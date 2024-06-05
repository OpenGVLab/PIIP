# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from flash_attn.flash_attention import FlashAttention
    from flash_attn.modules.mlp import FusedMLP
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_first'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            input_dtype = x.dtype
            x = x.to(torch.float32)
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None].to(torch.float32) * x + self.bias[:, None, None].to(torch.float32)
            x = x.to(input_dtype)
            return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from apex.normalization import FusedRMSNorm

    RMSNorm = FusedRMSNorm  # noqa

    print('Discovered apex.normalization.FusedRMSNorm - will use it instead of RMSNorm')
except ImportError:
    # using the normal RMSNorm
    pass


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False, force_fp32=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.force_fp32 = force_fp32

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if self.force_fp32:
            output_type = x.dtype
            out = x.float().mul_(self.gamma.float()) if self.inplace else x.float() * self.gamma.float()
            return out.to(dtype=output_type)
        else:
            out = x.mul_(self.gamma) if self.inplace else x * self.gamma
            return out


class WindowedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_flash_attn=False,
                 causal=False, norm_layer=nn.LayerNorm, qk_normalization=False, window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size

        self.causal = causal
        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            print("! Using flash attention")
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()

    def forward(self, x, H, W):
        B, N, C = x.shape
        N_ = self.window_size * self.window_size
        H_ = math.ceil(H / self.window_size) * self.window_size
        W_ = math.ceil(W / self.window_size) * self.window_size
        x = x.view(B, H, W, C)
        x = F.pad(x, [0, 0, 0, W_ - W, 0, H_ - H])
        
        x = window_partition(x, window_size=self.window_size)  # nW*B, window_size, window_size, C
        x = x.view(-1, N_, C)
        
        def _naive_attn(x):
            qkv = self.qkv(x).view(-1, N_, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            if self.qk_normalization:
                B_, H_, _, D_ = q.shape
                q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
                k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, L, num_head, N_, N_]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)  # [B, L, num_head, N_, N_]
            x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size, self.window_size, C)
            return x
        
        def _flash_attn(x):
            qkv = self.qkv(x).view(-1, N_, 3, self.num_heads, C // self.num_heads)
            
            if self.qk_normalization:
                q, k, v = qkv.unbind(2)
                q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
                qkv = torch.stack([q, k, v], dim=2)
            
            # For val w/o deepspeed
            qkv_dtype = qkv.dtype
            if qkv.dtype not in [torch.float16, torch.bfloat16]:
                qkv = qkv.to(torch.float16)      
            context, _ = self.inner_attn(qkv, causal=self.causal)
            if qkv_dtype not in [torch.float16, torch.bfloat16]:
                context = context.to(qkv_dtype)
            x = context.reshape(-1, self.window_size, self.window_size, C)
            return x
        
        x = _naive_attn(x) if not self.use_flash_attn else _flash_attn(x)
        x = x.contiguous()
        
        x = window_reverse(x, self.window_size, H_, W_)
        x = x[:, :H, :W, :].reshape(B, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_flash_attn=False,
                 causal=False, norm_layer=nn.LayerNorm, qk_normalization=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            print("! Using flash attention")
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()

    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        # For val w/o deepspeed
        qkv_dtype = qkv.dtype
        if qkv.dtype not in [torch.float16, torch.bfloat16]:
            qkv = qkv.to(torch.float16) 
        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal
        )
        if qkv_dtype not in [torch.float16, torch.bfloat16]:
            context = context.to(qkv_dtype)
        outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, x, H=None, W=None):
        x = self._naive_attn(x) if not self.use_flash_attn else self._flash_attn(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_flash_attn=False, with_cp=False,
            qk_normalization=False, layerscale_force_fp32=False, windowed=False, window_size=14,
            mlp_type="regular"):
        super().__init__()

        self.norm1 = norm_layer(dim)
        if windowed:
            self.attn = WindowedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                          proj_drop=drop, use_flash_attn=use_flash_attn, causal=False,
                                          norm_layer=norm_layer, qk_normalization=qk_normalization,
                                          window_size=window_size)
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                  proj_drop=drop, use_flash_attn=use_flash_attn, causal=False,
                                  norm_layer=norm_layer, qk_normalization=qk_normalization)
        self.ls1 = LayerScale(dim, init_values=init_values,
                              force_fp32=layerscale_force_fp32) if init_values else nn.Identity()

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        if mlp_type == "regular":
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif mlp_type == "fused_mlp":
            self.mlp = FusedMLP(in_features=dim, hidden_features=mlp_hidden_dim, activation="gelu_approx")
        else:
            raise NotImplementedError(mlp_type)
        
        self.ls2 = LayerScale(dim, init_values=init_values,
                              force_fp32=layerscale_force_fp32) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.with_cp = with_cp

    def forward(self, x, H=None, W=None):

        def _inner_forward(x, H, W):
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), H, W)))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x

        if self.with_cp:
            return checkpoint.checkpoint(_inner_forward, x, H, W)
        else:
            return _inner_forward(x, H, W)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, **kwargs):
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, H, W


@BACKBONES.register_module()
class InternViT6B(BaseModule):
    def __init__(self, in_chans=3, patch_size=14, pretrain_patch_size=14, pretrain_img_size=224, qkv_bias=False, drop_path_rate=0.0,
                 embed_dim=3200, num_heads=25, mlp_ratio=4, init_values=0.1, qk_normalization=True, depth=48,
                 use_flash_attn=True, with_cp=True, layerscale_force_fp32=False, out_indices=[7, 11, 15, 23],
                 with_fpn=False, with_final_norm=False, window_attn=False, window_size=14,
                 output_dtype="float16", pretrained=None, norm_layer_type='RMSNorm', use_simple_fpn=True,
                 mlp_type="regular", **kwargs):

        super().__init__()

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.drop_path_rate = drop_path_rate
        self.pretrain_img_size = pretrain_img_size
        self.pretrain_patch_size = pretrain_patch_size
        self.patch_size = patch_size
        self.out_indices = out_indices
        self.with_fpn = with_fpn
        self.use_simple_fpn = use_simple_fpn
        self.pretrained = pretrained
        
        if output_dtype == 'float16':
            self.output_dtype = torch.float16
        elif output_dtype == 'bfloat16':
            self.output_dtype = torch.bfloat16
        elif output_dtype == 'float32':
            self.output_dtype = torch.float32
        else:
            raise NotImplementedError

        use_flash_attn = use_flash_attn and has_flash_attn
        if use_flash_attn and not has_flash_attn:
            print('Warning: Flash Attention is not available, use_flash_attn is set to False.')
        use_flash_attn = [use_flash_attn] * depth if not isinstance(use_flash_attn, list) else use_flash_attn
        window_attn = [window_attn] * depth if not isinstance(window_attn, list) else window_attn
        window_size = [window_size] * depth if not isinstance(window_size, list) else window_size
        logging.info(f'use_flash_attn: {use_flash_attn}')
        logging.info(f'init values: {init_values}')
        logging.info(f'window attention: {window_attn}')
        logging.info(f'window size: {window_size}')

        self.norm_layer_for_blocks = partial(RMSNorm, eps=1e-6) if norm_layer_type == 'RMSNorm' else partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(pretrain_img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Identity()
        self.cls_token = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=self.norm_layer_for_blocks,
                  drop_path=dpr[i], init_values=init_values, attn_drop=0.,
                  use_flash_attn=use_flash_attn[i],
                  with_cp=with_cp,
                  qk_normalization=qk_normalization,
                  layerscale_force_fp32=layerscale_force_fp32,
                  windowed=window_attn[i],
                  window_size=window_size[i],
                  mlp_type=mlp_type)
            for i in range(depth)])

        self.init_weights(pretrained)

        if with_fpn:
            self.up1 = nn.Sequential(*[
                nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
                LayerNorm(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
                LayerNorm(embed_dim) if with_final_norm else nn.Identity()
            ])
            self.up2 = nn.Sequential(*[
                nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
                LayerNorm(embed_dim) if with_final_norm else nn.Identity()
            ])
            self.up3 = nn.Sequential(*[
                nn.Identity(),
                LayerNorm(embed_dim) if with_final_norm else nn.Identity()
            ])
            self.up4 = nn.Sequential(*[
                nn.MaxPool2d(kernel_size=2, stride=2),
                LayerNorm(embed_dim) if with_final_norm else nn.Identity()
            ])
            self.up1.apply(self._init_weights)
            self.up2.apply(self._init_weights)
            self.up3.apply(self._init_weights)
            self.up4.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):

        def resize_pos_embed(pos_embed, H, W):
            cls = pos_embed[:, :1, :]
            pos_embed = pos_embed[:, 1:, :].reshape(
                1, self.pretrain_img_size // self.pretrain_patch_size, self.pretrain_img_size // self.pretrain_patch_size, -1).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
                reshape(1, -1, H * W).permute(0, 2, 1)
            pos_embed = torch.cat([cls, pos_embed], dim=1)
            return pos_embed

        if isinstance(pretrained, str):
            logger = get_root_logger()
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']

            # resize pos_embed
            pos_embed = checkpoint['pos_embed'] if 'pos_embed' in checkpoint.keys() else self.pos_embed
            checkpoint['pos_embed'] = resize_pos_embed(
                pos_embed, self.pretrain_img_size // self.patch_size, self.pretrain_img_size // self.patch_size)
            # resize patch_embed
            patch_embed = checkpoint['patch_embed.proj.weight']
            checkpoint['patch_embed.proj.weight'] = F.interpolate(
                patch_embed, size=(self.patch_size, self.patch_size),
                mode='bicubic', align_corners=False)
            message = self.load_state_dict(checkpoint, strict=False)
            logger.info(message)

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_img_size // self.patch_size, self.pretrain_img_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed.float(), size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1).to(pos_embed.dtype)
        return pos_embed

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def forward(self, x):
        x, H, W = self.patch_embed(x.type(self.dtype))
        batch_size, seq_len, _ = x.size()
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)
        outs = list()
        for idx, blk in enumerate(self.blocks):
            x = blk(x, H, W)
            if idx in self.out_indices:
                out = x
                b, n, c = out.shape
                out = out.reshape(b, H, W, c).permute(0, 3, 1, 2)
                outs.append(out)

        if self.use_simple_fpn:
            outs = [outs[-1]]
        if not self.with_fpn:
            return [item.to(self.output_dtype).contiguous() for item in outs]
        else:
            if len(outs) == 1:
                x1 = x2 = x3 = x4 = outs[0]
            else:
                x1, x2, x3, x4 = outs
            f1 = self.up1(x1).to(self.output_dtype).contiguous()
            f2 = self.up2(x2).to(self.output_dtype).contiguous()
            f3 = self.up3(x3).to(self.output_dtype).contiguous()
            f4 = self.up4(x4).to(self.output_dtype).contiguous()
            return [f1, f2, f3, f4]