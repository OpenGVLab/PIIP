# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from functools import partial

from timm.models.vision_transformer import Mlp

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger

try:
    from flash_attn.flash_attention import FlashAttention
    from flash_attn.modules.mlp import FusedMLP
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False


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


class FlashAttn(nn.Module):
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
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()

    def _naive_attn(self, x, return_attn=False):
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
        if return_attn:
            return x, attn
        else:
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

    def forward(self, x, H=None, W=None, return_attn=False):
        if return_attn:
            x, attn = self._naive_attn(x, return_attn)
            return x, attn
        else:        
            x = self._naive_attn(x) if not self.use_flash_attn else self._flash_attn(x)
            return x


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, H, W


class WindowedFlashAttention(nn.Module):
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

        x = window_partition(x, window_size=self.window_size)  # nW*B, window_size, window_size, C
        x = x.view(-1, N_, C)

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


class WindowedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.,
                 proj_drop=0., qk_scale=None, window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size

    def forward(self, x, H, W):
        B, N, C = x.shape
        N_ = self.window_size * self.window_size
        H_ = math.ceil(H / self.window_size) * self.window_size
        W_ = math.ceil(W / self.window_size) * self.window_size

        qkv = self.qkv(x)  # [B, N, C]
        qkv = qkv.transpose(1, 2).reshape(B, C * 3, H, W)  # [B, C, H, W]
        qkv = F.pad(qkv, [0, W_ - W, 0, H_ - H], mode='constant')

        dtype = qkv.dtype
        qkv = F.unfold(qkv.float(), kernel_size=(self.window_size, self.window_size),
                       stride=(self.window_size, self.window_size)).to(dtype)
        
        B, C_kw_kw, L = qkv.shape  # L - the num of windows
        qkv = qkv.reshape(B, C * 3, N_, L).permute(0, 3, 2, 1)  # [B, L, N_, C]
        qkv = qkv.reshape(B, L, N_, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv.unbind(0)

        # q,k,v [B, L, num_head, N_, C/num_head]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, L, num_head, N_, N_]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, L, num_head, N_, N_]
        # attn @ v = [B, L, num_head, N_, C/num_head]
        x = (attn @ v).permute(0, 2, 4, 3, 1).reshape(B, C_kw_kw // 3, L)

        x = F.fold(x.float(), output_size=(H_, W_),
                   kernel_size=(self.window_size, self.window_size),
                   stride=(self.window_size, self.window_size)).to(dtype)  # [B, C, H_, W_]
        x = x[:, :, :H, :W].reshape(B, C, N).transpose(-1, -2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 
   
    
class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4, window_size=-1, use_flash_attn=False, with_cp=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.with_cp = with_cp
                
        if window_size > 0:
            print("! Using window attention")
            if use_flash_attn:
                print("! Using flash attention")
                self.attn = WindowedFlashAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, use_flash_attn=use_flash_attn, causal=False, window_size=window_size)
            else:
                self.attn = WindowedAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size)
        else:
            if use_flash_attn:
                print("! Using flash attention")
                self.attn = FlashAttn(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, use_flash_attn=use_flash_attn, causal=False)
            else:
                self.attn = Attention_block(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x, H=None, W=None, return_attn=False):
        def _inner_forward(x, H, W, return_attn):
            if not return_attn:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
                return x
            else:
                attn, attn_weight = self.attn(self.norm1(x), H, W, return_attn)
                x = x + self.drop_path(self.gamma_1 * attn)
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
                return x, attn_weight

        if self.with_cp:
            return checkpoint.checkpoint(_inner_forward, x, H, W, return_attn)
        else:
            return _inner_forward(x, H, W, return_attn)


class Layer_scale_init_Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_1_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        
    def forward(self, x):
        x = x + self.drop_path(self.gamma_1*self.attn(self.norm1(x))) + self.drop_path(self.gamma_1_1 * self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + self.drop_path(self.gamma_2_1 * self.mlp1(self.norm21(x)))
        return x
       
        
class Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.mlp1(self.norm21(x)))
        return x
        
        
class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, embed_dim=768,norm_layer=nn.SyncBatchNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                         ])
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


@BACKBONES.register_module()    
class vit_models(BaseModule):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, pretrain_img_size=224, patch_size=16, pretrain_patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 block_layers=Layer_scale_init_Block,
                 act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,
                 use_cls_token=False,
                 init_scale=1e-4, window_size=None, window_attn=None,
                 output_dtype="float32", pretrained=None,
                 out_indices=[7, 11, 15, 23], with_fpn=False, use_simple_fpn=True,
                 use_flash_attn=False, with_cp=False, cal_flops=False,):

        super().__init__()
        
        if Mlp_block == "fused_mlp":
            assert FusedMLP is not None
            class FusedMLPWrapper(FusedMLP):
                def __init__(self, in_features, hidden_features, act_layer, drop):
                    super().__init__(in_features=in_features, hidden_features=hidden_features, activation="gelu_approx")
            Mlp_block = FusedMLPWrapper
        
        if window_attn is None:
            window_attn = [False] * depth
        if window_size is None:
            window_size = [-1] * depth
        assert len(window_size) == len(window_attn) == depth
        
        self.dropout_rate = drop_rate
        self.pretrain_img_size = pretrain_img_size
        self.pretrain_patch_size = pretrain_patch_size
        self.patch_size = patch_size
        self.with_fpn = with_fpn
        self.out_indices = out_indices
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

        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
                img_size=pretrain_img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Identity()

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block,init_values=init_scale, 
                window_size=window_size[i] if window_attn[i] is True else -1, 
                use_flash_attn=use_flash_attn, with_cp=with_cp
                )
            for i in range(depth)])

        self.init_weights(pretrained)

        if with_fpn:
            self.up1 = nn.Sequential(*[
                nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
                LayerNorm(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
            ])
            self.up2 = nn.Sequential(*[
                nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
            ])
            self.up3 = nn.Identity()
            self.up4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.up1.apply(self._init_weights)
            self.up2.apply(self._init_weights)
            self.up3.apply(self._init_weights)
            self.up4.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
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
            len_pos = pos_embed.shape[1]
            if int(len_pos ** 0.5) ** 2 != len_pos:
                print("drop cls token pos embed.")
                pos_embed = pos_embed[:, 1:, :]
            pos_embed = pos_embed.reshape(
                1, self.pretrain_img_size // self.pretrain_patch_size, self.pretrain_img_size // self.pretrain_patch_size, -1).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
                reshape(1, -1, H * W).permute(0, 2, 1)
            return pos_embed

        if isinstance(pretrained, str):
            logger = get_root_logger()
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            # resize pos_embed
            pos_embed = checkpoint['pos_embed']
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
    
    def forward_features(self, x):
        x, H, W = self.patch_embed(x.type(self.dtype))
        pos_embed = self._get_pos_embed(self.pos_embed, H, W)
        x = self.pos_drop(x + pos_embed)
        
        outs = list() 
        for idx , blk in enumerate(self.blocks):
            x = blk(x, H, W)
            if idx in self.out_indices:
                out = x
                b, n, c = out.shape
                out = out.reshape(b, H, W, c).permute(0, 3, 1, 2)
                outs.append(out)

        return outs

    def forward(self, x):
        outs = self.forward_features(x)
        if self.use_simple_fpn:
            outs = [outs[-1]]
        if not self.with_fpn:
            return [item.contiguous().to(self.output_dtype) for item in outs]
        else:
            x1 = x2 = x3 = x4 = outs[-1]
            f1 = self.up1(x1).to(self.output_dtype).contiguous()
            f2 = self.up2(x2).to(self.output_dtype).contiguous()
            f3 = self.up3(x3).to(self.output_dtype).contiguous()
            f4 = self.up4(x4).to(self.output_dtype).contiguous()
            return [f1, f2, f3, f4]

