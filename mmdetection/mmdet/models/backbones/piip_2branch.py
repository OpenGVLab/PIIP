# --------------------------------------------------------
# PIIP
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from functools import partial
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger

try:
    from ops.deformable_attention.modules import MSDeformAttn
    has_deform_attn = True
except:
    has_deform_attn = False

from .deit import vit_models
from .internvit_6b import InternViT6B
from .piip_modules import deform_inputs_1_vit, deform_inputs_2_vit, TwoBranchInteractionBlock


@BACKBONES.register_module()
class PIIPTwoBranch(nn.Module):
    def __init__(self, n_points=4, deform_num_heads=6,
                 with_cffn=False, cffn_ratio=0.25,
                 deform_ratio=1.0,   
                 is_dino=False,   
                 interaction_proj=True,
                      
                 interact_attn_type='normal',
                 interaction_drop_path_rate=0.3,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 
                 branch1={},
                 branch2={},
                 pretrained=None
                 ):
        
        super().__init__()
        
        if norm_layer == "none":
            norm_layer = nn.Identity
        
        self.interact_attn_type = interact_attn_type
        
        self.w1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.img_norm_cfg_branch1 = branch1.pop("img_norm_cfg")
        self.img_norm_cfg_branch2 = branch2.pop("img_norm_cfg")
        
        self.branch1_interaction_indexes = branch1.pop("interaction_indexes")
        self.branch2_interaction_indexes = branch2.pop("interaction_indexes")
        
        self.branch1_real_size = branch1.pop("real_size")
        self.branch2_real_size = branch2.pop("real_size")
        
        self.branch1_w_cls_token = branch1.pop("branch1_w_cls_token", False)
        self.branch2_w_cls_token = branch2.pop("branch2_w_cls_token", False)
        
        if 'deit' in branch1['pretrained']:
            self.branch1 = vit_models(**branch1)
        else:
            self.branch1 = InternViT6B(**branch1)
            self.branch1_w_cls_token = True
        
        if 'deit' in branch2['pretrained']:
            self.branch2 = vit_models(**branch2)
        else:
            self.branch2 = InternViT6B(**branch2)
            self.branch2_w_cls_token = True
        
        assert len(self.branch1_interaction_indexes) == len(self.branch2_interaction_indexes)
        self.interactions = nn.Sequential(*[
            TwoBranchInteractionBlock(
                branch1_dim=self.branch1.embed_dim,
                branch2_dim=self.branch2.embed_dim,
                branch1_img_size=self.branch1.pretrain_img_size,
                branch2_img_size=self.branch2.pretrain_img_size,
                num_heads=deform_num_heads, n_points=n_points,
                drop_path=interaction_drop_path_rate,
                norm_layer=norm_layer, with_cffn=with_cffn,
                cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                attn_type=interact_attn_type,
                with_proj=interaction_proj,
            )
            for _ in range(len(self.branch1_interaction_indexes))
        ])
        
        dim1 = self.branch1.embed_dim
        dim2 = self.branch2.embed_dim
        assert dim1 >= dim2
        
        self.merge_branch1 = nn.Sequential(
            nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, dim1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, dim1),
            nn.ReLU(inplace=True),
        )
        
        self.merge_branch2 = nn.Sequential(
            nn.Conv2d(dim2, dim1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, dim1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, dim1),
            nn.ReLU(inplace=True),
        )
        
        self.merge_branch1.apply(self._init_weights)
        self.merge_branch2.apply(self._init_weights)
        
        out_dim = dim1
        self.is_dino = is_dino
        if not is_dino:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(out_dim, out_dim, 2, 2),
                nn.GroupNorm(32, out_dim),
                nn.GELU(),
                nn.ConvTranspose2d(out_dim, out_dim, 2, 2)
            )
            self.fpn1.apply(self._init_weights) 
            
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(out_dim, out_dim, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))


        self.fpn2.apply(self._init_weights)
        self.fpn3.apply(self._init_weights)
        self.fpn4.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        self.init_weights(pretrained)
        
    @property
    def dtype(self):
        return self.branch1.patch_embed.proj.weight.dtype
    
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

        if isinstance(pretrained, str):
            logger = get_root_logger()
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            if 'module' in checkpoint:
                checkpoint_old = checkpoint['module']
                checkpoint = {}
                for k, v in checkpoint_old.items():
                    checkpoint[k.replace('backbone.', '')] = v
            message = self.load_state_dict(checkpoint, strict=False)
            logger.info(message)

    def _get_pos_embed(self, pos_embed, pretrain_size, patch_size, H, W):
        pos_embed = pos_embed.reshape(
            1, pretrain_size[0] // patch_size[0], pretrain_size[1] // patch_size[1], -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed.type(self.dtype)

    def _init_deform_weights(self, m):
        if has_deform_attn:
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, x):
        # Resize images
        scale_factor_1to2 = self.branch1_real_size / self.branch2_real_size
        if scale_factor_1to2 < 1:
            x1 = F.interpolate(x, scale_factor=scale_factor_1to2, mode='bilinear', align_corners=False)
        else:
            x1 = x.clone()

        x2 = x.clone()
        
        x1 = x1.type(self.dtype)
        x2 = x2.type(self.dtype)

        deform_inputs = {}
        if self.interact_attn_type == "deform":
            deform_inputs["2to1"] = deform_inputs_1_vit(x1, x2)
            deform_inputs["1to2"] = deform_inputs_2_vit(x2, x1)
        else:
            deform_inputs["2to1"] = [None, None, None]
            deform_inputs["1to2"] = [None, None, None]
        

        # Patch embedding and position embedding
        x1, H1, W1 = self.branch1.patch_embed(x1)
        bs1, n1, dim1 = x1.shape
        pos_embed1 = self.branch1.pos_embed if not self.branch1_w_cls_token else self.branch1.pos_embed[:, 1:]
        pos_embed1 = self._get_pos_embed(pos_embed1.float(), (self.branch1.pretrain_img_size, self.branch1.pretrain_img_size),  
                                         (self.branch1.patch_size, self.branch1.patch_size), H1, W1) 
        x1 = self.branch1.pos_drop(x1 + pos_embed1)

        x2, H2, W2 = self.branch2.patch_embed(x2)
        bs2, n2, dim2 = x2.shape
        pos_embed2 = self.branch2.pos_embed if not self.branch2_w_cls_token else self.branch2.pos_embed[:, 1:]
        pos_embed2 = self._get_pos_embed(pos_embed2.float(), (self.branch2.pretrain_img_size, self.branch2.pretrain_img_size), 
                                         (self.branch2.patch_size, self.branch2.patch_size), H2, W2) 
        x2 = self.branch2.pos_drop(x2 + pos_embed2)

        # Blocks and interactions
        for i, layer in enumerate(self.interactions):
            indexes1 = self.branch1_interaction_indexes[i]
            branch1_blocks = self.branch1.blocks[indexes1[0]:indexes1[-1] + 1]
            indexes2 = self.branch2_interaction_indexes[i]
            branch2_blocks = self.branch2.blocks[indexes2[0]:indexes2[-1] + 1]

            x1, x2, _, _ = layer(x1, x2,
                        branch1_blocks, branch2_blocks,
                        H1=H1, W1=W1, H2=H2, W2=W2,
                        cls1=None, cls2=None,
                        deform_inputs=deform_inputs)

        # Branch merging
        x1 = x1.transpose(1, 2).view(bs1, dim1, H1, W1)
        x1 = self.merge_branch1(x1)
        x1 = x1.type(torch.float32)
        x1 = F.interpolate(x1, size=(H2, W2), mode='bilinear', align_corners=False)
        x1 = x1.type(self.dtype)

        x2 = x2.transpose(1, 2).view(bs2, dim2, H2, W2)
        x2 = self.merge_branch2(x2)
        
        out = x1 * self.w1 + x2 * self.w2
             
        
        # Outputs for fpn
        if not self.is_dino:
            f1 = self.fpn1(out).contiguous().float()
            f2 = self.fpn2(out).contiguous().float()
            f3 = self.fpn3(out).contiguous().float()
            f4 = self.fpn4(out).contiguous().float()
            return [f1, f2, f3, f4]
            
        else:
            f2 = self.fpn2(out).contiguous().float()
            f3 = self.fpn3(out).contiguous().float()
            f4 = self.fpn4(out).contiguous().float()
            return [f2, f3, f4]