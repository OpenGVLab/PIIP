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
       
try:
    from ops.modules import MSDeformAttn
    has_deform_attn = True
except:
    has_deform_attn = False

from deit import vit_models
from intern_vit_6b import InternViT6B
from piip_modules import deform_inputs_1, deform_inputs_2, Permute, ThreeBranchInteractionBlock


class PIIPThreeBranch(nn.Module):
    def __init__(self, n_points=4, deform_num_heads=6,
                 with_cffn=False, cffn_ratio=0.25,
                 deform_ratio=1.0,   
                 interact_attn_type='normal',
                 interaction_drop_path_rate=0.3,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 interaction_proj=True,
                 
                 branch1={},
                 branch2={},
                 branch3={},
                 
                 num_classes=1000,
                 use_cls_head=True,
                 separate_head=False
                 ):
        
        super().__init__()
        
        self.interact_attn_type = interact_attn_type
        
        branch1 = branch1.copy()
        branch2 = branch2.copy()
        branch3 = branch3.copy()
        
        if norm_layer == "none":
            norm_layer = nn.Identity
        
        if not separate_head:
            self.w1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
            self.w2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
            self.w3 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
        self.branch1_interaction_indexes = branch1.pop("interaction_indexes")
        self.branch2_interaction_indexes = branch2.pop("interaction_indexes")
        self.branch3_interaction_indexes = branch3.pop("interaction_indexes")
        
        self.branch1_model_type = branch1.pop("model_type", "deit")
        self.branch2_model_type = branch2.pop("model_type", "deit")
        self.branch3_model_type = branch3.pop("model_type", "deit")
        
        self.separate_head = separate_head
        
        self.branch1 = self.create_branch(self.branch1_model_type, branch1)
        self.branch2 = self.create_branch(self.branch2_model_type, branch2)
        self.branch3 = self.create_branch(self.branch3_model_type, branch3)
        
        assert len(self.branch1_interaction_indexes) == len(self.branch2_interaction_indexes) == len(self.branch3_interaction_indexes)
        self.interactions = nn.Sequential(*[
            ThreeBranchInteractionBlock(
                branch1_dim=self.branch1.embed_dim,
                branch2_dim=self.branch2.embed_dim, 
                branch3_dim=self.branch3.embed_dim, 
                branch1_img_size=self.branch1.img_size,
                branch2_img_size=self.branch2.img_size,
                branch3_img_size=self.branch3.img_size,
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
        dim3 = self.branch3.embed_dim
        assert dim1 >= dim2 >= dim3

        if not separate_head:
            self.merge_branch1 = nn.GroupNorm(32, dim1)
            self.merge_branch2 = nn.Sequential(
                nn.GroupNorm(32, dim2),
                Permute(0,2,3,1), # N,C,H,W -> N,H,W,C
                nn.Linear(dim2, dim1),
                Permute(0,3,1,2) , # N,H,W,C -> N,C,H,W
                nn.GroupNorm(32, dim1),
            )
            self.merge_branch3 = nn.Sequential(
                nn.GroupNorm(32, dim3),
                Permute(0,2,3,1), # N,C,H,W -> N,H,W,C
                nn.Linear(dim3, dim1),
                Permute(0,3,1,2) , # N,H,W,C -> N,C,H,W
                nn.GroupNorm(32, dim1),
            )
            self.merge_branch1.apply(self._init_weights)
            self.merge_branch2.apply(self._init_weights)
            self.merge_branch3.apply(self._init_weights)
            
            if use_cls_head:
                self.cls_norm = nn.LayerNorm(dim1, eps=1e-6)
                self.cls_head = nn.Linear(dim1, num_classes)
                self.cls_norm.apply(self._init_weights)
                self.cls_head.apply(self._init_weights)
            else:
                self.cls_head = None

        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        
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


    def _init_deform_weights(self, m):
        if has_deform_attn:
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
    
    
    def create_branch(self, branch_model_type, branch_config):
        if branch_model_type == "deit":
            return vit_models(**branch_config, use_final_norm_and_head=self.separate_head)
        elif branch_model_type == "augreg":
            # InternViT6B is used to load augreg models
            return InternViT6B(**branch_config, use_final_norm_and_head=self.separate_head)
        else:
            raise NotImplementedError(branch_model_type)
    
    
    def forward_embedding(self, branch, x, model_type):
        assert branch.patch_embed.patch_size == (16, 16)
        if model_type == "deit":
            x = branch.patch_embed(x)
            bs, n, dim = x.shape
            H = W = int(n ** 0.5)
            pos_embed = branch.pos_embed
            x = x + pos_embed
            if branch.cls_token is not None:
                cls_ = branch.cls_token.expand(bs, -1, -1)
            else:
                cls_ = None
            return x, cls_, bs, n, dim, H, W
        elif model_type == "augreg":
            x, H, W = branch.patch_embed(x)
            bs, n, dim = x.size()
            pos_embed = branch._get_pos_embed(branch.pos_embed[:, 1:], H, W)
            x = branch.pos_drop(x + pos_embed)
            if branch.cls_token is not None:
                cls_ = branch.cls_token.expand(bs, -1, -1)
            else:
                cls_ = None
            return x, cls_, bs, n, dim, H, W

    def forward(self, x):
        # Resize images
        scale_factor_1to3 = self.branch1.img_size / self.branch3.img_size
        if scale_factor_1to3 < 1:
            x1 = F.interpolate(x, scale_factor=scale_factor_1to3, mode='bilinear', align_corners=False)
        else:
            x1 = x.clone()
            
        scale_factor_2to3 = self.branch2.img_size / self.branch3.img_size
        if scale_factor_2to3 < 1:
            x2 = F.interpolate(x, scale_factor=scale_factor_2to3, mode='bilinear', align_corners=False)
        else:
            x2 = x.clone()
        scale_factor_1to2 = self.branch1.img_size / self.branch2.img_size
        x3 = x.clone()
        
        x1 = x1.type(self.dtype)
        x2 = x2.type(self.dtype)
        x3 = x3.type(self.dtype)

        deform_inputs = {}
        if self.interact_attn_type == "deform":
            deform_inputs["2to1"] = deform_inputs_1(x1, scale_factor_1to2)
            deform_inputs["1to2"] = deform_inputs_2(x2, scale_factor_1to2)
            deform_inputs["3to2"] = deform_inputs_1(x2, scale_factor_2to3)
            deform_inputs["2to3"] = deform_inputs_2(x3, scale_factor_2to3)
        else:
            deform_inputs["2to1"] = [None, None, None]
            deform_inputs["1to2"] = [None, None, None]
            deform_inputs["3to2"] = [None, None, None]
            deform_inputs["2to3"] = [None, None, None]
        
        # Patch embedding and position embedding
        x1, cls1, bs1, n1, dim1, H1, W1 = self.forward_embedding(self.branch1, x1, self.branch1_model_type)
        x2, cls2, bs2, n2, dim2, H2, W2 = self.forward_embedding(self.branch2, x2, self.branch2_model_type)
        x3, cls3, bs3, n3, dim3, H3, W3 = self.forward_embedding(self.branch3, x3, self.branch3_model_type)

        # Blocks and interactions
        for i, layer in enumerate(self.interactions):
            indexes1 = self.branch1_interaction_indexes[i]
            branch1_blocks = self.branch1.blocks[indexes1[0]:indexes1[-1] + 1]
            indexes2 = self.branch2_interaction_indexes[i]
            branch2_blocks = self.branch2.blocks[indexes2[0]:indexes2[-1] + 1]
            indexes3 = self.branch3_interaction_indexes[i]
            branch3_blocks = self.branch3.blocks[indexes3[0]:indexes3[-1] + 1]

            x1, x2, x3, cls1, cls2, cls3 = layer(x1, x2, x3,
                        branch1_blocks, branch2_blocks, branch3_blocks,
                        H1=H1, W1=W1, H2=H2, W2=W2, H3=H3, W3=W3,
                        cls1=cls1, cls2=cls2, cls3=cls3,
                        deform_inputs=deform_inputs)
        
        if self.separate_head:
            assert self.branch1.cls_token is not None and self.branch2.cls_token is not None and self.branch3.cls_token is not None
            x1 = self.branch1.head(self.branch1.norm(x1)[:, 0])
            x2 = self.branch2.head(self.branch2.norm(x2)[:, 0])
            x3 = self.branch3.head(self.branch3.norm(x3)[:, 0])
            return (x1 + x2 + x3) / 3

        x1 = x1.transpose(1, 2).view(bs1, dim1, H1, W1)
        x1 = self.merge_branch1(x1)
        x1 = x1.type(torch.float32)
        x1 = F.interpolate(x1, size=(H3, W3), mode='bilinear', align_corners=False)
        x1 = x1.type(self.dtype)

        x2 = x2.transpose(1, 2).view(bs2, dim2, H2, W2)
        x2 = self.merge_branch2(x2)
        x2 = x2.type(torch.float32)
        x2 = F.interpolate(x2, size=(H3, W3), mode='bilinear', align_corners=False)
        x2 = x2.type(self.dtype)
        
        x3 = x3.transpose(1, 2).view(bs3, dim3, H3, W3)
        x3 = self.merge_branch3(x3)
        
        out = x1 * self.w1 + x2 * self.w2 + x3 * self.w3
        if self.cls_head is None:
            # for calculating flops
            return out
        
        out = out.reshape(bs1, dim1, H3*W3).transpose(1,2)
        out = self.cls_norm(out)
        out = out.mean(dim=1)
        out = self.cls_head(out)
        
        return out