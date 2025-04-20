# --------------------------------------------------------
# PIIP
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel
from functools import partial
       
try:
    from .ops.modules import MSDeformAttn
    has_deform_attn = True
except:
    raise
    has_deform_attn = False

from .piip_modules import deform_inputs_1, deform_inputs_2, Permute, check_finite, TwoBranchInteractionBlock
from .clip_hf_wrapper import CLIPHFWrapper
from .configuration_piip_2branch import PIIPTwoBranchConfig


_logger = logging.get_logger(__name__)


class PIIPTwoBranchModel(PreTrainedModel):
    _no_split_modules = []
    config_class = PIIPTwoBranchConfig
    
    def __init__(self, config):
        
        super().__init__(config)
        self.config = config
        
        branch1 = config.branch1.copy()
        branch2 = config.branch2.copy()
        
        if config.norm_layer == "none":
            norm_layer = nn.Identity
        elif config.norm_layer == "ln":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            norm_layer = config.norm_layer
        
        
        self.branch1_interaction_indexes = branch1.pop("interaction_indexes")
        self.branch2_interaction_indexes = branch2.pop("interaction_indexes")
        
        self.branch1_model_type = branch1.pop("model_type", "clip_hf")
        self.branch2_model_type = branch2.pop("model_type", "clip_hf")
        
        self.branch1 = self.create_branch(self.branch1_model_type, branch1)
        self.branch2 = self.create_branch(self.branch2_model_type, branch2)
        
        
        assert len(self.branch1_interaction_indexes) == len(self.branch2_interaction_indexes)
        self.interactions = nn.Sequential(*[
            TwoBranchInteractionBlock(
                branch1_dim=self.branch1.embed_dim,
                branch2_dim=self.branch2.embed_dim, 
                branch1_img_size=self.branch1.img_size,
                branch2_img_size=self.branch2.img_size,
                branch1_patch_size=self.branch1.patch_size,
                branch2_patch_size=self.branch2.patch_size,
                num_heads=config.deform_num_heads, n_points=config.n_points,
                drop_path=config.interaction_drop_path_rate,
                norm_layer=norm_layer, with_cffn=config.with_cffn,
                cffn_ratio=config.cffn_ratio, deform_ratio=config.deform_ratio,
                attn_type=config.interact_attn_type,
                with_proj=config.interaction_proj,
            )
            for _ in range(len(self.branch1_interaction_indexes))
        ])
        
        dim1 = self.branch1.embed_dim
        dim2 = self.branch2.embed_dim
        self.out_dim = dim1

        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        
    @property
    def dtype(self):
        return self.branch1.clip_model.embeddings.patch_embedding.weight.dtype
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            dtype = m.weight.data.dtype
            m.weight.data = m.weight.data.float()
            trunc_normal_(m.weight, std=.02)
            m.weight.data = m.weight.data.to(dtype)
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
        if branch_model_type == "clip_hf":
            return CLIPHFWrapper(branch_config)
        else:
            raise NotImplementedError(branch_model_type)
    
    
    def _get_pos_embed(self, pos_embed, pretrain_size, patch_size, H, W):
        pos_embed = pos_embed.reshape(1, pretrain_size[0] // patch_size[0], pretrain_size[1] // patch_size[1], -1).permute(0, 3, 1, 2)
        dtype = pos_embed.dtype
        pos_embed = pos_embed.float()
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        pos_embed = pos_embed.to(dtype=dtype)
        return pos_embed
    
    
    def forward_embedding(self, branch, x,  branch_model_type, use_cls_token):
        if branch_model_type == "clip_hf":
            embed_layer = branch.clip_model.embeddings
            patch_embeds = embed_layer.patch_embedding(x)
            bs, _, H, W = patch_embeds.shape
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
            x = patch_embeds
            _, n, dim = x.shape
            pos_embed = self._get_pos_embed(embed_layer.position_embedding(embed_layer.position_ids)[:, 1:], (branch.pretrain_size, branch.pretrain_size), (branch.pretrain_patch_size, branch.pretrain_patch_size), H, W)
            
            class_embeds = embed_layer.class_embedding.expand(bs, 1, -1)
            x = torch.cat([class_embeds, x], dim=1)
            
            cls_pos_embed = embed_layer.position_embedding(embed_layer.position_ids)[:, :1]
            pos_embed = torch.cat([cls_pos_embed, pos_embed], dim=1)
            
            x = x + pos_embed
            x = branch.clip_model.pre_layrnorm(x)
            if use_cls_token:
                cls_, x = x[:, :1], x[:, 1:]
            else:
                x = x[:, 1:]
                cls_ = None
            return x, cls_, bs, n, dim, H, W
        else:
            raise NotImplementedError

    def forward(self, x):
        # Resize images
        dtype = x.dtype
        x = x.float()
            
        scale_factor_1to2 = self.branch1.img_size / self.branch2.img_size
        if scale_factor_1to2 < 1:
            x1 = F.interpolate(x, scale_factor=scale_factor_1to2, mode='bilinear', align_corners=False)
        else:
            x1 = x.clone()
        x2 = x.clone()
        
        x = x.to(dtype)
        x1 = x1.type(self.dtype)
        x2 = x2.type(self.dtype)

        deform_inputs = {}
        if self.config.interact_attn_type == "deform":
            deform_inputs["2to1"] = deform_inputs_1(x1, x2, self.branch1.patch_size, self.branch2.patch_size)
            deform_inputs["1to2"] = deform_inputs_2(x2, x1, self.branch1.patch_size, self.branch2.patch_size)
        else:
            deform_inputs["2to1"] = [None, None, None]
            deform_inputs["1to2"] = [None, None, None]
        
        
        # Patch embedding and position embedding
        x1, cls1, bs1, n1, dim1, H1, W1 = self.forward_embedding(self.branch1, x1, self.branch1_model_type, use_cls_token=self.branch1.config.get("use_cls_token", False))
        x2, cls2, bs2, n2, dim2, H2, W2 = self.forward_embedding(self.branch2, x2, self.branch2_model_type, use_cls_token=self.branch2.config.get("use_cls_token", False))
        
        assert check_finite(x1)
        assert check_finite(x2)

        # Blocks and interactions
        for i, layer in enumerate(self.interactions):
            indexes1 = self.branch1_interaction_indexes[i]
            branch1_blocks = self.branch1.blocks[indexes1[0]:indexes1[-1] + 1]
            indexes2 = self.branch2_interaction_indexes[i]
            branch2_blocks = self.branch2.blocks[indexes2[0]:indexes2[-1] + 1]

            x1, x2, cls1, cls2 = layer(x1, x2,
                        branch1_blocks, branch2_blocks,
                        H1=H1, W1=W1, H2=H2, W2=W2,
                        cls1=cls1, cls2=cls2,
                        deform_inputs=deform_inputs)
            assert check_finite(x1), f"layer{i}"
            assert check_finite(x2), f"layer{i}"
        
        
        x1 = x1.transpose(1, 2).view(bs1, dim1, H1, W1)
        x2 = x2.transpose(1, 2).view(bs2, dim2, H2, W2)
        return x1, x2 # for two mlps
