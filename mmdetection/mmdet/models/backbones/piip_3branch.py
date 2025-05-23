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
from .beit import BEiT
from .internvit_6b import InternViT6B
from .uniperceiver import UnifiedBertEncoder
from .convnext_hf_wrapper import ConvNextHFWrapper
from .piip_modules import deform_inputs_1_vit, deform_inputs_2_vit, ThreeBranchInteractionBlock


@BACKBONES.register_module()
class PIIPThreeBranch(nn.Module):
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
                 branch3={},
                 pretrained=None,
                 with_simple_fpn=True,
                 out_interaction_indexes=[],
                 cal_flops=False,
                 ):
        
        super().__init__()
        
        if norm_layer == "none":
            norm_layer = nn.Identity
        
        self.interact_attn_type = interact_attn_type
        self.out_interaction_indexes = out_interaction_indexes
        self.cal_flops = cal_flops
        branch1 = branch1.copy()
        branch2 = branch2.copy()
        branch3 = branch3.copy()
        
        self.w1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.w3 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
        self.branch1_interaction_indexes = branch1.pop("interaction_indexes")
        self.branch2_interaction_indexes = branch2.pop("interaction_indexes")
        self.branch3_interaction_indexes = branch3.pop("interaction_indexes")
        
        self.branch1_real_size = branch1.pop("real_size")
        self.branch2_real_size = branch2.pop("real_size")
        self.branch3_real_size = branch3.pop("real_size")
        
        self.branch1_w_cls_token = branch1.pop("branch1_w_cls_token", False)
        self.branch2_w_cls_token = branch2.pop("branch2_w_cls_token", False)
        self.branch3_w_cls_token = branch3.pop("branch3_w_cls_token", False)
        
        if 'deit' in branch1['pretrained']:
            self.branch1 = vit_models(**branch1)
        elif 'beit' in branch1['pretrained']:
            self.branch1 = BEiT(**branch1)
        elif 'perceiver' in branch1['pretrained']:
            self.branch1 = UnifiedBertEncoder(**branch1)
        elif 'convnext' in branch1['pretrained']:
            self.branch1 = ConvNextHFWrapper(**branch1, real_size=self.branch1_real_size, with_simple_fpn=False)
        else:
            self.branch1 = InternViT6B(**branch1)
            self.branch1_w_cls_token = True
        
        if 'deit' in branch2['pretrained']:
            self.branch2 = vit_models(**branch2)
        elif 'beit' in branch2['pretrained']:
            self.branch2 = BEiT(**branch2)
        elif 'perceiver' in branch2['pretrained']:
            self.branch2 = UnifiedBertEncoder(**branch2)
        elif 'convnext' in branch2['pretrained']:
            self.branch2 = ConvNextHFWrapper(**branch2, real_size=self.branch2_real_size, with_simple_fpn=False)
        else:
            self.branch2 = InternViT6B(**branch2)
            self.branch2_w_cls_token = True
            
        if 'deit' in branch3['pretrained']:
            self.branch3 = vit_models(**branch3)
        elif 'beit' in branch3['pretrained']:
            self.branch3 = BEiT(**branch3)
        elif 'perceiver' in branch3['pretrained']:
            self.branch3 = UnifiedBertEncoder(**branch3)
        elif 'convnext' in branch3['pretrained']:
            self.branch3 = ConvNextHFWrapper(**branch3, real_size=self.branch3_real_size, with_simple_fpn=False)
        else:
            self.branch3 = InternViT6B(**branch3)
            self.branch3_w_cls_token = True
        
        assert len(self.branch1_interaction_indexes) == len(self.branch2_interaction_indexes) == len(self.branch3_interaction_indexes)
        num_interactions = len(self.branch1_interaction_indexes)
        if 'convnext' in branch1['pretrained']:
            dims1 = self.branch1.out_dims
        else:
            if 'perceiver' in branch1['pretrained']:
                depth = len(self.branch1.layers)
            else:
                depth = len(self.branch1.blocks)
            dims1 = [self.branch1.embed_dim] * depth
            self.branch1.downsample_ratios = [self.branch1.patch_size] * depth
        
        if 'convnext' in branch2['pretrained']:
            dims2 = self.branch2.out_dims
        else:
            if 'perceiver' in branch2['pretrained']:
                depth = len(self.branch2.layers)
            else:
                depth = len(self.branch2.blocks)
            dims2 = [self.branch2.embed_dim] * depth
            self.branch2.downsample_ratios = [self.branch2.patch_size] * depth

        if 'convnext' in branch3['pretrained']:
            dims3 = self.branch3.out_dims
        else:
            if 'perceiver' in branch3['pretrained']:
                depth = len(self.branch3.layers)
            else:
                depth = len(self.branch3.blocks)
            dims3 = [self.branch3.embed_dim] * depth
            self.branch3.downsample_ratios = [self.branch3.patch_size] * depth
        
        self.branch1_interaction_layer_index = [self.branch1_interaction_indexes[idx][-1] for idx in range(num_interactions)]
        self.branch2_interaction_layer_index = [self.branch2_interaction_indexes[idx][-1] for idx in range(num_interactions)]
        self.branch3_interaction_layer_index = [self.branch3_interaction_indexes[idx][-1] for idx in range(num_interactions)]
        self.interactions = nn.Sequential(*[
            ThreeBranchInteractionBlock(
                branch1_dim=dims1[self.branch1_interaction_layer_index[idx]],
                branch2_dim=dims2[self.branch2_interaction_layer_index[idx]],
                branch3_dim=dims3[self.branch3_interaction_layer_index[idx]],
                branch1_feat_size=self.branch1_real_size // self.branch1.downsample_ratios[self.branch1_interaction_layer_index[idx]],
                branch2_feat_size=self.branch2_real_size // self.branch2.downsample_ratios[self.branch2_interaction_layer_index[idx]],
                branch3_feat_size=self.branch3_real_size // self.branch3.downsample_ratios[self.branch3_interaction_layer_index[idx]],
                num_heads=deform_num_heads, n_points=n_points,
                drop_path=interaction_drop_path_rate,
                norm_layer=norm_layer, with_cffn=with_cffn,
                cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                attn_type=interact_attn_type,
                with_proj=interaction_proj,
            )
            for idx in range(len(self.branch1_interaction_indexes))
        ])
        self.with_simple_fpn = with_simple_fpn
        # assert not is_dino
        out_dim = self.branch1.embed_dim
        self.branch3_is_cnn = "convnext" in self.branch3.pretrained
        # H,W of the final output is decided by branch3
        
        if with_simple_fpn: # ViT simple FPN
            assert not self.branch3_is_cnn
            
            # fpns are 4x, 2x, 1x, 1/2x
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(out_dim, out_dim, 2, 2),
                nn.GroupNorm(32, out_dim),
                nn.GELU(),
                nn.ConvTranspose2d(out_dim, out_dim, 2, 2)
            )
            self.fpn2 = nn.Sequential(nn.ConvTranspose2d(out_dim, out_dim, 2, 2))
            self.fpn3 = nn.Sequential(nn.Identity())
            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        
        elif self.branch3_is_cnn: # CNN regular FPN: no FPNs needed for upsampling
            self.fpn1 = nn.Identity()
            self.fpn2 = nn.Identity()
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.Identity()
        
        else: # ViT regular fpn
            # fpns are 4x, 2x, 1x, 1/2x
            dim_fpn1 = dims1[self.branch1_interaction_layer_index[self.out_interaction_indexes[0]]]
            dim_fpn2 = dims1[self.branch1_interaction_layer_index[self.out_interaction_indexes[1]]]
            
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(dim_fpn1, dim_fpn1, 2, 2),
                nn.GroupNorm(32, dim_fpn1),
                nn.GELU(),
                nn.ConvTranspose2d(dim_fpn1, dim_fpn1, 2, 2)
            )
            self.fpn2 = nn.Sequential(nn.ConvTranspose2d(dim_fpn2, dim_fpn2, 2, 2))
            self.fpn3 = nn.Sequential(nn.Identity())
            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.fpn1.apply(self._init_weights) 
        self.fpn2.apply(self._init_weights)
        self.fpn3.apply(self._init_weights)
        self.fpn4.apply(self._init_weights)

        dim1 = self.branch1.embed_dim
        dim2 = self.branch2.embed_dim
        dim3 = self.branch3.embed_dim
        
        if not with_simple_fpn:
            assert out_interaction_indexes is not None
            for out_idx in self.out_interaction_indexes:
                dim1_ = dims1[self.branch1_interaction_layer_index[out_idx]]
                dim2_ = dims2[self.branch2_interaction_layer_index[out_idx]]
                dim3_ = dims3[self.branch3_interaction_layer_index[out_idx]]
                self.create_intermediate_merge_module(out_idx, dim1_, dim2_, dim3_)
            
            self.merge_branch1 = nn.Sequential(
                nn.GroupNorm(32, dim1),
                nn.ReLU(inplace=True),
            )
            
            self.merge_branch2 = nn.Sequential(
                nn.Conv2d(dim2, dim1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, dim1),
                nn.ReLU(inplace=True),
            )
            
            self.merge_branch3 = nn.Sequential(
                nn.Conv2d(dim3, dim1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, dim1),
                nn.ReLU(inplace=True),
            )
        else:
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
            
            self.merge_branch3 = nn.Sequential(
                nn.Conv2d(dim3, dim1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, dim1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, dim1),
                nn.ReLU(inplace=True),
            )        
        
        self.merge_branch1.apply(self._init_weights)
        self.merge_branch2.apply(self._init_weights)
        self.merge_branch3.apply(self._init_weights)
        
        out_dim = dim1
        self.is_dino = is_dino


        self.fpn2.apply(self._init_weights)
        self.fpn3.apply(self._init_weights)
        self.fpn4.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        self.init_weights(pretrained)
        
        
    def create_intermediate_merge_module(self, idx, dim1_, dim2_, dim3_):
        merge_branch1 = nn.Sequential(
            nn.GroupNorm(32, dim1_),
            nn.ReLU(inplace=True),
        )
        merge_branch2 = nn.Sequential(
            nn.Conv2d(dim2_, dim1_, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, dim1_),
            nn.ReLU(inplace=True),
        )
        merge_branch3 = nn.Sequential(
            nn.Conv2d(dim3_, dim1_, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, dim1_),
            nn.ReLU(inplace=True),
        )
        merge_branch1.apply(self._init_weights)
        merge_branch2.apply(self._init_weights)
        merge_branch3.apply(self._init_weights)
        w1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        w2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        w3 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
        setattr(self, f"intermediate_merging_{idx}_branch1", merge_branch1)
        setattr(self, f"intermediate_merging_{idx}_branch2", merge_branch2)
        setattr(self, f"intermediate_merging_{idx}_branch3", merge_branch3)
        setattr(self, f"intermediate_merging_{idx}_w1", w1)
        setattr(self, f"intermediate_merging_{idx}_w2", w2)
        setattr(self, f"intermediate_merging_{idx}_w3", w3)
    @property
    def dtype(self):
        if 'convnext' in self.branch3.pretrained:
            return self.branch3.convnext_model.embeddings.patch_embeddings.weight.dtype
        else:
            return self.branch3.patch_embed.proj.weight.dtype
    
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
        scale_factor_1to3 = self.branch1_real_size / self.branch3_real_size
        if scale_factor_1to3 < 1:
            x1 = F.interpolate(x, scale_factor=scale_factor_1to3, mode='bilinear', align_corners=False)
        else:
            x1 = x.clone()
            
        scale_factor_2to3 = self.branch2_real_size / self.branch3_real_size
        if scale_factor_2to3 < 1:
            x2 = F.interpolate(x, scale_factor=scale_factor_2to3, mode='bilinear', align_corners=False)
        else:
            x2 = x.clone()

        x3 = x.clone()
        
        x1 = x1.type(self.dtype)
        x2 = x2.type(self.dtype)
        x3 = x3.type(self.dtype)

        deform_inputs_list = []
        for idx in range(len(self.interactions)):
            deform_inputs = {}
            if self.interact_attn_type == "deform":
                downsample_ratio1 = self.branch1.downsample_ratios[self.branch1_interaction_layer_index[idx]]
                downsample_ratio2 = self.branch2.downsample_ratios[self.branch2_interaction_layer_index[idx]]
                downsample_ratio3 = self.branch3.downsample_ratios[self.branch3_interaction_layer_index[idx]]
                deform_inputs["2to1"] = deform_inputs_1_vit(x1, x2, downsample_ratio1, downsample_ratio2)
                deform_inputs["1to2"] = deform_inputs_2_vit(x2, x1, downsample_ratio1, downsample_ratio2)
                deform_inputs["3to2"] = deform_inputs_1_vit(x2, x3, downsample_ratio2, downsample_ratio3)
                deform_inputs["2to3"] = deform_inputs_2_vit(x3, x2, downsample_ratio2, downsample_ratio3)
            else:
                deform_inputs["2to1"] = [None, None, None]
                deform_inputs["1to2"] = [None, None, None]
                deform_inputs["3to2"] = [None, None, None]
                deform_inputs["2to3"] = [None, None, None]
            deform_inputs_list.append(deform_inputs)
        

        # Patch embedding and position embedding
        if 'convnext' in self.branch1.pretrained:
            x1 = self.branch1.convnext_model.embeddings(x1)
            bs1, _, H1, W1 = x1.shape # here dim is not the final dim1
            x1 = x1.view(bs1, -1, H1*W1).transpose(1, 2) # (B,C,H,W) -> (B,C,N)-> (B,N,C)
        elif 'perceiver' in self.branch1.pretrained:
            x1, H1, W1 = self.branch1.visual_embed(x1)
            bs1, n1, dim1 = x1.shape
        else:
            x1, H1, W1 = self.branch1.patch_embed(x1)
            bs1, n1, dim1 = x1.shape
            if self.branch1.pos_embed is not None:
                pos_embed1 = self.branch1.pos_embed if not self.branch1_w_cls_token else self.branch1.pos_embed[:, 1:]
                pos_embed1 = self._get_pos_embed(pos_embed1.float(), (self.branch1.pretrain_img_size, self.branch1.pretrain_img_size),  
                                                (self.branch1.patch_size, self.branch1.patch_size), H1, W1) 
                x1 = x1 + pos_embed1
            x1 = self.branch1.pos_drop(x1)

        if 'convnext' in self.branch2.pretrained:
            x2 = self.branch2.convnext_model.embeddings(x2)
            bs2, _, H2, W2 = x2.shape
            x2 = x2.view(bs2, -1, H2*W2).transpose(1, 2) # (B,C,H,W) -> (B,C,N)-> (B,N,C)
        elif 'perceiver' in self.branch2.pretrained:
            x2, H2, W2 = self.branch2.visual_embed(x2)
            bs2, n2, dim2 = x2.shape
        else:
            x2, H2, W2 = self.branch2.patch_embed(x2)
            bs2, n2, dim2 = x2.shape
            if self.branch2.pos_embed is not None:
                pos_embed2 = self.branch2.pos_embed if not self.branch2_w_cls_token else self.branch2.pos_embed[:, 1:]
                pos_embed2 = self._get_pos_embed(pos_embed2.float(), (self.branch2.pretrain_img_size, self.branch2.pretrain_img_size), 
                                                (self.branch2.patch_size, self.branch2.patch_size), H2, W2) 
                x2 = x2 + pos_embed2
            x2 = self.branch2.pos_drop(x2)

        if 'convnext' in self.branch3.pretrained:
            x3 = self.branch3.convnext_model.embeddings(x3)
            bs3, _, H3, W3 = x3.shape
            x3 = x3.view(bs3, -1, H3*W3).transpose(1, 2) # (B,C,H,W) -> (B,C,N)-> (B,N,C)
        elif 'perceiver' in self.branch3.pretrained:
            x3, H3, W3 = self.branch3.visual_embed(x3)
            bs3, n3, dim3 = x3.shape
        else:
            x3, H3, W3 = self.branch3.patch_embed(x3)
            bs3, n3, dim3 = x3.shape
            if self.branch3.pos_embed is not None:
                pos_embed3 = self.branch3.pos_embed if not self.branch3_w_cls_token else self.branch3.pos_embed[:, 1:]
                pos_embed3 = self._get_pos_embed(pos_embed3.float(), (self.branch3.pretrain_img_size, self.branch3.pretrain_img_size), 
                                                (self.branch3.patch_size, self.branch3.patch_size), H3, W3) 
                x3 = x3 + pos_embed3
            x3 = self.branch3.pos_drop(x3)

        outs = []
        # Blocks and interactions
        for i, layer in enumerate(self.interactions):
            indexes1 = self.branch1_interaction_indexes[i]
            branch1_blocks = self.branch1.blocks[indexes1[0]:indexes1[-1] + 1]\
                if 'perceiver' not in self.branch1.pretrained else self.branch1.layers[indexes1[0]:indexes1[-1] + 1]
            indexes2 = self.branch2_interaction_indexes[i]
            branch2_blocks = self.branch2.blocks[indexes2[0]:indexes2[-1] + 1]\
                if 'perceiver' not in self.branch2.pretrained else self.branch2.layers[indexes2[0]:indexes2[-1] + 1]
            indexes3 = self.branch3_interaction_indexes[i]
            branch3_blocks = self.branch3.blocks[indexes3[0]:indexes3[-1] + 1]\
                if 'perceiver' not in self.branch3.pretrained else self.branch3.layers[indexes3[0]:indexes3[-1] + 1]

            x1, x2, x3, _, _, _, H1, W1, H2, W2, H3, W3 = layer(x1, x2, x3,
                        branch1_blocks, branch2_blocks, branch3_blocks,
                        H1=H1, W1=W1, H2=H2, W2=W2, H3=H3, W3=W3,
                        cls1=None, cls2=None, cls3=None,
                        deform_inputs=deform_inputs_list[i]
                        )

            if not self.with_simple_fpn and i in self.out_interaction_indexes:
                x1_ = x1.transpose(1, 2).view(bs1, -1, H1, W1)
                x1_ = getattr(self, f"intermediate_merging_{i}_branch1")(x1_)
                x1_ = x1_.type(torch.float32)
                x1_ = F.interpolate(x1_, size=(H3, W3), mode='bilinear', align_corners=False)
                x1_ = x1_.type(self.dtype)

                x2_ = x2.transpose(1, 2).view(bs2, -1, H2, W2)
                x2_ = getattr(self, f"intermediate_merging_{i}_branch2")(x2_)
                x2_ = x2_.type(torch.float32)
                x2_ = F.interpolate(x2_, size=(H3, W3), mode='bilinear', align_corners=False)
                x2_ = x2_.type(self.dtype)
                
                x3_ = x3.transpose(1, 2).view(bs3, -1, H3, W3)
                x3_ = getattr(self, f"intermediate_merging_{i}_branch3")(x3_)
                
                cur_out = x1_ * getattr(self, f"intermediate_merging_{i}_w1") + x2_ * getattr(self, f"intermediate_merging_{i}_w2") + x3_ * getattr(self, f"intermediate_merging_{i}_w3")

                outs.append(cur_out)
        # Branch merging
        x1 = x1.transpose(1, 2).view(bs1, self.branch1.embed_dim, H1, W1)
        x1 = self.merge_branch1(x1)
        x1 = x1.type(torch.float32)
        x1 = F.interpolate(x1, size=(H3, W3), mode='bilinear', align_corners=False)
        x1 = x1.type(self.dtype)

        x2 = x2.transpose(1, 2).view(bs2, self.branch2.embed_dim, H2, W2)
        x2 = self.merge_branch2(x2)
        x2 = x2.type(torch.float32)
        x2 = F.interpolate(x2, size=(H3, W3), mode='bilinear', align_corners=False)
        x2 = x2.type(self.dtype)
        
        x3 = x3.transpose(1, 2).view(bs3, self.branch3.embed_dim, H3, W3)
        x3 = self.merge_branch3(x3)
        
        out = x1 * self.w1 + x2 * self.w2 + x3 * self.w3
        
        if self.cal_flops:
            return
        
        # Outputs for fpn
        if self.is_dino:
            f2 = self.fpn2(out).contiguous().float()
            f3 = self.fpn3(out).contiguous().float()
            f4 = self.fpn4(out).contiguous().float()
            return [f2, f3, f4]
        
        
        if self.with_simple_fpn: # ViT or CNN simple fpn
            f1 = self.fpn1(out).contiguous().float()
            f2 = self.fpn2(out).contiguous().float()
            f3 = self.fpn3(out).contiguous().float()
            f4 = self.fpn4(out).contiguous().float()
        else:
            outs.append(out)
            assert len(outs) == 4
            f1 = self.fpn1(outs[0]).contiguous().float()
            f2 = self.fpn2(outs[1]).contiguous().float()
            f3 = self.fpn3(outs[2]).contiguous().float()
            f4 = self.fpn4(outs[3]).contiguous().float()
             
        
        return [f1, f2, f3, f4]