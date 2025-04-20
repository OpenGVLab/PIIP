from types import NotImplementedType
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math

from ..multimodal_encoder.piip.piip_modules import Permute, ReshapeAndLN

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class TwoMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vit_hidden_size = 3200
        self.mlp1 = nn.Sequential(
            nn.Linear(self.vit_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, inputs):
        images, queries = inputs
        images = self.mlp1(images)
        queries = self.mlp2(queries)
        out = torch.cat([queries, images], dim=1)
        assert out.size(1) == 576 + 96, f"Expected 576+96, got {out.size(1)}"

        return out



class TwoBranchMLP(nn.Module):
    def __init__(self, config, out_dim):
        super().__init__()
        dim1 = config["dim1"]
        dim2 = config["dim2"]
        norm = config["norm"]
        self.out_dim = out_dim
        self.out_shape = config.get("out_shape", None)
        
        if norm == "ln":
            self.branch1 = nn.Sequential(
                Permute(0,2,3,1), # N,C,H,W -> N,H,W,C
                nn.Linear(dim1, out_dim),
                ReshapeAndLN(),
                nn.GELU(),
                Permute(0,2,3,1), # N,C,H,W -> N,H,W,C
                nn.Linear(out_dim, out_dim),
                Permute(0,3,1,2), # N,H,W,C -> N,C,H,W
            )
            self.branch2 = nn.Sequential(
                Permute(0,2,3,1), # N,C,H,W -> N,H,W,C
                nn.Linear(dim2, out_dim),
                ReshapeAndLN(),
                nn.GELU(),
                Permute(0,2,3,1), # N,C,H,W -> N,H,W,C
                nn.Linear(out_dim, out_dim),
                Permute(0,3,1,2), # N,H,W,C -> N,C,H,W
            )
        elif norm == "gn":
            self.branch1 = nn.Sequential(
                Permute(0,2,3,1), # N,C,H,W -> N,H,W,C
                nn.Linear(dim1, out_dim),
                Permute(0,3,1,2), # N,H,W,C -> N,C,H,W
                nn.GroupNorm(num_groups=32, num_channels=out_dim),
                nn.GELU(),
                Permute(0,2,3,1), # N,C,H,W -> N,H,W,C
                nn.Linear(out_dim, out_dim),
                Permute(0,3,1,2), # N,H,W,C -> N,C,H,W
            )
            self.branch2 = nn.Sequential(
                Permute(0,2,3,1), # N,C,H,W -> N,H,W,C
                nn.Linear(dim2, out_dim),
                Permute(0,3,1,2), # N,H,W,C -> N,C,H,W
                nn.GroupNorm(num_groups=32, num_channels=out_dim),
                nn.GELU(),
                Permute(0,2,3,1), # N,C,H,W -> N,H,W,C
                nn.Linear(out_dim, out_dim),
                Permute(0,3,1,2), # N,H,W,C -> N,C,H,W
            )
        else:
            raise NotImplementedError
    
    
    def forward(self, x):
        x1, x2 = x
        bs1, dim1, H1, W1 = x1.shape
        bs2, dim2, H2, W2 = x2.shape
        dtype = x1.dtype
        
        if self.out_shape is not None:
            x1 = F.interpolate(x1.float(), size=(self.out_shape, self.out_shape), mode='bilinear', align_corners=False).to(dtype)
            x2 = F.interpolate(x2.float(), size=(self.out_shape, self.out_shape), mode='bilinear', align_corners=False).to(dtype)
        else:
            x1 = F.interpolate(x1.float(), size=(H2, W2), mode='bilinear', align_corners=False).to(dtype)

        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        
        out = x1 + x2
        if self.out_shape is not None:
            out = out.reshape(bs2, self.out_dim, self.out_shape * self.out_shape).transpose(1, 2)
        else:
            out = out.reshape(bs2, self.out_dim, H2 * W2).transpose(1, 2)
        return out





def build_vision_projector(config, delay_load=False, **kwargs):
    
    vision_tower_name = getattr(config, 'mm_vision_tower', getattr(config, 'vision_tower', None))
    if vision_tower_name is not None and vision_tower_name.endswith(".py"):
        config_file = __import__(vision_tower_name.replace(".py", "").replace("/", "."), fromlist=[''])
        if "projector" in config_file.model:
            cfg = config_file.model["projector"]
            
            if cfg["type"] == "separate_2branch_v1":
                return TwoBranchMLP(cfg, config.hidden_size)
            else:
                raise NotImplementedError(cfg["type"])
        else:
            print("!!! falling back to default projector")
            pass
    
    
    
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    if projector_type == 'two_mlp':
        return TwoMLP(config)

    raise ValueError(f'Unknown projector type: {projector_type}')
