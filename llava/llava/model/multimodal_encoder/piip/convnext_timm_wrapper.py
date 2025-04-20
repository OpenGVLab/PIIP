import torch
from torch import nn

import timm
from timm.models._manipulate import checkpoint_seq

class ConvNextTimmLayerWrapper(nn.Module):
    def __init__(self, layer, gradient_checkpoint=False):
        super().__init__()
        self.convnext_layer = layer
        self.gradient_checkpoint = gradient_checkpoint
        
        from timm.models.convnext import ConvNeXtBlock
        from llava.model.timm_convnext_monkey_patch import ConvNeXtBlockWithoutGamma
        if isinstance(layer, ConvNeXtBlockWithoutGamma) or isinstance(layer, ConvNeXtBlock):
            self.out_dim = layer.conv_dw.in_channels
        else:
            # downsampling layer
            assert isinstance(layer, nn.Sequential), type(layer)
            self.out_dim = layer[-1].out_channels
        self.relative_downsample_ratio = None
    
    def forward(self, x, H, W):
        assert self.relative_downsample_ratio is not None # should be assigned outside
        
        bs, n, dim = x.shape
        x = x.transpose(1,2).reshape(bs, dim, H, W) #(B,N,C)->(B,C,N)->(B,C,H,W)
        if self.gradient_checkpoint:
            x = checkpoint_seq(self.convnext_layer, x)
        else:
            x = self.convnext_layer(x)
        
        bs, dim_new, H_new, W_new = x.shape
        x = x.reshape(bs, dim_new, H_new * W_new).transpose(1,2) #(B,C,H,W)->(B,C,N)->(B,N,C)
        assert round(H/H_new) == round(W/W_new) == self.relative_downsample_ratio, f"{self.relative_downsample_ratio=}"
        return x, H_new, W_new
    
class ConvNextTimmWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.use_gradient_checkpointing = config.get("gradient_checkpointing", False)
        
        print("creating", config.get("pretrained"))

        self.convnext_model = timm.create_model(
            config.get("pretrained"),
            pretrained=config.get("load_timm_weights", True),
            num_classes=0,
        )
        # load_timm_weights = config.get("load_timm_weights", True)
        # print(f"!!! {load_timm_weights=}")
        
        
        if self.use_gradient_checkpointing:
            print("!!! Convnext Using gradient checkpointing")
        
        
        self.embed_dim = self.convnext_model.num_features
        self.blocks = []
        for stage in self.convnext_model.stages:
            if not isinstance(stage.downsample, nn.Identity):
                self.blocks.append(ConvNextTimmLayerWrapper(
                    stage.downsample, 
                    gradient_checkpoint=self.use_gradient_checkpointing,
                ))
            for layer in stage.blocks:
                self.blocks.append(ConvNextTimmLayerWrapper(
                    layer, 
                    gradient_checkpoint=self.use_gradient_checkpointing,
                ))
        
        self.img_size = config.get("img_size")
        
        
        self.downsample_ratios = config.get("downsample_ratios")
        
        assert len(self.downsample_ratios) == len(self.blocks)
        
        for idx, blk in enumerate(self.blocks):
            if idx == 0:
                blk.relative_downsample_ratio = 1
            else:
                blk.relative_downsample_ratio = self.downsample_ratios[idx] / self.downsample_ratios[idx-1]
        
        self.out_dims = [blk.out_dim for blk in self.blocks]
