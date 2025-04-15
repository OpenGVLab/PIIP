import torch
from torch import nn

from transformers.models.convnext.modeling_convnext import ConvNextModel, ConvNextLayer
from mmdet.models.builder import BACKBONES


class ConvNextLayerWrapper(nn.Module):
    def __init__(self, layer, gradient_checkpoint_func=None): #raw_img_size, 
        super().__init__()
        self.convnext_layer = layer
        self.gradient_checkpoint_func = gradient_checkpoint_func
        if isinstance(layer, ConvNextLayer):
            self.out_dim = layer.dwconv.in_channels
        else:
            # downsampling layer
            assert isinstance(layer, nn.Sequential), type(layer)
            self.out_dim = layer[-1].out_channels
        self.relative_downsample_ratio = None
    
    def forward(self, x, H, W):
        assert self.relative_downsample_ratio is not None # should be assigned outside
        
        bs, n, dim = x.shape
        x = x.transpose(1,2).reshape(bs, dim, H, W) #(B,N,C)->(B,C,N)->(B,C,H,W)
        
        if self.gradient_checkpoint_func is not None:
            x = self.gradient_checkpoint_func(
                        self.convnext_layer.__call__,
                        x
                    )
        else:
            x = self.convnext_layer(x)
        
        bs, dim_new, H_new, W_new = x.shape
        x = x.reshape(bs, dim_new, H_new * W_new).transpose(1,2) #(B,C,H,W)->(B,C,N)->(B,N,C)
        assert round(H/H_new) == round(W/W_new) == self.relative_downsample_ratio, f"{self.relative_downsample_ratio=}"
        return x, H_new, W_new


@BACKBONES.register_module()
class ConvNextHFWrapper(nn.Module):
    def __init__(self, pretrained, real_size, downsample_ratios=None, 
                 single_branch=False, with_simple_fpn=False, drop_path_rate=None,
                 cal_flops_wo_fpn=False): # , cal_flops=False
        super().__init__()
        
        
        self.use_gradient_checkpointing = False
        
        kwargs = {}
        if drop_path_rate is not None:
            kwargs["drop_path_rate"] = drop_path_rate
        self.convnext_model = ConvNextModel.from_pretrained(
            pretrained,
            **kwargs
        )
        if self.use_gradient_checkpointing:
            self.convnext_model.gradient_checkpointing_enable()
            print("!!! Convnext Using gradient checkpointing")
        
        
        self.embed_dim = self.convnext_model.config.hidden_sizes[-1]
        
        self.blocks = []
        for stage in self.convnext_model.encoder.stages:
            if not isinstance(stage.downsampling_layer, nn.Identity):
                self.blocks.append(ConvNextLayerWrapper(
                    stage.downsampling_layer, 
                    gradient_checkpoint_func=self.convnext_model.encoder._gradient_checkpointing_func if self.use_gradient_checkpointing else None,
                ))
            for layer in stage.layers:
                self.blocks.append(ConvNextLayerWrapper(
                    layer, 
                    gradient_checkpoint_func=self.convnext_model.encoder._gradient_checkpointing_func if self.use_gradient_checkpointing else None,
                ))
        
        self.pretrained = pretrained
        self.real_size = real_size
        
        self.pos_embed = nn.Identity() # for calculating flops
        self.patch_embed = nn.Identity() # for calculating flops
        
        self.with_simple_fpn = with_simple_fpn
        self.single_branch = single_branch
        self.cal_flops_wo_fpn = cal_flops_wo_fpn
        
        # if not cal_flops:
        if with_simple_fpn:
            out_dim = self.embed_dim
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(out_dim, out_dim, 2, 2),
                nn.GroupNorm(32, out_dim),
                nn.GELU(),
                nn.ConvTranspose2d(out_dim, out_dim, 2, 2),
                nn.GroupNorm(32, out_dim),
                nn.GELU(),
                nn.ConvTranspose2d(out_dim, out_dim, 2, 2),
            )
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(out_dim, out_dim, 2, 2),
                nn.GroupNorm(32, out_dim),
                nn.GELU(),
                nn.ConvTranspose2d(out_dim, out_dim, 2, 2)
            )
            self.fpn3 = nn.Sequential(nn.ConvTranspose2d(out_dim, out_dim, 2, 2))
            self.fpn4 = nn.Sequential(nn.Identity())
            
        else:
            if not single_branch:
                self.downsample_ratios = downsample_ratios
                assert len(downsample_ratios) == len(self.blocks)
                
                for idx, blk in enumerate(self.blocks):
                    if idx == 0:
                        blk.relative_downsample_ratio = 1
                    else:
                        blk.relative_downsample_ratio = downsample_ratios[idx] / downsample_ratios[idx-1]
                
                self.out_dims = [blk.out_dim for blk in self.blocks]
    
    @property
    def dtype(self):
        return self.convnext_model.embeddings.patch_embeddings.weight.dtype
    
    def forward(self, pixel_values):
        assert self.single_branch
        # for single-branch baselines
        pixel_values = pixel_values.type(self.dtype) #(B, 3, H, W)
        embedding_output = self.convnext_model.embeddings(pixel_values)

        encoder_outputs = self.convnext_model.encoder(embedding_output, output_hidden_states=True)

        last_hidden_state = encoder_outputs[0]
        hidden_states = encoder_outputs[1]
        
        if self.cal_flops_wo_fpn:
            return
        
        if self.with_simple_fpn: 
            f1 = self.fpn1(last_hidden_state).to(torch.float32).contiguous() #(B, dim1, H/4, W/4)
            f2 = self.fpn2(last_hidden_state).to(torch.float32).contiguous() #(B, dim2, H/8, W/8)
            f3 = self.fpn3(last_hidden_state).to(torch.float32).contiguous() #(B, dim3, H/16, W/16)
            f4 = self.fpn4(last_hidden_state).to(torch.float32).contiguous() #(B, dim4, H/32, W/32)
            assert f1.shape[-1] == pixel_values.shape[-1] / 4, f"{f1.shape=}, {pixel_values.shape=}"
            assert f2.shape[-1] == pixel_values.shape[-1] / 8, f"{f2.shape=}, {pixel_values.shape=}"
            assert f3.shape[-1] == pixel_values.shape[-1] / 16, f"{f3.shape=}, {pixel_values.shape=}"
            assert f4.shape[-1] == pixel_values.shape[-1] / 32, f"{f4.shape=}, {pixel_values.shape=}"
            
            return [f1, f2, f3, f4]
        else: # regular fpn
            assert len(hidden_states) == 5
            ret = [h.to(torch.float32).contiguous() for h in hidden_states[1:]]
            assert ret[0].shape[-1] == pixel_values.shape[-1] / 4, f"{ret[0].shape=}, {pixel_values.shape=}"
            assert ret[1].shape[-1] == pixel_values.shape[-1] / 8, f"{ret[1].shape=}, {pixel_values.shape=}"
            assert ret[2].shape[-1] == pixel_values.shape[-1] / 16, f"{ret[2].shape=}, {pixel_values.shape=}"
            assert ret[3].shape[-1] == pixel_values.shape[-1] / 32, f"{ret[3].shape=}, {pixel_values.shape=}"
            return ret