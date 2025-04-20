import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPVisionModel
from einops import rearrange


class CLIPLayerWrapper(nn.Module):
    def __init__(self, layer, config=None, gradient_checkpoint_func=None):
        super().__init__()
        self.clip_layer = layer
        self.gradient_checkpoint_func = gradient_checkpoint_func
        
    
    def forward(self, x, H, W):
        if self.gradient_checkpoint_func is not None:
            raise NotImplementedError
            return self.gradient_checkpoint_func(
                        self.clip_layer.__call__,
                        x,
                        None,
                        None,
                        False,
                    )[0]
        else:
            return self.clip_layer(x, attention_mask=None, causal_attention_mask=None, output_attentions=False)[0]
        

class CLIPHFWrapper(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        self.patch_size = config.get("patch_size")
        self.pretrain_patch_size = config.get("pretrain_patch_size")
        
        self.use_gradient_checkpointing = config.get("gradient_checkpointing", False)
        
        print("creating", config.get("pretrained"))

        self.clip_model = CLIPVisionModel.from_pretrained(
            config.get("pretrained"),
        )
        if self.use_gradient_checkpointing:
            self.clip_model.gradient_checkpointing_enable()
            print("!!! CLIP Using gradient checkpointing")
        self.clip_model = self.clip_model.vision_model
        
        
        self.embed_dim = self.clip_model.config.hidden_size
        self.blocks = [
            CLIPLayerWrapper(
                layer, 
                config=self.clip_model.config,
                gradient_checkpoint_func=self.clip_model.encoder._gradient_checkpointing_func if self.use_gradient_checkpointing else None,
            ) for layer in self.clip_model.encoder.layers
        ]
        
        self.pretrain_size = self.clip_model.config.image_size
        self.img_size = config.get("img_size")
        
        assert self.pretrain_patch_size == self.clip_model.embeddings.patch_size, f"{self.clip_model.embeddings.patch_size=}"
        
        
        if self.patch_size != self.pretrain_patch_size:
            print(f"!!! patch_size {self.patch_size} pretrain_patch_size {self.pretrain_patch_size}. Resize patch embed to {self.patch_size}")

            self.clip_model.embeddings.patch_embedding.weight.data = F.interpolate(self.clip_model.embeddings.patch_embedding.weight, size=(self.patch_size, self.patch_size), mode='bicubic', align_corners=False)
        
        assert self.img_size % self.patch_size == 0, f"{self.img_size=}, {self.patch_size=}"