from llava.model.timm_convnext_monkey_patch import replace_timm_convnext
replace_timm_convnext()

import torch
from torch import nn
import timm
timm.layers.trunc_normal_ = torch.nn.init.normal_

def _init_weights_new(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        torch.nn.init.normal_(module.weight, std=.02) #!
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, std=.02) #!
        nn.init.zeros_(module.bias)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)

timm.models.convnext._init_weights = _init_weights_new
# print("!!! replace timm.layers.trunc_normal_ with torch.nn.init.normal_")
# print("!!! replace timm.models.convnext._init_weights with torch.nn.init.normal_")


from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
