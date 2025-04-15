# --------------------------------------------------------
# PIIP
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import glob

import torch
from torch import nn


from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string, print_model_with_flops
from mmcv.cnn import get_model_complexity_info

from mmcv import Config
from mmdet.models import build_detector
import mmdet


def n_params(*args, return_int=False):
    count = 0
    for m in args:
        if isinstance(m, nn.Module):
            count += sum(p.numel() for p in m.parameters() if p.requires_grad)
        elif isinstance(m, nn.Parameter):
            if m.requires_grad:
                count += m.numel()
        elif isinstance(m, list):
            for ele in m:
                count += n_params(ele, return_int=True)
        else:
            raise NotImplementedError(m)

    if return_int:
        return count
    return f"{count / 1000 / 1000 : .1f} M"

def sa_flops(h, w, dim):
    return 2 * h * w * h * w * dim

def ca_flops(h1, w1, h2, w2, dim):
    return 2 * h1 * w1 * h2 * w2 * dim

def deformable_attn_flops(h, w, K, dim): # only support one level
    return 5 * h * w * K * dim

def window_sa_flops(h, w, dim, window_size):
    return 2 * h * w * window_size * window_size * dim


def get_sa_flops(module, input_shape):
    _, H, W = input_shape
    depth = module.get("depth", 12)
    if "window_attn" in module:
        window_attn = module["window_attn"]
        window_size = module["window_size"]
    elif "window_size" in module and isinstance(module["window_size"], int): # window attention in each layer
        window_attn =[True] * depth
        window_size = [module["window_size"]] * depth
    else:
        # print(module, "has no attr named window_attn. set to default")
        window_attn = [False] * depth
        window_size = [None] * depth
        
    dim = module["embed_dim"]
    print("embed dim", dim, "shape", H, W)
    ret = 0
    for flag, size in zip(window_attn, window_size):
        # print(flag, size)
        if flag == True:
            ret += window_sa_flops(H // 16, W // 16, dim, size)
        else:
            ret += sa_flops(H // 16, W // 16, dim)
    return ret



def get_backbone_flops(model, input_shape, model_config):
    # with torch.cuda.amp.autocast():
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False)
    

    if hasattr(model, "blocks") and "resnet" not in model_config["pretrained"] and "convnext" not in model_config["pretrained"]:
        fl = get_sa_flops(model_config, input_shape)
        flops += fl
        print("self attention flops", flops_to_string(fl))
    
    if "branch1" in model_config and "resnet" not in model_config["branch1"]["pretrained"] and "convnext" not in model_config["branch1"]["pretrained"]:
        input_shape1 = model.branch1_real_size
        fl = get_sa_flops(model_config["branch1"], (None, input_shape1, input_shape1))
        flops += fl
        print("branch1 self attention flops", flops_to_string(fl))
    
    if "branch2" in model_config and "resnet" not in model_config["branch2"]["pretrained"] and "convnext" not in model_config["branch2"]["pretrained"]:
        input_shape2 = model.branch2_real_size
        fl = get_sa_flops(model_config["branch2"], (None, input_shape2, input_shape2))
        flops += fl
        print("branch2 self attention flops", flops_to_string(fl))
    
    if "branch3" in model_config and "resnet" not in model_config["branch3"]["pretrained"] and "convnext" not in model_config["branch3"]["pretrained"]:
        input_shape3 = model.branch3_real_size
        fl = get_sa_flops(model_config["branch3"], (None, input_shape3, input_shape3))
        flops += fl
        print("branch3 self attention flops", flops_to_string(fl))
    
    if "branch4" in model_config and "resnet" not in model_config["branch4"]["pretrained"] and "convnext" not in model_config["branch4"]["pretrained"]:
        input_shape4 = model.branch4_real_size
        fl = get_sa_flops(model_config["branch4"], (None, input_shape4, input_shape4))
        flops += fl
        print("branch4 self attention flops", flops_to_string(fl))
    
    if hasattr(model, "interactions"):
        fl = 0
        for interaction in model.interactions:
            for interaction_unit in interaction.interaction_units:
                if model_config["interact_attn_type"] == 'deform':
                    if hasattr(interaction_unit, "branch1to2_injector"):
                        fl += deformable_attn_flops(
                            interaction_unit.branch2_feat_size, 
                            interaction_unit.branch2_feat_size, 
                            4,
                            interaction_unit.branch2_dim
                        )
                    if hasattr(interaction_unit, "branch2to1_injector"):
                        fl += deformable_attn_flops(
                            interaction_unit.branch1_feat_size, 
                            interaction_unit.branch1_feat_size, 
                            4,
                            interaction_unit.branch1_dim
                        )
                elif model_config["interact_attn_type"] == 'normal':
                    if hasattr(interaction_unit, "branch1to2_injector"):
                        fl += ca_flops(
                            interaction_unit.branch2_feat_size, 
                            interaction_unit.branch2_feat_size, 
                            interaction_unit.branch1_feat_size, 
                            interaction_unit.branch1_feat_size,
                            interaction_unit.branch2_dim
                        )
                    if hasattr(interaction_unit, "branch2to1_injector"):
                        fl += ca_flops(
                            interaction_unit.branch1_feat_size, 
                            interaction_unit.branch1_feat_size, 
                            interaction_unit.branch2_feat_size, 
                            interaction_unit.branch2_feat_size, 
                            interaction_unit.branch1_dim
                        )
                else:
                    raise NotImplementedError
        print(f"interaction deformable attention flops {flops_to_string(fl)}")
        
        flops += fl
    
    return flops_to_string(flops, precision=1), params_to_string(params, precision=1)

def main(config_name, out_file=None):
    cfg = Config.fromfile(config)

    
    cfg.model.neck.type = "FPNLateralOnly"
    cfg.model.backbone.cal_flops = True
        
    print("building detector")
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    
    model = model.backbone
    
    
    
    config_backbone = cfg.get("model").get("backbone")
    # import ipdb; ipdb.set_trace()
    if "branch4" in config_backbone:
        shape = config_backbone["branch4"]["real_size"]
    elif "branch3" in config_backbone:
        shape = config_backbone["branch3"]["real_size"]
    elif "branch2" in config_backbone:
        shape = config_backbone["branch2"]["real_size"]
    elif "real_size" in config_backbone:
        shape = config_backbone["real_size"]
    else:
        print("set shape=1024")
        shape = 1024
    
    assert torch.cuda.is_available()
    model.cuda()
    model.eval()
    
    if shape is None:
        shape = 1024
    h = w = shape
    input_shape = (3, h, w)
    
    branch1_params = branch2_params = branch3_params = branch4_params = interaction_params = None
    
    if hasattr(model, "branch1"):
        if hasattr(model.branch1, "pos_embed") and model.branch1.pos_embed is None:
            branch1_params = n_params(model.branch1.blocks, model.branch1.patch_embed)
        else:
            branch1_params = n_params(model.branch1.blocks, model.branch1.pos_embed, model.branch1.patch_embed)
    if hasattr(model, "branch2"):
        if hasattr(model.branch2, "pos_embed") and model.branch2.pos_embed is None:
            branch2_params = n_params(model.branch2.blocks, model.branch2.patch_embed)
        else:
            branch2_params = n_params(model.branch2.blocks, model.branch2.pos_embed, model.branch2.patch_embed)
    if hasattr(model, "branch3"):
        if hasattr(model.branch3, "pos_embed") and model.branch3.pos_embed is None:
            branch3_params = n_params(model.branch3.blocks, model.branch3.patch_embed)
        else:
            branch3_params = n_params(model.branch3.blocks, model.branch3.pos_embed, model.branch3.patch_embed)
    if hasattr(model, "branch4"):
        if hasattr(model.branch4, "pos_embed") and model.branch4.pos_embed is None:
            branch4_params = n_params(model.branch4.blocks, model.branch4.patch_embed)
        else:
            branch4_params = n_params(model.branch4.blocks, model.branch4.pos_embed, model.branch4.patch_embed)
    if hasattr(model, "interactions"):
        interaction_params = n_params(model.interactions)
    
    
    flops, params = get_backbone_flops(model, input_shape, config_backbone)
    
    
    merge_params = None
    if hasattr(model, "merge_branch4"):
        merge_params = n_params(model.merge_branch1, model.merge_branch2, model.merge_branch3, model.merge_branch4)
    elif hasattr(model, "merge_branch3"):
        merge_params = n_params(model.merge_branch1, model.merge_branch2, model.merge_branch3)
    elif hasattr(model, "merge_branch2"):
        merge_params = n_params(model.merge_branch1, model.merge_branch2)
    
    
    print(os.path.basename(config), "FLOPs", flops, "Params", params, "Shape", h)
    if out_file is not None:
        print(os.path.basename(config).replace(".py", "").ljust(60), "FLOPs", flops, "Params", params, "Shape", h, 
              "Branch1", branch1_params, "Branch2", branch2_params, "Branch3", branch3_params, "Branch4", branch4_params, "Interaction", interaction_params, "Merge", merge_params, file=out_file, flush=True)
       


if __name__ == "__main__":
    config_list = [
        # "configs/piip/convnext/*.py",
        # "configs/piip/2branch/dino_4scale_internvit_h6b_1024_768_fpn_1x_coco_bs32_ms.py",
        "configs/piip/3branch/*.py",
        "configs/piip/2branch/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms.py",
        "configs/piip/4branch/*.py",
        "configs/piip/baseline/*.py",
    ]
    
    new_config_list = []
    for pattern in config_list:
        assert len(glob.glob(pattern)) > 0, pattern
        for file in glob.glob(pattern):
            if "dino" in file or "uniperceiver" in file:
                continue
            if "speedup" in file:
                continue
            new_config_list.append(file)
    # config_list = sorted(new_config_list)
    config_list = new_config_list
    
    
    with open(f"flops.txt", "w") as f:
        for config in config_list:
            try:
                main(config, out_file=f)
            except:
                print("ERR CONFIG", config)
                raise
