# --------------------------------------------------------
# PIIP
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os

from piip_2branch import PIIPTwoBranch
from piip_3branch import PIIPThreeBranch
from piip_4branch import PIIPFourBranch

from deit import vit_models
from intern_vit_6b import InternViT6B


def convert_mmdet_config(cfg, config_name):
    # for flops calculation
    cfg.pop("pretrained", None)
    cfg.pop("start_level", None)
    cfg.pop("_delete_", None)
    branch_pop_keys = ["with_fpn", "img_norm_cfg", "layerscale_force_fp32"]
    for branch in ["branch1", "branch2", "branch3", "branch4"]:
        if branch in cfg:
            if 'beit' in cfg[branch]['pretrained']:
                raise NotImplementedError("flops calculation not implemented for beit")
            elif 'perceiver' in cfg[branch]['pretrained']:
                raise NotImplementedError("flops calculation not implemented for uni-perceiver")
            
            for key in branch_pop_keys:
                cfg[branch].pop(key, None)
            cfg[branch]["use_cls_token"] = False
            cfg[branch]["img_size"] = cfg[branch]["real_size"]
            cfg[branch].pop("real_size")
            
            cfg[branch]["model_type"] = "augreg"
            if "is_branch1_deit" in cfg[branch]:
                cfg[branch]["model_type"] = "deit" if cfg[branch]["is_branch1_deit"] else "augreg"
            if "deit" in cfg[branch]["pretrained"]:
                cfg[branch]["model_type"] = "deit"
    
    cfg.pop("output_dtype", None)
    cfg.pop("out_indices", None)
    cfg.pop("with_fpn", None)
    if cfg["type"] in ["vit_models", "InternViT6B"]:
        if "upernet" in config_name:
            pass
        elif "mask_rcnn" in config_name:
            cfg["img_size"] = 1024
        else:
            pass
        cfg["use_cls_token"] = False
        
    return cfg


def read_config(config_path):
    if config_path.startswith("configs/"):
        config_path = config_path[8:]
    assert config_path.endswith(".py") and os.path.exists("configs/" + config_path)
    config_file = __import__("configs." + config_path.replace(".py", "").replace("/", "."), fromlist=[''])
    
    config_name = os.path.basename(config_path)
    
    cfg = config_file.model["backbone"]
    if "type" in cfg:
        cfg = convert_mmdet_config(cfg, config_name)
    
    if "piip_3branch" in config_name or cfg.get("type") == "PIIPThreeBranch":
        img_size = cfg["branch3"]["img_size"]
        model_cls = PIIPThreeBranch
    elif "piip_4branch" in config_name or cfg.get("type") == "PIIPFourBranch":
        img_size = cfg["branch4"]["img_size"]
        model_cls = PIIPFourBranch
    elif "piip_2branch" in config_name or cfg.get("type") == "PIIPTwoBranch":
        img_size = cfg["branch2"]["img_size"]
        model_cls = PIIPTwoBranch
    elif "augreg" in config_name or "internvit" in config_name:
        img_size = cfg["img_size"]
        model_cls = InternViT6B
    elif "deit" in config_name or cfg.get("type") == "vit_models":
        img_size = cfg["img_size"]
        model_cls = vit_models
    else:
        raise NotImplementedError
    
    if hasattr(config_file, "input_size"):
        print(f"!!! set input size to {config_file.input_size}")
        img_size = config_file.input_size
    
    cfg.pop("type", None)
    return cfg, img_size, model_cls
