# --------------------------------------------------------
# PIIP
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
os.environ["CHECK_FINITE"] = "0"
import glob

import torch
from torch import nn


from flops_counter import flops_to_string, params_to_string, get_model_complexity_info


from types import SimpleNamespace
from llava.model.multimodal_encoder.builder import build_vision_tower


def n_params(*args):
    count = 0
    for m in args:
        if isinstance(m, nn.Module):
            count += sum(p.numel() for p in m.parameters() if p.requires_grad)
        elif isinstance(m, nn.Parameter):
            if m.requires_grad:
                count += m.numel()
        else:
            raise NotImplementedError(m)

    return f"{count / 1000 / 1000 : .1f} M"

def sa_flops(h, w, dim):
    return 2 * h * w * h * w * dim

def ca_flops(h1, w1, h2, w2, dim):
    return 2 * h1 * w1 * h2 * w2 * dim

def deformable_attn_flops(h, w, K, dim): # only support one level
    return 5 * h * w * K * dim

def window_sa_flops(h, w, dim, window_size):
    return 2 * h * w * window_size * window_size * dim


def get_sa_flops_clip(model, input_shape):
    # model: CLIPVisionTransformer
    _, H, W = input_shape
    if hasattr(model, "blocks"):
        depth = len(model.blocks)
        patch_size = model.patch_embed.patch_size[0]
        dim = model.patch_embed.proj.out_channels
    else:
        depth = len(model.encoder.layers)
        patch_size = model.embeddings.patch_size
        dim = model.config.hidden_size
    
    print("embed dim", dim, "shape", H, W, "patch size", patch_size)
    ret = sa_flops(H // patch_size, W // patch_size, dim) * depth
    return ret


def get_backbone_flops_clip(model, input_shape, model_config):
    with torch.cuda.amp.autocast():
        flops, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=True)
    

    if model.vision_tower_type == "clip":
        fl = get_sa_flops_clip(model.vision_tower.vision_model, input_shape)
        flops += fl
        print("self attention flops", flops_to_string(fl))
    
    elif model.vision_tower_type.startswith("piip"):
        input_shape1 = model.vision_tower.branch1.img_size
        if hasattr(model.vision_tower.branch1, "clip_model"):
            fl = get_sa_flops_clip(model.vision_tower.branch1.clip_model, (None, input_shape1, input_shape1))
        else:
            assert hasattr(model.vision_tower.branch1, "convnext_model")
            fl = 0
            
        flops += fl
        print("branch1 self attention flops", flops_to_string(fl))
    
        input_shape2 = model.vision_tower.branch2.img_size
        if hasattr(model.vision_tower.branch2, "clip_model"):
            fl = get_sa_flops_clip(model.vision_tower.branch2.clip_model, (None, input_shape2, input_shape2))
        else:
            assert hasattr(model.vision_tower.branch2, "convnext_model")
            fl = 0
        flops += fl
        print("branch2 self attention flops", flops_to_string(fl))

        if model.vision_tower_type == "piip_3branch":
            input_shape3 = model.vision_tower.branch3.img_size
            fl = get_sa_flops_clip(model.vision_tower.branch3.clip_model, (None, input_shape3, input_shape3))
            flops += fl
            print("branch3 self attention flops", flops_to_string(fl))
        
        
        fl = 0
        for interaction in model.vision_tower.interactions:
            for interaction_unit in interaction.interaction_units:
                if model_config["interact_attn_type"] == 'deform':
                    fl += deformable_attn_flops(
                        interaction_unit.branch2_img_size // interaction_unit.branch2_patch_size, 
                        interaction_unit.branch2_img_size // interaction_unit.branch2_patch_size, 
                        4,
                        interaction_unit.branch2_dim
                    )
                    fl += deformable_attn_flops(
                        interaction_unit.branch1_img_size // interaction_unit.branch1_patch_size, 
                        interaction_unit.branch1_img_size // interaction_unit.branch1_patch_size, 
                        4,
                        interaction_unit.branch1_dim
                    )
                elif model_config["interact_attn_type"] == 'normal':
                    fl += ca_flops(
                        interaction_unit.branch2_img_size // interaction_unit.branch2_patch_size, 
                        interaction_unit.branch2_img_size // interaction_unit.branch2_patch_size, 
                        interaction_unit.branch1_img_size // interaction_unit.branch1_patch_size, 
                        interaction_unit.branch1_img_size // interaction_unit.branch1_patch_size, 
                        interaction_unit.branch2_dim
                    )
                    fl += ca_flops(
                        interaction_unit.branch1_img_size // interaction_unit.branch1_patch_size, 
                        interaction_unit.branch1_img_size // interaction_unit.branch1_patch_size, 
                        interaction_unit.branch2_img_size // interaction_unit.branch2_patch_size, 
                        interaction_unit.branch2_img_size // interaction_unit.branch2_patch_size, 
                        interaction_unit.branch1_dim
                    )
                else:
                    raise NotImplementedError
        print(f"interaction deformable attention flops {flops_to_string(fl)}")
        
        flops += fl
    else:
        raise NotImplementedError(model.vision_tower_type)
    
        
    
    return flops_to_string(flops, precision=1), params_to_string(params, precision=1)


def main(config_name, out_file=None):
    args = SimpleNamespace(
        vision_tower=config_name,
        mm_vision_select_layer=-2, # for CLIP
    )
    
    print("building model")
    model = build_vision_tower(args, delay_load=False)
    
    if model.vision_tower_type == "clip":
        model.vision_tower.requires_grad_(True)
    
    shape = model.input_image_size
    
    assert torch.cuda.is_available()
    model.cuda()
    model.eval()
    
    if shape is None:
        shape = 336
        
    h = w = shape
    input_shape = (3, h, w)
    
    branch1_params = branch2_params = branch3_params = interaction_params = None
    
    if hasattr(model, "branch1"):
        assert model.branch1_model_type == "clip_hf"
        branch1_params = n_params(model.branch1.clip_model.embeddings, model.branch1.blocks)
    if hasattr(model, "branch2"):
        assert model.branch2_model_type == "clip_hf"
        branch2_params = n_params(model.branch2.clip_model.embeddings, model.branch2.blocks)
    if hasattr(model, "branch3"):
        assert model.branch3_model_type == "clip_hf"
        branch3_params = n_params(model.branch3.clip_model.embeddings, model.branch3.blocks)
    if hasattr(model, "interactions"):
        interaction_params = n_params(model.interactions)
    
    flops, params = get_backbone_flops_clip(model, input_shape, model.config_backbone)
    
    merge_params = None
    if hasattr(model, "merge_branch3"):
        merge_params = n_params(model.merge_branch1, model.merge_branch2, model.merge_branch3)
    elif hasattr(model, "merge_branch2"):
        merge_params = n_params(model.merge_branch1, model.merge_branch2)
    
    print(os.path.basename(config), "FLOPs", flops, "Params", params, "Shape", h)
    if out_file is not None:
        print(os.path.basename(config).replace(".py", "").ljust(60), "FLOPs", flops, "Params", params, "Shape", h, 
              "Branch1", branch1_params, "Branch2", branch2_params, "Branch3", branch3_params, "Interaction", interaction_params, "Merge", merge_params, file=out_file, flush=True)
       


if __name__ == "__main__":
    config_list = [
        # "configs/piip-llava_convnext-b_clip-l_640-224_7B.py",
        "configs/*.py",
    ]
    
    new_config_list = []
    for pattern in config_list:
        assert len(glob.glob(pattern)) > 0, pattern
        for file in glob.glob(pattern):
            new_config_list.append(file)
    config_list = sorted(new_config_list)
    
    with open(f"flops_llava.txt", "w") as f:
        for config in config_list:
            try:
                main(config, out_file=f)
            except:
                print("ERR CONFIG", config)
                raise
