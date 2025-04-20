import dataclasses
import os
import re
import sys

from tqdm import tqdm

import open_clip
from open_clip.hf_model import HFTextEncoder
from open_clip.model import CLIPVisionCfg
from transformers import CLIPVisionConfig, VisionTextDualEncoderConfig

from modeling_clip import OpenCLIPVisionTextDualEncoderModel


VISION_CONFIG_MAP = {
    "layers": "num_hidden_layers",
    "width": "hidden_size",
    "patch_size": "patch_size",
    "image_size": "image_size",
}
STATE_DICT_PATTERNS = [
    # Vision
    (r"visual\.class_embedding", "vision_model.vision_model.embeddings.class_embedding"),
    (r"visual\.positional_embedding", "vision_model.vision_model.embeddings.position_embedding.weight"),
    (r"visual\.conv1\.(\w+)", "vision_model.vision_model.embeddings.patch_embedding.{0}"),
    (r"visual\.ln_pre\.(\w+)", "vision_model.vision_model.pre_layrnorm.{0}"),
    (r"visual\.ln_post\.(\w+)", "vision_model.vision_model.post_layernorm.{0}"),
    (
        r"visual\.transformer\.resblocks\.(\w+)\.ln_1\.(\w+)",
        "vision_model.vision_model.encoder.layers.{0}.layer_norm1.{1}",
    ),
    (
        r"visual\.transformer\.resblocks\.(\w+)\.ln_2\.(\w+)",
        "vision_model.vision_model.encoder.layers.{0}.layer_norm2.{1}",
    ),
    (
        r"visual\.transformer\.resblocks\.(\w+)\.attn\.out_proj\.(\w+)",
        "vision_model.vision_model.encoder.layers.{0}.self_attn.out_proj.{1}",
    ),
    (
        r"visual\.transformer\.resblocks\.(\w+)\.mlp\.c_fc\.(\w+)",
        "vision_model.vision_model.encoder.layers.{0}.mlp.fc1.{1}",
    ),
    (
        r"visual\.transformer\.resblocks\.(\w+)\.mlp\.c_proj\.(\w+)",
        "vision_model.vision_model.encoder.layers.{0}.mlp.fc2.{1}",
    ),
    # Text
    (r"text\.transformer\.(.+)", "text_model.{0}"),
    (r"text\.proj\.(.+)", "text_projection.{0}"),
]


def convert_vision_config(config: CLIPVisionCfg):
    config = dataclasses.asdict(config)
    new_config = {
        "hidden_act": "gelu",
    }
    for key, value in config.items():
        if key in VISION_CONFIG_MAP:
            new_config[VISION_CONFIG_MAP[key]] = value
        elif key == "head_width":
            new_config["num_attention_heads"] = config["width"] // value
        elif key == "mlp_ratio":
            new_config["intermediate_size"] = int(config["width"] * value)
        elif not key.startswith("timm") and value:
            print(f"WARNING: Unknown key '{key}' in vision config.")

    return CLIPVisionConfig(**new_config)


def convert_state_dict(state_dict):
    new_state_dict = {}
    for k, v in tqdm(state_dict.items()):
        found = False
        # special handling of vision attention blocks
        if match := re.match(r"visual\.transformer\.resblocks\.(\w+)\.attn\.in_proj_(\w+)", k):
            # chunk weights into three
            chunks = v.chunk(3, dim=0)
            for proj_name, proj_v in zip(["q_proj", "k_proj", "v_proj"], chunks):
                new_k = f"vision_model.vision_model.encoder.layers.{match.group(1)}.self_attn.{proj_name}.{match.group(2)}"
                print(k, "--->", new_k)
                new_state_dict[new_k] = proj_v
                found = True
        # transpose visual projection
        elif k == "visual.proj":
            new_k = "visual_projection.weight"
            print(k, "--->", new_k)
            new_state_dict[new_k] = v.t()
            found = True
        else:
            for pattern, replacement in STATE_DICT_PATTERNS:
                if match := re.match(pattern, k):
                    new_k = replacement.format(*match.groups())
                    print(k, "--->", new_k)
                    new_state_dict[new_k] = v
                    found = True
                    break
        if not found:
            new_state_dict[k] = v

    return new_state_dict


if __name__ == "__main__":
    # model_name = sys.argv[1]
    # pretrained = sys.argv[2]
    print("111")
    model_name = "convnext_base_w_320"
    pretrained = "laion_aesthetic_s13b_b82k_augreg"
    openclip_config = open_clip.get_model_config(model_name)
    openclip_model = open_clip.create_model(model_name, pretrained=pretrained)
    
    print("222")
    # if not isinstance(openclip_model.text, HFTextEncoder):
    #     raise ValueError("Only HFTextEncoder is supported.")
    # if openclip_config["text_cfg"]["pooler_type"] != "mean_pooler":
    #     raise ValueError("Only mean_pooler is supported.")
    text_config = open_clip.create_model("xlm-roberta-base-ViT-B-32", pretrained="laion5b_s13b_b90k").text.config
    
    vision_config = convert_vision_config(CLIPVisionCfg(**openclip_config["vision_cfg"]))

    config = VisionTextDualEncoderConfig.from_vision_text_configs(
        vision_config,
        text_config,
        projection_dim=openclip_config["embed_dim"],
    )

    print("333")
    state_dict = convert_state_dict(openclip_model.state_dict())

    print("444")
    model, loading_info = OpenCLIPVisionTextDualEncoderModel.from_pretrained(
        None, config=config, state_dict=state_dict, output_loading_info=True
    )
    print(loading_info)

    print("555")
    out_path = os.path.join("../../pretrained", model_name + "." + pretrained)
    # 
    print("666")
    model.vision_model.save_pretrained(out_path)
    model.save_pretrained(out_path + "_dual")