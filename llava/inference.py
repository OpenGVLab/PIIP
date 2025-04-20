import os
import argparse

from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="/mnt/petrelfs/liqingyun/PIIP_llava_hf/PIIP-LLaVA-Plus_ConvNeXt-L_CLIP-L_1024-336_7B", type=str)
parser.add_argument('--img_path', default="images/llava_logo.png", type=str)
parser.add_argument('--prompt', default="Describe the image.", type=str)
args = parser.parse_args()


args = type('Args', (), {
    "model_path": args.model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(args.model_path),
    "query": args.prompt,
    "conv_mode": None,
    "image_file": args.img_path,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "dtype": "fp16",
})()

output = eval_model(args)


print("\n\n")
print("Image File:", args.image_file)
print("Prompt:", args.prompt)

print(f"Output:", output)
    