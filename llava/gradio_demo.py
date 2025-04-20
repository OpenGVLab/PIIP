import os
import argparse

import gradio as gr
import torch

from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.eval.run_llava import eval_model, load_images

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="/mnt/petrelfs/liqingyun/PIIP_llava_hf/PIIP-LLaVA-Plus_ConvNeXt-L_CLIP-L_1024-336_7B", type=str)
args = parser.parse_args()



print("\n\nInitializing model...\n\n")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.model_path, None, get_model_name_from_path(args.model_path)
)
model = model.to(torch.float16)
assert not model.config.mm_use_im_start_end


def get_prompt(query):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def run_models(image_path, query):
    if image_path is None:
        return None
    
    prompt = get_prompt(query)
    
    image_files = image_path.split(",")
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
        )
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    return output

        


demo = gr.Interface(
    fn=run_models,
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Textbox(lines=2, label="Text Prompt")
    ],
    outputs=gr.Textbox(label="Output"),
    title="Multimodal Dialogue Demo",
    description="Upload an image and input text",
    examples=[
        ["images/tokyo.jpeg", "Describe the image."],
    ]
)

import socket
def get_ip():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        print(f"IP: {ip_address}")
        return ip_address
    except:
        print(f"IP: 127.0.0.1")
        return "127.0.0.1"

if __name__ == "__main__":
    demo.launch(server_name=get_ip(), server_port=10013, share=True)
    
