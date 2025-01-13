import os
# os.environ['MMCV_WITH_DS'] = '0'
import argparse

import mmcv
import gradio as gr
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
import tempfile
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default="configs/piip/2branch/upernet_internvit_h6b_512_512_80k_ade20k_bs16_lr4e-5.py", type=str)
parser.add_argument('--checkpoint_file', default="work_dirs/upernet_internvit_h6b_512_512_80k_ade20k_bs16_lr4e-5/upernet_internvit_h6b_512_512_80k_ade20k_bs16_lr4e-5.pth", type=str)
parser.add_argument('--device', default="cuda:0", type=str)
args = parser.parse_args()


print("Initializing model...")
model = init_segmentor(args.config_file, args.checkpoint_file, device=args.device)


palette = model.PALETTE
classes = model.CLASSES


for i, (cls_name, color) in enumerate(zip(classes, palette)):
    print(f"category {i}: {cls_name} - RGB color: {color}")

def segment_objects(image):
    if image is None:
        return None
    
    img = mmcv.imread(image)
    result = inference_segmentor(model, img)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        output_path = tmp.name
    
    show_result_pyplot(
        model, 
        img, 
        result, 
        opacity=0.5,
        title='Segmentation Result',
        out_file=output_path
    )
    
    return output_path


demo = gr.Interface(
    fn=segment_objects,
    inputs=[
        gr.Image(type="filepath", label="upload image"),
    ],
    outputs=gr.Image(label="segmentation result"),
    title="semantic segmentation demo",
    description="Upload an image to view segmentation results.",
    examples=[
        ["demo/demo.png"],
    ]
)

import socket
def get_ip():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except:
        return "127.0.0.1"

if __name__ == "__main__":
    demo.launch(server_name=get_ip(),
        server_port=10013, share=True)
