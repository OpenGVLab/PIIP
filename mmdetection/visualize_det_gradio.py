import os
# os.environ['MMCV_WITH_DS'] = '0'
import argparse

import mmcv
import gradio as gr
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import tempfile
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default="configs/piip/2branch/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms.py", type=str)
parser.add_argument('--checkpoint_file', default="work_dirs/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms.pth", type=str)
parser.add_argument('--device', default="cuda:0", type=str)
args = parser.parse_args()


print("Initializing model...")
model = init_detector(args.config_file, args.checkpoint_file, device=args.device)


def detect_objects(image, confidence):
    if image is None:
        return None
    
    img = mmcv.imread(image)
    result = inference_detector(model, img)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        output_path = tmp.name
    
    show_result_pyplot(
        model, 
        img, 
        result, 
        score_thr=confidence,
        title='Detection Result',
        wait_time=0,
        out_file=output_path
    )
    
    return output_path


demo = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Image(type="filepath", label="upload image"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.3, label="confidence threshold")
    ],
    outputs=gr.Image(label="detection result"),
    title="object detection and instance segmentation demo",
    description="Upload an image and adjust the confidence threshold to view detection results.",
    examples=[
        ["demo/demo.jpg", 0.8],
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
        server_port=10012, share=True)
