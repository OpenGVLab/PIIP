import os
import argparse
import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default="configs/piip/2branch/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms.py", type=str)
parser.add_argument('--checkpoint_file', default="work_dirs/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms.pth", type=str)
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--img_path', default="demo/demo.jpg", type=str)
parser.add_argument('--confidence_threshold', default=0.7, type=float)
parser.add_argument('--out_path', default="visualization.jpg", type=str)
args = parser.parse_args()


print("initializing")
model = init_detector(args.config_file, args.checkpoint_file, device=args.device)
img = mmcv.imread(args.img_path)

result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=args.confidence_threshold, title='result', wait_time=0, palette=None, out_file=args.out_path)
print("done")