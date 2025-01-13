import os
import argparse
import mmcv
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default="configs/piip/2branch/upernet_internvit_h6b_512_512_80k_ade20k_bs16_lr4e-5.py", type=str)
parser.add_argument('--checkpoint_file', default="work_dirs/upernet_internvit_h6b_512_512_80k_ade20k_bs16_lr4e-5/upernet_internvit_h6b_512_512_80k_ade20k_bs16_lr4e-5.pth", type=str)
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--img_path', default="demo/demo.png", type=str)
parser.add_argument('--out_path', default="visualization.jpg", type=str)
args = parser.parse_args()


print("initializing")
model = init_segmentor(args.config_file, args.checkpoint_file, device=args.device)
img = mmcv.imread(args.img_path)

result = inference_segmentor(model, img)
show_result_pyplot(model, img, result, opacity=0.5, title='result', out_file=args.out_path)
print("done")