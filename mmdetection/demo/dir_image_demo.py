# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
import os.path as osp
import os
import json
from PIL import Image
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
# import mmdet_custom # noqa: F401,F403


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='random',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    if not osp.isdir(args.img):
        imgs = [args.img]
    else:
        imgs = [osp.join(args.img, img) for img in os.listdir(args.img)]

    format_anno = []
    format_image = []
    print(len(imgs))
    for img_id, img_path in enumerate(imgs):
        # test a single image
        h, w = Image.open(img_path).size
        basename = osp.basename(img_path)
        single_image_info = {"file_name": basename,
                             "height": h, "width": w, "id": img_id}
        format_image.append(single_image_info)
        result = inference_detector(model, img_path)
        for cat_id, res in enumerate(result):
            if res.shape[0] == 0:
                continue
            else:
                for single_res in res:
                    single_res = single_res.astype(float).tolist()
                    single_res_info = {
                        'image_id': img_id, 'bbox': single_res[:4], 'score': single_res[4], 'category_id': cat_id, "area": single_res[2] * single_res[3]}
                    format_anno.append(single_res_info)
        # show the results
        show_result_pyplot(
            model,
            img_path,
            result,
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=f'{args.out_file}/{basename}.png')
    with open(f'{args.out_file}/demo4owa.json', 'w') as f:
        save_info = json.dumps({'images': format_image, 'annotations': format_anno, 'categories': [
                               {'id': i, 'name': model.CLASSES[i]} for i in range(len(model.CLASSES))]})
        f.write(save_info)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
