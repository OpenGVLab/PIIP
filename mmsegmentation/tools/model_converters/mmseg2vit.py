# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_vit(ckpt):

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('head'):
            # continue
            new_k = k #!
        elif k.startswith('ln1'):
            new_k = k.replace('ln1.', 'norm.')
        elif k.startswith('patch_embed'):
            if 'projection' in k:
                new_k = k.replace('projection', 'proj')
            else:
                new_k = k
        elif k.startswith('layers'):
            if 'ln' in k:
                new_k = k.replace('ln', 'norm')
            elif 'ffn.layers.0.0' in k:
                new_k = k.replace('ffn.layers.0.0', 'mlp.fc1')
            elif 'ffn.layers.1' in k:
                new_k = k.replace('ffn.layers.1', 'mlp.fc2')
            elif 'attn.attn.in_proj_' in k:
                new_k = k.replace('attn.attn.in_proj_', 'attn.qkv.')
            elif 'attn.attn.out_proj' in k:
                new_k = k.replace('attn.attn.out_proj', 'attn.proj')
            else:
                new_k = k
            new_k = new_k.replace('layers.', 'blocks.')
        else:
            new_k = k
        new_ckpt[new_k] = v

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_vit(state_dict)
    import ipdb; ipdb.set_trace()
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
