# --------------------------------------------------------
# PIIP
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_instance.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]
deepspeed = True
deepspeed_config = 'zero_configs/adam_zero1_bf16.json'

model = dict(
    backbone=dict(
        _delete_=True,
        type='PIIPThreeBranch',
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cffn=True,
        interact_attn_type='deform',
        interaction_drop_path_rate=0.4,
        
        branch1=dict(
            real_size=448,
            pretrain_img_size=224,
            patch_size=16,
            pretrain_patch_size=16,
            depth=24,
            embed_dim=1024,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.4,
            init_scale=1.,
            with_fpn=False,
            interaction_indexes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23]],
            pretrained = "./pretrained/deit_3_large_224_21k.pth",
            window_attn=[True, True, True, True, True, True,
                         True, True, True, True, True, True,
                         True, True, True, True, True, True,
                         True, True, True, True, True, True,],
            window_size=[28, 28, 28, 28, 28, 28,
                         28, 28, 28, 28, 28, 28,
                         28, 28, 28, 28, 28, 28,
                         28, 28, 28, 28, 28, 28],
            use_flash_attn=True,
            img_norm_cfg = dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
        ),
        
        branch2=dict(
            real_size=672, 
            pretrain_img_size=224,
            patch_size=16,
            pretrain_patch_size=16,
            depth=12,
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.15,
            init_scale=1.,
            with_fpn=False,
            interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]],
            pretrained = "./pretrained/deit_3_base_224_21k.pth",
            window_attn=[True, True, True,
                         True, True, True,
                         True, True, True,
                         True, True, True,],
            window_size=[28, 28, 28,
                         28, 28, 28,
                         28, 28, 28,
                         28, 28, 28],
            use_flash_attn=True,
            img_norm_cfg = dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
        ),
        
        branch3=dict(
            real_size=1120,
            pretrain_img_size=224,
            patch_size=16,
            pretrain_patch_size=16,
            depth=12,
            embed_dim=384,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.05,
            init_scale=1.,
            with_fpn=False,
            interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]],
            pretrained="./pretrained/deit_3_small_224_21k.pth",
            window_attn=[True, True, True,
                         True, True, True,
                         True, True, True,
                         True, True, True,],
            window_size=[28, 28, 28,
                         28, 28, 28,
                         28, 28, 28,
                         28, 28, 28],
            use_flash_attn=True,
            img_norm_cfg = dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=256,
        num_outs=5))
# By default, models are trained on 8 GPUs with 2 images per GPU
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1120, (1120*800)//1333), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=224),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1120, (1120*800)//1333),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=224),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='CustomLayerDecayOptimizerConstructorMMDet',
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.85, skip_stride=[2, 2])
                 )
optimizer_config = dict(grad_clip=None)
if deepspeed:
    checkpoint_config = dict(deepspeed=deepspeed, interval=1, max_keep_ckpts=1)
else:
    checkpoint_config = dict(interval=1, max_keep_ckpts=1)
evaluation = dict(interval=1, save_best=None)

custom_imports = dict(
    imports=[
        'mmdet.mmcv_custom'],
    allow_failed_imports=False
)

if deepspeed:
    custom_hooks = [
        dict(
            type='ToBFloat16HookMMDet',
            priority=49),
    ]
