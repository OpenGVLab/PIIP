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
        type='PIIPTwoBranch',
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cffn=True,
        interact_attn_type='deform',
        interaction_drop_path_rate=0.4,
        interaction_proj=False,
        norm_layer='none',
        branch1=dict(
            real_size=512,
            pretrain_img_size=224,
            patch_size=16,
            pretrain_patch_size=14,
            depth=48,
            embed_dim=3200,
            num_heads=25,
            mlp_ratio=4,
            qkv_bias=False,
            init_values=0.1,
            with_cp=True,
            use_flash_attn=True,
            qk_normalization=True,
            layerscale_force_fp32=False,
            with_fpn=False,
            drop_path_rate=0.4,
            interaction_indexes=[[0, 3], [4, 7], [8, 11], [12, 15], [16, 19], [20, 23], [24, 27], [28, 31], [32, 35], [36, 39], [40, 43], [44, 47]],
            pretrained = 'pretrained/intern_vit_6b_224px.pth',
            norm_layer_type='RMSNorm',
            mlp_type='fused_mlp',
        ),
        
        branch2=dict(
            real_size=1024, 
            pretrain_img_size=224,
            patch_size=16,
            pretrain_patch_size=14,
            depth=32,
            embed_dim=1280,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            init_values=1.0,
            with_cp=True,
            use_flash_attn=True,
            qk_normalization=False,
            layerscale_force_fp32=False,
            with_fpn=False,
            drop_path_rate=0.4,
            interaction_indexes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 10], [11, 13], [14, 16], [17, 19], [20, 22], [23, 25], [26, 28], [29, 31]],
            pretrained = 'pretrained/mae_pretrain_vit_huge.pth',
            norm_layer_type='LayerNorm',
            mlp_type='fused_mlp',
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[3200, 3200, 3200, 3200],
        out_channels=256,
        num_outs=5))
# By default, models are trained on 8 GPUs with 2 images per GPU
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='RandomCrop',
         crop_type='absolute_range',
         crop_size=(1024, 1024),
         allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=64),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1333, 800),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=64),
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
                 paramwise_cfg=dict(num_layers=48, layer_decay_rate=0.85, skip_stride=[48/32])
                 )
optimizer_config = dict(grad_clip=None)
if deepspeed:
    checkpoint_config = dict(deepspeed=deepspeed, interval=1, max_keep_ckpts=1)
else:
    checkpoint_config = dict(interval=1, max_keep_ckpts=1)
evaluation = dict(interval=1, save_best=None)

custom_imports = dict(
    imports=[
        'mmcv_custom'],
    allow_failed_imports=False
)

if deepspeed:
    custom_hooks = [
        dict(
            type='ToBFloat16HookMMDet',
            priority=49),
    ]
