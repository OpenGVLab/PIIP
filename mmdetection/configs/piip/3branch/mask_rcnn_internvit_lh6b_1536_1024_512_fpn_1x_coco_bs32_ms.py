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
        interaction_proj=False,
        norm_layer="none",
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
            interaction_indexes=[[0, 11], [12, 23], [24, 35], [36, 47]],
            pretrained = 'pretrained/intern_vit_6b_224px.pth',
            norm_layer_type='RMSNorm',
            mlp_type="fused_mlp",
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
            interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]],
            pretrained = 'pretrained/mae_pretrain_vit_huge.pth',
            norm_layer_type='LayerNorm',
            mlp_type="fused_mlp",
        ),
        branch3=dict(
            real_size=1536,
            pretrain_img_size=384,
            patch_size=16,
            pretrain_patch_size=16,
            depth=24,
            embed_dim=1024,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.4,
            init_scale=1,
            with_fpn=False,
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
            pretrained = "./pretrained/deit_3_large_384_21k.pth",
            use_flash_attn=True,
            Mlp_block="fused_mlp",
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[3200, 3200, 3200, 3200],
        out_channels=256,
        num_outs=5))
# By default, models are trained on 16 GPUs with 2 images per GPU
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN
r = 1536 / 1024
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(int(480*r), int(1333*r)), (int(512*r), int(1333*r)), (int(544*r), int(1333*r)), (int(576*r), int(1333*r)),
                                 (int(608*r), int(1333*r)), (int(640*r), int(1333*r)), (int(672*r), int(1333*r)), (int(704*r), int(1333*r)),
                                 (int(736*r), int(1333*r)), (int(768*r), int(1333*r)), (int(800*r), int(1333*r))],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(int(400*r), int(1333*r)), (int(500*r), int(1333*r)), (int(600*r), int(1333*r))],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(int(384*r), int(600*r)),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(int(480*r), int(1333*r)), (int(512*r), int(1333*r)), (int(544*r), int(1333*r)),
                                 (int(576*r), int(1333*r)), (int(608*r), int(1333*r)), (int(640*r), int(1333*r)),
                                 (int(672*r), int(1333*r)), (int(704*r), int(1333*r)), (int(736*r), int(1333*r)),
                                 (int(768*r), int(1333*r)), (int(800*r), int(1333*r))],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='RandomCrop',
         crop_type='absolute_range',
         crop_size=(1536, 1536),
         allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=64),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
        img_scale=(int(1333*r), int(800*r)),
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
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='CustomLayerDecayOptimizerConstructorMMDet',
                 paramwise_cfg=dict(num_layers=48, layer_decay_rate=0.85, skip_stride=[48/32, 48/24])
                 )
optimizer_config = dict(grad_clip=None)
if deepspeed:
    checkpoint_config = dict(deepspeed=deepspeed, interval=1, max_keep_ckpts=1)
else:
    checkpoint_config = dict(interval=1, max_keep_ckpts=1)
evaluation = dict(interval=1, save_best='auto')

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
