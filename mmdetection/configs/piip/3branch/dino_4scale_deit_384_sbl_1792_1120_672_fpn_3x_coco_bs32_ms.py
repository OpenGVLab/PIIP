# --------------------------------------------------------
# PIIP
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_3x.py',
]
deepspeed = True
deepspeed_config = 'zero_configs/adam_zero1_bf16.json'

model = dict(
    type='DINO',
    backbone=dict(
        type='PIIPThreeBranch',
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cffn=True,
        interact_attn_type='deform',
        interaction_drop_path_rate=0.4,
        is_dino=True,
        
        branch1=dict(
            real_size=672,
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
            interaction_indexes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23]],
            pretrained = "./pretrained/deit_3_large_384_21k.pth",
            window_attn=[True, True, False, True, True, False,
                         True, True, False, True, True, False,
                         True, True, False, True, True, False,
                         True, True, False, True, True, False,],
            window_size=[24, 24, -1, 24, 24, -1,
                         24, 24, -1, 24, 24, -1,
                         24, 24, -1, 24, 24, -1,
                         24, 24, -1, 24, 24, -1],
            use_flash_attn=True,
        ),
        
        branch2=dict(
            real_size=1120, 
            pretrain_img_size=384,
            patch_size=16,
            pretrain_patch_size=16,
            depth=12,
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.15,
            init_scale=1,
            with_fpn=False,
            interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]],
            pretrained = "./pretrained/deit_3_base_384_21k.pth",
            window_attn=[True, True, False,
                         True, True, False,
                         True, True, False,
                         True, True, False,],
            window_size=[24, 24, -1,
                         24, 24, -1,
                         24, 24, -1,
                         24, 24, -1],
            use_flash_attn=True,
        ),
        
        branch3=dict(
            real_size=1792,
            pretrain_img_size=384,
            patch_size=16,
            pretrain_patch_size=16,
            depth=12,
            embed_dim=384,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.05,
            init_scale=1,
            with_fpn=False,
            interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]],
            pretrained="./pretrained/deit_3_small_384_21k.pth",
            window_attn=[True, True, False,
                         True, True, False,
                         True, True, False,
                         True, True, False,],
            window_size=[24, 24, -1,
                         24, 24, -1,
                         24, 24, -1,
                         24, 24, -1],
            use_flash_attn=True,
        ),
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[1024, 1024, 1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DINOHead',
        num_query=900,
        num_classes=80,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=1.0),
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        transformer=dict(
            type='DinoTransformer',
            two_stage_num_proposals=900,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,  # 0.1 for DeformDETR
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            dropout=0.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=300))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
r = 1792 / 1024
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True), 
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
         crop_size=(1792, 1792),
         allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=224),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(int(1333*r), int(800*r)),
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
# By default, models are trained on 16 GPUs with 2 images per GPU
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='CustomLayerDecayOptimizerConstructorMMDet',
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.85, skip_stride=[2, 2])
                 )
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
if deepspeed:
    checkpoint_config = dict(deepspeed=deepspeed, interval=1, max_keep_ckpts=1)
else:
    checkpoint_config = dict(interval=1, max_keep_ckpts=1)
evaluation = dict(interval=1, save_best=None, metric=['bbox'])

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