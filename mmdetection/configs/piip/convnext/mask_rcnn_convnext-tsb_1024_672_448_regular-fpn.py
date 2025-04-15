# --------------------------------------------------------
# InternVL
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
        with_simple_fpn=False,
        out_interaction_indexes=[0, 1, 10],
        
        branch1=dict(
            real_size=448, 
            interaction_indexes=[
                [0, 2], 
                [3, 6], 
                [7, 10], [11, 13], [14, 16], [17, 19], [20, 22], [23, 25], [26, 28], [29, 31], [32, 34],
                [35, 38],
            ],
            downsample_ratios=[
                4, 4, 4,
                8, # downsampling layer
                8, 8, 8, 
                16, # downsampling layer
                16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 16, 16, 16, 16, 16, 16, 16, 16,
                32, # downsampling layer
                32, 32, 32,
            ],
            pretrained="facebook/convnext-base-224", # out_dim=2048
            drop_path_rate=0.4,
        ),
        
        branch2=dict(
            real_size=672,
            interaction_indexes=[
                [0, 2], 
                [3, 6], 
                [7, 10], [11, 13], [14, 16], [17, 19], [20, 22], [23, 25], [26, 28], [29, 31], [32, 34],
                [35, 38],
            ],
            downsample_ratios=[
                4, 4, 4,
                8, # downsampling layer
                8, 8, 8, 
                16, # downsampling layer
                16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 16, 16, 16, 16, 16, 16, 16, 16,
                32, # downsampling layer
                32, 32, 32,
            ],
            pretrained="facebook/convnext-small-224", # out_dim=512
            drop_path_rate=0.3,
        ),
        
        branch3=dict(
            real_size=1024,
            interaction_indexes=[
                [0, 2], 
                [3, 6], 
                [7, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13], [14, 14], [15, 15], [16, 16],
                [17, 20],
            ],
            downsample_ratios=[
                4, 4, 4,
                8, # downsampling layer
                8, 8, 8, 
                16, # downsampling layer
                16, 16, 16, 16, 16, 16, 16, 16, 16,
                32, # downsampling layer
                32, 32, 32,
            ],
            pretrained="facebook/convnext-tiny-224", # out_dim=512
            drop_path_rate=0.3,
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='CustomLayerDecayOptimizerConstructorMMDet',
                 paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.8, skip_stride=[1, 3])
                )
# optimizer_config = dict(grad_clip=None)
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

log_config = dict(
    interval=50,
    hooks=[
        # dict(type='TextLoggerHook'),
        dict(
            type='MMDetWandbHook',
            init_kwargs=dict(project='',
                            name="",
                            tags=[]),
            interval=50,
            log_checkpoint=False,
            log_checkpoint_metadata=False,
            num_eval_images=0
        )
    ]
)