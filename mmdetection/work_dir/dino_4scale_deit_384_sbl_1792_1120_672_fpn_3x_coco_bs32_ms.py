dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[{
            'type':
            'Resize',
            'img_scale':
            [(840, 2332), (896, 2332), (952, 2332), (1008, 2332), (1064, 2332),
             (1120, 2332), (1176, 2332), (1232, 2332), (1288, 2332),
             (1344, 2332), (1400, 2332)],
            'multiscale_mode':
            'value',
            'keep_ratio':
            True
        }],
                  [{
                      'type': 'Resize',
                      'img_scale': [(700, 2332), (875, 2332), (1050, 2332)],
                      'multiscale_mode': 'value',
                      'keep_ratio': True
                  }, {
                      'type': 'RandomCrop',
                      'crop_type': 'absolute_range',
                      'crop_size': (672, 1050),
                      'allow_negative_crop': True
                  }, {
                      'type':
                      'Resize',
                      'img_scale': [(840, 2332), (896, 2332), (952, 2332),
                                    (1008, 2332), (1064, 2332), (1120, 2332),
                                    (1176, 2332), (1232, 2332), (1288, 2332),
                                    (1344, 2332), (1400, 2332)],
                      'multiscale_mode':
                      'value',
                      'override':
                      True,
                      'keep_ratio':
                      True
                  }]]),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(1792, 1792),
        allow_negative_crop=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=224),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2332, 1400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=224),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type':
                    'Resize',
                    'img_scale': [(840, 2332), (896, 2332), (952, 2332),
                                  (1008, 2332), (1064, 2332), (1120, 2332),
                                  (1176, 2332), (1232, 2332), (1288, 2332),
                                  (1344, 2332), (1400, 2332)],
                    'multiscale_mode':
                    'value',
                    'keep_ratio':
                    True
                }],
                          [{
                              'type':
                              'Resize',
                              'img_scale': [(700, 2332), (875, 2332),
                                            (1050, 2332)],
                              'multiscale_mode':
                              'value',
                              'keep_ratio':
                              True
                          }, {
                              'type': 'RandomCrop',
                              'crop_type': 'absolute_range',
                              'crop_size': (672, 1050),
                              'allow_negative_crop': True
                          }, {
                              'type':
                              'Resize',
                              'img_scale': [(840, 2332), (896, 2332),
                                            (952, 2332), (1008, 2332),
                                            (1064, 2332), (1120, 2332),
                                            (1176, 2332), (1232, 2332),
                                            (1288, 2332), (1344, 2332),
                                            (1400, 2332)],
                              'multiscale_mode':
                              'value',
                              'override':
                              True,
                              'keep_ratio':
                              True
                          }]]),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(1792, 1792),
                allow_negative_crop=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=224),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2332, 1400),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=224),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2332, 1400),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=224),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric=['bbox'], save_best=None)
checkpoint_config = dict(interval=1, deepspeed=True, max_keep_ckpts=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='ToBFloat16HookMMDet', priority=49)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructorMMDet',
    paramwise_cfg=dict(
        num_layers=24, layer_decay_rate=0.85, skip_stride=[2, 2]))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
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
            interaction_indexes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
                                 [10, 11], [12, 13], [14, 15], [16, 17],
                                 [18, 19], [20, 21], [22, 23]],
            pretrained='./pretrained/deit_3_large_384_21k.pth',
            window_attn=[
                True, True, False, True, True, False, True, True, False, True,
                True, False, True, True, False, True, True, False, True, True,
                False, True, True, False
            ],
            window_size=[
                24, 24, -1, 24, 24, -1, 24, 24, -1, 24, 24, -1, 24, 24, -1, 24,
                24, -1, 24, 24, -1, 24, 24, -1
            ],
            use_flash_attn=True),
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
            interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
                                 [5, 5], [6, 6], [7, 7], [8, 8], [9, 9],
                                 [10, 10], [11, 11]],
            pretrained='./pretrained/deit_3_base_384_21k.pth',
            window_attn=[
                True, True, False, True, True, False, True, True, False, True,
                True, False
            ],
            window_size=[24, 24, -1, 24, 24, -1, 24, 24, -1, 24, 24, -1],
            use_flash_attn=True),
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
            interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
                                 [5, 5], [6, 6], [7, 7], [8, 8], [9, 9],
                                 [10, 10], [11, 11]],
            pretrained='./pretrained/deit_3_small_384_21k.pth',
            window_attn=[
                True, True, False, True, True, False, True, True, False, True,
                True, False
            ],
            window_size=[24, 24, -1, 24, 24, -1, 24, 24, -1, 24, 24, -1],
            use_flash_attn=True)),
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
                    ffn_dropout=0.0,
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
                            dropout=0.0)
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
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=300))
r = 1.75
custom_imports = dict(imports=['mmcv_custom'], allow_failed_imports=False)
work_dir = './work_dir/'
auto_resume = False
gpu_ids = range(0, 1)
