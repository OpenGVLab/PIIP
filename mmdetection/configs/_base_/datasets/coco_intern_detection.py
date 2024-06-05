# dataset settings
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
file_client_args = dict(backend='petrel')

train_pipeline = [
    dict(type='LoadImageFromInternFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromInternFile', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data_root = '/mnt/petrelfs/share_data/huangzhenhang/datasets/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=[data_root + 'coco/annotations/instances_train2017.json', data_root + 'intern/tianchangyao.p/DETv1.0.list.train.json'],
        img_prefix=[data_root + 'coco/train2017/', 'hzh:s3://intern_data_detection/',],
        # ann_file=data_root + 'intern/tianchangyao.p/DETv1.0.list.train.json',
        # img_prefix='hzh:s3://intern_data_detection/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=[data_root + 'coco/annotations/instances_val2017.json', ],
        img_prefix=[data_root + 'coco/val2017/',],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=[data_root + 'coco/annotations/instances_val2017.json', ],
        img_prefix=[data_root + 'coco/val2017/',],        
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
