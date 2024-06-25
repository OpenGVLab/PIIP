# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

_base_ = [
    '../../_base_/models/upernet_r50.py',
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_80k.py'
]
deepspeed = True
deepspeed_config = 'zero_configs/adam_zero1_bf16.json'
pretrained = None
model = dict(
    pretrained=None,
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
            img_size=512,
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
            interaction_indexes=[[0, 3], [4, 7], [8, 11], [12, 15], [16, 19], [20, 23], [24, 27], [28, 31], [32, 35], [36, 39], [40, 43], [44, 47]], #TODO
            pretrained = 'pretrained/intern_vit_6b_224px.pth',
            norm_layer_type='RMSNorm',
            mlp_type='fused_mlp',
        ),
        
        branch2=dict(
            real_size=512,
            img_size=512,
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
            interaction_indexes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 10], [11, 13], [14, 16], [17, 19], [20, 22], [23, 25], [26, 28], [29, 31]], #TODO
            pretrained = 'pretrained/mae_pretrain_vit_huge.pth',
            norm_layer_type='LayerNorm',
            mlp_type='fused_mlp'
        ),
    ),
    decode_head=dict(num_classes=150,
                     channels=1536,
                     in_channels=[3200, 3200, 3200, 3200]),
    auxiliary_head=dict(num_classes=150,
                        channels=1536,
                        in_channels=3200),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)
optimizer = dict(_delete_=True, type='AdamW', lr=4e-5, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='CustomLayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=48, layer_decay_rate=0.95, skip_stride=1.5))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
runner = dict(type='IterBasedRunner')
if deepspeed:
    checkpoint_config = dict(deepspeed=deepspeed, by_epoch=False, interval=2000, max_keep_ckpts=1)
else:
    checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=1)
evaluation = dict(interval=1000, metric='mIoU', save_best=None)

if deepspeed:
    custom_hooks = [
        dict(
            type='ToBFloat16Hook',
            priority=49),
    ]
