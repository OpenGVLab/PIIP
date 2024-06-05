model = dict(
    backbone=dict(
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cffn=True,
        interact_attn_type='deform',
        interaction_drop_path_rate=0.4,
        separate_head=True,
        
        branch1=dict(
            model_type="augreg",
            img_size=128,
            patch_size=16,
            pretrain_img_size=224,
            pretrain_patch_size=16,
            depth=24,
            embed_dim=1024,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.4,
            interaction_indexes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23]],
            use_cls_token=True,
            use_flash_attn=True,
            with_cp=True,
            pretrained="pretrained/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.pth",
        ),
        
        branch2=dict(
            model_type="augreg",
            img_size=192,
            patch_size=16,
            pretrain_img_size=224,
            pretrain_patch_size=16,
            depth=12,
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.2,
            interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]],
            use_cls_token=True,
            use_flash_attn=True,
            with_cp=True,
            pretrained="pretrained/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.pth",
        ),
        
        branch3=dict(
            model_type="augreg",
            img_size=384,
            patch_size=16,
            pretrain_img_size=224,
            pretrain_patch_size=16,
            depth=12,
            embed_dim=384,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.05,
            interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]],
            use_cls_token=True,
            use_flash_attn=True,
            with_cp=True,
            pretrained="pretrained/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.pth",
        ),
    ),
)