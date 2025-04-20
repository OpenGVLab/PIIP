model = dict(
    backbone=dict(
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cffn=True,
        interact_attn_type='deform',
        interaction_drop_path_rate=0.4,
        model_cls="2branch_mix",
        unfreeze="none",
        
        use_branch_merging=False,
        
        
        branch1=dict(
            model_type="clip_hf",
            img_size=336,
            patch_size=16,
            pretrain_patch_size=16,
            interaction_indexes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23]],
            pretrained='OpenGVLab/clip-vit-large-patch14to16-336',
            use_cls_token=True,
        ),
        
        branch2=dict(
            model_type="convnext_timm",
            img_size=1024,
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
            pretrained="convnext_large_mlp.clip_laion2b_ft_320",
        ),
    ),
    
    projector=dict(
        type="separate_2branch_v1",
        dim1=1024,
        dim2=1536,
        norm="gn",
    )
)



