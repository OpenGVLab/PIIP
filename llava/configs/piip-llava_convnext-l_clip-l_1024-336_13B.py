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
            interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21], [22, 22], [23, 23]],
            pretrained='OpenGVLab/clip-vit-large-patch14to16-336',
            use_cls_token=True,
        ),
        
        branch2=dict(
            model_type="convnext_timm",
            img_size=1024,
            interaction_indexes=[
                [0, 1], [2, 2], 
                [3, 4], [5, 6], 
                [7, 9], [10, 10], [11, 12], [13, 13], [14, 15], [16, 16], [17, 18], [19, 19], [20, 21], [22, 22], [23, 24], [25, 25], [26, 27], [28, 28], [29, 30], [31, 31], [32, 33], [34, 34],
                [35, 36], [37, 38],
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



