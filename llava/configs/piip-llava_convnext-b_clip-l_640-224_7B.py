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
            img_size=224,
            patch_size=16,
            pretrain_patch_size=16,
            interaction_indexes=[[0, 11], [12, 23]],
            pretrained='OpenGVLab/clip-vit-large-patch14to16-224',
            use_cls_token=True,
        ),
        
        branch2=dict(
            model_type="convnext_timm",
            img_size=640,
            interaction_indexes=[
                [0, 19], [20, 38],
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
            pretrained="convnext_base.clip_laiona_augreg_320",
        ),
    ),
    
    projector=dict(
        type="separate_2branch_v1",
        dim1=1024,
        dim2=1024,
        norm="gn",
    )
)



