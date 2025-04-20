model = dict(
    backbone=dict(
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cffn=True,
        interact_attn_type='deform',
        interaction_drop_path_rate=0.4,
        model_cls="2branch_v1",
        unfreeze="none",
        
        use_branch_merging=False,
        
        
        branch1=dict(
            model_type="clip_hf",
            img_size=256,
            patch_size=16,
            pretrain_patch_size=16,
            interaction_indexes=[[0, 11], [12, 23]],
            pretrained='OpenGVLab/clip-vit-large-patch14to16-224',
            use_cls_token=True,
        ),
        
        branch2=dict(
            model_type="clip_hf",
            img_size=512,
            patch_size=16,
            pretrain_patch_size=16,
            interaction_indexes=[[0, 5], [6, 11]],
            pretrained='openai/clip-vit-base-patch16',
            use_cls_token=True,
        ),
    ),
    
    projector=dict(
        type="separate_2branch_v1",
        dim1=1024,
        dim2=768,
        norm="gn",
    )
)



