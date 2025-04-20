import torch
import torch.nn as nn

from transformers import CLIPImageProcessor

from .piip.configuration_piip_2branch import PIIPTwoBranchConfig
from .piip.modeling_piip_2branch import PIIPTwoBranchModel
from .piip.modeling_piip_2branch_mix import PIIPTwoBranchMixModel
from .piip.piip_modules import check_finite


class PIIPVisionTower(nn.Module):
    def __init__(self, vision_tower, delay_load=False, output_dir=None, is_pretraining=False):
        super().__init__()

        self.is_loaded = False
        self.output_dir = output_dir

        self.vision_tower_name = vision_tower
        
        assert self.vision_tower_name.endswith(".py")
        config_file = __import__(self.vision_tower_name.replace(".py", "").replace("/", "."), fromlist=[''])
        config_backbone = config_file.model["backbone"]
        self.config_file = config_file
        self.config_backbone = config_backbone
        
        self.model_cls = config_backbone.get("model_cls", "2branch_v1")
        
        self.is_pretraining = is_pretraining
        # if not delay_load:
        self.load_model(config_backbone)
        # else:
        self.cfg_only = self.config
            
    def load_model(self, config_backbone, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        if self.model_cls == "2branch_v1":
            self.vision_tower_type = "piip_2branch"
            vision_tower_cls = PIIPTwoBranchModel
            self.config = PIIPTwoBranchConfig(**config_backbone)
            self.input_image_size = config_backbone["branch2"]["img_size"]
        elif self.model_cls == "2branch_mix":
            self.vision_tower_type = "piip_2branch_mix"
            vision_tower_cls = PIIPTwoBranchMixModel
            self.config = PIIPTwoBranchConfig(**config_backbone)
            self.input_image_size = config_backbone["branch2"]["img_size"]
        else:
            raise NotImplementedError(self.model_cls)
        
        self.vision_tower = vision_tower_cls(self.config)
        if device_map is not None:
            # for eval
            self.vision_tower = self.vision_tower.cuda()
        
        
        self.image_processor = CLIPImageProcessor(
            crop_size=self.input_image_size, 
            do_center_crop=True, do_normalize=True, do_resize=True,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], 
            size=self.input_image_size
        )
        
        self.hidden_size = self.vision_tower.out_dim
        self.vision_tower.output_dir = self.output_dir

        self.is_loaded = True
        
        
    def unfreeze_all(self):
        print(f"!!! Unfreeze all layers of vision tower")
        for p in self.vision_tower.parameters():
            p.requires_grad = True

    
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                if isinstance(image_feature, torch.Tensor):
                    image_feature = image_feature.to(image.dtype)
                else:
                    image_feature = [x.to(image.dtype) for x in image_feature]
                assert check_finite(image_feature)
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            if isinstance(image_features, torch.Tensor):
                image_features = image_features.to(images.dtype)
                assert check_finite(image_features) #!!!
            else:
                image_features = [x.to(images.dtype) for x in image_features]
                assert all(check_finite(x) for x in image_features)
        
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
