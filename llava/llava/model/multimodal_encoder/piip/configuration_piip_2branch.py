from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
# from functools import partial


logger = logging.get_logger(__name__)


class PIIPTwoBranchConfig(PretrainedConfig):

    def __init__(
        self,
        n_points=4, 
        deform_num_heads=6,
        with_cffn=False, 
        cffn_ratio=0.25,
        deform_ratio=1.0,   
        interact_attn_type='normal',
        interaction_drop_path_rate=0.3,
        norm_layer="ln",
        interaction_proj=True,
                 
        branch1={},
        branch2={},
        
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_points = n_points
        self.deform_num_heads = deform_num_heads
        self.with_cffn = with_cffn
        self.cffn_ratio = cffn_ratio
        self.deform_ratio = deform_ratio
        self.interact_attn_type = interact_attn_type
        self.interaction_drop_path_rate = interaction_drop_path_rate
        self.norm_layer = norm_layer
        self.interaction_proj = interaction_proj
        
        self.branch1 = branch1
        self.branch2 = branch2
        
        

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)


        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)