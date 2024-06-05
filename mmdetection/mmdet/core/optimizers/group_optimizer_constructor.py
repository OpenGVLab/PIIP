# Copyright (c) SenseTime X-Lab. All rights reserved.
from mmcv.utils import build_from_cfg
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmcv.runner.optimizer import DefaultOptimizerConstructor


@OPTIMIZER_BUILDERS.register_module()
class GroupOptimizerConstructor(DefaultOptimizerConstructor):
    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module
        
        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        assert not self.paramwise_cfg, '\"paramwise_cfg\" is not supported for GroupOptimizerConstructor!'
        param_groups = []
        default_component_names = ['backbone', 'neck', 'rpn_head', 'roi_head', 'bbox_head']
        component_names = optimizer_cfg.pop('component_names', default_component_names)
        for component in component_names:
            if hasattr(model, component):
                fa_component = getattr(model, component)
                if hasattr(fa_component, 'mask_head'):
                    param_group = {'lr': self.base_lr, 
                                   'params': getattr(fa_component, 'bbox_head').parameters(),
                                   'name': component}
                    param_groups.append(param_group)
                    param_group = {'lr': self.base_lr, 
                                   'params': getattr(fa_component, 'mask_head').parameters(),
                                   'name': component}
                    param_groups.append(param_group)
                else:
                    param_group = {'lr': self.base_lr, 
                                   'params': getattr(model, component).parameters(),
                                   'name': component}
                    param_groups.append(param_group)

        optimizer_cfg['params'] = param_groups
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)