import torch
import torch.nn as nn
import torch.distributed as dist
from mmcv.parallel import scatter_kwargs
from torch._utils import _get_device_index
from torch.distributed.distributed_c10d import _get_default_group
from mmcv.parallel import MODULE_WRAPPERS


@MODULE_WRAPPERS.register_module()
class AgvmDistModule(nn.Module):
    def __init__(self, model, device_ids):
        super().__init__()
        self.module = model
        self.device_ids = device_ids
        self.dim = 0

        self.output_device = _get_device_index(device_ids[0], True)
        self.process_group = _get_default_group()
        self.device = list(self.module.parameters())[0].device

        self.broadcast_params(model)
    
    def broadcast_params(self, model):
        for name, p in model.state_dict().items():
            dist.broadcast(p, 0)
    
    def to_kwargs(self, inputs, kwargs, device_id):
        # Use `self.to_kwargs` instead of `self.scatter` in pytorch1.8
        # to move all tensors to device_id
        return scatter_kwargs(inputs, kwargs, [device_id], dim=self.dim)
    
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
    
    def train_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        output = self.module.train_step(*inputs[0], **kwargs[0])
        return output
    
    def val_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        output = self.module.val_step(*inputs[0], **kwargs[0])
        return output

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
        output = self.module(*inputs[0], **kwargs[0])
        return output
