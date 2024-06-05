# Copyright (c) OpenMMLab. All rights reserved.
from packaging import version as pkg_version

import torch
import json
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, MMDeepSpeedEngine, MMDeepSpeedInferenceEngine
from deepspeed.pipe import PipelineModule
from deepspeed.git_version_info import version, git_hash, git_branch
from deepspeed.utils import log_dist, OnDevice
from deepspeed.inference.config import DeepSpeedInferenceConfig

dp_factory = {'cuda': MMDataParallel, 'cpu': MMDataParallel}

ddp_factory = {'cuda': MMDistributedDataParallel}

ZeROddp_factory = {'cuda': MMDeepSpeedEngine}

ZeRO_inference_factory = {'cuda': MMDeepSpeedInferenceEngine}


def build_dp(model, device='cuda', dim=0, *args, **kwargs):
    """build DataParallel module by device type.

    if device is cuda, return a MMDataParallel model; if device is mlu,
    return a MLUDataParallel model.

    Args:
        model (:class:`nn.Module`): model to be parallelized.
        device (str): device type, cuda, cpu or mlu. Defaults to cuda.
        dim (int): Dimension used to scatter the data. Defaults to 0.

    Returns:
        nn.Module: the model to be parallelized.
    """
    if device == 'cuda':
        model = model.cuda(kwargs['device_ids'][0])
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDataParallel
        dp_factory['mlu'] = MLUDataParallel
        model = model.mlu()

    return dp_factory[device](model, dim=dim, *args, **kwargs)


def build_ddp(model, device='cuda', *args, **kwargs):
    """Build DistributedDataParallel module by device type.

    If device is cuda, return a MMDistributedDataParallel model;
    if device is mlu, return a MLUDistributedDataParallel model.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, mlu or cuda.

    Returns:
        :class:`nn.Module`: the module to be parallelized

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
                     DistributedDataParallel.html
    """
    assert device in ['cuda', 'mlu'], 'Only available for cuda or mlu devices.'
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDistributedDataParallel
        ddp_factory['mlu'] = MLUDistributedDataParallel
        model = model.mlu()

    return ddp_factory[device](model, *args, **kwargs)


def build_ZeROddp(model, optimizer=None, model_parameters=None, device='cuda', args=None):
    """Build DistributedDataParallel module by device type.

    If device is cuda, return a MMDistributedDataParallel model;
    if device is mlu, return a MLUDistributedDataParallel model.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, mlu or cuda.

    Returns:
        :class:`nn.Module`: the module to be parallelized

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
                     DistributedDataParallel.html
    """
    def _parse_version(version_str):
        '''Parse a version string and extract the major, minor, and patch versions.'''
        ver = pkg_version.parse(version_str)
        return ver.major, ver.minor, ver.micro
    
    # Export deepspeed version information
    __version__ = version
    __version_major__, __version_minor__, __version_patch__ = _parse_version(__version__)
    __git_hash__ = git_hash
    __git_branch__ = git_branch
    
    log_dist("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(
        __version__,
        __git_hash__,
        __git_branch__), ranks=[0])
    
    
    assert device == 'cuda', "zero only support cuda device for now."
    assert model is not None, "deepspeed.initialize requires a model"

    assert not isinstance(model, PipelineModule), "we only support deepspeed engine for now"
    model = model.cuda()
    
    ZeRO_engine = ZeROddp_factory[device](model=model,
                                          optimizer=optimizer,
                                          model_parameters=model_parameters,
                                          args=args)

    return_items = [
        ZeRO_engine,
        ZeRO_engine.optimizer,
        ZeRO_engine.training_dataloader,
        ZeRO_engine.lr_scheduler
    ]
    
    return tuple(return_items)


def build_ZeROddp_inference(model, device='cuda', config=None, *args, **kwargs):
    # Load config_dict from config first
    if config is None:
        config = {}
    if isinstance(config, str):
        with open(config, "r") as f:
            config_dict = json.load(f)
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError(f"'config' argument expected string or dictionary, got {type(config)}")

    # Update with values from kwargs, ensuring no conflicting overlap between config and kwargs
    overlap_keys = set(config_dict.keys()).intersection(kwargs.keys())
    # If there is overlap, error out if values are different
    for key in overlap_keys:
        if config_dict[key] != kwargs[key]:
            raise ValueError(f"Conflicting argument '{key}' in 'config':{config_dict[key]} and kwargs:{kwargs[key]}")
    config_dict.update(kwargs)

    ds_inference_config = DeepSpeedInferenceConfig(**config_dict)
    ds_inf_engine = ZeRO_inference_factory[device](model, config=ds_inference_config)
    
    return ds_inf_engine
    
    
def is_mlu_available():
    """Returns a bool indicating if MLU is currently available."""
    return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()


def get_device():
    """Returns an available device, cpu, cuda or mlu."""
    is_device_available = {
        'cuda': torch.cuda.is_available(),
        'mlu': is_mlu_available()
    }
    device_list = [k for k, v in is_device_available.items() if v]
    return device_list[0] if len(device_list) == 1 else 'cpu'
