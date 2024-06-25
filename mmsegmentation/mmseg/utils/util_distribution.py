# Copyright (c) OpenMMLab. All rights reserved.
from packaging import version as pkg_version

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, MMDeepSpeedEngine

from mmseg import digit_version
from deepspeed.pipe import PipelineModule
from deepspeed.git_version_info import version, git_hash, git_branch
from deepspeed.utils import log_dist, OnDevice

dp_factory = {'cuda': MMDataParallel, 'cpu': MMDataParallel}

ddp_factory = {'cuda': MMDistributedDataParallel}

ZeROddp_factory = {'cuda': MMDeepSpeedEngine}


def build_dp(model, device='cuda', dim=0, *args, **kwargs):
    """build DataParallel module by device type.

    if device is cuda, return a MMDataParallel module; if device is mlu,
    return a MLUDataParallel module.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, cuda, cpu or mlu. Defaults to cuda.
        dim (int): Dimension used to scatter the data. Defaults to 0.

    Returns:
        :class:`nn.Module`: parallelized module.
    """
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        assert digit_version(mmcv.__version__) >= digit_version('1.5.0'), \
                'Please use MMCV >= 1.5.0 for MLU training!'
        from mmcv.device.mlu import MLUDataParallel
        dp_factory['mlu'] = MLUDataParallel
        model = model.mlu()

    return dp_factory[device](model, dim=dim, *args, **kwargs)


def build_ddp(model, device='cuda', *args, **kwargs):
    """Build DistributedDataParallel module by device type.

    If device is cuda, return a MMDistributedDataParallel module;
    if device is mlu, return a MLUDistributedDataParallel module.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, mlu or cuda.

    Returns:
        :class:`nn.Module`: parallelized module.

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
                     DistributedDataParallel.html
    """
    assert device in ['cuda', 'mlu'], 'Only available for cuda or mlu devices.'
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        assert digit_version(mmcv.__version__) >= digit_version('1.5.0'), \
            'Please use MMCV >= 1.5.0 for MLU training!'
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
