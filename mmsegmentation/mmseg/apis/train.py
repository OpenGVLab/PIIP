# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         build_runner, get_dist_info, OptimizerHook,
                         Fp16OptimizerHook, GradientCumulativeFp16OptimizerHook,
                         GradientCumulativeOptimizerHook)
from mmcv.utils import build_from_cfg

from mmseg import digit_version
from mmseg.core import DistEvalHook, EvalHook, build_optimizer, DeepspeedDistEvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import (build_ddp, build_dp, build_ZeROddp,
                         find_latest_checkpoint, get_root_logger)
from mmcv.runner.checkpoint import load_checkpoint

def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })

    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    optimizer = build_optimizer(model, cfg.optimizer)
    deepspeed_enabled = 'deepspeed' in cfg.keys() and cfg.get('deepspeed') is True
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # DDP wrapper
        if deepspeed_enabled:
            if cfg.load_from:
                # load_checkpoint(model, cfg.load_from, map_location="cpu", strict=False)
                ckpt = torch.load(cfg.load_from, map_location="cpu")['module']
                new_ckpt = {}
                for k, v in ckpt.items():
                    if "conv_seg.weight" in k or "conv_seg.bias" in k:
                        pass
                    else:
                        new_ckpt[k] = v
                message = model.load_state_dict(new_ckpt, strict=False)
                print(1, torch.distributed.get_rank(), message)
                logger.info(message)
                import time
                time.sleep(10)
            cfg.optimizer = optimizer
            cfg.model = model
            model, optimizer, _, _ = build_ZeROddp(
                model=model,
                optimizer=optimizer,
                model_parameters=model.parameters(),
                args=cfg,
            )
            if cfg.load_from:
                message = model.load_state_dict({"module." + k: v for k, v in new_ckpt.items()}, strict=False)
                print(2, torch.distributed.get_rank(), message)
                logger.info(message)
                import time
                time.sleep(10)
            model.device_ids = [int(os.environ['LOCAL_RANK'])]
        else:
            model = build_ddp(
                model,
                cfg.device,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    # calculate the total number of trainable parameters
    logger.info(f"trainable parameters: {sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'auxiliary_head' not in name)}")
    logger.info(f"total parameters: {sum(p.numel() for name, p in model.named_parameters() if 'auxiliary_head' not in name)}")
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # fp16 setting
    '''
    If we use DeepSpeed, both fp16 and gradient_cumulative should be None.
    That means, we must use OptimizerHook for DeepSpeed.
    '''
    fp16_cfg = cfg.get('fp16', None)
    gradient_cumulative_cfg = cfg.get('gradient_cumulative', None)
    if fp16_cfg is not None and gradient_cumulative_cfg is not None:
        optimizer_config = GradientCumulativeFp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, **gradient_cumulative_cfg,
            distributed=distributed)
    elif fp16_cfg is not None and gradient_cumulative_cfg is None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif fp16_cfg is None and gradient_cumulative_cfg is not None:
        optimizer_config = GradientCumulativeOptimizerHook(
            **cfg.optimizer_config, **gradient_cumulative_cfg)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config
    
    if deepspeed_enabled:
        assert isinstance(optimizer_config, OptimizerHook), "deepspeed must use OptimizerHook"
    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config, # <class 'mmcv.runner.hooks.optimizer.OptimizerHook'>
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        # when distributed training by epoch, using`DistSamplerSeedHook` to set
        # the different seed to distributed sampler for each epoch, it will
        # shuffle dataset at each epoch and avoid overfitting.
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # The specific dataloader settings
        val_loader_cfg = {
            **loader_cfg,
            'samples_per_gpu': 1,
            'shuffle': False,  # Not shuffle by default
            **cfg.data.get('val_dataloader', {}),
        }
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        if deepspeed_enabled:
            eval_hook = DeepspeedDistEvalHook
        else:
            eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
        if deepspeed_enabled:
            try:
                resume_from = cfg.work_dir
                dirs = os.listdir(os.path.join(resume_from, "latest"))
                dirs = [item for item in dirs if item.startswith("global_")]
                cfg.tag = os.path.join("latest", dirs[0])
            except:
                pass

    if resume_from is not None:
        cfg.resume_from = resume_from
        
    if cfg.resume_from:
        if deepspeed_enabled:
            try:
                if cfg.get("deepspeed_load_module_only", False):
                    # does not load optimizer and lr scheduler
                    runner.resume_deepspeed(
                        cfg.resume_from, 
                        tag=cfg.tag,
                        load_optimizer_states=False,
                        load_lr_scheduler_states=False,
                        load_module_only=True,
                    )
                else:
                    runner.resume_deepspeed(
                        cfg.resume_from, 
                        tag=cfg.tag,
                        load_optimizer_states=True,
                        load_lr_scheduler_states=True,
                        load_module_only=False,
                    )
            except Exception as e:
                print(e)
        else:
            runner.resume(cfg.resume_from)
    elif cfg.load_from and not deepspeed_enabled:
        # DeepSpeed load_from has been handled before build_ZeROddp
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
