# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

import numpy as np
import torch
import torch.distributed as dist

from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_runner,
                         get_dist_info, GradientCumulativeFp16OptimizerHook,
                         GradientCumulativeOptimizerHook)

from mmdet.core import DistEvalHook, EvalHook, build_optimizer, DeepspeedDistEvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import (build_ddp, build_dp, build_ZeROddp, compat_cfg,
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


def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    samples_per_gpu = cfg.data.train_dataloader.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s) with {samples_per_gpu} '
                f'samples per GPU. The total batch size is {batch_size}.')

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)
    deepspeed_enabled = 'deepspeed' in cfg.keys() and cfg.get('deepspeed') is True
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        if deepspeed_enabled:
            # DeepSpeed will use different K/V pairs for pretrain checkpoint.
            # Therefore, we need to load pretrain checkpoint before build ZeRODDP.
            if cfg.load_from:
                load_checkpoint(model, cfg.load_from, strict=True)
            cfg.optimizer = optimizer
            cfg.model = model
            model, optimizer, _, _ = build_ZeROddp(
                model=model,
                optimizer=optimizer,
                model_parameters=model.parameters(),
                args=cfg,
            )
            model.device_ids = [int(os.environ['LOCAL_RANK'])]
        else:
            model = build_ddp(
                model,
                cfg.device,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    
    # build runner
    auto_scale_lr(cfg, distributed, logger)
    runner = build_runner( # {'type': 'IterBasedRunner', 'max_iters': 50000}
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

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
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
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
