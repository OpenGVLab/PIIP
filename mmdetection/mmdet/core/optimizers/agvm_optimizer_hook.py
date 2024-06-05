# Copyright (c) SenseTime X-Lab. All rights reserved.
import warnings

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad
from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class AgvmOptimizerHook(Hook):

    def __init__(self, 
                 lr_update_interval=5,
                 lr_update_start_iter=500,
                 lr_update_stop_iter=-1,
                 momentum=0.97, 
                 warmup_iters=500, 
                 use_adam=False, 
                 static_lr=False, 
                 grad_clip=None,
                 debug_mode=False,
                 grad_clip_iter_range=None):
        self.lr_update_interval = lr_update_interval
        self.lr_update_start_iter = lr_update_start_iter
        self.lr_update_stop_iter = lr_update_stop_iter if lr_update_stop_iter > lr_update_start_iter else 10e9
        self.momentum = momentum
        self.grad_clip = grad_clip
        self.warmup_iters = warmup_iters
        self.use_adam = use_adam
        self.static_lr = static_lr
        self.grad_clip_iter_range = grad_clip_iter_range
        self.debug_mode = debug_mode

        if self.grad_clip_iter_range is not None:
            assert isinstance(grad_clip_iter_range, (list, tuple)) and \
                   len(grad_clip_iter_range) == 2 and \
                   grad_clip_iter_range[0] < grad_clip_iter_range[1], \
                   '\"grad_clip_iter_range\" must be a tuple or a list with two elements: \
                    [iter_min, iter_max]'

        if self.lr_update_start_iter < self.warmup_iters:
            warnings.warn("It is recommended to start agvm after warm up.")

        self.scale_factors = []
        self.world_size = dist.get_world_size()
        self.rank_group_a = dist.new_group(ranks=range(0, self.world_size // 2))
        self.rank_group_b = dist.new_group(ranks=range(self.world_size // 2, self.world_size))
        self.rank_group_simi = dist.new_group(ranks=[0, min(self.world_size // 2 + 8, self.world_size - 1)])

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def _update_optimizer_lr(self, runner, scale_factors):
        for group_id, group in enumerate(runner.optimizer.param_groups):
            group['lr'] *= scale_factors[group_id]
    
    def _recover_optimizer_lr(self, runner, scale_factors):
        for group_id, group in enumerate(runner.optimizer.param_groups):
            group['lr'] /= scale_factors[group_id]
    
    def _cat_all_param_grads(self, optimizer, group):
        param_list = group['params']
        all_param_grads = []
        if not self.use_adam:
            all_param_grads = [param.grad.reshape(-1) for param in param_list if param.grad is not None]
        else:
            for param in param_list:
                if param.grad is not None:
                    exp_avg_sq, beta2 = optimizer.state[param]['exp_avg_sq'], group['betas'][1]
                    cur_grad_sq = param.grad.data ** 2
                    cur_exp_avg_sq = exp_avg_sq.mul(beta2) + cur_grad_sq.mul(1 - beta2)
                    new_grad = param.grad / (cur_exp_avg_sq.sqrt().add(group['eps']))
                    all_param_grads.append(new_grad.reshape(-1))
        return torch.cat(all_param_grads)
    
    def _normal_update(self, runner):
        loss = runner.outputs['loss'] / self.world_size
        loss.backward()

        for name, param in runner.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad.data)

    def _agvm_update(self, runner):
        loss = runner.outputs['loss'] / (self.world_size * self.world_size // 2)
        loss.backward()

        for _, param in runner.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad.data, group=self.rank_group_a)
                dist.all_reduce(param.grad.data, group=self.rank_group_b)

        all_param_grad_list = [self._cat_all_param_grads(runner.optimizer, group) * self.world_size
                               for group in runner.optimizer.param_groups]

        grad_len = [len(_) for _ in all_param_grad_list]
        all_param_grad = torch.cat(all_param_grad_list)

        grad_bucket = [torch.zeros_like(all_param_grad) for _ in range(2)]
        dist.all_gather(grad_bucket, all_param_grad, group=self.rank_group_simi)

        for name, param in runner.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad.data)

        similarities = torch.zeros(len(grad_len)).to(all_param_grad)
        if dist.get_rank() == 0:
            grad_a = torch.split(grad_bucket[0], grad_len)
            grad_b = torch.split(grad_bucket[1], grad_len)
            similarities = torch.cat([1 - torch.cosine_similarity(a, b, dim=0).unsqueeze(0)
                                     for a, b in zip(grad_a, grad_b)])
            if self.debug_mode:
                print(similarities)
        dist.broadcast(similarities, 0)

        new_scale_factors = [torch.sqrt(torch.clamp(((similarities[0] + 1e-6) / (simi + 1e-6)), 0.5, 10.0)).item() for simi in similarities]
        
        return new_scale_factors
                
    def after_train_iter(self, runner):
        if len(self.scale_factors) == 0:
            self.scale_factors = [1.0 for _ in range(len(runner.optimizer.param_groups))]

        runner.optimizer.zero_grad()
        if not self.static_lr and self.lr_update_start_iter <= runner.iter <= self.lr_update_stop_iter and runner.iter % self.lr_update_interval == 0:
            new_scale_factors = self._agvm_update(runner)
            self.scale_factors = [self.momentum * old + (1 - self.momentum) * new
                                  for old, new in zip(self.scale_factors, new_scale_factors)]
            if dist.get_rank() == 0 and self.debug_mode:
                print(self.scale_factors)
        else:
            self._normal_update(runner)

        if self.grad_clip is not None:
            if self.grad_clip_iter_range is None:
                clip = True
            else:
                iter_low, iter_high = self.grad_clip_iter_range
                clip = iter_low <= runner.iter <= iter_high
            if clip:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                            runner.outputs['num_samples'])
            else:
                # Remove grad norm from the logger
                runner.log_buffer.val_history.pop('grad_norm', None)

        if self.static_lr or runner.iter < self.lr_update_start_iter or runner.iter > self.lr_update_stop_iter:
            runner.optimizer.step()
        else:
            self._update_optimizer_lr(runner, self.scale_factors)
            runner.optimizer.step()
            self._recover_optimizer_lr(runner, self.scale_factors)