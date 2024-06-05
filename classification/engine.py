# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import util.misc


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    
    metric_logger = util.misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', util.misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = 200
    
    accum_iter = args.accum_iter
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
        
        allow_autocast = not args.disable_autocast
        with torch.cuda.amp.autocast(allow_autocast):
            outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            raise ValueError("Loss is {}, stopping training".format(loss_value))
            

        loss /= accum_iter
        update_grad = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=update_grad)
        if update_grad:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, disable_autocast=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = util.misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    print_freq = 200

    # switch to evaluation mode
    model.eval()
    
    normal_batch_size = next(iter(data_loader))[0].size(0)

    for data_iter_step, (images, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        current_batch_size = images.size(0)
        if current_batch_size < normal_batch_size:
            # pad to the standard batch size
            repeats = normal_batch_size // current_batch_size + 1
            images = images.repeat_interleave(repeats, dim=0)[:normal_batch_size]
            target = target.repeat_interleave(repeats)[:normal_batch_size]


        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        allow_autocast = not disable_autocast
        with torch.cuda.amp.autocast(allow_autocast):
            output = model(images)
            if current_batch_size < normal_batch_size:
                output = output[:current_batch_size]
                target = target[:current_batch_size]

            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=current_batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=current_batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
