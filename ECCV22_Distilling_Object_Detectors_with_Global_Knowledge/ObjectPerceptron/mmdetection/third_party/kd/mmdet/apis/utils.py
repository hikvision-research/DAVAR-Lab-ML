"""
# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# #
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Abstract: define some helper functions.
"""
import os.path as osp
import numpy as np
import json
import torch
import re
import torch.distributed as dist
from collections import OrderedDict 

from mmcv.runner import obj_from_dict


def build_optimizer(model,optimizer_cfg,train_params):
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, torch.optim,
                             dict(params=(model.parameters() if not train_params else train_params)))
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options
                or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        params = []
        
        for name, param in (model.named_parameters() if not train_params else train_params):
            
            param_group = {'params': [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)
        
        
def parse_losses(losses):
    """
    parse losses and sum all the loss values
    Args:
        losses (dict):

    Returns:
        loss (float)
        log_vars (OrderedDict):
    """
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def batch_processor(model, data, regularization_terms, ewc_lambda, train_mode):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    losses = model(**data)
    loss, log_vars = parse_losses(losses)
    
    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs
    
    
def ltrb2xywh(boxes, reduction=1):
    ltx = boxes[:, 0]
    lty = boxes[:, 1]
    rbx = boxes[:, 2]
    rby = boxes[:, 3]

    cx = (ltx + rbx) * 0.5
    cy = (lty + rby) * 0.5
    w = (rbx - ltx) + 1.0 / reduction
    h = (rby - lty) + 1.0 / reduction

    new_boxes = torch.zeros_like(boxes)
    new_boxes[:, 0] = cx
    new_boxes[:, 1] = cy
    new_boxes[:, 2] = w
    new_boxes[:, 3] = h

    return new_boxes


def bbox_ious(boxes1, boxes2):
    b1_len = boxes1.size(0)
    b2_len = boxes2.size(0)

    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions