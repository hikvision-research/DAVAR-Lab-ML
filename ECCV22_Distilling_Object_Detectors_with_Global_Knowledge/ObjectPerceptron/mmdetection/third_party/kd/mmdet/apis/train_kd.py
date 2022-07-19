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
# Abstract: the API for training with knowledge distillation.
"""
import random
import numpy as np
import torch
from collections import OrderedDict

from mmcv import Config
from mmcv.runner import Fp16OptimizerHook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, OptimizerHook, build_optimizer)
from mmcv.runner import load_checkpoint
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import build_dataset
from mmdet.utils import get_root_logger
from mmdet.models import build_detector

from third_party.kd.mmcv.runner import DistSamplerSeedHook_KD, DistOptimizerHookKD, PGMHook, RunnerKD
from third_party.kd.mmdet.datasets import build_dataloader


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


def parse_losses(losses):
    log_vars = OrderedDict()
    if 'acc' in losses.keys():
        losses['prob_acc'] = losses['acc']
        del losses['acc']
    loss_names = [k for k in losses.keys() if k.startswith('loss_')]
    probe_names = [k for k in losses.keys() if k.startswith('prob_')]
    assert(len(loss_names)+len(probe_names) == len(losses))

    for loss_name in loss_names:
        loss_value = losses[loss_name]
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if _key.startswith('loss_'))
    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    prob_vars = OrderedDict()
    for prob_name in probe_names:
        prob_value = losses[prob_name]
        prob_vars[prob_name] = prob_value

    return loss, log_vars, prob_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)

    loss, log_vars, prob_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, prob_vars=prob_vars,
        num_samples=len(data['img']['train_data'].data))

    return outputs


def _dist_train(model, dataset, cfg, logger, timestamp):
    data_loader = build_dataloader(
                    dataset,
                    cfg.data.samples_per_gpu,
                    cfg.data.workers_per_gpu,
                    sampler=cfg.data.get('sampler', None),
                    dist=True)

    # put model on gpus
    model = MMDistributedDataParallel(model.cuda(),
       device_ids=[torch.cuda.current_device()],
       broadcast_buffers=True,
       find_unused_parameters=True)

    # build runner
    optimizer_stu = build_optimizer(model.module.stu, cfg.optimizer)
    optimizer = dict(stu=optimizer_stu)
    if model.module.tea is not None:
        optimizer_tea = build_optimizer(model.module.tea, cfg.optimizer)
        optimizer.update(dict(tea=optimizer_tea))

    runner = RunnerKD(model, batch_processor, optimizer, cfg.work_dir,
                      cfg.log_level, logger, kd_cfg=cfg.model.kd_cfg,
                      ckpt_cfg=cfg.checkpoint_config)

    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        raise NotImplementedError('Support SSL Method should be added')
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                             **fp16_cfg)
    else:
        optimizer_config = DistOptimizerHookKD(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    runner.register_hook(DistSamplerSeedHook_KD())
    kd_loss = cfg.model.kd_cfg.strategy.kd_loss
    if 'use_pair_kd' in kd_loss.keys():
        if kd_loss.pair_wise in ['use_pgm', 'robustness_for_point'] and kd_loss.use_pair_kd:
            runner.register_hook(PGMHook(cfg.pgm_cfg, cfg.data))

    # register eval hooks
    distributed = True
    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    val_dataloader = build_dataloader(
        val_dataset,
        1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    eval_cfg = cfg.get('evaluation', {})
    eval_hook = DistEvalHook if distributed else EvalHook
    runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    if 'total_epochs' in cfg.keys():
        runner.run([data_loader], cfg.workflow, cfg.total_epochs)
    else:
        runner.run([data_loader], cfg.workflow, cfg.runner.max_epochs)


def train_detector_kd(model,
                      dataset,
                      cfg,
                      distributed=False,
                      logger=None,
                      timestamp=None,
                      meta=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, logger, timestamp)
    else:
        raise NotImplementedError