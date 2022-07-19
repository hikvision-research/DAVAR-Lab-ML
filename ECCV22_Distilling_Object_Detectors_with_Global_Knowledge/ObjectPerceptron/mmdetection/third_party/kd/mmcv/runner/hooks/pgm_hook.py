# !/usr/bin/python
# -*- coding: utf8 -*-
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
# Abstract: the hook function for generating prototypes in Prototype Generation Module (PGM).
"""
import os
import tempfile
import shutil

import torch
import torch.distributed as dist

import mmcv
from mmcv.runner.hooks import Hook, HOOKS
from mmcv.runner.dist_utils import get_dist_info

from third_party.kd.mmdet.strategy.builder import build_strategy


@HOOKS.register_module()
class PGMHook(Hook):
    def __init__(self, pgm_cfg, data):
        super(PGMHook, self).__init__()
        self.pgm_cfg = pgm_cfg
        self.data = data
        self.openmetric_strategy = build_strategy(pgm_cfg.strategy_cfg)

    def before_train_epoch(self, runner):
        rank, _ = get_dist_info()
        epoch_now = int(runner.epoch / runner.prototype_each_epoch)
        pgm_feat_path = os.path.join(runner.work_dir, 'pgm/epoch_' + str(epoch_now))
        if not os.path.exists(pgm_feat_path):
            dist.barrier()
            if rank == 0:
                print('***********************************************')
                print('Begin to generate prototype in: {}'.format(pgm_feat_path))
                mmcv.mkdir_or_exist(pgm_feat_path)
            dist.barrier()
            self.pgm_generator(pgm_feat_path, runner, self.data.train)
        if rank == 0:
            if runner.epoch + 1 <= self.pgm_cfg.strategy_cfg['save_ckpt_min']:
                ckpt_path = os.path.join(runner.work_dir, 'epoch_{}.pth').format(runner.epoch)
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
        dist.barrier()


    def pgm_generator(self, save_path, runner, pgm_dataset):
        tea_checkpoint = self.pgm_cfg.runtime_cfg.checkpoints[0]

        if runner.epoch == 0:
            stu_checkpoint = self.pgm_cfg.runtime_cfg.checkpoints[1]
        else:
            stu_checkpoint = os.path.join(runner.work_dir, 'epoch_' + str(runner.epoch) + '.pth')

        cls_names = []
        max_samples = []
        max_num_prototypes = []
        cls_names_file = mmcv.load(self.data.train.classes_config)

        for cls in cls_names_file['classes']:
            cls_names.append(cls)
            max_samples.append(self.pgm_cfg.runtime_cfg.max_samples)
            max_num_prototypes.append(self.pgm_cfg.runtime_cfg.max_num_prototypes)

        self.pgm_cfg.runtime_cfg.checkpoints = [tea_checkpoint, stu_checkpoint]
        self.pgm_cfg.runtime_cfg.pseudolabel_path = pgm_dataset.ann_file
        self.pgm_cfg.runtime_cfg.cls_names = cls_names
        self.pgm_cfg.runtime_cfg.max_samples = max_samples
        self.pgm_cfg.runtime_cfg.pgm_dataset = pgm_dataset
        self.pgm_cfg.runtime_cfg.max_num_prototypes = max_num_prototypes
        self.pgm_cfg.runtime_cfg.save_path = save_path
        self.openmetric_strategy(**self.pgm_cfg.runtime_cfg)