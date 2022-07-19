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
# #
# Abstract       :    init model and logger, check config and start trainning.
"""
import os
import time
import argparse

import torch
import torch.distributed as dist
from torch import nn

import mmcv
from mmcv import Config
from mmcv.runner import init_dist
from mmdet.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed

from third_party.kd.mmdet.apis import train_detector_kd
from third_party.kd.manager.dataset_manager import KDDatasetManager
from third_party.kd.manager.model_manager import KDModelManager


class TRAINPROCESS():
    """
    TRAINPROCESS Class for preparing for training and starting the main process.
    """

    def __init__(self,
                model_def_file={},
                distributed=True,
                work_dir=None,
                ):
                
        self.model_def_file = model_def_file
        self.distributed = distributed
        self._init_config()
        if work_dir is not None:
            self.model_def_file.work_dir = work_dir
        self._init_logger()
        self._model = None
        self.config_train_check()

    def main_process(self):
        '''
        main process for training.
        '''
        # build models
        built_models = self.model_manager.build_single_model('kd_model',
                                                             self.cfg.model)
        self.model = built_models['kd_model']

        # build labeled dataset
        train_dataset = self.dataset_manager.build_single_dataset('train_data', self.cfg.data.train)

        # start trainning
        train_detector_kd(
            self.model,
            train_dataset,
            self.cfg,
            distributed=True,
            logger=self.logger,
            timestamp=self.timestamp)

    def config_train_check(self):
        '''
        check config.
        '''
        self.cfg = self.model_def_file

        if not os.path.exists(self.cfg.model.student.pretrained):
            raise IOError('{} is not exist！'.format(self.cfg.model.student.pretrained))

        if not os.path.exists(self.cfg.data.train.ann_file):
            raise IOError('{} is not exist！'.format(self.cfg.data.train.ann_file))

        if self.cfg.model.teacher is not None and (not os.path.exists(self.cfg.model.teacher.pretrained)):
            raise IOError('{} is not exist！'.format(self.cfg.model.teacher.pretrained))

        # build model_manager and dataset_manager
        self.model_manager = KDModelManager(self.cfg)
        self.dataset_manager = KDDatasetManager(self.cfg)

        if self.cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in checkpoints as meta data
            self.cfg.checkpoint_config.meta = dict(config=self.cfg.text)

    def _init_logger(self):
        '''
        init logger.
        '''
        work_dir = self.model_def_file.get('work_dir', '') if self.model_def_file else ''

        # generate logger file
        mmcv.mkdir_or_exist(work_dir)
        log_level = self.model_def_file.get('log_level', 'INFO') if self.model_def_file else 'INFO'
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(work_dir, '{}.log'.format(self.timestamp))
        self.logger = get_root_logger(log_file=log_file, log_level=log_level)

        self.meta = dict()

        # save the information of environment
        env_info_dict = collect_env()
        env_info = '\n'.join([('{}: {}'.format(k, v)) for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        self.meta['env_info'] = env_info

        # save the information of trainning
        self.logger.info('Distributed training: {}'.format(self.distributed))
        self.logger.info('Config:\n{}'.format(self.model_def_file.text))

        # set random seed 
        seed = self.model_def_file.get('seed', None) if self.model_def_file else None
        if seed is not None:
            self.logger.info('Set random seed to {}, deterministic: {}'.format(seed, True))
            set_random_seed(seed, deterministic=True)
        self.meta['seed'] = seed

    def _init_config(self):
        '''
        init config.
        '''
        if self.model_def_file is not None:
            if isinstance(self.model_def_file, str):
                if not os.path.exists(self.model_def_file):
                    raise IOError('{} is not exist！'.format(self.model_def_file))

                self.model_def_file = mmcv.Config.fromfile(self.model_def_file)
            elif not isinstance(self.model_def_file, mmcv.Config):
                raise ValueError('config must be a str or mmcv.Config,'
                                 'current type is ：{}'.format(type(self.model_def_file)))


def parse_args():
    parser = argparse.ArgumentParser(description='Training process')
    parser.add_argument('config_file', type=str, help='path to the config file')
    parser.add_argument('--gpus', type=int, default=1, help='number of training gpus')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--work_dir', type=str, help='path to the work direction')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config_file)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # distributed environment initialization  
        init_dist(args.launcher, **cfg.dist_params)

    KD = TRAINPROCESS(
        model_def_file = args.config_file,
        distributed = distributed,
        work_dir = args.work_dir,
    )

    KD.main_process()

if __name__ == '__main__':
    main()