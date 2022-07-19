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
# Abstract: the base dataset manager.
"""
from abc import abstractmethod

from mmdet.datasets.builder import build_dataset, build_dataloader
from mmdet.utils import get_root_logger
import torch

class BaseDatasetManager(object):
    """
    Args:
        cfg: config file
    """
    def __init__(self, cfg):
        self._datasets = dict()
        self.config = cfg
        self.logger = get_root_logger(log_level='INFO')

    def build_single_dataset(self, dataset_name, dataset_cfg=None):
        """ build single dataset

        input:
            dataset_name : dataset name for save in dictionary
            dataset_cfg : dataset config info, self.config.data in default
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        if dataset_cfg is None:
            dataset_cfg = self.config.data
        dataset = build_dataset(dataset_cfg)
        self._datasets[dataset_name] = dataset
        return dataset

    def build_multi_datasets(self, dataset_list):
        """ given dataset key list, build multiple datasets for each key

        input:
            dataset_list : dataset key list,
                e.g., ['ds_train', 'ds_val', 'ds_test', 'ds_unlabeled', 'ds_al_labeled', 'full'] in Active Learning Tasks
        """
        for split in dataset_list:
            self._datasets[split] = self.build_single_dataset(split, getattr(self.config.data, split))
        return self._datasets

    @property
    def dataset(self):
        """ return dataset dict"""
        return self._datasets

    def get_dataset(self, dataset_name):
        """ given dataset key list, build multiple datasets for each key

        input:
            dataset_list : dataset key list,
                e.g., ['ds_train', 'ds_val', 'ds_test', 'ds_unlabeled', 'ds_al_labeled', 'full'] in Active Learning Tasks
        """

        if dataset_name not in self._datasets:
            raise ValueError('{} has not built yet!'.format(dataset_name))
        return self._datasets[dataset_name]

    def build_dataloader(self, dataset_name, **kwargs):
        """ build dataloader for specific dataset_name

        input:
            dataset_name : dataset name
        """
        dataset = self.get_dataset(dataset_name)
        loader = build_dataloader(dataset,
                                  imgs_per_gpu=self.config.data.imgs_per_gpu,
                                  workers_per_gpu=self.config.data.workers_per_gpu,
                                  **kwargs)
        return loader

    def reset(self, dataset_name=None):
        """ reset dataset manager, clean up all the builded datasets

        input:
            dataset_name(optional) : dataset to delete, if not given, delete all the datasets

        """
        if dataset_name:
            self._datasets[dataset_name] = None
        del self._datasets
        self._datasets = dict()
        torch.cuda.empty_cache()

    @abstractmethod
    def build_datasets(self):
        pass


