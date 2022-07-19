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
# Abstract: the base model manager.
"""
from abc import abstractmethod
import os.path as osp
import torch

from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger


class BaseModelManager(object):
    def __init__(self, cfg):
        """
        Args:
            cfg: config file
        """

        self.config = cfg
        self._model = dict()
        self.logger = get_root_logger(log_level='INFO')

    @property
    def model(self):
        """ return model dict"""
        return self._model

    def get_model(self, model_name='model'):
        """
         Get specific model by model name.
         input:
              model_name : key of model_name saved in dictionary,
                  can be listed in self.config.model.keys()

         output:
            self._model: dict of model instance, e.g., Detector instance, Recognizor instance, etc.
         """

        if model_name not in self._model:
            raise ValueError('{} has not built yet!'.format(model_name))
        return self._model[model_name]

    def build_single_model(self, model_name='model'): # mode='train'
        """
        Build model
        input:
            model_name : key of model_name saved in dictionary,
                can be listed in self.config.model.keys()

        output:
            self._model: dict of model instance, e.g., Detector instance, Recognizor instance, etc.
        """

        model = build_detector(self.config.model[model_name],
                               train_cfg=self.config.train_cfg,
                               test_cfg=self.config.test_cfg)
        self._model[model_name] = model
        return self._model

    def build_all_models(self):
        """Build all models in configs

        output:
            self._model: dict of model instance, e.g., Detector instance, Recognizor instance, etc.
        """
        for model_name in self.config.model.keys():
            model = build_detector(self.config.model[model_name],
                               train_cfg=self.config.train_cfg,
                               test_cfg=self.config.test_cfg)

            self._model[model_name] = model

        return self._model

    def load_checkpoint(self, checkpoint_file, model):
        assert osp.isfile(checkpoint_file)
        checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
        return checkpoint

    def reset(self):
        """ reset dataset manager, clean up all the builded datasets

        input:
            dataset_name(optional) : dataset to delete, if not given, delete all the datasets

        """
        del self._model
        self._model = dict()
        torch.cuda.empty_cache()