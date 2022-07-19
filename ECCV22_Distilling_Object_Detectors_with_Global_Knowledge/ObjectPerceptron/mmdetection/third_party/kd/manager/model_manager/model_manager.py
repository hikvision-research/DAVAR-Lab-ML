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
# Abstract: the model manager for teacher and student model in knowledge distillation task.
"""
from mmdet.utils import get_root_logger
from mmdet.models import build_detector
from .base import BaseModelManager


class KDModelManager(BaseModelManager):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger = get_root_logger(log_level='INFO')

    def build_single_model(self,
                           model_name,
                           model_cfg,
                           train_cfg=None,
                           test_cfg=None,
                           checkpoint=None):
        model = build_detector(model_cfg,
                               train_cfg=train_cfg,
                               test_cfg=test_cfg)
        self.logger.info("{} initialized.".format(model_name))
        if checkpoint is not None:
            # call the parent method to load checkpoint
            self.load_checkpoint(checkpoint, model)
            self.logger.info("{} weights loaded".format(model_name))
        self._model[model_name] = model
        return self._model
