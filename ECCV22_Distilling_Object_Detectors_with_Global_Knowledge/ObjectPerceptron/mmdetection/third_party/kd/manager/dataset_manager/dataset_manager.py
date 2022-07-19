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
# Abstract: the dataset manager for knowledge distillation.
"""
from .base import BaseDatasetManager


class KDDatasetManager(BaseDatasetManager):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_datasets(self, dataset_names, dataset_cfgs):
        for name, cfg in zip(dataset_names, dataset_cfgs):
            self.build_single_dataset(name, dataset_cfg=cfg)
