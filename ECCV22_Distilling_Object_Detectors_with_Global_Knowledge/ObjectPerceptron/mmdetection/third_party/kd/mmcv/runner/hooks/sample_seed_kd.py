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
# Abstract: the sampler seed hook for setting the sampler seed before each training epoch.
"""
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class DistSamplerSeedHook_KD(Hook):
    def before_epoch(self, runner):
        pass

    def before_train_epoch(self, runner):
        runner.data_loader.sampler.set_epoch(runner.epoch)