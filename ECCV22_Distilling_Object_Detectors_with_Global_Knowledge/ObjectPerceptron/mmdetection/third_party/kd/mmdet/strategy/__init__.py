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
# Abstract: the kd strategy module.
"""
from .base import Strategy
from .builder import build_strategy
from .kd_one_stage_strategy import OneStageDetStrategy
from .kd_two_stage_strategy import TwoStageDetStrategy
from .prototype_generation_module import PrototypeGenerationModule

__all__ = ['Strategy', 'build_strategy',
          'OneStageDetStrategy', 'TwoStageDetStrategy', 'PrototypeGenerationModule']
