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
# Abstract: The base strategy.
"""
from abc import abstractmethod

from mmdet.utils import get_root_logger


class Strategy(object):
    """
    Base Class for strategy pool in auto-system.
         offer examples in al_strategies/random.py for randomly selecting strategy

    Info: 2020/07/12 added by xuyunlu

    """

    def __init__(self, **kwargs):
        self.logger = get_root_logger(log_level='INFO')

    @abstractmethod
    def __call__(self, inputs):
        # outputs一般为ModelManager.test()返回的内容
        pass