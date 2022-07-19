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
# Abstract: the definition of MSE loss for knowledge distillation.
"""
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.losses import weight_reduce_loss
from mmdet.models import LOSSES

def mse_loss(pred, label, weight=None, reduction='mean', avg_factor=None):
    loss = F.mse_loss(pred, label, reduction='none')
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@LOSSES.register_module()
class KDMSELoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        loss = self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            avg_factor=avg_factor)
        return loss