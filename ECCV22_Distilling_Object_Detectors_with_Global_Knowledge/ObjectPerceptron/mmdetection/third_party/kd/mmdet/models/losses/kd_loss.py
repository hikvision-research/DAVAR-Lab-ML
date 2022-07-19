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
# Abstract: the definition of kd loss.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import LOSSES
from mmdet.models.builder import build_loss

@LOSSES.register_module()
class KDLoss(nn.Module):
    def __init__(self,
                 kd_t=15):
        super().__init__()
        self.kd_t = kd_t
        mse_loss = dict(type='KDMSELoss',
                        reduction='mean',
                        loss_weight=1.0)
        self.loss_mse = build_loss(mse_loss)

    def forward(self, kd_param=None, neck_kd_lambda=None, st_feat=None, tc_feat=None,
                feat_mask=None, st_pred=None, tc_pred=None, p_str='decouple', soft_weights=None):
        feat_loss = torch.Tensor([0]).cuda()
        feat_bg_loss = torch.Tensor([0]).cuda()
        cls_kd_loss = torch.Tensor([0]).cuda()
        loss_kd = {'feat':[], 'pred':[]}

        if (st_feat is not None) and (tc_feat is not None):
            assert len(st_feat) == len(tc_feat)
            if p_str == 'decouple':
                feat_mask_onehot = copy.deepcopy(feat_mask[0])
                feat_mask_onehot[torch.where(feat_mask_onehot > 0)] = 1
                norms = max(1.0, feat_mask[0].sum() * 2)
                feat_loss = (torch.pow(st_feat[0] - tc_feat[0], 2) * feat_mask[0]).sum() / norms * kd_param[2][1] * kd_param[0][0]
                norms_back = max(1.0, (1 - feat_mask_onehot).sum() * 2)
                feat_bg_loss = (torch.pow(st_feat[0] - tc_feat[0], 2) * 
                            (1 - feat_mask_onehot)).sum() / norms_back * kd_param[2][0] * kd_param[0][0]
            else:
                pass

        if (st_pred is not None) and (tc_pred is not None):
            assert len(st_pred) == len(tc_pred)
            if soft_weights is not None:
                cls_kd_tmp = self.knowledge_distillation_kl_div_loss(st_pred, tc_pred, self.kd_t, soft_weights)
            else:
                cls_kd_tmp = F.kl_div(F.log_softmax(st_pred / self.kd_t, dim=1), F.softmax(tc_pred / self.kd_t, dim=1), reduction='mean') \
                                  * self.kd_t * self.kd_t * kd_param[1][0]
            cls_kd_loss += (cls_kd_tmp * kd_param[1][0])
        else:
            pass

        loss_kd['feat'] = feat_loss
        loss_kd['feat_bg'] = feat_bg_loss
        loss_kd['pred'] = cls_kd_loss

        return loss_kd


    def knowledge_distillation_kl_div_loss(self, pred, soft_label, T, weights):
        """Loss function for knowledge distilling using KL divergence.
           from mmdetection2.11.0/models/losses/kd_loss.py

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            T (int): Temperature for distillation.
            detach_target (bool): Remove soft_label from automatic differentiation

        Returns:
            torch.Tensor: Loss tensor with shape (N,).
        """
        assert pred.size() == soft_label.size()
        target = F.softmax(soft_label / T, dim=1)

        kd_loss = F.kl_div(
            F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (
                T * T)

        if weights is not None:
            kd_loss *= weights

        return kd_loss.mean()
        
