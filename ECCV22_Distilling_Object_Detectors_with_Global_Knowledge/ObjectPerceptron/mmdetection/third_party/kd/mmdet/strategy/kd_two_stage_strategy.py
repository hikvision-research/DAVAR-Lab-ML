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
# Abstract: KD strategy (compute loss) for two-stage detector, e.g., Faster R-CNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import mmcv
from mmdet.core import multi_apply
from mmdet.models.builder import build_loss

from .registry import STRATEGY
from .base import Strategy
from .kd_one_stage_strategy import OneStageDetStrategy
from third_party.kd.mmdet.apis.utils import ltrb2xywh, bbox_ious


@STRATEGY.register_module()
class TwoStageDetStrategy(OneStageDetStrategy):
    def __init__(self,
                 start_epoch=0,
                 distributed=True,
                 kd_attenuation = False,
                 use_point_kd = False,
                 point_strategy = 'decouple',
                 use_soft_label = False,
                 use_pair_kd = False,
                 pair_wise = 'use_pgm',
                 kd_param=None,
                 neck_kd_lambda=None,
                 kd_t=None,
                 pair_wise_factor = 20,
                 **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch
        self.kd_attenuation = kd_attenuation
        self.use_point_kd = use_point_kd
        self.point_strategy = point_strategy
        self.use_soft_label = use_soft_label
        self.use_pair_kd = use_pair_kd
        self.pair_wise = pair_wise
        self.kd_param = kd_param
        self.neck_kd_lambda = neck_kd_lambda
        self.kd_t = kd_t
        self.pair_wise_factor = pair_wise_factor

        self.pool_ops = nn.AdaptiveAvgPool2d(1)
        loss_kd=dict(type='KDLoss', kd_t=self.kd_t)
        self.loss_kd = build_loss(loss_kd)


    def r2i_soft_weight(self, protos_stu, protos_tea, cls_start_index, roi_feats_stu,
                        roi_feats_tea, target_lvls, pos_assigned_gt_inds):
        robust_weights = []
        roi_num = len(roi_feats_stu)
        for ind in range(roi_num):
            roi_feat_stu = roi_feats_stu[ind]
            roi_feat_tea = roi_feats_tea[ind]
            level = target_lvls[ind]
            label = pos_assigned_gt_inds[ind]
            start_ind = cls_start_index[level][label]
            end_ind = cls_start_index[level][label+1]

            proto_stu = protos_stu[level][start_ind:end_ind]
            proto_tea = protos_tea[level][start_ind:end_ind]

            pooling_st = self.pool_ops(roi_feat_stu).squeeze()
            pooling_tc = self.pool_ops(roi_feat_tea).squeeze()

            st_qscore = self.template_box_distance(pooling_st.unsqueeze(0), proto_stu)
            tc_qscore = self.template_box_distance(pooling_tc.unsqueeze(0), proto_tea)
            st_qscore = st_qscore.max()
            tc_qscore = tc_qscore.max()
            target = 1-abs(st_qscore-tc_qscore)
            robust_weights.append(target.detach())

        if len(robust_weights) == 0:
            return None

        return torch.stack(robust_weights)


    def __call__(self, runner, st_feature, tc_feature, bbox_head, gt_bboxes, gt_labels, 
                cls_score_stu, cls_score_tea, proto_st=None, proto_tc=None, 
                cls_start_index=None, train_meta=None):
        if runner.epoch < self.start_epoch:
            loss_dict = dict()
            return loss_dict

        self.gt_bboxes = gt_bboxes
        self.gt_labels = gt_labels
        self.bbox_head = bbox_head
        self.train_meta = train_meta
        kd_soft_loss = torch.Tensor([0]).cuda()

        self.kd_att_val = 1
        if self.kd_attenuation:
            self.kd_att_val = 1 - (runner.epoch / runner.max_epochs)

        st_feature = [st_feature]
        tc_feature = [tc_feature]
        feature_num = len(st_feature[0]['neck_feat'])
        assert st_feature, 'The student feature should not be None.'
        assert len(st_feature[0]['neck_feat']) == feature_num, 'The input feature dimention is wrong: {} != {}'.format(len(st_feature[0]['neck_feat']), feature_num)

        st_feat = self.convert(st_feature, feature_num=feature_num)
        tc_feat = self.convert(tc_feature, feature_num=feature_num)

        anchor_strides = bbox_head.anchor_generator.strides
        # do not use prototype
        if proto_st is None:
            proto_st = [None for _ in range(len(anchor_strides))]
            proto_tc = [None for _ in range(len(anchor_strides))]
            cls_start_index = [None for _ in range(len(anchor_strides))]
        else:
         assert len(st_feat) == len(proto_st), 'st_feat adn proto_st have different dimension'

        self.level_num = len(st_feat)
        levels = [i for i in range(self.level_num)]
        point_loss, bg_point_loss, rkd_angle_loss, \
        rkd_dis_loss, r2i_loss = multi_apply(self.forward_single,
                                             st_feat,
                                             tc_feat,
                                             bbox_head.anchor_generator.base_sizes,
                                             proto_st,
                                             proto_tc,
                                             cls_start_index,
                                             self.neck_kd_lambda,
                                             levels)
        
        if self.use_soft_label:
            roi_feats_stu = st_feature[0]['roi_feats']
            roi_feats_tea = tc_feature[0]['roi_feats']
            target_lvls = st_feature[0]['target_lvls']
            pos_assigned_gt_inds = st_feature[0]['pos_assigned_gt_inds']
            soft_weights = None
            if self.use_pair_kd:
                soft_weights = self.r2i_soft_weight(proto_st, proto_tc, cls_start_index, 
                        roi_feats_stu, roi_feats_tea, target_lvls, pos_assigned_gt_inds)
            kd_soft_loss = self.loss_kd(self.kd_param,
                                        st_pred=cls_score_stu,
                                        tc_pred=cls_score_tea,
                                        soft_weights=soft_weights)

        loss_dict = dict()
        if self.use_point_kd:
            loss_dict["loss_kd_point"] = sum(_loss for _loss in point_loss) / self.level_num
            if self.point_strategy == 'decouple':
                loss_dict["loss_bg_point"] = sum(_loss for _loss in bg_point_loss) / self.level_num
        if self.use_soft_label:
            loss_dict["loss_kd_soft"] = kd_soft_loss['pred'] 
        if self.use_pair_kd and self.pair_wise == 'use_rkd':
            loss_dict["loss_kd_rkd_angle"] = rkd_angle_loss
            loss_dict["loss_kd_rkd_dis"] = rkd_dis_loss
        if self.use_pair_kd and self.pair_wise == 'use_r2i':
            loss_dict["loss_kd_pair"] = r2i_loss

        return loss_dict