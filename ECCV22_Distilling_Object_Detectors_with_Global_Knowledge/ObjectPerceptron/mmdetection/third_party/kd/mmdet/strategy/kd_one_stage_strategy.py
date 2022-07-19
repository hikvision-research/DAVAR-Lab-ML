# -*- coding: utf-8 -*-
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
# Abstract: KD strategy (compute loss) for single stage detector, e.g. Retina-Net
"""
import os
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import mmcv
from mmdet.core import multi_apply
from mmdet.models.builder import build_loss

from .registry import STRATEGY
from .base import Strategy
from third_party.kd.mmdet.apis.utils import ltrb2xywh, bbox_ious


@STRATEGY.register_module()
class OneStageDetStrategy(Strategy):
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

    
    def pdist(self, feature, squared=False, eps=1e-12):
        e_square = feature.pow(2).sum(dim=1)
        prod = feature @ feature.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()
        res = res.clone()
        res[range(len(feature)), range(len(feature))] = 0

        return res

    def rkd_loss(self, gt_list, st_feature, tc_feature, anchor_stride):
        assert len(gt_list) == len(st_feature)
        pool_features_st = []
        pool_features_tc = []
        pool_st_simloss = []
        pool_tc_simloss = []
        box_flag = False
        device_now = st_feature.device
        angle_diff = torch.tensor(0, requires_grad=False, device=device_now)
        dis_diff = torch.tensor(0, requires_grad=False, device=device_now)

        for image_id in range(len(gt_list)):
            if (len(gt_list[image_id]) == 0):
                continue
            else:
                box_flag = True
                for bbox in range(gt_list[image_id].shape[0]):
                    [cx_low, cy_low, cx_high, cy_high] = gt_list[image_id][bbox].cpu()
                    cy_low = int(cy_low / anchor_stride)
                    cy_high = int(cy_high / anchor_stride) + 1
                    cx_low = int(cx_low / anchor_stride)
                    cx_high = int(cx_high / anchor_stride) + 1

                    bbox_st_feat = st_feature[image_id, :, cy_low:cy_high, cx_low:cx_high]
                    bbox_tc_feat = tc_feature[image_id, :, cy_low:cy_high, cx_low:cx_high]

                    pooling_st = self.pool_ops(bbox_st_feat).squeeze()
                    pooling_tc = self.pool_ops(bbox_tc_feat).squeeze()
                    pool_features_st.append(pooling_st)
                    pool_features_tc.append(pooling_tc)

        if box_flag is False:
            return angle_diff, dis_diff

        features_st = torch.stack(pool_features_st, dim=0)
        features_tc = torch.stack(pool_features_tc, dim=0)
        
        sim_map_st = features_st.unsqueeze(0) - features_st.unsqueeze(1)
        norm_st = F.normalize(sim_map_st, p=2, dim=2)
        st_angle = torch.bmm(norm_st, norm_st.transpose(1, 2)).view(-1)
        sim_map_tc = features_tc.unsqueeze(0) - features_tc.unsqueeze(1)
        norm_tc = F.normalize(sim_map_tc, p=2, dim=2)
        tc_angle = torch.bmm(norm_tc, norm_tc.transpose(1, 2)).view(-1)

        tc_d = self.pdist(features_tc, squared=False)
        mean_td = tc_d[tc_d>0].mean()
        tc_d = tc_d / mean_td
        st_d = self.pdist(features_st, squared=False)
        mean_d = st_d[st_d>0].mean()
        st_d = st_d / mean_d

        dis_diff = F.smooth_l1_loss(st_d, tc_d, reduction='elementwise_mean') * self.pair_wise_factor * self.kd_att_val
        angle_diff = F.smooth_l1_loss(st_angle, tc_angle, reduction='elementwise_mean') * self.pair_wise_factor * 2 * self.kd_att_val

        return angle_diff, dis_diff


    def _map_roi_levels(self, rois, num_levels):
        scale = torch.sqrt(
            (rois[:, 2] - rois[:, 0] + 1) * (rois[:, 3] - rois[:, 1] + 1))
        target_lvls = torch.floor(torch.log2(scale / 56 + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls


    def get_gt_mask(self, bbox_head, feature, gt_bboxes, level, weights=None):
        if len(feature) == 0:
            return None
        if weights is not None:
            assert len(weights) == len(gt_bboxes), 'weights and gt_bboxes have different dimension.'

        featmap_size = [feature.size()[-2:]]
        featmap_strides = bbox_head.anchor_generator.strides
        featmap_stride = featmap_strides[level][0]  # featmap_strides:((8,8) , (16,16))
        imit_range = [0, 0, 0, 0, 0]
        with torch.no_grad():
            mask_batch = []
            for batch in range(len(gt_bboxes)):
                target_lvls = self._map_roi_levels(gt_bboxes[batch], self.level_num)
                gt_level = gt_bboxes[batch][target_lvls==level]  # gt_bboxes: BatchsizexNpointx4coordinate
                gt_level_ind = torch.where([target_lvls==level][0] == True)[0]
                h, w = featmap_size[0][0], featmap_size[0][1]
                mask_per_img = torch.zeros([h, w], dtype=torch.double).cuda()
                for ins in range(gt_level.shape[0]):
                    gt_level_map = gt_level[ins] / featmap_stride
                    lx = max(int(gt_level_map[0]) - imit_range[level], 0)
                    rx = min(int(gt_level_map[2]) + imit_range[level], w)
                    ly = max(int(gt_level_map[1]) - imit_range[level], 0)
                    ry = min(int(gt_level_map[3]) + imit_range[level], h)

                    if weights is None:
                        if (lx == rx) or (ly == ry):
                            mask_per_img[ly, lx] += 1
                        else:
                            mask_per_img[ly:ry, lx:rx] += 1
                    else:
                        weight = weights[batch][gt_level_ind[ins]].cuda()
                        if (lx == rx) or (ly == ry):
                            mask_per_img[ly, lx] = max(mask_per_img[ly, lx], weight)
                        else:
                            mask_per_img[ly:ry, lx:rx] = weight

                if weights is None:
                    mask_per_img = (mask_per_img > 0).double()
                    mask_per_img = mask_per_img.unsqueeze(0).repeat(feature.size()[1], 1, 1)
                    mask_batch.append(mask_per_img)
                else:
                    mask_per_img = mask_per_img.unsqueeze(0).repeat(feature.size()[1], 1, 1)
                    mask_batch.append(mask_per_img)

        return torch.stack(mask_batch)


    def template_box_distance(self, pooling_box, template_feat):
        box_norm = F.normalize(pooling_box)
        template_norm = F.normalize(template_feat).to(box_norm[0].device)
        distance = box_norm.mm(template_norm.t())
        return distance


    def r2i_om_loss(self, gt_list, st_feature, tc_feature, template_student, template_teacher,
                    anchor_stride, cls_start_index, use_omloss, level=None):
        assert len(gt_list) == len(st_feature)
        pool_features_st = []
        pool_features_tc = []
        scores_for_point = []
        scores_for_rkd = []
        box_flag = False
        device_now = st_feature.device
        sum_diff = torch.tensor(0, requires_grad=False, device=device_now).float()

        for image_id in range(len(gt_list)):
            if (len(gt_list[image_id]) == 0):
                robustness_tmp = torch.zeros(1).to(device_now)
                scores_for_point.append(robustness_tmp)
                continue
            else:
                box_flag = True
                robustness_tmp = []

                for bbox in range(len(gt_list[image_id])):
                    [cx_low, cy_low, cx_high, cy_high] = gt_list[image_id][bbox].cpu()
                    cy_low = int(abs(cy_low) / anchor_stride)
                    cy_high = int(abs(cy_high) / anchor_stride) + 1
                    cx_low = int(abs(cx_low) / anchor_stride)
                    cx_high = int(abs(cx_high) / anchor_stride) + 1

                    bbox_st_feat = st_feature[image_id, :, cy_low:cy_high, cx_low:cx_high]
                    bbox_tc_feat = tc_feature[image_id, :, cy_low:cy_high, cx_low:cx_high]
                    pooling_st = self.pool_ops(bbox_st_feat).squeeze()
                    pooling_tc = self.pool_ops(bbox_tc_feat).squeeze()
                    pool_features_st.append(pooling_st)
                    pool_features_tc.append(pooling_tc)

                    label_b = int(self.gt_labels[image_id][bbox])
                    start_index = cls_start_index[label_b]
                    end_index = cls_start_index[label_b+1]
                    same_class_st = template_student[start_index: end_index]
                    same_class_tc = template_teacher[start_index: end_index]
                    st_qscore = self.template_box_distance(pooling_st.unsqueeze(0), same_class_st)
                    tc_qscore = self.template_box_distance(pooling_tc.unsqueeze(0), same_class_tc)
                    st_qscore = st_qscore.max()
                    tc_qscore = tc_qscore.max()
                    target = 1 - abs(st_qscore - tc_qscore)
                    robustness_tmp.append(target.detach())
                    scores_for_rkd.append(target.detach())

                robustness_tmp = torch.tensor(robustness_tmp)
                scores_for_point.append(robustness_tmp)

        if use_omloss is False:
            return scores_for_point
        if box_flag is False:
            return sum_diff, scores_for_point

        if len(pool_features_st) > 0:
            scores_for_rkd = torch.stack(scores_for_rkd, dim=0).unsqueeze(1)
            features_st = torch.stack(pool_features_st, dim=0)
            features_tc = torch.stack(pool_features_tc, dim=0)
            dist_st = self.template_box_distance(features_st, template_student) 
            dist_tc = self.template_box_distance(features_tc, template_teacher) 

            st_distance = dist_st * scores_for_rkd
            tc_distance = dist_tc * scores_for_rkd

            sum_diff = F.smooth_l1_loss(st_distance, tc_distance, reduction='elementwise_mean') * self.pair_wise_factor * self.kd_att_val

        return sum_diff, scores_for_point


    def forward_single(self, st_feat, tc_feat, anchor_stride, proto_student_feat,
                       proto_teacher_feat, cls_start_index, neck_kd_lambda, level):
        device_now=st_feat['neck_feat'][0].device
        point_loss = torch.tensor(0, requires_grad=False, device=device_now).float()
        bg_point_loss = torch.tensor(0, requires_grad=False, device=device_now).float()
        pair_loss_rkd_angle = torch.tensor(0, requires_grad=False, device=device_now).float()
        pair_loss_rkd_dis = torch.tensor(0, requires_grad=False, device=device_now).float()
        pair_loss_r2i = torch.tensor(0, requires_grad=False, device=device_now).float()

        st_neck_feat = st_feat['neck_feat'][0]
        st_neck_adp_feat = st_feat['neck_adp_feat'][0]
        tc_neck_feat = tc_feat['neck_feat'][0]
        student_feat, teacher_feat = None, None
        robust_weights = None

        if self.use_pair_kd:
            if self.pair_wise == 'use_rkd':
                pair_loss_rkd_angle, pair_loss_rkd_dis = \
                            self.rkd_loss(self.gt_bboxes, st_neck_feat, tc_neck_feat, anchor_stride)
            elif self.pair_wise == 'use_r2i':
                pair_loss_r2i, robust_weights = \
                        self.r2i_om_loss(self.gt_bboxes, st_neck_feat, tc_neck_feat, 
                            proto_student_feat, proto_teacher_feat, anchor_stride, 
                            cls_start_index, True, level=level)
            else:
                robust_weights = self.r2i_om_loss(self.gt_bboxes, st_neck_feat, tc_neck_feat, 
                                            proto_student_feat, proto_teacher_feat, anchor_stride, 
                                            cls_start_index, False, level=level)

        if self.use_point_kd:
            student_feat = [st_neck_adp_feat]
            teacher_feat = [tc_neck_feat]

            if self.point_strategy == 'decouple':
                feature_mask_kd = self.get_gt_mask(self.bbox_head,
                                                   st_neck_adp_feat,
                                                   self.gt_bboxes,
                                                   level,
                                                   robust_weights)
            else:
                feature_mask_kd = None
            kd_mask = [feature_mask_kd, None]

            if self.use_point_kd:
                loss_dict = self.loss_kd(self.kd_param,
                                          neck_kd_lambda,
                                          student_feat,
                                          teacher_feat,
                                          kd_mask,
                                          p_str=self.point_strategy)
                point_loss = loss_dict['feat'] * self.kd_att_val
                if self.point_strategy == 'decouple':
                    bg_point_loss = loss_dict['feat_bg'] * self.kd_att_val

        return point_loss, bg_point_loss, pair_loss_rkd_angle, pair_loss_rkd_dis, pair_loss_r2i


    def convert(self, feature_dict, feature_num):
        feats = []
        for j in range(feature_num):
            feats.append({})
        for i in range(len(feature_dict)):
            for j in range(feature_num):
                if i == 0:
                    feats[j]['bb_feat'], feats[j]['neck_feat'], \
                    feats[j]['bb_adp_feat'], feats[j]['neck_adp_feat']= [], [], [], []

                feats[j]['neck_feat'].append(feature_dict[i]['neck_feat'][j])
                if 'neck_adp_feat' in feature_dict[i].keys():
                    feats[j]['neck_adp_feat'].append(feature_dict[i]['neck_adp_feat'][j])

                if 'bb_feat' in feature_dict[i].keys():
                    if j < len(feature_dict[i]['bb_feat']):
                        feats[j]['bb_feat'].append(feature_dict[i]['bb_feat'][j])
                        if 'bb_adp_feat' in feature_dict[i].keys():
                            feats[j]['bb_adp_feat'].append(feature_dict[i]['bb_adp_feat'][j])

        return feats


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
        anchor_strides = bbox_head.anchor_generator.strides
        feature_num = len(anchor_strides)
        assert st_feature, 'The student feature should not be None.'
        assert len(st_feature[0]['neck_feat']) == feature_num, 'The input feature dimention is wrong: {} != {}'.format(len(st_feature[0]['feat']), feature_num)

        st_feat = self.convert(st_feature, feature_num=feature_num)
        tc_feat = self.convert(tc_feature, feature_num=feature_num)

        # do not use prototype
        if proto_st is None:
            proto_st = [None for _ in range(feature_num)]
            proto_tc = [None for _ in range(feature_num)]
            cls_start_index = [None for _ in range(feature_num)]
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
            pos_assigned_gt_inds = st_feature[0]['pos_assigned_gt_inds']
            soft_weights = None
            if self.use_pair_kd:
                soft_weights = self.soft_weight(anchor_strides, proto_st, proto_tc, cls_start_index, 
                                                    st_feat, tc_feat, pos_assigned_gt_inds)
            kd_soft_loss = self.loss_kd(self.kd_param,
                                        st_pred=cls_score_stu,
                                        tc_pred=cls_score_tea,
                                        soft_weights=soft_weights)

        loss_dict = dict()
        if self.use_point_kd:
            loss_dict["loss_kd_point"] = point_loss
            if self.point_strategy == 'decouple':
                loss_dict["loss_bg_point"] = bg_point_loss
        if self.use_soft_label:
            loss_dict["loss_kd_soft"] = kd_soft_loss['pred'] 
        if self.use_pair_kd and self.pair_wise == 'use_rkd':
            loss_dict["loss_kd_rkd_angle"] = rkd_angle_loss
            loss_dict["loss_kd_rkd_dis"] = rkd_dis_loss
        if self.use_pair_kd and self.pair_wise == 'use_r2i':
            loss_dict["loss_kd_pair"] = r2i_loss

        return loss_dict


    def soft_weight(self, anchor_strides, protos_stu, protos_tea, cls_start_index,
                        st_feat, tc_feat, pos_assigned_gt_inds):
        robust_weights = []
        roi_num = len(pos_assigned_gt_inds)
        roi_box = torch.cat((self.gt_bboxes))

        image_id = []
        for id, gt_bbox in enumerate(self.gt_bboxes):
            image_id.extend([id for _ in range(len(gt_bbox))])
        target_lvls = self._map_roi_levels(roi_box, self.level_num)

        for ind in range(roi_num):
            level = target_lvls[ind]
            anchor_stride = anchor_strides[level][0]
            label = pos_assigned_gt_inds[ind]
            start_ind = cls_start_index[level][label]
            end_ind = cls_start_index[level][label+1]

            [cx_low, cy_low, cx_high, cy_high] = roi_box[ind].cpu()
            cy_low = int(abs(cy_low) / anchor_stride)
            cy_high = int(abs(cy_high) / anchor_stride) + 1
            cx_low = int(abs(cx_low) / anchor_stride)
            cx_high = int(abs(cx_high) / anchor_stride) + 1

            bbox_st_feat = st_feat[level]['neck_feat'][0][image_id[ind], :, cy_low:cy_high, cx_low:cx_high]
            bbox_tc_feat = tc_feat[level]['neck_feat'][0][image_id[ind], :, cy_low:cy_high, cx_low:cx_high]
            pooling_st = self.pool_ops(bbox_st_feat).squeeze()
            pooling_tc = self.pool_ops(bbox_tc_feat).squeeze()
            proto_stu = protos_stu[level][start_ind:end_ind]
            proto_tea = protos_tea[level][start_ind:end_ind]

            st_qscore = self.template_box_distance(pooling_st.unsqueeze(0), proto_stu)
            tc_qscore = self.template_box_distance(pooling_tc.unsqueeze(0), proto_tea)
            st_qscore = st_qscore.max()
            tc_qscore = tc_qscore.max()
            target = 1 - abs(st_qscore - tc_qscore)
            robust_weights.append(target.detach())

        if len(robust_weights) == 0:
            return None

        return torch.stack(robust_weights)