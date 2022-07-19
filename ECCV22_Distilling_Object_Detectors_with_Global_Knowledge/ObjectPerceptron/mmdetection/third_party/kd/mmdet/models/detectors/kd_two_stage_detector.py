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
# Abstract: the knowledge distillation framework for two stage detector.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import kaiming_init
from mmdet.models import builder, DETECTORS
from mmdet.core import bbox2roi

from third_party.kd.mmdet.strategy.builder import build_strategy
from ..detach_param import (detach_param, load_detector_ckpt_two_stage)
from .kd_one_stage_detector import KD_One_Stage_Detector


@DETECTORS.register_module()
class KD_Two_Stage_Detector(KD_One_Stage_Detector):
    def __init__(self,
                 student,
                 teacher,
                 kd_cfg,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(KD_Two_Stage_Detector, self).__init__(
            student,
            teacher,
            kd_cfg)
        
        self.stu_cfg = student
        self.tea_cfg = teacher
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.kd_cfg = kd_cfg

        self.stu = nn.ModuleDict()
        self.stu['backbone'], self.stu['neck'], self.stu['rpn_head'], \
        self.stu['roi_head'], self.stu['adap_neck'] = self.build_student()

        self.tea = nn.ModuleDict()
        self.tea['backbone'], self.tea['neck'], self.tea['rpn_head'], \
        self.tea['roi_head'] = self.build_teacher()

        load_detector_ckpt_two_stage(self.stu_cfg.pretrained, self.stu['backbone'], self.stu['neck'], 
                           self.stu['rpn_head'], self.stu['roi_head'])
        load_detector_ckpt_two_stage(self.tea_cfg.pretrained, self.tea['backbone'], self.tea['neck'],
                           self.tea['rpn_head'], self.tea['roi_head'])

        for param in self.tea.parameters():
            param.requires_grad = False
        self.tea.eval()


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      runner,
                      gt_bboxes_ignore=None,
                      proto_st=None,
                      proto_tc=None,
                      cls_start_index=None):
        train_data = img['train_data']      # NCHW Tensor
        train_meta = img_metas['train_data']   # list of dict
        gt_bboxes = gt_bboxes['train_data']    # list of gpu tensor
        gt_labels = gt_labels['train_data']    # list of gpu tensor

        rpn_loss, bbox_loss, stu_kd_feat, tea_kd_feat, \
        cls_score_stu, cls_score_tea, pos_bboxes, pos_labels = \
            self.det_loss_kd_feature(train_data, gt_bboxes, gt_labels, train_meta)

        if self.kd_cfg.strategy.kd_loss['proposal_replace_gt']:
            gt_bboxes = pos_bboxes
            gt_labels = pos_labels
        kd_loss = self.kd_loss_strategy(runner,
                                        stu_kd_feat,
                                        tea_kd_feat,
                                        self.stu['rpn_head'],
                                        gt_bboxes,
                                        gt_labels,
                                        cls_score_stu,
                                        cls_score_tea,
                                        proto_st=proto_st,
                                        proto_tc=proto_tc,
                                        cls_start_index=cls_start_index,
                                        train_meta=train_meta)
        loss_ret = {}
        loss_ret.update(rpn_loss)
        loss_ret.update(bbox_loss)
        loss_ret.update(kd_loss)
        return loss_ret

    def det_loss_kd_feature(self, train_data, gt_bboxes, gt_labels, train_meta):
        backbone_stu = self.stu['backbone']
        neck_stu = self.stu['neck']
        rpn_head_stu = self.stu['rpn_head']
        roi_head_stu = self.stu['roi_head']
        adap_neck = self.stu['adap_neck']

        backbone_tea = self.tea['backbone']
        neck_tea = self.tea['neck']
        rpn_head_tea = self.tea['rpn_head']
        roi_head_tea = self.tea['roi_head']

        # student feature for kd
        _, fpn_stu = self.extract_feat(backbone_stu, neck_stu, train_data)
        kd_fpn_stu = [fpn_stu[0], fpn_stu[1], fpn_stu[2], fpn_stu[3], fpn_stu[4]]
        kd_fpn_adp_stu = []
        if adap_neck is not None:
            for i in range(len(kd_fpn_stu)):
                kd_fpn_adp_stu.append(adap_neck[i](kd_fpn_stu[i]))
        kd_fpn_stu = tuple(kd_fpn_stu)
        kd_fpn_adp_stu = tuple(kd_fpn_adp_stu)
        stu_kd_feat = {'neck_feat':kd_fpn_stu, 'neck_adp_feat':kd_fpn_adp_stu}

        # teacher feature for kd
        with torch.no_grad():
            _, fpn_tea = self.extract_feat(backbone_tea, neck_tea, train_data)
            kd_fpn_tea = [fpn_tea[0], fpn_tea[1], fpn_tea[2], fpn_tea[3], fpn_tea[4]]
            kd_fpn_tea = tuple(kd_fpn_tea)
            tea_kd_feat = {'neck_feat':kd_fpn_tea}

        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                            self.test_cfg.rpn)
        rpn_losses, proposal_list = rpn_head_stu.forward_train(
            fpn_stu,
            train_meta,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=None,
            proposal_cfg=proposal_cfg,
            return_outs=True)

        roi_losses, sampling_results = \
            roi_head_stu.forward_train(fpn_stu, train_meta, proposal_list,
                                gt_bboxes, gt_labels, gt_bboxes_ignore=None,
                                gt_masks=None, return_sampling_results=True)

        pos_bboxes = [res.pos_bboxes for res in sampling_results]
        pos_labels = []
        pos_assigned_gt_inds = []
        for img_id, res in enumerate(sampling_results):
            labels = gt_labels[img_id][res.pos_assigned_gt_inds]
            pos_assigned_gt_inds.extend(labels)
            pos_labels.append(labels)

        rois = bbox2roi(pos_bboxes)
        bbox_results_stu, roi_feats_stu, target_lvls = roi_head_stu._bbox_forward(fpn_stu, rois, return_roi_feats=True)
        bbox_results_tea, roi_feats_tea, _ = roi_head_tea._bbox_forward(fpn_tea, rois, return_roi_feats=True)
        stu_kd_feat['roi_feats'] = roi_feats_stu
        stu_kd_feat['target_lvls'] = target_lvls
        stu_kd_feat['pos_assigned_gt_inds'] = pos_assigned_gt_inds
        tea_kd_feat['roi_feats'] = roi_feats_tea
        cls_score_stu = bbox_results_stu['cls_score']
        cls_score_tea = bbox_results_tea['cls_score']

        return rpn_losses, roi_losses, stu_kd_feat, tea_kd_feat, cls_score_stu, cls_score_tea, pos_bboxes, pos_labels

    def build_student(self):
        backbone = builder.build_backbone(self.stu_cfg.backbone)
        neck = builder.build_neck(self.stu_cfg.neck)
        if self.kd_cfg.st_neck_channels is not None:
            adap_neck = self.build_adaptation_layers(self.kd_cfg.st_neck_channels, self.kd_cfg.tc_neck_channels)
        else:
            adap_neck = None
        self.stu_cfg.rpn_head['train_cfg'] = self.train_cfg.rpn
        self.stu_cfg.rpn_head['test_cfg'] = self.test_cfg.rpn
        rpn_head = builder.build_head(self.stu_cfg.rpn_head)
        self.stu_cfg.roi_head['train_cfg'] = self.train_cfg.rcnn
        self.stu_cfg.roi_head['test_cfg'] = self.test_cfg.rcnn
        roi_head = builder.build_head(self.stu_cfg.roi_head)

        return backbone, neck, rpn_head, roi_head, adap_neck

    def build_teacher(self):
        backbone = builder.build_backbone(self.tea_cfg.backbone)
        neck = builder.build_neck(self.tea_cfg.neck)
        self.tea_cfg.rpn_head['train_cfg'] = self.train_cfg.rpn
        self.tea_cfg.rpn_head['test_cfg'] = self.test_cfg.rpn
        self.tea_cfg.roi_head['train_cfg'] = self.train_cfg.rcnn
        self.tea_cfg.roi_head['test_cfg'] = self.test_cfg.rcnn
        rpn_head = builder.build_head(self.tea_cfg.rpn_head)
        roi_head = builder.build_head(self.tea_cfg.roi_head)

        return backbone, neck, rpn_head, roi_head

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        # test student
        branch = 'stu'
        backbone = getattr(self, branch)['backbone']
        neck = getattr(self, branch)['neck']
        rpn_head = getattr(self, branch)['rpn_head']
        roi_head = getattr(self, branch)['roi_head']

        _, x = self.extract_feat(backbone, neck, img)

        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape

        if proposals is None:
            proposal_list = rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)