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
# Abstract: the knowledge distillation framework for single stage detector.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import kaiming_init
from mmcv.runner import auto_fp16
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import builder, DETECTORS

from third_party.kd.mmdet.strategy.builder import build_strategy
from ..detach_param import (detach_param, load_detector_ckpt)


@DETECTORS.register_module()
class KD_One_Stage_Detector(nn.Module):
    def __init__(self,
                 student,
                 teacher,
                 kd_cfg,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(KD_One_Stage_Detector, self).__init__()

        self.stu_cfg = student
        self.tea_cfg = teacher
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.kd_cfg = kd_cfg

        if 'rpn_head' not in self.stu_cfg.keys():
            self.stu_cfg = student
            self.tea_cfg = teacher
            self.train_cfg = train_cfg
            self.test_cfg = test_cfg
            self.kd_cfg = kd_cfg

            self.stu = nn.ModuleDict()
            self.stu['backbone'], self.stu['neck'], self.stu['bbox_head'], \
                self.stu['neck_adp']= self.build_student()

            self.tea = nn.ModuleDict()
            self.tea['backbone'], self.tea['neck'], self.tea['bbox_head'] = \
                self.build_teacher()
            detach_param(self.tea)

            load_detector_ckpt(self.stu_cfg.pretrained, self.stu['backbone'],
                               self.stu['neck'], self.stu['bbox_head'])
            load_detector_ckpt(self.tea_cfg.pretrained, self.tea['backbone'],
                               self.tea['neck'], self.tea['bbox_head'])
            
            for param in self.tea.parameters():
                param.requires_grad = False
            self.tea.eval()

        self.kd_loss_strategy = build_strategy(self.kd_cfg.strategy.kd_loss)


    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)


    def build_adaptation_layers(self, in_channels, out_channels, activation=None):
        base_feat_adap = nn.ModuleList()
        for idx in range(len(in_channels)):
            if activation == 'relu':
                base_feat_adap.append(nn.Sequential(nn.Conv2d(in_channels[idx], out_channels[idx], kernel_size=3,
                                                              stride=1, padding=0, bias=True),
                                      nn.ReLU(inplace=True)).cuda())
            else:
                base_feat_adap.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels[idx], out_channels[idx], kernel_size=3, padding=1),
                        )
                    )
        for i in range(len(base_feat_adap)):
            if isinstance(base_feat_adap[i], nn.Conv2d):
                base_feat_adap[i][0].weight.data.normal_().fmod_(2).mul_(0.0001).add_(0)
            elif isinstance(base_feat_adap[i], nn.BatchNorm2d):
                base_feat_adap[i].weight.data.fill_(1)
                base_feat_adap[i].bias.data.zero_()

        return base_feat_adap

    def extract_feat(self, backbone, neck, img):
        bb_x = backbone(img)
        x = neck(bb_x)
        return bb_x, x

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

        det_loss, stu_kd_feat, tea_kd_feat, \
        cls_score_stu, cls_score_tea, pos_bboxes, pos_labels = \
            self.det_loss_kd_feature(train_data, gt_bboxes, gt_labels, train_meta)

        if self.kd_cfg.strategy.kd_loss['proposal_replace_gt']:
            gt_bboxes = pos_bboxes
            gt_labels = pos_labels
        kd_loss = self.kd_loss_strategy(runner,
                                        stu_kd_feat,
                                        tea_kd_feat,
                                        self.stu['bbox_head'],
                                        gt_bboxes,
                                        gt_labels,
                                        cls_score_stu,
                                        cls_score_tea,
                                        proto_st=proto_st,
                                        proto_tc=proto_tc,
                                        cls_start_index=cls_start_index,
                                        train_meta=train_meta)
        loss_ret = {}
        loss_ret.update(det_loss)
        loss_ret.update(kd_loss)
        return loss_ret


    def det_loss_kd_feature(self, train_data, gt_bboxes, gt_labels, train_meta):
        backbone_stu = self.stu['backbone']
        neck_stu = self.stu['neck']
        head_stu = self.stu['bbox_head']
        neck_adp = self.stu['neck_adp']

        backbone_tea = self.tea['backbone']
        neck_tea = self.tea['neck']
        head_tea = self.tea['bbox_head']

        _, neck_feat_stu = self.extract_feat(backbone_stu, neck_stu, train_data)

        with torch.no_grad():
            bb_feat_tea, neck_feat_tea = self.extract_feat(backbone_tea, neck_tea, train_data)
            outs_tea = head_tea(neck_feat_tea)

        kd_fpn_adp_stu = []
        if neck_adp is not None:
            for i in range(len(neck_feat_stu)):
                kd_fpn_adp_stu.append(neck_adp[i](neck_feat_stu[i]))

        kd_feat_stu = {'neck_feat':neck_feat_stu, 'neck_adp_feat':kd_fpn_adp_stu}
        kd_feat_tea = {'bb_feat':bb_feat_tea, 'neck_feat':neck_feat_tea}

        det_loss, cls_score_stu, cls_score_tea, sampling_results = \
            head_stu.forward_train(neck_feat_stu, train_meta, gt_bboxes, gt_labels,
               gt_bboxes_ignore=None, soft_target=outs_tea[0], return_sampling_results=True)
        cls_score_stu = torch.cat((cls_score_stu))
        cls_score_tea = torch.cat((cls_score_tea))

        pos_bboxes = [res.pos_bboxes for res in sampling_results]
        pos_labels = []
        pos_assigned_gt_inds = []
        for img_id, res in enumerate(sampling_results):
            labels = gt_labels[img_id][res.pos_assigned_gt_inds]
            pos_assigned_gt_inds.extend(labels)
            pos_labels.append(labels)
        kd_feat_stu['pos_assigned_gt_inds'] = pos_assigned_gt_inds

        return det_loss, kd_feat_stu, kd_feat_tea, cls_score_stu, cls_score_tea, pos_bboxes, pos_labels


    def build_student(self):
        backbone = builder.build_backbone(self.stu_cfg.backbone)
        neck = builder.build_neck(self.stu_cfg.neck)
        self.stu_cfg.bbox_head['train_cfg'] = self.train_cfg
        self.stu_cfg.bbox_head['test_cfg'] = self.test_cfg
        head = builder.build_head(self.stu_cfg.bbox_head)

        feat_adap_neck = None
        if self.kd_cfg.st_neck_channels is not None:
            feat_adap_neck = self.build_adaptation_layers(self.kd_cfg.st_neck_channels, self.kd_cfg.tc_neck_channels, self.kd_cfg.adpt_activation)
        return backbone, neck, head, feat_adap_neck

    def build_teacher(self):
        backbone = builder.build_backbone(self.tea_cfg.backbone)
        neck = builder.build_neck(self.tea_cfg.neck)
        self.tea_cfg.bbox_head['train_cfg'] = self.train_cfg
        self.tea_cfg.bbox_head['test_cfg'] = self.test_cfg
        head = builder.build_head(self.tea_cfg.bbox_head)
        return backbone, neck, head

    def simple_test(self, img, img_meta, rescale=False):
        # test student
        branch = 'stu'
        backbone = getattr(self, branch)['backbone']
        neck = getattr(self, branch)['neck']
        bbox_head = getattr(self, branch)['bbox_head']
        
        _, x = self.extract_feat(backbone, neck, img)
        outs = bbox_head(x)
        # get origin input shape to support onnx dynamic shape
        if torch.onnx.is_in_onnx_export():
            # get shape as tensor
            img_shape = torch._shape_as_tensor(img)[2:]
            img_meta[0]['img_shape_for_onnx'] = img_shape
        bbox_list = bbox_head.get_bboxes(
            *outs, img_meta, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            self.bbox2result(det_bboxes, det_labels, bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_meta (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return ValueError('multiscale test is not supported now')

    def bbox2result(self, bboxes, labels, num_classes, cls_ret=False):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (Tensor): shape (n, 5)
            labels (Tensor): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class
        """
        if bboxes.shape[0] == 0:
            if not cls_ret:
                return [
                    np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)
                ]
            else:
                return [
                    np.zeros((0, 5+1+num_classes), dtype=np.float32) for i in range(num_classes)
                ]
        else:
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            return [bboxes[labels == i, :] for i in range(num_classes)]