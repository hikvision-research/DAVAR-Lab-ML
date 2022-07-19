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
# Abstract: helper functions for detaching the parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def detach_param(model):
    if model is None:
        return

    assert isinstance(model, nn.Module)
    for param in model.parameters():
        param.detach_()


def load_detector_ckpt(pretrained, backbone, neck, head):
    if pretrained is not None:
        pretrained_ckpt = torch.load(pretrained, map_location='cpu')
        if "state_dict" in pretrained_ckpt:
            weights = pretrained_ckpt["state_dict"]
        else:
            weights = pretrained_ckpt

        backbone_dict = OrderedDict()
        neck_dict = OrderedDict()
        bbox_head_dict = OrderedDict()
        for key in weights.keys():
            key_prefix = key.split('.')[0]
            if key_prefix == 'backbone':
                new_key = key[9:]
                backbone_dict[new_key] = weights[key]
            if key_prefix == "neck":
                new_key = key[5:]
                neck_dict[new_key] = weights[key]
            if key_prefix == "bbox_head":
                new_key = key[10:]
                bbox_head_dict[new_key] = weights[key]
        backbone.load_state_dict(backbone_dict)
        if len(neck_dict.keys()) == 0:
            neck.init_weights()
        else:
            neck.load_state_dict(neck_dict)
        if len(neck_dict.keys()) == 0:
            head.init_weights()
        else:
            head.load_state_dict(bbox_head_dict)
    else:
        backbone.init_weights(pretrained=None)
        if isinstance(neck, nn.Sequential):
            for m in neck:
                m.init_weights()
        else:
            neck.init_weights()
        head.init_weights()

def load_detector_ckpt_two_stage(pretrained, backbone, neck, rpn_head, roi_head):
    if pretrained is not None:
        pretrained_ckpt = torch.load(pretrained, map_location='cpu')
        if "state_dict" in pretrained_ckpt:
            weights = pretrained_ckpt["state_dict"]
        else:
            weights = pretrained_ckpt

        backbone_dict = OrderedDict()
        neck_dict = OrderedDict()
        rpn_head_dict = OrderedDict()
        roi_head_dict = OrderedDict()
        for key in weights.keys():
            key_prefix = key.split('.')[0]
            if key_prefix == 'backbone':
                new_key = key[9:]
                backbone_dict[new_key] = weights[key]
            if key_prefix == "neck":
                new_key = key[5:]
                neck_dict[new_key] = weights[key]
            if key_prefix == "rpn_head":
                new_key = key[9:]
                rpn_head_dict[new_key] = weights[key]
            if key_prefix == "roi_head":
                new_key = key[9:]
                roi_head_dict[new_key] = weights[key]

        backbone.load_state_dict(backbone_dict)
        if len(neck_dict.keys()) == 0:
            neck.init_weights()
        else:
            neck.load_state_dict(neck_dict)
        if len(rpn_head_dict.keys()) == 0:
            rpn_head.init_weights()
        else:
            rpn_head.load_state_dict(rpn_head_dict)
        if len(roi_head_dict.keys()) == 0:
            roi_head.init_weights(pretrained=None)
        else:
            roi_head.load_state_dict(roi_head_dict)
    else:
        backbone.init_weights(pretrained=None)
        if isinstance(neck, nn.Sequential):
            for m in neck:
                m.init_weights()
        else:
            neck.init_weights()
        rpn_head.init_weights()
        roi_head.init_weights(pretrained=None)