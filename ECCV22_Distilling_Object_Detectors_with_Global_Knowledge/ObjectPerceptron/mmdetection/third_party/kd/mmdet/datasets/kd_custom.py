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
# Abstract: the dataset for knowledge distillation.
"""
import os.path as osp
import collections
import json

import mmcv
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import CustomDataset,DATASETS


@DATASETS.register_module()
class KDDataset(CustomDataset):
    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 classes_config=None,
                 imgs_per_gpu=1):

        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.imgs_per_gpu = imgs_per_gpu
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals),
        data_infos = self.load_annotations(self.ann_file)
        self.data_infos = self._cvt_list(data_infos)
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)
        if classes_config is not None:
            self.classes_config = mmcv.load(classes_config)
        else:
            self.classes_config = None



    @staticmethod
    def _load_json(json_file):
        with open(json_file, 'r') as f:
            ann = json.load(f, object_pairs_hook=collections.OrderedDict)
        return ann

    def _cvt_list(self, img_info):
        result_dict = []

        if "###" in img_info.keys():
            del img_info["###"]

        for key in img_info.keys():
            tmp_dict = dict()
            tmp_dict["filename"] = key
            tmp_dict["height"] = img_info[key]["height"]
            tmp_dict["width"] = img_info[key]["width"]
            tmp_dict["ann"] = img_info[key]["content_ann"]
            tmp_dict["ann2"] = img_info[key].get("content_ann2", None)

            result_dict.append(tmp_dict)

        return result_dict

    def load_annotations(self, ann_file):
        return self._load_json(ann_file)

    def load_proposals(self, proposal_file):
        return self._load_json(proposal_file)

    def get_ann_info(self, idx):
        return self.data_infos[idx].get('ann', None)

    def get_ann_info_2(self, idx):
        return self.data_infos[idx].get('ann2', None)

    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        results['cbbox_fields'] = []

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        ann_info_2 = self.get_ann_info_2(idx)
        results = dict(img_info=img_info, ann_info=ann_info, ann_info_2=ann_info_2, classes_config=self.classes_config)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        ann_info_2 = self.get_ann_info_2(idx)
        results = dict(img_info=img_info, ann_info=ann_info, ann_info_2=ann_info_2)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)
