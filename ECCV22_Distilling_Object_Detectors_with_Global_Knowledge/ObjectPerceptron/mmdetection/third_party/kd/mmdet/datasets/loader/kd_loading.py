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
# Abstract: the loading pipelines for knolwedge distillation.
"""
import os.path as osp
import mmcv
import cv2
import numpy as np
from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class KDLoadAnnotations():
    def __init__(self,
                 with_bbox=False,
                 with_poly_bbox=False,
                 with_poly_mask=False,
                 with_care=False,
                 with_label=False,
                 with_multi_label=False,
                 with_text=False,
                 with_cbbox=False,
                 text_profile=None,
                 label_start_index=1,
                 ):
        self.with_bbox = with_bbox
        self.with_poly_bbox = with_poly_bbox
        self.with_poly_mask = with_poly_mask
        self.with_care = with_care
        self.with_label = with_label
        self.with_multi_label = with_multi_label
        self.with_text = with_text
        self.text_profile = text_profile
        self.label_start_index=label_start_index
        self.with_cbbox = with_cbbox

        assert not (self.with_label and self.with_multi_label), \
            "Only one of with_label and with_multi_label can be true"

        if text_profile is not None:
            if 'character' in self.text_profile:
                if osp.exists(self.text_profile['character']):
                    print("loading characters from file: %s" % self.text_profile['character'])
                    with open(self.text_profile['character'], 'r', encoding='utf8') as character_file:
                        characters = character_file.readline().strip().split(' ')
                        self.character = ''.join(characters)
                elif isinstance(self.text_profile['character'], str):
                    self.character = self.text_profile['character']
                else:
                    raise NotImplementedError
            else:
                self.character = ''

            if 'text_max_length' in self.text_profile:
                self.text_max_length = self.text_profile['text_max_length']
            else:
                self.text_max_length = 25

            if 'sensitive' in self.text_profile:
                self.sensitive = self.text_profile['sensitive']
            else:
                self.sensitive = True

            if 'filtered' in self.text_profile:
                self.filtered = self.text_profile['filtered']
            else:
                self.filtered = True

    def _load_cares(self, results):
        ann = results['ann_info']

        bboxes_length = len(ann['bboxes']) if 'bboxes' in ann else 1

        if self.with_care:
            cares = np.array(ann.get('cares', np.ones(bboxes_length)))
        else:
            cares = np.ones(bboxes_length)

        results["cares"] = cares
        return results

    def _load_bboxes(self, results):
        ann = results['ann_info']
        cares = results['cares']

        tmp_gt_bboxes = ann.get('bboxes', [])

        gt_bboxes = []
        gt_bboxes_ignore = []
        for i, box in enumerate(tmp_gt_bboxes):
            box_i = np.array(box, dtype=np.double)
            x_coords = box_i[0::2]
            y_coords = box_i[1::2]
            aligned_box = [
                np.min(x_coords),
                np.min(y_coords),
                np.max(x_coords),
                np.max(y_coords)
            ]
            if cares[i] == 1:
                gt_bboxes.append(aligned_box)
            else:
                gt_bboxes_ignore.append(aligned_box)

        if len(gt_bboxes) == 0:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        else:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)

        if len(gt_bboxes_ignore) == 0:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        else:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_ignore'] = gt_bboxes_ignore

        results['bbox_fields'].append('gt_bboxes')
        results['bbox_fields'].append('gt_bboxes_ignore')
        return results


    def _load_labels(self, results):
        ann = results['ann_info']
        cares = results['cares']
        tmp_labels = ann.get("labels", None)
        bboxes_length = len(ann['bboxes']) if 'bboxes' in ann else 1

        if tmp_labels is None:
            tmp_labels = [[1]] * bboxes_length

        if isinstance(self.label_start_index, list):
            self.label_start_index = self.label_start_index[0]

        gt_labels=[]

        for i, label in enumerate(tmp_labels):
            if cares[i] == 1:
                gt_labels.append(label[0])

        if len(gt_labels) > 0 and isinstance(gt_labels[0], str):
            assert results['classes_config'] is not None
            assert 'classes' in results['classes_config'] or 'classes_0' in results['classes_config']
            for i, _ in enumerate(gt_labels):
                if 'classes_0' in results['classes_config']:
                    classes_config = results['classes_config']["classes_0"]
                else:
                    classes_config = results['classes_config']['classes']

                if self.label_start_index == -1:
                    assert 'NotLabeled' or 'NoLabel' in classes_config
                    if 'NotLabeled' in classes_config:
                        notlabeled_index = classes_config.index('NotLabeled')
                    else:
                        notlabeled_index = classes_config.index('NoLabel')
                    if notlabeled_index > 0:
                        classes_config[notlabeled_index], classes_config[0] = \
                            classes_config[0], classes_config[notlabeled_index]

                gt_labels[i] = classes_config.index(gt_labels[i]) + self.label_start_index

        results['gt_labels'] = gt_labels

        return results


    def __call__(self, results):
        assert 'ann_info' in results
        results = self._load_cares(results)

        if self.with_bbox:
            results = self._load_bboxes(results)
        if self.with_label:
            results = self._load_labels(results)



        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            '(with_bbox={}, with_poly_bbox={}, with_poly_mask={}, '
            'with_care={}, with_label={}, with_multi_lalbel={}, '
            'with_text={}, with_cbbox={}').format(
            self.with_bbox, self.with_poly_bbox, self.with_poly_mask,
            self.with_care, self.with_label,
            self.with_multi_label, self.with_text, self.with_cbbox)
        return repr_str
