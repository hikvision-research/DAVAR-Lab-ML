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
# Abstract: generate prototypes.
"""
import os
import argparse
import random
import shutil
import cv2
import torch
import json
import copy
import numpy as np
import torch.nn as nn
import torch.distributed as dist

import mmcv
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint, init_dist
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmcv.runner.dist_utils import master_only

from .base import Strategy
from .registry import STRATEGY


@STRATEGY.register_module()
class PrototypeGenerationModule(Strategy):
    """
    PrototypeGenerationModule Class for generating prototypes.
    """

    def __init__(self,
                 seed=1,
                 silent_mode=False,
                 **kwargs):

        super(PrototypeGenerationModule, self).__init__(**kwargs)
        self._prototype_proposal_json_name = 'tmp_valid_predictions_{}.json'
        self.silent_mode = silent_mode
        np.random.seed(seed)

    def _print(self, content):
        if not self.silent_mode:
            rank, _ = get_dist_info()
            if rank == 0:
                print(content)

    def _filter_predictions(self,
                            predictions,
                            conf_thre=0.95,
                            min_size=(30, 30),
                            cls_names=('person', )):
        '''
        filter the valid bboxes in predictions.
        '''
        valid_predictions = dict()
        for key, pred in predictions.items():
            width = pred['width']
            height = pred['height']
            content_ann = pred['content_ann']
            bboxes = np.array(content_ann['bboxes'])
            labels = np.array(content_ann['labels'])
            confs = np.ones(len(labels)).astype(np.float32)


            if len(bboxes) == 0:
                continue

            box_w = np.abs(bboxes[:, 2] - bboxes[:, 0]).reshape(-1)
            box_h = np.abs(bboxes[:, 3] - bboxes[:, 1]).reshape(-1)
            valid_idx = (confs > conf_thre) & (box_w > min_size[0]) & (
                    box_h > min_size[1])
            valid_idx = np.where(valid_idx)[0]
            valid_cls_idx = []

            for idx in valid_idx:
                if labels[idx][0] in cls_names:
                    valid_cls_idx.append(idx)
            valid_idx = valid_cls_idx
            if len(valid_idx) == 0:
                continue
            bboxes = bboxes[valid_idx, :]
            labels = labels[valid_idx]
            confs = confs[valid_idx]
            valid_predictions[key] = dict(
                width=width,
                height=height,
                content_ann=dict(
                    bboxes=bboxes.tolist(),
                    labels=labels.tolist(),
                    confs=confs.tolist()
                )
            )
        return valid_predictions

    def _collect_results_cpu(self,
                             result_part,
                             size,
                             tmp_dir=None,
                             save_path='./'):
        """
        Merge the results of each cpu.
        """
        if tmp_dir is None:
            tmp_dir = 'collect_tmp/'
        rank, world_size = get_dist_info()
        tmpdir = os.path.join(save_path, tmp_dir)
        mmcv.mkdir_or_exist(tmpdir)
        
        # dump the part result to the dir
        mmcv.dump(result_part, os.path.join(tmpdir, 'part_{}.pkl'.format(rank)))
        dist.barrier()
        # collect all parts
        if rank != 0:
            return None
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = os.path.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(self._load(part_file))

        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        shutil.rmtree(tmpdir)
        return ordered_results


    def _fetch_roi_features(self,
                            models,
                            data_loader,
                            save_path):
        '''
        fetch the region of interest features.
        '''
        results = []
        for model in models:
            model.eval()
            results.append([])
        dataset = data_loader.dataset
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))

        pool_operation = nn.AdaptiveAvgPool2d(1)
        for _, data in enumerate(data_loader):
            gt_bboxes = data['gt_bboxes'][0].numpy().squeeze(0)
            img = data['img'][0]
            img_pad_h, img_pad_w = img.shape[2:]
            for model_id, model in enumerate(models):
                with torch.no_grad():
                    x = model.module.backbone(img.cuda())
                    x_fpn_all = model.module.neck(x)
                    layer_num = len(x_fpn_all)
                    if _ == 0:
                        results[model_id] = [[] for i in range(layer_num)]
                    for layer_id, x_fpn_i in enumerate(x_fpn_all):
                        img_result = []
                        w_ratio = float(x_fpn_i.shape[3]) / img_pad_w
                        h_ratio = float(x_fpn_i.shape[2]) / img_pad_h
                        for box_id in range(gt_bboxes.shape[0]):
                            box = gt_bboxes[box_id, :].astype(np.float32)
                            box[0::2] *= w_ratio
                            box[1::2] *= h_ratio
                            x1_, y1_, x2_, y2_ = box.astype(np.int)
                            roi_feat = x_fpn_i[:, :, y1_:y2_+1, x1_:x2_+1]
                            roi_feat = pool_operation(roi_feat).squeeze().cpu().numpy()
                            img_result.append(roi_feat.reshape(1, -1))
                        img_result = np.concatenate(img_result, axis=0)
                        results[model_id][layer_id].append(img_result)

            if rank == 0:
                batch_size = data['img'][0].size(0)
                for _ in range(batch_size * world_size):
                    prog_bar.update()

        dist.barrier()
        # collect results of each model
        for idx, _ in enumerate(results):
            for layer, _ in enumerate(results[idx]):
                results[idx][layer] = self._collect_results_cpu(results[idx][layer], len(dataset),
                                                     'collect_tmp_model{}'.format(layer),
                                                     save_path=save_path)
                dist.barrier()
        return results

    def get_features_from_detector(self,
                                   pseudolabel_path,
                                   models,
                                   model_cfgs,
                                   cls_names,
                                   conf_thre=0.95,
                                   min_size=(30, 30),
                                   max_samples=1000,
                                   img_prefix=None,
                                   pgm_dataset={},
                                   save_path='./',
                                   max_num_prototype=10):
        '''
        get features by using detection models.
        '''
        assert len(cls_names) == 1, 'only 1 class per time to generate features, got {}'.format(len(cls_names))

        rank, world_size = get_dist_info()
        if rank == 0:
            if not os.path.exists(os.path.join(save_path,
                               self._prototype_proposal_json_name.format(cls_names[0]))):
                if isinstance(pseudolabel_path, str):
                    with open(pseudolabel_path, 'r') as file:
                        predictions = json.load(file)
                elif isinstance(pseudolabel_path, dict):
                    pass
                else:
                    raise TypeError('Unrecognized prediction type :{}, only '
                                    'Pathlike or dict type are supported'.format(type(pseudolabel_path)))

                valid_predictions = self._filter_predictions(predictions, conf_thre, min_size, cls_names)
                if len(valid_predictions) == 0:
                    print('[Warning] Valid predictions is empty, please decrease the conf_thre')
                if len(valid_predictions) >= world_size:
                    if len(valid_predictions) > max_samples:
                        # filter valid images
                        keys = np.array(list(valid_predictions.keys()))
                        used_idx = np.arange(len(keys))
                        np.random.shuffle(used_idx)
                        used_idx = used_idx[:max_samples]
                        valid_predictions = {key: valid_predictions[key]
                                             for key in keys[used_idx]}
                    # save valid image in self._prototype_proposal_json_name
                    with open(os.path.join(save_path,
                                           self._prototype_proposal_json_name.format(cls_names[0])), 'w') as file:
                        json.dump(valid_predictions, file)
        dist.barrier()
        # failed to generate valid file, return empty list
        # occasionally occurs in the unlabeled dataset
        if not os.path.exists(os.path.join(save_path, self._prototype_proposal_json_name.format(cls_names[0]))):
            return []

        dataset_cfg = copy.deepcopy(pgm_dataset)
        dataset_cfg.pipeline = model_cfgs[0].data.test.pipeline
        dataset_cfg.ann_file = os.path.join(save_path, self._prototype_proposal_json_name.format(cls_names[0]))
        dataset = build_dataset(dataset_cfg)
        if len(dataset) == 0:
            return []

        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=1,
            dist=True,
            shuffle=False)

        feats = self._fetch_roi_features(models, data_loader, save_path)
        return feats

    def _norm_feats(self, feats):
        '''
        normalize features.
        '''
        norm_feat_factor = np.sqrt(np.power(feats, 2).sum(1))
        norm_feat_factor = norm_feat_factor.reshape(-1, 1)
        norm_feats = torch.from_numpy(feats / norm_feat_factor).half().cuda()
        return norm_feats

    def _generate_prototypes_single(self, ma_cls_feats, mb_cls_feats, cls_name,
                                    max_num_prototypes=20, alpha=10):
        '''
        generate prototypes by using matching persuit.
        '''

        assert ma_cls_feats.shape[0] == mb_cls_feats.shape[0], 'but found {}!={}'.format(ma_cls_feats.shape[0],
                                                                                         mb_cls_feats.shape[0])

        # feature normalization
        ma_cls_feats_normed = self._norm_feats(ma_cls_feats)
        mb_cls_feats_normed = self._norm_feats(mb_cls_feats)

        # filter according to density
        dist_mat_a = 1 - torch.mm(ma_cls_feats_normed, ma_cls_feats_normed.T)
        dist_mat_b = 1 - torch.mm(mb_cls_feats_normed, mb_cls_feats_normed.T)

        density_a = torch.exp(-1 * torch.pow(dist_mat_a / 0.1, 2)).sum(axis=1)
        density_b = torch.exp(-1 * torch.pow(dist_mat_b / 0.1, 2)).sum(axis=1)

        middle = int(len(density_a) * 0.4)
        density_a_lb = density_a.sort().values[middle]
        density_b_lb = density_b.sort().values[middle]

        proposal_gamma = torch.where((density_a > density_a_lb) & (density_b > density_b_lb))[0]

        ma_cls_feats_normed = ma_cls_feats_normed.T  # transpose to Da x N
        mb_cls_feats_normed = mb_cls_feats_normed.T  # transpose to Db x N
        proposal_feats_a = ma_cls_feats_normed[:, proposal_gamma]
        proposal_feats_b = mb_cls_feats_normed[:, proposal_gamma]

        K = proposal_feats_a.shape[1]
        N = ma_cls_feats_normed.shape[1]
        Da = ma_cls_feats_normed.shape[0]
        Db = mb_cls_feats_normed.shape[0]

        prototype_indexes = []
        cnt = 0
        residual_a = ma_cls_feats_normed.clone()  # Da x N, N samples in Da dimension
        residual_b = mb_cls_feats_normed.clone()  # Db x N, N samples in Db dimension

        if type(max_num_prototypes) == list:
            max_num_prototypes = 10

        while cnt < max_num_prototypes:
            inner_a = torch.mm(proposal_feats_a.T, residual_a) # (K, Da) * (Da x N) -> (K, N)
            inner_b = torch.mm(proposal_feats_b.T, residual_b) # (K, Da) * (Da x N) -> (K, N)

            w_a = (alpha * (inner_a + inner_b) + inner_a) / (1 + 2 * alpha) # (K, N)
            w_b = (alpha * (inner_a + inner_b) + inner_b) / (1 + 2 * alpha) # (K, N)

            res_a = proposal_feats_a.T.view(K, Da, 1) * w_a.view(K, 1, N) # (K, Da, N)
            res_b = proposal_feats_b.T.view(K, Db, 1) * w_b.view(K, 1, N) # (K, Db, N)

            err_a = res_a - residual_a.view(1, Da, N) # (K, Da, N)
            err_b = res_b - residual_b.view(1, Db, N) # (K, Db, N)
            err_w = w_a - w_b # (K, N)
            err_sum = (err_a ** 2).sum(-1).sum(-1) + (err_b ** 2).sum(-1).sum(-1) + alpha * (err_w ** 2).sum(-1) # (K, )
            gamma = torch.argmin(err_sum.float())
            prototype_indexes.append(proposal_gamma[gamma])

            residual_a = residual_a - torch.mm(proposal_feats_a[:, gamma].view(-1, 1), w_a[gamma, :].view(1, -1))
            residual_b = residual_b - torch.mm(proposal_feats_b[:, gamma].view(-1, 1), w_b[gamma, :].view(1, -1))

            cnt += 1

        prototype_indexes = torch.tensor(prototype_indexes).cpu().numpy()
        prototype_feats_a = ma_cls_feats[prototype_indexes, :]
        prototype_feats_b = mb_cls_feats[prototype_indexes, :]

        return prototype_feats_a, prototype_feats_b, prototype_indexes

    @master_only
    def visualize_prototypes(self,
                            prototype_ids,
                            img_prefix='./',
                            predictions=None,
                            save_path=None,
                            tmp_valid_path=None,
                            neck_i=0):
        '''
        visualize the prototypes.
        '''
        mmcv.mkdir_or_exist(save_path)
        cls_names = list(prototype_ids.keys())
        decryptor = None

        for cls_name in cls_names:
            if predictions is None:
                with open(os.path.join(tmp_valid_path,
                                       self._prototype_proposal_json_name.format(cls_name)), 'r') as file:
                    predictions = json.load(file)
            elif isinstance(predictions, str):
                with open(predictions, 'r')  as file:
                    predictions = json.load(file)
            elif isinstance(predictions, dict):
                pass
            else:
                raise TypeError('Unrecognized data format : {}'.format(type(predictions)))
            filenames = list(predictions.keys())

            mmcv.mkdir_or_exist(os.path.join(save_path, 'prototype',
                                             cls_name, str(neck_i), 'sub'))
            mmcv.mkdir_or_exist(os.path.join(save_path, 'prototype',
                                             cls_name, str(neck_i), 'all'))
            cls_prototypes_ids = prototype_ids[cls_name]
            if len(cls_prototypes_ids) == 0:
                continue
            for (img_id, box_id) in cls_prototypes_ids:
                key = filenames[img_id]
                content_ann = predictions[key]['content_ann']
                box = np.array(content_ann['bboxes'][box_id], dtype=np.int)
                label = str(content_ann['labels'][box_id][0])
                conf = float(content_ann['confs'][box_id]) if 'confs' in \
                                                              content_ann else -1
                img_path = os.path.join(img_prefix, key)
                img = cv2.imread(img_path)

                _, filename = os.path.split(key)
                filename, _ = os.path.splitext(filename)

                # save the image of the prototype area 
                sub_img = img[box[1]: box[3], box[0]:box[2], :]
                out_path = os.path.join(save_path, 'prototype', cls_name, str(neck_i),
                                        'sub', cls_name + '_' + filename +
                                        '_%d_%.3f'%(box_id, conf) + '.jpg')
                cv2.imwrite(out_path, sub_img)

                # save the full image, circle the prototype area
                cv2.rectangle(img, tuple(box[0:2]), tuple(box[2:4]),
                              (255, 0, 0), 2)
                cv2.putText(img, "%s:%.3f" % (label, conf), tuple(box[0:2]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
                out_path = os.path.join(save_path, 'prototype', cls_name, str(neck_i),
                                        'all', cls_name + '_' + filename +
                                        '_%d_%.3f'%(box_id, conf) + '.jpg')
                cv2.imwrite(out_path, img)

    def generate_prototypes(self, 
                           ma_feats,
                           mb_feats,
                           predictions=None,
                           cls_names=('person', ),
                           max_num_prototypes=20, 
                           save_path='./',
                           conf_thre=0.9,
                           alpha=10):
        '''
        generate prototypes.
        '''
        rank, _ = get_dist_info()
        assert len(cls_names) == 1, 'process one class per round, found {}'.format(len(cls_names))
        assert len(ma_feats) == len(mb_feats), "symetric feats len must be the same, but found {}!={}". \
                                    format(len(ma_feats), len(mb_feats))
        if rank != 0:
            return None, None, None

        if predictions is None:
            prediction_path = os.path.join(save_path,
                                           self._prototype_proposal_json_name.format(cls_names[0]))
            with open(prediction_path, 'r') as file:
                predictions = json.load(file)
        elif isinstance(predictions, str):
            with open(predictions, 'r') as file:
                predictions = json.load(file)
        elif isinstance(predictions, dict):
            pass
        else:
            raise TypeError('Unrecognized type of predictions, found {}, which '
                            'should be one on [None, Pathlike, dict]'.format(type(predictions)))
        prediction_list = []
        for key, value in predictions.items():
            value.update({'filename': key})
            prediction_list.append(value)
        cls_prototype_ids = dict()

        for cls_index, cls_name in enumerate(cls_names):
            ma_cls_feats = []
            mb_cls_feats = []
            mp_cls_confs = []
            id_inv_mapping = []
            img_id = 0
            assert len(ma_feats) == len(mb_feats), "but found {}!={}".format(
                len(ma_feats), len(mb_feats))
            assert len(ma_feats) == len(prediction_list), \
                "but found {}!={}".format(len(ma_feats), len(prediction_list))

            for ma_img_feats, mb_img_feats, img_predictions in \
                    zip(ma_feats, mb_feats, prediction_list):
                labels = img_predictions['content_ann']['labels']
                labels = np.array([label[0] for label in labels])
                if conf_thre == 0:
                    confs = np.ones(len(labels)).astype(np.float32)
                else:
                    confs = np.array(img_predictions['content_ann']['confs']). \
                        astype(np.float32)
                cls_ids = np.where(labels==cls_name)[0]
                if len(cls_ids) == 0:
                    img_id += 1
                    continue
                ma_cls_feats.append(ma_img_feats[cls_ids, :])

                mb_cls_feats.append(mb_img_feats[cls_ids, :])
                mp_cls_confs.append(confs[cls_ids])
                for idx in cls_ids:
                    id_inv_mapping.append((img_id, idx))
                img_id += 1
            ma_cls_feats = np.concatenate(ma_cls_feats, axis=0)
            mb_cls_feats = np.concatenate(mb_cls_feats, axis=0)
            mp_cls_confs = np.concatenate(mp_cls_confs, axis=0)

            # generate prototypes for each class
            prototype_feats_stu, prototype_feats_tea, cls_prototype_idx = \
                 self._generate_prototypes_single(ma_cls_feats, 
                                                  mb_cls_feats, 
                                                  cls_name=cls_name,
                                                  max_num_prototypes=max_num_prototypes,
                                                  alpha=alpha)

            if len(prototype_feats_stu) > 0:
                prototype_image_names = []
                prototype_bboxes = []
                for idx in cls_prototype_idx:
                    img_idx, box_idx = id_inv_mapping[idx]
                    prototype_image_names.append(prediction_list[img_idx]['filename'])
                    prototype_bboxes.append(prediction_list[img_idx]['content_ann']['bboxes'][box_idx])
                cls_prototype_ids[cls_name] = [id_inv_mapping[idx]
                                              for idx in cls_prototype_idx]

        return prototype_feats_stu, prototype_feats_tea, cls_prototype_ids


    @master_only
    def _save(self, data, file):
        mmcv.dump(data, file)

    def _load(self, file):
        return mmcv.load(file)

    def __call__(self, model_cfg_paths, checkpoints, pseudolabel_path, cls_names, conf_thr, 
                 img_prefix, max_samples, max_num_prototypes, alpha=10, vis_prototype=True, 
                 save_path='./', pgm_dataset={}, alter_path='./'):
        rank, world_size = get_dist_info()
        model_cfgs = [mmcv.Config.fromfile(cfg_path) for cfg_path in model_cfg_paths]

        # save_path is like '/*/*/*/epoch_x' or '/*/*/*/epoch_xx'
        root_path = save_path[:-8]
        tea_fea_path = os.path.join(root_path, 'teacher_feature')
        tmp_valid_path = os.path.join(root_path, 'tmp_valid')
        mmcv.mkdir_or_exist(tea_fea_path)
        mmcv.mkdir_or_exist(tmp_valid_path)

        # build models
        models = [build_detector(cfg.model, train_cfg=None,
                                 test_cfg=cfg.test_cfg)
                  for cfg in model_cfgs]
        for model_id, _ in enumerate(models):
            load_checkpoint(models[model_id], checkpoints[model_id],
                            map_location='cpu')
            models[model_id] = MMDistributedDataParallel(
                models[model_id].cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)

        prototype_feats_all_neck_stu = []
        prototype_feats_all_neck_tea = []
        cls_start_index = [0]
        cls_id = 0
        dist.barrier()

        for cls_name, max_sample, max_num_prototype in zip(cls_names, max_samples, max_num_prototypes):
            if rank == 0:
                print('Get features from detectors for ', cls_name, ', class id:', cls_id)
            if not os.path.exists(os.path.join(tea_fea_path, 'tea_det_feats_{}.pkl'.format(cls_name))):
                # generate teacher features before the 1st epoch
                det_feats = self.get_features_from_detector(
                    pseudolabel_path, models, model_cfgs, [cls_name], conf_thre=conf_thr,
                    img_prefix=img_prefix, max_samples=max_sample, pgm_dataset=pgm_dataset, 
                    save_path=tmp_valid_path)

                if len(det_feats) == 0:
                    # no valid sample, will only appear in the unlabeled dataset
                    # load features and index from the alternative path
                    alter_stu = self._load(os.path.join(alter_path, 'prototype_feats_allneck_allcls_student.json'))
                    alter_tea = self._load(os.path.join(alter_path, 'prototype_feats_allneck_allcls_teacher.json'))
                    alter_index = self._load(os.path.join(alter_path, 'prototype_feats_allneck_allcls_index.json'))
                    for neck_i in range(len(prototype_feats_all_neck_stu)):
                        # load the corresponding features and index 
                        if rank == 0:
                            alter_start = alter_index[neck_i][cls_id]
                            alter_end = alter_index[neck_i][cls_id + 1]
                            index_start = cls_start_index[neck_i][cls_id]
                            prototype_feats_all_neck_stu[neck_i].extend(alter_stu[neck_i][alter_start : alter_end])
                            prototype_feats_all_neck_tea[neck_i].extend(alter_tea[neck_i][alter_start : alter_end])
                            index_end = index_start + alter_end - alter_start
                            cls_start_index[neck_i].append(index_end)
                    dist.barrier()
                    cls_id += 1
                    continue
                det_feats_tea, det_feats_stu = det_feats
                if rank == 0:
                    self._save(det_feats_tea, os.path.join(tea_fea_path, 'tea_det_feats_{}.pkl'.format(cls_name)))
            else:
                det_feats = self.get_features_from_detector(
                    pseudolabel_path, [models[1]], model_cfgs, [cls_name], conf_thre=conf_thr,
                    img_prefix=img_prefix, max_samples=max_sample, pgm_dataset=pgm_dataset, 
                    save_path=tmp_valid_path)
                if len(det_feats) == 0:
                    # no valid sample, will only appear in the unlabeled dataset
                    # load features and index from the alternative path
                    alter_stu = self._load(os.path.join(alter_path, 'prototype_feats_allneck_allcls_student.json'))
                    alter_tea = self._load(os.path.join(alter_path, 'prototype_feats_allneck_allcls_teacher.json'))
                    alter_index = self._load(os.path.join(alter_path, 'prototype_feats_allneck_allcls_index.json'))
                    for neck_i in range(len(prototype_feats_all_neck_stu)):
                        # load the corresponding features
                        if rank == 0:
                            alter_start = alter_index[neck_i][cls_id]
                            alter_end = alter_index[neck_i][cls_id + 1]
                            index_start = cls_start_index[neck_i][cls_id]
                            prototype_feats_all_neck_stu[neck_i].extend(alter_stu[neck_i][alter_start : alter_end])
                            prototype_feats_all_neck_tea[neck_i].extend(alter_tea[neck_i][alter_start : alter_end])
                            index_end = index_start + alter_end - alter_start
                            cls_start_index[neck_i].append(index_end)
                    dist.barrier()
                    cls_id += 1
                    continue
                det_feats_stu = det_feats[0]
                if rank == 0:
                    det_feats_tea = self._load(os.path.join(tea_fea_path, 'tea_det_feats_{}.pkl'.format(cls_name)))
            dist.barrier()

            # generate prototypes
            if rank == 0:
                self._print('\nStarting generating prototypes for class: {}'.format(cls_name))
            for neck_i in range(len(det_feats_stu)):
                if rank == 0:
                    if len(prototype_feats_all_neck_stu) == 0:
                        prototype_feats_all_neck_stu = [[] for _ in range(len(det_feats_stu))]
                        prototype_feats_all_neck_tea = [[] for _ in range(len(det_feats_stu))]
                        cls_start_index = [[0] for _ in range(len(det_feats_stu))]
                    index_start = cls_start_index[neck_i][cls_id]
                    prototype_feats_stu, prototype_feats_tea, prototype_ids = \
                                                   self.generate_prototypes(det_feats_stu[neck_i], 
                                                                           det_feats_tea[neck_i],
                                                                           max_num_prototypes=max_num_prototype,
                                                                           cls_names=[cls_name],
                                                                           save_path=tmp_valid_path,
                                                                           conf_thre=conf_thr,
                                                                           alpha=alpha)
                    # will only appear in the unlabeled dataset
                    # load prototypes from the alternative path
                    if len(prototype_feats_stu) == 0:
                        alter_stu = self._load(os.path.join(alter_path, 'prototype_feats_allneck_allcls_student.json'))
                        alter_tea = self._load(os.path.join(alter_path, 'prototype_feats_allneck_allcls_teacher.json'))
                        alter_index = self._load(os.path.join(alter_path, 'prototype_feats_allneck_allcls_index.json'))
                        alter_start = alter_index[neck_i][cls_id]
                        alter_end = alter_index[neck_i][cls_id + 1]
                        prototype_feats_all_neck_stu[neck_i].extend(alter_stu[neck_i][alter_start : alter_end])
                        prototype_feats_all_neck_tea[neck_i].extend(alter_tea[neck_i][alter_start : alter_end])
                        index_end = index_start + alter_end - alter_start
                    else:
                        prototype_feats_all_neck_stu[neck_i].extend(prototype_feats_stu.tolist())
                        prototype_feats_all_neck_tea[neck_i].extend(prototype_feats_tea.tolist())
                        index_end = index_start + len(prototype_feats_stu)

                        if vis_prototype:
                            # visualize prototypes
                            img_prefix = pgm_dataset.img_prefix
                            self.visualize_prototypes(prototype_ids,
                                                     img_prefix=img_prefix,
                                                     save_path=save_path,
                                                     tmp_valid_path=tmp_valid_path,
                                                     neck_i=neck_i)
                    cls_start_index[neck_i].append(index_end)
                else:
                    prototype_feats_stu, prototype_feats_tea, prototype_ids = None, None, None
                dist.barrier()
            if rank == 0:
                self._print('Generating prototypes for class: {} Done!\n'.format(cls_name))
            cls_id += 1

        # save features of prototypes
        if rank == 0:
            self._save(prototype_feats_all_neck_stu, os.path.join(save_path, 'prototype_feats_allneck_allcls_student.json'))
            self._save(prototype_feats_all_neck_tea, os.path.join(save_path, 'prototype_feats_allneck_allcls_teacher.json'))
            self._save(cls_start_index, os.path.join(save_path, 'prototype_feats_allneck_allcls_index.json'))
        dist.barrier()        
        return

def parse_args():
    '''
    parse arguments.
    '''

    parser = argparse.ArgumentParser(description='Prototype Generation Module')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='automatically set by pytorch, never config it '
                             'by hand')
    args_ = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args_.local_rank)

    return args_
