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
# Abstract: the runner (inherated from the base runner) for knowledge distillation.
"""
import logging
import os.path as osp
import time
import collections

import mmcv
import torch

from mmcv.runner.hooks import CheckpointHook, IterTimerHook, HOOKS
from mmcv.runner.checkpoint import weights_to_cpu
from mmcv.runner import Runner
from mmcv.runner.dist_utils import master_only

class RunnerKD(Runner):
    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None,
                 kd_cfg=None,
                 ckpt_cfg=None):
        super(RunnerKD, self).__init__(model=model,
                                       batch_processor=batch_processor,
                                       work_dir=work_dir,
                                       logger=logger)

        if optimizer is not None:
            self.optimizer = dict()
            for _type in optimizer:
                _optimizer = optimizer[_type]
                self.optimizer[_type] = self.init_optimizer(_optimizer)
        else:
            self.optimizer = None

        self.kd_cfg = kd_cfg
        self.ckpt_cfg = ckpt_cfg
        self.prototype_each_epoch = self.kd_cfg.prototype_each_epoch


    def init_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            optimizer = mmcv.runner.obj_from_dict(optimizer, torch.optim,
                                      dict(params=self.model.parameters()))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        return optimizer

    def current_forward_branch(self):
        # branch only set to 'stu'
        branch = 'stu'
        return branch

    def current_lr(self):
        """
        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')

        group_ret = dict(stu=[], tea=[])
        for _type in self.optimizer:
            group_ret[_type] = [group['lr'] for group in
                                self.optimizer[_type].param_groups]
        return group_ret


    def train(self, data_loader, **kwargs):
        self.data_loader = data_loader
        self.mode = 'train'
        self.model.train()

        if self.kd_cfg.tea_mode == 'eval':
            self.model.module.tea.eval()

        self.call_hook('before_train_epoch')
        self.call_hook('before_epoch')  # IterTimerHook
        iters_per_epoch = len(data_loader)
        self._max_iters = self._max_epochs * iters_per_epoch

        use_pair_kd = self.kd_cfg.strategy.kd_loss.get('use_pair_kd', False)
        pair_wise = self.kd_cfg.strategy.kd_loss.get('pair_wise', False)

        use_pgm = False
        if use_pair_kd and pair_wise in ['use_pgm', 'robustness_for_point']:
            use_pgm = True
            epoch_now = int(self._epoch / self.prototype_each_epoch)

            pgm_path = osp.join(self.work_dir, 'pgm/epoch_' + str(epoch_now))
            proto_student = mmcv.load(osp.join(pgm_path, 'prototype_feats_allneck_allcls_student.json'))
            proto_teacher = mmcv.load(osp.join(pgm_path, 'prototype_feats_allneck_allcls_teacher.json'))
            cls_start_index = mmcv.load(osp.join(pgm_path, 'prototype_feats_allneck_allcls_index.json'))

            device_id = torch.cuda.current_device()
            proto_student = [torch.tensor(proto_student[i]).to(device_id) for i in range(len(proto_student))]
            proto_teacher = [torch.tensor(proto_teacher[i]).to(device_id) for i in range(len(proto_teacher))]
            cls_start_index = [torch.tensor(cls_start_index[i]).to(device_id) for i in range(len(cls_start_index))]

        data_iter = iter(data_loader)

        for i in range(iters_per_epoch):
            try:
                data = next(data_iter)
            except StopIteration:
                self.epoch += 1
                self.call_hook('before_train_epoch')
                data_iter = iter(data_loader)
                data = next(data_iter)

            img_v = dict(train_data=data['img'])
            img_meta_v = dict(train_data=data['img_metas'])
            gt_bboxes_v = dict(train_data=data['gt_bboxes'])
            gt_labels_v = dict(train_data=data['gt_labels'])

            self._inner_iter = i
            self.call_hook('before_train_iter')  # warmup
            self.call_hook('before_iter')  # IterTimerHook

            if use_pgm:
                data_batch = dict(img=img_v, img_metas=img_meta_v, gt_bboxes=gt_bboxes_v, gt_labels=gt_labels_v,
                                  runner=self, proto_st=proto_student, proto_tc=proto_teacher, 
                                  cls_start_index=cls_start_index)
            else:
                data_batch = dict(img=img_v, img_metas=img_meta_v, gt_bboxes=gt_bboxes_v,
                                  gt_labels=gt_labels_v, runner=self)

            outputs = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)

            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')

            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            if 'prob_vars' in outputs:
                self.log_buffer.update(outputs['prob_vars'], 1)

            self.outputs = outputs
            self.call_hook('after_train_iter')
            self.call_hook('after_iter')  # IterTimerHook
            self._iter += 1

        self.call_hook('after_train_epoch')  # evaluation
        self.save_checkpoint_only_stu(out_dir=self.work_dir)

        self._epoch += 1
        self.epoch += 1

    def resume(self, checkpoint, resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            load_optimizer = checkpoint['optimizer']
            for _type in load_optimizer:
                self.optimizer[_type].load_state_dict(load_optimizer[_type])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def register_lr_hook(self, lr_config):
        if lr_config is None:
            return
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            policy_type = lr_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of Lr updater.
            # Since this is not applicable for `
            # CosineAnnealingLrUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'LrUpdaterHook'
            lr_config['type'] = hook_type
            hook = mmcv.build_from_cfg(lr_config, HOOKS)
        else:
            hook = lr_config
        self.register_hook(hook)


    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """

        self.register_lr_hook(lr_config)
        self.register_optimizer_hook(optimizer_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config)


    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None

        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError('meta must be a dict or None, but got {}'.format(
                type(meta)))
        meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

        mmcv.mkdir_or_exist(osp.dirname(filepath))
        if hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(model.state_dict())
        }
        if optimizer is not None:
            if isinstance(optimizer, dict):
                checkpoint['optimizer'] = dict()
                for _type in optimizer:
                    checkpoint['optimizer'][_type] = optimizer[
                        _type].state_dict()
            else:  # regular optimizer
                checkpoint['optimizer'] = optimizer.state_dict()

        torch.save(checkpoint, filepath)

        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))


    @master_only
    def save_checkpoint_only_stu(self,
                                 out_dir,
                                 filename_tmpl='epoch_{}.pth',
                                 save_optimizer=True,
                                 meta=None,
                                 create_symlink=True):
        if self.ckpt_cfg is not None and (self.epoch + 1) % self.ckpt_cfg.interval != 0:
            return

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)

        mmcv.mkdir_or_exist(osp.dirname(filepath))

        if hasattr(self.model, 'module'):
            model = self.model.module
            model = model.stu
        else:
            model = self.model.stu

        meta = dict(meta_info=None)
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(model.state_dict())
        }
        print('Save checkpoint only stu!')
        torch.save(checkpoint, filepath)


    def call_hook(self, fn_name):
        """
        根据Hook注册的函数名调用hook
        Args:
            fn_name (str): hook函数名，如"after_train_epoch"
        """
        for hook in self._hooks:
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self)

