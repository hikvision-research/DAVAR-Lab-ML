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
# Abstract: the interface for building the dataloader.
"""
import platform
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader

from mmdet.datasets.samplers import DistributedGroupSampler, DistributedSampler, GroupSampler
from ..builder import build_sampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     **kwargs):
    """
    Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        imgs_per_gpu (int): Number of images on each GPU, i.e., batch size of
            each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        sampler = kwargs.pop('sampler', None)
        if shuffle:
            if sampler is None:
                sampler = DistributedGroupSampler(dataset, imgs_per_gpu, world_size, rank)
            else:
                sampler['dataset'] = dataset
                sampler['samples_per_gpu'] = imgs_per_gpu
                sampler['num_replicas'] = world_size
                sampler['rank'] = rank
                sampler = build_sampler(sampler)
        else:
            if sampler is None:
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False)
            else:
                sampler['dataset'] = dataset
                sampler['samples_per_gpu'] = imgs_per_gpu
                sampler['num_replicas'] = world_size
                sampler['rank'] = rank
                sampler = build_sampler(sampler)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = kwargs.pop('sampler', None)
        if shuffle:
            if sampler is None:
                sampler = GroupSampler(dataset, imgs_per_gpu)
            else:
                sampler['dataset'] = dataset
                sampler['samples_per_gpu'] = imgs_per_gpu
                sampler = build_sampler(sampler)
        else:
            sampler = None
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
