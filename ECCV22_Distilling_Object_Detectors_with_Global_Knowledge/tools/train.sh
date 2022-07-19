#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH

/usr/local/miniconda3/bin/python -m pip install --upgrade pip

# current path
CURRENT_FOLDER=$PWD

############ install mmcv and mmdet ############
PIP=${PIP:-"pip"}
$PIP install mmcv-full==1.3.4
$PIP install addict cython numpy albumentations==0.3.2 imagecorruptions matplotlib Pillow==6.2.2 six terminaltables pytest pytest-cov pytest-runner mmlvis scipy sklearn mmpycocotools yapf

MMDET_PATH=$CURRENT_FOLDER/ObjectPerceptron/mmdetection/
MMCV_PATH=$CURRENT_FOLDER/ObjectPerceptron/mmcv2/

cd $MMCV_PATH
python setup.py install

cd $MMDET_PATH
python setup.py install

# gpu number
GPUS=8

# Distributed training needs a master port
MASTER_PORT=12917

work_dir=$CURRENT_FOLDER/work_dir
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    $CURRENT_FOLDER/tools/train.py \
    $CURRENT_FOLDER/configs/faster_rcnn_coco_kd_config.py \
    --gpus=$GPUS \
    --launcher='pytorch' \
    --work_dir=$work_dir

