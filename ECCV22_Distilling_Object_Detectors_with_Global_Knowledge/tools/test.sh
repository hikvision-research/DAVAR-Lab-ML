#!/usr/bin/env bash

# current path
CURRENT_FOLDER=$PWD

GPUS=2
PORT=${PORT:-29501}

CONFIG=${CURRENT_FOLDER}/configs/faster_rcnn_coco_50.py
CHECKPOINT=${CURRENT_FOLDER}/epoch_24.pth

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $CURRENT_FOLDER/tools/test.py $CONFIG $CHECKPOINT \
    --out $FILE/test_result.pkl \
    --eval bbox \
    --launcher pytorch ${@:4}