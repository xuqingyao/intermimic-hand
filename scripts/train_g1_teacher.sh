#!/bin/bash

CUDA_DEV=$1
SUB_NAME=$2

: "${CUDA_DEV:=0}"     
: "${SUB_NAME:=sub0}"  

echo "Using CUDA device: $CUDA_DEV"
echo "Using sub name: $SUB_NAME"

CUDA_VISIBLE_DEVICES=$CUDA_DEV python intermimic/run.py \
    --sub $SUB_NAME \
    --task InterMimicG1 \
    --cfg_env intermimic/data/cfg/omomo_g1_29dof_with_hand.yaml \
    --cfg_train intermimic/data/cfg/train/rlg/omomo_g1_29dof_with_hand.yaml \
    --headless \
    --output checkpoints
