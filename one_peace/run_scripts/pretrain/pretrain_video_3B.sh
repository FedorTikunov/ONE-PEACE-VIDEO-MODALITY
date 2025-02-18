#!/usr/bin/env bash

export MASTER_PORT=6082
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8

config_dir=.
config_name=pretrain_video_3B
save_dir=./checkpoints/one_peace_video
restore_file=./one_peace.pt

torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} train.py \
    --config-dir=${config_dir} \
    --config-name=${config_name} \
    checkpoint.save_dir=${save_dir} \
    checkpoint.restore_file=${restore_file}