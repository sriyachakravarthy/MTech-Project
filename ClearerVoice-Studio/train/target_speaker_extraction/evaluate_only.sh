#!/bin/sh

#####
# Modify these lines
gpu_id=2													# Visible GPUs
n_gpu=1														# Number of GPU used, currently only support 1
checkpoint_dir=''

train_from_last_checkpoint=1
config_pth=${checkpoint_dir}/config.yaml

# call evaluation
export PYTHONWARNINGS="ignore"
CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=$n_gpu \
--master_port=$(date '+88%S') \
train.py \
--evaluate_only 1 \
--config $config_pth \
--checkpoint_dir $checkpoint_dir \
--train_from_last_checkpoint $train_from_last_checkpoint \
--evaluate_only 1 \
>>${checkpoint_dir}/evaluation.txt 2>&1


