#!/bin/bash

# ===== Set Parameters =====
CONFIG_FILE=opencood/hypes_yaml/v2xverse/codriving_multiclass_config.yaml
# CHECKPOINT_FOLDER=/home/hjh/carla/Gym/V2Xverse/opencood/logs/point_pillar_Select2Col_2024_08_22_17_07_34

# ===== Activate Env =====
cd /home/hjh/carla/Gym/V2Xverse
source /home/hjh/miniconda3/bin/activate v2xverse

# ===== Train =====
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 --use_env opencood/tools/train_ddp.py -y ${CONFIG_FILE} \
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
#     --nproc_per_node=2 --use_env opencood/tools/train_ddp.py -y ${CONFIG_FILE} \
    # [--model_dir ${CHECKPOINT_FOLDER}]