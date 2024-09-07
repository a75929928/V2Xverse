#!/bin/bash

# ===== Set Parameters =====
export CUDA_VISIBLE_DEVICES="0"
export nproc_per_node=1
export perception_model_dir=
export config=codebook
# config-file=./codriving/hypes_yaml/codriving/end2end_${4:-codriving}.yaml
export resume=None
export log=log

# ===== Activate Env and Run =====
cd /home/hjh/carla/Gym/V2Xverse
source /home/hjh/miniconda3/bin/activate v2xverse
# Evaluation upon different scenarios on each routes
## CUDA_VISIBLE_DEVICES=0 ${CARLA_ROOT}/CarlaUE4.sh --world-port=2000 -prefer-nvidia
# CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=${Carla_port} -prefer-nvidia

bash scripts/train_planner_e2e.sh $CUDA_VISIBLE_DEVICES $nproc_per_node $perception_model_dir ${config} $resume ${log}