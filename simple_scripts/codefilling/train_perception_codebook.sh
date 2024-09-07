#!/bin/bash

# ===== Set Parameters =====
CONFIG_FILE=opencood/hypes_yaml/v2xverse/where2comm_codebook.yaml
# could also start with checkpoint
# model_dir=
# ===== Check GPU =====

# ===== Activate Env and Run =====
cd /home/hjh/carla/Gym/V2Xverse
source /home/hjh/miniconda3/bin/activate v2xverse

python opencood/tools/train.py -y=${CONFIG_FILE}
# python opencood/tools/train.py -y=${CONFIG_FILE} --model_dir=$model_dir