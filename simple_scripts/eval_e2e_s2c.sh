#!/bin/bash

# ===== Initialization =====
# read -p "Default evaluation config for select2col? (default '1'): " Flag
# Flag=${Flag:-"1"}

# if [ "${Flag}" -ne 1 ]; then
#     read -p "Enter method (default 'select2col'): " Method_tag
#     # Config files named pnp_config_{codriving_5_10}
#     # simulation/leaderboard/team_code/agent_config
#     read -p "Enter Agent_config (default 'select2col_5_10'): " Agent_config
#     # Route files named town05_short_r{0~31}
#     # simulation/leaderboard/data/evaluation_routes
#     read -p "Enter testing routes (default '0'): " Route_ids
#     # Scenario files named scenario_parameter_${1~5}.yaml
#     # simulation/leaderboard/leaderboard/scenarios
#     read -p "Enter scenario config (default '1'): " Scenario_configs
#     read -p "Enter repetition for each route (default '0'): " Repeat_times
#     read -p "Enter Carla Port (default '2000'): " Carla_port
# fi

# Meant to use for long loop evaluation, here demonstrate single one
Method_tag=${Method_tag:-"select2col"}
Agent_config=${Agent_config:-"${Method_tag}_5_10"}

Route_id=${Route_id:-0}
Scenario_config=${Scenario_config:-1}
Repeat_id=${Repeat_id:-0}
Carla_port=${Carla_port:-2000}

# ===== Activate Env and Run =====
cd /home/hjh/carla/Gym/V2Xverse
source /home/hjh/miniconda3/bin/activate v2xverse
#  CUDA_VISIBLE_DEVICES=0 ${CARLA_ROOT}/CarlaUE4.sh --world-port=2000 -prefer-nvidia
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_driving_e2e.sh ${Route_id} ${Carla_port} ${Method_tag} ${Repeat_id} ${Agent_config} ${Scenario_config}
# CUDA_VISIBLE_DEVICES=0 bash scripts/eval_driving_select2col.sh 0 2000 select2col 0 select2col_5_10 1

# ===== Example Loop =====

# Method_tag=${Method_tag:-"select2col"}
# Agent_config=${Agent_config:-"${Method_tag}_5_10"}

# Route_ids=${Route_ids:-0}
# Scenario_configs=${Scenario_configs:-1}
# Repeat_ids=${Repeat_ids:-0}
# Carla_port=${Carla_port:-2000}

# Evaluation upon different scenarios on each routes
## CUDA_VISIBLE_DEVICES=0 ${CARLA_ROOT}/CarlaUE4.sh --world-port=2000 -prefer-nvidia
# CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=${Carla_port} -prefer-nvidia

# for ((Scenario_config=1; Scenario_config<=${Scenario_configs}; Scenario_config++ )); do
#     echo "Current Scenario_config: $Scenario_config"
#     for (( Route_id=0; Route_id<=${Route_ids}; Route_id++ )); do
#         for (( Repeat_id=0; Repeat_id<=${Repeat_ids}; Repeat_id++ )); do
#             echo "Current Route: $Route_id for $Repeat_id time"
#             CUDA_VISIBLE_DEVICES=0 bash scripts/eval_driving_e2e.sh ${Route_id} ${Carla_port} ${Method_tag} ${Repeat_id} ${Agent_config} ${Scenario_config}
#         done
#     done
# done