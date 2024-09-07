#!/bin/bash

# Default run all scenarios on all routes at once 
# ===== Initialization =====

# $1, from which route to start
# $2, end with which route
# $3, Carla port

Method_tag=${Method_tag:-"select2col"}
Agent_config=${Agent_config:-"${Method_tag}_5_10"}

Route_id_start=${Route_id_start:-0}
Route_id_end=${Route_id_end:-15}
# Route_id_start=${Route_id_start:-0}
# Route_id_end=${Route_id_end:-31}

Scenario_config_start=${Scenario_config_start:-1}
Scenario_config_end=${Scenario_config_end:-5}

Carla_port=${Carla_port:-2000}
# Carla_port=${Carla_port:-2000}

Repeat_id_end=${Repeat_id_end:-0}

cd /home/hjh/carla/Gym/V2Xverse
source /home/hjh/miniconda3/bin/activate v2xverse
# Evaluation upon different scenarios on each routes
for ((Scenario_config=$Scenario_config_start; Scenario_config<=$Scenario_config_end; Scenario_config++ )); do
    echo "Current Scenario_config: $Scenario_config"
    for (( Route_id=$Route_id_start; Route_id<=$Route_id_end; Route_id++ )); do
        for (( Repeat_id=0; Repeat_id<=$Repeat_id_end; Repeat_id++ )); do
            echo "Current Route: $Route_id for $Repeat_id time"
            
            # KILL EXISTING CARLA PROCESS AS VIOLENT RESTART
            PID=$(lsof -t -i:$Carla_port)
                if [ -n "$PID" ]; then
                    kill -SIGTERM "$PID"
                fi
            tmux send-keys -t 1 "CUDA_VISIBLE_DEVICES=0 ${CARLA_ROOT}/CarlaUE4.sh --world-port=$Carla_port -prefer-nvidia" C-m 
            sleep 6

            # additionally add tee to record simulation error
            CUDA_VISIBLE_DEVICES=0 bash scripts/eval_driving_e2e.sh ${Route_id} ${Carla_port} ${Method_tag} ${Repeat_id} ${Agent_config} ${Scenario_config} | tee results/eval_e2e_s2c_${Carla_port}.log
        done
    done
done