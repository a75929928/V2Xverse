# Generate batch evaluation commands for different routes and run them in parallel.
''' Example output(should be run in different terminals):
CUDA_VISIBLE_DEVICES=0 ${CARLA_ROOT}/CarlaUE4.sh --world-port=2000 -prefer-nvidia
bash simple_scripts/batch_eval_e2e_s2c.sh 0 7 2000
CUDA_VISIBLE_DEVICES=0 ${CARLA_ROOT}/CarlaUE4.sh --world-port=2008 -prefer-nvidia
bash simple_scripts/batch_eval_e2e_s2c.sh 8 15 2008
CUDA_VISIBLE_DEVICES=0 ${CARLA_ROOT}/CarlaUE4.sh --world-port=2016 -prefer-nvidia
bash simple_scripts/batch_eval_e2e_s2c.sh 16 23 2016
CUDA_VISIBLE_DEVICES=0 ${CARLA_ROOT}/CarlaUE4.sh --world-port=2024 -prefer-nvidia
bash simple_scripts/batch_eval_e2e_s2c.sh 24 31 2024
'''

for route_start in (0, 8, 16, 24):
    print("CUDA_VISIBLE_DEVICES=0 ${CARLA_ROOT}/CarlaUE4.sh --world-port="+ f"{2000+route_start} -prefer-nvidia")
    print(f"bash simple_scripts/batch_eval_e2e_s2c.sh {route_start} {route_start+7} {2000+route_start}")