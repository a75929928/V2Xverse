import re

''' echo command first, then match patterns using re '''
bash = "--scenarios=simulation/leaderboard/data/scenarios/town05_all_scenarios_2.json  --scenario_parameter=simulation/leaderboard/leaderboard/scenarios/scenario_parameter_.yaml  --routes=simulation/leaderboard/data/evaluation_routes/town05_short_r0.xml --repetitions=1 --track=SENSORS --checkpoint=results/results_driving_debug/v2x_final/town05_short_collab/r0_repeat_0/results.json --agent=simulation/leaderboard/team_code/pnp_agent_e2e.py --agent-config=simulation/leaderboard/team_code/agent_config/pnp_config_.yaml --debug=0 --record= --resume=0 --port=40000 --trafficManagerPort=40005 --carlaProviderSeed=2000 --trafficManagerSeed=2000 --ego-num=1 --timeout 600 --skip_existed=0"
# bash = "python simulation/leaderboard/leaderboard/leaderboard_evaluator_parameter.py --scenarios=simulation/leaderboard/data/scenarios/town05_all_scenarios_2.json  --scenario_parameter=simulation/leaderboard/leaderboard/scenarios/scenario_parameter_.yaml  --routes=simulation/leaderboard/data/evaluation_routes/town05_short_r0.xml --repetitions=1 --track=SENSORS --checkpoint=results/results_driving_debug/v2x_final/town05_short_collab/r0_repeat_0/results.json --agent=simulation/leaderboard/team_code/pnp_agent_e2e.py --agent-config=simulation/leaderboard/team_code/agent_config/pnp_config_.yaml --debug=0 --record= --resume=0 --port=40000 --trafficManagerPort=40005 --carlaProviderSeed=2000 --trafficManagerSeed=2000 --ego-num=1 --timeout 600 --skip_existed=0"

parts = bash.split()
results = [f'"{part}"' if part.startswith('--') else part for part in parts if part]
# results = [[f'["--{part}"]' if not part.startswith('--') else part] for part in parts if part]
# print(results)
for result in results:
    formatted_result = str(result) + ","
    print(formatted_result)