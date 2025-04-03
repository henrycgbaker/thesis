#!/usr/bin/env python
import subprocess
import os
import json
import tempfile

# Ensure the current directory (project root) is in sys.path.
os.chdir(os.path.abspath(os.path.dirname(__file__)))

from configs.b_scenario_configs import (
    scenario_a_max_throughput_exploit,
    scenario_b_precision_gaming,
    scenario_c_gpu_overdrive,
    scenario_d_standard_production,
    scenario_e_low_latency_real_time,
    scenario_f_balanced_performance_mode
)

# List of configuration dictionaries.
scenarios_list = [
    scenario_a_max_throughput_exploit,
    scenario_b_precision_gaming,
    scenario_c_gpu_overdrive,
    scenario_d_standard_production,
    scenario_e_low_latency_real_time,
    scenario_f_balanced_performance_mode,
]

# Loop over each configuration.
for config in scenarios_list:
    # Write config to a temporary JSON file.
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as temp_config_file:
        json.dump(config, temp_config_file, indent=2)
        temp_config_path = temp_config_file.name

    print(f"Launching experiment with config: {config['config_name']}")

    # Launch the single experiment script using 'accelerate launch'
    result = subprocess.run([
        "accelerate", "launch", "run_single_experiment.py",
        "--config", temp_config_path
    ], check=True)

    print(f"Experiment with {config['config_name']} completed.\n")

print("All experiments have been executed sequentially.")
