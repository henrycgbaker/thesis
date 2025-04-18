import copy
from configs.a_default_config import base_config
from configs.config_utils import update_nested_dict, update_multiple_config, generate_config_name_from_variation
import copy

def generate_controlled_configs(base_config, controlled_variations):
    configs = []
    for param, values in controlled_variations.items():
        for val in values:
            # Create a fresh copy of the base configuration
            cfg = copy.deepcopy(base_config)
            cfg = update_nested_dict(cfg, param, val)
            variation = {param: val}
            cfg["controlled_variation"] = variation
            cfg["suite"] = "controlled"
            cfg["config_name"] = generate_config_name_from_variation(variation)
            configs.append(cfg)
    return configs

# ---------------------------
# Generate Controlled Experiments
# ---------------------------

# (i) parallelisation Strategies:
parallelisation_variations = {
    "num_processes": [1, 2, 3, 4]
}
parallelisation_configs = generate_controlled_configs(base_config, parallelisation_variations)

# (ii) Batching Strategies:
batching_variations = {
    "batching_options.batch_size___fixed_batching": [1, 2, 4, 8, 16, 32, 64]
}
batching_configs = generate_controlled_configs(base_config, batching_variations)

# (iii) Precision & Quantisation Methods:
precision_quantisation_configs = []

# Variation 1: FP32, no quantisation.
updates = {
    "fp_precision": "float32",
    "quantization_config.quantization": False,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": False,
}
cfg1 = update_multiple_config(base_config, updates)
cfg1["controlled_variation"] = updates
cfg1["suite"] = "controlled"   
cfg1["config_name"] = generate_config_name_from_variation(updates)
precision_quantisation_configs.append(cfg1)

# Variation 2: FP16, no quantisation.
updates = {
    "fp_precision": "float16",
    "quantization_config.quantization": False,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": False,
}
cfg2 = update_multiple_config(base_config, updates)
cfg2["controlled_variation"] = updates
cfg2["suite"] = "controlled"    
cfg2["config_name"] = generate_config_name_from_variation(updates)
precision_quantisation_configs.append(cfg2)

# Variation 3: FP16 with 8-bit quantisation.
updates = {
    "fp_precision": "float16",
    "quantization_config.quantization": True,
    "quantization_config.load_in_8bit": True,
    "quantization_config.load_in_4bit": False,
}
cfg3 = update_multiple_config(base_config, updates)
cfg3["controlled_variation"] = updates
cfg3["suite"] = "controlled"    
cfg3["config_name"] = generate_config_name_from_variation(updates)
precision_quantisation_configs.append(cfg3)

# Variation 4: FP16 with 4-bit quantisation.
updates = {
    "fp_precision": "float16",
    "quantization_config.quantization": True,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": True,
}
cfg4 = update_multiple_config(base_config, updates)
cfg4["controlled_variation"] = updates
cfg4["suite"] = "controlled"    
cfg4["config_name"] = generate_config_name_from_variation(updates)
precision_quantisation_configs.append(cfg4)


# (iv) decoder mode Variations: ____________________________________________________
# List to hold all decoder configuration variations
decoder_mode_configs = []

# List of temperature values to test
temperature_variations = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 14]

# Values for top_k and top_p
top_k_values = [20, 50, 100, 200, 500]
top_p_values = [0.7, 0.8, 0.9, 0.98]

### Variation A: Greedy Decoding
# Greedy decoding always picks the highest probability token so only temperature is varied here.
for temp in temperature_variations:
    updates = {
        "decoder_config.decoding_mode": "greedy",
        "decoder_config.decoder_temperature": temp,
    }
    cfg = update_multiple_config(base_config, updates)
    cfg["controlled_variation"] = updates
    cfg["suite"] = "controlled"
    cfg["config_name"] = generate_config_name_from_variation(updates)
    decoder_mode_configs.append(cfg)

### Variation B: Top-k Sampling
# For top-k sampling, grid search over both temperature and top_k values.
for temp in temperature_variations:
    for top_k in top_k_values:
        updates = {
            "decoder_config.decoding_mode": "top_k",
            "decoder_config.decoder_top_k": top_k,
            "decoder_config.decoder_temperature": temp,
        }
        cfg = update_multiple_config(base_config, updates)
        cfg["controlled_variation"] = updates
        cfg["suite"] = "controlled"
        cfg["config_name"] = generate_config_name_from_variation(updates)
        decoder_mode_configs.append(cfg)

### Variation C: Top-p (Nucleus) Sampling
# For top-p sampling, grid search over both temperature and top_p values.
for temp in temperature_variations:
    for top_p in top_p_values:
        updates = {
            "decoder_config.decoding_mode": "top_p",
            "decoder_config.decoder_top_p": top_p,
            "decoder_config.decoder_temperature": temp,
        }
        cfg = update_multiple_config(base_config, updates)
        cfg["controlled_variation"] = updates
        cfg["suite"] = "controlled"
        cfg["config_name"] = generate_config_name_from_variation(updates)
        decoder_mode_configs.append(cfg)

# (v) Latency Simulation Variations:____________________________________________________

latency_configs = []

# Variation 1: Baseline (no latency simulation)
updates = {
    "latency_simulation.simulate": False
}
cfg_latency_baseline = update_multiple_config(base_config, updates)
cfg_latency_baseline["controlled_variation"] = updates
cfg_latency_baseline["suite"] = "controlled"    # <-- Added here
cfg_latency_baseline["config_name"] = generate_config_name_from_variation(updates)
latency_configs.append(cfg_latency_baseline)

# Variation 2: Constant Moderate Latency
updates = {
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.05,
    "latency_simulation.delay_max": 0.2,
    "latency_simulation.simulate_burst": False,
}
cfg_latency_mod = update_multiple_config(base_config, updates)
cfg_latency_mod["controlled_variation"] = updates
cfg_latency_mod["suite"] = "controlled"    # <-- Added here
cfg_latency_mod["config_name"] = generate_config_name_from_variation(updates)
latency_configs.append(cfg_latency_mod)

# Variation 3: Constant High Latency
updates = {
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.2,
    "latency_simulation.delay_max": 0.6,
    "latency_simulation.simulate_burst": False,
}
cfg_latency_high = update_multiple_config(base_config, updates)
cfg_latency_high["controlled_variation"] = updates
cfg_latency_high["suite"] = "controlled"    # <-- Added here
cfg_latency_high["config_name"] = generate_config_name_from_variation(updates)
latency_configs.append(cfg_latency_high)

# Variation 4: Bursty Moderate Latency
updates = {
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.05,
    "latency_simulation.delay_max": 0.2,
    "latency_simulation.simulate_burst": True,
    "latency_simulation.burst_interval": 4.0,
    "latency_simulation.burst_size": 5,
}
cfg_latency_bursty_mod = update_multiple_config(base_config, updates)
cfg_latency_bursty_mod["controlled_variation"] = updates
cfg_latency_bursty_mod["suite"] = "controlled"    # <-- Added here
cfg_latency_bursty_mod["config_name"] = generate_config_name_from_variation(updates)
latency_configs.append(cfg_latency_bursty_mod)

# Variation 5: Bursty High Latency
updates = {
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.2,
    "latency_simulation.delay_max": 0.6,
    "latency_simulation.simulate_burst": True,
    "latency_simulation.burst_interval": 5.0,
    "latency_simulation.burst_size": 8,
}
cfg_latency_bursty_high = update_multiple_config(base_config, updates)
cfg_latency_bursty_high["controlled_variation"] = updates
cfg_latency_bursty_high["suite"] = "controlled"    # <-- Added here
cfg_latency_bursty_high["config_name"] = generate_config_name_from_variation(updates)
latency_configs.append(cfg_latency_bursty_high)


# Combine all controlled configurations ========================================================
controlled_config_list = (
    parallelisation_configs +
    batching_configs +
    precision_quantisation_configs +
    decoder_mode_configs +
    latency_configs
)

__all__ = ["controlled_config_list"]

if __name__ == "__main__":
    for cfg in controlled_config_list:
        print(cfg["config_name"])
