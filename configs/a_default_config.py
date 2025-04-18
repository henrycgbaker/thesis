base_config = {
    # Default values that will be remain constant across experiments 
    "config_name": "default",
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    "is_encoder_decoder": False,
    "task_type": "text_generation",
    "inference_type": "pure_generative",
    "gpu_list": [0, 1, 2, 3],
    "backend": "pytorch",
    "save_outputs": True,
    
    # Default values that will be overridden by the grid:
    "max_input_tokens": 128, #2048 is Llama's limit
    "max_output_tokens": 128,
    "num_input_prompts": 128,  # 500 * 200 = 100,000 output tokens.
    "decode_token_to_text": True,
    "num_processes": 4,
    "batching_options": {
        "batch_size___fixed_batching": 16,
        "adaptive_batching": False,
        "adaptive_max_tokens": 0,
        "max_batch_size___adaptive_batching": 0,
    },
    "sharding_config": {
        "fsdp_config": {
            "use_orig_params": False,
            "cpu_offload": False
        },
        "sharding_strategy": "NO_SHARD" #FULL_SHARD doesn't work
    },
    "query_rate": 1.0,
    "latency_simulation" : {
        "simulate": False,        # If True, introduce artificial delays.
        "delay_min": 0,           # Minimum delay (in seconds, e.g. 50ms).
        "delay_max": 0,         # Maximum delay (e.g. 300ms).
        "simulate_burst": False,   # If True, simulate burst traffic conditions.
        "burst_interval": 0.0,    # After a burst, wait for this many seconds.
        "burst_size": 0           # Define a burst as every 5 batches.
    },
    "decoder_config":{
        "decoder_temperature": 1.0,
        "decoder_top_k": 0,      
        "decoder_top_p": 0.0
    }, 
    "fp_precision": "float32", # float16
    "quantization_config": {
        "quantization": True,
        "load_in_8bit": False,
        "load_in_4bit": True,
        "cached_flops_for_quantised_models": 16949970993152 # For quantized models, store a cached FLOPs value! 
    }
}