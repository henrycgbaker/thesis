base_config = {
    # Default values that will be remain constant across experiments 
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "is_encoder_decoder": False,
    "task_type": "text_generation",
    "inference_type": "pure_generative",
    "gpu_list": [0, 1, 2, 3],
    "backend": "pytorch",
    "save_outputs": False,
    
    # Default values that will be overridden by the grid:
    "max_input_tokens": 500, #2048 is Llama's limit
    "max_output_tokens": 200,
    "num_input_prompts": 500,  # 500 * 200 = 100,000 output tokens.
    "decode_token_to_text": True,
    "num_processes": 4,
    "batching_options": {
        "batch_size___fixed_batching": 16,
        "adaptive_batching": False,
        "adaptive_max_tokens": 3000,
        "max_batch_size___adaptive_batching": 100,
    },
    "sharding_config": {
        "fsdp_config": {
            "use_orig_params": True,
            "cpu_offload": True
        },
        "sharding_strategy": "NO_SHARD" #FULL_SHARD doesn't work
    },
    "query_rate": 1.0,
    "decoder_temperature": 1.0,
    "fp_precision": "float32",
    "quantization_config": {
        "quantization": False,
        "load_in_8bit": False,
        "load_in_4bit": False,
        # For quantized models, store a cached FLOPs value:
        "cached_flops_for_quantised_models": None
    }
}


grid_params = {
    "max_input_tokens": [256, 512, 768],
    "max_output_tokens": [100, 200, 300],
    "num_input_prompts": [1000, 500, 250],  
    # The product of max_output_tokens and num_input_prompts should equal 100,000.
    "decode_token_to_text": [True, False],
    "batching_options": [
        {"batch_size___fixed_batching": 16, "adaptive_batching": False},
        {"batch_size___fixed_batching": 32, "adaptive_batching": False},
    ],
    "sharding_config": [
        {"fsdp_config": {"use_orig_params": True, "cpu_offload": True}, "sharding_strategy": "NO_SHARD"},
        {"fsdp_config": {"use_orig_params": True, "cpu_offload": False}, "sharding_strategy": "FULL_SHARD"}
    ],
    "query_rate": [0.5, 1, 1.2, 1.5, 2, 4, 8],
    "decoder_temperature": [0.0, 1.0, 2.0],
    "fp_precision": ["float16", "float32"],
    "quantization_config": [
        {"quantization": False, "load_in_8bit": False, "load_in_4bit": False},
        {"quantization": True, "load_in_8bit": True, "load_in_4bit": False},
    ]
}