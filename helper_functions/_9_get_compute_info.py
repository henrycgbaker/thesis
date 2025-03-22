import io
import contextlib
import os
import torch
import psutil
import subprocess
import ptflops
import logging
logger = logging.getLogger(__name__)


def get_flops(model, input_ids_batch):
    """
    Computes total FLOPs for a batch of tokenised input samples.
    Each sample is measured individually to account for variable sequence lengths.

    Parameters:
        model: The model to measure.
        input_ids_batch: Tensor of shape (batch_size, seq_len)

    Returns:
        Total estimated FLOPs (float) for the batch.
    """
    total_flops = 0.0
    batch_size = input_ids_batch.shape[0]


    for i in range(batch_size):
        sample_length = input_ids_batch[i].shape[0]

        def input_constructor(input_res):
            dummy_input = torch.zeros((1,) + input_res, dtype=torch.long).to(input_ids_batch.device)
            return {"input_ids": dummy_input}

        try:
            with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                flops_single, _ = ptflops.get_model_complexity_info(
                    model,
                    input_res=(sample_length,),
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False,
                    input_constructor=input_constructor
                )
            if flops_single is None:
                logger.warning(f"FLOPs computation returned None for sample {i}. Skipping this sample.")
                continue
            total_flops += flops_single
        except Exception as e:
            logger.warning(f"FLOPs computation failed for sample {i}: {e}")
            continue

    return total_flops


def get_memory(device):
    """
    Returns a dictionary with current and peak memory usage on the given CUDA device.
    """
    torch.cuda.reset_peak_memory_stats(device)
    
    return {
        "GPU_max_mem_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "GPU_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    

def get_global_gpu_usage():
    """
    global, so only need to run on main process!
    """
    gpu_utilisation = {}
    
    # GPU utilization using nvidia-smi
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        )
        lines = result.decode("utf-8").strip().splitlines()
        gpu_utils = [float(line.strip()) for line in lines if line.strip()]
        gpu_utilisation["gpu_utilization_percent"] = gpu_utils
    except Exception as e:
        gpu_utilisation["gpu_utilization_percent"] = f"Error: {str(e)}"

    return gpu_utilisation

def get_gpu_utilisation(device):
    """
    Returns the GPU utilisation (in percent) for the given torch.device.
    Adjusts for modifications to CUDA_VISIBLE_DEVICES by mapping the effective device index
    to the physical GPU index.

    Returns a dictionary with:
      - 'effective_device': the torch device (e.g. "cuda:0")
      - 'physical_device': the physical GPU id (if CUDA_VISIBLE_DEVICES is set, else same as effective)
      - 'gpu_utilization_percent': the utilisation percentage (float)
    """
    gpu_utilisation = {}
    try:
        # Run nvidia-smi to get utilisation for all GPUs.
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        )
        lines = result.decode("utf-8").strip().splitlines()
        all_gpu_utils = [float(line.strip()) for line in lines if line.strip()]

        # Get the effective device index from torch.device.
        if isinstance(device, torch.device):
            effective_index = device.index if device.index is not None else 0
        else:
            # If provided as string, try to extract index from "cuda:0"
            try:
                effective_index = int(str(device).split("cuda:")[1])
            except Exception:
                effective_index = 0

        # Check if CUDA_VISIBLE_DEVICES is set.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible is not None:
            # Map the effective index to the physical GPU id.
            mapping = [int(x.strip()) for x in visible.split(",") if x.strip()]
            # In many setups, if CUDA_VISIBLE_DEVICES is set, the CUDA runtime renumbers GPUs to 0,1,...
            # But to be safe, we return both the effective and physical indices.
            physical_index = mapping[effective_index] if effective_index < len(mapping) else effective_index
        else:
            physical_index = effective_index

        gpu_utilisation["effective_device"] = f"cuda:{effective_index}"
        gpu_utilisation["physical_device"] = physical_index
        gpu_utilisation["gpu_utilization_percent"] = float(all_gpu_utils[physical_index])
    except Exception as e:
        gpu_utilisation["error"] = f"Error: {str(e)}"

    return gpu_utilisation


def get_cpu_utilisation():
    
    cpu_utilisation = {}
    process = psutil.Process(os.getpid())

    cpu_utilisation["pid"] = process.pid
    cpu_utilisation["cpu_usage_percent"] = process.cpu_percent(interval=1.0)
    cpu_utilisation["cpu_memory_usage_bytes"] = process.memory_info().rss
    
    return cpu_utilisation       
        
