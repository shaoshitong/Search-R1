"""Process-local vLLM0.6 profiling for authorized shared GPUs.

The installed global-free delta assumes foreign processes never allocate/free.
Run the SAME required profile once; cap KV bytes by both the requested engine
budget and currently available VRAM after transient activation and headroom.
No sampling, model, attention kernel or global environment is changed.
"""
from contextlib import contextmanager
import gc
import json


def kv_budget(total, utilization, free_now, resident, peak, reserve):
    if min(total,free_now,resident,peak,reserve)<0 or peak<resident or not 0<utilization<=1:
        raise ValueError('Invalid actual process memory accounting')
    transient=peak-resident
    return max(0,int(min(total*utilization-peak,free_now-transient-reserve)))


@contextmanager
def process_local_profile(reserve_bytes):
    import torch
    from vllm.worker.worker import Worker
    original=Worker.determine_num_available_blocks

    @torch.inference_mode()
    def determine(worker):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        worker.model_runner.profile_run()
        torch.cuda.synchronize()
        free,total=torch.cuda.mem_get_info()
        resident=torch.cuda.memory_allocated();peak=torch.cuda.max_memory_allocated()
        budget=kv_budget(total,worker.cache_config.gpu_memory_utilization,
                         free,resident,peak,reserve_bytes)
        size=worker.get_cache_block_size_bytes()
        gpu=budget//size if size else 0
        cpu=int(worker.cache_config.swap_space_bytes//size)if size else 0
        print('SHARED_GPU_PROCESS_LOCAL_KV_PROFILE='+json.dumps(dict(
            free_now=free,total=total,torch_resident=resident,torch_peak=peak,
            reserve_bytes=reserve_bytes,kv_budget_bytes=budget,block_bytes=size,
            gpu_blocks=gpu,cpu_blocks=cpu,foreign_global_delta_used=False)),flush=True)
        if worker.model_runner.lora_manager:worker.model_runner.remove_all_loras()
        gc.collect();torch.cuda.empty_cache()
        return gpu,cpu

    Worker.determine_num_available_blocks=determine
    try:yield
    finally:Worker.determine_num_available_blocks=original
