"""Private vLLM 0.6.3 executor for GPUs shared with unrelated CUDA contexts.

Uses the installed executor's worker factory; no installed class/file is
modified. The actual model profile still runs on every TP rank. Cache capacity
is bounded by this process's CUDA peak and current free memory, so another
process releasing memory cannot make a global free-memory delta negative.
"""
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time

import torch
import vllm
from vllm.executor.multiproc_gpu_executor import MultiprocessingGPUExecutor
from vllm.worker.worker import Worker

ORIGINAL_PROFILE_SHA256 = hashlib.sha256(inspect.getsource(Worker.determine_num_available_blocks).encode()).hexdigest()


class SharedVramWorker(Worker):
    @torch.inference_mode()
    def determine_num_available_blocks(self):
        # This is the one intentionally replaced 0.6.3 method. In particular,
        # init_device/load_model/profile_run/cache initialization/warm-up and
        # the original warm-up seed reset retain their installed behavior.
        device = self.device
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        own_before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        self.model_runner.profile_run()
        torch.cuda.synchronize(device)
        own_peak = torch.cuda.max_memory_allocated(device)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        own_retained = torch.cuda.memory_allocated(device)
        own_reserved = torch.cuda.memory_reserved(device)
        free, total = torch.cuda.mem_get_info(device)
        # Empty-cache above releases unused allocator reserves. Retain space
        # for the real profiled activation peak to recur after KV allocation.
        future_own_peak = max(0, own_peak - own_retained)
        reserve = int(os.environ['Q7B_SHARED_KV_RESERVE_BYTES'])
        cap = int(os.environ['Q7B_SHARED_KV_BLOCK_CAP'])
        block_bytes = self.get_cache_block_size_bytes()
        available = max(0, free - reserve - future_own_peak)
        gpu_blocks = min(cap, available // block_bytes) if block_bytes else 0
        cpu_blocks = self.cache_config.swap_space_bytes // block_bytes if block_bytes else 0
        proof = dict(status='REAL_PROFILE_WITH_BOUNDED_SHARED_VRAM_KV_CACHE', pid=os.getpid(), rank=self.rank,
            local_rank=self.local_rank, vllm_version=vllm.__version__, original_profile_method_sha256=ORIGINAL_PROFILE_SHA256,
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            numeric_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            original_init_free_bytes=self.init_gpu_memory, current_free_bytes=free, total_bytes=total,
            own_torch_allocated_before_profile=own_before, own_torch_peak_allocated=own_peak,
            own_torch_allocated_after_profile=own_retained, own_torch_reserved_after_cleanup=own_reserved,
            own_profiled_future_peak_bytes=future_own_peak, reserve_bytes=reserve, bounded_available_bytes=available,
            block_size=self.cache_config.block_size, block_bytes=block_bytes, requested_block_cap=cap,
            num_gpu_blocks=int(gpu_blocks), num_cpu_blocks=int(cpu_blocks),
            actual_kv_bytes=int(gpu_blocks * block_bytes), max_model_len=self.model_config.max_model_len,
            minimum_full_context_blocks=math.ceil(self.model_config.max_model_len / self.cache_config.block_size),
            real_profile_run=True, global_free_delta_used=False, init_free_modified=False, epoch=time.time())
        folder = Path(os.environ['Q7B_SHARED_KV_PROFILE_DIR'])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / ('rank-%d-pid-%d.json' % (self.rank, os.getpid()))
        temporary = path.with_name(path.name + '.tmp-' + str(time.time_ns()))
        try:
            with temporary.open('x') as stream:
                json.dump(proof, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print('SHARED_VRAM_REAL_KV_PROFILE=' + json.dumps(proof, sort_keys=True), flush=True)
        # The installed distributed executor takes the minimum across TP
        # ranks. Its unmodified initialize_cache still checks full context
        # length and performs the real allocations, warm-up and seed reset.
        return int(gpu_blocks), int(cpu_blocks)


class SharedVramExecutor(MultiprocessingGPUExecutor):
    def _get_worker_module_and_class(self):
        if self.scheduler_config.is_multi_step or self.speculative_config:
            raise ValueError('Shared-VRAM adapter supports the existing single-step non-speculative G5 protocol')
        return (__name__, 'SharedVramWorker', None)
