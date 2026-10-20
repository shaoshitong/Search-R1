"""Read-only runner composition adding same-process CUDA peak audit only."""
import argparse
from pathlib import Path
import runpy
import time
import torch
from .probe_ablation_train import atomic


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args();base=args.base.resolve()
    if not str(base).startswith('/data/user/jxu957/experiments/'):
        raise ValueError('Qualification artifacts must remain remote shared storage')
    started=time.time();normal=False
    try:
        runpy.run_module('reproduction.operations.qwen7b_warmup.probe_g10_train',run_name='__main__')
        normal=True
    finally:
        if torch.cuda.is_initialized():
            device=torch.cuda.current_device()
            atomic(base/'gpu-memory.json',dict(status='ACTUAL_SAME_PROCESS_CUDA_ALLOCATOR_PEAK',
                device=torch.cuda.get_device_name(device),peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),started=started,time=time.time(),
                exit_normally=normal,scope='process lifetime, CUDA allocator only; excludes external allocations'))


if __name__=='__main__':main()
