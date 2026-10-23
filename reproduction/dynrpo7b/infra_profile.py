"""The mature native4 7B fast path, independently bound to DynRPO.

This module has no torch imports, so deployment and configuration checks run on
the CPU.  It changes placement, not the GRPO objective or selection algorithm.
"""
from pathlib import Path
import os
import resource

PROFILE = 'fa3-lean-batched-search-stop-v1'
BASE = '/data/user/jxu957/experiments/dynrpo7b-formal-20261023'
ENV_SOURCE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021'
MODEL = ('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/'
         'assets/qwen25-7b-instruct-a09a35458c70')
WIKI = '/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source/assets-root/assets/wiki'
RETRIEVAL_PORT = 18126
PROXY_PORT = 18110
PROBE_PORT = 18112
EXPORT_PORT = 18116


def native4(run):
    return (run.get('dynrpo7b_selection') is True
        and run.get('profile') == 'common8'
        and run.get('execution_site') == 'hpc3'
        and run.get('execution_backend', 'slurm') == 'slurm'
        and run.get('training_world_size') == 4
        and run.get('data_selection_mode') == 'dynrpo7b'
        and run.get('actor_logical_micro_batch_per_rank') == 2
        and isinstance(run.get('question_probe'), dict)
        and bool(run['question_probe'])
        and not run.get('easy2560_selection', False))


def settings(actor_micro=8):
    """Fields merged into the ordinary common8 run manifest by the driver."""
    from search_g1.easy7b_infra import profile_flags
    from search_g1.rollout_phase_contract import DEFAULT_ROLLOUT_PHASE
    if actor_micro not in (4, 8, 16, 32):
        raise ValueError('Actor physical micro must contain complete logical2 micros')
    return dict(dynrpo7b_selection=True, easy2560_selection=False,
        data_selection_mode='dynrpo7b', actor_micro_batch_per_rank=actor_micro,
        actor_logical_micro_batch_per_rank=2, infra_profile=PROFILE,
        performance_profile='lean_v1', extra_policy_monitor_forwards=False,
        infra_optimizations=profile_flags(PROFILE),
        infra_rollout=dict(DEFAULT_ROLLOUT_PHASE),
        precision_profile='native4_bf16_varlen_fp32_master',
        training_world_size=4, execution_site='hpc3', execution_backend='slurm',
        vllm_gpu_memory_utilization=.1)


def ray_options(run):
    if not native4(run):
        raise ValueError('Ray resources require the explicit DynRPO native4 binding')
    return dict(num_cpus=16, object_store_memory=8 * 1024**3)


def configure_ray(run):
    """Raise only this process's thread limit; no user/host settings change."""
    result = ray_options(run)
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    desired = 8192 if hard == resource.RLIM_INFINITY else min(8192, hard)
    if soft != resource.RLIM_INFINITY and soft < desired:
        resource.setrlimit(resource.RLIMIT_NPROC, (desired, hard))
    return result


def own_base(value):
    path = Path(value)
    if (not path.is_absolute() or path.resolve() != path or
            not str(path).startswith('/data/user/jxu957/experiments/dynrpo7b-formal-')):
        raise ValueError('DynRPO requires its independent HPC3 run namespace')
    return path


def environment(base, commit):
    """All outputs/caches remain in this run's remote shared-storage directory."""
    base = own_base(base)
    return dict(SEARCH_G1_BASE=str(base), SEARCH_G1_SOURCE_COMMIT=commit,
        SEARCH_G1_PROFILE='common8', SEARCH_G1_TRAINING_WORLD_SIZE='4',
        SEARCH_G1_EXECUTION_PROFILE='native4_bf16_varlen_fp32_master',
        QPROBE_SOURCE=str(base / 'repo'),
        PATH=str(base / 'env/bin') + ':/usr/local/bin:/usr/bin:/bin',
        PYTHONPATH=str(base / 'retriever-gpu-lib') + ':' + str(base / 'repo'),
        HF_HOME=str(base / 'cache/hf'), XDG_CACHE_HOME=str(base / 'cache'),
        TMPDIR=str(base / 'tmp'), TRITON_CACHE_DIR=str(base / 'cache/triton'),
        TORCH_EXTENSIONS_DIR=str(base / 'cache/torch-extensions'),
        TORCHINDUCTOR_CACHE_DIR=str(base / 'cache/inductor'),
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='8', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        NCCL_NVLS_ENABLE='0', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1',
        LD_PRELOAD=str(base / 'preparation/flash3-runtime/libcudart.so.12'))
