"""Place the CPU Ray driver's probe on one already allocated Slurm actor GPU."""
import hashlib
import json
import os
from pathlib import Path
import socket
import time


def source_root(base,run):
    root = Path(base)/'source'/run['source_commit']
    if (root.resolve() != root or root.is_symlink() or
            (Path(base)/'repo').resolve() != root or root.stat().st_uid != os.getuid()):
        raise ValueError('PERIODIC_IMMUTABLE_SOURCE_ALIAS_CHANGED')
    return root


def validate_route(run,allocation,claim,hostname,job_id):
    if (run.get('execution_site') != 'hpc3' or run.get('execution_backend','slurm') != 'slurm' or
            run.get('training_world_size') != 4 or
            run.get('question_probe',{}).get('head_kind') != 'qwen_native_periodic_lastblock_binary_v1' or
            run['question_probe'].get('head_device') != 'cuda:0' or
            allocation.get('accepted') is not True or allocation.get('node') != hostname.split('.')[0] or
            allocation.get('training_world_size') != 4 or allocation.get('slurm_job_id') != job_id or
            allocation.get('run_id') != run['run_id'] or claim.get('run_id') != run['run_id'] or
            claim.get('slurm_job_id') != job_id):
        raise ValueError('PERIODIC_DRIVER_SLURM_ALLOCATION_MISMATCH')
    uuids = allocation['actual_cuda_runtime_uuids']
    if len(uuids) != len(set(uuids)) or len(uuids) != 4 or any(not u.startswith('GPU-') for u in uuids):
        raise ValueError('PERIODIC_DRIVER_REQUIRES_FOUR_ADMITTED_UUIDS')
    return dict(gpu_uuid=uuids[0],training_gpu_uuids=uuids,ray_gpu_resources_reserved=0,
                shares_admitted_actor_training_gpu=True,head_device='cuda:0')


def bind(config):
    import ray
    import torch
    from search_g1.qprobe_driver_gpu import install_driver_visibility, _birth
    from reproduction.common8.native_allocation import cuda_uuids, _write_local_once
    base = Path(os.environ['SEARCH_G1_BASE'])
    def read(name):
        path = base/name
        if path.resolve() != path or path.is_symlink() or path.stat().st_uid != os.getuid():
            raise ValueError('PERIODIC_DRIVER_UNOWNED_ADMISSION_RECORD')
        return json.loads(path.read_text())
    run = read('logs/run-manifest.json')
    allocation = read('logs/native-allocation-admission.json')
    claim = read('logs/training-launch-claim.json')
    if ray.get_gpu_ids() != [] or config.trainer.n_gpus_per_node != 4:
        raise ValueError('PERIODIC_DRIVER_MUST_REMAIN_CPU_TASK_WITH_FOUR_ACTORS')
    root = source_root(base,run)
    if Path(__file__).resolve() != root/'search_g1/qprobe_periodic_driver.py':
        raise ValueError('PERIODIC_DRIVER_LOADED_DIFFERENT_SOURCE')
    for name in ('search_g1/qprobe_periodic_driver.py','search_g1/qprobe_driver_gpu.py','verl/trainer/main_ppo_eco.py'):
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != run['source_sha256'][name]:
            raise ValueError('PERIODIC_DRIVER_SOURCE_SHA_CHANGED')
    shell = _birth(claim['shell_pid'])
    if shell['start_ticks'] != claim['shell_start_ticks']:
        raise ValueError('PERIODIC_DRIVER_TRAINING_SHELL_BIRTH_CHANGED')
    route = validate_route(run,allocation,claim,socket.gethostname(),os.environ.get('SLURM_JOB_ID'))
    incoming = os.environ.get('CUDA_VISIBLE_DEVICES')
    if incoming not in (None,'',route['gpu_uuid'],','.join(allocation['visible_devices']),','.join(route['training_gpu_uuids'])):
        raise ValueError('PERIODIC_DRIVER_UNBOUND_INCOMING_GPU_VISIBILITY')
    actual = install_driver_visibility(route,environ=os.environ,cuda_initialized=torch.cuda.is_initialized,
        cuda_device_count=torch.cuda.device_count,cuda_uuids=cuda_uuids)
    proof = dict(stage='PERIODIC_PROBE_DRIVER_ADMITTED_ACTOR_GPU_BOUND',run_id=run['run_id'],
        source_commit=run['source_commit'],slurm_job_id=os.environ['SLURM_JOB_ID'],
        producer_identity=_birth(os.getpid()),incoming_cuda_visible_devices=incoming,
        allocation_sha256=hashlib.sha256((base/'logs/native-allocation-admission.json').read_bytes()).hexdigest(),
        child_actor_cuda_assignment_unchanged=True,**route,**actual,checked_unix=time.time())
    _write_local_once(base/'logs/periodic-qwen-driver-gpu-route.json',proof)
    return proof
