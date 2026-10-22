"""Reuse the measured native4 Ray sizing and raise only this process's NPROC."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

from .control import own_base, publish


def raise_self_soft(*, getlimit=resource.getrlimit, setlimit=resource.setrlimit):
    before = tuple(getlimit(resource.RLIMIT_NPROC)); soft, hard = before
    desired = 8192 if hard == resource.RLIM_INFINITY else min(8192, hard)
    changed = soft != resource.RLIM_INFINITY and soft < desired
    if changed: setlimit(resource.RLIMIT_NPROC, (desired, hard))
    after = tuple(getlimit(resource.RLIMIT_NPROC))
    return dict(before=list(before), after=list(after), soft_raised=changed,
                hard_unchanged=after[1] == hard, scope='current process only; fork/exec descendants inherit')


def options(run):
    if (run.get('profile') != 'common8' or run.get('execution_site') != 'hpc3' or
            run.get('execution_backend', 'slurm') != 'slurm' or run.get('training_world_size') != 4 or
            run.get('data_selection_mode') != 'mixed_first' or run.get('easy2560_selection') is not True or
            run.get('question_probe') is not None or run.get('all_correct_virtual_zero') is not False):
        raise ValueError('EASY2560_NATIVE4_RAY_SCOPE_DIFFERS')
    return dict(num_cpus=16, object_store_memory=8 * 1024**3)


def allocator_env_vars():
    """Carry the launcher's actual allocator setting into new Ray processes."""
    value = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
    return {'PYTORCH_CUDA_ALLOC_CONF': value} if value else {}


def configure(base, run, *, phase, role='train'):
    base = own_base(base); kwargs = options(run)
    if role != 'train' or phase not in ('entry', 'ray'):
        raise ValueError('EASY2560_NATIVE4_RAY_PHASE_DIFFERS')
    job = os.environ.get('SLURM_JOB_ID', '')
    if os.getuid() != 14090 or not job.isdecimal() or os.environ.get('SEARCH_G1_BASE') != str(base):
        raise ValueError('EASY2560_NATIVE4_SLURM_OWNER_REQUIRED')
    proof = dict(run_id=run['run_id'], source_commit=run['source_commit'], slurm_job_id=job,
                 phase=phase, role=role, pid=os.getpid(), public_ray_init_kwargs=kwargs,
                 torch_allocator_env=allocator_env_vars(),
                 checked_unix=time.time(), **raise_self_soft())
    publish(base / 'logs' / ('native-ray-resources-train-' + phase + '-' + str(os.getpid()) + '.json'), proof)
    print('EASY2560_NATIVE_RAY_RESOURCES=' + json.dumps(proof), flush=True)
    return kwargs


def main():
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True); a = p.parse_args()
    base = own_base(a.base); run = json.loads((base / 'logs/run-manifest.json').read_text())
    configure(base, run, phase='entry')
    os.execv(sys.executable, [sys.executable, '-B', '-m', 'reproduction.common8.run_pipeline',
                            '--base', str(base), '--export-port', '18016'])


if __name__ == '__main__': main()
