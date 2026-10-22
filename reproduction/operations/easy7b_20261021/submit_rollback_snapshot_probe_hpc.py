"""Submit one independent native4 snapshot-timing probe; never touch production."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    request = json.load(sys.stdin)
    base = Path(request['base'])
    commit = request['commit']
    source = Path(request['source_base'])
    sys.path.insert(0, str(base/'source'/commit))
    from reproduction.easy2560_joint4.control import own_base, publish
    own_base(base)
    root = base/'diagnostics'/'native4-rollback-snapshot-r1-20261023'
    root.mkdir(parents=True, exist_ok=True)
    claim = root/'submit-claim.json'
    if claim.exists():
        receipt = root/'submit-receipt.json'
        print(json.dumps(dict(status='EXISTING_ATTEMPT_INSPECT_NO_REPLAY',
            root=str(root), receipt=json.loads(receipt.read_text()) if receipt.is_file() else None)), flush=True)
        return
    checkpoint = Path(request['checkpoint'])
    script = base/'source'/commit/'reproduction/operations/easy7b_20261021/benchmark_rollback_snapshot_native4.py'
    command = ['env', 'LD_PRELOAD='+str(source/'preparation/flash3-runtime/libcudart.so.12'),
        'PYTHONDONTWRITEBYTECODE=1', 'PYTHONUNBUFFERED=1', 'OMP_NUM_THREADS=8',
        'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'NCCL_NVLS_ENABLE=0',
        'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
        'TRITON_CACHE_DIR='+str(root/'cache/triton'), 'XDG_CACHE_HOME='+str(root/'cache'),
        str(source/'env/bin/python'), '-B', '-m', 'torch.distributed.run',
        '--standalone', '--nnodes=1', '--nproc_per_node=4', str(script),
        '--source-base', str(source), '--checkpoint', str(checkpoint),
        '--mode', 'legacy', '--output', str(root), '--steps', '20', '--selection-step', '51']
    argv = ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user',
        '--account=jxu957', '--nodes=1', '--ntasks=1', '--gres=gpu:4',
        '--exclude=ACD1-13', '--cpus-per-task=48', '--mem=512G', '--time=00:30:00',
        '--job-name=easy7b-rollback-cost', '--export=NONE',
        '--chdir='+str(base/'source'/commit), '--output='+str(root/'slurm-%j.log'),
        '--error='+str(root/'slurm-%j.log'), '--wrap='+shlex.join(command)]
    publish(claim, dict(source_commit=commit, command=argv, checkpoint=str(checkpoint),
        source_base=str(source), source_selection_step=51, scientific_math_changed=False,
        role='independent legacy constructor timing; no production edits or rollout',
        production_job_untouched='749598', created_unix=time.time()))
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    receipt = dict(source_commit=commit, root=str(root), command=argv,
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        checked_unix=time.time(), production_modified=False, snapshot_timing_complete=False)
    if result.returncode == 0:
        job = result.stdout.strip().split(';')[0]
        if not job.isdecimal():
            raise RuntimeError('Unknown sbatch result; inspect the saved launch claim')
        receipt['job'] = job
        with (root/'job.id').open('x') as stream:
            stream.write(job+'\n'); stream.flush(); os.fsync(stream.fileno())
    publish(root/'submit-receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
