"""Submit one isolated H100 reference against the existing saved replay tokens."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    p = json.load(sys.stdin)
    base = Path(p['base']); commit = p['commit']; source = Path(p['source_base'])
    sys.path.insert(0, str(base/'source'/commit))
    from reproduction.easy2560_joint4.control import own_base, publish
    own_base(base)
    root = base/'diagnostics'/p.get('diagnostic_name', 'rollout-sdpa-reference-r1')
    root.mkdir(parents=True, exist_ok=True)
    claim = root/'submit-claim.json'; receipt_path = root/'submit-receipt.json'
    if claim.exists():
        print(json.dumps(dict(status='ALREADY_CLAIMED', receipt=
            json.loads(receipt_path.read_text()) if receipt_path.exists() else None)))
        return
    entry = base/'source'/commit/'reproduction/operations/easy7b_20261021/reference_rollout_sdpa_math.py'
    command = ['env', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONUNBUFFERED=1',
        'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'OMP_NUM_THREADS=8',
        'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
        'XDG_CACHE_HOME='+str(root/'cache'), str(source/'env/bin/python'), '-B', str(entry),
        '--root', p['replay_root'], '--checkpoint', p['checkpoint'],
        '--output', str(root/'result.json')]
    argv = ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user',
        '--account=jxu957', '--nodes=1', '--ntasks=1', '--gres=gpu:1',
        '--cpus-per-task=12', '--mem=128G', '--time=00:30:00',
        '--job-name=easy7b-sdpa-reference', '--export=NONE',
        '--chdir='+str(base/'source'/commit), '--output='+str(root/'slurm-%j.log'),
        '--error='+str(root/'slurm-%j.log'), '--wrap='+shlex.join(command)]
    publish(claim, dict(source_commit=commit, command=argv, checkpoint=p['checkpoint'],
        replay_root=p['replay_root'], production_modified=False, created_unix=time.time()))
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    receipt = dict(root=str(root), returncode=result.returncode, stdout=result.stdout,
        stderr=result.stderr, source_commit=commit, checked_unix=time.time(), production_modified=False)
    if not result.returncode:
        job = result.stdout.strip().split(';')[0]
        if not job.isdecimal():raise RuntimeError('Unknown outcome; inspect existing submission claim')
        receipt['job'] = job
        with (root/'job.id').open('x') as stream:
            stream.write(job+'\n'); stream.flush(); os.fsync(stream.fileno())
    publish(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':main()
