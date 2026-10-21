"""Submit one isolated H100 SDPA diagnostic; no policy is loaded or changed."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


def main():
    request = json.loads(sys.stdin.read())
    commit = request['commit']
    if os.getuid() != 14090 or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Require owned HPC account and published full commit')
    target = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp12-20261022')
    sys.path.insert(0, str(target/'source'/commit))
    from reproduction.easy2560_joint4.control import publish
    from reproduction.easy2560_joint4.flash3_runtime import binding
    root = target/'diagnostics/fa3-sdpa-math'
    root.mkdir(parents=True, exist_ok=True)
    claim = root/'submit-claim.json'
    if claim.exists():
        print(json.dumps(dict(status='EXISTING_SDPA_SUBMISSION_INSPECT_ACTUAL',
                              claim=json.loads(claim.read_text())))); return
    (root/'logs').mkdir()
    manifest = dict(training_world_size=4, source_commit=commit,
                    infra_flash_attention_3=binding(target))
    publish(root/'logs/run-manifest.json', manifest)
    private = target/'preparation/flash3-runtime/libcudart.so.12'
    code = target/'source'/commit/'reproduction/operations/easy7b_20261021/qualify_fa3_sdpa_cuda.py'
    command = ['env', 'LD_PRELOAD='+str(private), 'PYTHONDONTWRITEBYTECODE=1',
               'PYTHONUNBUFFERED=1', 'OMP_NUM_THREADS=4',
               str(target/'env/bin/python'), '-B', str(code), '--base', str(root)]
    argv = ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user',
            '--account=jxu957', '--nodes=1', '--ntasks=1', '--gres=gpu:1',
            '--cpus-per-task=8', '--mem=64G', '--time=00:20:00',
            '--job-name=easy7b-fa3-sdpa', '--export=NONE', '--chdir='+str(target/'repo'),
            '--output='+str(root/'slurm-%j.log'), '--error='+str(root/'slurm-%j.log'),
            '--wrap='+shlex.join(command)]
    publish(claim, dict(source_commit=commit, command=argv, checked_unix=time.time(),
                        diagnostic_only=True, model_loaded=False, production_modified=False))
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    value = dict(source_commit=commit, returncode=result.returncode, stdout=result.stdout,
                 stderr=result.stderr, checked_unix=time.time(), root=str(root))
    if result.returncode == 0:
        job = result.stdout.strip().split(';')[0]
        if not job.isdecimal(): raise ValueError('Inspect unknown sbatch outcome')
        value['job'] = job
        with (root/'job.id').open('x') as stream:
            stream.write(job+'\n'); stream.flush(); os.fsync(stream.fileno())
    publish(root/'submit-receipt.json', value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
