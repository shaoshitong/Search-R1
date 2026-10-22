"""Submit a single independent H100 inference replay; never touches live jobs."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    request = json.loads(sys.stdin.read())
    source_base = Path(request['source_base'])
    base = Path(request['base'])
    commit = request['commit']
    sys.path.insert(0, str(base / 'source' / commit))
    from reproduction.easy2560_joint4.control import own_base, publish
    own_base(base)
    root = base / 'diagnostics' / request.get('diagnostic_name', 'rollout-flash3')
    root.mkdir(parents=True, exist_ok=True)
    claim = root / 'submit-claim.json'
    if claim.exists():
        print(json.dumps(dict(status='ALREADY_CLAIMED', claim=json.loads(claim.read_text()),
                              receipt=json.loads((root/'submit-receipt.json').read_text())
                              if (root/'submit-receipt.json').is_file() else None)), flush=True)
        return
    checkpoint = Path(request['checkpoint'])
    if not (checkpoint/'complete.json').is_file():
        raise ValueError('Use an already completed source checkpoint')
    code = base/'source'/commit/'reproduction/operations/easy7b_20261021/benchmark_rollout_flash3.py'
    command = ['env',
        'LD_PRELOAD='+str(source_base/'preparation/flash3-runtime/libcudart.so.12'),
        'PYTHONDONTWRITEBYTECODE=1', 'PYTHONUNBUFFERED=1', 'OMP_NUM_THREADS=8',
        'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'VLLM_NO_USAGE_STATS=1',
        'TRITON_CACHE_DIR='+str(root/'cache/triton'), 'XDG_CACHE_HOME='+str(root/'cache'),
        str(source_base/'env/bin/python'), '-B', str(code), '--root', str(root),
        '--source-base', str(source_base), '--checkpoint', str(checkpoint),
        '--candidate-file', request['candidate_file']]
    # Independent replay geometry only; production configuration is untouched.
    for key, default in (('batch', 64), ('output_tokens', 128), ('warmup', 2), ('repeats', 5)):
        value = request.get(key, default)
        if type(value) is not int or value < (0 if key == 'warmup' else 1):
            raise ValueError('Invalid diagnostic replay argument: ' + key)
        command.extend(['--' + key.replace('_', '-'), str(value)])
    argv = ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user',
        '--account=jxu957', '--nodes=1', '--ntasks=1', '--gres=gpu:1',
        '--cpus-per-task=12', '--mem=128G', '--time=02:00:00',
        '--job-name=easy7b-fa3-rollout-test', '--export=NONE',
        '--chdir='+str(base/'source'/commit),
        '--output='+str(root/'slurm-%j.log'), '--error='+str(root/'slurm-%j.log'),
        '--wrap='+shlex.join(command)]
    publish(claim, dict(source_commit=commit, source_checkpoint=str(checkpoint), command=argv,
                        production_modified=False, created_unix=time.time()))
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    receipt = dict(source_commit=commit, returncode=result.returncode,
                   stdout=result.stdout, stderr=result.stderr, checked_unix=time.time(),
                   root=str(root), production_modified=False)
    if result.returncode == 0:
        job = result.stdout.strip().split(';')[0]
        if not job.isdecimal():
            raise RuntimeError('Unknown submission outcome: inspect claim before retry')
        receipt['job'] = job
        with (root/'job.id').open('x') as f:
            f.write(job+'\n'); f.flush(); os.fsync(f.fileno())
    publish(root/'submit-receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
