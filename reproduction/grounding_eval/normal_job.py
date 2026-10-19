"""One explicitly authorized normal-only worker; never starts CF or training."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    if job.get('normal_only') is not True:
        raise ValueError('Explicit normal-only authorization binding required')
    command = job['normal_command']
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError('Explicit argv required')
    output = Path(job['control_output']); output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'binding.json').exists():
        if json.loads((output / 'binding.json').read_text()) != job:
            raise ValueError('Normal-only controller binding changed')
    else:
        atomic(output / 'binding.json', job)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    status = dict(state='RUNNING', pid=os.getpid(), normal_only=True,
                  started_at=time.time(), command=command,
                  job_sha256=hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest())
    atomic(output / 'worker.json', status)
    with (output / 'normal.log').open('ab', buffering=0) as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        status['child_pid'] = child.pid
        atomic(output / 'worker.json', status)
        code = child.wait()
    status.update(state='COMPLETE' if code == 0 else 'FAILED',
                  exit_code=code, completed_at=time.time())
    atomic(output / 'exit.json', status)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
