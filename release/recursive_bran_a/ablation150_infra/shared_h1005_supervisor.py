"""Authorized h100-5 shared continuation; no Slurm job is borrowed."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import fcntl
import argparse


def write(path, value):
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, sort_keys=True) + '\n')
    os.replace(temp, path)


def main(reuse_admitted=False):
    assert socket.gethostname().split('.')[0] == 'h100-5' and os.getuid() == 0
    base = Path('/dataset/shaoshitong/recursive-ablation150-20261014')
    source = base / 'source-h1005-shared-d15f311'
    reference = Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
    runtime = Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
    control = base / ('shared-h1005-study-retry-20261018' if reuse_admitted else 'shared-h1005-20261018')
    control.mkdir(exist_ok=True)
    (base / 'budget-5/logs').mkdir(parents=True, exist_ok=True)
    lock = (control / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (control / 'actual-exit.json').exists():
        raise RuntimeError('Terminal supervisor exists; inspect it before a new explicit recovery')
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('SLURM_') or key in ('LD_LIBRARY_PATH', 'HTTP_PROXY', 'HTTPS_PROXY',
                'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'RAY_ADDRESS'):
            env.pop(key)
    env.update(RBA_H100_SHARED_EXECUTION_ID='shared-h1005-ablation150-20261018',
        RBA_ABLATION_BASE=str(base), RBA_ABLATION_SOURCE_COMMIT='233b0dc95de3f7b797d5b424d3b7dcaafca5328f',
        RBA_ABLATION_ENTRY_COMMIT='d15f3119643b4a95364f3180f7730db102651106',
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1', NVIDIA_TF32_OVERRIDE='0',
        PYTHONPATH=str(source), PATH=str(runtime / 'env/bin') + ':/usr/local/bin:/usr/bin:/bin')
    start = dict(time=time.time(), pid=os.getpid(), uid=os.getuid(),
        tick=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        backend='h100_shared', Slurm_allocation=None, target='h100-5',
        resume_completed_outer=111, resume_Adam_each_rank=2220,
        execution_id=env['RBA_H100_SHARED_EXECUTION_ID'])
    write(control / 'started.json', start)
    server = json.loads((reference / 'retriever/receipts/h1005-actual-overlay-launch.json').read_text())
    stages = [
        ('activate-retriever', [str(runtime / 'env/bin/python'), '-B',
            str(reference / 'source-resume100-v1/release/recursive_bran_a/h100_migration/activate.py'),
            str(reference / 'retriever'), str(server['pid'])]),
        ('new-service-numeric-gate', [str(runtime / 'env/bin/python'), '-B',
            str(reference / 'source-resume100-v1/release/recursive_bran_a/h100_migration/gate.py'),
            '--base', str(reference / 'retriever')]),
        ('native-study-continuation', ['bash', str(source / 'release/recursive_bran_a/ablation150_v1/study.sh')]),
    ]
    if reuse_admitted:
        stages = [('reuse-live-admitted-retriever', [str(runtime / 'env/bin/python'), '-B',
            str(source / 'release/recursive_bran_a/h100_migration/reuse_retriever.py'),
            str(reference / 'retriever')]), stages[-1]]
    for stage, command in stages:
        with (control / (stage + '.log')).open('xb') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(control / 'stage.json', dict(time=time.time(), stage=stage,
                pid=child.pid, command=command, status='RUNNING'))
            rc = child.wait()
        write(control / (stage + '-wait.json'), dict(time=time.time(), pid=child.pid, rc=rc))
        if rc:
            write(control / 'actual-exit.json', dict(time=time.time(), stage=stage, rc=rc,
                training_completion_claim=False))
            return rc
    write(control / 'actual-exit.json', dict(time=time.time(), stage='study-returned', rc=0,
        training_completion_claim='Inspect native150 results and original remaining conditions'))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-admitted-retriever', action='store_true')
    args = parser.parse_args()
    raise SystemExit(main(args.reuse_admitted_retriever))
