"""Replay saved tokens in spare CPUs of our idle private retrieval allocation.

No new GPU allocation, service restart, model load or production modification.
The persistent child records the real srun exit; the caller need not hold SSH.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def child(request):
    root, source = Path(request['root']), Path(request['source'])
    target = Path(request['environment_base'])
    command = ['srun', '--jobid=' + request['allocation'], '--overlap', '--ntasks=1',
        '--cpus-per-task=2', '--mem=16G', '--gres=none', '--cpu-bind=none',
        '--export=NONE', '/usr/bin/env', 'PYTHONDONTWRITEBYTECODE=1',
        'PYTHONUNBUFFERED=1', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=1',
        'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'TOKENIZERS_PARALLELISM=false',
        'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'PYTHONPATH=' + str(source),
        str(target / 'env/bin/python'), '-B', str(source /
            'reproduction/operations/easy7b_20261021/qualify_search_action_stop_saved_cpu.py')]
    for filename in request['candidate_files']:
        command += ['--candidate-file', filename]
    command += ['--checkpoint', request['checkpoint'], '--output', str(root / 'result.json')]
    began = time.time()
    done = subprocess.run(command, stdin=subprocess.DEVNULL)
    (root / 'exit.json').write_text(json.dumps(dict(started_unix=began,
        finished_unix=time.time(), returncode=done.returncode, command=command,
        allocation=request['allocation'], production_modified=False), indent=2) + '\n')
    return done.returncode


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--child':
        raise SystemExit(child(json.loads(Path(sys.argv[2]).read_text())))
    request = json.load(sys.stdin)
    root = Path(request['root'])
    root.mkdir(parents=True, exist_ok=True)
    claim = root / 'launch-claim.json'
    if claim.exists():
        print(json.dumps(dict(status='EXISTING_REPLAY_NO_DUPLICATE', root=str(root))))
        return
    with claim.open('x') as stream:
        json.dump(dict(request, claimed_unix=time.time(), owner_uid=os.getuid()), stream, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    with (root / 'replay.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
            '--child', str(claim)], stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=process.pid, root=str(root), allocation=request['allocation'],
        source=request['source'], started_unix=time.time(), production_modified=False,
        optimizer_loaded=False, gpu_allocation_requested=False)
    (root / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
