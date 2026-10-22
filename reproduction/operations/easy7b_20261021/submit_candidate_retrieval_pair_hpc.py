"""One independent native4 allocation for sequential cold-cache candidate A/B."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    request = json.load(sys.stdin)
    root = Path(request['root'])
    root.mkdir(parents=True, exist_ok=True)
    claim, receipt_path = root / 'submit-claim.json', root / 'submit-receipt.json'
    if claim.exists() or (root / 'job.id').exists():
        print(json.dumps(dict(status='EXISTING_ATTEMPT_NO_REPLAY',
            receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None)))
        return
    source = Path(request['base']) / 'source' / request['commit']
    target = Path(request['source_base'])
    pin = Path(request['pin_base'])
    command = ['env', 'LD_PRELOAD=' + str(target / 'preparation/flash3-runtime/libcudart.so.12'),
        'PYTHONDONTWRITEBYTECODE=1', 'PYTHONUNBUFFERED=1', 'OMP_NUM_THREADS=8',
        'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'VLLM_NO_USAGE_STATS=1',
        'PYTHONPATH=' + str(source), 'RAY_DEDUP_LOGS=0',
        'TRITON_CACHE_DIR=' + str(root / 'cache/triton'), 'XDG_CACHE_HOME=' + str(root / 'cache'),
        str(target / 'env/bin/python'), '-B',
        str(source / 'reproduction/operations/easy7b_20261021/run_candidate_retrieval_pair.py'),
        '--source-base', str(target), '--checkpoint', str(pin / 'policy-step-66'),
        '--candidate-files', *[str(pin / 'inputs' / (str(offset) + '.pt')) for offset in range(0, 2560, 512)],
        '--service-base', request['service_base'], '--root', str(root / 'runs'),
        '--python', str(target / 'env/bin/python')]
    argv = ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user',
        '--account=jxu957', '--nodes=1', '--ntasks=1', '--gres=gpu:4',
        '--cpus-per-task=48', '--mem=256G', '--time=03:00:00', '--exclude=ACD1-13',
        '--job-name=easy7b-candidate-retrieval-pair', '--export=NONE', '--chdir=' + str(source),
        '--output=' + str(root / 'slurm-%j.log'), '--error=' + str(root / 'slurm-%j.log'),
        '--wrap=exec ' + shlex.join(command)]
    with claim.open('x') as stream:
        json.dump(dict(request, command=argv, created_unix=time.time(),
            production_training_job_untouched='750029', production_retrieval_job_untouched='745198',
            optimizer_loaded=False, optimizer_writes=0), stream, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    receipt = dict(root=str(root), source_commit=request['commit'], returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr, checked_unix=time.time(), production_modified=False)
    if result.returncode == 0:
        job = result.stdout.strip().split(';')[0]
        if not job.isdecimal():
            raise RuntimeError('Unknown submission outcome; inspect existing claim')
        receipt['job'] = job
        with (root / 'job.id').open('x') as stream:
            stream.write(job + '\n'); stream.flush(); os.fsync(stream.fileno())
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
