"""Submit one private two-GPU service; leave production services untouched."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid


def main():
    request = json.load(sys.stdin)
    base = Path(request['base'])
    source = base / 'source' / request['commit']
    (base / 'logs').mkdir(parents=True, exist_ok=True)
    (base / 'preparation').mkdir(exist_ok=True)
    for name in ('cache', 'tmp'):
        (base / name).mkdir(exist_ok=True)
    claim = base / 'preparation/service-submit-claim.json'
    receipt_path = base / 'preparation/service-submission.json'
    if claim.exists() or (base / 'job.id').exists():
        print(json.dumps(dict(status='EXISTING_ATTEMPT_NO_REPLAY',
            receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None)))
        return
    service = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-service2-20261021')
    libraries = list((service / 'env/lib/python3.10/site-packages/nvidia').glob('*/lib'))
    libraries += [service / 'env/lib/python3.10/site-packages/torch/lib']
    run_id = uuid.uuid4().hex
    env = dict(PATH=str(service / 'env/bin') + ':/usr/local/bin:/usr/bin:/bin',
        PYTHONPATH=str(service / 'retriever-gpu-lib') + ':' + str(source),
        LD_LIBRARY_PATH=':'.join(map(str, libraries)), OMP_NUM_THREADS='8',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HOME=str(base / 'cache/hf'), XDG_CACHE_HOME=str(base / 'cache'), TMPDIR=str(base / 'tmp'))
    # Insert this immutable source ahead of the reused environment's old repo.
    entry = "import sys,runpy;sys.path.insert(0," + repr(str(source)) + ");runpy.run_module('reproduction.easy2560_joint4.retrieval_service',run_name='__main__')"
    program = ['env', *[key + '=' + value for key, value in env.items()],
        str(service / 'env/bin/python'), '-B', '-c', entry, '--base', str(base),
        '--run-id', run_id, '--source-commit', request['commit'],
        '--index', request['index'], '--corpus', request['corpus'],
        '--encoder', str(service / 'assets/e5-base-v2'),
        '--port', '18022', '--faiss-batch-size', '16', '--faiss-max-wait-ms', '2']
    argv = ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user',
        '--account=jxu957', '--nodes=1', '--ntasks=1', '--gres=gpu:2',
        '--cpus-per-task=24', '--mem=384G', '--time=04:00:00', '--exclude=ACD1-13',
        '--job-name=easy7b-private-retrieval-batch', '--export=NONE', '--chdir=' + str(source),
        '--output=' + str(base / 'logs/service-%j.log'), '--error=' + str(base / 'logs/service-%j.log'),
        '--wrap=exec ' + shlex.join(program)]
    with claim.open('x') as stream:
        json.dump(dict(request, command=argv, created_unix=time.time(), service_run_id=run_id,
            production_training_job_untouched='750029', production_retrieval_job_untouched='745198'),
            stream, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    receipt = dict(base=str(base), source_commit=request['commit'], service_run_id=run_id,
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        checked_unix=time.time(), production_modified=False)
    if result.returncode == 0:
        job = result.stdout.strip().split(';')[0]
        if not job.isdecimal():
            raise RuntimeError('Unknown submission outcome; inspect existing claim')
        receipt['job'] = job
        with (base / 'job.id').open('x') as stream:
            stream.write(job + '\n'); stream.flush(); os.fsync(stream.fileno())
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
