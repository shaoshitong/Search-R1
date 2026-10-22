"""One independent2GPU retrieval batching probe, existing assets/environment only."""
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
    source = base/'source'/request['commit']
    root = base/'diagnostics'/request.get('diagnostic_name', 'retrieval-batching-r1-20261023')
    root.mkdir(parents=True, exist_ok=True)
    claim = root/'submit-claim.json'
    if claim.exists():
        receipt = root/'submit-receipt.json'
        print(json.dumps(dict(status='EXISTING_ATTEMPT_NO_REPLAY', root=str(root),
            receipt=json.loads(receipt.read_text()) if receipt.exists() else None)), flush=True)
        return
    service = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-service2-20261021')
    libraries = list((service/'env/lib/python3.10/site-packages/nvidia').glob('*/lib'))
    libraries += [service/'env/lib/python3.10/site-packages/torch/lib']
    env = dict(PATH=str(service/'env/bin')+':/usr/local/bin:/usr/bin:/bin',
        PYTHONPATH=str(service/'retriever-gpu-lib')+':'+str(source),
        LD_LIBRARY_PATH=':'.join(map(str,libraries)), OMP_NUM_THREADS='8',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        XDG_CACHE_HOME=str(root/'cache'), HF_HOME=str(root/'cache/hf'))
    command = ['env', *[k+'='+v for k,v in env.items()], str(service/'env/bin/python'),
        '-B', str(source/'reproduction/operations/easy7b_20261021/benchmark_retrieval_batching.py'),
        '--raw-pool-file', request['raw_pool_file'], '--index', request['index'],
        '--encoder', request['encoder'], '--output', str(root/'result.json')]
    if request.get('singleton_encoder', False):
        command.append('--singleton-encoder')
    if request.get('cases') is not None:
        command.extend(['--cases', str(request['cases'])])
    argv = ['sbatch','--parsable','--partition=emergency_acd','--qos=formal-user',
        '--account=jxu957','--nodes=1','--ntasks=1','--gres=gpu:2','--exclude=ACD1-13',
        '--cpus-per-task=16','--mem=128G','--time=00:20:00',
        '--job-name=easy7b-retrieval-batch-probe','--export=NONE','--chdir='+str(source),
        '--output='+str(root/'slurm-%j.log'),'--error='+str(root/'slurm-%j.log'),
        '--wrap=exec '+shlex.join(command)]
    with claim.open('x') as stream:
        json.dump(dict(request,command=argv,created_unix=time.time(),
            production_job_untouched='750029',production_service_untouched='745198',
            reads_existing_assets_only=True,actor_loaded=False,corpus_loaded=False),stream,indent=2)
        stream.flush();os.fsync(stream.fileno())
    result = subprocess.run(argv,capture_output=True,text=True,timeout=30)
    receipt = dict(root=str(root),source_commit=request['commit'],returncode=result.returncode,
        stdout=result.stdout,stderr=result.stderr,checked_unix=time.time(),production_modified=False)
    if result.returncode == 0:
        job = result.stdout.strip().split(';')[0]
        if not job.isdecimal():
            raise RuntimeError('Unknown submission response; inspect existing claim')
        receipt['job'] = job
        with (root/'job.id').open('x') as stream:
            stream.write(job+'\n');stream.flush();os.fsync(stream.fileno())
    (root/'submit-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
