"""Read-only compact progress observer for the immutable CP1 dense recovery."""
import json
import re
import subprocess
import time
from pathlib import Path

BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021')
SERVICE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-service2-20261021')


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return {'observation_error': str(error)}


def tail(path, limit=14000):
    try:
        stat = path.stat()
        with path.open('rb') as stream:
            stream.seek(max(0, stat.st_size-limit))
            content = stream.read().decode(errors='replace')
        return dict(bytes=stat.st_size, mtime=stat.st_mtime, tail=content)
    except OSError as error:
        return {'observation_error': str(error)}


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=12)
        return dict(rc=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired:
        return dict(observation_timeout=True)


def main():
    job = (BASE/'job.id').read_text().strip()
    result = dict(epoch=time.time(), base=str(BASE), job=job,
                  read_only=True, training_runtime_edited=False)
    result['queue'] = command(['squeue', '-j', job+',745198', '-h', '-o', '%i|%T|%N|%R|%S|%L'])
    result['accounting'] = command(['sacct', '-j', job+',745198', '-n', '-P', '-o', 'JobIDRaw,State,ExitCode,Elapsed,NodeList'])
    for name in ['logs/pipeline-status.json', 'logs/full-state-restored-workers.json',
                 'logs/full-state-restored-driver.json', 'logs/easy2560-restored-stream-before-next-outer.json',
                 'preparation/rolling-checkpoints/latest.json', 'logs/cleanup-status.json']:
        if (BASE/name).is_file():
            result[name] = read_json(BASE/name)
    result['ledgers'] = {}
    for name in ['actual-updates.jsonl', 'actual-attempts.jsonl'] + [f'optimizer-rank-{rank}.jsonl' for rank in range(4)]:
        path = BASE/'logs'/name
        if path.is_file():
            result['ledgers'][name] = tail(path, 18000)
    result['candidate_pools'] = []
    pools = BASE/'artifacts/mixed-candidate-pools'
    if pools.is_dir():
        for path in sorted(pools.iterdir(), key=lambda p:p.stat().st_mtime)[-3:]:
            value = read_json(path/'progress.json')
            result['candidate_pools'].append(dict(name=path.name, step=value.get('step'),
                completed_questions=len(value.get('complete_ordinals', [])),
                reserved_questions=len(value.get('reserved', [])), source_consumed=value.get('source_consumed'),
                saved_chunk_files=len(list(path.glob('[0-9]*.pt'))),
                progress_error=value.get('observation_error')))
    result['training_log'] = tail(BASE/'logs/search-g1-common8-training.log', 20000)
    result['trainer_log'] = tail(BASE/'logs/trainer.log', 5000)
    gpu = BASE/'logs/gpu-samples-232f20e112ac4a978c5aba52a0367195.jsonl'
    result['gpu_sample_tail'] = tail(gpu, 3000)
    try:
        import requests
        ready = read_json(SERVICE/'logs/service-ready.json')
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(ready['endpoint'].replace('/retrieve','/health'), timeout=8)
            health = response.json()
        result['service_health'] = {key:health.get(key) for key in ['ok','service_job_id','hostname','physical_gpu_count','statistics']}
    except Exception as error:
        result['service_observation_error'] = str(error)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
