import io,sys,json
sys.stdin=io.StringIO('{"service_base": "/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-batch-wait16-service-r1-20261023", "pair_root": "/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/candidate-retrieval-wait16-r1-20261023/attempt-r1", "production_base": "/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023"}')
"""One readonly small-file/log snapshot of a private retrieval/candidate pair.

JSON stdin: {service_base, pair_root, production_base}. No HTTP, CUDA, srun,
model hashes, tensor loads, polling loop, or writes are performed.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys
import time

MAX_JSON = 4 * 1024**2
ERROR = re.compile(r'Traceback|CUDA.*(?:error|out of memory)|RuntimeError:|'
                   r'AssertionError:|ValueError:|NCCL.*(?:error|failed)|FAILED', re.I)
ANSI = re.compile(r'\x1b\[[0-9;]*m')
DROP = {'queries', 'query', 'prompts', 'prompt', 'tokens', 'input_ids', 'attention_mask',
        'position_ids', 'command', 'argv', 'env', 'environment', 'password', 'credentials',
        'secret', 'source_sha256', 'source_code_sha256', 'files', 'input_fingerprints',
        'trajectory_ids', 'question_ids', 'selected_ordinals', 'actual_request_token_ids'}
RECORDS = ('submit-receipt.json', 'submission.json', 'runner-submit-receipt.json',
    'runner-submission.json', 'pair-submit-receipt.json', 'pair-submission.json',
    'launch.json', 'progress.json', 'runner-progress.json', 'runner-status.json',
    'runner-exit.json', 'exit.json', 'result.json', 'failure.json')
SERVICE_KEYS = ('phase', 'ready', 'ready_unix', 'checked_unix', 'run_id', 'service_run_id',
    'service_job_id', 'source_commit', 'hostname', 'endpoint', 'physical_gpu_count',
    'logical_shards', 'logical_to_physical_gpu', 'encoder_dtype', 'ntotal', 'dimension',
    'exact_query_cache_enabled', 'request_rebatching_enabled', 'singleton_e5',
    'faiss_batch_size', 'faiss_max_wait_ms', 'backend_seconds_scope', 'statistics',
    'error_type', 'error', 'rss_bytes')


def small(value, depth=0):
    if depth > 5:
        return '<nested summary omitted>'
    if isinstance(value, dict):
        return {key: small(item, depth + 1) for key, item in list(value.items())[:48]
                if key not in DROP and not re.search(r'password|credential|secret', key, re.I)}
    if isinstance(value, list):
        rows = [small(item, depth + 1) for item in value[:6]]
        return rows if len(value) <= 6 else dict(count=len(value), first=rows)
    if isinstance(value, str):
        return value[:1200]
    return value


def metadata(path):
    value = path.stat()
    return dict(path=str(path), bytes=value.st_size, mtime=value.st_mtime,
                mtime_ns=value.st_mtime_ns)


def document(path):
    if not path.is_file():
        return dict(path=str(path), present=False), None
    try:
        record = dict(metadata(path), present=True)
        if record['bytes'] > MAX_JSON:
            return dict(record, observation_error='Small JSON size limit'), None
        value = json.loads(path.read_text())
        return dict(record, summary=small(value)), value
    except (OSError, ValueError) as error:
        return dict(path=str(path), present=True, observation_error=type(error).__name__), None


def tail(path, read_bytes=4096):
    try:
        record = metadata(path)
        with path.open('rb') as stream:
            stream.seek(max(0, record['bytes'] - read_bytes))
            raw = ANSI.sub('', stream.read(read_bytes).decode('utf-8', errors='replace'))
        lines = raw.splitlines()
        mentions = []
        for line in lines:
            matches = list(re.finditer(r'\bstep:\s*(\d+)|"(?:logical_update|step|outer)"\s*:\s*(\d+)', line))
            if matches:
                mentions.append(dict(outer=int(next(x for x in matches[-1].groups() if x is not None)),
                                     line_end=line[-250:]))
        return dict(record, tail='\n'.join(line[-700:] for line in lines[-18:])[-4096:],
            errors=[line[-1000:] for line in lines if ERROR.search(line)][-6:],
            last_outer_mentions=mentions[-3:])
    except OSError as error:
        return dict(path=str(path), present=False, observation_error=type(error).__name__)


def job_file(root):
    try:
        value = (root / 'job.id').read_text()[:128].strip()
        return value if value.isdecimal() else None
    except OSError:
        return None


def jobs_in(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('job', 'job_id', 'slurm_job_id', 'service_job_id', 'pair_job') and str(item).isdecimal():
                found.add(str(item))
            elif isinstance(item, (dict, list)):
                found.update(jobs_in(item))
    elif isinstance(value, list):
        for item in value:
            found.update(jobs_in(item))
    return found


def log_paths(root, limit=2):
    paths = set(root.glob('*.log')) | set((root / 'logs').glob('*.log'))
    return sorted((p for p in paths if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


def section(root, jobs, names=RECORDS, log_limit=2):
    result = dict(root=str(root), job=job_file(root), documents={})
    if result['job']:
        jobs.add(result['job'])
    for name in names:
        for path in (root / name, root / 'preparation' / name):
            if path.is_file():
                record, value = document(path)
                result['documents'][str(path.relative_to(root))] = record
                jobs.update(jobs_in(value))
    result['logs'] = [tail(path) for path in log_paths(root, log_limit)]
    return result


def variant_roots(pair):
    """Immediate variants, or one level below conventional variant containers."""
    result = []
    if not pair.is_dir():
        return result
    ignored = {'source', 'repo', 'env', 'cache', 'raw', 'logs', 'preparation', 'tmp'}
    for child in sorted(pair.iterdir()):
        if not child.is_dir() or child.name in ignored:
            continue
        choices = list(child.iterdir()) if child.name in ('variants', 'runs', 'results') else [child]
        for candidate in choices:
            if candidate.is_dir() and (any((candidate / name).is_file() for name in RECORDS) or
                                      any(candidate.glob('chunk-*.json'))):
                result.append(candidate)
    return result[:12]


def slurm(jobs):
    requested = sorted(jobs, key=int)[:16]
    result = dict(requested_job_ids=requested, owner=pwd.getpwuid(os.getuid()).pw_name,
                  actual_owned_job_ids=[], calls=[])
    if not requested:
        return result
    commands = [
        ['squeue', '-h', '-j', ','.join(requested), '-o', '%i|%u|%T|%N|%M|%R'],
        ['sacct', '-n', '-X', '-j', ','.join(requested), '-P',
         '--format=JobIDRaw,User,State,ExitCode,Elapsed,NodeList'],
    ]
    actual = set()
    for command in commands:
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=15)
            rows = []
            for line in done.stdout.splitlines():
                fields = line.strip().split('|')
                if len(fields) >= 3 and fields[0] in requested and fields[1] == result['owner']:
                    rows.append(line.strip()); actual.add(fields[0])
            result['calls'].append(dict(command=command, returncode=done.returncode,
                own_rows=rows[:32], stderr=done.stderr[-1200:]))
        except (OSError, subprocess.TimeoutExpired) as error:
            result['calls'].append(dict(command=command, observation_error=type(error).__name__))
    result['actual_owned_job_ids'] = sorted(actual, key=int)
    return result


def observe(request):
    service, pair, production = (Path(request[key]) for key in ('service_base', 'pair_root', 'production_base'))
    jobs = set()
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(), epoch=time.time(), read_only=True,
        tensor_contents_read=False, model_hashed=False, http_requests=0, gpu_calls=0, srun_calls=0,
        training_modified=False, automatic_retry=False,
        growth_scope='Single snapshot; compare completed chunks/questions and file bytes/mtime across observations')
    result['service'] = section(service, jobs, names=('service-submission.json',), log_limit=1)
    for name in ('status.json', 'logs/service-ready.json'):
        record, value = document(service / name)
        if isinstance(value, dict):
            record['summary'] = {key: small(value[key]) for key in SERVICE_KEYS if key in value}
        result['service']['documents'][name] = record
        jobs.update(jobs_in(value))
    result['pair'] = section(pair, jobs, log_limit=4)
    if (pair / 'runs').is_dir():
        result['pair']['runner'] = section(pair / 'runs', jobs, log_limit=0)
    variants = []
    for root in variant_roots(pair):
        variant = section(root, jobs, log_limit=1)
        candidate = root / 'candidate' if (root / 'candidate').is_dir() else root
        if candidate != root:
            variant['candidate'] = section(candidate, jobs, log_limit=0)
        variant['chunks'] = [document(path)[0] for path in sorted(candidate.glob('chunk-*.json'))[:5]]
        # Metadata only: never open or deserialize a raw candidate tensor.
        variant['raw_chunk_metadata'] = [metadata(path) for path in sorted((candidate / 'raw').glob('chunk-*.pt'))[:6]]
        variants.append(variant)
    result['pair']['variants'] = variants
    production_job = job_file(production)
    if production_job:
        jobs.add(production_job)
    sealed, _ = document(production / 'preparation/rolling-checkpoints/latest.json')
    train = production / 'logs/search-g1-common8-training.log'
    if not train.is_file():
        train = production / 'logs/trainer.log'
    result['production'] = dict(base=str(production), job=production_job, latest_sealed_pointer=sealed,
        train_log=tail(train, read_bytes=64 * 1024), observation_is_training_gate=False)
    result['slurm'] = slurm(jobs)
    return result


if __name__ == '__main__':
    print(json.dumps(observe(json.load(sys.stdin)), sort_keys=True, allow_nan=False), flush=True)
