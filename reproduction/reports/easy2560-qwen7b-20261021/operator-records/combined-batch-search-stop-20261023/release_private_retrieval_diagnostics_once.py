"""Release only the two completed experiments' private diagnostic GPU services.

Prepared locally; run once from the published operator copy as jxu957 on HPC3.
No tensors, CUDA, FAISS, HTTP calls, source edits, file removals, or resubmits.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/data/user/jxu957/experiments')
SERVICES = (
    dict(job='750396', base='easy2560-corollout-joint4-retrieval-batch-service-r1-20261023',
         commit='0f48a49fed951870c7f45dd7fb1b37d3c6a816c1',
         run_id='c95ec3253c5740a7ab0ca404024a935f',
         endpoint='http://ACD1-4:18022/retrieve', wait_ms=2),
    dict(job='750432', base='easy2560-corollout-joint4-retrieval-batch-wait16-service-r1-20261023',
         commit='2303dd444e077ac28d65bef95e427ff7de9fcb38',
         run_id='7e798ae2f8de46bab2ab0a3bfa3ed0b7',
         endpoint='http://ACD1-7:18024/retrieve', wait_ms=16),
)
CONSUMERS = ('750423', '750433', '750449', '750454', '750432.0')
PROTECTED = ('750029', '745198', '750471')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def read(path):
    if not path.is_file() or path.stat().st_size > 4*1024**2:
        raise ValueError('Required small JSON absent/oversized: '+str(path))
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def accounting():
    jobs = [item['job'] for item in SERVICES] + list(CONSUMERS)
    result = subprocess.run(['sacct', '-n', '-P', '-j', ','.join(dict.fromkeys(jobs)),
        '--format=JobIDRaw,User,State,ExitCode,NodeList,JobName%64'],
        capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError('sacct failed; no service cancellation attempted: '+result.stderr[-1000:])
    rows = []
    for line in result.stdout.splitlines():
        columns = line.strip().split('|')
        if len(columns) < 6:
            continue
        rows.append(dict(zip(('job', 'user', 'state', 'exit_code', 'nodes', 'name'), columns[:6])))
    return rows, result.stdout


def state(row):
    return row['state'].split()[0].rstrip('+') if row['state'].split() else ''


def exact_row(rows, job):
    matches = [row for row in rows if row['job'] == job]
    if len(matches) != 1:
        raise ValueError('Missing/ambiguous actual accounting row: '+job)
    row = matches[0]
    if row['user'] != 'jxu957' and not (job == '750432.0' and row['user'] == ''):
        raise ValueError('Unexpected accounting owner: '+job)
    return row


def service_snapshot(spec, rows):
    base = ROOT/spec['base']; source = base/'source'/spec['commit']
    if base.resolve() != base or (base/'job.id').read_text().strip() != spec['job']:
        raise ValueError('Private diagnostic base/job binding differs')
    publication_path = source/'source-publication.json'; publication = read(publication_path)
    ready_path = base/'logs/service-ready.json'; ready = read(ready_path)
    if (publication.get('source_commit') != spec['commit'] or
            publication.get('fresh_source_all_files_sha_verified') is not True or
            ready.get('ready') is not True or ready.get('source_commit') != spec['commit'] or
            ready.get('service_job_id') != spec['job'] or ready.get('slurm_job_id') != spec['job'] or
            ready.get('service_run_id') != spec['run_id'] or ready.get('endpoint') != spec['endpoint'] or
            ready.get('physical_gpu_count') != 2 or ready.get('logical_shards') != 4 or
            ready.get('logical_to_physical_gpu') != [0,1,0,1] or
            ready.get('singleton_e5') is not True or ready.get('faiss_batch_size') != 16 or
            ready.get('faiss_max_wait_ms') != spec['wait_ms']):
        raise ValueError('Private diagnostic publication/READY identity differs: '+spec['job'])
    source_hashes = {}
    for name, expected in ready['source_code_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unexpected source member')
        actual = sha(source/relative)
        if actual != expected or publication['files'][name]['sha256'] != actual:
            raise ValueError('Private diagnostic small source SHA differs: '+name)
        source_hashes[name] = actual
    row = exact_row(rows, spec['job'])
    if row['name'] != 'easy7b-private-retrieval-batch':
        raise ValueError('Unexpected private diagnostic Slurm job name')
    preserved = []
    for folder in (base/'logs', base/'preparation'):
        for path in folder.glob('*failure*.json'):
            preserved.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
    status_path = base/'status.json'
    status = read(status_path) if status_path.exists() else None
    return dict(base=str(base), job=spec['job'], source_commit=spec['commit'],
        source_publication_path=str(publication_path), source_publication_sha256=sha(publication_path),
        published_source_file_count=len(publication['files']), small_source_sha256=source_hashes,
        service_ready_path=str(ready_path), service_ready_sha256=sha(ready_path), service_ready=ready,
        existing_status=status, accounting=row, failure_history_preserved=preserved,
        checked_unix=time.time(), reason='Private candidate/replay diagnostics finished; production replacement service READY',
        protected_jobs=list(PROTECTED), source_cache_assets_models_results_deleted=False)


def main():
    if os.getuid() != 14090:
        raise ValueError('Owned jxu957 HPC3 operator required')
    if {item['job'] for item in SERVICES} != {'750396', '750432'}:
        raise ValueError('Release scope differs')
    claims = [ROOT/item['base']/'preparation/diagnostic-release-claim.json' for item in SERVICES]
    if any(path.exists() for path in claims):
        print(json.dumps(dict(status='EXISTING_RELEASE_ATTEMPT_NO_REPLAY', claims=list(map(str, claims)))))
        return
    rows, raw = accounting()
    consumers = [exact_row(rows, job) for job in CONSUMERS]
    if any(state(row) not in TERMINAL for row in consumers):
        raise ValueError('An actual diagnostic consumer is not terminal; no cancellation attempted')
    snapshots = [service_snapshot(spec, rows) for spec in SERVICES]
    # Save both identities and all actual consumer exit states before cancellation.
    for snapshot, claim in zip(snapshots, claims):
        snapshot.update(actual_consumers=consumers, accounting_stdout=raw)
        publish(claim.with_name('diagnostic-release-closing.json'), snapshot)
        publish(claim, dict(job=snapshot['job'], source_commit=snapshot['source_commit'],
            closing_sha256=sha(claim.with_name('diagnostic-release-closing.json')),
            command=['scancel', snapshot['job']], checked_unix=time.time(), automatic_replay=False))
    results = []
    for snapshot, claim in zip(snapshots, claims):
        job = snapshot['job']
        if job in PROTECTED or job not in ('750396', '750432'):
            raise ValueError('Cancellation scope differs')
        if state(snapshot['accounting']) in TERMINAL:
            result = dict(job=job, status='ALREADY_TERMINAL_NO_SCANCEL', returncode=0)
        else:
            call = subprocess.run(['scancel', job], capture_output=True, text=True, timeout=20)
            result = dict(job=job, status='SCANCEL_RETURNED', returncode=call.returncode,
                          stdout=call.stdout, stderr=call.stderr)
        result.update(checked_unix=time.time(), actual_terminal_requires_followup=True,
                      protected_jobs=list(PROTECTED), files_deleted=False)
        publish(claim.with_name('diagnostic-release-result.json'), result); results.append(result)
    print(json.dumps(dict(status='PRIVATE_DIAGNOSTIC_RELEASE_COMMANDS_RECORDED', results=results)))


if __name__ == '__main__':
    main()
