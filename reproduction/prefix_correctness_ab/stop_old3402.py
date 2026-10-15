"""Latest human-authorized stop of exactly old job3402; never submits a job.

Execute on h100-8 through the existing management transport. Large logs and
all model artifacts remain remote. Existing intent means inspect, not replay.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

BASE = Path('/mnt/data/shaoshitong/search-g1-qprobe3-binary10-c5fc096-20261015')
JOB = 3402
RUN = 'f77c3245dcfe49a88a5a99b1dec161e6'
PREFIX = BASE / 'preparation/prefix5-human-stop'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def once(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def same_birth(row):
    path = Path('/proc') / str(row['pid']) / 'stat'
    try:
        return path.read_text().rsplit(')', 1)[1].split()[19] == str(row['start_ticks'])
    except FileNotFoundError:
        return False


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def native():
    result = subprocess.run(['scontrol', 'show', 'job', '-o', str(JOB)],
                            text=True, capture_output=True, timeout=20)
    return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                fields=dict(re.findall(r'(\w+)=(\S+)', result.stdout)))


def identities():
    ledger = read(BASE / 'logs/process-ledger.json')
    return [row for job in ledger['jobs'].values() for row in job['ledger'].values()]


def request():
    assert read(BASE / 'logs/run-manifest.json')['run_id'] == RUN
    path = Path(str(PREFIX) + '-intent.json')
    if path.exists():
        return dict(existing_intent=True, action='Observe same3402; never replay stop',
                    intent_path=str(path))
    state = native()
    f = state['fields']
    assert state['returncode'] == 0 and f['JobId'] == str(JOB)
    assert f['JobState'] == 'RUNNING' and f['NodeList'] == 'h100-8'
    assert f['Command'] == str(BASE / 'preparation/launch-h1008.sh')
    assert f['StdOut'] == f['StdErr'] == str(BASE / 'logs/native-3402.log')
    births = identities()
    assert any(row['pid'] == 1544802 and row['start_ticks'] == 2301002451
               and same_birth(row) for row in births), 'Original driver identity differs'
    logs = [dict(path=str(p), inode=p.stat().st_ino, device=p.stat().st_dev)
            for p in (BASE / 'logs').rglob('*') if p.is_file() and not p.is_symlink()]
    once(path, dict(utc=utc(), job_id=JOB, run_id=RUN, native_before=state,
        human_quotes=['停掉现在跑的吧', '理解一下最新的思路，整理一下方案，然后之前的实验停掉吧'],
        scope='Discontinue only old3402; no automatic restore or replacement200. '
              'Organize offline prefix5 correctness A/B; preserve caches/artifacts/foreign tasks.',
        human_original_timestamp='not exposed; latest current-turn explicit instruction',
        owned_births=births, original_log_inodes=logs))
    result = subprocess.run(['scancel', '--signal=TERM', '--batch', str(JOB)],
                            capture_output=True, text=True, timeout=30)
    receipt = dict(utc=utc(), returncode=result.returncode,
                   stdout=result.stdout, stderr=result.stderr, job_id=JOB)
    once(Path(str(PREFIX) + '-signal-receipt.json'), receipt)
    assert result.returncode == 0, result.stderr
    return dict(stop_requested=True, job_id=JOB, foreign_tasks_untouched=True,
                no_submission=True, utc=utc())


def observe():
    assert read(BASE / 'logs/run-manifest.json')['run_id'] == RUN
    intent = read(Path(str(PREFIX) + '-intent.json'))
    births = list({(r['pid'], r['start_ticks']): r
                   for r in intent['owned_births'] + identities()}.values())
    alive = [r for r in births if same_birth(r)]
    state = native()
    f = state['fields']
    terminal = state['returncode'] == 0 and f.get('JobId') == str(JOB) and f.get('JobState') in (
        'CANCELLED', 'FAILED', 'COMPLETED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY')
    if terminal:
        path = Path(str(PREFIX) + '-native-terminal.json')
        if not path.exists():
            once(path, dict(utc=utc(), native=state))
    terminal_path = Path(str(PREFIX) + '-native-terminal.json')
    saved_terminal = read(terminal_path) if terminal_path.exists() else None
    groups = [str(p) for p in Path('/sys/fs/cgroup').glob('**/job_' + str(JOB))]
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'],
                         capture_output=True, text=True, timeout=20)
    assert gpu.returncode == 0, gpu.stderr
    owned_pids = {r['pid'] for r in births}
    owned_gpu = [line for line in gpu.stdout.splitlines()
                 if line.strip() and int(line.split(',')[0]) in owned_pids]
    cleanup_path = BASE / 'logs/cleanup-status.json'
    cleanup = read(cleanup_path) if cleanup_path.exists() else None
    released = not alive and not groups and not owned_gpu
    result = dict(utc=utc(), job_id=JOB, run_id=RUN, native=state,
        saved_native_terminal=saved_terminal, original_owned_birth_count=len(births),
        live_owned_births=len(alive), owned_cgroups=groups, owned_gpu_applications=owned_gpu,
        cleanup=cleanup, resources_released=released, foreign_tasks_untouched=True,
        no_new_job_submitted=True, manual_stop=True, experiment_completed=False)
    if released and saved_terminal and (cleanup is None or cleanup.get('cleaned') is True):
        hash_path = Path(str(PREFIX) + '-terminal-log-shas.json')
        if not hash_path.exists():
            hashes = []
            for row in intent['original_log_inodes']:
                path = Path(row['path'])
                if not path.exists() and path.name.endswith('.tmp'):
                    hashes.append(dict(row, transient_atomic_temporary_removed=True,
                                       full_sha_performed=False))
                    continue
                st = path.stat()
                same = (st.st_ino, st.st_dev) == (row['inode'], row['device'])
                if path.suffix in ('.log', '.out', '.err'):
                    assert same, str(path)
                hashes.append(dict(row, terminal_inode=st.st_ino, terminal_device=st.st_dev,
                                   same_original_inode=same, bytes=st.st_size, full_sha256=sha(path)))
            once(hash_path, hashes)
        result.update(terminal_original_log_manifest=str(hash_path),
                      terminal_original_log_manifest_sha256=sha(hash_path),
                      original_log_files=sum('full_sha256' in r for r in read(hash_path)),
                      transient_atomic_temporary_removed=[r['path'] for r in read(hash_path)
                          if r.get('transient_atomic_temporary_removed')], stopped_verified=True)
        receipt = Path(str(PREFIX) + '-terminal-receipt.json')
        if not receipt.exists():
            once(receipt, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['request', 'observe'])
    args = parser.parse_args()
    if args.phase == 'request':
        print(json.dumps(request(), ensure_ascii=False))
    else:
        observe()
