"""One read-only node10 cohort observation; send this source in one transport.

Reads native state, /proc identities/owned lock FDs and bounded JSON metadata.
No HTTP, GPU calls, signals, writes, trajectory reads or workload launches.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import time

RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
BASE = Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
PHYSICAL = Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
SLOTS = ('gpu1-tp1', 'gpu2-tp1', 'gpu3-tp1', 'gpu5-tp1', 'tp4-0-4-6-7')
MAX_JSON_BYTES = 512 * 1024


def path(recorded):
    recorded = Path(os.path.normpath(recorded))
    for root in (PHYSICAL, BASE):
        if recorded.is_relative_to(root):
            return PHYSICAL / recorded.relative_to(root)
    raise ValueError('Metadata path is outside the fixed same-run NAS root')


def read(where):
    where = Path(where)
    try:
        size = where.stat().st_size
        if size > MAX_JSON_BYTES:
            return dict(read_error='metadata_size_limit', path=str(where), bytes=size)
        return json.loads(where.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        return dict(read_error=type(error).__name__, path=str(where), error=str(error))


def file_metadata(where):
    where = Path(where)
    value = read(where)
    if value is None:
        return None
    try:
        info = where.stat()
        return dict(path=str(where), bytes=info.st_size, mtime_ns=info.st_mtime_ns, value=value)
    except FileNotFoundError:
        return dict(path=str(where), disappeared_during_read=True)


def process(pid):
    try:
        root = Path('/proc') / str(pid)
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, birth=int(fields[19]), state=fields[0], cgroup=(root / 'cgroup').read_text())
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return dict(pid=pid, present=False)


def active_phases(allocation, children):
    result, seen = [], set()
    group = allocation.get('native_cgroup')
    if not group:
        return result
    for pid in children.get('own_process_groups', []):
        actual = process(pid)
        if not any(line.split(':', 2)[-1] == group for line in actual.get('cgroup', '').splitlines()):
            continue
        try:
            descriptors = list((Path('/proc') / str(pid) / 'fd').iterdir())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for fd in descriptors:
            try:
                target = os.readlink(fd)
                if '/pool/batch-locks/' not in target:
                    continue
                prefix, relative = target.split('/pool/batch-locks/', 1)
                tail = Path(relative)
                if not prefix.endswith('/shaoshitong/' + PHYSICAL.name) or len(tail.parts) != 2 or tail.suffix != '.lock':
                    continue
                # An opened FD alone is not a held batch claim.
                info = (Path('/proc') / str(pid) / 'fdinfo' / fd.name).read_text()
                if not any(line.startswith('lock:') and 'POSIX' in line and 'WRITE' in line
                           and str(pid) in line.split() for line in info.splitlines()):
                    continue
                key = (pid, str(tail))
                if key in seen:
                    continue
                seen.add(key)
                batch = PHYSICAL / 'sampling' / tail.parent / 'batches' / tail.stem
                phases = {}
                for phase in ('prefix', 'full'):
                    root = batch / 'outers/outer-000001' / phase
                    phases[phase] = dict(rank_progress=file_metadata(root / 'rank-0-progress.json'),
                        rank_complete=file_metadata(root / 'rank-0-complete.json'), complete=file_metadata(root / 'complete.json'))
                result.append(dict(child=actual, fd=int(fd.name), lock_target=target, shard=str(tail.parent),
                    batch=tail.stem, whole_complete=file_metadata(batch / 'complete.json'), phases=phases))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    return result


def attempt(job):
    allocation, children = read(job / 'allocation.json') or {}, read(job / 'children.json') or {}
    return dict(directory=str(job), allocation=allocation, children=children,
        child_actual=[process(pid) for pid in children.get('own_process_groups', [])],
        progress=file_metadata(job / 'progress.json'), failure=file_metadata(job / 'failure.json'),
        complete=file_metadata(job / 'complete.json'), sampler_exit=file_metadata(job / 'sampler-exit.json'),
        model_initialized=[file_metadata(p) for p in sorted((job / 'sampler-launches').glob('launch-*/model-initialized.json'))],
        launch_failures=[file_metadata(p) for p in sorted((job / 'sampler-launches').glob('launch-*/failure.json'))],
        active_phases=active_phases(allocation, children))


def sealed_counts():
    count = questions = trajectories = 0
    latest, errors = [], []
    for receipt in sorted((PHYSICAL / 'sampling').glob('shard-*/batches/batch-*/complete.json')):
        value = read(receipt)
        if not isinstance(value, dict) or value.get('run_id') != RUN_ID or not all(
                type(value.get(key)) is int and value[key] >= 0 for key in ('questions', 'trajectories')) \
                or value['trajectories'] != 5 * value['questions']:
            errors.append(str(receipt))
            continue
        count += 1; questions += value['questions']; trajectories += value['trajectories']
        latest.append(dict(path=str(receipt), batch=value.get('batch'), epoch=value.get('epoch'),
            questions=value['questions'], trajectories=value['trajectories']))
    return dict(sealed_batches=count, questions=questions, trajectories=trajectories,
        latest=sorted(latest, key=lambda row: row['epoch'] or 0)[-8:], invalid_or_unreadable=errors,
        metadata_observation_only=True, independent_full_audit_performed=False)


def main():
    assert socket.gethostname().split('.')[0] == 'h100-10', 'This read is limited to node10'
    units = {slot: 'sst-q7b-g5-primary10-' + slot + '-v2-' + RUN_ID[:12] + '.service' for slot in SLOTS}
    units['cpu'] = 'sst-q7b-g5-cpu-relay10-v1-' + RUN_ID[:12] + '.service'
    properties = ('Id', 'LoadState', 'MainPID', 'ActiveState', 'SubState', 'ControlGroup', 'Result',
                  'ExecMainCode', 'ExecMainStatus', 'ExecMainStartTimestamp', 'ExecMainExitTimestamp')
    native = subprocess.run(['systemctl', 'show', *units.values(), *[item for prop in properties for item in ('-p', prop)]],
        text=True, capture_output=True, timeout=20)
    observed = {}
    for block in native.stdout.strip().split('\n\n'):
        props = dict(line.split('=', 1) for line in block.splitlines() if '=' in line)
        if props.get('Id'):
            observed[props['Id']] = props
    result = dict(status='READ_ONLY_NODE10_PRIMARY_V2_COHORT', epoch=time.time(), run_id=RUN_ID,
        node='h100-10', native_query_returncode=native.returncode, native_query_stderr=native.stderr, units={})
    for slot, unit in units.items():
        directory = 'legacy-relay10-independent-v1' if slot == 'cpu' else 'relay-primary10-' + slot + '-v2'
        root = PHYSICAL / 'preparation' / directory
        receipt = read(root / ('submit-receipt.json' if slot == 'cpu' else 'native-submit-receipt.json')) or {}
        props = observed.get(unit, {})
        pid = props.get('MainPID', '0')
        historical = pid == '0'
        if historical:
            pid = ((receipt.get('observed') or {}).get('properties') or {}).get('MainPID', '0')
        pid = int(pid)
        prefix = 'job-legacy-relay-owner-' if slot == 'cpu' else 'job-relay10-primary-'
        jobs = sorted((PHYSICAL / 'pool/jobs' / (prefix + str(pid))).glob('attempt-*')) if pid else []
        row = dict(unit=unit, native=props, controller_pid=pid, historical_pid_from_immutable_receipt=historical,
            controller_actual=process(pid) if pid else None, submit_receipt=receipt, attempts=[attempt(job) for job in jobs])
        if slot == 'cpu':
            row['owner_ready'] = file_metadata(root / 'owner-ready.json')
            row['cpu_metadata'] = [dict(directory=str(job), **{name[:-5].replace('-', '_'): file_metadata(job / name)
                for name in ('backpressure-stats.json', 'health.json', 'drain-owner.json', 'drain-request.json', 'draining.json')}) for job in jobs]
        result['units'][slot] = row
    result['global_sealed'] = sealed_counts()
    print(json.dumps(result))


if __name__ == '__main__': main()
