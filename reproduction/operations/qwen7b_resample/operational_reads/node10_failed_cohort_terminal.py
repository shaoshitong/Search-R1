"""Read-only terminal proof for four actual native node10 failures, not a stop.

Run once after publication on physical NAS10. Only this run's terminal proof
directory is written. No signal, service mutation, model or trajectory read.
Historical births are reported only with their real source; missing births
remain missing and both observations instead require complete PID absence.
"""
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time

BASE = Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
RECORDED = Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
STATUS = 'OWN_NODE10_NATIVE_FAILURES_ORIGINAL_SHA_RELEASE_PASS'
ROOT = BASE / 'preparation/node10-native-failures-20261021-v1'
COHORT = (
    (2371322, 'consumer10-v5', 'relay10', 'relay-consumer10', 'relay-consumer10-control-v1', 'native-v5', '1958624331'),
    (2452973, 'consumer10-parallel-v1', 'relay10', 'relay-consumer10-parallel', 'relay-consumer10-parallel-v1', 'native', '1958806278'),
    (2830315, 'consumer10-tp1-gpu1-v1', 'relay10-tp1', 'relay-consumer10-tp1', 'relay-consumer10-tp1-gpu1-v1', 'native', None),
    (2830845, 'consumer10-tp1-gpu2-v1', 'relay10-tp1', 'relay-consumer10-tp1', 'relay-consumer10-tp1-gpu2-v1', 'native', None),
)
UNITS = {'sst-q7b-g5-' + row[1] + '-' + RUN_ID[:12] + '.service' for row in COHORT}
FORBIDDEN = {'sampling', 'states', 'features', 'cache', 'tmp', 'retrieval-cache', 'assets', 'code'}


def read(path):
    return json.loads(Path(path).read_text())


def physical(path):
    path = Path(os.path.normpath(path))
    if path.is_relative_to(RECORDED):
        path = BASE / path.relative_to(RECORDED)
    assert path.is_relative_to(BASE)
    return path


def identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def originals(folders):
    result = set()
    for folder in folders:
        assert folder.is_dir() and not folder.is_symlink()
        for path in folder.rglob('*'):
            if any(part in FORBIDDEN for part in path.relative_to(folder).parts):
                continue
            if path.is_file() and not path.is_symlink() and path.suffix in ('.json', '.log', '.txt', '.out', '.err', '.py', '.sh'):
                assert 'secret' not in path.name.lower() and 'token' not in path.name.lower()
                result.add(path)
    return sorted(result)


def original(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        assert stat.S_ISREG(before.st_mode) and identity(before) == identity(path.stat())
        digest, blocks = hashlib.sha256(), []
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block);blocks.append(block)
        assert identity(before) == identity(os.fstat(stream.fileno())) == identity(path.stat())
    return dict(path=str(path), bytes=before.st_size, sha256=digest.hexdigest(), device=before.st_dev,
                inode=before.st_ino, mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns), b''.join(blocks)


def stable(records):
    for row in records:
        assert identity(Path(row['path']).stat()) == (row['device'], row['inode'], row['bytes'], row['mtime_ns'], row['ctime_ns'])


def collect_births(value, path, entries, pointer=''):
    if isinstance(value, dict):
        unit = value.get('unit') or value.get('native_unit')
        groups = [line.split(':', 2)[-1] for line in str(value.get('cgroup', '')).splitlines()]
        own = value.get('node', 'h100-10') == 'h100-10' and (
            unit in UNITS or any('/system.slice/' + name in groups for name in UNITS))
        if own and type(value.get('pid')) is int and str(value.get('birth', '')).isdigit():
            pid = value['pid'];birth = str(value['birth'])
            entry = entries.setdefault(pid, dict(pid=pid, birth=None, sources=[]))
            assert entry['birth'] in (None, birth), 'Historical birth sources disagree'
            entry['birth'] = birth
            entry['sources'].append(dict(path=str(path), field=pointer or '/', kind='original_process_snapshot'))
        for key, child in value.items():
            collect_births(child, path, entries, pointer + '/' + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            collect_births(child, path, entries, pointer + '/' + str(index))


def listeners():
    found = []
    for name in ('tcp', 'tcp6'):
        path = Path('/proc/net') / name
        for line in path.read_text().splitlines()[1:]:
            parts = line.split()
            if len(parts) < 4 or parts[3] != '0A':
                continue
            address, port = parts[1].split(':')
            if int(port, 16) != 28831:
                continue
            raw = bytes.fromhex(address)
            raw = raw[::-1] if name == 'tcp' else b''.join(raw[i:i + 4][::-1] for i in range(0, 16, 4))
            ip = ipaddress.ip_address(raw)
            if ip.is_loopback or ip.is_unspecified:
                found.append(dict(table=name, address=str(ip), port=28831))
    return found


def release(entries):
    observations = []
    for unit in sorted(UNITS):
        detail = subprocess.check_output(['systemctl', 'show', unit, '-p', 'Id', '-p', 'MainPID', '-p', 'ActiveState',
            '-p', 'Result', '-p', 'ExecMainStatus', '-p', 'ControlGroup'], text=True, timeout=15)
        props = dict(line.split('=', 1) for line in detail.splitlines() if '=' in line)
        assert props['Id'] == unit and props['MainPID'] == '0' and props['ActiveState'] == 'failed'
        assert props['Result'] == 'exit-code' and props['ExecMainStatus'] == '1'
        assert props['ControlGroup'] in ('', '/system.slice/' + unit)
        group = Path('/sys/fs/cgroup/system.slice') / unit
        pids = sorted({int(pid) for path in group.rglob('cgroup.procs') for pid in path.read_text().split()}) if group.exists() else []
        assert not pids
        observations.append(dict(unit=unit, observed=props, cgroup_exists=group.exists(), remaining_cgroup_pids=pids))
    remaining = [pid for pid in entries if Path('/proc/%d' % pid).exists()]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    gpu_pids = {int(line.strip()) for line in apps.splitlines() if line.strip().isdigit()}
    own_gpu = sorted(set(entries) & gpu_pids)
    active_listeners = listeners()
    assert not remaining and not own_gpu and not active_listeners
    return dict(epoch=time.time(), units=observations, original_births_gone=True,
        birth_claim_basis='Every evidenced own PID is absent; historical missing birth ticks were not inferred',
        own_cgroups_empty=True, remaining_known_pids=remaining, own_gpu_contexts=own_gpu,
        loopback_28831_listeners=active_listeners, foreign_signalled=False)


def publish(path, value):
    raw = (json.dumps(value, indent=2) + '\n').encode()
    temporary = path.with_name(path.name + '.tmp-%d-%d' % (os.getpid(), time.time_ns()))
    try:
        with temporary.open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary, path)
        except FileExistsError:assert path.read_bytes() == raw, 'Immutable proof already differs'
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def summarize(path, value):
    assert value['status'] == STATUS and value['run_id'] == RUN_ID and value['node'] == 'h100-10'
    return dict(status=STATUS, proof=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        units=len(value['units']), original_files=len(value['original_files']), original_bytes=value['original_bytes'],
        known_owned_pids=len(value['pid_evidence']), historical_births_recorded=len(value['original_owned_births']),
        historical_births_missing=value['pids_without_recorded_historical_birth'], release_observations=2,
        native_failure=True, controlled_stop=False, global_goal_complete=False)


def main():
    assert os.geteuid() == 0 and socket.gethostname().split('.')[0] == 'h100-10'
    assert os.stat('/proc/self/ns/net').st_ino == os.stat('/proc/1/ns/net').st_ino
    assert read(BASE / 'config.json')['run_id'] == RUN_ID
    ROOT.mkdir(parents=True, exist_ok=True)
    complete = ROOT / 'complete.json'
    with (ROOT / 'seal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if complete.exists():
            print(json.dumps(dict(summarize(complete, read(complete)), existing_proof_read_no_rehash=True)))
            return
        folders, units, entries, paths, expected_hashes = [], [], {}, set(), {}
        for pid, short, job_prefix, native_prefix, prep, intent_prefix, birth in COHORT:
            unit = 'sst-q7b-g5-' + short + '-' + RUN_ID[:12] + '.service'
            jobs = list((BASE / 'pool/jobs' / ('job-%s-%d' % (job_prefix, pid))).glob('attempt-*'))
            assert len(jobs) == 1
            job = jobs[0];allocation = read(job / 'allocation.json')
            assert allocation['controller_pid'] == pid and allocation['native_unit'] == unit and allocation['node'] == 'h100-10'
            assert allocation['native_cgroup'] == '/system.slice/' + unit
            assert 'run_id' not in allocation or allocation['run_id'] == RUN_ID
            evidence = physical(allocation['allocation_evidence'])
            native = BASE / native_prefix / 'launch' / ('native-' + str(pid))
            assert evidence == native / 'allocation-evidence.json'
            source = read(evidence)
            assert source['main_pid'] == pid and source['unit'] == unit and source['node'] == 'h100-10'
            assert source['cgroup'] == allocation['native_cgroup']
            expected_hashes[evidence] = allocation['allocation_evidence_sha256']
            failure = read(job / 'failure.json')
            assert failure['foreign_processes_signalled'] is False and failure['upstream_retriever_signalled'] is False
            assert not (job / 'complete.json').exists()
            folders.extend((job, native))
            paths.update(originals((job, native)))
            control = BASE / 'preparation' / prep
            for name in (intent_prefix + '-submit-intent.json', intent_prefix + '-submit-receipt.json'):
                path = control / name
                assert path.is_file();paths.add(path)
                if name.endswith('-intent.json'):
                    assert read(path)['run_id'] == RUN_ID
            if 'slot_context' in allocation:
                context = physical(allocation['slot_context'])
                assert context == control / 'slot-context.json' and read(context)['run_id'] == RUN_ID
                paths.add(context);expected_hashes[context] = allocation['slot_context_sha256']
            entries[pid] = dict(pid=pid, birth=birth, sources=[dict(kind='controller_PID_bound_by_allocation_and_native_evidence',
                path=str(job / 'allocation.json'), field='/controller_pid')])
            if birth:
                entries[pid]['sources'].append(dict(kind='root_task_supplied_retained_actual_birth',
                    source='Root task: actual node10 failure cohort identity', birth=birth,
                    limitation='Own allocation controller_pid does not itself record birth ticks'))
            children = read(job / 'children.json')
            for child in children['own_process_groups']:
                entries.setdefault(child, dict(pid=child, birth=None, sources=[]))['sources'].append(dict(path=str(job / 'children.json'), field='/own_process_groups', kind='own_child_PID'))
            units.append(dict(unit=unit, pid=pid, birth=birth, node='h100-10', job_directory=str(job),
                allocation_path=str(job / 'allocation.json'), native_launch=str(native), failure=failure,
                progress=read(job / 'progress.json') if (job / 'progress.json').exists() else None))
        records, error_markers = [], {}
        for path in sorted(paths):
            row, raw = original(path);records.append(row)
            if path in expected_hashes:
                assert row['sha256'] == expected_hashes[path]
            text = raw.decode(errors='replace')
            for number, line in enumerate(text.splitlines(), 1):
                try:collect_births(json.loads(line), path, entries, '/line-%d' % number)
                except json.JSONDecodeError:pass
                if path.name == 'sampler.log':
                    for value in re.findall(r'VllmWorkerProcess pid=(\d+)', line):
                        pid = int(value)
                        entries.setdefault(pid, dict(pid=pid, birth=None, sources=[]))['sources'].append(dict(path=str(path), field='line-%d' % number, kind='own_sampler_logged_MP_PID'))
            if path.suffix == '.json':
                value = json.loads(raw)
                collect_births(value, path, entries)
                if path.parent.name == 'kv-profile' and type(value.get('pid')) is int:
                    pid = value['pid']
                    entries.setdefault(pid, dict(pid=pid, birth=None, sources=[]))['sources'].append(dict(
                        path=str(path), field='/pid', kind='own_sampler_KV_profile_PID'))
            if path.name == 'sampler.log':
                error_markers[str(path)] = {name: name in text for name in ('async_retrieval_attempts_exhausted', 'HTTPError')}
        assert not {1490214, 1490465, 1654561} & set(entries), 'Source5 processes are never node10-owned'
        assert sum(all(markers.values()) for markers in error_markers.values()) >= 3
        assert any('Own local relay exited' in unit['failure']['error'] for unit in units)
        for unit in units:
            unit['birth'] = entries[unit['pid']]['birth']
        original_membership = sorted(paths)
        stable(records)
        observations = [release(entries)];time.sleep(3);observations.append(release(entries))
        assert observations[1]['epoch'] - observations[0]['epoch'] >= 3
        assert set(originals(folders)).issubset(set(original_membership))
        stable(records)
        value = dict(status=STATUS, predecessor_compatibility_status='OWN_PREDECESSOR_GENERATION_TERMINAL',
            termination_kind='OBSERVED_NATIVE_FAILURE_NOT_CONTROLLED_STOP', run_id=RUN_ID, node='h100-10',
            predecessor_compatibility_scope='Actual failed terminal release only; no controlled stop or full historical birth claim',
            source_unit=units[0]['unit'], source_pid=2371322, source_birth='1958624331', units=units,
            pid_evidence=sorted(entries.values(), key=lambda entry: entry['pid']),
            original_owned_births={str(pid): entry['birth'] for pid, entry in entries.items() if entry['birth'] is not None},
            pids_without_recorded_historical_birth=sorted(pid for pid, entry in entries.items() if entry['birth'] is None),
            all_historical_births_recorded=all(entry['birth'] is not None for entry in entries.values()),
            original_files=records, original_bytes=sum(row['bytes'] for row in records), release_observations=observations,
            error_markers=error_markers, original_partial_outputs_retained=True, original_files_not_moved_or_rewritten=True,
            controlled_stop=False, foreign_signalled=False, source5_not_queried_or_signalled=True,
            source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), global_goal_complete=False, epoch=time.time())
        publish(complete, value)
        print(json.dumps(summarize(complete, value)))


if __name__ == '__main__':
    main()
