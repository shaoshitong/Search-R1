"""Once-only original-inode proof of five real pre-model primary v1 failures.

Run only after exact publication on physical NAS10. This helper never signals,
loads a model, reads trajectories or requires the borrowed CPU relay port free.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import time

BASE = Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
RECORDED = Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
ROOT = BASE / 'preparation/node10-primary-v1-pre-model-terminal-20261021'
STATUS = 'OWN_NODE10_PRIMARY_V1_PRE_MODEL_FAILURES_ORIGINAL_SHA_RELEASE_PASS'
COHORT = (('gpu1-tp1', 2886544), ('gpu2-tp1', 2887149), ('gpu3-tp1', 2888798),
          ('gpu5-tp1', 2890177), ('tp4-0-4-6-7', 2891150))
UNITS = {'sst-q7b-g5-primary10-' + slot + '-v1-' + RUN_ID[:12] + '.service' for slot, _ in COHORT}
PROTECTED = {2882823, 2882852, 1490214, 1490465, 1654561}


def read(path):return json.loads(Path(path).read_text())


def physical(path):
    path = Path(os.path.normpath(path))
    if path.is_relative_to(RECORDED):path = BASE / path.relative_to(RECORDED)
    assert path.is_relative_to(BASE)
    return path


def identity(s):return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def metadata(folder):
    assert folder.is_dir() and not folder.is_symlink()
    return {p for p in folder.rglob('*') if not any(x in ('sampling', 'states', 'features', 'cache', 'tmp')
        for x in p.relative_to(folder).parts) and p.is_file() and not p.is_symlink()
        and p.suffix in ('.json', '.log', '.txt', '.out', '.err', '.py', '.sh')
        and not any(x in p.name.lower() for x in ('secret', 'token'))}


def original(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        assert stat.S_ISREG(before.st_mode) and identity(before) == identity(path.stat())
        h = hashlib.sha256()
        for raw in iter(lambda: stream.read(1024 * 1024), b''):h.update(raw)
        assert identity(before) == identity(os.fstat(stream.fileno())) == identity(path.stat())
    return dict(path=str(path), bytes=before.st_size, sha256=h.hexdigest(), device=before.st_dev,
        inode=before.st_ino, mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns)


def stable(rows):
    for row in rows:
        assert identity(Path(row['path']).stat()) == tuple(row[k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns'))


def binding(slot, pid, allocation, evidence):
    unit = 'sst-q7b-g5-primary10-' + slot + '-v1-' + RUN_ID[:12] + '.service'
    assert unit in UNITS and allocation['run_id'] == RUN_ID and allocation['node'] == evidence['node'] == 'h100-10'
    assert allocation['controller_pid'] == evidence['main_pid'] == pid
    assert type(allocation['controller_birth']) is int and allocation['controller_birth'] > 0
    assert allocation['controller_birth'] == evidence['controller_birth']
    assert allocation['boot_id'] == evidence['boot_id']
    assert allocation['native_unit'] == evidence['unit'] == unit
    assert allocation['native_cgroup'] == evidence['cgroup'] == '/system.slice/' + unit
    assert allocation['source_owner']['pid'] == 2882823 and allocation['local_relay_process']['pid'] == 2882852
    assert allocation['local_relay_owned'] is False and allocation['upstream_retriever_owned'] is False
    return unit


def own_pids(pid, birth, job, children):
    entries = {pid: dict(pid=pid, birth=str(birth), sources=[dict(path=str(job / 'allocation.json'), field='/controller_birth')])}
    for child in children['own_process_groups']:
        assert type(child) is int and child > 0 and child not in PROTECTED and child != pid
        entries[child] = dict(pid=child, birth=None, sources=[dict(path=str(job / 'children.json'),
            field='/own_process_groups', limitation='PID recorded; historical birth not recorded in children metadata')])
    assert not PROTECTED.intersection(entries)
    return entries


def pre_model(job):
    assert not (job / 'complete.json').exists()
    assert not list(job.rglob('model-initialized.json')) and not list(job.rglob('typed-model-initialization-oom.json'))
    progress = read(job / 'progress.json') if (job / 'progress.json').exists() else {}
    assert all(progress.get(key, 0) == 0 for key in ('completed_own_batches', 'questions', 'trajectories'))
    if progress:assert progress['run_id'] == RUN_ID
    leaves = list((job / 'sampler-launches').glob('launch-*/failure.json'))
    assert len(leaves) == 1
    leaf = read(leaves[0])
    assert leaf['error_type'] == 'KeyError' and leaf['error'] == "'original_cuda_visible_devices'"
    text = (job / 'sampler.log').read_text()
    assert "KeyError: 'original_cuda_visible_devices'" in text and '_configure_one_visible_gpu' in text
    assert 'shared_vram_worker.py' in text
    failure = read(job / 'failure.json')
    assert failure['foreign_processes_signalled'] is False and failure['upstream_retriever_signalled'] is False
    return dict(error_type='KeyError', error=leaf['error'], callback='shared_vram_worker.device',
        actual_typed_oom=False, actual_model_initialized=False, new_complete_batches=0, new_trajectories=0,
        failure=failure, sampler_failure=leaf, progress=progress)


def release(entries):
    assert entries and not PROTECTED.intersection(entries)
    units = []
    for unit in sorted(UNITS):
        raw = subprocess.check_output(['systemctl', 'show', unit, '-p', 'Id', '-p', 'MainPID', '-p', 'ActiveState',
            '-p', 'Result', '-p', 'ExecMainStatus', '-p', 'ControlGroup'], text=True, timeout=15)
        props = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
        assert props['Id'] == unit and props['MainPID'] == '0' and props['ActiveState'] == 'failed'
        assert props['Result'] == 'exit-code' and props['ExecMainStatus'] == '1'
        assert props['ControlGroup'] in ('', '/system.slice/' + unit)
        group = Path('/sys/fs/cgroup/system.slice') / unit
        pids = sorted({int(x) for p in group.rglob('cgroup.procs') for x in p.read_text().split()}) if group.exists() else []
        assert not pids
        units.append(dict(unit=unit, observed=props, remaining_cgroup_pids=pids))
    remaining = sorted(pid for pid in entries if Path('/proc/%d' % pid).exists())
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    gpu = sorted(set(entries) & {int(x.strip()) for x in raw.splitlines() if x.strip().isdigit()})
    assert not remaining and not gpu
    return dict(epoch=time.time(), units=units, remaining_known_pids=remaining, own_gpu_contexts=gpu,
        own_cgroups_empty=True, original_births_gone=True,
        birth_claim_basis='Every evidenced own PID is absent; missing historical child births were not inferred',
        borrowed_cpu_relay_pids_excluded=[2882823, 2882852], port_28831_free_not_required=True, foreign_signalled=False)


def publish(path, value):
    raw = (json.dumps(value, indent=2) + '\n').encode()
    temporary = path.with_name(path.name + '.tmp-%d-%d' % (os.getpid(), time.time_ns()))
    try:
        with temporary.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary, path)
        except FileExistsError:assert path.read_bytes() == raw
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temporary.unlink(missing_ok=True)


def summary(path, value):
    assert value['status'] == STATUS and value['run_id'] == RUN_ID and value['node'] == 'h100-10'
    return dict(status=STATUS, proof=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        units=len(value['units']), original_files=len(value['original_files']), original_bytes=value['original_bytes'],
        historical_births_missing=value['pids_without_recorded_historical_birth'], release_observations=2,
        actual_model_initialized=False, new_trajectories=0, controlled_stop=False, global_goal_complete=False)


def main():
    assert os.geteuid() == 0 and socket.gethostname().split('.')[0] == 'h100-10'
    assert os.stat('/proc/self/ns/net').st_ino == os.stat('/proc/1/ns/net').st_ino
    assert read(BASE / 'config.json')['run_id'] == RUN_ID
    ROOT.mkdir(parents=True, exist_ok=True)
    complete = ROOT / 'complete.json'
    with (ROOT / 'seal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if complete.exists():
            print(json.dumps(dict(summary(complete, read(complete)), existing_proof_read_no_rehash=True)));return
        folders, paths, expected, units, entries = [], set(), {}, [], {}
        for slot, pid in COHORT:
            jobs = list((BASE / 'pool/jobs' / ('job-relay10-primary-' + str(pid))).glob('attempt-*'))
            assert len(jobs) == 1
            job = jobs[0];allocation = read(job / 'allocation.json')
            native = BASE / 'relay-primary10/launch' / ('native-' + str(pid))
            evidence = physical(allocation['allocation_evidence']);assert evidence == native / 'allocation-evidence.json'
            unit = binding(slot, pid, allocation, read(evidence));expected[evidence] = allocation['allocation_evidence_sha256']
            folders.extend((job, native));paths.update(metadata(job) | metadata(native))
            control = BASE / 'preparation' / ('relay-primary10-' + slot + '-v1')
            intent = read(control / 'native-submit-intent.json');assert intent['run_id'] == RUN_ID and intent['unit'] == unit
            context = physical(allocation['slot_context']);assert context == control / 'slot-context.json'
            assert read(context)['run_id'] == RUN_ID and read(context)['unit'] == unit
            expected[context] = allocation['slot_context_sha256'];assert expected[context] == intent['dependency_context_sha256']
            stage = control / intent['source_bundle_sha256']
            paths.update((context, control / 'native-submit-intent.json', control / 'native-submit-receipt.json',
                stage / 'manifest.json', stage / 'stage-receipt.json', stage / 'relay_primary_bootstrap.py', stage / 'consumer_paths.sh'))
            fresh = own_pids(pid, allocation['controller_birth'], job, read(job / 'children.json'))
            assert not set(fresh) & set(entries);entries.update(fresh)
            units.append(dict(slot=slot, unit=unit, pid=pid, birth=str(allocation['controller_birth']), boot_id=allocation['boot_id'],
                job_directory=str(job), allocation_path=str(job / 'allocation.json'), native_launch=str(native), **pre_model(job)))
        records = [original(path) for path in sorted(paths)]
        for row in records:
            if Path(row['path']) in expected:assert row['sha256'] == expected[Path(row['path'])]
        stable(records);observations = [release(entries)];time.sleep(3);observations.append(release(entries))
        assert observations[1]['epoch'] - observations[0]['epoch'] >= 3
        assert set().union(*(metadata(folder) for folder in folders)).issubset(paths);stable(records)
        value = dict(status=STATUS, termination_kind='OBSERVED_PRE_MODEL_KEYERROR_FAILURE_NOT_CONTROLLED_STOP', run_id=RUN_ID,
            node='h100-10', units=units, pid_evidence=sorted(entries.values(), key=lambda row: row['pid']),
            original_owned_births={str(pid): row['birth'] for pid, row in entries.items() if row['birth'] is not None},
            pids_without_recorded_historical_birth=sorted(pid for pid, row in entries.items() if row['birth'] is None),
            original_files=records, original_bytes=sum(row['bytes'] for row in records), release_observations=observations,
            actual_model_initialized=False, new_complete_batches=0, new_trajectories=0, actual_typed_oom=False,
            borrowed_cpu_relay_pids_excluded=[2882823, 2882852], source5_not_queried_or_signalled=True,
            original_shared_batch_partials_retained=True, original_files_not_moved_or_rewritten=True,
            controlled_stop=False, foreign_signalled=False, global_goal_complete=False,
            source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), epoch=time.time())
        publish(complete, value);print(json.dumps(summary(complete, value)))


if __name__ == '__main__':main()
