"""Seal consumer10 V2's real weight-loading OOM and observed own release.

Only this V2 terminal directory is written. No process is signalled, and the
productive V6 and upstream CPU relay are neither queried nor controlled.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time


BASE = Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
UNIT = 'sst-q7b-g5-consumer10-v2-70188b8e50fd.service'
PID = 2327023
STATUS = 'OWN_CONSUMER_V2_TERMINAL_FULL_SHA_RELEASE_PASS'
JOB = BASE / 'pool/jobs/job-relay10-2327023/attempt-1792524696315153174-2327023'
NATIVE = BASE / 'relay-consumer10/launch/native-2327023'
ROOT = BASE / 'preparation/relay-consumer10-control-v1/v2-terminal'
COMPLETE = ROOT / 'complete.json'


def read(path):
    return json.loads(path.read_text())


def identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def original_paths():
    paths = sorted(p for folder in (JOB, NATIVE) for p in folder.rglob('*') if p.is_file())
    assert paths and len(paths) <= 100
    assert all(not p.is_symlink() and p.stat().st_size <= 8 * 1024 ** 2 for p in paths)
    assert sum(p.stat().st_size for p in paths) <= 16 * 1024 ** 2
    return paths


def hash_original(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        assert stat.S_ISREG(before.st_mode) and identity(before) == identity(path.stat())
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        assert identity(before) == identity(os.fstat(stream.fileno())) == identity(path.stat())
    return dict(path=str(path), dev=before.st_dev, inode=before.st_ino, bytes=before.st_size,
                mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns, sha256=digest.hexdigest())


def assert_originals_stable(records):
    assert [str(p) for p in original_paths()] == [r['path'] for r in records]
    for row in records:
        actual = Path(row['path']).stat()
        assert identity(actual) == (row['dev'], row['inode'], row['bytes'], row['mtime_ns'], row['ctime_ns'])


def release_observation(pids, cgroup):
    detail = subprocess.check_output(['systemctl', 'show', UNIT, '-p', 'MainPID', '-p', 'ControlGroup',
        '-p', 'ActiveState', '-p', 'Result', '-p', 'ExecMainStatus'], text=True, timeout=15)
    props = dict(line.split('=', 1) for line in detail.splitlines() if '=' in line)
    assert props['MainPID'] == '0' and props['ActiveState'] == 'failed'
    assert props['Result'] == 'exit-code' and props['ExecMainStatus'] == '1'
    assert props['ControlGroup'] in ('', cgroup)
    group = Path('/sys/fs/cgroup') / cgroup.lstrip('/')
    group_files = sorted(group.rglob('cgroup.procs')) if group.exists() else []
    group_pids = sorted({int(x) for p in group_files for x in p.read_text().split()})
    birth_checks = []
    for pid in pids:
        proc = Path('/proc/%d' % pid)
        if proc.exists():
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            birth_checks.append(dict(pid=pid, proc_exists=True, observed_birth_ticks=fields[19],
                                     state=fields[0], cgroup=(proc / 'cgroup').read_text()))
        else:
            birth_checks.append(dict(pid=pid, proc_exists=False, observed_birth_ticks=None))
    remaining = [x['pid'] for x in birth_checks if x['proc_exists']]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    gpu_pids = {int(line.split(',')[1].strip()) for line in apps.splitlines() if line.strip()}
    own_gpu = sorted(set(pids) & gpu_pids)
    unlisted_unit_gpu = []
    for pid in gpu_pids - set(pids):
        path = Path('/proc/%d/cgroup' % pid)
        try:
            if cgroup in path.read_text():
                unlisted_unit_gpu.append(pid)
        except FileNotFoundError:
            pass
    assert not group_pids and not remaining and not own_gpu and not unlisted_unit_gpu
    return dict(epoch=time.time(), unit=detail, original_birth_checks=birth_checks,
        original_births_not_recorded_in_launch_receipts=True,
        absence_proves_all_listed_original_pids_gone=True, remaining_known_pids=remaining,
        remaining_original_births={}, expected_cgroup=cgroup, cgroup_exists=group.exists(),
        remaining_cgroup_pids=group_pids, own_gpu_pids=own_gpu,
        unlisted_unit_gpu_pids=unlisted_unit_gpu, gpu_compute_apps=apps)


def summary(value):
    assert value['status'] == STATUS and value['controller_pid'] == PID and value['unit'] == UNIT
    return dict(status=value['status'], proof=str(COMPLETE),
        sha256=hashlib.sha256(COMPLETE.read_bytes()).hexdigest(),
        original_files=len(value['original_files']), original_bytes=value['original_bytes'],
        release_observations=len(value['actual_release']), checked_known_pids=value['known_owned_pids'],
        all_logged_mp_pids=value['all_logged_mp_pids'], new_complete_batches=value['new_complete_batches'])


def main():
    assert socket.gethostname().split('.')[0] == 'h100-10'
    if COMPLETE.exists():
        print(json.dumps(dict(summary(read(COMPLETE)), existing_proof_read_no_replay=True)))
        return
    allocation, child = read(JOB / 'allocation.json'), read(JOB / 'children.json')
    assert allocation['controller_pid'] == PID and allocation['native_unit'] == UNIT
    assert allocation['job_id'] == 'relay10-2327023' and allocation['node'] == 'h100-10'
    cgroup = allocation['native_cgroup']
    assert cgroup == '/system.slice/' + UNIT
    evidence = read(NATIVE / 'allocation-evidence.json')
    assert evidence['main_pid'] == PID and evidence['unit'] == UNIT and evidence['cgroup'] == cgroup
    failure = read(JOB / 'failure.json')
    assert failure['status'] == 'OWN_RELAY_CONSUMER_PARTIAL_OR_FAILED'
    assert failure['error_type'] == 'ValueError' and 'eligible typed constructor OOM retry' in failure['error']
    assert failure['foreign_processes_signalled'] is False and failure['upstream_retriever_signalled'] is False
    assert not any((JOB / name).exists() for name in ('progress.json', 'complete.json', 'sampler-exit.json'))
    launches = sorted((JOB / 'sampler-launches').glob('launch-*'))
    assert [p.name for p in launches] == ['launch-00', 'launch-01', 'launch-02']
    leaf_failures, typed_ooms = [], []
    for launch in launches:
        leaf, typed, initialization = (read(launch / name) for name in
            ('failure.json', 'typed-model-initialization-oom.json', 'model-initialization.json'))
        assert leaf['error_type'] == 'OutOfMemoryError' and 'CUDA out of memory' in leaf['error']
        assert 'load_model' in leaf['traceback'] and 'create_weights' in leaf['traceback']
        assert typed['actual_typed_cuda_oom'] and typed['before_first_new_sampling_request']
        assert initialization['first_new_sampling_request_started'] is False
        leaf_failures.append(dict(launch=str(launch), **leaf))
        typed_ooms.append(typed)
    sampler_text = (JOB / 'sampler.log').read_text(errors='replace')
    mp_pids = sorted({int(x) for x in re.findall(r'VllmWorkerProcess pid=(\d+)', sampler_text)})
    assert {2327194, 2327195, 2327196, 2328271, 2328272, 2328273, 2329072, 2329073, 2329074} <= set(mp_pids)
    assert 'method load_model' in sampler_text and 'torch.OutOfMemoryError' in sampler_text
    pids = sorted({PID, *child['own_process_groups'], *mp_pids})
    assert not {1490214, 1490465, 1654561} & set(pids)
    records = [hash_original(path) for path in original_paths()]
    observations = []
    for ordinal in range(2):
        assert_originals_stable(records)
        observations.append(release_observation(pids, cgroup))
        if ordinal == 0:
            time.sleep(3)
    assert observations[1]['epoch'] - observations[0]['epoch'] >= 3
    assert_originals_stable(records)
    value = dict(status=STATUS, unit=UNIT, controller_pid=PID,
        job_directory=str(JOB), original_native_launch=str(NATIVE),
        failure=failure, leaf_weight_loading_failures=leaf_failures, typed_oom_attempts=typed_ooms,
        all_logged_mp_pids=mp_pids, known_owned_pids=pids,
        original_files=records, original_files_count=len(records), original_bytes=sum(r['bytes'] for r in records),
        original_inodes_stable_across_observations=True, actual_release=observations,
        observations_elapsed_seconds=observations[1]['epoch'] - observations[0]['epoch'],
        real_gpu_weight_loading_attempted=True, real_gpu_model_initialized=False,
        new_sampling_request_started=False, new_complete_batches=0, unsealed_partials_retained=True,
        healthy_v6_not_queried_or_signalled=True, upstream_cpu_relay_1654561_not_queried_or_signalled=True,
        foreign_processes_signalled=False, global_goal_complete=False, epoch=time.time())
    ROOT.mkdir(exist_ok=True)
    temporary = ROOT / ('complete.json.tmp-%d-%d' % (os.getpid(), time.time_ns()))
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, COMPLETE)
        except FileExistsError:
            value = read(COMPLETE)
        directory = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(summary(value)))


if __name__ == '__main__':
    main()
