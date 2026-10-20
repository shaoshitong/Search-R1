"""Seal consumer10 V3's original mixed profile errors and observed own release.

Only this V3 terminal directory is written; any execution dump stays remote. No process is signalled, and the
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
UNIT = 'sst-q7b-g5-consumer10-v3-70188b8e50fd.service'
PID = 2351003
STATUS = 'OWN_CONSUMER_V3_TERMINAL_FULL_SHA_RELEASE_PASS'
JOB = BASE / 'pool/jobs/job-relay10-2351003/attempt-1792525220715374225-2351003'
NATIVE = BASE / 'relay-consumer10/launch/native-2351003'
ROOT = BASE / 'preparation/relay-consumer10-control-v1/v3-terminal'
COMPLETE = ROOT / 'complete.json'
DUMP = Path('/tmp/err_execute_model_input_20261021-034044.pkl')
SOURCE_STAGE = '0aeed1c00102797f3aaf5645d291d84982be38b6d9fd59d7edd5533ddfec5f5d'


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


def release_observation(pids, cgroup, births):
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
            birth_checks.append(dict(pid=pid, original_birth_ticks=births.get(str(pid)),
                                     proc_exists=True, observed_birth_ticks=fields[19],
                                     state=fields[0], cgroup=(proc / 'cgroup').read_text()))
        else:
            birth_checks.append(dict(pid=pid, original_birth_ticks=births.get(str(pid)),
                                     proc_exists=False, observed_birth_ticks=None))
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
        recorded_original_births=births,
        original_births_unrecorded_for_pids=[pid for pid in pids if str(pid) not in births],
        absence_proves_all_listed_original_pids_gone=True, remaining_known_pids=remaining,
        remaining_original_births={}, expected_cgroup=cgroup, cgroup_exists=group.exists(),
        remaining_cgroup_pids=group_pids, own_gpu_pids=own_gpu,
        unlisted_unit_gpu_pids=unlisted_unit_gpu, gpu_compute_apps=apps)


def preserve_dump():
    if not DUMP.exists():
        return dict(original_path=str(DUMP), existed_at_terminal=False,
                    remote_only=True, unpickled=False)
    target = ROOT / DUMP.name
    temporary = ROOT / ('dump.tmp-%d-%d' % (os.getpid(), time.time_ns()))
    try:
        with os.fdopen(os.open(DUMP, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as source:
            before = os.fstat(source.fileno())
            assert stat.S_ISREG(before.st_mode) and identity(before) == identity(DUMP.stat())
            digest = hashlib.sha256()
            with temporary.open('xb') as destination:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
                    destination.write(block)
                destination.flush()
                os.fsync(destination.fileno())
            assert identity(before) == identity(os.fstat(source.fileno())) == identity(DUMP.stat())
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
        copied = hash_original(target)
        assert copied['bytes'] == before.st_size and copied['sha256'] == digest.hexdigest()
        return dict(original_path=str(DUMP), existed_at_terminal=True, remote_only=True,
            unpickled=False, source=dict(dev=before.st_dev, inode=before.st_ino,
                bytes=before.st_size, mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns,
                sha256=digest.hexdigest()), retained_copy=copied,
            original_inode_stable_during_copy=True)
    finally:
        temporary.unlink(missing_ok=True)


def summary(value):
    assert value['status'] == STATUS and value['controller_pid'] == PID and value['unit'] == UNIT
    return dict(status=value['status'], proof=str(COMPLETE),
        sha256=hashlib.sha256(COMPLETE.read_bytes()).hexdigest(),
        original_files=len(value['original_files']), original_bytes=value['original_bytes'],
        release_observations=len(value['actual_release']), checked_known_pids=value['known_owned_pids'],
        all_logged_mp_pids=value['all_logged_mp_pids'], new_complete_batches=value['new_complete_batches'],
        parent_initialization_error_type=value['parent_initialization_error_type'],
        actual_profile_leaf_error_types=value['actual_profile_leaf_error_types'],
        remote_execution_dump=value['remote_execution_dump'])


def main():
    assert socket.gethostname().split('.')[0] == 'h100-10'
    if COMPLETE.exists():
        print(json.dumps(dict(summary(read(COMPLETE)), existing_proof_read_no_replay=True)))
        return
    allocation, child = read(JOB / 'allocation.json'), read(JOB / 'children.json')
    assert allocation['controller_pid'] == PID and allocation['native_unit'] == UNIT
    assert allocation['job_id'] == 'relay10-2351003' and allocation['node'] == 'h100-10'
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
    assert [p.name for p in launches] == ['launch-00']
    launch = launches[0]
    leaf = read(launch / 'failure.json')
    initialization = read(launch / 'model-initialization.json')
    assert leaf['error_type'] == 'DistBackendError'
    assert 'Failed to CUDA calloc 10485760 bytes' in leaf['error']
    assert 'profile_run' in leaf['traceback'] and 'torch.distributed.all_reduce' in leaf['traceback']
    assert SOURCE_STAGE in leaf['traceback']
    assert initialization['first_new_sampling_request_started'] is False
    assert not (launch / 'typed-model-initialization-oom.json').exists()
    assert not (launch / 'model-initialized.json').exists()
    sampler_text = (JOB / 'sampler.log').read_text(errors='replace')
    mp_pids = sorted({int(x) for x in re.findall(r'VllmWorkerProcess pid[= ](\d+)', sampler_text)})
    assert mp_pids == [2351222, 2351223, 2351224]
    lines = sampler_text.splitlines()
    selected = [dict(line_number=i + 1, text=line) for i, line in enumerate(lines)
        if any(token in line for token in ('Loading model weights took',
            'NCCL_NVLS_ENABLE set by environment', 'coll channels', 'torch.OutOfMemoryError:',
            'torch.distributed.DistBackendError:', 'Failed to CUDA calloc'))]
    weight_lines = [row for row in selected if 'Loading model weights took' in row['text']]
    nvls_lines = [row for row in selected if 'NCCL_NVLS_ENABLE set by environment to 0.' in row['text']]
    assert len(weight_lines) == 4 and len(nvls_lines) == 4
    assert {int(re.search(r'\[(\d+)\]', row['text']).group(1)) for row in nvls_lines} == {0, 1, 2, 3}
    assert 'Failed to CUDA calloc 2097152 bytes' in sampler_text
    assert 'Failed to CUDA calloc 10485760 bytes' in sampler_text
    assert 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 56.00 MiB.' in sampler_text
    assert evidence['communication_environment']['NCCL_NVLS_ENABLE'] == '0'
    relay = read(JOB / 'relay-ready.json')['process']
    assert relay['pid'] in child['own_process_groups'] and cgroup in relay['cgroup']
    births = {str(relay['pid']): relay['birth']}
    pids = sorted({PID, *child['own_process_groups'], *mp_pids})
    assert not {1490214, 1490465, 1654561} & set(pids)
    records = [hash_original(path) for path in original_paths()]
    observations = []
    for ordinal in range(2):
        assert_originals_stable(records)
        observations.append(release_observation(pids, cgroup, births))
        if ordinal == 0:
            time.sleep(3)
    assert observations[1]['epoch'] - observations[0]['epoch'] >= 3
    assert_originals_stable(records)
    ROOT.mkdir(exist_ok=True)
    value = dict(status=STATUS, unit=UNIT, controller_pid=PID,
        job_directory=str(JOB), original_native_launch=str(NATIVE),
        failure=failure, original_parent_initialization_failure=leaf, model_initialization=initialization,
        original_profile_error_lines=selected, original_source_stage=SOURCE_STAGE,
        recorded_original_births=births, typed_oom_attempts=[],
        actual_profile_leaf_error_types=['OutOfMemoryError', 'DistBackendError'],
        parent_initialization_error_type='DistBackendError',
        typed_constructor_oom_retry_eligible=False,
        unique_failure_root_cause_claimed=False, remote_execution_dump=preserve_dump(),
        all_logged_mp_pids=mp_pids, known_owned_pids=pids,
        original_files=records, original_files_count=len(records), original_bytes=sum(r['bytes'] for r in records),
        original_inodes_stable_across_observations=True, actual_release=observations,
        observations_elapsed_seconds=observations[1]['epoch'] - observations[0]['epoch'],
        real_gpu_weight_loading_attempted=True, all_four_ranks_weight_loading_completed=True,
        real_gpu_model_initialized=False, real_profile_run_failed=True,
        new_sampling_request_started=False, new_complete_batches=0, unsealed_partials_retained=True,
        healthy_v6_not_queried_or_signalled=True, upstream_cpu_relay_1654561_not_queried_or_signalled=True,
        foreign_processes_signalled=False, global_goal_complete=False, epoch=time.time())
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
