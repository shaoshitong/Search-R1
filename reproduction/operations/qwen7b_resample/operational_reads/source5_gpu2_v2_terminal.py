"""Seal only GPU2 V2's pre-model contract-import failure; never signal anything.

Execute as root on h100-5 through nsenter --target 1490214 --mount --net.
Physical NAS10 paths are recorded; ACCESS_BASE is that source namespace's
existing NAS10 mount alias. Existing terminal proof is read without replay.
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
ACCESS_BASE = Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
UNIT = 'sst-q7b-g5-source5-extra-gpu2-v2-70188b8e50fd.service'
PID, PARENT_PID = 2175836, 1490214
STATUS = 'OWN_SOURCE5_GPU2_V2_TERMINAL_FULL_SHA_RELEASE_PASS'
MESSAGE = 'Only the original four-rank'
NATIVE = ACCESS_BASE / 'source-sublease/launch/native-2175836'
ROOT = ACCESS_BASE / 'preparation/source5-model-sublease-gpu2-v2/v2-terminal'
COMPLETE = ROOT / 'complete.json'


def read(path):
    return json.loads(path.read_text())


def physical(path):
    return str(BASE / path.relative_to(ACCESS_BASE))


def identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def original_paths(job):
    return sorted(p for folder in (job, NATIVE) for p in folder.rglob('*') if p.is_file())


def hash_original(path):
    assert not path.is_symlink()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        assert stat.S_ISREG(before.st_mode) and identity(before) == identity(path.stat())
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        assert identity(before) == identity(os.fstat(stream.fileno())) == identity(path.stat())
    return dict(path=physical(path), source_access_path=str(path), dev=before.st_dev, inode=before.st_ino,
                bytes=before.st_size, mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns, sha256=digest.hexdigest())


def stable(job, records):
    assert [str(p) for p in original_paths(job)] == [row['source_access_path'] for row in records]
    for row in records:
        assert identity(Path(row['source_access_path']).stat()) == (
            row['dev'], row['inode'], row['bytes'], row['mtime_ns'], row['ctime_ns'])


def release(pids, cgroup):
    detail = subprocess.check_output(['systemctl', 'show', UNIT, '-p', 'MainPID', '-p', 'ControlGroup',
        '-p', 'ActiveState', '-p', 'Result', '-p', 'ExecMainStatus'], text=True, timeout=15)
    props = dict(line.split('=', 1) for line in detail.splitlines() if '=' in line)
    assert props['MainPID'] == '0' and props['ActiveState'] == 'failed'
    assert props['Result'] == 'exit-code' and props['ExecMainStatus'] == '1'
    assert props['ControlGroup'] in ('', cgroup)
    group = Path('/sys/fs/cgroup') / cgroup.lstrip('/')
    group_files = sorted(group.rglob('cgroup.procs')) if group.exists() else []
    group_pids = sorted({int(x) for p in group_files for x in p.read_text().split()})
    remaining = [pid for pid in pids if Path('/proc/%d' % pid).exists()]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    gpu_pids = {int(x.strip()) for x in apps.splitlines() if x.strip().isdigit()}
    own_gpu = sorted(set(pids) & gpu_pids)
    assert not remaining and not group_pids and not own_gpu
    return dict(epoch=time.time(), unit=detail, remaining_known_pids=remaining,
        original_births_not_recorded_in_launch_receipts=True, remaining_original_births={},
        expected_cgroup=cgroup, cgroup_exists=group.exists(), remaining_cgroup_pids=group_pids,
        own_gpu_pids=own_gpu, gpu_query_returncode=0)


def summary(value):
    assert value['status'] == STATUS and value['controller_pid'] == PID and value['unit'] == UNIT
    assert value['new_complete_batches'] == value['new_trajectories'] == 0 and not value['real_gpu_model_initialized']
    return dict(status=STATUS, proof=physical(COMPLETE), source_access_proof=str(COMPLETE),
        sha256=hashlib.sha256(COMPLETE.read_bytes()).hexdigest(), original_files=value['original_files_count'],
        original_bytes=value['original_bytes'], release_observations=len(value['actual_release']),
        new_complete_batches=0, new_trajectories=0, real_gpu_model_initialized=False)


def main():
    assert os.geteuid() == 0 and socket.gethostname().split('.')[0] == 'h100-5'
    for namespace in ('mnt', 'net'):
        assert os.stat('/proc/self/ns/' + namespace).st_ino == os.stat('/proc/%d/ns/%s' % (PARENT_PID, namespace)).st_ino
    assert Path('/proc/%d/stat' % PARENT_PID).read_text().rsplit(')', 1)[1].split()[19] == '6807965'
    assert read(ACCESS_BASE / 'config.json')['run_id'] == RUN_ID
    if COMPLETE.exists():
        print(json.dumps(dict(summary(read(COMPLETE)), existing_proof_read_no_replay=True)))
        return
    attempts = list((ACCESS_BASE / 'pool/jobs/job-source5-model-2175836').glob('attempt-*'))
    assert len(attempts) == 1 and NATIVE.is_dir()
    job = attempts[0]
    allocation, children = read(job / 'allocation.json'), read(job / 'children.json')
    assert allocation['controller_pid'] == PID and allocation['native_unit'] == UNIT
    assert allocation['run_id'] == RUN_ID and allocation['node'] == 'h100-5'
    assert allocation['model_indices'] == [2] and allocation['tensor_parallel_size'] == 1
    cgroup = allocation['native_cgroup']
    assert cgroup == '/system.slice/' + UNIT
    evidence = read(NATIVE / 'allocation-evidence.json')
    assert evidence['main_pid'] == PID and evidence['unit'] == UNIT and evidence['cgroup'] == cgroup
    failure = read(job / 'failure.json')
    assert failure['foreign_processes_signalled'] is False and failure['upstream_retriever_signalled'] is False
    assert not any((job / name).exists() for name in ('progress.json', 'complete.json', 'sampler-exit.json'))
    assert not list(job.rglob('model-initialized.json')) and not list(job.rglob('model-initialization.json'))
    launches = sorted((job / 'sampler-launches').glob('launch-*'))
    assert [p.name for p in launches] == ['launch-00']
    leaf = read(launches[0] / 'failure.json')
    assert leaf['error_type'] == 'ValueError' and MESSAGE in leaf['error']
    assert MESSAGE in leaf['traceback'] and 'contract.py' in leaf['traceback']
    sampler_text = (job / 'sampler.log').read_text(errors='replace')
    assert MESSAGE in sampler_text and not re.findall(r'VllmWorkerProcess pid=(\d+)', sampler_text)
    pids = sorted({PID, *children['own_process_groups']})
    assert not {1490214, 1490465, 2013773, 2021698, 2371322, 2452973} & set(pids)
    records = [hash_original(path) for path in original_paths(job)]
    observations = []
    for ordinal in range(2):
        stable(job, records)
        observations.append(release(pids, cgroup))
        if ordinal == 0:
            time.sleep(3)
    assert observations[1]['epoch'] - observations[0]['epoch'] >= 3
    stable(job, records)
    value = dict(status=STATUS, run_id=RUN_ID, unit=UNIT, controller_pid=PID,
        job_directory=physical(job), original_native_launch=physical(NATIVE),
        failure=failure, leaf_import_failure=leaf, expected_contract_error=MESSAGE,
        known_owned_pids=pids, original_files=records, original_files_count=len(records),
        original_bytes=sum(row['bytes'] for row in records), original_inodes_stable_across_observations=True,
        actual_release=observations, observations_elapsed_seconds=observations[1]['epoch'] - observations[0]['epoch'],
        physical_world_size_bootstrap_was_invalid=True, configured_sampler_world_size=1, tensor_parallel_size=1,
        real_gpu_model_initialized=False, real_gpu_weight_loading_attempted=False,
        new_sampling_request_started=False, new_complete_batches=0, new_trajectories=0,
        original_job_and_launch_retained=True, healthy_source_groups_not_signalled=True,
        foreign_processes_signalled=False, global_goal_complete=False, epoch=time.time())
    ROOT.mkdir(parents=True, exist_ok=True)
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
