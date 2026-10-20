"""Explicitly authorized once-stop of three still-waiting primary v2 controllers.

No healthy sampler, borrowed CPU relay or source5 process is a stop target.
Any new sampler/ready/progress/batch-lock evidence aborts the stop decision.
"""
import argparse
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
ROOT = BASE / 'preparation/node10-primary-v2-readiness-repair-20261021'
STATUS = 'OWN_NODE10_PRIMARY_V2_READINESS_CONTROLLED_STOP_ORIGINAL_SHA_RELEASE_PASS'
COHORT = (('gpu2-tp1', 2905668, '1959784299'), ('gpu5-tp1', 2909621, '1959792986'),
          ('tp4-0-4-6-7', 2910951, '1959794962'))
PROTECTED_LOCAL = {2882823, 2882852, 2904782, 2906630}


def read(path):return json.loads(Path(path).read_text())


def physical(path):
    path = Path(os.path.normpath(path))
    if path.is_relative_to(RECORDED):path = BASE / path.relative_to(RECORDED)
    assert path.is_relative_to(BASE)
    return path


def identity(s):return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def original(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno());assert stat.S_ISREG(before.st_mode) and identity(before) == identity(path.stat())
        h = hashlib.sha256()
        for raw in iter(lambda:stream.read(1024 * 1024), b''):h.update(raw)
        assert identity(before) == identity(os.fstat(stream.fileno())) == identity(path.stat())
    return dict(path=str(path), bytes=before.st_size, sha256=h.hexdigest(), device=before.st_dev,
        inode=before.st_ino, mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns)


def stable(rows):
    for row in rows:
        assert identity(Path(row['path']).stat()) == tuple(row[k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns'))


def metadata(folder):
    assert folder.is_dir() and not folder.is_symlink()
    return {p for p in folder.rglob('*') if not any(x in ('sampling', 'states', 'features', 'cache', 'tmp')
        for x in p.relative_to(folder).parts) and p.is_file() and not p.is_symlink()
        and p.suffix in ('.json', '.log', '.txt', '.out', '.err', '.py', '.sh')
        and not any(x in p.name.lower() for x in ('secret', 'token'))}


def publish(path, value):
    raw = (json.dumps(value, indent=2) + '\n').encode();temporary = path.with_name(path.name + '.tmp-%d-%d' % (os.getpid(), time.time_ns()))
    try:
        with temporary.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary, path)
        except FileExistsError:assert path.read_bytes() == raw
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temporary.unlink(missing_ok=True)


def process(pid):
    root = Path('/proc/%d' % pid);fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    assert fields[0] != 'Z'
    return dict(pid=pid, birth=fields[19], cgroup=(root / 'cgroup').read_text())


def query(unit):
    raw = subprocess.check_output(['systemctl', 'show', unit, '-p', 'Id', '-p', 'MainPID', '-p', 'ActiveState',
        '-p', 'Result', '-p', 'ExecMainStatus', '-p', 'ControlGroup'], text=True, timeout=15)
    return dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)


def cgroup_pids(unit):
    folder = Path('/sys/fs/cgroup/system.slice') / unit
    return sorted({int(x) for p in folder.rglob('cgroup.procs') for x in p.read_text().split()}) if folder.exists() else []


def batch_fds(pid):
    found = []
    for path in (Path('/proc/%d/fd' % pid)).iterdir():
        try:target = Path(os.readlink(path).removesuffix(' (deleted)'))
        except FileNotFoundError:continue
        if any(target.is_relative_to(root / 'pool/batch-locks') for root in (BASE, RECORDED)):
            found.append(dict(fd=int(path.name), target=str(target)))
    return found


def zero_work(job, allow_terminal=False):
    forbidden = ('sampler-launches', 'sampler.log', 'progress.json', 'complete.json', 'sampler-exit.json', 'retriever-ready.json')
    assert not any((job / name).exists() for name in forbidden), 'Controller is no longer waiting without sampler work'
    assert not list(job.rglob('model-initialized.json'))
    children = read(job / 'children.json') if (job / 'children.json').exists() else {}
    assert not children.get('own_process_groups', []), 'A sampler child was recorded; require a new decision'
    if not allow_terminal:assert not (job / 'failure.json').exists()
    return dict(children=children, sampler_children=0, model_initialized=False, progress_present=False, new_trajectories=0)


def capture():
    rows = []
    for slot, pid, birth in COHORT:
        unit = 'sst-q7b-g5-primary10-' + slot + '-v2-' + RUN_ID[:12] + '.service'
        jobs = list((BASE / 'pool/jobs' / ('job-relay10-primary-' + str(pid))).glob('attempt-*'));assert len(jobs) == 1
        job = jobs[0];allocation_path = job / 'allocation.json';allocation = read(allocation_path)
        evidence_path = physical(allocation['allocation_evidence']);native = BASE / 'relay-primary10/launch' / ('native-' + str(pid))
        assert evidence_path == native / 'allocation-evidence.json'
        evidence = read(evidence_path);originals = [original(allocation_path), original(evidence_path)]
        assert originals[1]['sha256'] == allocation['allocation_evidence_sha256']
        assert allocation['run_id'] == RUN_ID and allocation['instance'] == 'v2' and allocation['node'] == evidence['node'] == 'h100-10'
        assert allocation['controller_pid'] == evidence['main_pid'] == pid
        assert str(allocation['controller_birth']) == str(evidence['controller_birth']) == birth
        assert allocation['boot_id'] == evidence['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        assert allocation['native_unit'] == evidence['unit'] == unit
        assert allocation['native_cgroup'] == evidence['cgroup'] == '/system.slice/' + unit
        assert allocation['source_owner']['pid'] == 2882823 and allocation['local_relay_process']['pid'] == 2882852
        assert allocation['local_relay_owned'] is False and allocation['upstream_retriever_owned'] is False
        control = BASE / 'preparation' / ('relay-primary10-' + slot + '-v2')
        context = physical(allocation['slot_context']);assert context == control / 'slot-context.json'
        context_original = original(context);assert context_original['sha256'] == allocation['slot_context_sha256']
        originals.append(context_original)
        intent = read(control / 'native-submit-intent.json');assert intent['unit'] == unit and intent['run_id'] == RUN_ID
        assert intent['dependency_context_sha256'] == context_original['sha256']
        rows.append(dict(slot=slot, unit=unit, pid=pid, birth=birth, job_directory=str(job), native_launch=str(native),
            allocation_path=str(allocation_path), allocation_sha256=originals[0]['sha256'],
            allocation_evidence_path=str(evidence_path), allocation_evidence_sha256=originals[1]['sha256'],
            control_directory=str(control), source_bundle_sha256=intent['source_bundle_sha256'], prehashed_originals=originals))
    return rows


def preflight(row):
    assert (row['slot'], row['pid'], row['birth']) in COHORT and row['pid'] not in PROTECTED_LOCAL
    stable(row['prehashed_originals']);props = query(row['unit']);actual = process(row['pid'])
    group = '/system.slice/' + row['unit']
    assert props['Id'] == row['unit'] and props['MainPID'] == str(row['pid']) and props['ActiveState'] == 'active'
    assert props['ControlGroup'] == group and actual['birth'] == row['birth']
    assert group in [line.split(':', 2)[-1] for line in actual['cgroup'].splitlines()]
    assert cgroup_pids(row['unit']) == [row['pid']], 'Owned cgroup gained a child'
    threads = Path('/proc/%d/task' % row['pid'])
    children = sorted({int(x) for path in threads.glob('*/children') for x in path.read_text().split()})
    assert not children
    work = zero_work(Path(row['job_directory']));fds = batch_fds(row['pid']);assert not fds, 'Controller has an input batch lock FD'
    return dict(epoch=time.time(), unit=row['unit'], process=actual, observed=props, cgroup_pids=[row['pid']],
        task_children=children, batch_lock_fds=fds, **work)


def stop_once(rows, intent_path, receipt_path):
    assert [(r['slot'], r['pid'], r['birth']) for r in rows] == list(COHORT)
    assert all(row['unit'] == 'sst-q7b-g5-primary10-' + row['slot'] + '-v2-' + RUN_ID[:12] + '.service' for row in rows)
    assert not intent_path.exists(), 'Existing stop intent: reconcile actual outcome; never replay stop'
    before = [preflight(row) for row in rows]
    command = ['systemctl', 'stop', *[row['unit'] for row in rows]]
    intent = dict(status='EXPLICIT_AUTHORIZED_THREE_WAITING_CONTROLLERS_READINESS_REPAIR_STOP_INTENT',
        run_id=RUN_ID, node='h100-10', reason='Existing three-second health readiness loop; controlled replacement, not failure/OOM',
        units=rows, command=command, preflight=before, protected_local_pids=sorted(PROTECTED_LOCAL),
        authorization_source=dict(active_goal_created_epoch=1792511267, full_fresh_g5_trajectories=845140,
            use_all_h100_resources=True, root_necessary_admission_runtime_adjustment=True,
            human_direct_three_unit_stop_request=False),
        source5_not_queried_or_signalled=True, parent_goal_continues_845140=True, epoch=time.time())
    publish(intent_path, intent)
    fresh = [preflight(row) for row in rows]
    result = subprocess.run(command, text=True, capture_output=True, timeout=90)
    receipt = dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        final_preflight=fresh, epoch=time.time(), foreign_signalled=False)
    publish(receipt_path, receipt);assert result.returncode == 0, 'Actual owned stop failed; no complete proof'
    return intent, receipt


def release(rows, protected):
    units = []
    for row in rows:
        props = query(row['unit']);assert props['Id'] == row['unit'] and props['MainPID'] == '0'
        assert props['ActiveState'] in ('inactive', 'failed') and props['ControlGroup'] in ('', '/system.slice/' + row['unit'])
        remaining = cgroup_pids(row['unit']);assert not remaining
        assert not Path('/proc/%d' % row['pid']).exists()
        zero_work(Path(row['job_directory']), allow_terminal=True)
        units.append(dict(unit=row['unit'], observed=props, remaining_cgroup_pids=remaining))
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    gpu = sorted({row['pid'] for row in rows} & {int(x.strip()) for x in raw.splitlines() if x.strip().isdigit()});assert not gpu
    preserved = [process(pid) for pid in sorted(PROTECTED_LOCAL)];assert preserved == protected
    return dict(epoch=time.time(), units=units, remaining_known_pids=[], own_gpu_contexts=gpu,
        original_births_gone=True, own_cgroups_empty=True, protected_local_processes=preserved,
        port_28831_free_not_required=True, foreign_signalled=False)


def summary(path, value):
    assert value['status'] == STATUS and value['run_id'] == RUN_ID
    return dict(status=STATUS, proof=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_files=len(value['original_files']), original_bytes=value['original_bytes'], controlled_stop=True,
        actual_typed_oom=False, new_trajectories=0, parent_goal_continues_845140=True, global_goal_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute-authorized-stop', action='store_true');args = parser.parse_args()
    assert args.execute_authorized_stop, 'Explicit root execution authorization flag required'
    assert os.geteuid() == 0 and socket.gethostname().split('.')[0] == 'h100-10'
    assert os.stat('/proc/self/ns/net').st_ino == os.stat('/proc/1/ns/net').st_ino
    assert read(BASE / 'config.json')['run_id'] == RUN_ID
    ROOT.mkdir(parents=True, exist_ok=True);complete = ROOT / 'complete.json'
    with (ROOT / 'seal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if complete.exists():print(json.dumps(dict(summary(complete, read(complete)), existing_proof_read_no_rehash=True)));return
        rows = capture();protected = [process(pid) for pid in sorted(PROTECTED_LOCAL)]
        intent, receipt = stop_once(rows, ROOT / 'stop-intent.json', ROOT / 'stop-receipt.json')
        observations = [release(rows, protected)];paths, folders = set(), []
        cached = {Path(record['path']): record for row in rows for record in row['prehashed_originals']}
        for row in rows:
            job, native, control = map(Path, (row['job_directory'], row['native_launch'], row['control_directory']))
            folders.extend((job, native));paths.update(metadata(job) | metadata(native));stage = control / row['source_bundle_sha256']
            paths.update((control / 'slot-context.json', control / 'native-submit-intent.json', control / 'native-submit-receipt.json',
                stage / 'manifest.json', stage / 'stage-receipt.json', stage / 'relay_primary_bootstrap.py', stage / 'consumer_paths.sh'))
        records = [cached[path] if path in cached else original(path) for path in sorted(paths)];stable(records)
        time.sleep(3);observations.append(release(rows, protected));assert observations[1]['epoch'] - observations[0]['epoch'] >= 3
        assert set().union(*(metadata(folder) for folder in folders)).issubset(paths);stable(records)
        value = dict(status=STATUS, run_id=RUN_ID, node='h100-10', termination_kind='EXPLICIT_AUTHORIZED_READINESS_REPAIR_NOT_FAILURE_OR_OOM',
            units=rows, stop_intent=intent, stop_receipt=receipt, release_observations=observations,
            original_owned_births={str(row['pid']): row['birth'] for row in rows}, original_files=records,
            original_bytes=sum(row['bytes'] for row in records), protected_local_pids=sorted(PROTECTED_LOCAL),
            source5_not_queried_or_signalled=True, original_shared_batch_partials_retained=True, controlled_stop=True,
            actual_model_initialized=False, new_trajectories=0, actual_typed_oom=False, foreign_signalled=False,
            parent_goal_continues_845140=True, global_goal_complete=False,
            source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), epoch=time.time())
        publish(complete, value);print(json.dumps(summary(complete, value)))


if __name__ == '__main__':main()
