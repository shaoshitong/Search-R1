"""Retire only own extra TP4 after a complete own G5 batch for a TP1 continuation.

No stop intent is created before one own complete batch. Frozen old loop has
no drain hook: a newly claimed incomplete phase may exist when the unit stops;
it is retained and never counted or stitched. Borrowed V6/retriever stay alive.
Execute in original V6 mount/network namespace. Existing intent is read only.
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

BASE = Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
PHYSICAL = Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
UNIT = 'sst-q7b-g5-source5-extra-model-v1-70188b8e50fd.service'
PID, BIRTH = 2021698, '7430669'
PARENT_PID, PARENT_BIRTH, RETRIEVER_PID, RETRIEVER_BIRTH = 1490214, '6807965', 1490465, '6808057'
PARENT_UNIT = 'sst-q7b-g5-sharedvram-v6-70188b8e50fd.service'
STATUS = 'OWN_SOURCE5_MODEL_TP4_CONTROLLED_THROUGHPUT_STOP_FULL_SHA_RELEASE_PASS'
ROOT = BASE / 'preparation/source5-model-tp4-to-tp1-v1'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical(path):
    return str(PHYSICAL / path.relative_to(BASE))


def birth(pid):
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return None if fields[0] in ('Z', 'X') else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def query(unit):
    detail = subprocess.check_output(['systemctl', 'show', unit, '-p', 'Id', '-p', 'MainPID',
        '-p', 'ActiveState', '-p', 'ControlGroup', '-p', 'Result', '-p', 'ExecMainStatus'],
        text=True, timeout=20)
    return dict(line.split('=', 1) for line in detail.splitlines() if '=' in line)


def parent_preserved():
    q = query(PARENT_UNIT)
    assert q['MainPID'] == str(PARENT_PID) and q['ActiveState'] == 'active'
    assert birth(PARENT_PID) == PARENT_BIRTH and birth(RETRIEVER_PID) == RETRIEVER_BIRTH
    for namespace in ('mnt', 'net'):
        assert os.stat('/proc/self/ns/' + namespace).st_ino == os.stat('/proc/%d/ns/%s' % (PARENT_PID, namespace)).st_ino
    return dict(pid=PARENT_PID, birth=PARENT_BIRTH, unit=PARENT_UNIT,
        retriever_pid=RETRIEVER_PID, retriever_birth=RETRIEVER_BIRTH, unit_show=q)


def owned(group):
    result = {}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            paths = [x.split(':', 2)[-1] for x in (p / 'cgroup').read_text().splitlines()]
            if any(x == group or x.startswith(group + '/') for x in paths):
                b = birth(int(p.name))
                if b is not None:
                    result[p.name] = b
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return result


def publish(path, value):
    raw = (json.dumps(value, indent=2) + '\n').encode()
    temporary = path.with_name(path.name + '.tmp-%d-%d' % (os.getpid(), time.time_ns()))
    try:
        with temporary.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def phases(known):
    paths = set()
    for pid in known:
        folder = Path('/proc/%s/fd' % pid)
        if not folder.exists():
            continue
        for fd in folder.iterdir():
            try:
                target = os.readlink(fd)
            except (FileNotFoundError, ProcessLookupError):
                continue
            if '/pool/batch-locks/' not in target:
                continue
            prefix, relative = target.split('/pool/batch-locks/', 1)
            tail = Path(relative)
            if not prefix.endswith('/shaoshitong/' + BASE.name) or len(tail.parts) != 2 or tail.suffix != '.lock':
                continue
            outer = BASE/'sampling'/tail.parent/'batches'/tail.stem/'outers/outer-000001'
            paths.add(str(outer))
    return sorted(paths)


def phase_records(outers):
    records = []
    for raw in outers:
        outer = Path(raw)
        for phase in ('prefix', 'full'):
            folder = outer/phase
            files = []
            if folder.exists():
                for p in sorted(folder.glob('rank-0-*')):
                    if p.is_file():
                        item = dict(path=physical(p),bytes=p.stat().st_size)
                        if p.suffix == '.json' and p.stat().st_size <= 1024*1024:
                            item.update(sha256=digest(p),value=read(p))
                        files.append(item)
            records.append(dict(outer=physical(outer),phase=phase,files=files))
    return records


def file_identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def hash_original(path):
    assert not path.is_symlink()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        assert stat.S_ISREG(before.st_mode) and file_identity(before) == file_identity(path.stat())
        d = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            d.update(block)
        assert file_identity(before) == file_identity(os.fstat(stream.fileno())) == file_identity(path.stat())
    return dict(path=physical(path), source_access_path=str(path), bytes=before.st_size,
        dev=before.st_dev, inode=before.st_ino, mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns, sha256=d.hexdigest())


def release(known, group):
    q = query(UNIT)
    assert q['MainPID'] == '0' and q['ActiveState'] in ('inactive', 'failed')
    remaining = {p:b for p,b in known.items() if birth(int(p)) == b}
    group_pids = owned(group)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    gpu_pids = {int(x.strip()) for x in apps.splitlines() if x.strip().isdigit()}
    own_gpu = sorted(gpu_pids & {int(p) for p in known})
    path = Path('/sys/fs/cgroup') / group.lstrip('/')
    files = list(path.rglob('cgroup.procs')) if path.exists() else []
    actual_pids = {int(x) for p in files for x in p.read_text().split()}
    assert not remaining and not group_pids and not actual_pids and not own_gpu
    parent_preserved()
    return dict(epoch=time.time(), unit_main_pid=0, unit=q, original_births_gone=True,
        remaining_original_births=remaining, own_cgroup_gone=True, cgroup_exists=path.exists(),
        remaining_cgroup_pids=sorted(actual_pids), own_gpu_contexts=own_gpu, foreign_signalled=False)


def main():
    assert os.geteuid() == 0 and socket.gethostname().split('.')[0] == 'h100-5'
    assert read(BASE / 'config.json')['run_id'] == RUN_ID
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete, intent, receipt = ROOT/'complete.json', ROOT/'intent.json', ROOT/'receipt.json'
        if complete.exists():
            v = read(complete); assert v['status'] == STATUS
            print(json.dumps(dict(status=STATUS, proof=physical(complete), sha256=digest(complete), existing_proof_read_no_replay=True)))
            return
        if intent.exists():
            print(json.dumps(dict(status='EXISTING_STOP_INTENT_NO_REPLAY', intent=read(intent),
                receipt=read(receipt) if receipt.exists() else None, unit=query(UNIT))))
            return
        parent = parent_preserved()
        current = query(UNIT)
        assert current['Id'] == UNIT and current['MainPID'] == str(PID) and current['ActiveState'] == 'active'
        assert birth(PID) == BIRTH
        group = current['ControlGroup']; assert group == '/system.slice/' + UNIT
        attempts = list((BASE/'pool/jobs'/('job-source5-model-'+str(PID))).glob('attempt-*'))
        assert len(attempts) == 1
        job = attempts[0]; allocation_path = job/'allocation.json'; allocation = read(allocation_path)
        assert allocation['controller_pid'] == PID and allocation['native_unit'] == UNIT and allocation['native_cgroup'] == group
        context_path = Path(allocation['source_context'])
        assert str(context_path).startswith(str(BASE) + '/preparation/') and digest(context_path) == allocation['source_context_sha256']
        context = read(context_path)
        assert context['run_id'] == RUN_ID and context['parent']['pid'] == PARENT_PID and context['parent']['birth'] == PARENT_BIRTH
        assert allocation.get('run_id', context['run_id']) == RUN_ID and allocation['model_indices'] == context['model_indices'] == [3,4,6,7] and allocation['tensor_parallel_size'] == 4
        assert allocation['source_parent']['pid'] == PARENT_PID and allocation['source_parent']['birth'] == PARENT_BIRTH
        progress = read(job/'progress.json')
        assert progress['run_id'] == RUN_ID
        if progress['completed_own_batches'] < 1:
            print(json.dumps(dict(status='WAIT_FIRST_OWN_WHOLE_BATCH_NO_STOP_INTENT_CREATED', progress=progress)))
            return
        known = owned(group); assert known.get(str(PID)) == BIRTH and not {'1490214','1490465','2013773','2193217','2194351'} & set(known)
        active_outers = phases(known)
        original_active_phases = phase_records(active_outers)
        retired = dict(unit=UNIT,pid=PID,birth=known[str(PID)],allocation_path=physical(allocation_path),allocation_sha256=digest(allocation_path))
        proposed = dict(status='CONTROLLED_TP4_TO_TP1_STOP_INTENT',run_id=RUN_ID,retired=retired,source_parent=parent,
            progress=progress,original_owned_births=known,original_active_phases=original_active_phases,original_partial_outputs_retained=True,
            no_zero_partial_loss_claim=True,foreign_signalled=False,source_code_sha256=digest(Path(__file__)),epoch=time.time())
        publish(intent, proposed)
        try:
            r = subprocess.run(['systemctl','stop',UNIT],text=True,capture_output=True,timeout=75)
            result = dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        except subprocess.TimeoutExpired as error:
            result = dict(error_type=type(error).__name__,error=str(error),outcome_unknown=True)
        result.update(status='CONTROLLED_TP4_STOP_QUERIED',unit=query(UNIT),epoch=time.time(),foreign_signalled=False)
        publish(receipt, result)
        native = BASE/'source-sublease/launch'/('native-'+str(PID))
        paths = sorted(p for folder in (job,native) for p in folder.rglob('*') if p.is_file())
        records = [hash_original(p) for p in paths]
        observations = [release(known, group)]; time.sleep(3); observations.append(release(known,group))
        for row in records:
            assert file_identity(Path(row['source_access_path']).stat()) == (row['dev'],row['inode'],row['bytes'],row['mtime_ns'],row['ctime_ns'])
        final_progress = read(job/'progress.json')
        value = dict(status=STATUS,run_id=RUN_ID,retired=retired,source_parent=parent,
            source_parent_and_retriever_preserved=True,stopped_after_at_least_own_batches=final_progress['completed_own_batches'],
            original_files=records,original_files_count=len(records),original_bytes=sum(x['bytes'] for x in records),
            release_observations=observations,original_owned_births=known,
            original_partial_outputs_retained=True,old_loop_had_no_drain_hook=True,no_zero_partial_loss_claim=True,
            source_intent=physical(intent),source_intent_sha256=digest(intent),source_stop_receipt=physical(receipt),
            final_progress=final_progress,original_active_phases_before_stop=original_active_phases,retained_active_phases_after_stop=phase_records(active_outers),source_code_sha256=digest(Path(__file__)),foreign_signalled=False,global_goal_complete=False,epoch=time.time())
        publish(complete,value)
        print(json.dumps(dict(status=STATUS,proof=physical(complete),source_access_proof=str(complete),sha256=digest(complete),
            completed_own_batches=final_progress['completed_own_batches'],original_files=len(records),original_bytes=value['original_bytes'],
            own_release_observations=2,partial_outputs_retained=True,source_parent_and_retriever_preserved=True,global_goal_complete=False)))


if __name__ == '__main__':
    main()
