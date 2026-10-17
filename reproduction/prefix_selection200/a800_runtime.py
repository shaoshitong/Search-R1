"""A800 process admission for the four-rank prefix5 actor experiment.

No Slurm, global Ray cleanup, GPU reservation process, or foreign signal. The
published config binds four idle training UUIDs and two auxiliary UUIDs once.
OFD descriptors stay open in the parent and are inherited by every child.
"""
import csv
import fcntl
import io
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

from reproduction.a800.owned_processes import identity, same
from reproduction.prefix20_online.launch import once, publish, read


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()

PREFIX = '/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017'


def gpu_inventory():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    return {int(row[0]): dict(index=int(row[0]), uuid=row[1].strip(), memory_MiB=int(row[2]))
            for row in csv.reader(io.StringIO(out))}


def compute_processes():
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    result = []
    for row in csv.reader(io.StringIO(out)):
        if row and row[0].strip().startswith('GPU-'):
            result.append(dict(uuid=row[0].strip(), pid=int(row[1])))
    return result


class UUIDLocks:
    def __init__(self, uuids):
        root = Path('/data/user/jxu957/experiments/.thread5-prefix5-gpu-uuid-locks')
        root.mkdir(exist_ok=True)
        self.fds = []
        try:
            for uuid in sorted(uuids):
                if not uuid.startswith('GPU-') or Path(uuid).name != uuid:
                    raise ValueError('Invalid actual GPU UUID')
                fd = os.open(root / (uuid + '.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                self.fds.append(fd)
                fcntl.fcntl(fd, fcntl.F_OFD_SETLK,
                    struct.pack('hhqqi4x', fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0))
        except BaseException:
            self.close()
            raise

    def close(self):
        for fd in self.fds:
            os.close(fd)
        self.fds = []


def native_allocation(base, c):
    """Validate one actual mapping before any CUDA child, while holding leases."""
    if not str(base).startswith(PREFIX):
        raise ValueError('Wrong isolated A800 namespace')
    training = c['physical_gpu_indices']
    aux = c['auxiliary_gpu_indices']
    if len(training) != 4 or len(set(training)) != 4 or len(aux) != 4 or len(set(aux)) != 4 or set(training) & set(aux):
        raise ValueError('Four Slurm training GPUs and four distinct shared auxiliary GPUs required')
    if not os.environ.get('SLURM_JOB_ID') or c.get('execution_backend') != 'slurm_hpc3':
        raise ValueError('HPC3 requires a real owned Slurm allocation')
    mapping = gpu_inventory()
    ids = training + aux
    expected = c['physical_gpu_uuids'] + c['auxiliary_gpu_uuids']
    if [mapping[i]['uuid'] for i in ids] != expected:
        raise ValueError('Actual indices/UUIDs changed after preparation')
    used = {x['uuid'] for x in compute_processes()}
    if used & set(expected) or any(mapping[i]['memory_MiB'] > 64 for i in ids):
        raise RuntimeError('GPU admission rejected: an assigned card is occupied; do not stop its owner')
    value = dict(run_id=c['run_id'], source_commit=c['source_commit'],
        physical_gpu_indices=training, physical_gpu_uuids=c['physical_gpu_uuids'],
        auxiliary_gpu_indices=aux, auxiliary_gpu_uuids=c['auxiliary_gpu_uuids'],
        checked_unix=time.time(), idle_memory_threshold_MiB=64,
        actual_inventory=[mapping[i] for i in ids], foreign_processes_stopped=False)
    once(base / 'logs/native-allocation.json', value)
    return value


def query(base):
    prep = Path(base) / 'preparation'
    path = prep / 'submit-receipt.json'
    if not path.exists():
        return dict(status='NO_SUBMIT_RECEIPT_UNKNOWN_NOT_EXIT', base=str(base))
    result = read(path)
    leader = result.get('leader')
    return dict(result, leader_same_birth=(same(leader) if leader else None))


def submit(base, validate):
    """One owned detached process; unknown launch outcomes are never replayed."""
    base = Path(base).resolve()
    prep = base / 'preparation'
    with (prep / 'submit-control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (prep / 'submit-intent.json').exists():
            return query(base)
        c = validate(read(prep / 'launch-config.json'), base)
        # Qualify the exact chosen source before stopping any previous task.
        source = Path(c['probe_source_checkpoint'])
        if not source.is_file() or sha(source) != c['probe_source_sha256']:
            raise ValueError('Chosen complete warmup checkpoint is missing or changed')
        once(prep / 'submit-intent.json', dict(run_id=c['run_id'], source_commit=c['source_commit'],
            config_sha256=sha(prep / 'launch-config.json'), created_unix=time.time()))
        env = dict(os.environ, PYTHONPATH=str(base / 'repo'), PYTHONUNBUFFERED='1',
                   PYTHONDONTWRITEBYTECODE='1')
        python = str(Path(c['environment_root']) / 'bin/python')
        command = [python, '-m', 'reproduction.prefix_selection200.launch', 'run', '--base', str(base)]
        with (base / 'logs/native-pipeline.log').open('ab') as output:
            child = subprocess.Popen(command, env=env, cwd=base / 'repo', stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True)
        who = identity(child.pid)
        if who['sid'] != child.pid or who['uid'] != os.getuid():
            raise RuntimeError('Unknown wrapper identity; inspect, never replay')
        receipt = dict(status='ONCE_LAUNCHED_NOT_TRAINING_ACCEPTANCE', leader=who,
            run_id=c['run_id'], source_commit=c['source_commit'], command=command,
            launched_unix=time.time(), config_sha256=sha(prep / 'launch-config.json'))
        once(prep / 'submit-receipt.json', receipt)
        return receipt
