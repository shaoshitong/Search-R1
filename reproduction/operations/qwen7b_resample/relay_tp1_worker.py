"""Four finite h100-10 TP1 slots borrowing existing primary leases and V5 relay.

Primary owners, relay owner and the source5 retriever are dependencies, never
children. Only the new per-UUID child lease and its sampler belong to this unit.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import relay_consumer_worker as consumer

shared = consumer.shared
ORIGINAL_BORROWED_RELAY = consumer.borrowed_relay
RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
SLOTS = {'gpu1': 1, 'gpu2': 2, 'gpu3': 3, 'gpu5': 5}
V5_UNIT = 'sst-q7b-g5-consumer10-v5-' + RUN_ID[:12] + '.service'
PARALLEL_UNIT = 'sst-q7b-g5-consumer10-parallel-v1-' + RUN_ID[:12] + '.service'
OWNERS = {index: dict(pid=2371322, birth='1958624331', unit=V5_UNIT) for index in (1, 2, 3)}
OWNERS[5] = dict(pid=2452973, birth='1958806278', unit=PARALLEL_UNIT)
RELAY_PID, RELAY_BIRTH = 2371352, '1958624426'
LIMITS = dict(seqs=64, tokens=8192, blocks=4096, reserve=1.0)
MAX_SECONDS, CUTOFF, BATCH_RESERVE = 14400, 600, 1200
SOURCE_UNIT = 'sst-q7b-g5-sharedvram-v6-' + RUN_ID[:12] + '.service'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(base, recorded):
    base, recorded = Path(os.path.normpath(base)), Path(os.path.normpath(recorded))
    physical = Path('/mnt/data/shaoshitong') / base.name
    if recorded.is_relative_to(physical):
        recorded = base / recorded.relative_to(physical)
    shared.require(recorded.is_relative_to(base), 'Dependency path leaves the same NAS data root')
    return recorded


def process(pid):
    path = Path('/proc/%d' % pid)
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    shared.require(fields[0] not in ('Z', 'X'), 'Dependency process is terminal')
    return dict(pid=pid, birth=fields[19], cgroup=(path / 'cgroup').read_text(),
                netns_inode=os.stat(path / 'ns/net').st_ino)


def alive(proof):
    try:
        actual = process(proof['pid'])
        return all(actual[key] == proof[key] for key in ('pid', 'birth', 'cgroup', 'netns_inode'))
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return False


def query(unit):
    output = subprocess.check_output(['systemctl', 'show', unit, '-p', 'MainPID', '-p', 'ActiveState',
                                     '-p', 'ControlGroup', '-p', 'LoadState'], text=True, timeout=15)
    return dict(line.split('=', 1) for line in output.splitlines() if '=' in line)


def held_primary(base, owner, uuid, expected=None):
    """Observe the actual primary POSIX lock without opening or changing it."""
    path = Path(base) / 'pool/gpu-locks' / (uuid + '.lock')
    actual = path.stat()
    candidates = [Path('/proc/%d/fd/%d' % (owner['pid'], expected['fd']))] if expected else list(
        Path('/proc/%d/fd' % owner['pid']).iterdir())
    for descriptor in candidates:
        try:
            held = descriptor.stat()
            if (held.st_dev, held.st_ino) != (actual.st_dev, actual.st_ino):
                continue
            fd = int(descriptor.name)
            info = Path('/proc/%d/fdinfo/%d' % (owner['pid'], fd)).read_text()
            pattern = (r'^lock:\s+\d+:\s+POSIX\s+ADVISORY\s+WRITE\s+' + str(owner['pid'])
                       + r'\s+([0-9a-fA-F]+):([0-9a-fA-F]+):(\d+)\s+0\s+EOF\s*$')
            matches = re.findall(pattern, info, re.MULTILINE)
            if not any((int(major, 16), int(minor, 16), int(inode)) ==
                       (os.major(actual.st_dev), os.minor(actual.st_dev), actual.st_ino)
                       for major, minor, inode in matches):
                continue
            result = dict(fd=fd, device=actual.st_dev, inode=actual.st_ino)
            shared.require(expected is None or result == expected, 'Original primary lock identity changed')
            return result
        except FileNotFoundError:
            continue
    raise ValueError('Actual primary owner does not hold the selected UUID POSIX WRITE lock')


def owner_snapshot(base, expected, canonical_base):
    props = query(expected['unit'])
    proof = process(expected['pid'])
    shared.require(props.get('LoadState') == 'loaded' and props.get('ActiveState') == 'active'
                   and props.get('MainPID') == str(expected['pid']) and proof['birth'] == expected['birth']
                   and props.get('ControlGroup') == '/system.slice/' + expected['unit']
                   and any(line.split(':', 2)[-1] == props['ControlGroup'] for line in proof['cgroup'].splitlines()),
                   'Current cohort primary/relay owner PID, birth or unit differs')
    attempts = list((Path(base) / 'pool/jobs' / ('job-relay10-' + str(expected['pid']))).glob('attempt-*'))
    shared.require(len(attempts) == 1, 'Expected one original owner attempt')
    allocation_path, progress_path = attempts[0] / 'allocation.json', attempts[0] / 'progress.json'
    allocation, progress = read(allocation_path), read(progress_path)
    shared.require(allocation['node'] == 'h100-10' and allocation['controller_pid'] == expected['pid']
                   and allocation['native_unit'] == expected['unit']
                   and allocation['native_cgroup'] == props['ControlGroup']
                   and allocation.get('run_id', RUN_ID) == RUN_ID, 'Primary owner allocation differs')
    proof.update(unit=expected['unit'], allocation_path=str(Path(canonical_base) / allocation_path.relative_to(base)),
                 allocation_sha256=sha(allocation_path), deadline_epoch=progress['deadline_epoch'])
    return proof, allocation, attempts[0]


def validate_configuration(value, slot):
    shared.require(slot in SLOTS and value['schema_version'] == 1 and value['mode'] == 'NODE10_FINITE_TP1_CHILD_LEASE'
                   and value['run_id'] == RUN_ID and value['slot'] == slot
                   and value['model_indices'] == [SLOTS[slot]] and len(value['model_cuda_uuids']) == 1
                   and value['tensor_parallel_size'] == 1 and value['model_limits'] == LIMITS
                   and value['unit'] == 'sst-q7b-g5-consumer10-tp1-' + slot + '-v1-' + RUN_ID[:12] + '.service',
                   'Finite slot identity, TP1 topology or initial capacity differs')
    expected = OWNERS[SLOTS[slot]]
    shared.require(all(value['lease_owner'][key] == expected[key] for key in ('pid', 'birth', 'unit'))
                   and value['relay_owner']['pid'] == 2371322 and value['relay_owner']['birth'] == '1958624331'
                   and value['relay_owner']['unit'] == V5_UNIT
                   and value['relay']['pid'] == RELAY_PID and value['relay']['birth'] == RELAY_BIRTH,
                   'Current cohort runtime dependency identity differs')
    shared.require(value['model_deadline_epoch'] <= value['relay_owner']['deadline_epoch'] - CUTOFF
                   and value['model_deadline_epoch'] <= value['lease_owner']['deadline_epoch'] - CUTOFF
                   and value['model_deadline_epoch'] <= value['source']['source_sampling_deadline_epoch'] - CUTOFF
                   and value['model_deadline_epoch'] <= value['captured_epoch'] + MAX_SECONDS,
                   'TP1 absolute cutoff exceeds an owner, source or four-hour limit')
    shared.require(value['source']['source_unit'] == SOURCE_UNIT
                   and value['source']['source_main_pid'] == 1490214
                   and value['source']['source_main_birth'] == '6807965'
                   and value['source']['source_pid'] == 1490465 and value['source']['source_birth'] == '6808057',
                   'Original source5 controller/retriever birth binding differs')


def capture_context(base, slot, canonical_base=None):
    """Submit-time read-only observation. Does not create/open any primary lease."""
    base = Path(base)
    canonical_base = str(canonical_base or base)
    shared.require(slot in SLOTS, 'Only gpu1/gpu2/gpu3/gpu5 slots are allowed')
    index = SLOTS[slot]
    owner, allocation, _ = owner_snapshot(base, OWNERS[index], canonical_base)
    relay_owner, _, relay_attempt = owner_snapshot(base, OWNERS[1], canonical_base)
    inventory = {row['index']: row for row in shared.load_pool(Path(__file__).with_name('pool_worker.py')).nvml().values()}
    uuid = inventory[index]['uuid']
    shared.require(dict(zip(allocation['model_indices'], allocation['model_cuda_uuids'])).get(index) == uuid,
                   'Selected numeric GPU does not map to the real primary owner UUID')
    primary = held_primary(base, owner, uuid)
    ready_path = relay_attempt / 'relay-ready.json'
    ready = read(ready_path)
    relay = process(RELAY_PID)
    shared.require(ready['status'] == 'RELAY_LISTENING' and ready['role'] == 'downstream'
                   and ready['listen'] == dict(host='127.0.0.1', port=28831)
                   and ready['listen_url'] == 'http://127.0.0.1:28831'
                   and all(ready['process'][key] == relay[key] for key in ('pid', 'birth', 'cgroup', 'netns_inode'))
                   and relay['birth'] == RELAY_BIRTH
                   and relay['cgroup'] == relay_owner['cgroup']
                   and relay['netns_inode'] == relay_owner['netns_inode'] == os.stat('/proc/1/ns/net').st_ino,
                   'Existing V5 local relay PID, birth, namespace or endpoint differs')
    sockets = {os.readlink(path) for path in Path('/proc/%d/fd' % RELAY_PID).iterdir() if path.is_symlink()}
    listeners = [line.split() for line in Path('/proc/net/tcp').read_text().splitlines()[1:]]
    shared.require(any(row[1] == '0100007F:709F' and row[3] == '0A'
                       and 'socket:[' + row[9] + ']' in sockets for row in listeners),
                   'The evidenced relay does not own its localhost listener')
    relay_root = base / 'preparation/relay-consumer10-control-v1'
    source_pathname, upstream_path = relay_root / 'source-context.json', relay_root / 'upstream-ready.json'
    source, upstream = read(source_pathname), read(upstream_path)
    shared.require(ready['upstream_proof']['sha256'] == sha(upstream_path)
                   and ready['source_pid'] == upstream['source_pid'] == source['source_pid']
                   and ready['source_birth'] == upstream['source_birth'] == source['source_birth']
                   and ready['source_netns_inode'] == upstream['source_netns_inode'] == source['source_netns_inode']
                   and upstream['role'] == 'upstream' and ready['code_sha256'] == sha(Path(__file__).with_name('retrieval_relay.py')),
                   'Original source context, upstream ready or relay code differs')
    canonical = lambda path: str(Path(canonical_base) / path.relative_to(base))
    now = time.time()
    value = dict(schema_version=1, mode='NODE10_FINITE_TP1_CHILD_LEASE', run_id=RUN_ID,
        base=canonical_base, slot=slot, unit='sst-q7b-g5-consumer10-tp1-' + slot + '-v1-' + RUN_ID[:12] + '.service',
        model_indices=[index], model_cuda_uuids=[uuid], tensor_parallel_size=1,
        model_limits=dict(LIMITS), lease_owner=owner, relay_owner=relay_owner, relay=relay, source=source,
        primary_gpu_lock=primary, initial_gpu_memory=inventory[index],
        reuse_local_relay_ready=canonical(ready_path), reuse_local_relay_sha256=sha(ready_path),
        upstream_ready=canonical(upstream_path), upstream_ready_sha256=sha(upstream_path),
        source_context=canonical(source_pathname), source_context_sha256=sha(source_pathname),
        model_deadline_epoch=min(now + MAX_SECONDS, relay_owner['deadline_epoch'] - CUTOFF,
                                 owner['deadline_epoch'] - CUTOFF, source['source_sampling_deadline_epoch'] - CUTOFF),
        local_relay_owned=False, primary_owner_owned=False, upstream_retriever_owned=False,
        captured_epoch=now)
    validate_configuration(value, slot)
    shared.require(value['model_deadline_epoch'] > now + BATCH_RESERVE + 60,
                   'Current cohort lifetime cannot accommodate a reserved batch')
    return value


def context(pool, args):
    path = Path(os.environ['Q7B_NODE10_TP1_CONTEXT'])
    value = pool.read(path)
    shared.require(pool.sha(path) == os.environ['Q7B_NODE10_TP1_CONTEXT_SHA256'], 'Immutable slot context SHA differs')
    validate_configuration(value, value['slot'])
    shared.require(value['base'] == str(args.base) and args.tp == 1
                   and args.model_gpus == str(SLOTS[value['slot']])
                   and (args.model_max_seqs, args.model_batch_tokens, args.model_kv_block_cap, args.model_reserve_gib)
                       == (LIMITS['seqs'], LIMITS['tokens'], LIMITS['blocks'], LIMITS['reserve'])
                   and args.batch_reserve_seconds == BATCH_RESERVE,
                   'Slot arguments differ from the immutable TP1 configuration')
    shared.require(Path(args.reuse_local_relay_ready).samefile(source_path(args.base, value['reuse_local_relay_ready']))
                   and args.reuse_local_relay_sha256 == value['reuse_local_relay_sha256']
                   and args.reuse_local_relay_unit == V5_UNIT
                   and Path(args.upstream_ready).samefile(source_path(args.base, value['upstream_ready'])),
                   'Borrowed relay/upstream arguments differ from the slot context')
    return path, value


def dependencies_live(base, value):
    try:
        if time.time() >= value['model_deadline_epoch'] or not all(
                alive(value[key]) for key in ('lease_owner', 'relay_owner', 'relay')):
            return False
        if sha(source_path(base, value['source_context'])) != value['source_context_sha256']:
            return False
        held_primary(base, value['lease_owner'], value['model_cuda_uuids'][0], value['primary_gpu_lock'])
        return True
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return False


class BorrowedDependencies:
    """Observation only: deliberately has no stop, wait or signal interface."""
    def __init__(self, base, value, relay):
        self.base, self.value, self.relay = base, value, relay
        self.pid, self.proof = relay.pid, dict(relay.proof, process_owned=False,
            lease_owner=value['lease_owner'], relay_owner=value['relay_owner'], source=value['source'])

    def poll(self):
        return None if self.relay.poll() is None and dependencies_live(self.base, self.value) else 1


def borrowed_relay(pool, args):
    _, value = context(pool, args)
    relay = ORIGINAL_BORROWED_RELAY(pool, args)
    shared.require(relay is not None, 'This slot always borrows the original V5 local relay')
    borrowed = BorrowedDependencies(args.base, value, relay)
    shared.require(borrowed.poll() is None, 'A current cohort dependency is no longer live')
    return borrowed


def lease_path(base, value):
    owner, uuid = value['lease_owner'], value['model_cuda_uuids'][0]
    return Path(base) / 'pool/gpu-subleases' / RUN_ID / (
        'h100-10-%d-%s' % (owner['pid'], owner['birth'])) / (uuid + '.lock')


def admit(pool, args):
    shared.require(socket.gethostname().split('.')[0] == 'h100-10', 'Only retained h100-10 is admitted')
    path, value = context(pool, args)
    shared.require(os.stat('/proc/self/ns/mnt').st_ino != os.stat('/proc/1/ns/mnt').st_ino
                   and os.stat('/proc/self/ns/net').st_ino == os.stat('/proc/1/ns/net').st_ino,
                   'TP1 slot needs a private retained-path mount and the existing host network')
    borrowed_relay(pool, args)
    evidence = pool.read(args.allocation_evidence)
    props = dict(re.findall(r'^(\w+)=(.*)$', evidence['unit_show'], re.MULTILINE))
    group = evidence['cgroup']
    shared.require(evidence['node'] == 'h100-10' and evidence['main_pid'] == os.getpid()
                   and props.get('Id') == evidence['unit'] and props.get('MainPID') == str(os.getpid())
                   and props.get('ActiveState') == 'active' and props.get('ControlGroup') == group
                   and group == '/system.slice/' + evidence['unit'] and evidence['unit'] == value['unit']
                   and group in Path('/proc/self/cgroup').read_text()
                   and evidence['resource_limits']['runtime_max_seconds'] <= MAX_SECONDS,
                   'Slot controller is outside its own bounded unit')
    inventory = {row['index']: row for row in pool.nvml().values()}
    index, uuid = value['model_indices'][0], value['model_cuda_uuids'][0]
    shared.require(inventory[index]['uuid'] == uuid, 'Actual numeric GPU UUID changed after submit')
    owner = value['lease_owner']
    allocation_path = source_path(args.base, owner['allocation_path'])
    shared.require(pool.sha(allocation_path) == owner['allocation_sha256'], 'Original primary allocation SHA differs')
    locks = []
    try:
        lease = lease_path(args.base, value)
        fd = pool.try_lock(lease)
        shared.require(fd is not None, 'Another cooperating TP1 slot holds this primary UUID child lease')
        locks.append(fd)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(index), CUDA_DEVICE_ORDER='PCI_BUS_ID')
        result = subprocess.check_output([sys.executable, str(Path(consumer.__file__).resolve()),
            '--cuda-probe', '--expected-uuids', uuid], env=env, text=True, timeout=45)
        shared.require(json.loads(result)['actual_cuda_runtime_uuids'] == [uuid], 'Actual CUDA UUID differs')
        shared.require(dependencies_live(args.base, value)
                       and value['model_deadline_epoch'] > time.time() + BATCH_RESERVE + 60,
                       'Dependency identity or reserved lifetime changed during slot admission')
        return dict(job_id='relay10-tp1-' + str(os.getpid()), node='h100-10', controller_pid=os.getpid(), run_id=RUN_ID,
            allocation_kind='HUMAN_AUTHORIZED_NODE10_PRIMARY_CHILD_TP1', native_unit=evidence['unit'], native_cgroup=group,
            model_indices=[index], model_cuda_uuids=[uuid], model_cuda_visible_devices=str(index),
            actual_cuda_runtime_uuids=[uuid], tensor_parallel_size=1, model_engine_count=1,
            initial_gpu_memory=[inventory[index]], foreign_initial_contexts=shared.foreign_contexts({uuid}),
            foreign_contexts_allowed=True, foreign_processes_signalled=False,
            source_model_deadline_epoch=value['model_deadline_epoch'], slot_context=str(path), slot_context_sha256=pool.sha(path),
            lease_owner=owner, relay_owner=value['relay_owner'], gpu_sublease_paths={uuid: str(lease)},
            primary_gpu_locks_modified=False, primary_owner_owned=False, local_relay_reused=True,
            local_relay_owned=False, local_relay_process=dict(value['relay'], process_owned=False),
            upstream_retriever_owned=False, shared_retriever_owned_by_consumer=False,
            original_cuda_visible_devices=evidence.get('original_cuda_visible_devices'),
            upstream_ready=str(args.upstream_ready), upstream_ready_sha256=pool.sha(args.upstream_ready),
            allocation_evidence=str(args.allocation_evidence), allocation_evidence_sha256=pool.sha(args.allocation_evidence),
            native_resource_limits=evidence['resource_limits'], captured_epoch=time.time()), locks
    except BaseException:
        for fd in locks:
            os.close(fd)
        raise


def main():
    consumer.admit = admit
    consumer.borrowed_relay = borrowed_relay
    consumer.main()


if __name__ == '__main__':
    main()
