"""Finite node10 model owners borrowing the independent old-source CPU relay.

Each controller owns its primary UUID locks, one CPU-owner live-slot lease and
only its sampler children. Neither the CPU relay nor source5 is a child.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

import relay_consumer_worker as consumer
import legacy_relay_owner as owner

shared = consumer.shared
RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
TOPOLOGIES = owner.TOPOLOGIES
MAX_SECONDS, CUTOFF, BATCH_RESERVE = 14400, 600, 1200


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(base, recorded):
    """Resolve the same NAS object from host physical or retained mount paths."""
    base, recorded = Path(os.path.normpath(base)), Path(os.path.normpath(recorded))
    for root in (Path('/mnt/data/shaoshitong') / base.name,
                 Path('/mnt/shared/workspace/training_data_node_10/shaoshitong') / base.name, base):
        if recorded.is_relative_to(root):
            return base / recorded.relative_to(root)
    raise ValueError('Dependency path leaves the same NAS data root')


def process(pid):
    return owner.process(pid)


def alive(proof):
    try:
        actual = process(proof['pid'])
        return all(str(actual[key]) == str(proof[key]) for key in
                   ('pid', 'birth', 'cgroup', 'netns_inode', 'mntns_inode'))
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return False


def names(slot, instance='v2'):
    shared.require(any(slot in slots for slots in TOPOLOGIES.values()), 'Unknown finite node10 slot')
    shared.require(instance in ('v1', 'v2'), 'Unknown primary instance')
    return ('sst-q7b-g5-primary10-' + slot + '-' + instance + '-' + RUN_ID[:12] + '.service',
            'relay-primary10-' + slot + '-' + instance)


def validate_configuration(value, slot):
    slots = TOPOLOGIES[value['topology']]
    shared.require(slot in slots and value['schema_version'] == 1
        and value['mode'] == 'NODE10_INDEPENDENT_PRIMARY_RELAY_CONSUMER'
        and value['run_id'] == RUN_ID and value['slot'] == slot and value['unit'] == names(slot, value['instance'])[0]
        and value['slot_configuration'] == slots[slot]
        and value['model_indices'] == slots[slot]['indices']
        and len(value['model_cuda_uuids']) == slots[slot]['tp']
        and len(set(value['model_cuda_uuids'])) == slots[slot]['tp'], 'Finite primary model configuration differs')
    ready, source = value['cpu_ready'], value['source']
    shared.require(ready['status'] == 'LEGACY_SOURCE_RELAY_OWNER_READY'
        and ready['role'] == 'current-oldV6-source' and ready['run_id'] == RUN_ID
        and ready['node'] == 'h100-10' and ready['slots'] == slots
        and not ready['gpu_primary_leases_owned'] and not ready['model_processes_owned'],
        'Independent CPU relay identity or explicit slot topology differs')
    shared.require(source['source_main_pid'] == 1490214 and source['source_pid'] == 1490465
        and str(source['source_main_birth']) == '6807965' and str(source['source_birth']) == '6808057'
        and source['source_unit'] == 'sst-q7b-g5-sharedvram-v6-' + RUN_ID[:12] + '.service',
        'Current old V6 source identity differs')
    shared.require(value['model_deadline_epoch'] <= ready['consumer_deadline_epoch']
        and value['model_deadline_epoch'] <= ready['deadline_epoch'] - CUTOFF
        and value['model_deadline_epoch'] <= source['source_sampling_deadline_epoch'] - CUTOFF
        and value['model_deadline_epoch'] <= value['captured_epoch'] + MAX_SECONDS,
        'Model deadline exceeds CPU/source cutoff or four hours')


def capture_context(base, slot, canonical_base, cpu_ready_path, cpu_ready_sha256, topology, instance='v2'):
    """Submit-time observations only; primary leases are acquired by the model."""
    base = Path(base)
    shared.require(topology in TOPOLOGIES and slot in TOPOLOGIES[topology], 'Slot is outside the chosen topology')
    path = source_path(base, cpu_ready_path)
    shared.require(sha(path) == cpu_ready_sha256, 'CPU owner ready SHA differs')
    ready = read(path)
    canonical = lambda p: str(Path(canonical_base) / source_path(base, p).relative_to(base))
    for key in ('allocation', 'source_context', 'upstream_ready', 'transport_ready'):
        shared.require(sha(source_path(base, ready[key + '_path'])) == ready[key + '_sha256'],
                       'CPU dependency SHA differs: ' + key)
    shared.require(sha(source_path(base, ready['job_ready_path'])) == cpu_ready_sha256,
                   'CPU owner job ready differs from published ready')
    source = read(source_path(base, ready['source_context_path']))
    upstream = read(source_path(base, ready['upstream_ready_path']))
    transport = read(source_path(base, ready['transport_ready_path']))
    shared.require(transport['status'] == 'RELAY_LISTENING' and transport['role'] == 'downstream'
        and transport['listen_url'] == 'http://127.0.0.1:28831'
        and transport['upstream_proof']['sha256'] == ready['upstream_ready_sha256']
        and transport['code_sha256'] == sha(Path(__file__).with_name('retrieval_relay_backpressure.py'))
        and transport['transport_core_sha256'] == sha(Path(__file__).with_name('retrieval_relay.py'))
        and all(str(transport['process'][key]) == str(ready['relay'][key]) for key in
                ('pid', 'birth', 'cgroup', 'netns_inode')),
        'Original full-recipe backpressure transport proof differs')
    shared.require(upstream['role'] == 'upstream' and all(
        upstream[key] == source[key] == transport[key] for key in ('source_pid', 'source_birth', 'source_netns_inode')),
        'Original source/upstream identity differs')
    shared.require(alive(ready['owner']) and alive(ready['relay']), 'Original CPU owner/relay is no longer live')
    props = subprocess.check_output(['systemctl', 'show', ready['owner']['unit'], '-p', 'Id', '-p', 'MainPID',
        '-p', 'ActiveState', '-p', 'ControlGroup'], text=True, timeout=15)
    props = dict(line.split('=', 1) for line in props.splitlines() if '=' in line)
    shared.require(props.get('Id') == ready['owner']['unit'] and props.get('MainPID') == str(ready['owner']['pid'])
        and props.get('ActiveState') == 'active' and any(line.split(':', 2)[-1] == props.get('ControlGroup')
        for line in ready['owner']['cgroup'].splitlines()), 'Actual CPU owner unit differs')
    job = source_path(base, ready['job_ready_path']).parent
    shared.require(not any((job / name).exists() for name in ('failure.json', 'complete.json', 'drain-owner.json')),
                   'CPU owner is terminal or draining; no new model admission')
    inventory = {row['index']: row for row in shared.load_pool(Path(__file__).with_name('pool_worker.py')).nvml().values()}
    config = TOPOLOGIES[topology][slot]
    now = time.time()
    value = dict(schema_version=1, mode='NODE10_INDEPENDENT_PRIMARY_RELAY_CONSUMER', run_id=RUN_ID,
        base=str(canonical_base), slot=slot, topology=topology, instance=instance, unit=names(slot, instance)[0], slot_configuration=config,
        model_indices=config['indices'], model_cuda_uuids=[inventory[i]['uuid'] for i in config['indices']],
        initial_gpu_memory=[inventory[i] for i in config['indices']], cpu_ready=ready, source=source,
        cpu_ready_path=canonical(path), cpu_ready_sha256=cpu_ready_sha256,
        reuse_local_relay_ready=canonical(ready['transport_ready_path']), reuse_local_relay_sha256=ready['transport_ready_sha256'],
        upstream_ready=canonical(ready['upstream_ready_path']), upstream_ready_sha256=ready['upstream_ready_sha256'],
        source_context=canonical(ready['source_context_path']), source_context_sha256=ready['source_context_sha256'],
        owner_drain_path=canonical(job / 'drain-owner.json'), captured_epoch=now,
        model_deadline_epoch=min(now + MAX_SECONDS, ready['consumer_deadline_epoch'], ready['deadline_epoch'] - CUTOFF,
                                source['source_sampling_deadline_epoch'] - CUTOFF))
    validate_configuration(value, slot)
    shared.require(value['model_deadline_epoch'] > now + BATCH_RESERVE + 60, 'No reserved whole-batch lifetime remains')
    return value


def context(pool, args):
    path = Path(os.environ['Q7B_NODE10_PRIMARY_CONTEXT'])
    value = pool.read(path)
    shared.require(pool.sha(path) == os.environ['Q7B_NODE10_PRIMARY_CONTEXT_SHA256'], 'Immutable primary context SHA differs')
    validate_configuration(value, value['slot'])
    config = value['slot_configuration']
    shared.require(value['base'] == str(args.base) and args.tp == config['tp']
        and args.model_gpus == ','.join(map(str, config['indices']))
        and (args.model_max_seqs, args.model_batch_tokens, args.model_kv_block_cap, args.model_reserve_gib)
            == (config['seqs'], config['tokens'], config['blocks'], config['reserve'])
        and args.batch_reserve_seconds == BATCH_RESERVE
        and str(args.reuse_local_relay_ready) == value['reuse_local_relay_ready']
        and args.reuse_local_relay_sha256 == value['reuse_local_relay_sha256']
        and args.reuse_local_relay_unit == value['cpu_ready']['owner']['unit']
        and str(args.upstream_ready) == value['upstream_ready']
        and os.environ['Q7B_SLOT_OWNER_DRAIN'] == value['owner_drain_path'], 'Primary arguments differ from immutable context')
    return path, value


class BorrowedDependencies:
    """Observation only: no stop/wait/signal methods and no owned children."""
    def __init__(self, base, value):
        self.base, self.value = Path(base), value
        self.pid = value['cpu_ready']['relay']['pid']
        self.proof = dict(value['cpu_ready']['relay'], process_owned=False, cpu_owner=value['cpu_ready']['owner'])

    def poll(self):
        ready = self.value['cpu_ready']
        job = source_path(self.base, ready['job_ready_path']).parent
        # Drain is handled before the next atomic claim, allowing a current batch to finish.
        return None if (alive(ready['owner']) and alive(ready['relay'])
            and not (job / 'failure.json').exists() and not (job / 'complete.json').exists()) else 1


def borrowed_relay(pool, args):
    _, value = context(pool, args)
    ready = value['cpu_ready']
    for path, digest in ((value['cpu_ready_path'], value['cpu_ready_sha256']),
                         (value['source_context'], value['source_context_sha256']),
                         (value['reuse_local_relay_ready'], value['reuse_local_relay_sha256']),
                         (value['upstream_ready'], value['upstream_ready_sha256']),
                         (ready['allocation_path'], ready['allocation_sha256'])):
        shared.require(pool.sha(source_path(args.base, path)) == digest, 'Borrowed dependency SHA changed')
    borrowed = BorrowedDependencies(args.base, value)
    shared.require(borrowed.poll() is None, 'Independent CPU relay dependency is no longer live')
    return borrowed


def admit(pool, args):
    shared.require(socket.gethostname().split('.')[0] == 'h100-10', 'Only retained node10 is admitted')
    path, value = context(pool, args)
    shared.require(os.stat('/proc/self/ns/mnt').st_ino != os.stat('/proc/1/ns/mnt').st_ino
        and os.stat('/proc/self/ns/net').st_ino == os.stat('/proc/1/ns/net').st_ino,
        'Primary controller needs private retained mounts and shared hostNET')
    borrowed_relay(pool, args)
    evidence = pool.read(args.allocation_evidence)
    actual = process(os.getpid())
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    props = dict(re.findall(r'^(\w+)=(.*)$', evidence['unit_show'], re.MULTILINE))
    group = evidence['cgroup']
    shared.require(evidence['node'] == 'h100-10' and evidence['main_pid'] == os.getpid()
        and type(evidence['controller_birth']) is int and evidence['controller_birth'] == int(actual['birth'])
        and evidence['boot_id'] == boot_id and props.get('Id') == value['unit'] == evidence['unit']
        and props.get('MainPID') == str(os.getpid()) and props.get('ActiveState') == 'active'
        and props.get('ControlGroup') == group == '/system.slice/' + evidence['unit']
        and any(line.split(':', 2)[-1] == group for line in actual['cgroup'].splitlines())
        and evidence['resource_limits']['runtime_max_seconds'] <= MAX_SECONDS, 'Own unit/birth/boot/cgroup differs')
    inventory = {row['index']: row for row in pool.nvml().values()}
    model, uuids = value['model_indices'], value['model_cuda_uuids']
    shared.require([inventory[i]['uuid'] for i in model] == uuids, 'Actual numeric GPU UUID changed')
    locks, primary_paths = [], {}
    lease = owner.lease_path(args.base, RUN_ID, value['cpu_ready']['owner'], value['slot'])
    try:
        for uuid in sorted(uuids):
            primary = Path(args.base) / 'pool/gpu-locks' / (uuid + '.lock')
            fd = pool.try_lock(primary)
            shared.require(fd is not None, 'Another cooperating model owns primary UUID ' + uuid)
            locks.append(fd); primary_paths[uuid] = str(primary)
        fd = pool.try_lock(lease)
        shared.require(fd is not None, 'CPU owner live-slot lease is already held')
        locks.append(fd)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, model)), CUDA_DEVICE_ORDER='PCI_BUS_ID')
        result = subprocess.check_output([sys.executable, str(Path(__file__).resolve()), '--cuda-probe',
            '--expected-uuids', ','.join(uuids)], env=env, text=True, timeout=45)
        shared.require(json.loads(result)['actual_cuda_runtime_uuids'] == uuids, 'Actual CUDA UUID order differs')
        borrowed_relay(pool, args)
        shared.require(not source_path(args.base, value['owner_drain_path']).exists()
            and value['model_deadline_epoch'] > time.time() + BATCH_RESERVE + 60, 'CPU owner drained or lifetime expired during admission')
        return dict(job_id='relay10-primary-' + str(os.getpid()), node='h100-10', run_id=RUN_ID,
            instance=value['instance'],
            controller_pid=os.getpid(), controller_birth=int(actual['birth']), boot_id=boot_id,
            allocation_kind='HUMAN_AUTHORIZED_NODE10_INDEPENDENT_PRIMARY', native_unit=evidence['unit'], native_cgroup=group,
            model_indices=model, model_cuda_uuids=uuids, model_cuda_visible_devices=','.join(map(str, model)),
            actual_cuda_runtime_uuids=uuids, tensor_parallel_size=args.tp, model_engine_count=1,
            original_cuda_visible_devices=evidence.get('original_cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES')),
            initial_gpu_memory=[inventory[i] for i in model], foreign_initial_contexts=shared.foreign_contexts(set(uuids)),
            foreign_contexts_allowed=True, foreign_processes_signalled=False, primary_uuid_lock_paths=primary_paths,
            cpu_live_slot_lease=str(lease), source_owner=value['cpu_ready']['owner'],
            source_owner_allocation_sha256=value['cpu_ready']['allocation_sha256'],
            source_model_deadline_epoch=value['model_deadline_epoch'], source_retrieval_context=value['source'],
            slot_context=str(path), slot_context_sha256=pool.sha(path), local_relay_reused=True,
            local_relay_owned=False, local_relay_process=dict(value['cpu_ready']['relay'], process_owned=False),
            upstream_retriever_owned=False, shared_retriever_owned_by_consumer=False,
            upstream_ready=str(args.upstream_ready), upstream_ready_sha256=pool.sha(args.upstream_ready),
            allocation_evidence=str(args.allocation_evidence), allocation_evidence_sha256=pool.sha(args.allocation_evidence),
            native_resource_limits=evidence['resource_limits'], captured_epoch=time.time()), locks
    except BaseException:
        for fd in locks: os.close(fd)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'upstream-ready', 'token-file', 'reuse-local-relay-ready', 'reuse-local-relay-sha256',
                 'reuse-local-relay-unit', 'allocation-evidence', 'expected-uuids', 'job-dir', 'launch-dir', 'handoff-marker'):
        parser.add_argument('--' + name)
    parser.add_argument('--pool-worker', default=str(Path(__file__).with_name('draining_pool_worker.py')))
    parser.add_argument('--relay-script', default=str(Path(__file__).with_name('retrieval_relay_backpressure.py')))
    parser.add_argument('--model-gpus')
    parser.add_argument('--tp', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--model-max-seqs', type=int, default=64)
    parser.add_argument('--model-batch-tokens', type=int, default=8192)
    parser.add_argument('--model-kv-block-cap', type=int, default=4096)
    parser.add_argument('--model-reserve-gib', type=float, default=1.)
    parser.add_argument('--run-seconds', type=int, default=MAX_SECONDS)
    parser.add_argument('--batch-reserve-seconds', type=int, default=BATCH_RESERVE)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--sampler-loop', action='store_true')
    parser.add_argument('--cuda-probe', action='store_true')
    args = parser.parse_args()
    if args.cuda_probe:
        print(json.dumps(dict(actual_cuda_runtime_uuids=shared.actual_uuids(args.expected_uuids.split(',')))))
        return
    shared.require(args.base and sys.platform == 'linux' and args.model_max_seqs > 0
        and args.model_batch_tokens >= 4096 and args.model_kv_block_cap >= 256
        and 0 < args.model_reserve_gib <= 1.5 and args.batch_reserve_seconds == BATCH_RESERVE,
        'Invalid remote capacity/whole-batch reserve settings')
    pool = shared.load_pool(args.pool_worker)
    def terminate(signum, frame): raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate); signal.signal(signal.SIGINT, terminate)
    if args.sampler_loop:
        pool.single_gpu_sampler = lambda binding: shared.multi_sampler(pool, binding, args)
        try: pool.sample_loop(args)
        except BaseException as error:
            pool.publish(Path(args.launch_dir) / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
                traceback=traceback.format_exc(), global_goal_complete=False, epoch=time.time()))
            raise
    else:
        original = consumer.admit, consumer.borrowed_relay, consumer.__file__
        try:
            consumer.admit, consumer.borrowed_relay = admit, borrowed_relay
            # The reused lifecycle starts this same entry for probes and sampler children.
            consumer.__file__ = __file__
            consumer.controller(pool, args)
        finally:
            consumer.admit, consumer.borrowed_relay, consumer.__file__ = original


if __name__ == '__main__': main()
