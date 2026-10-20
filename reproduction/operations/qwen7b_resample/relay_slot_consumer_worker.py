"""Finite h100-10 model slots borrowing an independently owned CPU relay."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import relay_consumer_worker as consumer
import retrieval_relay_owner as relay_owner
import source_slot_consumer_worker as source_slot

shared = consumer.shared


def ready(pool, args):
    path = Path(args.reuse_local_relay_ready)
    value = relay_owner.bound_read(pool, path, args.reuse_local_relay_sha256)
    shared.require(value['status'] == 'RETRIEVAL_RELAY_OWNER_READY' and value['role'] == 'downstream'
        and value['run_id'] == pool.RUN_ID and value['owner']['unit'] == args.reuse_local_relay_unit,
        'Independent node10 relay owner generation differs')
    allocation = relay_owner.bound_read(pool, value['allocation_path'], value['allocation_sha256'])
    shared.require(allocation['allocation_role'] == 'RETRIEVAL_RELAY_CPU_OWNER' and allocation['role'] == 'downstream'
        and allocation['owner'] == value['owner'] and allocation['slots'] == value['slots']
        and allocation['run_id'] == pool.RUN_ID, 'CPU relay primary allocation differs')
    relay_owner.check_terminal(pool, value['job_ready_path'])
    shared.require(not (Path(value['job_ready_path']).parent / 'drain-request.json').exists(), 'CPU relay owner is draining')
    transport = relay_owner.bound_read(pool, value['transport_ready_path'], value['transport_ready_sha256'])
    upstream = relay_owner.bound_read(pool, value['upstream_owner_ready'], value['upstream_owner_ready_sha256'])
    relay_owner.check_terminal(pool, upstream['job_ready_path'])
    shared.require(transport['role'] == 'downstream' and transport['status'] == 'RELAY_LISTENING'
        and transport['listen_url'] == 'http://127.0.0.1:28831'
        and transport['code_sha256'] == pool.sha(args.relay_script)
        and Path(args.upstream_ready).samefile(upstream['transport_ready_path'])
        and transport['upstream_proof']['sha256'] == pool.sha(args.upstream_ready)
        and upstream['source_owner'] == value['source_owner'], 'Original downstream relay/upstream identity differs')
    return path, value, allocation


class BorrowedCPUOwner:
    def __init__(self, value):
        self.value = value
        self.pid = value['relay']['pid']
        self.proof = dict(value['relay'], cpu_owner=value['owner'], process_owned=False)

    def poll(self):
        try:
            relay_owner.same_process(self.value['owner'])
            relay_owner.same_process(self.value['relay'])
            directory = Path(self.value['job_ready_path']).parent
            if (directory / 'complete.json').exists() or (directory / 'failure.json').exists():return 1
        except (FileNotFoundError, ProcessLookupError, ValueError):
            return 1
        return None


def borrowed_relay(pool, args):
    _, value, _ = ready(pool, args)
    borrowed = BorrowedCPUOwner(value)
    shared.require(borrowed.poll() is None, 'Original independent CPU relay owner is no longer live')
    return borrowed


def admit(pool, args):
    shared.require(socket.gethostname().split('.')[0] == 'h100-10', 'Remote model slots support only h100-10')
    path, value, allocation = ready(pool, args)
    borrowed = borrowed_relay(pool, args)
    evidence, actual = relay_owner.own_evidence(pool, args, 'h100-10')
    owner = value['owner']
    shared.require(actual['mntns_inode'] == owner['mntns_inode'] and actual['netns_inode'] == owner['netns_inode']
        and actual['unit'] != owner['unit'], 'Model slot must share CPU owner namespaces with an independent cgroup')
    name = os.environ['Q7B_RELAY_SLOT_NAME']
    shared.require(name in relay_owner.SLOTS and name in value['slots'], 'Undeclared node10 model slot')
    slot = value['slots'][name]
    shared.require(all(slot[key] == expected for key, expected in relay_owner.SLOTS[name].items()), 'Node10 finite slot specification differs')
    model = shared.indices(args.model_gpus, args.tp)
    shared.require(model == slot['indices'] and args.tp == slot['tp']
        and args.model_max_seqs == slot['seqs'] and args.model_batch_tokens == slot['tokens']
        and args.model_kv_block_cap == slot['blocks'] and args.model_reserve_gib == slot['reserve'], 'Node10 slot model topology/capacity differs')
    deadline = min(value['consumer_deadline_epoch'], evidence['unit_hard_end_epoch'] - 45)
    shared.require(deadline > time.time() + args.batch_reserve_seconds, 'Model slot absolute deadline is exhausted')
    inventory = {row['index']: row for row in pool.nvml().values()}
    uuids = [inventory[index]['uuid'] for index in model]
    shared.require(uuids == slot['cuda_uuids'], 'Node10 slot physical UUID mapping differs')
    base = Path(args.base)
    proofs = {uuid: source_slot.held_lock(base / 'pool/gpu-locks' / (uuid + '.lock'), owner,
        allocation['gpu_lock_proofs'][uuid]) for uuid in uuids}
    locks, paths = [], {}
    try:
        for uuid in sorted(uuids):
            lease = relay_owner.source.slot_directory(base, pool.RUN_ID, owner, name) / (uuid + '.lock')
            fd = pool.try_lock(lease)
            shared.require(fd is not None, 'This finite node10 model slot is already occupied')
            locks.append(fd);paths[uuid] = str(lease)
        output = subprocess.check_output([sys.executable, str(Path(consumer.__file__).resolve()), '--cuda-probe',
            '--expected-uuids', ','.join(uuids)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, model)),
            CUDA_DEVICE_ORDER='PCI_BUS_ID'), text=True, timeout=45)
        shared.require(json.loads(output)['actual_cuda_runtime_uuids'] == uuids and borrowed.poll() is None,
                       'Actual node10 model CUDA UUID/CPU owner differs')
        return dict(job_id='relay-slot-%s-%d' % (name, os.getpid()), run_id=pool.RUN_ID, node='h100-10',
            controller_pid=os.getpid(), native_unit=evidence['unit'], native_cgroup=evidence['cgroup'],
            model_indices=model, model_cuda_uuids=uuids, model_cuda_visible_devices=','.join(map(str, model)),
            tensor_parallel_size=args.tp, model_engine_count=1, actual_cuda_runtime_uuids=uuids,
            original_cuda_visible_devices=evidence.get('original_cuda_visible_devices'),
            source_owner=owner, source_owner_allocation_sha256=value['allocation_sha256'],
            source_owner_ready=str(path), source_owner_ready_sha256=pool.sha(path),
            source_model_deadline_epoch=deadline, finite_slot=name, slot_lease_paths=paths,
            original_primary_lock_proofs=proofs, primary_gpu_locks_modified=False,
            source_retrieval_owner=value['source_owner'], local_relay_process=borrowed.proof,
            shared_retriever_owned_by_consumer=False, source_parent_owned=False,
            foreign_processes_signalled=False, allocation_evidence=str(args.allocation_evidence),
            allocation_evidence_sha256=pool.sha(args.allocation_evidence),
            native_resource_limits=evidence['resource_limits'], epoch=time.time()), locks
    except BaseException:
        for fd in locks:os.close(fd)
        raise


def main():
    consumer.admit = admit
    consumer.borrowed_relay = borrowed_relay
    consumer.main()


if __name__ == '__main__':
    main()
