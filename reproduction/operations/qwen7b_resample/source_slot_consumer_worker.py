"""A finite declared model slot borrowing an independent retrieval owner."""
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
import retrieval_only_owner as owner_module

shared = consumer.shared


class BorrowedOwner:
    def __init__(self, proof):
        self.value = proof
        self.pid = proof['retriever']['pid']
        self.proof = dict(proof['retriever'], source_owner=proof['owner'], process_owned=False)

    def poll(self):
        try:
            for expected in (self.value['owner'], self.value['retriever']):
                actual = owner_module.process(expected['pid'])
                if any(actual[key] != expected[key] for key in ('birth', 'cgroup', 'netns_inode', 'mntns_inode')):
                    return 1
        except (FileNotFoundError, ProcessLookupError, ValueError):
            return 1
        return None


def ready(pool, args):
    path = Path(os.environ['Q7B_SLOT_OWNER_READY'])
    shared.require(path.samefile(args.reuse_local_relay_ready)
        and pool.sha(path) == os.environ['Q7B_SLOT_OWNER_READY_SHA256'] == args.reuse_local_relay_sha256,
        'Resource owner ready path/SHA differs')
    value = pool.read(path)
    shared.require(value['status'] == 'RETRIEVAL_ONLY_OWNER_READY' and value['run_id'] == pool.RUN_ID
        and value['owner']['unit'] == args.reuse_local_relay_unit, 'Resource owner generation differs')
    shared.require(not (path.parent / 'drain-request.json').exists(), 'Resource owner is draining; no new slot admission')
    return path, value


def borrowed_relay(pool, args):
    _, value = ready(pool, args)
    borrowed = BorrowedOwner(value)
    shared.require(borrowed.poll() is None, 'Original retrieval resource owner is no longer live')
    return borrowed


def held_lock(path, owner, expected):
    actual = path.stat()
    held = os.stat('/proc/%d/fd/%d' % (owner['pid'], expected['fd']))
    shared.require((actual.st_dev, actual.st_ino) == (held.st_dev, held.st_ino) == (expected['device'], expected['inode']),
        'Independent owner primary GPU lock FD differs')
    detail = Path('/proc/%d/fdinfo/%d' % (owner['pid'], expected['fd'])).read_text()
    pattern = r'^lock:\s+\d+:\s+POSIX\s+ADVISORY\s+WRITE\s+%d\s+([0-9a-fA-F]+):([0-9a-fA-F]+):(\d+)\s+0\s+EOF\s*$' % owner['pid']
    shared.require(any((int(a, 16), int(b, 16), int(c)) == (os.major(actual.st_dev), os.minor(actual.st_dev), actual.st_ino)
        for a, b, c in re.findall(pattern, detail, re.MULTILINE)), 'Owner POSIX WRITE GPU lease is not evidenced')
    return dict(expected, fdinfo_sha256=hashlib.sha256(detail.encode()).hexdigest())


def admit(pool, args):
    shared.require(socket.gethostname().split('.')[0] == 'h100-5', 'Source slots currently support only h100-5')
    path, value = ready(pool, args)
    borrowed = borrowed_relay(pool, args)
    owner = value['owner']
    actual = owner_module.process(os.getpid())
    shared.require(actual['netns_inode'] == owner['netns_inode'] and actual['mntns_inode'] == owner['mntns_inode'],
        'Model slot must use the exact independent owner namespaces')
    allocation = pool.read(value['allocation_path'])
    shared.require(pool.sha(value['allocation_path']) == value['allocation_sha256']
        and allocation['allocation_role'] == 'RETRIEVAL_ONLY_PRIMARY_LEASE_OWNER'
        and allocation['owner'] == owner and allocation['run_id'] == pool.RUN_ID, 'Owner allocation binding differs')
    name = os.environ['Q7B_SLOT_NAME']
    shared.require(name in value['slots'] and value['slots'] == allocation['slots'], 'Undeclared resource-owner model slot')
    slot = value['slots'][name]
    shared.require(name in owner_module.SLOTS
        and all(slot[key] == expected for key, expected in owner_module.SLOTS[name].items()), 'Slot specification differs')
    model = shared.indices(args.model_gpus, args.tp)
    shared.require(model == slot['indices'] and args.tp == slot['tp']
        and args.model_max_seqs == slot['seqs'] and args.model_batch_tokens == slot['tokens']
        and args.model_kv_block_cap == slot['blocks'] and args.model_reserve_gib == slot['reserve'], 'Slot model topology/capacity differs')
    shared.require(time.time() + args.run_seconds <= value['consumer_deadline_epoch'], 'Model deadline exceeds owner drain margin')
    upstream = pool.read(args.upstream_ready)
    shared.require(upstream['role'] == 'upstream' and upstream['source_pid'] == value['retriever']['pid']
        and upstream['listen_url'] == 'http://127.0.0.1:28831', 'Original owner retrieval endpoint differs')
    evidence = pool.read(args.allocation_evidence)
    props = dict(re.findall(r'^(\w+)=(.*)$', evidence['unit_show'], re.MULTILINE))
    group = evidence['cgroup']
    shared.require(evidence['node'] == 'h100-5' and evidence['main_pid'] == os.getpid()
        and props.get('Id') == evidence['unit'] and props.get('MainPID') == str(os.getpid())
        and props.get('ActiveState') == 'active' and props.get('ControlGroup') == group
        and evidence['unit'] != owner['unit'] and group.endswith('/' + evidence['unit'])
        and any(line.split(':', 2)[-1] == group for line in actual['cgroup'].splitlines()), 'Slot controller own cgroup differs')
    deadline = value['consumer_deadline_epoch']
    if 'unit_hard_end_epoch' in evidence:
        deadline = min(deadline, evidence['unit_hard_end_epoch'] - 45)
    inventory = {row['index']: row for row in pool.nvml().values()}
    uuids = [inventory[index]['uuid'] for index in model]
    shared.require(uuids == slot['cuda_uuids'], 'Declared slot physical UUID mapping differs')
    base = Path(args.base)
    proofs = {uuid: held_lock(base / 'pool/gpu-locks' / (uuid + '.lock'), owner, allocation['gpu_lock_proofs'][uuid]) for uuid in uuids}
    locks, paths = [], {}
    try:
        for uuid in sorted(uuids):
            lease = owner_module.slot_directory(base, pool.RUN_ID, owner, name) / (uuid + '.lock')
            fd = pool.try_lock(lease)
            shared.require(fd is not None, 'Declared model slot is already occupied')
            locks.append(fd)
            paths[uuid] = str(lease)
        result = subprocess.check_output([sys.executable, str(Path(consumer.__file__).resolve()), '--cuda-probe',
            '--expected-uuids', ','.join(uuids)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, model)),
            CUDA_DEVICE_ORDER='PCI_BUS_ID'), text=True, timeout=45)
        shared.require(json.loads(result)['actual_cuda_runtime_uuids'] == uuids and borrowed.poll() is None,
            'Actual model slot CUDA UUID/source identity differs')
        return dict(job_id='source-slot-%s-%d' % (name, os.getpid()), run_id=pool.RUN_ID, node='h100-5',
            controller_pid=os.getpid(), native_unit=evidence['unit'], native_cgroup=group,
            model_indices=model, model_cuda_uuids=uuids, model_cuda_visible_devices=','.join(map(str, model)),
            tensor_parallel_size=args.tp, model_engine_count=1, actual_cuda_runtime_uuids=uuids,
            original_cuda_visible_devices=evidence.get('original_cuda_visible_devices'),
            initial_gpu_memory=[inventory[index] for index in model], source_owner=owner,
            source_owner_allocation_sha256=value['allocation_sha256'], source_owner_ready=str(path),
            source_owner_ready_sha256=pool.sha(path), source_model_deadline_epoch=deadline,
            finite_slot=name, slot_lease_paths=paths,
            original_primary_lock_proofs=proofs, primary_gpu_locks_modified=False,
            local_relay_process=borrowed.proof, shared_retriever_owned_by_consumer=False,
            source_parent_owned=False, foreign_processes_signalled=False,
            allocation_evidence=str(args.allocation_evidence), allocation_evidence_sha256=pool.sha(args.allocation_evidence),
            native_resource_limits=evidence['resource_limits'], epoch=time.time()), locks
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
