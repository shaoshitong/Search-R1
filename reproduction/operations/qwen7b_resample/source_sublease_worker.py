"""Model-only h100-5 consumer holding child leases of the live V6 allocation.

The original controller and retriever are observed dependencies, never owned
children. Only distinct per-UUID sublease files are locked by this controller.
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
PARENT_PID, RETRIEVER_PID, PEER_PID = 1490214, 1490465, 2013773


def context(pool, args):
    path = Path(os.environ['Q7B_SOURCE_SUBLEASE_CONTEXT'])
    shared.require(path.samefile(args.reuse_local_relay_ready)
                   and pool.sha(path) == args.reuse_local_relay_sha256,
                   'Source sublease context path or SHA differs')
    value = pool.read(path)
    shared.require(value['schema_version'] in (1, 2) and value['mode'] == 'SOURCE5_SAME_RUN_GPU_SUBLEASE'
                   and value['run_id'] == pool.RUN_ID, 'Source sublease run identity differs')
    tp = 4 if value['schema_version'] == 1 else 1
    shared.require(len(value['model_indices']) == len(value['model_cuda_uuids']) == tp == args.tp
                   and value.get('tensor_parallel_size', 4) == tp,
                   'Source sublease tensor parallel size differs from its schema and arguments')
    parent, retriever = value['parent'], value['retriever']
    shared.require(parent['pid'] == PARENT_PID and retriever['pid'] == RETRIEVER_PID
                   and parent['unit'] == 'sst-q7b-g5-sharedvram-v6-' + pool.RUN_ID[:12] + '.service'
                   and args.reuse_local_relay_unit == parent['unit']
                   and parent['unit'] in parent['cgroup']
                   and retriever['cgroup'] == parent['cgroup']
                   and retriever['netns_inode'] == parent['netns_inode'],
                   'Original parent or retriever binding differs')
    return path, value


def alive(proof):
    try:
        pid = int(proof['pid'])
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return (fields[0] not in ('Z', 'X') and fields[19] == str(proof['birth'])
                and Path('/proc/%d/cgroup' % pid).read_text() == proof['cgroup']
                and os.stat('/proc/%d/ns/net' % pid).st_ino == proof['netns_inode'])
    except (FileNotFoundError, ProcessLookupError):
        return False


class BorrowedSource:
    """Popen-shaped liveness observation with no stop/wait/ownership methods."""
    def __init__(self, parent, retriever):
        self.parent = dict(parent)
        self.pid = int(retriever['pid'])
        self.proof = dict(retriever, source_parent=dict(parent), process_owned=False)

    def poll(self):
        return None if alive(self.parent) and alive(self.proof) else 1


def borrowed_relay(pool, args):
    _, value = context(pool, args)
    borrowed = BorrowedSource(value['parent'], value['retriever'])
    shared.require(borrowed.poll() is None, 'Original source parent or retriever is no longer live')
    return borrowed


def held_lock(path, owner, expected, label):
    """Read another controller's FD and POSIX lease without opening its file."""
    fd = expected['fd']
    shared.require(type(fd) is int and fd >= 0, 'Invalid ' + label + ' GPU lock FD')
    actual = path.stat()
    held = os.stat('/proc/%d/fd/%d' % (owner['pid'], fd))
    shared.require((actual.st_dev, actual.st_ino) == (held.st_dev, held.st_ino)
                   == (expected['device'], expected['inode']), label + ' GPU lock FD device/inode differs')
    detail = Path('/proc/%d/fdinfo/%d' % (owner['pid'], fd)).read_text()
    pattern = (r'^lock:\s+\d+:\s+POSIX\s+ADVISORY\s+WRITE\s+' + str(owner['pid'])
               + r'\s+([0-9a-fA-F]+):([0-9a-fA-F]+):(\d+)\s+0\s+EOF\s*$')
    matches = re.findall(pattern, detail, re.MULTILINE)
    shared.require(any((int(major, 16), int(minor, 16), int(inode))
                       == (os.major(actual.st_dev), os.minor(actual.st_dev), actual.st_ino)
                       for major, minor, inode in matches), label + ' POSIX WRITE GPU lock is not evidenced')
    return dict(path=str(path), fd=fd, device=actual.st_dev, inode=actual.st_ino,
                fdinfo_sha256=hashlib.sha256(detail.encode()).hexdigest())


def parent_lock(base, parent, uuid, expected):
    return held_lock(base / 'pool/gpu-locks' / (uuid + '.lock'), parent, expected, 'Original parent')


def peer_admission(pool, value, directory, model, uuids):
    """A schema2 peer authorizes admission only; it is not a runtime dependency."""
    peer, parent = value['peer_sublease'], value['parent']
    shared.require(peer['pid'] == PEER_PID and alive(peer)
                   and peer['unit'] != parent['unit'] and peer['unit'] in peer['cgroup']
                   and peer['netns_inode'] == parent['netns_inode']
                   and os.stat('/proc/%d/ns/mnt' % PEER_PID).st_ino == peer['mntns_inode'] == parent['mntns_inode'],
                   'Original peer sublease process identity differs')
    shared.require(pool.sha(peer['allocation_path']) == peer['allocation_sha256'], 'Peer allocation SHA differs')
    allocation = pool.read(peer['allocation_path'])
    peer_context = pool.read(allocation['source_context'])
    shared.require(pool.sha(allocation['source_context']) == allocation['source_context_sha256']
                   and peer_context['run_id'] == allocation.get('run_id', pool.RUN_ID) == pool.RUN_ID
                   and peer_context['parent']['pid'] == parent['pid']
                   and str(peer_context['parent']['birth']) == str(parent['birth'])
                   and allocation['node'] == 'h100-5' and allocation['controller_pid'] == PEER_PID
                   and allocation['native_unit'] == peer['unit']
                   and any(line.split(':', 2)[-1] == allocation['native_cgroup'] for line in peer['cgroup'].splitlines())
                   and allocation['source_parent']['pid'] == parent['pid']
                   and str(allocation['source_parent']['birth']) == str(parent['birth'])
                   and allocation['source_parent']['allocation_sha256'] == parent['allocation_sha256'],
                   'Peer allocation does not bind this same-run original parent')
    mapping = dict(zip(allocation['model_indices'], allocation['model_cuda_uuids']))
    shared.require(all(mapping.get(index) == uuid for index, uuid in zip(model, uuids)),
                   'Peer allocation does not contain the selected physical GPU UUID')
    proofs = {}
    for uuid in uuids:
        path = directory / (uuid + '.lock')
        shared.require(Path(allocation['gpu_sublease_paths'][uuid]).samefile(path), 'Peer root-child lease path differs')
        proofs[uuid] = held_lock(path, peer, peer['gpu_locks'][uuid], 'Original peer')
    return dict(peer, verified_gpu_locks=proofs, runtime_dependency=False)


def admit(pool, args):
    shared.require(socket.gethostname().split('.')[0] == 'h100-5', 'Source sublease is restricted to h100-5')
    path, value = context(pool, args)
    parent = value['parent']
    borrowed = borrowed_relay(pool, args)
    shared.require(os.stat('/proc/self/ns/net').st_ino == parent['netns_inode']
                   and os.stat('/proc/self/ns/mnt').st_ino == parent['mntns_inode']
                   and os.stat('/proc/%d/ns/mnt' % PARENT_PID).st_ino == parent['mntns_inode'],
                   'Consumer is outside the evidenced parent mount/network namespaces')
    evidence = pool.read(args.allocation_evidence)
    props = dict(re.findall(r'^(\w+)=(.*)$', evidence['unit_show'], re.MULTILINE))
    group = evidence['cgroup']
    shared.require(evidence['node'] == 'h100-5' and evidence['main_pid'] == os.getpid()
                   and props.get('Id') == evidence['unit'] and props.get('MainPID') == str(os.getpid())
                   and props.get('ActiveState') == 'active' and props.get('ControlGroup') == group
                   and group.endswith('/' + evidence['unit']) and evidence['unit'] != parent['unit']
                   and any(line.split(':', 2)[-1] == group for line in Path('/proc/self/cgroup').read_text().splitlines()),
                   'Sublease controller is outside its own active bounded unit')
    allocation_path = Path(parent['allocation_path'])
    shared.require(pool.sha(allocation_path) == parent['allocation_sha256'], 'Original parent allocation SHA differs')
    allocation = pool.read(allocation_path)
    shared.require(allocation['node'] == 'h100-5' and allocation['controller_pid'] == PARENT_PID
                   and allocation['native_unit'] == parent['unit']
                   and any(line.split(':', 2)[-1] == allocation['native_cgroup'] for line in parent['cgroup'].splitlines())
                   and allocation.get('run_id', pool.RUN_ID) == pool.RUN_ID, 'Original parent allocation identity differs')
    model = shared.indices(args.model_gpus, args.tp)
    inventory = {item['index']: item for item in pool.nvml().values()}
    uuids = [inventory[index]['uuid'] for index in model]
    parent_mapping = dict(zip(allocation['model_indices'] + allocation['retrieval_indices'],
                              allocation['model_cuda_uuids'] + allocation['retrieval_cuda_uuids']))
    shared.require(model == value['model_indices'] and uuids == value['model_cuda_uuids']
                   and all(parent_mapping.get(index) == uuid for index, uuid in zip(model, uuids)),
                   'Subleased physical GPU UUID mapping differs from the parent allocation')
    base = Path(args.base)
    parent_locks = {uuid: parent_lock(base, parent, uuid, value['parent_gpu_locks'][uuid]) for uuid in uuids}
    directory = base / 'pool/gpu-subleases' / pool.RUN_ID / ('%d-%s' % (PARENT_PID, parent['birth']))
    peer = None
    if value['schema_version'] == 2:
        peer = peer_admission(pool, value, directory, model, uuids)
        directory = directory / ('co-share-with-%d-%s' % (peer['pid'], peer['birth']))
    locks, leases = [], {}
    try:
        for uuid in sorted(uuids):
            lease = directory / (uuid + '.lock')
            fd = pool.try_lock(lease)
            shared.require(fd is not None, 'Another same-run child holds GPU sublease ' + uuid)
            locks.append(fd)
            leases[uuid] = str(lease)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, model)), CUDA_DEVICE_ORDER='PCI_BUS_ID')
        result = subprocess.check_output([sys.executable, str(Path(consumer.__file__).resolve()),
            '--cuda-probe', '--expected-uuids', ','.join(uuids)], env=env, text=True, timeout=45)
        shared.require(json.loads(result)['actual_cuda_runtime_uuids'] == uuids,
                       'Subleased numeric CUDA mapping differs from the actual UUIDs')
        shared.require(borrowed.poll() is None, 'Original source identity changed during sublease admission')
        upstream = pool.read(args.upstream_ready)
        shared.require(upstream['role'] == 'upstream' and upstream['source_pid'] == RETRIEVER_PID
                       and upstream['listen_url'].rstrip('/') == 'http://127.0.0.1:28831',
                       'Direct original retriever endpoint differs')
        return dict(job_id='source5-model-' + str(os.getpid()), node='h100-5', controller_pid=os.getpid(), run_id=pool.RUN_ID,
            allocation_kind='HUMAN_AUTHORIZED_SAME_RUN_SOURCE_GPU_SUBLEASE', gpu_exclusive_allocation=False,
            native_unit=evidence['unit'], native_cgroup=group, tensor_parallel_size=args.tp, model_engine_count=1,
            model_indices=model, model_cuda_uuids=uuids, model_cuda_visible_devices=','.join(map(str, model)),
            actual_cuda_runtime_uuids=uuids, original_cuda_visible_devices=evidence.get('original_cuda_visible_devices'),
            initial_gpu_memory=[inventory[index] for index in model],
            observed_existing_gpu_contexts=shared.foreign_contexts(set(uuids)),
            existing_contexts_may_include_same_run_parent=True, foreign_contexts_allowed=True,
            foreign_processes_signalled=False, source_parent=parent, original_parent_gpu_locks=parent_locks,
            peer_sublease_startup_proof=peer, peer_is_runtime_dependency=False,
            gpu_sublease_paths=leases, primary_gpu_locks_modified=False,
            source_context=str(path), source_context_sha256=pool.sha(path),
            source_sublease_worker_sha256=pool.sha(Path(__file__)),
            direct_original_retriever=True, local_relay_owned=False, local_relay_process=borrowed.proof,
            shared_retriever_owned_by_consumer=False, source_parent_owned=False,
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
