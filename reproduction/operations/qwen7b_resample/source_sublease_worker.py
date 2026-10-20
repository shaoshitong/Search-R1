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
PARENT_PID, RETRIEVER_PID = 1490214, 1490465


def context(pool, args):
    path = Path(os.environ['Q7B_SOURCE_SUBLEASE_CONTEXT'])
    shared.require(path.samefile(args.reuse_local_relay_ready)
                   and pool.sha(path) == args.reuse_local_relay_sha256,
                   'Source sublease context path or SHA differs')
    value = pool.read(path)
    shared.require(value['schema_version'] == 1 and value['mode'] == 'SOURCE5_SAME_RUN_GPU_SUBLEASE'
                   and value['run_id'] == pool.RUN_ID, 'Source sublease run identity differs')
    parent, retriever = value['parent'], value['retriever']
    shared.require(parent['pid'] == PARENT_PID and retriever['pid'] == RETRIEVER_PID
                   and parent['unit'] == 'sst-q7b-g5-sharedvram-v6-' + pool.RUN_ID[:12] + '.service'
                   and args.reuse_local_relay_unit == parent['unit']
                   and parent['unit'] in parent['cgroup']
                   and retriever['cgroup'] == parent['cgroup']
                   and retriever['netns_inode'] == parent['netns_inode'],
                   'Original parent or retriever binding differs')
    return path, value


class BorrowedSource:
    """Popen-shaped liveness observation with no stop/wait/ownership methods."""
    def __init__(self, parent, retriever):
        self.parent = dict(parent)
        self.pid = int(retriever['pid'])
        self.proof = dict(retriever, source_parent=dict(parent), process_owned=False)

    def poll(self):
        try:
            for proof in (self.parent, self.proof):
                pid = int(proof['pid'])
                fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
                if (fields[0] in ('Z', 'X') or fields[19] != str(proof['birth'])
                        or Path('/proc/%d/cgroup' % pid).read_text() != proof['cgroup']
                        or os.stat('/proc/%d/ns/net' % pid).st_ino != proof['netns_inode']):
                    return 1
        except (FileNotFoundError, ProcessLookupError):
            return 1
        return None


def borrowed_relay(pool, args):
    _, value = context(pool, args)
    borrowed = BorrowedSource(value['parent'], value['retriever'])
    shared.require(borrowed.poll() is None, 'Original source parent or retriever is no longer live')
    return borrowed


def parent_lock(base, parent, uuid, expected):
    """Read the parent's original FD and POSIX lease; never open/lock its file."""
    fd = expected['fd']
    shared.require(type(fd) is int and fd >= 0, 'Invalid parent GPU lock FD')
    path = base / 'pool/gpu-locks' / (uuid + '.lock')
    actual = path.stat()
    held = os.stat('/proc/%d/fd/%d' % (parent['pid'], fd))
    shared.require((actual.st_dev, actual.st_ino) == (held.st_dev, held.st_ino)
                   == (expected['device'], expected['inode']), 'Parent GPU lock FD device/inode differs')
    detail = Path('/proc/%d/fdinfo/%d' % (parent['pid'], fd)).read_text()
    pattern = (r'^lock:\s+\d+:\s+POSIX\s+ADVISORY\s+WRITE\s+' + str(parent['pid'])
               + r'\s+([0-9a-fA-F]+):([0-9a-fA-F]+):(\d+)\s+0\s+EOF\s*$')
    matches = re.findall(pattern, detail, re.MULTILINE)
    shared.require(any((int(major, 16), int(minor, 16), int(inode))
                       == (os.major(actual.st_dev), os.minor(actual.st_dev), actual.st_ino)
                       for major, minor, inode in matches), 'Original parent POSIX WRITE GPU lock is not evidenced')
    return dict(path=str(path), fd=fd, device=actual.st_dev, inode=actual.st_ino,
                fdinfo_sha256=hashlib.sha256(detail.encode()).hexdigest())


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
    model = shared.indices(args.model_gpus, 4)
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
        return dict(job_id='source5-model-' + str(os.getpid()), node='h100-5', controller_pid=os.getpid(),
            allocation_kind='HUMAN_AUTHORIZED_SAME_RUN_SOURCE_GPU_SUBLEASE', gpu_exclusive_allocation=False,
            native_unit=evidence['unit'], native_cgroup=group, tensor_parallel_size=4, model_engine_count=1,
            model_indices=model, model_cuda_uuids=uuids, model_cuda_visible_devices=','.join(map(str, model)),
            actual_cuda_runtime_uuids=uuids, original_cuda_visible_devices=evidence.get('original_cuda_visible_devices'),
            initial_gpu_memory=[inventory[index] for index in model],
            observed_existing_gpu_contexts=shared.foreign_contexts(set(uuids)),
            existing_contexts_may_include_same_run_parent=True, foreign_contexts_allowed=True,
            foreign_processes_signalled=False, source_parent=parent, original_parent_gpu_locks=parent_locks,
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
