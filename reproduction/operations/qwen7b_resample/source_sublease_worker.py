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
RETIRED_PID, RETIRED_BIRTH = 2021698, '7430669'
RETIRED_UNIT = 'sst-q7b-g5-source5-extra-model-v1-70188b8e50fd.service'
RETIRED_STATUS = 'OWN_SOURCE5_MODEL_TP4_CONTROLLED_THROUGHPUT_STOP_FULL_SHA_RELEASE_PASS'


def context(pool, args):
    path = Path(os.environ['Q7B_SOURCE_SUBLEASE_CONTEXT'])
    shared.require(path.samefile(args.reuse_local_relay_ready)
                   and pool.sha(path) == args.reuse_local_relay_sha256,
                   'Source sublease context path or SHA differs')
    value = pool.read(path)
    shared.require(value['schema_version'] in (1, 2, 3) and value['mode'] == 'SOURCE5_SAME_RUN_GPU_SUBLEASE'
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
    if value['schema_version'] == 3:
        shared.require('peer_sublease' not in value and value['model_indices'] in ([3], [4], [6], [7])
                       and value['base'] == str(args.base)
                       and bool(value['retired_terminal'])
                       and value['model_deadline_epoch'] <= value['source_sampling_deadline_epoch'] - 3600,
                       'Schema3 requires an independent retired root-child lease and conservative source cutoff')
        limits = dict(seqs=8, tokens=4096, blocks=512, reserve=1.0) if value['model_indices'] == [3] else dict(seqs=32, tokens=8192, blocks=4096, reserve=1.0)
        shared.require(value['model_limits'] == limits
                       and (args.model_max_seqs, args.model_batch_tokens, args.model_kv_block_cap, args.model_reserve_gib)
                           == (limits['seqs'], limits['tokens'], limits['blocks'], limits['reserve']), 'Schema3 initial TP1 capacity limits differ')
    return path, value


def source_path(base, recorded):
    """Access physical-NAS proof references through the original private mount."""
    base, recorded = Path(os.path.normpath(base)), Path(os.path.normpath(recorded))
    physical = Path('/mnt/data/shaoshitong') / base.name
    if recorded.is_relative_to(physical):
        recorded = base / recorded.relative_to(physical)
    shared.require(recorded.is_relative_to(base), 'Retired proof references another data root')
    return recorded


def validate_retired_terminal(pool, value, parent, model, uuids):
    shared.require(len(model) == len(uuids) == 1, 'Retired replacement must be one TP1 physical GPU')
    binding = value['retired_terminal']
    base = Path(value['base'])
    path = source_path(base, binding['path'])
    shared.require(pool.sha(path) == binding['sha256'], 'Retired terminal proof SHA differs')
    proof = pool.read(path)
    retired = proof['retired']
    shared.require(proof['status'] == RETIRED_STATUS and proof['run_id'] == pool.RUN_ID
                   and retired['pid'] == RETIRED_PID and str(retired['birth']) == RETIRED_BIRTH
                   and retired['unit'] == RETIRED_UNIT, 'Retired terminal identity differs')
    shared.require(all(proof['source_parent'][key] == parent[key] for key in ('pid', 'birth', 'unit'))
                   and proof['source_parent_and_retriever_preserved'] is True
                   and proof['stopped_after_at_least_own_batches'] >= 1
                   and proof['original_partial_outputs_retained'] is True
                   and proof['global_goal_complete'] is False, 'Retired terminal source or data retention proof differs')
    observations = proof['release_observations']
    shared.require(len(observations) >= 2 and observations[-1]['epoch'] - observations[0]['epoch'] >= 3
                   and all(row['unit_main_pid'] == 0 and row['original_births_gone'] is True
                           and row['own_cgroup_gone'] is True and row['own_gpu_contexts'] == []
                           and row['foreign_signalled'] is False for row in observations),
                   'Retired terminal lacks two actual owned release observations')
    known = proof['original_owned_births']
    shared.require(known.get(str(RETIRED_PID)) == RETIRED_BIRTH
                   and not {str(PARENT_PID), str(RETRIEVER_PID), str(PEER_PID)} & set(known),
                   'Retired owned birth set includes a retained source or differs')
    allocation_path = source_path(base, retired['allocation_path'])
    shared.require(pool.sha(allocation_path) == retired['allocation_sha256']
                   and any(source_path(base, row['path']) == allocation_path
                           and row['sha256'] == retired['allocation_sha256'] for row in proof['original_files']),
                   'Retired allocation is not bound by the retained original-file proof')
    allocation = pool.read(allocation_path)
    context_path = source_path(base, allocation['source_context'])
    shared.require(pool.sha(context_path) == allocation['source_context_sha256'], 'Retired source context SHA differs')
    retired_context = pool.read(context_path)
    shared.require(allocation['controller_pid'] == RETIRED_PID and allocation['native_unit'] == RETIRED_UNIT
                   and allocation['native_cgroup'] == '/system.slice/' + RETIRED_UNIT
                   and allocation['node'] == 'h100-5' and retired_context['run_id'] == pool.RUN_ID
                   and ('run_id' not in allocation or allocation['run_id'] == pool.RUN_ID)
                   and allocation['tensor_parallel_size'] == 4 and allocation['model_indices'] == [3, 4, 6, 7]
                   and allocation['source_parent'] == retired_context['parent'] == parent
                   and retired_context['model_indices'] == allocation['model_indices']
                   and retired_context['model_cuda_uuids'] == allocation['model_cuda_uuids']
                   and model in ([4], [6], [7]),
                   'Retired allocation or original source binding differs')
    mapping = dict(zip(allocation['model_indices'], allocation['model_cuda_uuids']))
    directory = base / 'pool/gpu-subleases' / pool.RUN_ID / ('%d-%s' % (PARENT_PID, parent['birth']))
    shared.require(all(mapping.get(index) == uuid
                       and source_path(base, allocation['gpu_sublease_paths'][uuid]) == directory / (uuid + '.lock')
                       for index, uuid in zip(model, uuids)), 'Retired allocation does not bind this physical root-child GPU lease')
    return dict(path=str(path), sha256=binding['sha256'], retired=retired, original_owned_births=known,
                source_parent_preserved=True, partial_outputs_retained=True)


def retired_release(binding):
    """Observe only the retired unit/owned births; never signal a process."""
    raw = subprocess.check_output(['systemctl', 'show', RETIRED_UNIT, '-p', 'MainPID', '-p', 'ActiveState',
                                  '-p', 'ControlGroup'], text=True, timeout=15)
    props = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    group = '/system.slice/' + RETIRED_UNIT
    shared.require(props['MainPID'] == '0' and props['ActiveState'] in ('inactive', 'failed')
                   and props['ControlGroup'] in ('', group), 'Retired unit is not actually terminal')
    remaining, original_or_absent = {}, set()
    for pid, expected in binding['original_owned_births'].items():
        try:
            fields = Path('/proc/%s/stat' % pid).read_text().rsplit(')', 1)[1].split()
            if fields[19] == str(expected):
                remaining[pid] = expected
                original_or_absent.add(int(pid))
        except (FileNotFoundError, ProcessLookupError):
            original_or_absent.add(int(pid))
    directory = Path('/sys/fs/cgroup') / group.lstrip('/')
    members = sorted({int(x) for p in directory.rglob('cgroup.procs') for x in p.read_text().split()}) if directory.exists() else []
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    own_gpu = sorted({int(x.strip()) for x in apps.splitlines() if x.strip().isdigit()} & original_or_absent)
    shared.require(not remaining and not members and not own_gpu, 'Retired owned processes, cgroup or GPU contexts remain')
    return dict(epoch=time.time(), unit_main_pid=0, original_births_gone=True, own_cgroup_gone=True,
                own_gpu_contexts=own_gpu, foreign_signalled=False)


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
    retirement, release_observations = None, []
    if value['schema_version'] == 2:
        peer = peer_admission(pool, value, directory, model, uuids)
        directory = directory / ('co-share-with-%d-%s' % (peer['pid'], peer['birth']))
    elif value['schema_version'] == 3:
        retirement = validate_retired_terminal(pool, value, parent, model, uuids)
        release_observations.append(retired_release(retirement))
        shared.require(inventory[model[0]]['free_bytes'] > 1024 ** 3,
                       'Observed free VRAM does not leave the fixed one-GiB model reserve')
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
        if retirement is not None:
            release_observations.append(retired_release(retirement))
            shared.require(value['model_deadline_epoch'] > time.time() + args.batch_reserve_seconds + 60,
                           'Conservative source lifetime is too short for a new batch')
        shared.require(borrowed.poll() is None, 'Original source identity changed during sublease admission')
        upstream = pool.read(args.upstream_ready)
        shared.require(upstream['role'] == 'upstream' and upstream['source_pid'] == RETRIEVER_PID
                       and upstream['listen_url'].rstrip('/') == 'http://127.0.0.1:28831',
                       'Direct original retriever endpoint differs')
        binding = dict(job_id='source5-model-' + str(os.getpid()), node='h100-5', controller_pid=os.getpid(), run_id=pool.RUN_ID,
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
            native_resource_limits=evidence['resource_limits'], captured_epoch=time.time())
        if retirement is not None:
            binding.update(source_model_deadline_epoch=value['model_deadline_epoch'],
                source_deadline_reserve_seconds=3600, source_parent_may_return_early=True,
                retired_terminal_startup_proof=retirement, retired_current_release_observations=release_observations,
                independent_root_child_lease=True, peer_sublease_startup_proof=None)
        return binding, locks
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
