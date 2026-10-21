"""Fresh TP1 models on GPU2/5 with original GPU6/7 lifecycle anchors.

The unchanged source owner retains every primary lock. Its busy_slots sees
an original GPU6/7 UUID lease until the new controller reaps its own children.
No model CUDA context is placed on either lifecycle anchor GPU.
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
import source_supplementary_worker as original

slot_worker = original.slot_worker
consumer, shared, owner_module = original.consumer, original.shared, original.owner_module
rollover_source_slots = original.rollover_source_slots
SPECIFICATIONS = {
    'gpu6-tp1': dict(indices=[2], tp=1, seqs=64, tokens=8192, blocks=2048, reserve=1.5),
    'gpu7-tp1': dict(indices=[5], tp=1, seqs=128, tokens=16384, blocks=4096, reserve=1.5),
}


def specification(pool, name):
    path = Path(os.environ['Q7B_SUPPLEMENTARY_MANIFEST'])
    shared.require(pool.sha(path) == os.environ['Q7B_SUPPLEMENTARY_MANIFEST_SHA256'], 'Supplementary manifest SHA differs')
    value = pool.read(path)
    shared.require(value['role'] == 'slot' and value['slot'] == name and name in SPECIFICATIONS
        and value['supplementary_slots'] == {name: SPECIFICATIONS[name]}, 'Undeclared supplementary TP1 specification')
    for filename, expected in value['files'].items():
        raw = (path.parent / filename).read_bytes()
        shared.require(len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256'],
            'Supplementary immutable source differs: ' + filename)
    return value, SPECIFICATIONS[name]


def admit(pool, args):
    shared.require(socket.gethostname().split('.')[0] == 'h100-5', 'Supplementary source slots support only h100-5')
    path, value = slot_worker.ready(pool, args)
    borrowed = slot_worker.borrowed_relay(pool, args)
    owner = value['owner']
    actual = owner_module.process(os.getpid())
    shared.require(actual['netns_inode'] == owner['netns_inode'] and actual['mntns_inode'] == owner['mntns_inode'],
        'Supplementary model must use the exact independent owner namespaces')
    allocation = pool.read(value['allocation_path'])
    shared.require(pool.sha(value['allocation_path']) == value['allocation_sha256']
        and allocation['allocation_role'] == 'RETRIEVAL_ONLY_PRIMARY_LEASE_OWNER'
        and allocation['owner'] == owner and allocation['run_id'] == pool.RUN_ID, 'Owner allocation binding differs')
    name = os.environ['Q7B_SLOT_NAME']
    shared.require(name in SPECIFICATIONS and name in value['slots'] and value['slots'] == allocation['slots'],
        'Original anchor slot is absent or owner ready/allocation slots differ')
    anchor = value['slots'][name]
    standard = rollover_source_slots.specifications(owner_module.SLOTS)[name]
    shared.require(all(anchor[key] == expected for key, expected in standard.items())
        and len(anchor['indices']) == len(anchor['cuda_uuids']) == 1, 'Original anchor standard specification differs')
    manifest, slot = specification(pool, name)
    context_path = Path(os.environ['Q7B_SUPPLEMENTARY_CONTEXT'])
    shared.require(pool.sha(context_path) == os.environ['Q7B_SUPPLEMENTARY_CONTEXT_SHA256'], 'Supplementary submit context SHA differs')
    context = pool.read(context_path)
    previous = context['anchor_terminal']
    original = pool.read(previous['intent_path'])
    shared.require(pool.sha(previous['intent_path']) == previous['intent_sha256']
        and original['unit'] == previous['unit'] == manifest['anchor_unit'] and original['manifest']['role'] == 'slot'
        and original['manifest']['slot'] == name and original['manifest']['instance'] == 'r1'
        and original['context']['owner_ready_sha256'] == pool.sha(path)
        and previous['observed']['MainPID'] == '0' and previous['observed']['ActiveState'] in ('inactive', 'failed'),
        'Original exact r1 anchor terminal binding differs')
    model = shared.indices(args.model_gpus, args.tp)
    shared.require(model == slot['indices'] and args.tp == slot['tp']
        and args.model_max_seqs == slot['seqs'] and args.model_batch_tokens == slot['tokens']
        and args.model_kv_block_cap == slot['blocks'] and args.model_reserve_gib == slot['reserve'], 'Supplementary TP1 topology/capacity differs')
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
        and evidence['unit'] not in (owner['unit'], previous['unit']) and group.endswith('/' + evidence['unit'])
        and any(line.split(':', 2)[-1] == group for line in actual['cgroup'].splitlines())
        and evidence['source_bundle_sha256'] == manifest['source_bundle_sha256'], 'Supplementary controller own cgroup/bundle differs')
    deadline = value['consumer_deadline_epoch']
    if 'unit_hard_end_epoch' in evidence:deadline = min(deadline, evidence['unit_hard_end_epoch'] - 45)
    inventory = {row['index']: row for row in pool.nvml().values()}
    uuids = [inventory[index]['uuid'] for index in model]
    shared.require(anchor['cuda_uuids'] == [inventory[anchor['indices'][0]]['uuid']],
        'Original anchor physical UUID mapping differs')
    base = Path(args.base)
    proofs = {uuid: slot_worker.held_lock(base / 'pool/gpu-locks' / (uuid + '.lock'), owner,
        allocation['gpu_lock_proofs'][uuid]) for uuid in dict.fromkeys(anchor['cuda_uuids'] + uuids)}
    locks, paths = [], {}
    try:
        # Anchor first: the original unmodified busy_slots sees this model.
        order = anchor['cuda_uuids'] + sorted(set(uuids) - set(anchor['cuda_uuids']))
        for uuid in order:
            lease = owner_module.slot_directory(base, pool.RUN_ID, owner, name) / (uuid + '.lock')
            fd = pool.try_lock(lease)
            shared.require(fd is not None, 'Original anchor/supplementary UUID slot lease is already occupied')
            locks.append(fd);paths[uuid] = str(lease)
        result = subprocess.check_output([sys.executable, str(Path(consumer.__file__).resolve()), '--cuda-probe',
            '--expected-uuids', ','.join(uuids)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, model)),
            CUDA_DEVICE_ORDER='PCI_BUS_ID'), text=True, timeout=45)
        shared.require(json.loads(result)['actual_cuda_runtime_uuids'] == uuids and borrowed.poll() is None,
            'Actual supplementary CUDA UUID/source identity differs')
        return dict(job_id='source-slot-%s-%d' % (name, os.getpid()), run_id=pool.RUN_ID, node='h100-5',
            controller_pid=os.getpid(), native_unit=evidence['unit'], native_cgroup=group,
            model_indices=model, model_cuda_uuids=uuids, model_cuda_visible_devices=','.join(map(str, model)),
            tensor_parallel_size=args.tp, model_engine_count=1, actual_cuda_runtime_uuids=uuids,
            original_cuda_visible_devices=evidence.get('original_cuda_visible_devices'),
            initial_gpu_memory=[inventory[index] for index in model], source_owner=owner,
            source_owner_allocation_sha256=value['allocation_sha256'], source_owner_ready=str(path),
            source_owner_ready_sha256=pool.sha(path), source_model_deadline_epoch=deadline,
            finite_slot=name, slot_lease_paths=paths, supplementary_specification=slot,
            supplementary_manifest_sha256=pool.sha(Path(os.environ['Q7B_SUPPLEMENTARY_MANIFEST'])),
            anchor_terminal=previous, original_primary_lock_proofs=proofs, primary_gpu_locks_modified=False,
            local_relay_process=borrowed.proof, shared_retriever_owned_by_consumer=False,
            source_parent_owned=False, foreign_processes_signalled=False,
            allocation_evidence=str(args.allocation_evidence), allocation_evidence_sha256=pool.sha(args.allocation_evidence),
            native_resource_limits=evidence['resource_limits'], epoch=time.time()), locks
    except BaseException:
        for fd in locks:os.close(fd)
        raise


def main():
    import queued_health
    queued_health.install()
    saved = consumer.admit, consumer.borrowed_relay
    try:
        consumer.admit = admit
        consumer.borrowed_relay = slot_worker.borrowed_relay
        consumer.main()
    finally:
        consumer.admit, consumer.borrowed_relay = saved


if __name__ == '__main__': main()
