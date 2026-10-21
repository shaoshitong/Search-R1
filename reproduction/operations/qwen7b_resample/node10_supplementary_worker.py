"""One GPU4 TP1 borrowing a terminal GPU1 anchor of the live node10 owner.

The anchor is a lifecycle lease only. No CUDA context or model is placed on
GPU1. The unchanged owner retains every primary UUID lock and sees the anchor
as busy until this controller has cleaned up its own sampler children.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import relay_slot_consumer_worker as slot_worker
import queued_relay_slots
import rollover_paths

consumer, relay_owner = slot_worker.consumer, slot_worker.relay_owner
shared = consumer.shared
SLOT, ANCHOR = 'gpu4-tp1', 'gpu1-tp1'
SPECIFICATION = dict(indices=[4], tp=1, seqs=64, tokens=8192, blocks=4096, reserve=1.0)
ANCHOR_UNIT = 'sst-q7b-g5-relaygen-queued-owner-v2-slot-gpu1-tp1-r1-70188b8e50fd.service'
TERMINAL_STATUS = 'OWN_OWNER_V2_SPARE_ADMISSION_CAPACITY_FAILURE_TERMINAL_FULL_SHA_RELEASE_PASS'


def resolved(base, recorded):
    base = Path(base)
    return rollover_paths.readable(recorded, base, Path('/mnt/data/shaoshitong') / base.name)


def anchor_state(unit):
    raw = subprocess.check_output(['systemctl', 'show', unit, '-p', 'Id', '-p', 'LoadState', '-p', 'MainPID',
                                  '-p', 'ActiveState', '-p', 'ControlGroup'], text=True, timeout=15)
    props = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    shared.require(props.get('Id') == unit and props.get('LoadState') == 'loaded'
        and props.get('MainPID') == '0' and props.get('ActiveState') in ('inactive', 'failed'),
        'The original exact GPU1 r1 anchor is not terminal')
    group = '/system.slice/' + unit
    for proc in Path('/proc').glob('[0-9]*'):
        try: groups = [line.split(':', 2)[-1] for line in (proc / 'cgroup').read_text().splitlines()]
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
        shared.require(not any(value == group or value.startswith(group + '/') for value in groups),
                       'The original anchor cgroup still contains a process')
    return props


def anchor_binding(pool, base, ready, ready_sha, intent_path, intent_sha, proof_path, proof_sha):
    """Small original metadata only; never signals or replays an old intent."""
    intent_path, proof_path = resolved(base, intent_path), resolved(base, proof_path)
    shared.require(pool.sha(intent_path) == intent_sha and pool.sha(proof_path) == proof_sha,
                   'Original anchor intent/terminal proof SHA differs')
    intent, proof = pool.read(intent_path), pool.read(proof_path)
    prior, context = intent['manifest'], intent['context']
    shared.require(intent['unit'] == prior['unit'] == ANCHOR_UNIT and prior['role'] == 'slot'
        and prior['slot'] == ANCHOR and prior['instance'] == 'r1'
        and prior['run_id'] == pool.RUN_ID and prior['generation'] == 'queued-owner-v2'
        and context['downstream_owner_ready_sha256'] == ready_sha
        and pool.sha(resolved(base, context['downstream_owner_ready'])) == ready_sha,
        'Original GPU1 r1 intent does not belong to this unchanged CPU owner')
    shared.require(proof['status'] == TERMINAL_STATUS and proof['node'] == 'h100-10'
        and proof['run_id'] == pool.RUN_ID and proof['controlled_stop'] is False,
        'Original node10 capacity-failure terminal proof differs')
    observations = proof['release_observations']
    shared.require(len(observations) == 2 and observations[1]['epoch'] - observations[0]['epoch'] >= 3
        and all(not row['remaining_known_pids'] and not row['remaining_original_births']
                and not row['own_gpu_contexts'] and row['own_cgroups_empty']
                for row in observations), 'Original two-observation release proof is incomplete')
    rows = [row for row in proof['units'] if row['unit'] == ANCHOR_UNIT and row['key'] == 'node10:' + ANCHOR]
    shared.require(len(rows) == 1, 'Terminal proof has no unique exact GPU1 anchor')
    row = rows[0]
    job = resolved(base, row['job_directory'])
    allocation_path, failure_path = resolved(base, row['allocation_path']), resolved(base, row['failure_path'])
    shared.require(allocation_path.parent == failure_path.parent == job,
                   'Original anchor allocation/failure job paths differ')
    shared.require(pool.sha(allocation_path) == row['allocation_sha256']
        and pool.sha(failure_path) == row['failure_sha256'], 'Original anchor allocation/failure SHA changed')
    allocation = pool.read(allocation_path)
    anchor = ready['slots'][ANCHOR]
    shared.require(allocation['native_unit'] == ANCHOR_UNIT and allocation['controller_pid'] == row['controller_pid']
        and allocation['model_indices'] == [1] and allocation['model_cuda_uuids'] == anchor['cuda_uuids'],
        'Original anchor allocation/physical UUID differs')
    observed = anchor_state(ANCHOR_UNIT)
    return dict(unit=ANCHOR_UNIT, slot=ANCHOR, controller_pid=row['controller_pid'], intent_path=str(intent_path),
        intent_sha256=intent_sha, terminal_proof=str(proof_path), terminal_proof_sha256=proof_sha,
        allocation_path=str(allocation_path), allocation_sha256=row['allocation_sha256'],
        failure_path=str(failure_path), failure_sha256=row['failure_sha256'], observed=observed)


def specification(pool):
    path = Path(os.environ['Q7B_NODE10_SUPPLEMENT_MANIFEST'])
    shared.require(pool.sha(path) == os.environ['Q7B_NODE10_SUPPLEMENT_MANIFEST_SHA256'], 'Supplement manifest SHA differs')
    value = pool.read(path)
    shared.require(value['role'] == 'slot' and value['slot'] == SLOT and value['run_id'] == pool.RUN_ID
        and value['supplementary_slots'] == {SLOT: SPECIFICATION} and value['anchor_slot'] == ANCHOR
        and value['anchor_unit'] == ANCHOR_UNIT and value['owner_registry_unchanged'] is True,
        'Undeclared node10 GPU4 supplementary model')
    for filename, expected in value['files'].items():
        raw = (path.parent / filename).read_bytes()
        shared.require(len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256'],
                       'Supplement immutable source differs: ' + filename)
    return path, value


def lease_paths(base, run_id, owner, anchor_uuid, model_uuid):
    directory = relay_owner.source.slot_directory
    return (directory(base, run_id, owner, ANCHOR) / (anchor_uuid + '.lock'),
            directory(base, run_id, owner, 'node10-supplement-' + SLOT) / (model_uuid + '.lock'))


def admit(pool, args):
    shared.require(socket.gethostname().split('.')[0] == 'h100-10', 'Supplement supports only node10')
    path, ready, allocation = slot_worker.ready(pool, args)
    borrowed = slot_worker.borrowed_relay(pool, args)
    evidence, actual = relay_owner.own_evidence(pool, args, 'h100-10')
    owner = ready['owner']
    shared.require(actual['mntns_inode'] == owner['mntns_inode'] and actual['netns_inode'] == owner['netns_inode']
        and actual['unit'] != owner['unit'], 'Supplement requires owner namespaces and an independent cgroup')
    manifest_path, manifest = specification(pool)
    shared.require(os.environ['Q7B_RELAY_SLOT_NAME'] == SLOT and evidence['unit'] == manifest['unit']
        and evidence['unit'] != ANCHOR_UNIT, 'Supplement unit/slot differs')
    anchor = ready['slots'][ANCHOR]
    standard = queued_relay_slots.TP1_SLOTS[ANCHOR]
    shared.require(all(anchor[key] == expected for key, expected in standard.items())
        and anchor['indices'] == [1] and len(anchor['cuda_uuids']) == 1,
        'Original lifecycle anchor specification differs')
    context_path = Path(os.environ['Q7B_NODE10_SUPPLEMENT_CONTEXT'])
    shared.require(pool.sha(context_path) == os.environ['Q7B_NODE10_SUPPLEMENT_CONTEXT_SHA256'], 'Supplement context SHA differs')
    context = pool.read(context_path)
    previous = context['anchor_terminal']
    current = anchor_binding(pool, args.base, ready, pool.sha(path), previous['intent_path'], previous['intent_sha256'],
        previous['terminal_proof'], previous['terminal_proof_sha256'])
    shared.require(all(current[key] == previous[key] for key in ('unit', 'slot', 'controller_pid', 'intent_sha256',
        'terminal_proof_sha256', 'allocation_sha256', 'failure_sha256')), 'Anchor binding changed after submit')
    model = shared.indices(args.model_gpus, args.tp)
    shared.require(model == [4] and args.tp == 1 and args.model_max_seqs == 64 and args.model_batch_tokens == 8192
        and args.model_kv_block_cap == 4096 and args.model_reserve_gib == 1.0 and args.batch_reserve_seconds == 1200,
        'Only fixed GPU4 TP1 initial capacity is admitted')
    deadline = min(ready['consumer_deadline_epoch'], evidence['unit_hard_end_epoch'] - 45)
    shared.require(time.time() + args.run_seconds <= ready['consumer_deadline_epoch']
        and deadline > time.time() + args.batch_reserve_seconds, 'Supplement owner lifetime is exhausted')
    inventory = {row['index']: row for row in pool.nvml().values()}
    anchor_uuid, model_uuid = anchor['cuda_uuids'][0], inventory[4]['uuid']
    shared.require(anchor_uuid == inventory[1]['uuid'] and model_uuid != anchor_uuid,
                   'Anchor GPU1 and model GPU4 UUID mapping differ')
    base = Path(args.base)
    proofs = {uuid: slot_worker.source_slot.held_lock(base / 'pool/gpu-locks' / (uuid + '.lock'), owner,
        allocation['gpu_lock_proofs'][uuid]) for uuid in (anchor_uuid, model_uuid)}
    paths = lease_paths(base, pool.RUN_ID, owner, anchor_uuid, model_uuid)
    locks = []
    try:
        # First lease is seen by the unmodified owner's busy_slots. The second
        # independently excludes another GPU4 supplement, never the live TP4.
        for lease in paths:
            fd = pool.try_lock(lease)
            shared.require(fd is not None, 'Lifecycle anchor or supplementary GPU4 model lease is occupied')
            locks.append(fd)
        output = subprocess.check_output([sys.executable, str(Path(consumer.__file__).resolve()), '--cuda-probe',
            '--expected-uuids', model_uuid], env=dict(os.environ, CUDA_VISIBLE_DEVICES='4', CUDA_DEVICE_ORDER='PCI_BUS_ID'),
            text=True, timeout=45)
        shared.require(json.loads(output)['actual_cuda_runtime_uuids'] == [model_uuid] and borrowed.poll() is None,
                       'Actual supplementary GPU4 CUDA UUID/CPU owner differs')
        birth = int(actual['birth']);boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return dict(job_id='relay-slot-%s-%d' % (SLOT, os.getpid()), run_id=pool.RUN_ID, node='h100-10',
            controller_pid=os.getpid(), controller_birth=birth, boot_id=boot_id,
            native_unit=evidence['unit'], native_cgroup=evidence['cgroup'], model_indices=[4], model_cuda_uuids=[model_uuid],
            model_cuda_visible_devices='4', tensor_parallel_size=1, model_engine_count=1, actual_cuda_runtime_uuids=[model_uuid],
            original_cuda_visible_devices=evidence.get('original_cuda_visible_devices'), initial_gpu_memory=[inventory[4]],
            source_owner=owner, source_owner_allocation_sha256=ready['allocation_sha256'],
            source_owner_ready=str(path), source_owner_ready_sha256=pool.sha(path), source_model_deadline_epoch=deadline,
            finite_slot=SLOT, slot_lease_paths={str(i): str(p) for i, p in enumerate(paths)},
            lifecycle_anchor_slot=ANCHOR, lifecycle_anchor_cuda_uuid=anchor_uuid, lifecycle_anchor_lease=str(paths[0]),
            supplementary_model_lease=str(paths[1]), supplementary_specification=SPECIFICATION,
            supplementary_manifest=str(manifest_path), supplementary_manifest_sha256=pool.sha(manifest_path),
            anchor_terminal=current, original_primary_lock_proofs=proofs, primary_gpu_locks_modified=False,
            source_retrieval_owner=ready['source_owner'],
            local_relay_process=borrowed.proof, shared_retriever_owned_by_consumer=False,
            source_parent_owned=False, foreign_processes_signalled=False, upstream_ready=str(args.upstream_ready),
            upstream_ready_sha256=pool.sha(args.upstream_ready), allocation_evidence=str(args.allocation_evidence),
            allocation_evidence_sha256=pool.sha(args.allocation_evidence), native_resource_limits=evidence['resource_limits'],
            captured_epoch=time.time()), locks
    except BaseException:
        for fd in locks: os.close(fd)
        raise


def main():
    import queued_health
    queued_health.install()
    original = consumer.admit, consumer.borrowed_relay
    try:
        consumer.admit, consumer.borrowed_relay = admit, slot_worker.borrowed_relay
        consumer.main()
    finally:
        consumer.admit, consumer.borrowed_relay = original


if __name__ == '__main__': main()
