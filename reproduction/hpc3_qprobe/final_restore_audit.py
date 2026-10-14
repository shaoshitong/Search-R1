"""Independent native4 final200 restore audit. Never enters the training loop.

Run through final_restore_audit.sh in a new, owner-controlled native allocation.
The ordinary paired-training binder deliberately continues to reject step200.
CPU tests exercise guards only; only the native worker receipts can accept CUDA.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid

ROOT = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
OWNER = 14090
ROLE = 'final200_native4_restore_only_audit'
MODULE = 'reproduction.hpc3_qprobe.final_restore_audit'
ENTRY = 'reproduction/hpc3_qprobe/final_restore_audit.sh'
FORBIDDEN_OUTPUTS = ('training-launch-claim.json', 'actual-updates.jsonl',
                     'actual-attempts.jsonl', 'five-updates-acceptance.json')


def need(condition, message):
    if not condition:
        raise ValueError('FINAL_RESTORE_AUDIT: ' + message)


def digest(path):
    from search_g1.full_state_restore import digest as full_digest
    return full_digest(path)


def publish(path, value):
    from search_g1.full_state_restore import publish as immutable_publish
    immutable_publish(path, value)


def tree_sha(value):
    """Typed, deterministic digest; no tensor values are printed or saved here."""
    import numpy as np
    import torch
    h = hashlib.sha256()

    def add(v):
        def tag(s):
            raw = s.encode(); h.update(str(len(raw)).encode() + b':' + raw)
        if isinstance(v, torch.Tensor):
            x = v.detach().cpu().contiguous()
            tag('tensor:' + str(x.dtype) + ':' + str(tuple(x.shape)))
            raw = memoryview(x.reshape(-1).view(torch.uint8).numpy()).cast('B')
            for offset in range(0, len(raw), 8 * 1024 * 1024):
                h.update(raw[offset:offset + 8 * 1024 * 1024])
        elif isinstance(v, np.ndarray):
            tag('numpy:' + str(v.dtype) + ':' + str(v.shape))
            if v.dtype.hasobject:
                for item in v.flat: add(item)
            else: h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, np.generic):
            tag('numpy-scalar:' + str(v.dtype)); add(v.item())
        elif isinstance(v, dict):
            tag('dict:' + str(len(v)))
            for key in sorted(v, key=lambda k: (type(k).__name__, repr(k))):
                add(key); add(v[key])
        elif isinstance(v, (list, tuple)):
            tag(type(v).__name__ + ':' + str(len(v)))
            for item in v: add(item)
        elif all(hasattr(v, key) for key in ('batch', 'non_tensor_batch', 'meta_info')):
            tag('DataProto:' + type(v).__name__)
            add({key: v.batch[key] for key in v.batch.keys()})
            add(v.non_tensor_batch); add(v.meta_info)
        elif isinstance(v, bytes):
            tag('bytes:' + str(len(v))); h.update(v)
        elif v is None or type(v) in (str, bool, int, float):
            tag(type(v).__name__ + ':' + repr(v))
        else:
            raise ValueError('Unsupported actual state type: ' + type(v).__name__)
    add(value)
    return h.hexdigest()


def final_pair_gate(receipt, parent):
    from search_g1.qprobe_checkpoint import PAIR_STATUS, PAIR_KIND
    need(receipt.get('status') == PAIR_STATUS and receipt.get('kind') == PAIR_KIND,
         'not an actual completed question-probe pair')
    need(type(receipt.get('step')) is int and receipt['step'] == 200,
         'only completed200 is accepted')
    need(type(receipt.get('world_size')) is int and receipt['world_size'] == 4 and
         parent.get('training_world_size') == 4 and parent.get('actor_micro_batch_per_rank') == 4,
         'only the original four-rank shard layout is accepted')
    need(receipt['run_id'] == parent['run_id'] and
         receipt['actor_snapshot_id'] == parent['run_id'] + ':200', 'source actor identity differs')
    need(receipt['attempt_progress']['completed_outer'] == 200 and
         receipt['driver_state']['completed_outer'] == 200, 'source counters differ')
    need(parent.get('execution_site') == 'hpc3' and
         parent.get('execution_backend', 'slurm') == 'slurm', 'source is not native HPC3')
    need(parent.get('question_probe') is not None, 'source has no question probe')


def full_records(receipt):
    """Use the transport publisher's complete recursive record schema.

    Recovery adds foreign-checkpoint and original-producer provenance records.
    They are independent dependencies, not optional reporting metadata. Keep
    the exact archive file set and reject conflicting repeated identities.
    """
    from reproduction.hpc3_qprobe.archive_completed_pair import file_records
    records={}
    for row in file_records(receipt):
        path=row['path']
        need(isinstance(path,str) and path,'malformed dependency file path')
        previous=records.get(path)
        if previous is not None:
            need(all(previous[key]==row[key] for key in ('inode','bytes','mtime_ns','full_read_sha256')),
                 'duplicate dependency path has conflicting identity/SHA')
        else:records[path]=row
    return [records[path] for path in sorted(records)]


def verify_record_bytes(records):
    """Fresh full reads, including every large model/Adam file, never metadata-only."""
    checked = {}
    for row in records:
        path = Path(row['path'])
        need(path.is_absolute() and path.resolve() == path and not path.is_symlink() and path.is_file(),
             'unsafe source file')
        before = path.stat()
        need((before.st_ino, before.st_size, before.st_mtime_ns) ==
             (row['inode'], row['bytes'], row['mtime_ns']), 'source file identity differs')
        previous = checked.get(str(path))
        if previous is not None:
            need(previous == row['full_read_sha256'], 'duplicate source path has conflicting hashes')
        else:
            need(digest(path) == row['full_read_sha256'], 'fresh full source SHA differs')
            checked[str(path)] = row['full_read_sha256']
        after = path.stat()
        need((before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_ino, after.st_size, after.st_mtime_ns), 'source changed during read')
    return checked


def source_proof(receipt_path):
    from search_g1.qprobe_checkpoint import read_source
    from reproduction.common8.formal_contract import assert_formal_run
    path = Path(receipt_path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink() and
         path.stat().st_uid == OWNER, 'source receipt must be canonical and owner-authored')
    p, receipt, checkpoint, origin, parent = read_source(path)
    final_pair_gate(receipt, parent); assert_formal_run(parent)
    need(origin != checkpoint and origin.stat().st_uid == OWNER and
         Path(receipt['source_run_manifest']['path']).stat().st_uid == OWNER,
         'source owner/base differs')
    hashes = verify_record_bytes(full_records(receipt))
    expected_repo = ROOT / 'source' / parent['source_commit']
    # A relocated target retains the original producer manifest bytes. It need
    # not contain a repo link or pretend to be the producer's original base.
    need(expected_repo.is_dir() and expected_repo.resolve() == expected_repo, 'producer source root differs')
    for name, sha in parent['source_sha256'].items():
        value = expected_repo / name
        need(value.resolve() == value and not value.is_symlink() and digest(value) == sha,
             'producer source bytes differ: ' + name)
    return p, receipt, checkpoint, origin, parent, dict(
        source_receipt=str(p), source_receipt_sha256=digest(p),
        source_manifest_sha256=receipt['source_run_manifest']['full_read_sha256'],
        producer_commit=parent['source_commit'], producer_source_sha256=parent['source_sha256'],
        full_read_files=hashes, full_read_file_count=len(hashes), checked_unix=time.time())


def verify_transport(proof_path, receipt_path, *, expected_step=200):
    """Independent three-way full-byte acceptance; no receipt/report SHA cycle."""
    path = Path(proof_path); target_path = Path(receipt_path)
    need(type(expected_step) is int and expected_step in (5, 50, 100, 150, 200),
         'only fixed checkpoint transport milestones are accepted')
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink() and
         path.stat().st_uid == OWNER, 'transport proof must be canonical and owner-authored')
    proof = json.loads(path.read_text())
    need(proof.get('schema_version', proof.get('schema')) == 1 and
         proof.get('status') == 'QPROBE_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS' and
         type(proof.get('step')) is int and proof['step'] == expected_step and proof.get('world_size') == 4,
         'complete expected-step native4 transport proof required')
    receipts = {}
    for role in ('source', 'archive', 'target'):
        record = proof[role + '_receipt']; value = Path(record['path'])
        need(value.is_absolute() and value.resolve() == value and not value.is_symlink() and
             value.stat().st_uid == OWNER and digest(value) == record['sha256'], 'transport receipt SHA differs')
        receipts[role] = json.loads(value.read_text())
    need(target_path == Path(proof['target_receipt']['path']), 'audit must load the verified target receipt')
    need(len({proof[role + '_receipt']['path'] for role in receipts}) == 3, 'transport receipts are not distinct')
    for role, receipt in receipts.items():
        need(receipt['run_id'] == proof['run_id'] and receipt['step'] == expected_step and receipt['world_size'] == 4 and
             receipt['source_run_manifest']['full_read_sha256'] == proof['source_manifest_sha256'],
             'transport run/producer binding differs')
    roots = {role: Path(receipts[role]['source_base']) for role in receipts}
    need(len(set(roots.values())) == 3, 'three distinct transport roots required')
    for role, root in roots.items():
        need(root.resolve() == root and root.is_dir(), 'unsafe transport root')
        for other in roots.values():
            need(root == other or not root.is_relative_to(other), 'transport roots overlap')
    need(not path.is_relative_to(roots['archive']) and not path.is_relative_to(roots['target']),
         'transport proof must be outside copied file sets')
    copied = {role: {} for role in receipts}
    for row in proof['files']:
        need(type(row['bytes']) is int and row['bytes'] >= 0, 'invalid transported byte count')
        for role in receipts:
            value = Path(row[role + '_path'])
            # Source may include an immutable encoder outside the old run base;
            # archive/target must be self-contained copies inside their own roots.
            need(value.resolve() == value and value.is_file() and not value.is_symlink() and
                 value.stat().st_size == row['bytes'], 'transport file path/size differs')
            if role != 'source':
                need(value.is_relative_to(roots[role]) and value.stat().st_uid == OWNER, 'transport copy is not owned/self-contained')
            need(str(value) not in copied[role], 'duplicate transport file mapping')
            need(digest(value) == row['sha256'], 'fresh source/archive/target full SHA differs')
            copied[role][str(value)] = row['sha256']
    # Every source record and every relocated target record must be covered;
    # an omitted Adam/RNG/reward-encoder dependency can never pass this audit.
    for role, receipt in receipts.items():
        expected = {row['path']: row['full_read_sha256'] for row in full_records(receipt)}
        need(copied[role] == expected, 'transport payload file set differs from paired receipt')
    for role in ('archive', 'target'):
        expected = set(copied[role]) | {proof[role + '_receipt']['path']}
        need(set(proof[role + '_file_set']) == expected and len(proof[role + '_file_set']) == len(expected),
             'declared transport file set is incomplete/duplicated')
        actual = {str(p) for p in roots[role].rglob('*') if p.is_file()}
        need(actual == expected, 'actual transport root has missing/extra files')
    return dict(path=str(path), sha256=digest(path), target_receipt=proof['target_receipt'],
                step=expected_step,
                source_receipt=proof['source_receipt'], archive_receipt=proof['archive_receipt'],
                source_manifest_sha256=proof['source_manifest_sha256'],
                full_read_file_count=len(copied['target']), full_read_bytes=sum(row['bytes'] for row in proof['files']),
                verified_by_uid=os.getuid(), checked_unix=time.time())


def install_audit_reward_probe(base, receipt):
    """Bind only verified target reward files; original manifests retain their bytes."""
    from reproduction.hpc3_qprobe.control import link
    base = Path(base); live_dir = base / 'artifacts/live'; live_dir.mkdir(parents=True, exist_ok=True)
    need(not (live_dir / 'manifest.json').exists(), 'audit reward binding already exists')
    live = copy.deepcopy(json.loads(Path(receipt['reward_probe_manifest']['path']).read_text()))
    target_base = Path(receipt['source_base'])
    by_name = {row['relative_path']: row for row in receipt['reward_files']}
    for key, name in (('commit', 'commit_probe.pkl'), ('boundary', 'boundary_probe.pkl')):
        row = by_name[key]; path = Path(row['path'])
        need(path.is_relative_to(target_base) and digest(path) == row['full_read_sha256'] == live['head_sha256'][name],
             'target reward head binding differs')
        live[key] = str(path)
    encoder = base / 'artifacts/target-reward-encoder'; encoder.mkdir(parents=True, exist_ok=False)
    # Encoder records can refer to the target actor itself, so identify them by
    # their verified target file basename and original manifest digest as well.
    for name, sha in live['encoder_sha256'].items():
        matches = [row for row in receipt['reward_files'] if
                   Path(row['path']).name == name and row['full_read_sha256'] == sha and row['relative_path'] not in ('commit', 'boundary')]
        need(len(matches) == 1, 'target reward encoder file binding ambiguous/missing')
        path = Path(matches[0]['path'])
        need(path.is_relative_to(target_base) and digest(path) == sha, 'target reward encoder SHA differs')
        link(encoder / name, path)
    live['encoder'] = str(encoder)
    need(live['run_id'] == receipt['run_id'] and live['version'] == receipt['reward_probe_version'] and
         live['round'] == receipt['reward_probe_round'], 'target reward version differs')
    publish(live_dir / 'manifest.json', live)
    return live


def native_owner_gate(job, base, source, job_id, uid):
    need(uid == OWNER and job['JobId'] == job_id and job['UserId'] == 'jxu957(14090)' and
         job['Account'] == 'jxu957' and job['QOS'] == 'formal-user', 'actual native owner differs')
    need(job['JobState'] == 'RUNNING' and job['Partition'] == 'emergency_acd' and
         job['NumNodes'] == '1' and job['NodeList'] != 'ACD1-67' and
         job['NumCPUs'] == '48' and job['MinMemoryNode'] == '512G', 'actual allocation budget differs')
    need(job['TresPerNode'] in ('gres:gpu:4', 'gres:gpu:h100:4', 'gres/gpu:4'), 'four reserved GPUs required')
    need(job['Command'] == str(source / ENTRY) and job['WorkDir'] == str(base) and
         job['JobName'] == 'qprobe-final200audit', 'native audit entry/workdir/name differs')


def verify_runtime_source(commit):
    from reproduction.common8.configuration import REPO
    from reproduction.common8.source_files import runtime_files
    need(re.fullmatch('[0-9a-f]{40}', commit) is not None, 'invalid audit source commit')
    repo = REPO.resolve()
    need(repo == ROOT / 'source' / commit, 'audit must load its published immutable source root')
    publication = ROOT / 'preparation' / ('source-' + commit + '-verification.json')
    need(publication.stat().st_uid == OWNER, 'source publication is not owner-authored')
    proof = json.loads(publication.read_text())
    need(proof['status'] == 'PUBLISHED_SOURCE_FULL_SHA_PASS' and proof['commit'] == commit,
         'audit source publication is incomplete')
    for name, sha in proof['files'].items():
        path = repo / name
        need(path.resolve() == path and path.is_relative_to(repo) and not path.is_symlink() and
             digest(path) == sha, 'published audit source bytes differ')
    source = {str(path.relative_to(repo)): digest(path) for path in runtime_files(repo)}
    need(all(proof['files'].get(name) == sha for name, sha in source.items()),
         'runtime file omitted from source publication')
    return repo, source, dict(commit=commit, publication_path=str(publication),
                             publication_sha256=digest(publication), files=source)


def bind_audit(parent, receipt, source_path, base, commit, runtime_sources, audit_id):
    """A distinct final-only binder; never calls or relaxes bind_restore(step<200)."""
    final_pair_gate(receipt, parent)
    # Loading/training-state code must match the producer exactly. New audit-only
    # files and unrelated HPC control fixes are outside these protected prefixes.
    for name, sha in parent['source_sha256'].items():
        if name.startswith(('verl/', 'search_g1/', 'search_r1/', 'reproduction/common8/')) and \
                not Path(name).name.startswith('test_'):
            need(runtime_sources.get(name) == sha, 'restore/scientific implementation changed: ' + name)
    run = copy.deepcopy(parent)
    for key in ('slurm_gpu_sharing_policy', 'slurm_shared_authorization_path',
                'slurm_shared_authorization_sha256', 'shared_gpu_cohosts', 'shared_gpu_policy',
                'shared_physical_gpu_indices', 'shared_physical_gpu_uuids', 'ray_tmp_dir',
                'ray_tmp_dir_identity', 'ray_local_resources', 'execution_id'):
        run.pop(key, None)
    progress = receipt['attempt_progress']
    run.update(source_commit=commit, source_sha256=runtime_sources, completed_updates=0,
               native_ray_execution_id=uuid.uuid4().hex, checkpoint_dir=str(base / 'checkpoints/search-g1-common8'),
               train_log=str(base / 'logs/forbidden-training.log'), initial_probe_version=receipt['reward_probe_version'],
               pretraining_baseline_policy='reuse_historical', mixed_first_selection=False,
               final_restore_audit=dict(role=ROLE, audit_id=audit_id, fit_permitted=False,
                                       source_completed_outer=200, target_outer=200),
               full_state_restore=dict(kind='question_probe_pair', source_receipt=str(source_path),
                   source_receipt_sha256=digest(source_path), source_checkpoint=receipt['checkpoint'],
                   source_base=receipt['source_base'],
                   source_run_manifest_sha256=receipt['source_run_manifest']['full_read_sha256'],
                   completed_outer=200, actual_optimizer_writes_per_rank=progress['cumulative_writes'],
                   attempt_index=progress['attempt_index'], aborted_attempts=progress['aborted_attempts'],
                   candidate_source_cursor=receipt['driver_state']['source_consumed'],
                   reward_probe_version=receipt['reward_probe_version'], reward_probe_round=receipt['reward_probe_round'],
                   question_probe_state_path=receipt['question_probe_state_path'], feature_spec_sha=receipt['feature_spec_sha']))
    run['question_probe']['output_dir'] = str(base / 'artifacts/question-probe')
    run['question_probe']['initial_head_checkpoint'] = None
    run['question_probe']['snapshot_path'] = None
    run.pop('training_physical_indices', None)
    return run


def prepare_audit(base, commit, receipt_path, transport_path):
    from reproduction.common8.topology import validate_base
    from reproduction.common8.native_allocation import fields, admit
    from reproduction.common8.native_ray import prepare as prepare_ray
    from reproduction.hpc3_qprobe.control import prepare, link
    need(os.getuid() == OWNER, 'actual owner required')
    base = Path(base)
    need(base.is_absolute() and base.resolve() == base and
         base.name.startswith('qprobe-teacher-warmup200-final200-audit-'), 'new dedicated audit namespace required')
    need(not (base / 'logs/run-manifest.json').exists() and
         not (base / 'preparation/final-audit-launch-claim.json').exists(), 'audit already attempted; never replay')
    repo, runtime_sources, publication = verify_runtime_source(commit)
    transport = verify_transport(transport_path, receipt_path)
    p, receipt, _, origin, parent, source = source_proof(receipt_path)
    need(base != origin and not base.is_relative_to(origin) and not origin.is_relative_to(base), 'audit overlaps source')
    job_id = os.environ.get('SLURM_JOB_ID', '')
    need(re.fullmatch('[1-9][0-9]*', job_id) is not None, 'actual native Slurm job required')
    detail = subprocess.check_output(['scontrol', 'show', 'job', '-d', job_id], text=True, timeout=30)
    native_owner_gate(fields(detail), base, repo, job_id, os.getuid())
    run = bind_audit(parent, receipt, p, base, commit, runtime_sources, uuid.uuid4().hex)
    run['slurm_job_id'] = job_id; validate_base(base, run)
    prepare(base); link(base / 'repo', repo)
    data_origin = Path(parent['checkpoint_dir']).parents[1]
    for name in ('calibration-300.parquet', 'audit.json'):
        link(base / 'data' / name, data_origin / 'data' / name)
    for name, sha in run['input_sha256'].items():
        need(digest(base / 'data' / name) == sha, 'audit canonical data differs: ' + name)
    prepare_ray(run)
    publish(base / 'preparation/final-audit-launch-claim.json', dict(
        role=ROLE, audit_id=run['final_restore_audit']['audit_id'], uid=os.getuid(), slurm_job_id=job_id,
        native_job_detail=detail, source=source, transport=transport, audit_source=publication, training_claim_permitted=False,
        created_unix=time.time()))
    publish(base / 'logs/run-manifest.json', run)
    allocation = admit(base)
    need(allocation['training_world_size'] == 4 and len(set(allocation['actual_cuda_runtime_uuids'])) == 4,
         'actual native UUID admission incomplete')
    install_audit_reward_probe(base, receipt)
    publish(base / 'preparation/final-audit-prepared-binding.json', dict(
        audit_id=run['final_restore_audit']['audit_id'],
        run_manifest_sha256=digest(base / 'logs/run-manifest.json'),
        launch_claim_sha256=digest(base / 'preparation/final-audit-launch-claim.json'),
        native_allocation_sha256=digest(base / 'logs/native-allocation-admission.json'),
        transport_proof_path=str(transport_path), transport_proof_sha256=digest(transport_path),
        source_receipt_sha256=digest(p), source_manifest_sha256=source['source_manifest_sha256']))
    return run


def forbid(*args, **kwargs):
    raise RuntimeError('FINAL_RESTORE_AUDIT_FORBIDS_TRAINING_GENERATION_VALIDATION_OR_WRITES')


def lock_writes(worker):
    worker.actor_optimizer.step = forbid
    # LRScheduler.state_dict includes instance attributes. An instance-level
    # replacement of step would corrupt the otherwise exact scheduler readback.
    class ReadOnlyScheduler(type(worker.actor_lr_scheduler)):
        step = forbid
    worker.actor_lr_scheduler.__class__ = ReadOnlyScheduler
    worker.actor.update_policy = forbid


def worker_readback(worker, receipt_path, phase):
    import torch
    from search_g1.full_state_restore import equal_tree
    from search_g1.training_state import cpu_tree, worker_rng
    from search_g1.qprobe_checkpoint import read_source, boundary_worker_rng, validate_rank_state
    from reproduction.a800.owned_processes import identity
    from reproduction.common8.native_allocation import cuda_uuids
    need(phase in ('first', 'second') and os.getuid() == OWNER, 'invalid actual worker readback')
    _, receipt, _, _, _ = read_source(receipt_path)
    need(torch.distributed.get_world_size() == 4, 'actual four-rank group required')
    rank = worker.rank
    row = next(v for v in receipt['files'] if v['relative_path'] == 'training_state/rank-%d.pt' % rank)
    need(digest(row['path']) == row['full_read_sha256'], 'actual Adam source changed')
    saved = torch.load(row['path'], map_location='cpu', weights_only=False)
    validate_rank_state(saved, rank, 4, 200, expected_writes=receipt['attempt_progress']['cumulative_writes'])
    actual_optimizer = cpu_tree(worker.actor_optimizer.state_dict())
    actual_scheduler = cpu_tree(worker.actor_lr_scheduler.state_dict())
    rng = boundary_worker_rng(receipt, rank, saved)
    equal_tree(actual_optimizer, saved['optimizer']); equal_tree(actual_scheduler, saved['scheduler'])
    equal_tree(worker_rng(worker), rng)
    counters = dict(actual_optimizer_writes=getattr(worker.actor, '_optimizer_writes', None),
                    completed_outer=getattr(worker.actor, '_complete_outers', None),
                    last_attempt_index=getattr(worker.actor, '_last_attempt_index', None),
                    used_attempt_ids=sorted(getattr(worker.actor, '_used_attempt_ids', [])),
                    policy_rejections=getattr(worker.actor, '_policy_rejections', None))
    expected = {key: saved[key] for key in counters}
    expected['used_attempt_ids'] = sorted(expected['used_attempt_ids'])
    equal_tree(counters, expected)
    model = {name: parameter for name, parameter in worker.actor_module_fsdp.named_parameters()}
    need(sum(v.numel() for v in model.values()) > 0, 'no actual FSDP model shard read back')
    result = dict(rank=rank, phase=phase, process_identity=identity(os.getpid()),
                  slurm_job_id=os.environ['SLURM_JOB_ID'], actual_cuda_uuid=cuda_uuids(torch.cuda.device_count())[torch.cuda.current_device()],
                  counters=counters, model_shard_sha256=tree_sha(model), optimizer_sha256=tree_sha(actual_optimizer),
                  scheduler_sha256=tree_sha(actual_scheduler), rng_sha256=tree_sha(worker_rng(worker)),
                  actual_adam_scheduler_readback_equal=True, actual_rng_readback_equal=True, checked_unix=time.time())
    publish(Path(os.environ['SEARCH_G1_BASE']) / 'logs' / ('final-audit-%s-rank-%d.json' % (phase, rank)), result)
    return result


def audit_worker_class():
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.single_controller.base.decorator import register, Dispatch

    class AuditWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def restore_mixed_full_state(self, receipt_path):
            from search_g1.full_state_restore import restore_worker
            proof = restore_worker(self, receipt_path)
            lock_writes(self)
            return proof

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def final_audit_readback(self, receipt_path, phase):
            return worker_readback(self, receipt_path, phase)

        @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
        def update_actor(self, *args, **kwargs): return forbid()

        @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
        def generate_sequences(self, *args, **kwargs): return forbid()

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def save_checkpoint(self, *args, **kwargs): return forbid()
    return AuditWorker


def driver_readback(trainer, runtime, stream, receipt, phase):
    import torch
    from search_g1.full_state_restore import equal_tree
    from search_g1.training_state import compact_stream_state, driver_rng
    need(phase in ('first', 'second'), 'invalid driver readback phase')
    boundary = torch.load(receipt['boundary_driver']['path'], map_location='cpu', weights_only=False)
    state = torch.load(receipt['question_probe_state_path'], map_location='cpu', weights_only=False)
    head = torch.load(receipt['head_path'], map_location='cpu', weights_only=True)
    need(head['completed_outer'] == head['update_count'] == 200 and
         len(head['completed_attempts']) == 200 and runtime.completed_outer == 200 and runtime.pending is None,
         'head is not the completed200 successful-outer state')
    need(trainer.global_steps == 200, 'actual driver boundary is not200')
    need(set(head['completed_attempts']) == {receipt['run_id'] + ':' + str(outer) for outer in range(1, 201)} and
         set(head['completed_attempts'].values()) == set(state['completed_attempts']),
         'head and runtime successful-attempt histories differ')
    equal_tree(runtime.head.state_dict(), head)
    equal_tree(runtime.selection_rng.bit_generator.state, state['selection_rng'])
    equal_tree(sorted(runtime.completed_attempts), sorted(state['completed_attempts']))
    actual_stream = compact_stream_state(stream.state())
    equal_tree(actual_stream, compact_stream_state(boundary['candidate_stream']))
    equal_tree(actual_stream, compact_stream_state(state['candidate_state']))
    equal_tree(driver_rng(), boundary['rng'])
    equal_tree(trainer._restored_attempt_progress, boundary['attempt_progress'])
    need(runtime.version == receipt['head_version'] and runtime.feature_spec_sha == receipt['feature_spec_sha'],
         'actual head/feature namespace differs')
    need(not hasattr(runtime.head, 'scheduler'), 'head unexpectedly acquired a scheduler')
    return dict(phase=phase, completed_outer=200, head_update_count=runtime.head.update_count,
                head_scheduler=None, head_schedule='constant_lr_Adam', head_sha256=tree_sha(runtime.head.state_dict()),
                selector_rng_sha256=tree_sha(runtime.selection_rng.bit_generator.state),
                candidate_stream_sha256=tree_sha(actual_stream), driver_rng_sha256=tree_sha(driver_rng()),
                actual_head_model_adam_private_rng_replay_readback_equal=True,
                actual_selector_rng_readback_equal=True, actual_candidate_stream_readback_equal=True,
                actual_driver_rng_readback_equal=True, checked_unix=time.time())


def verify_two_boundaries(first, second, allocation, receipt, driver_first, driver_second):
    from search_g1.full_state_restore import equal_tree
    need(allocation.get('accepted') is True and allocation.get('training_world_size') == 4 and
         allocation.get('run_id') == receipt['run_id'], 'actual native allocation not accepted for this run')
    for rows, phase in ((first, 'first'), (second, 'second')):
        need(len(rows) == 4 and sorted(v['rank'] for v in rows) == list(range(4)), 'actual worker ranks incomplete')
        need({v['actual_cuda_uuid'] for v in rows} == set(allocation['actual_cuda_runtime_uuids']) and
             len(set(allocation['actual_cuda_runtime_uuids'])) == 4, 'native worker UUIDs differ')
        need(len({(v['process_identity']['pid'], v['process_identity']['start_ticks']) for v in rows}) == 4,
             'worker process identities not distinct')
        for row in rows:
            need(row['phase'] == phase and row['slurm_job_id'] == allocation['slurm_job_id'] and
                 row['process_identity']['uid'] == OWNER and row['actual_adam_scheduler_readback_equal'] is True and
                 row['actual_rng_readback_equal'] is True, 'worker readback ownership/phase differs')
            need(row['counters']['completed_outer'] == 200 and row['counters']['actual_optimizer_writes'] ==
                 receipt['attempt_progress']['cumulative_writes'], 'worker advanced during final audit')
    def stable(row):
        value = {k: v for k, v in row.items() if k not in ('phase', 'checked_unix')}
        if 'process_identity' in value:
            # Ray changes process titles and /proc running/sleeping states
            # between RPCs; only immutable process birth/owner identifies it.
            value['process_identity'] = {key: value['process_identity'][key] for key in ('pid', 'start_ticks', 'sid', 'uid')}
        return value
    equal_tree([stable(r) for r in sorted(first, key=lambda v: v['rank'])],
               [stable(r) for r in sorted(second, key=lambda v: v['rank'])])
    equal_tree(stable(driver_first), stable(driver_second))
    need(driver_first['completed_outer'] == driver_second['completed_outer'] == 200 and
         driver_first['head_update_count'] == driver_second['head_update_count'] == 200,
         'head/driver advanced during audit')


def native_task(config, base):
    """Separate Ray producer; the production main_task's unconditional fit is never called."""
    import ray
    from verl.utils import hf_tokenizer
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.trainer.main_ppo_eco import EcoRewardManager
    from search_g1.qprobe_checkpoint import read_source, verified_runtime_prefix
    from search_g1.full_state_restore import restore_driver
    from search_g1.qprobe_runtime import QProbeRuntime, QProbeCandidateStream
    from reproduction.a800.owned_processes import identity
    base = Path(base)
    actual_driver = identity(os.getpid())
    need(actual_driver['uid'] == OWNER, 'actual native Ray driver has another owner')
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    need(run['final_restore_audit']['role'] == ROLE and run['full_state_restore']['completed_outer'] == 200,
         'not a final restore-only producer')
    receipt_path = run['full_state_restore']['source_receipt']
    _, receipt, _, _, _ = read_source(receipt_path)
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
    worker_cls = audit_worker_class()
    roles = {Role.ActorRollout: ray.remote(worker_cls), Role.RefPolicy: ray.remote(worker_cls)}
    pool = ResourcePoolManager(resource_pool_spec={'audit': [4]}, mapping={key: 'audit' for key in roles})

    class AuditTrainer(RayPPOTrainer):
        fit = forbid
        _validate = forbid
        _save_checkpoint = forbid

    class AuditReward(EcoRewardManager):
        __call__ = forbid
    reward = AuditReward(tokenizer=tokenizer, num_examine=0, reward_mode='eco')
    trainer = AuditTrainer(config=config, tokenizer=tokenizer, role_worker_mapping=roles,
                          resource_pool_manager=pool, ray_worker_group_cls=RayWorkerGroup,
                          reward_fn=reward, val_reward_fn=None)
    trainer.init_workers()  # Actual FSDP/HF + Adam/scheduler + sampling RNG restore on all4.
    first = trainer.actor_rollout_wg.final_audit_readback(receipt_path, 'first')
    restore_driver(trainer, receipt_path)
    trainer.global_steps = trainer._restored_attempt_progress['completed_outer']
    stream = QProbeCandidateStream(trainer.train_dataloader, restored_state=trainer._restored_candidate_state)
    runtime = QProbeRuntime(config.question_probe, base, tokenizer)
    feature_proof = runtime.cache.verify_all()
    runtime.restore(receipt['question_probe_state_path'], receipt['actor_snapshot_id'])
    runtime.head.optimizer.step = forbid
    runtime.head._fit = forbid
    runtime.complete_outer = forbid
    runtime.begin_outer = forbid
    first_driver = driver_readback(trainer, runtime, stream, receipt, 'first')
    publish(base / 'logs/final-audit-first-driver.json', first_driver)
    second = trainer.actor_rollout_wg.final_audit_readback(receipt_path, 'second')
    second_driver = driver_readback(trainer, runtime, stream, receipt, 'second')
    publish(base / 'logs/final-audit-second-driver.json', second_driver)
    allocation = json.loads((base / 'logs/native-allocation-admission.json').read_text())
    verify_two_boundaries(first, second, allocation, receipt, first_driver, second_driver)
    prefix = verified_runtime_prefix(base, run)
    need(prefix['completed_outer'] == 200, 'actual restore prefix is not200')
    for name in FORBIDDEN_OUTPUTS:
        need(not (base / 'logs' / name).exists(), 'forbidden training output was created')
    proof = dict(schema_version=1, role=ROLE, audit_id=run['final_restore_audit']['audit_id'],
                 run_id=run['run_id'], source_receipt_sha256=digest(receipt_path),
                 actual_loaded_target_receipt=receipt_path, actual_loaded_target_checkpoint=receipt['checkpoint'],
                 actual_loaded_target_head=receipt['head_path'],
                 producer_commit=json.loads((base / 'preparation/final-audit-launch-claim.json').read_text())['source']['producer_commit'],
                 audit_commit=run['source_commit'], slurm_job_id=allocation['slurm_job_id'], world_size=4,
                 actual_driver_process=actual_driver,
                 actual_cuda_restore_accepted=True, actual_completed_outer=200,
                 actual_actor_optimizer_write_delta=0, actual_actor_scheduler_advance_delta=0,
                 actual_head_update_delta=0, actual_head_optimizer_write_delta=0,
                 head_scheduler=None, generated_trajectories=0, fit_called=False, update201_performed=False,
                 workers_first=first, workers_second=second, driver_first=first_driver, driver_second=second_driver,
                 actual_feature_cache=feature_proof,
                 actual_worker_restore_prefix=prefix, checked_unix=time.time(),
                 scope='actual native4 final200 restore and two equal readbacks; owned cleanup and final goal audit remain separate')
    publish(base / 'logs/final-restore-readback.json', proof)
    return proof


def worker_main(base):
    import ray
    from reproduction.common8.configuration import resolved
    from reproduction.common8.native_ray import verify
    from reproduction.a800.owned_processes import identity
    base = Path(base)
    binding = json.loads((base / 'preparation/final-audit-prepared-binding.json').read_text())
    for name, key in (('logs/run-manifest.json', 'run_manifest_sha256'),
                      ('preparation/final-audit-launch-claim.json', 'launch_claim_sha256'),
                      ('logs/native-allocation-admission.json', 'native_allocation_sha256')):
        need(digest(base / name) == binding[key], 'prepared binding changed: ' + name)
    run, config, _ = resolved(base)
    claim = json.loads((base / 'preparation/final-audit-launch-claim.json').read_text())
    need(claim['audit_id'] == run['final_restore_audit']['audit_id'] and os.getuid() == OWNER and
         claim['slurm_job_id'] == os.environ['SLURM_JOB_ID'], 'worker audit claim differs')
    restore = run['full_state_restore']
    need(binding['audit_id'] == run['final_restore_audit']['audit_id'] and
         digest(restore['source_receipt']) == binding['source_receipt_sha256'] == restore['source_receipt_sha256'] and
         restore['source_run_manifest_sha256'] == binding['source_manifest_sha256'], 'prepared source receipt differs')
    need(digest(binding['transport_proof_path']) == binding['transport_proof_sha256'], 'prepared transport proof changed')
    ledger = json.loads((base / 'logs/final-audit-process-ledger.json').read_text())
    leader = ledger['jobs']['restore-only']['leader']; current = identity(os.getpid())
    need(ledger['run_id'] == run['run_id'] and all(current[k] == leader[k] for k in ('pid', 'start_ticks', 'sid', 'uid')),
         'only the retained owner child may launch this worker')
    publish(base / 'logs/final-audit-worker-once.json', dict(audit_id=binding['audit_id'],
                                                         actual_process=current, checked_unix=time.time()))
    need(not ray.is_initialized(), 'private fresh local Ray required')
    root = verify(run, os.environ['RAY_TMPDIR'])
    ray.init(address='local', num_cpus=48, num_gpus=4, object_store_memory=8 * 1024**3, _temp_dir=root,
             runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    try:
        node = ray._private.worker._global_node
        need(node is not None and isinstance(node.all_processes, dict) and node.all_processes,
             'actual private Ray service process registry unavailable')
        services = []
        for kind, processes in node.all_processes.items():
            for item in processes:
                birth = identity(item.process.pid)
                need(birth['uid'] == OWNER, 'private Ray service has another owner')
                services.append(dict(component=kind, **{key: birth[key] for key in ('pid', 'start_ticks', 'sid', 'uid')}))
        need(services, 'no actual Ray service births captured')
        publish(base / 'logs/final-audit-ray-service-births.json', dict(audit_id=binding['audit_id'], processes=services))
        ray.get(ray.remote(native_task).remote(config, str(base)))
    finally:
        ray.shutdown()


def cleanup_retained_child(owner, child):
    """Clean only the held child's authenticated session, including a zombie leader."""
    from reproduction.a800.owned_processes import signal_owned_child_group
    if child is None:
        return
    job = owner.jobs['restore-only']
    try:
        for sig, seconds in ((signal.SIGINT, 15), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
            if not owner.observe('restore-only'):
                job['state'] = 'STOPPED'; owner.save(); return
            job['state'] = 'CLEANUP_PENDING'; owner.save()
            signal_owned_child_group(job['leader'], child, sig)
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                if not owner.observe('restore-only'):
                    job['state'] = 'STOPPED'; owner.save(); return
                time.sleep(.2)
        need(not owner.observe('restore-only'), 'actual held-child cleanup incomplete')
        job['state'] = 'STOPPED'; owner.save()
    except Exception as error:
        job.update(state='CLEANUP_FAILED', cleanup_error=repr(error)); owner.save(); raise


def run(base, commit, receipt_path, transport_path):
    from reproduction.a800.owned_processes import OwnedProcesses
    from reproduction.hpc3_qprobe.native_jit import prepare as prepare_jit
    base = Path(base); run_manifest = prepare_audit(base, commit, receipt_path, transport_path)
    env = dict(os.environ, SEARCH_G1_BASE=str(base), SEARCH_G1_PROFILE='common8',
               SEARCH_G1_TRAINING_WORLD_SIZE='4', QPROBE_SOURCE=str(base / 'repo'),
               RAY_TMPDIR=run_manifest['ray_tmp_dir'], OMP_NUM_THREADS='8', HF_HUB_OFFLINE='1',
               HF_HOME=str(base / 'cache/hf'), XDG_CACHE_HOME=str(base / 'cache'),
               HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', ECO_NO_SELF='1', ECO_USE_CF='0',
               ECO_TRUST_VARIANT='balanced', ECO_TRUST_SIGNAL='probe', ECO_PROBE_GATE='1',
               ECO_BAL_GROUND='0.6', ECO_BAL_NOSEARCH_BONUS='0.5', ECO_W_COST='0.02',
               ECO_WRONG_GROUND_SCALE='0', ECO_PROBE_BASE='0', ECO_PROBE_ALPHA_COST='0',
               C_INCLUDE_PATH=str(base / 'bootstrap-tools/python-dev/usr/include/python3.10') + ':' +
                              str(base / 'bootstrap-tools/python-dev/usr/include'))
    for key in ('PYTHONHOME', 'RAY_ADDRESS', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    env.update(prepare_jit(base, run_manifest))
    owner = OwnedProcesses(base / 'logs/final-audit-process-ledger.json', run_manifest['run_id'])
    child = None
    try:
        child = owner.start('restore-only', [str(base / 'env/bin/python'), '-m', MODULE,
                                            '--worker', '--base', str(base)], env,
                            base / 'logs/final-audit-worker.log', base / 'repo')
        # WNOWAIT preserves the direct-child PID for safe old-kernel cleanup.
        while True:
            owner.observe('restore-only')
            status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if status is not None:
                need(status.si_code == os.CLD_EXITED and status.si_status == 0,
                     'actual restore-only child failed; preserve worker log')
                break
            time.sleep(1)
        need((base / 'logs/final-restore-readback.json').is_file(), 'native readback receipt missing')
    finally:
        try:
            cleanup_retained_child(owner, child)
        finally:
            if child is not None and not owner.observe('restore-only'):
                child.wait(timeout=5)
    # Final acceptance follows actual owned-process cleanup, not merely Ray.shutdown.
    need(all(v['state'] == 'STOPPED' for v in owner.jobs.values()), 'owned cleanup incomplete')
    from reproduction.a800.owned_processes import same, identity
    readback = json.loads((base / 'logs/final-restore-readback.json').read_text())
    services = json.loads((base / 'logs/final-audit-ray-service-births.json').read_text())['processes']
    births = services + [row['process_identity'] for row in readback['workers_second']] + [readback['actual_driver_process']]
    deadline = time.monotonic() + 30
    while True:
        live = [birth for birth in births if same(birth) and identity(birth['pid'])['state'] != 'Z']
        if not live: break
        need(time.monotonic() < deadline, 'actual Ray service/restored worker survived owned cleanup')
        time.sleep(.2)
    original = source_proof(receipt_path)[-1]
    transport = verify_transport(transport_path, receipt_path)
    before = json.loads((base / 'preparation/final-audit-launch-claim.json').read_text())['source']
    need(original['source_receipt_sha256'] == before['source_receipt_sha256'] and
         original['full_read_files'] == before['full_read_files'], 'source changed across actual restore audit')
    need(transport['sha256'] == json.loads((base / 'preparation/final-audit-launch-claim.json').read_text())['transport']['sha256'],
         'transport proof changed across audit')
    need(readback['actual_cuda_restore_accepted'] is True, 'actual native restore was not accepted')
    publish(base / 'logs/final-restore-audit-accepted.json', dict(
        role=ROLE, audit_id=run_manifest['final_restore_audit']['audit_id'], accepted=True,
        actual_cuda_restore_accepted=True, readback_sha256=digest(base / 'logs/final-restore-readback.json'),
        source_receipt_sha256=original['source_receipt_sha256'], owned_cleanup=True,
        transport_proof_sha256=transport['sha256'],
        original_source_receipt=transport['source_receipt'], archive_receipt=transport['archive_receipt'],
        actual_loaded_target_receipt=transport['target_receipt'],
        actual_completed_outer=200, update201_performed=False, checked_unix=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--commit')
    parser.add_argument('--source-receipt')
    parser.add_argument('--transport-proof')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        need(args.commit is None and args.source_receipt is None and args.transport_proof is None, 'worker cannot rebind source')
        worker_main(args.base)
    else:
        need(args.commit is not None and args.source_receipt is not None and args.transport_proof is not None,
             'immutable source and three-way transport bindings required')
        run(args.base, args.commit, args.source_receipt, args.transport_proof)


if __name__ == '__main__':
    main()
