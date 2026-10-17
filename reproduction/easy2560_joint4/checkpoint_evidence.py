"""Immutable CPU evidence sidecar for the reviewed fresh Easy2560 run.

This independent operator does not load Torch, allocate GPUs, train, restore,
modify a producer checkpoint, or amend an accepted archive. Large model/data
and selected token traces remain references to the existing remote objects.
Only complete, exact checkpoint prefixes from one original job are supported;
restored/cross-job runs require a separately implemented authenticated chain.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time
import uuid

REVIEWED_COMMIT = '93eda66a978dae7a1086fdb2e6f07ef4a7fc3d48'
STEPS = (5, 50, 100, 150, 200)
STATUS = 'EASY2560_CHECKPOINT_EVIDENCE_SEALED'
ARCHIVE_STATUS = 'EASY2560_FULL_CHECKPOINT_SOURCE_ARCHIVE_SHA_PASS'
MAX_SMALL = 128 * 1024**2
MAX_LINE = 32 * 1024**2


def need(ok, message):
    if not ok:
        raise ValueError('EASY2560_EVIDENCE_' + message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + '\n').encode()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_uid)


def owned_base(value):
    p = Path(value)
    need(p.is_absolute() and p.resolve() == p and re.fullmatch(
        r'/data/user/jxu957/experiments/easy2560-corollout-joint4-[A-Za-z0-9._-]+', str(p)),
        'DEDICATED_CANONICAL_BASE_REQUIRED')
    # The human-authorized shared experiment root was created by mou176;
    # jxu957 owns the actual run/evidence files and executes this operator.
    # Directory ownership is not equivalent to execution/file ownership.
    need(os.getuid() == os.geteuid() == 14090, 'JXU_EXECUTION_UID_REQUIRED')
    st = p.stat()
    need(stat.S_ISDIR(st.st_mode) and st.st_uid in (14090, 21930) and st.st_gid == 21919,
         'AUTHORIZED_SHARED_BASE_REQUIRED')
    need(os.access(p, os.W_OK | os.X_OK), 'SHARED_BASE_NOT_WRITABLE')
    return p


def regular(path, *, external=False):
    path = Path(path)
    need(path.is_absolute(), 'ABSOLUTE_SOURCE_REQUIRED')
    if external:
        path = path.resolve(strict=True)
    need(path.resolve() == path and not path.is_symlink(), 'SYMLINK_SOURCE')
    st = path.stat()
    need(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid(), 'REGULAR_OWNED_SOURCE_REQUIRED')
    return path


def read_small(path, *, expected=None, external=False):
    path = regular(path, external=external)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        need(before.st_size <= MAX_SMALL, 'SMALL_SOURCE_LIMIT')
        raw = stream.read(MAX_SMALL + 1)
        after = os.fstat(stream.fileno())
    need(identity(before) == identity(after) == identity(path.stat()), 'SOURCE_CHANGED_DURING_READ')
    actual = digest(raw)
    need(expected is None or actual == expected, 'SOURCE_SHA_DIFFERS')
    return raw, dict(path=str(path), bytes=len(raw), sha256=actual,
                     device=after.st_dev, inode=after.st_ino)


def json_small(path, **kwargs):
    raw, record = read_small(path, **kwargs)
    return json.loads(raw), raw, record


def prefix(path, terminal):
    """Stop at the requested complete row; never parse a later partial suffix."""
    path = regular(path)
    rows = []; chunks = []
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        while True:
            raw = stream.readline(MAX_LINE + 1)
            need(raw and raw.endswith(b'\n') and len(raw) <= MAX_LINE, 'MISSING_OR_PARTIAL_CP_PREFIX')
            row = json.loads(raw); rows.append(row); chunks.append(raw)
            if terminal(row):
                break
        length = stream.tell()
        after = os.fstat(stream.fileno())
        need((before.st_dev, before.st_ino, before.st_uid) ==
             (after.st_dev, after.st_ino, after.st_uid) and after.st_size >= length,
             'LEDGER_REPLACED_DURING_READ')
        stream.seek(0)
        need(stream.read(length) == b''.join(chunks), 'LEDGER_PREFIX_CHANGED_DURING_READ')
    need((path.stat().st_dev, path.stat().st_ino) == (before.st_dev, before.st_ino),
         'LEDGER_REPLACED')
    raw = b''.join(chunks)
    return rows, raw, dict(path=str(path), bytes=len(raw), sha256=digest(raw),
                           prefix_only=True, later_suffix_excluded=True,
                           device=before.st_dev, inode=before.st_ino)


def attempt_identity(row):
    value = row if 'attempt_index' in row else row.get('context', {})
    return value['attempt_index'], value['attempt_id']


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Bundle:
    def __init__(self, root):
        self.root = root; self.files = {}; self.source_reads = []

    def put(self, name, raw, source=None):
        part = Path(name)
        need(not part.is_absolute() and '..' not in part.parts and name not in self.files,
             'UNSAFE_OR_DUPLICATE_OUTPUT')
        target = self.root / part; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        actual = target.read_bytes()
        need(actual == raw, 'DESTINATION_READBACK_DIFFERS')
        self.files[name] = dict(bytes=len(raw), sha256=digest(actual))
        if source is not None:
            self.source_reads.append(dict(source, sealed_as=name))

    def copy(self, name, source, *, parse_json=True, **kwargs):
        raw, record = read_small(source, **kwargs); self.put(name, raw, record)
        return json.loads(raw) if parse_json else None


def _validate_ledgers(base, run, step, driver, bundle, job):
    progress = driver['attempt_progress']; cutoff = progress['attempt_index']
    updates, raw, record = prefix(base/'logs/actual-updates.jsonl', lambda r: r.get('step') == step)
    need([r['step'] for r in updates] == list(range(1, step + 1)), 'UPDATE_PREFIX_NOT_FRESH_CONTIGUOUS')
    need(all(r['run_id'] == run['run_id'] for r in updates), 'FOREIGN_UPDATE_RUN')
    need(updates[-1]['probe_version'] == driver['probe_version'], 'SAVED_PROBE_AND_UPDATE_DIFFER')
    bundle.put('prefix/actual-updates.jsonl', raw, record)
    attempts, raw, record = prefix(base/'logs/actual-attempts.jsonl',
                                   lambda r: r.get('attempt_index') == cutoff)
    need([r['attempt_index'] for r in attempts] == list(range(1, cutoff + 1)), 'ATTEMPT_PREFIX_NOT_FRESH_CONTIGUOUS')
    need(len({r['attempt_id'] for r in attempts}) == cutoff, 'ATTEMPT_UUID_REUSED')
    writes = completed = aborted = full = 0
    successful = []
    for r in attempts:
        need(r['run_id'] == run['run_id'] and r['slurm_job_id'] == job and
             r['all_ranks_agreed'] is True and r['real_rank_count'] == 4,
             'CROSS_JOB_OR_FOREIGN_ATTEMPT')
        need(r['logical_update'] == completed + 1 and r['optimizer_writes_before'] == writes,
             'ATTEMPT_COUNTER_CHAIN_DIFFERS')
        count = r['optimizer_writes_this_attempt']; processed = r['opportunities_processed']
        need(type(count) is int and 0 <= count <= processed <= 20 and
             r['optimizer_writes_after'] == writes + count and r['complete_outer'] is (count > 0),
             'ACTUAL_WRITES_OR_COMPLETION_DIFFERS')
        writes += count; completed += int(count > 0); aborted += int(count == 0)
        full += int(count == 20 and processed == 20)
        need(r['completed_outer'] == completed and r['cumulative_aborted_attempts'] == aborted,
             'ATTEMPT_PROGRESS_DIFFERS')
        if count: successful.append(r)
    need(completed == step and writes == progress['cumulative_writes'] and
         aborted == progress['aborted_attempts'] and full == progress['full_budget_outer'],
         'DRIVER_AND_LEDGER_PROGRESS_DIFFER')
    need(len(successful) == len(updates), 'UPDATE_ATTEMPT_LENGTH_DIFFERS')
    for u, a in zip(updates, successful):
        need((u['step'], u['attempt_id'], u['attempt_index'], u['optimizer_writes_after']) ==
             (a['completed_outer'], a['attempt_id'], a['attempt_index'], a['optimizer_writes_after']),
             'UPDATE_ATTEMPT_BINDING_DIFFERS')
    bundle.put('prefix/actual-attempts.jsonl', raw, record)
    by_id = {r['attempt_id']: r for r in attempts}
    for rank in range(4):
        for role, terminal in (('scheduler', ('COMPLETE', 'ABORTED_ATTEMPT')),
                               ('optimizer', ('OUTER_BUDGET', 'ABORTED_ATTEMPT'))):
            path = base/'logs'/f'{role}-rank-{rank}.jsonl'
            rows, raw, record = prefix(path, lambda r: attempt_identity(r)[0] == cutoff and r['kind'] in terminal)
            seen = []; groups = {}
            for row in rows:
                index, aid = attempt_identity(row)
                need(aid in by_id and index == by_id[aid]['attempt_index'] and
                     row['run_id'] == run['run_id'] and row['rank'] == rank,
                     'RANK_PREFIX_IDENTITY_DIFFERS')
                if aid not in groups: groups[aid] = []; seen.append(aid)
                groups[aid].append(row)
                if role == 'scheduler':
                    need(row['slurm_job_id'] == job and row['world_size'] == 4,
                         'RANK_PREFIX_CROSS_JOB')
                if role == 'optimizer' and row['kind'] == 'POLICY_MONITOR':
                    need(row['objective_weight_definition'] ==
                         'equal four-row logical micro masked_mean; weighted global numerator/denominator',
                         'LOGICAL_MICRO_LABEL_DIFFERS')
                    need(len(row['monitor_identity']['selection_indices']) == 8,
                         'MONITOR_COUNT_DIFFERS')
            need(seen == [r['attempt_id'] for r in attempts], 'MISSING_OR_REORDERED_RANK_ATTEMPT')
            for aid, events in groups.items():
                attempt = by_id[aid]; terminal_row = events[-1]
                expected = ('COMPLETE' if attempt['complete_outer'] else 'ABORTED_ATTEMPT') if role == 'scheduler' else (
                            'OUTER_BUDGET' if attempt['complete_outer'] else 'ABORTED_ATTEMPT')
                need(terminal_row['kind'] == expected, 'INCOMPLETE_RANK_ATTEMPT')
                if role == 'scheduler':
                    need([r['kind'] for r in events] == (['BEGIN', 'COMMIT_PREPARED', 'COMPLETE']
                         if attempt['complete_outer'] else ['BEGIN', 'ABORTED_ATTEMPT']), 'SCHEDULER_PREFIX_SHAPE')
                    need(terminal_row['writes_after'] == attempt['optimizer_writes_after'], 'SCHEDULER_WRITES_DIFFER')
                else:
                    need(terminal_row['cumulative_writes'] == attempt['optimizer_writes_after'], 'WORKER_WRITES_DIFFER')
                    need(sum(r['kind'] == 'OPTIMIZER_WRITE' for r in events) ==
                         attempt['optimizer_writes_this_attempt'], 'WORKER_WRITE_RECEIPTS_DIFFER')
            bundle.put(f'prefix/{role}-rank-{rank}.jsonl', raw, record)
    return updates, attempts


def _trace_references(base, run, updates, attempts, bundle):
    references = []
    for attempt in attempts:
        aid = attempt['attempt_id']; name = f'logs/behavior-probability-attempt-{aid}.json'
        gate = bundle.copy('gates/' + Path(name).name, base/name)
        need(gate['run_id'] == run['run_id'] and gate['attempt_id'] == aid and
             gate['attempt_index'] == attempt['attempt_index'] and gate['logical_update'] == attempt['logical_update']
             and gate['accepted'] is True and gate['slurm_job_id'] == attempt['slurm_job_id']
             and gate['source_sha256'] == run['source_sha256'], 'GATE_ATTEMPT_DIFFERS')
        trace = regular(base/'artifacts/behavior-traces'/f'attempt-{aid}.json')
        h = hashlib.sha256()
        with trace.open('rb') as stream:
            before = os.fstat(stream.fileno())
            for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
            after = os.fstat(stream.fileno())
        need(identity(before) == identity(after) == identity(trace.stat()) and
             h.hexdigest() == gate['private_trace_sha256'] == gate['trace_sha256'], 'SELECTED_TRACE_SHA_DIFFERS')
        references.append(dict(path=str(trace), bytes=after.st_size, sha256=h.hexdigest(),
                               copied=False, requires_remote_retention=True, attempt_id=aid))
    for update in updates:
        gate_path = f'gates/behavior-probability-attempt-{update["attempt_id"]}.json'
        need(bundle.files[gate_path]['sha256'] == update['behavior_probability_gate_sha256'], 'UPDATE_GATE_SHA_DIFFERS')
        selection = bundle.copy('selection/' + update['attempt_id'] + '.json',
                    base/'artifacts/mixed-selection'/(update['attempt_id'] + '.json'))
        need(selection['step'] == update['step'] and selection['source_commit'] == run['source_commit'] and
             selection['policy_snapshot_id'] == run['run_id']+':'+str(update['step'])+':policy-'+update['attempt_id']
             and selection['fixed_candidate_budget'] is True and selection['attempted_candidates'] == 2560
             and selection['selected_groups'] == 512, 'SELECTION_CP_BINDING_DIFFERS')
    bundle.put('selected-large-references.json', encoded(references))
    return references


def _validation_observation(base, step, bundle):
    path = regular(base/'logs/validation-identities.jsonl')
    # Bounded current observation: complete records only; later rows are not a
    # continuation RNG boundary and do not change the immutable CP phase.
    with path.open('rb') as stream:
        bound = os.fstat(stream.fileno()).st_size
        raw = stream.read(bound)
    rows = []; chunks = []
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'): break
        value = json.loads(line)
        if value['step'] > step: break
        rows.append(value); chunks.append(line)
    sealed = b''.join(chunks)
    with path.open('rb') as stream:
        need(stream.read(len(sealed)) == sealed, 'VALIDATION_PREFIX_CHANGED')
    bundle.put('prefix/validation-identities.jsonl', sealed,
               dict(path=str(path), bytes=len(sealed), sha256=digest(sealed), prefix_only=True))
    fixed = [r for r in rows if r['step'] == step and r['split'] == 'validation']
    need(step == 5 or len(fixed) == 1, 'PRE_SAVE_FIXED512_EVIDENCE_MISSING_OR_DUPLICATED')
    return dict(fixed512_seen_at_observation=bool(fixed),
                final51713_seen_at_observation=any(r['step'] == step and r['split'] == 'final' for r in rows))


def verify(directory):
    directory = Path(directory).resolve()
    marker, _, marker_record = json_small(directory/'complete.json')
    manifest, _, _ = json_small(directory/'supplement.json', expected=marker['supplement_sha256'])
    need(marker['status'] == STATUS and marker['actual_cuda_restore_accepted'] is False and
         marker['post_callback_rng_captured'] is False and
         manifest['actual_cuda_restore_accepted'] is False and
         manifest['role'] == 'easy2560_checkpoint_evidence_supplement' and
         manifest['post_callback_rng_captured'] is False and
         manifest['single_original_job_prefix'] is True and not manifest['inherited_history_assembled'],
         'UNSUPPORTED_ACCEPTANCE_CLAIM')
    need(all(marker[k] == manifest[k] for k in ('source_commit', 'run_id', 'source_job', 'step')),
         'SEALED_IDENTITY_DIFFERS')
    for name, record in manifest['files'].items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts and
             (directory/name).resolve().is_relative_to(directory), 'SEALED_FILE_ESCAPE')
        _, actual = read_small(directory/name, expected=record['sha256'])
        need(actual['bytes'] == record['bytes'], 'SEALED_FILE_SIZE_DIFFERS')
    core = manifest['core_archive']
    archive, _, _ = json_small(directory/core['source_acceptance']['relative_path'],
                              expected=core['source_acceptance']['sha256'])
    snapshot, _, _ = json_small(directory/core['source_snapshot']['relative_path'],
                               expected=core['source_snapshot']['sha256'])
    need(archive['status'] == ARCHIVE_STATUS == core['status'] and
         archive['snapshot_sha256'] == core['source_snapshot']['sha256'] and
         all(archive[k] == snapshot[k] for k in ('base', 'run_id', 'source_commit', 'step', 'total_bytes')) and
         archive['base'] == manifest['source_base'] and archive['run_id'] == manifest['run_id'] and
         archive['source_commit'] == manifest['source_commit'] and archive['step'] == manifest['step'] and
         archive['actual_cuda_restore_accepted'] is False, 'SEALED_CORE_ARCHIVE_IDENTITY_DIFFERS')
    files = dict(manifest['files'])
    for name in ('supplement.json', 'complete.json'):
        raw, record = read_small(directory/name); files[name] = dict(bytes=len(raw), sha256=record['sha256'])
    return dict(marker, source_base=manifest['source_base'], source_checkpoint=manifest['source_checkpoint'],
                core_archive=manifest['core_archive'], single_original_job_prefix=True,
                inherited_history_assembled=False, selected_trace_bytes_copied=0,
                checkpoint_large_files_rehashed_by_this_operator=False,
                scope=manifest['scope'], directory=str(directory), files_to_archive=[dict(
        relative_path=name, path=str(directory/name), **record) for name, record in sorted(files.items())])


def recheck_sources(records):
    """Copied prefixes remain stable across harmless later append operations."""
    for source in records:
        path = regular(source['path'])
        with path.open('rb') as f:
            if not source.get('prefix_only', False):
                need(os.fstat(f.fileno()).st_size == source['bytes'], 'IMMUTABLE_SOURCE_SIZE_CHANGED')
            raw = f.read(source['bytes'])
        need(len(raw) == source['bytes'] and digest(raw) == source['sha256'], 'SOURCE_AFTER_COPY_SHA_DIFFERS')


def seal(base, training_commit, run_id, source_job, step):
    base = owned_base(base)
    need(training_commit == REVIEWED_COMMIT and step in STEPS, 'REVIEWED_COMMIT_AND_FIXED_CP_REQUIRED')
    root = base/'preparation/checkpoint-evidence'; root.mkdir(parents=True, exist_ok=True)
    need(root.resolve() == root, 'EVIDENCE_ROOT_SYMLINK')
    fsync_directory(root.parent)
    with (root/'seal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = root/f'global_step_{step}'
        if target.exists():
            result = verify(target)
            need((result['source_commit'], result['run_id'], result['source_job'], result['step']) ==
                 (training_commit, run_id, source_job, step), 'EXISTING_SIDECAR_BINDING_DIFFERS')
            manifest, _, _ = json_small(target/'supplement.json')
            recheck_sources(manifest['source_reads'])
            return result
        stage = root/('.partial-' + uuid.uuid4().hex); stage.mkdir()
        try:
            bundle = Bundle(stage)
            run = bundle.copy('identity/run-manifest.json', base/'logs/run-manifest.json')
            need(run['run_id'] == run_id and run['source_commit'] == training_commit and
                 run.get('easy2560_selection') is True and run.get('profile') == 'common8' and
                 run.get('execution_site') == 'hpc3' and run.get('execution_backend', 'slurm') == 'slurm' and
                 run['training_world_size'] == 4 and
                 run['data_selection_mode'] == 'mixed_first' and run.get('question_probe') is None and
                 run['actor_logical_micro_batch_per_rank'] == 4 and
                 type(run['actor_micro_batch_per_rank']) is int and run['actor_micro_batch_per_rank'] in (4, 8, 16, 32) and
                 run.get('selection_candidate_questions') == 2560 and run.get('selection_selected_questions') == 512 and
                 run.get('selection_fixed_candidate_budget') is True and run['all_correct_virtual_zero'] is False,
                 'EXACT_EASY_SCOPE_DIFFERS')
            need(not run.get('full_state_restore'), 'CROSS_JOB_AUTHENTICATED_PREFIX_NOT_IMPLEMENTED')
            allocation = bundle.copy('identity/native-allocation-admission.json', base/'logs/native-allocation-admission.json')
            need(allocation['accepted'] is True and allocation['run_id'] == run_id and
                 allocation['slurm_job_id'] == source_job, 'SOURCE_JOB_DIFFERS')
            cp = base/'checkpoints/search-g1-common8/actor'/f'global_step_{step}'
            complete = bundle.copy('identity/checkpoint-complete.json', cp/'complete.json')
            need(complete['run_id'] == run_id and complete['step'] == step and
                 complete['finite_all_saved_tensors'] is True, 'CHECKPOINT_NOT_COMPLETE')
            archive_root = base/'preparation/checkpoint-archives'/f'global_step_{step}'
            snap = bundle.copy('identity/source-snapshot.json', archive_root/'source-snapshot.json')
            archive = bundle.copy('identity/accepted-archive.json', archive_root/'complete.json')
            need(all(snap[k] == v for k, v in dict(base=str(base), run_id=run_id, source_commit=training_commit,
                 step=step, training_world_size=4).items()) and
                 snap['role'] == 'easy2560_full_checkpoint_source_snapshot', 'ARCHIVE_CP_BINDING_DIFFERS')
            need(archive['status'] == ARCHIVE_STATUS and archive['actual_cuda_restore_accepted'] is False and
                 archive['fresh_source_after_archive_sha_verified'] is True and
                 archive['fresh_destination_sha_verified'] is True and archive['destination_fsynced'] is True and
                 archive['file_count'] == len(snap['files']) and
                 archive['snapshot_sha256'] == bundle.files['identity/source-snapshot.json']['sha256'] and
                 all(archive[k] == snap[k] for k in ('run_id', 'source_commit', 'step', 'base', 'total_bytes')),
                 'ARCHIVE_ACCEPTANCE_BINDING_DIFFERS')
            need(snap['files']['checkpoint/complete.json']['sha256'] ==
                 bundle.files['identity/checkpoint-complete.json']['sha256'] and
                 snap['files']['metadata/run-manifest.json']['sha256'] ==
                 bundle.files['identity/run-manifest.json']['sha256'], 'ARCHIVED_METADATA_DIFFERS')
            state = bundle.copy('identity/training-state-manifest.json', cp/'training_state/manifest.json',
                                expected=complete['files_sha256']['training_state/manifest.json'])
            need(state['completed_outer'] == step and state['question_probe'] is None and
                 sorted(r['rank'] for r in state['worker_files']) == list(range(4)), 'NATIVE4_STATE_REQUIRED')
            from reproduction.prefix20_joint4.cpu_audit.safe_pt import Archive
            driver_raw, driver_record = read_small(cp/'training_state/driver.pt',
                          expected=complete['files_sha256']['training_state/driver.pt'])
            need(driver_record['sha256'] == snap['files']['checkpoint/training_state/driver.pt']['sha256'], 'ARCHIVED_DRIVER_DIFFERS')
            with Archive(driver_raw) as inert:
                driver = inert.value
                need(driver['schema_version'] == 1 and driver['completed_outer'] == step and driver['run_id'] == run_id and
                     driver['source_commit'] == training_commit and driver['question_probe'] is None and
                     driver['selection_mode'] == 'mixed_first' and driver['easy2560_selection'] is True,
                     'DRIVER_CP_BINDING_DIFFERS')
                need(driver['reward_refresh_pending'] is (step in (50, 100, 150)), 'SAVED_CALLBACK_FLAG_DIFFERS')
                stream = driver['candidate_stream']
                summary = {k: driver[k] for k in ('completed_outer', 'probe_version', 'probe_round',
                            'reward_refresh_pending', 'attempt_progress')}
                summary.update(driver_file=driver_record, stream={k: stream[k] for k in
                    ('candidate_cursor', 'epoch', 'consumed_source_rows', 'pending_offset', 'prefix20_pool_binding')},
                    candidate_order_count=len(stream['candidate_order']),
                    candidate_order_sha256=digest(canonical(stream['candidate_order'])),
                    candidate_stream_storage_signature=digest(canonical(inert.signature(stream))),
                    rng_storage_signature=digest(canonical(inert.signature(driver['rng']))),
                    tensor_or_model_load_performed=False, actual_restore_performed=False)
            progress = driver['attempt_progress']
            need(progress['completed_outer'] == step and complete['actual_optimizer_writes_per_rank'] ==
                 progress['cumulative_writes'] and all(r['run_id'] == run_id and
                 r['actual_optimizer_writes'] == progress['cumulative_writes'] for r in state['worker_files']),
                 'SAVED_COUNTERS_DIFFER')
            for rank in range(4):
                key = f'training_state/rank-{rank}.pt'
                need(complete['files_sha256'][key] == snap['files']['checkpoint/' + key]['sha256'], 'ARCHIVED_RANK_SHA_DIFFERS')
            bundle.put('saved-driver-summary.json', encoded(summary))
            resolved = bundle.copy('identity/resolved-training-config.json', base/'logs/resolved-training-config.json')
            need(resolved['run_id'] == run_id and digest(json.dumps(resolved['resolved_config'], sort_keys=True, separators=(',', ':')).encode()) ==
                 resolved['resolved_config_sha256'], 'RESOLVED_CONFIG_IDENTITY_DIFFERS')
            cfg = resolved['resolved_config']
            need(cfg['data']['selection_mode'] == 'mixed_first' and cfg['algorithm']['all_correct_virtual_zero'] is False
                 and cfg['actor_rollout_ref']['actor']['ppo_logical_micro_batch_size'] == 4, 'RESOLVED_EASY_MATH_DIFFERS')
            bundle.put('identity/source-data-identities.json', encoded(dict(
                source_sha256=run['source_sha256'], input_sha256=run['input_sha256'],
                source_and_large_data_bytes_in_existing_archive=True, large_file_SHA_recomputed_by_this_sidecar=False)))
            for name, expected in run['input_sha256'].items():
                need(snap['files']['data/' + name]['sha256'] == expected, 'ARCHIVED_DATA_IDENTITY_DIFFERS')
            for name, expected in run['source_sha256'].items():
                need(snap['files']['source/' + name]['sha256'] == expected, 'ARCHIVED_SOURCE_IDENTITY_DIFFERS')
            bundle.copy('identity/eligible-question-pool.json', Path(run['selection_eligible_pool_path']),
                        expected=stream['prefix20_pool_binding']['eligible_pool_sha256'], external=True)
            manifests = [r for name, r in snap['files'].items() if name.startswith('grounding/') and name.endswith('/version-manifest.json')]
            pinned = []
            for record in manifests:
                value, _, _ = json_small(record['path'], expected=record['sha256'])
                if value['version'] == driver['probe_version']: pinned.append((value, record))
            need(len(pinned) == 1, 'EXACT_SAVED_GROUNDING_VERSION_MISSING_OR_AMBIGUOUS')
            probe, record = pinned[0]
            need(probe['run_id'] == run_id and probe['round'] == driver['probe_round'], 'SAVED_GROUNDING_IDENTITY_DIFFERS')
            bundle.copy('grounding/version-manifest.json', record['path'], expected=record['sha256'])
            for name, key in (('commit_probe.pkl', 'commit'), ('boundary_probe.pkl', 'boundary')):
                bundle.copy('grounding/' + name, probe[key], expected=probe['head_sha256'][name], parse_json=False)
            bundle.copy('grounding/calibration-report.json', Path(probe['commit']).parent/'calibration-report.json',
                        expected=probe['calibration_report_sha256'])
            updates, attempts = _validate_ledgers(base, run, step, driver, bundle, source_job)
            need(snap['completed_update'] == updates[-1], 'ARCHIVE_UPDATE_AND_PREFIX_DIFFER')
            refs = _trace_references(base, run, updates, attempts, bundle)
            observed = _validation_observation(base, step, bundle)
            phase = dict(checkpoint_saved_after=('stage64' if step == 5 else 'fixed512'),
                fixed512_callback_pending_at_checkpoint=step == 5,
                grounding_refresh_pending_at_checkpoint=step in (50, 100, 150),
                final51713_pending_at_checkpoint=step == 200,
                post_callback_rng_captured=False, current_observation_is_not_checkpoint_RNG_state=True,
                **observed)
            bundle.put('checkpoint-callback-boundary.json', encoded(phase))
            skipped = base/'logs/formal-inline-development-initial-skipped.json'
            if skipped.exists(): bundle.copy('identity/initial-evaluation-skipped.json', skipped)
            manifest = dict(schema_version=1, role='easy2560_checkpoint_evidence_supplement',
                source_base=str(base), source_commit=training_commit, run_id=run_id, source_job=source_job, step=step,
                single_original_job_prefix=True, inherited_history_assembled=False,
                source_checkpoint=str(cp), actual_cuda_restore_accepted=False, actual_restore_performed=False,
                post_callback_rng_captured=False, new_training_updates=0, gpu_allocations=0,
                files=bundle.files, source_reads=bundle.source_reads,
                selected_large_trace_references=len(refs), copied_selected_trace_bytes=0,
                checkpoint_large_files_rehashed_by_this_operator=False,
                core_archive=dict(status=ARCHIVE_STATUS, training_world_size=4,
                    source_snapshot=dict(relative_path='identity/source-snapshot.json',
                        sha256=bundle.files['identity/source-snapshot.json']['sha256']),
                    source_acceptance=dict(relative_path='identity/accepted-archive.json',
                        sha256=bundle.files['identity/accepted-archive.json']['sha256']),
                    checkpoint_complete_sha256=bundle.files['identity/checkpoint-complete.json']['sha256'],
                    file_count=archive['file_count'], total_bytes=archive['total_bytes'],
                    actual_cuda_restore_accepted=False,
                    remote_location=dict(host='hkust3', directory='/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/checkpoints/'+run_id+'/global_step_'+str(step),
                        location_source='reviewed archive_controller.HK_ROOT layout', current_remote_read_performed=False)),
                operator_source_sha256=digest(Path(__file__).read_bytes()), checked_unix=time.time(),
                scope='sealed exact original-job prefixes and saved callback phase; byte preservation, not recovery acceptance')
            bundle.put('supplement.json', encoded(manifest))
            marker = dict(status=STATUS, source_commit=training_commit, run_id=run_id, source_job=source_job,
                          step=step, supplement_sha256=bundle.files['supplement.json']['sha256'],
                          actual_cuda_restore_accepted=False, post_callback_rng_captured=False)
            bundle.put('complete.json', encoded(marker))
            verify(stage)
            # Re-read every copied source prefix/file after sealing; appends are
            # allowed, replacement or changed prefix bytes are not.
            recheck_sources(bundle.source_reads)
            for directory, _, _ in os.walk(stage, topdown=False): fsync_directory(directory)
            stage.rename(target); fsync_directory(root)
            return verify(target)
        except Exception:
            # Remove only this newly created operator-owned unpublished staging.
            shutil.rmtree(stage)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('seal'); p.add_argument('--base', required=True)
    p.add_argument('--training-commit', required=True); p.add_argument('--run-id', required=True)
    p.add_argument('--source-job', required=True); p.add_argument('--step', type=int, required=True, choices=STEPS)
    p = sub.add_parser('verify'); p.add_argument('--directory', required=True)
    args = parser.parse_args()
    result = seal(args.base, args.training_commit, args.run_id, args.source_job, args.step) if args.mode == 'seal' else verify(args.directory)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
