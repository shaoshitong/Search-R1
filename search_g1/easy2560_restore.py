"""Native4 Easy saved-checkpoint recovery provenance, independent of qprobe pairs.

The publisher reads all checkpoint bytes and CPU Adam/scheduler states. It never
loads a model or claims a CUDA restart. CP5 RNG predates development: an existing
after5 result is historical evidence, not a fabricated post-development RNG.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

KIND = 'easy2560_native4'
STATUS = 'EASY2560_NATIVE4_FULL_SHA_CPU_STATES_PASS'
STEPS = (5, 50, 100, 150, 200)


def _need(condition, message):
    if not condition:
        raise ValueError('EASY2560_RESTORE_' + message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _publish(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    _need(not path.exists() and not path.is_symlink(), 'OUTPUT_ALREADY_EXISTS')
    temporary = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(_encoded(value)); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)  # Publish without replacing another writer.
        temporary.unlink()
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if temporary.exists(): temporary.unlink()


def _regular(path):
    path = Path(path)
    _need(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(),
          'CANONICAL_REGULAR_FILE_REQUIRED')
    return path


def _stat(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_uid)


def _record(path, *, relative_path=None, expected=None, prefix=None):
    path = _regular(path); before = path.stat()
    if prefix is None:
        sha = digest(path); count = before.st_size
    else:
        count = len(prefix); sha = hashlib.sha256(prefix).hexdigest()
        with path.open('rb') as stream: _need(stream.read(count) == prefix, 'PREFIX_CHANGED')
    after = path.stat(); _need(_stat(before) == _stat(after), 'FILE_CHANGED_DURING_SHA')
    _need(expected is None or sha == expected, 'ACTUAL_FILE_SHA_DIFFERS')
    return dict(path=str(path), relative_path=relative_path, full_read_sha256=sha, bytes=count,
                device=after.st_dev, inode=after.st_ino, mtime_ns=after.st_mtime_ns, uid=after.st_uid,
                prefix_only=prefix is not None, observed_source_bytes=after.st_size)


def _identity(record, *, rehash=False):
    path = _regular(record['path']); st = path.stat()
    # Receipts are published on the login node and consumed on GPU nodes.
    # st_dev identifies the client-local mount, not the shared file's content.
    # Keep it as a publisher observation; it is valid only in same-host
    # before/after checks (_stat), never as a cross-node admission condition.
    _need((st.st_ino, st.st_uid) == (record['inode'], record['uid']),
          'SOURCE_FILE_IDENTITY_CHANGED path=' + str(path) +
          ' expected_inode_uid=' + str((record['inode'], record['uid'])) +
          ' actual_inode_uid=' + str((st.st_ino, st.st_uid)))
    if record.get('prefix_only'):
        _need(st.st_size >= record['bytes'], 'SOURCE_PREFIX_TRUNCATED')
    else:
        _need((st.st_size, st.st_mtime_ns) == (record['bytes'], record['mtime_ns']), 'SOURCE_FILE_CHANGED')
    if rehash:
        with path.open('rb') as stream: raw = stream.read(record['bytes'])
        _need(len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['full_read_sha256'],
              'SOURCE_SMALL_SHA_CHANGED')
    return path


def _scope(run, *, physical_micros=(16,)):
    from reproduction.common8.precision_contract import easy2560_native4
    _need(easy2560_native4(run) and run.get('question_probe') is None and
          type(run.get('actor_micro_batch_per_rank')) is int and
          run['actor_micro_batch_per_rank'] in physical_micros and run.get('all_correct_virtual_zero') is False and
          run.get('selection_candidate_questions') == 2560 and run.get('selection_selected_questions') == 512 and
          run.get('selection_fixed_candidate_budget') is True, 'EXACT_NATIVE4_EASY_SCOPE_REQUIRED')


def physical_micro_transition(parent, target_micro):
    """Allow the owned OOM fallback while retaining the logical four-row loss.

    CP5 is the original physical16/native4 source. Its optimizer shard layout
    does not depend on forward micro size; only the number of accumulation
    passes changes. This does not claim bitwise continuation after regrouping.
    """
    source_micro = parent.get('actor_micro_batch_per_rank')
    _need(type(source_micro) is int and source_micro == 16 and
          type(target_micro) is int and target_micro in (16, 8, 4) and target_micro <= source_micro and
          parent.get('training_world_size') == 4 and
          parent.get('actor_logical_micro_batch_per_rank') == 4,
          'PHYSICAL_MICRO_FALLBACK_DIFFERS')
    return dict(source_physical_micro_per_rank=source_micro, target_physical_micro_per_rank=target_micro,
                source_accumulation_steps=32 // source_micro, target_accumulation_steps=32 // target_micro,
                global_mini_batch=128, training_world_size=4, logical_micro_per_rank=4,
                objective_reduction='original four-row micro masked_mean',
                bitwise_continuation_claimed=False)


def _compatible_execution_budget(run, parent):
    transition = physical_micro_transition(parent, run.get('actor_micro_batch_per_rank'))
    before, after = parent['budget'], run['budget']
    allowed = {'ppo_micro_batch_size', 'per_rank_actor_micro_batch'}
    _need(before.keys() == after.keys() and all(before[k] == after[k] for k in before if k not in allowed),
          'SCIENTIFIC_INPUT_OR_BATCH_DIFFERS_budget')
    if transition['source_physical_micro_per_rank'] != transition['target_physical_micro_per_rank']:
        for budget, micro in ((before, transition['source_physical_micro_per_rank']),
                              (after, transition['target_physical_micro_per_rank'])):
            _need(budget.get('ppo_micro_batch_size') == micro * 4 and
                  budget.get('per_rank_actor_micro_batch') == micro and
                  budget.get('ppo_mini_batch_size') == 128 and
                  budget.get('per_rank_actor_logical_micro_batch') == 4 and
                  budget.get('objective_reduction') == transition['objective_reduction'],
                  'PHYSICAL_MICRO_BUDGET_GEOMETRY_DIFFERS')
    else:
        _need(before == after, 'SCIENTIFIC_INPUT_OR_BATCH_DIFFERS_budget')
    return transition


def _load_cpu(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def _validate_rank(saved, rank, outer, writes):
    from search_g1.qprobe_checkpoint import validate_rank_state
    return validate_rank_state(saved, rank, 4, outer, expected_writes=writes)


def _prefix(path, relative, predicate):
    raw = _regular(path).read_bytes(); rows = []; parts = []
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'): break
        row = json.loads(line)
        if not predicate(row): break
        rows.append(row); parts.append(line)
    _need(rows, 'EMPTY_REQUIRED_PREFIX_' + relative)
    return rows, _record(path, relative_path=relative, prefix=b''.join(parts))


def _worker_prefix_row(row, cutoff):
    context = row.get('context') or row
    index = context.get('attempt_index')
    if index is None:
        _need(row.get('kind') == 'INITIAL_OPTIMIZER' and
              context.get('logical_update', 0) == 0 and row.get('optimizer_writes', 0) == 0,
              'WORKER_PREFIX_ATTEMPT_IDENTITY_MISSING')
        return True
    _need(type(index) is int and index >= 0, 'WORKER_PREFIX_ATTEMPT_INDEX_INVALID')
    return index <= cutoff


def _driver(saved, run, outer, ranks):
    _need(saved.get('schema_version') == 1 and saved.get('easy2560_selection') is True and
          saved.get('selection_mode') == 'mixed_first' and saved.get('question_probe') is None and
          saved['run_id'] == run['run_id'] and saved['source_commit'] == run['source_commit'] and
          saved['completed_outer'] == outer, 'DRIVER_SAVED_IDENTITY_DIFFERS')
    progress = saved['attempt_progress']; stream = saved['candidate_stream']
    _need(all(type(progress[k]) is int and progress[k] >= 0 for k in
        ('completed_outer', 'cumulative_writes', 'attempt_index', 'aborted_attempts', 'full_budget_outer')),
        'DRIVER_PROGRESS_INVALID')
    _need(progress['completed_outer'] == outer and progress['cumulative_writes'] == ranks[0]['actual_optimizer_writes']
          and outer <= progress['cumulative_writes'] <= outer * 20
          and progress['attempt_index'] == ranks[0]['last_attempt_index'] and
          progress['aborted_attempts'] == progress['attempt_index'] - outer and
          progress['full_budget_outer'] <= outer, 'DRIVER_WORKER_COUNTS_DIFFER')
    _need(saved['reward_refresh_pending'] is (outer in (50, 100, 150)), 'SAVED_REFIT_PHASE_DIFFERS')
    consumed = stream['consumed_source_rows']; order = stream['candidate_order']; binding = stream['prefix20_pool_binding']
    _need(binding['canonical_train_sha256'] == run['input_sha256']['train.parquet'],
          'SOURCE_POOL_CANONICAL_TRAIN_DIFFERS')
    _need(type(consumed) is int and consumed >= outer * 2560 and consumed % 2560 == 0,
          'ACTUAL_2560_SOURCE_CURSOR_INVALID')
    _need(len(order) == binding['eligible_questions'] and len(set(order)) == len(order) and
          all(type(i) is int for i in order) and set(order) == set(range(len(order))) and
          type(stream['candidate_cursor']) is int and 0 <= stream['candidate_cursor'] <= len(order) and
          type(stream['epoch']) is int and stream['epoch'] >= 0, 'SAVED_CANDIDATE_ORDER_INVALID')
    pending = stream['pending_source']; offset = stream['pending_offset']
    _need(type(offset) is int and offset >= 0 and (offset == 0 if pending is None else offset < len(pending)),
          'SAVED_PENDING_SOURCE_OFFSET_INVALID')
    _need(isinstance(saved.get('rng'), dict) and {'torch_cpu', 'numpy', 'python'} <= set(saved['rng']),
          'ACTUAL_DRIVER_RNG_MISSING')
    return dict(completed_outer=outer, source_consumed=consumed, candidate_cursor=stream['candidate_cursor'],
                candidate_order_count=len(order), candidate_order_sha256=hashlib.sha256(_encoded(order)).hexdigest(),
                epoch=stream['epoch'], pending_offset=offset, pending_source_present=pending is not None,
                prefix20_pool_binding=copy.deepcopy(binding), probe_version=saved['probe_version'],
                probe_round=saved['probe_round'], reward_refresh_pending=saved['reward_refresh_pending'],
                actual_driver_rng_saved=True, question_probe=None)


def create_receipt(base, checkpoint, output):
    base = Path(base).resolve(); checkpoint = Path(checkpoint).resolve(); output = Path(output).resolve()
    _need(checkpoint.is_relative_to(base) and not output.is_relative_to(checkpoint), 'TASK_PATH_SCOPE_DIFFERS')
    _need(not output.exists(), 'OUTPUT_ALREADY_EXISTS')
    run = _json(base/'logs/run-manifest.json'); _scope(run)
    _need(not run.get('full_state_restore'), 'ORIGINAL_JOB_PREFIX_REQUIRED_NO_IMPLICIT_CHAIN')
    complete = _json(checkpoint/'complete.json'); outer = complete['step']
    _need(type(outer) is int and outer in STEPS and complete['run_id'] == run['run_id'] and
          complete['finite_all_saved_tensors'] is True and Path(complete['checkpoint']) == checkpoint,
          'COMPLETE_CHECKPOINT_REQUIRED')
    files = [_record(p, relative_path=str(p.relative_to(checkpoint))) for p in sorted(checkpoint.rglob('*')) if p.is_file()]
    by_name = {item['relative_path']: item for item in files}
    for name, sha in complete['files_sha256'].items():
        _need(name in by_name and by_name[name]['full_read_sha256'] == sha, 'COMPLETE_SHA_DIFFERS')
    index = _json(checkpoint/'model.safetensors.index.json')
    _need(len(index['weight_map']) == complete['validated_tensor_count'] and
          all(name in by_name for name in set(index['weight_map'].values())), 'HF_SHARD_INDEX_DIFFERS')
    state = _json(checkpoint/'training_state/manifest.json')
    _need(state['completed_outer'] == outer and state.get('question_probe') is None and
          sorted(item['rank'] for item in state['worker_files']) == list(range(4)), 'FOUR_SAVED_RANKS_REQUIRED')
    writes = complete['actual_optimizer_writes_per_rank']; ranks = []
    for item in sorted(state['worker_files'], key=lambda item: item['rank']):
        rank = item['rank']; path = checkpoint/'training_state'/('rank-' + str(rank) + '.pt')
        _need(Path(item['path']) == path and by_name[str(path.relative_to(checkpoint))]['full_read_sha256'] == item['sha256'],
              'SAVED_RANK_FILE_DIFFERS')
        saved = _load_cpu(path)
        _need(saved['run_id'] == run['run_id'] and saved['source_commit'] == run['source_commit'] and
              isinstance(saved.get('rng'), dict), 'SAVED_RANK_OR_RNG_IDENTITY_DIFFERS')
        ranks.append(_validate_rank(saved, rank, outer, writes)); del saved
    signature = lambda row: (row['actual_optimizer_writes'], row['last_attempt_index'],
                             tuple(row['used_attempt_ids']), row['policy_rejections'])
    _need(len({signature(row) for row in ranks}) == 1, 'SAVED_RANK_COUNTERS_DIVERGE')
    driver_file = by_name['training_state/driver.pt']; driver = _load_cpu(driver_file['path'])
    _need(state['driver_file']['sha256'] == driver_file['full_read_sha256'], 'MANIFEST_DRIVER_SHA_DIFFERS')
    summary = _driver(driver, run, outer, ranks); progress = copy.deepcopy(driver['attempt_progress'])
    pool = _record(Path(run['selection_eligible_pool_path']).resolve(), relative_path='external/eligible-question-pool.json',
                   expected=summary['prefix20_pool_binding']['eligible_pool_sha256'])
    matches = []
    for path in (base/'artifacts').glob('round_*/version-manifest.json'):
        value = _json(path)
        if value.get('version') == driver['probe_version'] and value.get('run_id') == run['run_id']:
            matches.append((path, value))
    _need(len(matches) == 1, 'EXACT_SAVED_GROUNDING_VERSION_REQUIRED')
    probe_path, probe = matches[0]
    _need(probe['round'] == driver['probe_round'] == min((outer - 1)//50, 3)*50,
          'SAVED_GROUNDING_ROUND_DIFFERS')
    reward_files = []
    for key, name in (('commit', 'commit_probe.pkl'), ('boundary', 'boundary_probe.pkl')):
        reward_files.append(_record(Path(probe[key]).resolve(), relative_path='heads/'+name,
                                    expected=probe['head_sha256'][name]))
    report = Path(probe['commit']).parent/'calibration-report.json'
    if probe.get('calibration_report_sha256'):
        reward_files.append(_record(report, relative_path='heads/calibration-report.json',
                                    expected=probe['calibration_report_sha256']))
    encoder_files = [_record((Path(probe['encoder'])/name).resolve(), relative_path=name, expected=sha)
                     for name, sha in sorted(probe['encoder_sha256'].items())]
    allocation = _json(base/'logs/native-allocation-admission.json')
    _need(allocation['accepted'] is True and allocation['run_id'] == run['run_id'] and
          allocation['training_world_size'] == 4, 'SOURCE_ALLOCATION_DIFFERS')
    historical = {}
    def pin(relative, *, expected=None):
        record = _record(base/relative, relative_path=relative, expected=expected)
        historical[relative] = record; return record
    source_run = pin('logs/run-manifest.json')
    pin('logs/native-allocation-admission.json')
    updates, record = _prefix(base/'logs/actual-updates.jsonl', 'logs/actual-updates.jsonl', lambda row: row['step'] <= outer)
    historical[record['relative_path']] = record
    _need([row['step'] for row in updates] == list(range(1, outer+1)) and
          all(row['run_id'] == run['run_id'] for row in updates) and
          updates[-1]['optimizer_writes_after'] == writes and
          updates[-1]['attempt_index'] == progress['attempt_index'], 'ACTUAL_UPDATE_PREFIX_DIFFERS')
    attempts, record = _prefix(base/'logs/actual-attempts.jsonl', 'logs/actual-attempts.jsonl',
                              lambda row: row['attempt_index'] <= progress['attempt_index'])
    historical[record['relative_path']] = record
    _need([row['attempt_index'] for row in attempts] == list(range(1, progress['attempt_index']+1)) and
          all(row['run_id'] == run['run_id'] and row['slurm_job_id'] == allocation['slurm_job_id'] for row in attempts) and
          sorted(row['attempt_id'] for row in attempts) == sorted(ranks[0]['used_attempt_ids']), 'ACTUAL_ATTEMPT_PREFIX_DIFFERS')
    for role in ('optimizer', 'scheduler'):
        for rank in range(4):
            relative = 'logs/' + role + '-rank-' + str(rank) + '.jsonl'
            rows, record = _prefix(base/relative, relative,
                lambda row: _worker_prefix_row(row, progress['attempt_index']))
            _need(all(row['run_id'] == run['run_id'] and row['rank'] == rank for row in rows), 'ACTUAL_RANK_PREFIX_DIFFERS')
            historical[relative] = record
    for row in updates:
        aid = row['attempt_id']
        pin('logs/behavior-probability-attempt-' + aid + '.json', expected=row['behavior_probability_gate_sha256'])
        pin('artifacts/mixed-selection/' + aid + '.json')
    for relative in ('logs/validation-identities.jsonl',):
        if (base/relative).is_file():
            _, record = _prefix(base/relative, relative, lambda row: row['step'] <= outer)
            historical[relative] = record
    for relative in ('logs/formal-inline-development-after.json', 'logs/formal-inline-development-initial-skipped.json',
                     'logs/latest-stage-identity.json', 'logs/resolved-training-config.json',
                     'logs/policy-safety-calibration.json', 'preparation/run-source.json',
                     'source/'+run['source_commit']+'/source-publication.json'):
        if (base/relative).is_file(): pin(relative)
    after = historical.get('logs/formal-inline-development-after.json')
    _need(after is not None, 'REAL_HISTORICAL_QA5_REQUIRED_NO_INVENTED_BASELINE')
    qa = _json(after['path'])
    _need(qa['run_id'] == run['run_id'] and qa['step'] == 5, 'HISTORICAL_QA5_IDENTITY_DIFFERS')
    for name in ('source-snapshot.json', 'complete.json'):
        relative = 'preparation/checkpoint-archives/global_step_' + str(outer) + '/' + name
        if (base/relative).is_file(): pin(relative)
    historical['checkpoint/complete.json'] = dict(by_name['complete.json'], relative_path='checkpoint/complete.json')
    historical['checkpoint/training_state/manifest.json'] = dict(by_name['training_state/manifest.json'],
                                                               relative_path='checkpoint/training_state/manifest.json')
    receipt = dict(schema_version=1, kind=KIND, status=STATUS, step=outer, world_size=4,
        run_id=run['run_id'], source_commit=run['source_commit'], source_job=allocation['slurm_job_id'],
        checkpoint=str(checkpoint), source_base=str(base), files=files, ranks=ranks,
        hf_saved_tensor_count=complete['validated_tensor_count'], actual_optimizer_writes_per_rank=writes,
        driver_file=driver_file, boundary_driver=driver_file, driver_state=summary, attempt_progress=progress,
        source_run_manifest=source_run, eligible_pool=pool,
        reward_probe_manifest=_record(probe_path, relative_path='grounding/version-manifest.json'),
        reward_probe_version=probe['version'], reward_probe_round=probe['round'], reward_files=reward_files,
        reward_encoder_files=encoder_files, question_probe=None,
        historical_files=historical,
        historical_artifacts=dict(checkpoints=[dict(step=outer, **by_name['complete.json'])],
            development=dict(after5=after), prefix_files=list(historical.values()),
            reward_archives=[dict(version=probe['version'], round=probe['round'], path=str(probe_path),
                                 full_read_sha256=digest(probe_path))]),
        checkpoint_saved_before_development=outer == 5, post_callback_rng_captured=False,
        sampling_rng_stage='saved_checkpoint_before_development' if outer == 5 else 'saved_checkpoint_before_pending_callback',
        reward_refresh_pending=driver['reward_refresh_pending'], final_evaluation_pending=outer == 200,
        historical_QA5_is_post_save_result_only=True, completed_QA5_replay_required=False,
        source_archive_receipt_available=('preparation/checkpoint-archives/global_step_' + str(outer) + '/complete.json') in historical,
        same_job_continuation=False, actual_cuda_restore_accepted=False, model_tensor_reload_performed=False,
        checked_unix=time.time(), scope='full saved native4 CP bytes/CPU Adam scheduler and genuine historical prefix; CUDA/dataequal restore pending')
    # No source record may have changed while the long CPU/SHA pass ran.
    for record in files + reward_files + encoder_files + [source_run, pool, receipt['reward_probe_manifest']]:
        _identity(record)
    _publish(output, receipt)
    return dict(path=str(output), sha256=digest(output), status=STATUS, step=outer, world_size=4,
                run_id=run['run_id'], actual_cuda_restore_accepted=False)


def source(receipt_path):
    path = _regular(Path(receipt_path).resolve()); receipt = _json(path)
    _need(receipt.get('schema_version') == 1 and receipt.get('kind') == KIND and receipt.get('status') == STATUS,
          'RECEIPT_SCHEMA_DIFFERS')
    checkpoint = Path(receipt['checkpoint']); base = Path(receipt['source_base'])
    _need(receipt['world_size'] == 4 and receipt['step'] in STEPS and checkpoint.resolve() == checkpoint and
          checkpoint.is_relative_to(base) and receipt['actual_cuda_restore_accepted'] is False,
          'RECEIPT_SCOPE_DIFFERS')
    _need({str(p.relative_to(checkpoint)) for p in checkpoint.rglob('*') if p.is_file()} ==
          {record['relative_path'] for record in receipt['files']}, 'CHECKPOINT_FILE_SET_CHANGED')
    for record in receipt['files']:
        _need(Path(record['path']) == checkpoint/record['relative_path'], 'CHECKPOINT_MEMBER_PATH_DIFFERS')
        _identity(record, rehash=record['relative_path'] in ('complete.json', 'training_state/manifest.json'))
    for record in receipt['reward_files'] + receipt['reward_encoder_files'] + [receipt['eligible_pool']]:
        _identity(record)
    for record in receipt['historical_artifacts']['prefix_files']:
        _identity(record)  # Full prefix SHA was computed by publisher; no repeated large log scans.
    for record in (receipt['source_run_manifest'], receipt['reward_probe_manifest'],
                   receipt['historical_artifacts']['development']['after5']):
        _identity(record, rehash=True)
    parent = _json(receipt['source_run_manifest']['path']); _scope(parent)
    _need(Path(receipt['source_run_manifest']['path']) == base/'logs/run-manifest.json' and
          Path(receipt['historical_artifacts']['development']['after5']['path']) == base/'logs/formal-inline-development-after.json' and
          parent['run_id'] == receipt['run_id'] and parent['source_commit'] == receipt['source_commit'],
          'SOURCE_RUN_BINDING_DIFFERS')
    complete = _json(checkpoint/'complete.json'); state = _json(checkpoint/'training_state/manifest.json')
    progress = receipt['attempt_progress']; ranks = receipt['ranks']; outer = receipt['step']
    by_name = {r['relative_path']: r for r in receipt['files']}
    _need(complete['step'] == state['completed_outer'] == progress['completed_outer'] == outer and
          complete['run_id'] == receipt['run_id'] and complete['finite_all_saved_tensors'] is True and
          complete['actual_optimizer_writes_per_rank'] == progress['cumulative_writes'] ==
          receipt['actual_optimizer_writes_per_rank'] and
          receipt['hf_saved_tensor_count'] == complete['validated_tensor_count'], 'SOURCE_CP_COUNTER_BINDING_DIFFERS')
    _need([r['rank'] for r in ranks] == list(range(4)) and
          sorted(r['rank'] for r in state['worker_files']) == list(range(4)), 'SOURCE_CP_NATIVE4_RANKS_DIFFER')
    _need(set(complete['files_sha256']) <= set(by_name) and
          all(complete['files_sha256'][name] == record['full_read_sha256']
              for name, record in by_name.items() if name in complete['files_sha256']),
          'SOURCE_CP_COMPLETE_SHA_MAP_DIFFERS')
    for rank, row in enumerate(ranks):
        native = next(r for r in state['worker_files'] if r['rank'] == rank)
        key = 'training_state/rank-' + str(rank) + '.pt'
        _need(native['path'] == by_name[key]['path'] and native['sha256'] == by_name[key]['full_read_sha256'] and
              native['actual_optimizer_writes'] == row['actual_optimizer_writes'] == progress['cumulative_writes'] and
              native['run_id'] == receipt['run_id'] and row['completed_outer'] == outer and
              row['scheduler_last_epoch'] == outer and row['scheduler_step_count'] == outer+1 and
              row['optimizer_tensors_finite'] is True and row['last_attempt_index'] == progress['attempt_index'] and
              len(set(row['used_attempt_ids'])) == len(row['used_attempt_ids']) == progress['attempt_index'] and
              row['used_attempt_ids'] == ranks[0]['used_attempt_ids'] and
              row['policy_rejections'] == ranks[0]['policy_rejections'], 'SOURCE_RANK_COUNTER_BINDING_DIFFERS')
    _need(progress['aborted_attempts'] == progress['attempt_index']-outer and
          state['driver_file']['sha256'] == receipt['driver_file']['full_read_sha256'] ==
          by_name['training_state/driver.pt']['full_read_sha256'] and
          receipt['driver_file']['path'] == by_name['training_state/driver.pt']['path'] and
          receipt['driver_state']['completed_outer'] == outer and
          receipt['driver_state']['probe_version'] == receipt['reward_probe_version'] and
          receipt['driver_state']['probe_round'] == receipt['reward_probe_round'], 'SOURCE_DRIVER_COUNTER_BINDING_DIFFERS')
    _need(receipt['checkpoint_saved_before_development'] is (outer == 5) and
          receipt['post_callback_rng_captured'] is False and
          receipt['driver_state']['reward_refresh_pending'] is receipt['reward_refresh_pending'] is (outer in (50,100,150)) and
          receipt['completed_QA5_replay_required'] is False, 'SOURCE_SAVED_CALLBACK_PHASE_DIFFERS')
    return path, receipt, checkpoint, base, parent


def bind(run, receipt_path):
    path, receipt, checkpoint, base, parent = source(receipt_path); _scope(run, physical_micros=(16, 8, 4))
    _need(receipt['step'] == 5, 'ONLY_CP5_TRAINING_CONTINUATION_IMPLEMENTED')
    _need(not run.get('full_state_restore') and run.get('completed_updates', 0) == 0,
          'TARGET_MUST_BE_PRISTINE_UNFINISHED')
    transition = _compatible_execution_budget(run, parent)
    for key in ('input_sha256', 'initial_model_sha256', 'model_revision',
                'training_world_size', 'actor_logical_micro_batch_per_rank',
                'all_correct_virtual_zero', 'selection_candidate_questions', 'selection_selected_questions',
                'selection_fixed_candidate_budget'):
        _need(run.get(key) == parent.get(key), 'SCIENTIFIC_INPUT_OR_BATCH_DIFFERS_' + key)
    for key in ('retriever_version', 'retriever_manifest_sha256', 'retriever_topk'):
        if key in parent:
            _need(run.get(key, parent[key]) == parent[key], 'RETRIEVER_IDENTITY_DIFFERS_' + key)
            run[key] = parent[key]
    if 'infra_rollout' in parent:
        _need(run.get('infra_rollout', parent['infra_rollout']) == parent['infra_rollout'], 'ROLLOUT_CONTRACT_DIFFERS')
        run['infra_rollout'] = copy.deepcopy(parent['infra_rollout'])
    progress = receipt['attempt_progress']
    run.update(run_id=parent['run_id'], native_ray_execution_id=uuid.uuid4().hex,
        initial_probe_version=receipt['reward_probe_version'],
        pretraining_baseline_policy='reuse_historical',
        full_state_restore=dict(kind=KIND, source_receipt=str(path), source_receipt_sha256=digest(path),
            physical_micro_transition=transition,
            source_checkpoint=str(checkpoint), source_base=str(base), source_job=receipt['source_job'],
            source_commit=receipt['source_commit'], source_run_manifest_sha256=receipt['source_run_manifest']['full_read_sha256'],
            completed_outer=receipt['step'], actual_optimizer_writes_per_rank=progress['cumulative_writes'],
            attempt_progress=copy.deepcopy(progress), attempt_index=progress['attempt_index'],
            aborted_attempts=progress['aborted_attempts'], candidate_source_cursor=receipt['driver_state']['source_consumed'],
            reward_probe_round=receipt['reward_probe_round'], reward_probe_version=receipt['reward_probe_version'],
            checkpoint_saved_before_development=receipt['checkpoint_saved_before_development'],
            post_callback_rng_captured=False, same_job_continuation=False,
            followup_acceptance_outer=min(receipt['step'] + 5, 200)),
        optimizer_semantics='actual saved native4 HF/Adam/scheduler/RNG/data stream; cross-job prefix kept as history; CUDA readback pending')
    return run


def install_restored_reward_probe(base, run):
    path, receipt, _, _, _ = source(run['full_state_restore']['source_receipt'])
    _need(digest(path) == run['full_state_restore']['source_receipt_sha256'], 'BOUND_RECEIPT_SHA_DIFFERS')
    base = Path(base).resolve(); live = base/'artifacts/live/manifest.json'
    _need(not live.exists(), 'TARGET_REWARD_PROBE_ALREADY_EXISTS')
    old = _json(receipt['reward_probe_manifest']['path'])
    directory = base/'artifacts'/('round_' + str(old['round']) + '-restored-' + uuid.uuid4().hex)
    heads = directory/'heads'; heads.mkdir(parents=True)
    for record in receipt['reward_files']:
        _identity(record, rehash=True); destination = directory/record['relative_path']
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(record['path'], destination)
        _need(digest(destination) == record['full_read_sha256'], 'RESTORED_HEAD_COPY_SHA_DIFFERS')
    manifest = dict(old, commit=str(heads/'commit_probe.pkl'), boundary=str(heads/'boundary_probe.pkl'))
    _publish(directory/'version-manifest.json', manifest); _publish(live, manifest)
    return dict(version=manifest['version'], round=manifest['round'], archive=str(directory/'version-manifest.json'),
                source_receipt_sha256=digest(path), actual_rpc_adoption_pending=True,
                new_calibration_generated=False, encoder_weights_copied=False)


def restore_driver(trainer, path):
    path, receipt, checkpoint, _, _ = source(path)
    base = Path(os.environ['SEARCH_G1_BASE']).resolve(); run = _json(base/'logs/run-manifest.json')
    _need(run['full_state_restore']['kind'] == KIND and
          run['full_state_restore']['source_receipt_sha256'] == digest(path), 'BOUND_RECEIPT_SHA_DIFFERS')
    _need(digest(receipt['driver_file']['path']) == receipt['driver_file']['full_read_sha256'], 'SAVED_DRIVER_SHA_DIFFERS')
    saved = _load_cpu(receipt['driver_file']['path'])
    from search_g1.training_state import restore_easy2560_driver
    result = restore_easy2560_driver(trainer, saved, run, source_run_id=receipt['run_id'],
                                   live_probe=_json(base/'artifacts/live/manifest.json'))
    result.update(schema_version=1, role='actual_easy2560_native4_restored_driver', run_id=run['run_id'],
                  kind=KIND, source_checkpoint=str(checkpoint), source_step=receipt['step'],
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  source_job=receipt['source_job'], source_commit=receipt['source_commit'],
                  source_receipt_sha256=digest(path), source_driver_full_sha256=receipt['driver_file']['full_read_sha256'],
                  same_job_continuation=False, checkpoint_saved_before_development=receipt['checkpoint_saved_before_development'],
                  post_callback_rng_captured=False, completed_QA5_replay_required=False,
                  actual_driver_dataequal=bool(result['actual_driver_rng_readback_equal'] and
                                               result['actual_candidate_stream_readback_equal']),
                  actual_cuda_restore_accepted=False, checked_unix=time.time())
    _publish(base/'logs/full-state-restored-driver.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True); parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(create_receipt(args.base, args.checkpoint, args.output), sort_keys=True))


if __name__ == '__main__': main()
