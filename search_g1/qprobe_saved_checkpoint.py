"""CPU qualification of the saved CP5 boundary, separate from a completed pair.

The actor, Adam, scheduler, worker RNG, driver RNG, candidate stream and question
head are restored from their actual checkpoint files.  The later dev512 result
is retained as historical evidence; its post-evaluation RNG was not persisted.
This module never evaluates, starts a worker, rewrites a source artifact, or
claims CUDA/FSDP acceptance.  The expensive qualifier is an explicit CLI only.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch

from search_g1.qprobe_checkpoint import (
    SAVED_CHECKPOINT_KIND, SAVED_CHECKPOINT_STATUS, _file_record,
    _historical_artifacts, _identity_record, _publish, _read_json,
    _validate_qstate, digest, validate_rank_state,
)


BOUNDARY_ORIGIN = 'saved_checkpoint_before_postcheckpoint_development'
BOUNDARY_SCOPE = ('actual saved CP5 actor/head/Adam/scheduler/worker RNG/driver RNG/source stream; '
                  'existing later dev512 is reused; post-evaluation RNG was not persisted')
QUALIFICATION_ROLE = 'qprobe_saved_checkpoint_cpu_qualification'
HF_SIDECARS = frozenset(('generation_config.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt', 'tokenizer.json'))
HF_SHARDS = frozenset(f'model-{index:05d}-of-00003.safetensors' for index in range(1, 4))
STATE_FILES = frozenset(('training_state/manifest.json', 'training_state/driver.pt',
    'training_state/question-probe-state.pt', 'training_state/question-probe-state.pt.head.pt')) | \
    frozenset(f'training_state/rank-{rank}.pt' for rank in range(4))
DECLARED_FILES = STATE_FILES | HF_SHARDS | {'config.json', 'model.safetensors.index.json'}
INVENTORY_FILES = DECLARED_FILES | HF_SIDECARS | {'complete.json'}
_SOURCE_CACHE = {}


def _require(condition, message):
    if not condition:
        raise ValueError('QPROBE_SAVED_CHECKPOINT_' + message)


def _remote(path):
    original = Path(path)
    resolved = original.resolve()
    _require(original.is_absolute() and original == resolved and
             str(resolved).startswith(('/data/', '/mnt/', '/backup/')),
             'REQUIRES_CANONICAL_REMOTE_STORAGE')
    return resolved


def _owned(path, base=None):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink() and path == path.resolve() and
             path.stat().st_uid == os.getuid() and (base is None or path.is_relative_to(base)),
             'SOURCE_FILE_NOT_OWNED_CANONICAL_REGULAR')
    return path


def _rows(path):
    raw = Path(path).read_bytes()
    _require(raw and raw.endswith(b'\n'), 'INCOMPLETE_OR_EMPTY_LEDGER')
    return [json.loads(line) for line in raw.splitlines()]


def _rng_bytes(value, name):
    _require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
             value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0,
             'INVALID_RNG_' + name)


def _validate_rng(rng, worker=False):
    _require(isinstance(rng, dict), 'RNG_MISSING')
    _rng_bytes(rng['torch_cpu'], 'TORCH_CPU')
    torch.Generator(device='cpu').set_state(rng['torch_cpu'])
    np.random.RandomState().set_state(rng['numpy'])
    random.Random().setstate(rng['python'])
    if worker:
        for key in ('torch_cuda', 'actor_cuda_saved', 'rollout_cuda_saved'):
            _rng_bytes(rng[key], key.upper())
        _require(type(rng['cuda_device']) is int and rng['cuda_device'] >= 0 and
                 rng['vllm_version'] == '0.6.3' and
                 rng['boundary'] == 'idle synchronous generation engine' and
                 isinstance(rng['request_generators'], dict), 'INVALID_IDLE_WORKER_RNG')
        for value in rng['request_generators'].values():
            _require(isinstance(value['device'], str) and value['device'].startswith('cuda'),
                     'REQUEST_GENERATOR_DEVICE_CHANGED')
            _rng_bytes(value['state'], 'REQUEST_GENERATOR')
    else:
        _require(set(rng) == {'torch_cpu', 'numpy', 'python'}, 'DRIVER_RNG_SCHEMA_CHANGED')


def _validate_stream(stream):
    from search_g1.training_state import compact_stream_state
    compact_stream_state(stream)
    order = stream['candidate_order']
    _require(isinstance(order, list) and len(order) > 0 and
             all(type(value) is int for value in order) and sorted(order) == list(range(len(order))),
             'CANDIDATE_ORDER_NOT_FULL_SOURCE_PERMUTATION')
    _require(type(stream['candidate_cursor']) is int and 0 <= stream['candidate_cursor'] <= len(order) and
             type(stream['epoch']) is int and stream['epoch'] >= 0 and
             type(stream['consumed_source_rows']) is int and stream['consumed_source_rows'] >= 5 * 5120,
             'CANDIDATE_CURSOR_OR_CONSUMED_INVALID')
    if stream['pending_source'] is not None:
        _require(type(stream['pending_offset']) is int and
                 0 <= stream['pending_offset'] <= len(stream['pending_source']), 'PENDING_SOURCE_OFFSET_INVALID')


def _actual_prefix(base, run):
    """Check actual five rows/returns without replacing the separate safety audit."""
    attempts = _rows(base / 'logs/actual-attempts.jsonl')
    updates = _rows(base / 'logs/actual-updates.jsonl')
    _require([row['attempt_index'] for row in attempts] == list(range(1, 6)) and
             [row['step'] for row in updates] == list(range(1, 6)) and
             len({row['attempt_id'] for row in attempts}) == 5, 'ACTUAL_FIVE_PREFIX_INCOMPLETE')
    for step, (attempt, update) in enumerate(zip(attempts, updates), 1):
        shared = dict(run_id=run['run_id'], attempt_index=step,
                      optimizer_writes_before=(step - 1) * 20, optimizer_writes_after=step * 20,
                      opportunities_processed=20)
        _require(all(attempt.get(key) == value and update.get(key) == value for key, value in shared.items()) and
                 attempt['logical_update'] == attempt['completed_outer'] == step and
                 attempt['complete_outer'] is True and attempt['status'] == 'complete' and
                 attempt['all_ranks_agreed'] is True and attempt['real_rank_count'] == 4 and
                 attempt['optimizer_writes_this_attempt'] == update['actual_optimizer_writes'] == 20 and
                 attempt['scheduler_advances'] == 1 and attempt['cumulative_aborted_attempts'] == 0 and
                 attempt['opportunity_statuses'] == ['written'] * 20 and
                 update['attempt_id'] == attempt['attempt_id'] and
                 update['probe_version'] == run['initial_probe_version'] and
                 all(type(value) in (int, float) and math.isfinite(value) for value in update['metrics'].values()),
                 'ACTUAL_FIVE_COUNTERS_IDENTITY_OR_METRICS_CHANGED')
        if run.get('execution_backend') == 'ssh_local':
            _require(attempt['execution_id'] == run['execution_id'], 'ACTUAL_ATTEMPT_EXECUTION_CHANGED')
    return dict(attempt_ids=sorted(row['attempt_id'] for row in attempts), completed_outer=5,
                cumulative_writes=100, attempt_index=5, aborted_attempts=0, full_budget_outer=5)


def _check_proof(proof, records, relative):
    record = records[relative]
    _require(Path(proof['path']) == Path(record['path']) and proof['bytes'] == record['bytes'] and
             proof['sha256'] == record['full_read_sha256'], 'PRODUCER_PROOF_MISMATCH_' + relative)
    return record


def _serialization_value(value):
    """Canonicalize tokenizer AddedToken values without dropping their properties."""
    if isinstance(value, dict):
        return {str(key): _serialization_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialization_value(item) for item in value]
    if type(value).__name__ == 'AddedToken':
        return _serialization_value(value.__getstate__())
    return value


def _official_serialization_identity(checkpoint, initial):
    """Require the exact official tokenizer/generation semantics after HF saving."""
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    official = AutoTokenizer.from_pretrained(str(initial), local_files_only=True, trust_remote_code=False)
    saved = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    _require(type(official) is type(saved) and official.is_fast and saved.is_fast,
             'TOKENIZER_IMPLEMENTATION_CHANGED')
    _require(json.loads(official.backend_tokenizer.to_str()) == json.loads(saved.backend_tokenizer.to_str()) and
             official.get_vocab() == saved.get_vocab() and official.get_added_vocab() == saved.get_added_vocab(),
             'TOKENIZER_BACKEND_OR_VOCAB_CHANGED')
    for name in ('special_tokens_map_extended', 'all_special_ids', 'chat_template', 'model_max_length',
                 'padding_side', 'truncation_side', 'clean_up_tokenization_spaces', 'split_special_tokens'):
        _require(_serialization_value(getattr(official, name)) == _serialization_value(getattr(saved, name)),
                 'TOKENIZER_SEMANTIC_CHANGED_' + name)
    transport = {'name_or_path', 'tokenizer_file', 'vocab_file', 'merges_file', '_commit_hash'}
    kwargs = lambda tokenizer: _serialization_value({key: value for key, value in tokenizer.init_kwargs.items()
                                                     if key not in transport})
    _require(kwargs(official) == kwargs(saved), 'TOKENIZER_INITIALIZATION_SEMANTICS_CHANGED')
    # These are exactly the original worker's tokenizer-derived config overrides.
    expected = AutoConfig.from_pretrained(str(initial), local_files_only=True, trust_remote_code=False).to_dict()
    actual = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False).to_dict()
    _require(actual['torch_dtype'] == 'float32', 'SAVED_MODEL_CONFIG_NOT_FP32_MASTER')
    for key in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
        expected[key] = getattr(official, key)
    expected['torch_dtype'] = actual['torch_dtype'] = 'float32'
    metadata = {'_name_or_path', '_commit_hash', 'transformers_version'}
    strip = lambda values: {key: value for key, value in values.items() if key not in metadata}
    _require(strip(expected) == strip(actual), 'OFFICIAL_MODEL_CONFIG_SEMANTICS_CHANGED')
    expected_generation = GenerationConfig.from_pretrained(str(initial), local_files_only=True).to_dict()
    actual_generation = GenerationConfig.from_pretrained(str(checkpoint), local_files_only=True).to_dict()
    _require(strip(expected_generation) == strip(actual_generation), 'GENERATION_CONFIG_SEMANTICS_CHANGED')
    return dict(tokenizer_backend_and_vocabulary_equal=True, tokenizer_special_tokens_and_init_equal=True,
        generation_config_equal=True, model_config_equal_after_original_worker_overrides=True,
        model_master_dtype='float32', ignored_serialization_metadata=sorted(metadata),
        model_token_id_overrides={key: getattr(official, key) for key in ('bos_token_id', 'eos_token_id', 'pad_token_id')})


def _first_five_safety_audit(base, run):
    """Read original durable evidence through the corrected source-bound schema."""
    from reproduction.common8.behavior_runtime import read_gate_receipt, verify_write_receipt_links
    from reproduction.common8.runtime import verify_scheduler_receipts
    from reproduction.common8.formal_inline_safety import verify_first_five
    updates = _rows(base / 'logs/actual-updates.jsonl')
    gates = {str(row['step']): read_gate_receipt(base, run, row['step'], update_record=row) for row in updates}
    verify_write_receipt_links(base, gates, 1, 5)
    schedulers = verify_scheduler_receipts(base, run, 1, 5, gates)
    audit = verify_first_five(base, run, gates, schedulers)
    _require(audit['structural_accepted'] is True and audit['actual_complete_outers'] == 5 and
             audit['actual_full_budget_outers'] == 5 and audit['actual_optimizer_writes_per_rank'] == 100 and
             audit['completed_outer_optimizer_opportunities_per_rank'] == 100 and
             audit['guard_triggered_monitor_count'] == 0, 'ACTUAL_FIRST_FIVE_SAFETY_AUDIT_REJECTED')
    return audit


def _evidence_paths(base, run):
    from reproduction.common8.behavior_runtime import gate_path
    paths = [base / 'logs' / name for name in (
        'actual-attempts.jsonl', 'actual-updates.jsonl', 'latest-stage-identity.json',
        'run-manifest.json', 'policy-safety-calibration.json', 'resolved-training-config.json',
        'process-ledger.json', 'training-launch-claim.json', 'native-allocation-admission.json')]
    paths.append(Path(run['train_log']))
    for rank in range(4):
        paths += [base / 'logs' / name for name in (f'optimizer-rank-{rank}.jsonl',
            f'scheduler-rank-{rank}.jsonl', f'formal-inline-optimizer-step-0-rank-{rank}.json',
            f'formal-inline-optimizer-step-5-rank-{rank}.json')]
    paths += [gate_path(base, run, row['step'], attempt_id=row['attempt_id'])
              for row in _rows(base / 'logs/actual-updates.jsonl')]
    return set(paths)


def qualify_saved_checkpoint(base, directory, *, expected_run_id, expected_execution_id, expected_source_commit):
    """Explicit remote CPU operation, defined here and never run during import."""
    from safetensors import safe_open
    from search_g1.full_state_restore import equal_tree
    from search_g1.qprobe_features import FeatureCache
    from verl.utils.checkpoint_integrity import validate_saved_checkpoint
    base, directory = _remote(base), _remote(directory)
    _require(not directory.exists() and not directory.is_relative_to(base) and
             not base.is_relative_to(directory), 'QUALIFICATION_MUST_USE_NEW_SEPARATE_NAMESPACE')
    _require(re.fullmatch('[0-9a-f]{32}', expected_run_id) is not None and
             re.fullmatch('[0-9a-f]{32}', expected_execution_id) is not None and
             re.fullmatch('[0-9a-f]{40}', expected_source_commit) is not None, 'EXPECTED_IDENTITY_MALFORMED')
    run_path = _owned(base / 'logs/run-manifest.json', base)
    run = _read_json(run_path)
    _require((run['run_id'], run['execution_id'], run['source_commit']) ==
             (expected_run_id, expected_execution_id, expected_source_commit) and
             run['training_world_size'] == 4 and run['actor_micro_batch_per_rank'] == 4 and
             run.get('full_state_restore') is None and run['completed_updates'] == 0 and
             run['question_probe']['exploration_questions'] == 0 and
             run['question_probe']['selected_questions'] == 512 and
             run['question_probe']['responses_per_question'] == 5, 'WRONG_SOURCE_EXECUTION_OR_METHOD')
    checkpoint = _remote(Path(run['checkpoint_dir']) / 'actor/global_step_5')
    _require(checkpoint.is_relative_to(base), 'SOURCE_CHECKPOINT_OUTSIDE_NAMESPACE')
    _require(not (base / 'artifacts/question-probe/completed-pairs/global_step_5/source-receipt.json').exists() and
             not (base / 'logs/five-updates-acceptance.json').exists(),
             'SOURCE_ALREADY_HAS_NATIVE_FIVE_OR_COMPLETED_PAIR')
    files = [_file_record(_owned(path, base), relative_path=str(path.relative_to(checkpoint)))
             for path in sorted(checkpoint.rglob('*')) if path.is_file()]
    by_name = {row['relative_path']: row for row in files}
    _require(len(files) == 21 and set(by_name) == INVENTORY_FILES,
             'EXPECTED_EXACT_TWENTY_ONE_CHECKPOINT_FILES_MISSING')
    complete = _read_json(checkpoint / 'complete.json')
    _require(complete['run_id'] == run['run_id'] and complete['step'] == 5 and
             Path(complete['checkpoint']) == checkpoint and complete['finite_all_saved_tensors'] is True and
             complete['validated_tensor_count'] == 435 and complete['actual_optimizer_writes_per_rank'] == 100,
             'ORIGINAL_COMPLETION_PROOF_MISMATCH')
    _require(len(complete['files_sha256']) == 13 and set(complete['files_sha256']) == DECLARED_FILES,
             'ORIGINAL_THIRTEEN_DECLARED_FILES_CHANGED')
    for name, sha in complete['files_sha256'].items():
        _require(name in by_name and by_name[name]['full_read_sha256'] == sha, 'COMPLETE_DECLARED_SHA_MISMATCH')
    initial = _remote(run['initial_model'])
    canonical_files = []
    for name, sha in run['initial_model_sha256'].items():
        _require(Path(name).name == name, 'CANONICAL_MODEL_FILE_ESCAPES_DIRECTORY')
        row = _file_record(_owned(initial / name), relative_path=name)
        _require(row['full_read_sha256'] == sha, 'CANONICAL_MODEL_FULL_SHA_MISMATCH')
        canonical_files.append(row)
    weight_map = _read_json(checkpoint / 'model.safetensors.index.json')['weight_map']
    _require(len(weight_map) == 435 and set(weight_map.values()) == HF_SHARDS,
             'SAVED_HF_INDEX_ESCAPES_OR_HAS_WRONG_COVERAGE')
    count = validate_saved_checkpoint(checkpoint, initial, run['started_unix'])
    _require(count == 435, 'FULL_CANONICAL_435_FINITE_VALIDATION_FAILED')
    official_serialization = _official_serialization_identity(checkpoint, initial)
    master_tensors = master_numel = 0
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(checkpoint / shard), framework='pt', device='cpu') as reader:
            for name in reader.keys():
                tensor = reader.get_tensor(name)
                _require(tensor.dtype == torch.float32 and torch.isfinite(tensor).all().item(),
                         'MODEL_MASTER_NOT_FINITE_FP32')
                master_tensors += 1; master_numel += tensor.numel()
                del tensor
    _require(master_tensors == 435 and master_numel > 0, 'MASTER_TENSOR_COVERAGE_MISMATCH')
    state_manifest = _read_json(checkpoint / 'training_state/manifest.json')
    _require(state_manifest['completed_outer'] == 5 and
             sorted(row['rank'] for row in state_manifest['worker_files']) == [0, 1, 2, 3],
             'FOUR_SAVED_RANKS_MISSING')
    prefix = _actual_prefix(base, run)
    safety_audit = _first_five_safety_audit(base, run)
    actor_id = run['run_id'] + ':5'
    qstate, qpath, head_path = _validate_qstate(checkpoint, 5, actor_id)
    qproof = state_manifest['question_probe']
    _check_proof(qproof, by_name, 'training_state/question-probe-state.pt')
    _check_proof(qproof['head'], by_name, 'training_state/question-probe-state.pt.head.pt')
    _require(qstate['schema_version'] == 1 and qstate['completed_attempts'] == prefix['attempt_ids'] and
             qstate['reward_probe_version'] == run['initial_probe_version'] and
             qstate['head_version'].startswith('qprobe:5:5:'), 'HEAD5_ATTEMPTS_OR_REWARD_BINDING_MISMATCH')
    np.random.default_rng().bit_generator.state = qstate['selection_rng']
    driver_record = _check_proof(state_manifest['driver_file'], by_name, 'training_state/driver.pt')
    driver = torch.load(driver_record['path'], map_location='cpu', weights_only=False)
    expected_progress = {key: prefix[key] for key in
                         ('completed_outer', 'cumulative_writes', 'attempt_index', 'aborted_attempts', 'full_budget_outer')}
    equal_tree(driver['attempt_progress'], expected_progress)
    equal_tree(driver['question_probe'], qproof)
    equal_tree(driver['candidate_stream'], qstate['candidate_state'])
    _require(driver['completed_outer'] == 5 and driver['probe_version'] == run['initial_probe_version'] and
             driver.get('pending') is None and qstate.get('pending') is None, 'DRIVER_OR_HEAD_BOUNDARY_INCOMPLETE')
    _validate_stream(driver['candidate_stream']); _validate_rng(driver['rng'])
    ranks = []
    for proof in state_manifest['worker_files']:
        row = _check_proof(proof, by_name, f'training_state/rank-{proof["rank"]}.pt')
        saved = torch.load(row['path'], map_location='cpu', weights_only=False)
        _require(saved['run_id'] == run['run_id'] and saved['source_commit'] == run['source_commit'] and
                 saved['full_budget_outers'] == 5 and saved['optimizer_calls'] == 100 and
                 saved['optimizer_rollbacks'] == 0 and saved.get('pending') is None,
                 'WORKER_RUN_FULL_BUDGET_OR_UNCOMMITTED_STATE_MISMATCH')
        summary = validate_rank_state(saved, proof['rank'], 4, 5, expected_writes=100)
        _require(summary['last_attempt_index'] == 5 and summary['used_attempt_ids'] == prefix['attempt_ids'],
                 'WORKER_ATTEMPT_IDS_MISMATCH')
        for values in saved['optimizer']['state'].values():
            for name in ('exp_avg', 'exp_avg_sq'):
                _require(values[name].dtype == torch.float32, 'ADAM_MOMENTS_NOT_FP32')
        _validate_rng(saved['rng'], worker=True)
        summary.update(adam_moments_dtype='float32', full_budget_outers=5,
                       optimizer_calls=100, optimizer_rollbacks=0, saved_worker_rng_validated=True)
        ranks.append(summary)
        del saved
    ranks.sort(key=lambda row: row['rank'])
    _require(len({row['policy_rejections'] for row in ranks}) == 1, 'RANK_POLICY_REJECTION_COUNTERS_DIVERGE')
    live_path = _owned(base / 'artifacts/live/manifest.json', base)
    live = _read_json(live_path)
    _require(live['run_id'] == run['run_id'] and live['round'] == 0 and
             live['version'] == run['initial_probe_version'], 'REWARD_ROUND0_CHANGED')
    reward_files = []
    for key, name in (('commit', 'commit_probe.pkl'), ('boundary', 'boundary_probe.pkl')):
        row = _file_record(_owned(Path(live[key])), relative_path=key)
        _require(row['full_read_sha256'] == live['head_sha256'][name], 'REWARD_HEAD_SHA_MISMATCH')
        reward_files.append(row)
    for name, sha in live['encoder_sha256'].items():
        encoder_file = _owned(_remote(live['encoder']) / name)
        row = next((item for item in canonical_files if item['path'] == str(encoder_file)), None)
        if row is None:
            row = _file_record(encoder_file, relative_path='reward_encoder/' + name)
        _require(row['full_read_sha256'] == sha, 'REWARD_ENCODER_SHA_MISMATCH')
        reward_files.append(row)
    cache = FeatureCache(_remote(run['question_probe']['cache_dir']), expected_spec_sha=qstate['feature_spec_sha'])
    cache.verify_all()
    feature_files = [_file_record(_owned(cache.path / name)) for name in ('manifest.json', 'index.jsonl', 'features.npy')]
    history = _historical_artifacts(base, run, 5)
    _require(set(history['development']) == {'before', 'after5'}, 'EXISTING_DEVELOPMENT_PAIR_MISSING')
    from reproduction.common8.formal_inline_evaluation import verify_pair
    development = verify_pair(base, run)
    evidence = [_file_record(_owned(path, base)) for path in sorted(_evidence_paths(base, run))]
    qualification = dict(schema_version=1, role=QUALIFICATION_ROLE, run_id=run['run_id'],
        execution_id=run['execution_id'], source_commit=run['source_commit'], step=5,
        full_file_sha_count=21, producer_declared_file_count=13,
        hf_saved_tensor_count=435, master_tensors_dtype='float32',
        master_parameter_numel=master_numel, attempt_progress=expected_progress, ranks=ranks,
        driver_rng_validated=True, candidate_stream_bound_to_head=True,
        head_completed_outer=5, actual_first_five_rows_present=True,
        existing_development_pair=development, post_evaluation_rng_persisted=False,
        original_native_five_accepted=False, actual_cuda_restore_accepted=False,
        official_serialization_identity=official_serialization,
        original_first_five_safety_logs_revalidated=True,
        source_first_five_selected_evidence_sha256=safety_audit['selected_evidence_sha256'],
        scope='saved-state CPU qualification with original safety logs revalidated; no native-five receipt reconstructed')
    receipt = dict(schema_version=1, status=SAVED_CHECKPOINT_STATUS, kind=SAVED_CHECKPOINT_KIND,
        run_id=run['run_id'], source_execution_id=run['execution_id'], source_commit=run['source_commit'],
        step=5, world_size=4, checkpoint=str(checkpoint), source_base=str(base), actor_snapshot_id=actor_id,
        source_run_manifest=_file_record(run_path), files=files, driver_file=driver_record, ranks=ranks,
        canonical_model_files=canonical_files,
        attempt_progress=expected_progress, driver_state=dict(completed_outer=5,
            source_consumed=driver['candidate_stream']['consumed_source_rows']),
        question_probe_state_path=str(qpath), head_path=str(head_path), head_version=qstate['head_version'],
        feature_spec_sha=qstate['feature_spec_sha'], feature_files=feature_files,
        feature_cache_manifest_sha256=feature_files[0]['full_read_sha256'],
        reward_probe_manifest=_file_record(live_path), reward_files=reward_files,
        reward_probe_version=live['version'], reward_probe_round=0, historical_artifacts=history,
        hf_saved_tensor_count=435, boundary_origin=BOUNDARY_ORIGIN, boundary_scope=BOUNDARY_SCOPE,
        post_evaluation_rng_persisted=False, completed_pair_published=False,
        original_native_five_accepted=False, existing_development_after5_reused=True,
        actual_cpu_readback=True, actual_cuda_restore_accepted=False,
        evidence_files=evidence, qualification=qualification,
        scope=BOUNDARY_SCOPE, checked_unix=time.time())
    # Bind the publication to the exact unmodified files observed above.
    for row in files + evidence + feature_files + reward_files + canonical_files: _identity_record(row)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / 'source-receipt.json'
    _publish(directory / 'first-five-safety-audit.json', safety_audit)
    receipt['first_five_safety_audit'] = _file_record(directory / 'first-five-safety-audit.json')
    _publish(path, receipt)
    read_source(path)
    return dict(path=str(path), sha256=digest(path), status=SAVED_CHECKPOINT_STATUS,
                kind=SAVED_CHECKPOINT_KIND, step=5, actual_cuda_restore_accepted=False)


def read_source(receipt_path):
    """Read the distinct saved-state contract; never treat it as a completed pair."""
    path = _owned(Path(receipt_path))
    receipt = _read_json(path)
    _require(receipt.get('schema_version') == 1 and receipt.get('status') == SAVED_CHECKPOINT_STATUS and
             receipt.get('kind') == SAVED_CHECKPOINT_KIND and receipt['step'] == 5 and receipt['world_size'] == 4 and
             receipt.get('boundary_origin') == BOUNDARY_ORIGIN and receipt.get('boundary_scope') == BOUNDARY_SCOPE and
             receipt.get('post_evaluation_rng_persisted') is False and receipt.get('completed_pair_published') is False and
             receipt.get('original_native_five_accepted') is False and
             receipt.get('actual_cpu_readback') is True and receipt.get('actual_cuda_restore_accepted') is False,
             'RECEIPT_NOT_DISTINCT_SAVED_CP5_CONTRACT')
    checkpoint, base = _remote(receipt['checkpoint']), _remote(receipt['source_base'])
    _require(checkpoint.is_relative_to(base) and not path.is_relative_to(base) and
             receipt['actor_snapshot_id'] == receipt['run_id'] + ':5', 'RECEIPT_PATH_OR_ACTOR_BINDING_MISMATCH')
    _require(Path(receipt['source_run_manifest']['path']) == base / 'logs/run-manifest.json' and
             Path(receipt['reward_probe_manifest']['path']) == base / 'artifacts/live/manifest.json' and
             Path(receipt['question_probe_state_path']) == checkpoint / 'training_state/question-probe-state.pt' and
             Path(receipt['head_path']) == checkpoint / 'training_state/question-probe-state.pt.head.pt' and
             Path(receipt['first_five_safety_audit']['path']) == path.parent / 'first-five-safety-audit.json',
             'SOURCE_MANIFEST_HEAD_OR_DERIVED_AUDIT_PATH_SUBSTITUTED')
    records = receipt['files']
    by_name = {row['relative_path']: row for row in records}
    actual = {str(item.relative_to(checkpoint)) for item in checkpoint.rglob('*') if item.is_file()}
    _require(len(records) == len(by_name) == 21 and actual == set(by_name) == INVENTORY_FILES,
             'SOURCE_CHECKPOINT_FILE_SET_CHANGED')
    for row in records:
        _require(Path(row['path']) == checkpoint / row['relative_path'], 'SOURCE_RELATIVE_PATH_MISMATCH')
    history = receipt['historical_artifacts']
    _require(set(history['development']) == {'before', 'after5'}, 'PINNED_EXISTING_DEV512_MISSING')
    _require(all(Path(history['development'][key]['path']) == base / 'logs' / name
                 for key, name in (('before', 'formal-inline-development-before.json'),
                                   ('after5', 'formal-inline-development-after.json'))),
             'SOURCE_DEVELOPMENT_PATH_SUBSTITUTED')
    historical = history['checkpoints'] + list(history['development'].values()) + history['reward_archives']
    small = [receipt['source_run_manifest'], receipt['reward_probe_manifest'], receipt['first_five_safety_audit']] + historical + receipt['evidence_files']
    small += [by_name[name] for name in ('complete.json', 'training_state/manifest.json',
              'training_state/driver.pt', 'training_state/question-probe-state.pt',
              'training_state/question-probe-state.pt.head.pt')]
    all_records = records + small + receipt['reward_files'] + receipt['feature_files'] + receipt['canonical_model_files']
    identity = tuple(_identity_record(row) for row in all_records)
    key = (str(path), digest(path), identity)
    if key not in _SOURCE_CACHE:
        for row in small:
            _require(digest(row['path']) == row['full_read_sha256'], 'SOURCE_SMALL_BINDING_SHA_MISMATCH')
        complete = _read_json(checkpoint / 'complete.json')
        _require(set(complete['files_sha256']) == DECLARED_FILES and len(complete['files_sha256']) == 13 and
                 all(by_name[name]['full_read_sha256'] == sha for name, sha in complete['files_sha256'].items()) and
                 complete['run_id'] == receipt['run_id'] and complete['step'] == 5 and
                 complete['validated_tensor_count'] == 435 and complete['actual_optimizer_writes_per_rank'] == 100,
                 'ORIGINAL_DECLARED_SHA_OR_CP5_PROOF_SUBSTITUTED')
        parent = _read_json(receipt['source_run_manifest']['path'])
        _require(checkpoint == _remote(Path(parent['checkpoint_dir']) / 'actor/global_step_5'),
                 'CHECKPOINT_NOT_PARENT_CANONICAL_CP5')
        evidence_paths = {Path(row['path']) for row in receipt['evidence_files']}
        _require(len(evidence_paths) == len(receipt['evidence_files']) and
                 evidence_paths == _evidence_paths(base, parent) and
                 all(_owned(item, base) == item for item in evidence_paths),
                 'SOURCE_EVIDENCE_PATH_SET_SUBSTITUTED')
        canonical = {row['relative_path']: row for row in receipt['canonical_model_files']}
        _require(set(canonical) == set(parent['initial_model_sha256']) and
                 all(Path(row['path']) == _remote(parent['initial_model']) / name and
                     row['full_read_sha256'] == parent['initial_model_sha256'][name]
                     for name, row in canonical.items()), 'CANONICAL_MODEL_RECORDS_SUBSTITUTED')
        feature_paths = {Path(row['path']) for row in receipt['feature_files']}
        _require(feature_paths == {_remote(parent['question_probe']['cache_dir']) / name
                     for name in ('manifest.json', 'index.jsonl', 'features.npy')} and
                 len(feature_paths) == len(receipt['feature_files']) == 3 and
                 next(row['full_read_sha256'] for row in receipt['feature_files']
                      if row['path'].endswith('/manifest.json')) == receipt['feature_cache_manifest_sha256'],
                 'FIXED_FEATURE_CACHE_RECORDS_SUBSTITUTED')
        _require(parent['run_id'] == receipt['run_id'] and parent['execution_id'] == receipt['source_execution_id'] and
                 parent['source_commit'] == receipt['source_commit'] and parent['training_world_size'] == 4 and
                 parent['question_probe']['exploration_questions'] == 0, 'SOURCE_RUN_OR_METHOD_BINDING_CHANGED')
        qualification = receipt['qualification']
        _require(qualification['role'] == QUALIFICATION_ROLE and qualification['run_id'] == receipt['run_id'] and
                 qualification['step'] == 5 and qualification['hf_saved_tensor_count'] == receipt['hf_saved_tensor_count'] == 435 and
                 qualification['full_file_sha_count'] == 21 and qualification['producer_declared_file_count'] == 13 and
                 qualification['master_tensors_dtype'] == 'float32' and
                 qualification['post_evaluation_rng_persisted'] is False and
                 qualification['actual_cuda_restore_accepted'] is False, 'CPU_QUALIFICATION_CONTRACT_CHANGED')
        safety_audit = _read_json(receipt['first_five_safety_audit']['path'])
        _require(safety_audit['run_id'] == receipt['run_id'] and safety_audit['structural_accepted'] is True and
                 safety_audit['actual_complete_outers'] == safety_audit['actual_full_budget_outers'] == 5 and
                 safety_audit['actual_optimizer_writes_per_rank'] == 100 and
                 safety_audit['selected_evidence_sha256'] == qualification['source_first_five_selected_evidence_sha256'] and
                 qualification['original_first_five_safety_logs_revalidated'] is True,
                 'PINNED_FIRST_FIVE_SAFETY_AUDIT_CHANGED')
        from search_g1.full_state_restore import equal_tree
        actual_prefix = _actual_prefix(base, parent)
        progress = {name: actual_prefix[name] for name in
                    ('completed_outer', 'cumulative_writes', 'attempt_index', 'aborted_attempts', 'full_budget_outer')}
        equal_tree(receipt['attempt_progress'], progress)
        equal_tree(qualification['attempt_progress'], progress)
        ranks = receipt['ranks']
        _require(sorted(row['rank'] for row in ranks) == [0, 1, 2, 3] and
                 all(row['completed_outer'] == row['scheduler_last_epoch'] == row['last_attempt_index'] == 5 and
                     row['scheduler_step_count'] == 6 and row['actual_optimizer_writes'] == 100 and
                     row['used_attempt_ids'] == actual_prefix['attempt_ids'] and
                     row['optimizer_tensors_finite'] is True and row['saved_worker_rng_validated'] is True and
                     row['adam_moments_dtype'] == 'float32' and row['full_budget_outers'] == 5 and
                     row['optimizer_calls'] == 100 and row['optimizer_rollbacks'] == 0 for row in ranks),
                 'QUALIFIED_RANK_STATE_COUNTERS_CHANGED')
        equal_tree(qualification['ranks'], ranks)
        _require(receipt['driver_file'] == by_name['training_state/driver.pt'], 'DRIVER_FILE_NOT_ORIGINAL_CP_STATE')
        driver = torch.load(receipt['driver_file']['path'], map_location='cpu', weights_only=False)
        equal_tree(driver['attempt_progress'], progress)
        _validate_rng(driver['rng']); _validate_stream(driver['candidate_stream'])
        qstate, _, _ = _validate_qstate(checkpoint, 5, receipt['actor_snapshot_id'])
        equal_tree(qstate['candidate_state'], driver['candidate_stream'])
        _require(driver['completed_outer'] == 5 and driver['probe_version'] == receipt['reward_probe_version'] and
                 qstate['completed_attempts'] == actual_prefix['attempt_ids'] and
                 qstate['head_version'] == receipt['head_version'] and qstate['feature_spec_sha'] == receipt['feature_spec_sha'] and
                 driver['candidate_stream']['consumed_source_rows'] == receipt['driver_state']['source_consumed'],
                 'QUALIFIED_DRIVER_HEAD_OR_CANDIDATE_BINDING_CHANGED')
        live = _read_json(receipt['reward_probe_manifest']['path'])
        _require(live['run_id'] == receipt['run_id'] and live['version'] == receipt['reward_probe_version'] and
                 live['round'] == receipt['reward_probe_round'] == 0, 'PINNED_REWARD_PROBE_CHANGED')
        _SOURCE_CACHE[key] = parent
    return path, receipt, checkpoint, base, _SOURCE_CACHE[key]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--qualification-dir', required=True, type=Path)
    parser.add_argument('--expected-run-id', required=True)
    parser.add_argument('--expected-execution-id', required=True)
    parser.add_argument('--expected-source-commit', required=True)
    args = parser.parse_args()
    print(json.dumps(qualify_saved_checkpoint(args.base, args.qualification_dir,
        expected_run_id=args.expected_run_id, expected_execution_id=args.expected_execution_id,
        expected_source_commit=args.expected_source_commit), sort_keys=True))
