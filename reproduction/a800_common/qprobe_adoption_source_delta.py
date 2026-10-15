"""Exact source provenance for reuse of the original frozen G1 initial0 heads.

This is a source-inventory tool, not calibration or training. The complete
approved question-probe baseline is pinned independently of the new document.
Only the explicitly named integration edits may differ from that baseline.
The document excludes only its own byte hash, avoiding a self-referential map.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess


ORIGINAL_RUN_ID = '6de29bcb7be741cfa1bfacdf9e34efc9'
ORIGINAL_SOURCE_COMMIT = 'f820ba6cb2db4f5cbe2ffc370cf515c3fe22d43c'
ORIGINAL_BASE = '/data/shaoshitong/Search-G1-a800-fresh200-7895330a2c584bba9940720e98f006ef'
ORIGINAL_SOURCE_MAP_SHA256 = '0a72ca74abf61ce836ca8ddad2106b02470d6705c9a6bad927dd10031da5950d'
APPROVED_BASE_COMMIT = '1552de146c2dd44e4fd40eb94b27839ac8648ef1'
APPROVED_BASE_MAP_SHA256 = 'e442af63217a1a2ecf9de6833c4ca18a51f1721f5e1f4df3b202feb8946ff750'
DELTA_PATH = 'reproduction/a800_common/qprobe-initial0-adoption-source-delta.json'
ROLE = 'a800_qprobe_frozen_g1_initial0_source_delta'
PRECISION_PROFILE = 'bf16_policy_fp32_master'
MODEL_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
NATIVE_HEAD_KIND = 'qwen_native_lastblock_binary_v1'
NATIVE_WARMUP_BASE = Path('/data/shaoshitong/Search-G1-qprobe-qwen-lastblock-warmup10-096c8297cba74f8089063e763686671e')
NATIVE_WARMUP_SOURCE = 'e81445905bf8b15dbb0bdec3afa29d99ab0b11d8'
NATIVE_WARMUP_SOURCE_MAP = '006912cc7817b2ae8abde32a85c42d21b2c1976bf0b5618066c6aa9d2f0674ad'
_NATIVE_INITIAL_CACHE = {}

# No directory or suffix exception: these are the explicitly assigned changes
# after the published question-probe baseline. Their actual bytes are still
# recorded and compared in the complete target map and changed-source pairs.
INTEGRATION_EDITS = frozenset((
    'reproduction/a800_common/adopt_initial_probes.py',
    'reproduction/a800_common/qprobe_adoption_source_delta.py',
    'reproduction/a800_common/test_qprobe_initial_probe_adoption.py',
    'reproduction/a800_common/deploy_qprobe.py',
    'reproduction/a800_common/deploy.py',
    'reproduction/a800_common/prepare_run.py',
    'reproduction/a800_common/run_pipeline.py',
    'reproduction/a800/probe_admission.py',
    'reproduction/common8/run_pipeline.py',
    'reproduction/common8/verify_full.py',
    'reproduction/a800_common/train.sh',
    'reproduction/common8/native_allocation.py',
    'reproduction/common8/formal_inline_safety.py',
    'reproduction/common8/gpu_training_admission.py',
    'reproduction/common8/test_local_gpu_placement.py',
    'reproduction/a800_common/precompute_qprobe.py',
    'reproduction/a800_common/test_deploy_qprobe.py',
    'search_g1/qprobe_runtime.py',
    'search_g1/test_qprobe_runtime.py',
    'search_g1/test_qprobe_selection.py',
    # Derived prequential diagnostics use the same saved predictions/labels.
    # Their exact source pair does not change frozen G1 reward calibration.
    'search_g1/qprobe_selection.py',
    'search_g1/qprobe_warmup.py',
    'search_g1/test_qprobe_warmup.py',
    # Bounded-context and recoverable-update repair. Each actual source pair
    # remains inventoried against the pinned baseline and original receipt.
    'reproduction/common8/context_budget.py',
    'reproduction/common8/configuration.py',
    'reproduction/common8/formal_contract.py',
    'reproduction/common8/qprobe_bind.py',
    # The new question head loss changes neither frozen G1 reward heads nor
    # the teacher's accepted trajectories; exact edited bytes remain bound.
    'reproduction/common8/qprobe_configuration.py',
    'reproduction/common8/qprobe-head-defaults.json',
    'reproduction/common8/test_qprobe_bind.py',
    'reproduction/common8/test_qprobe_configuration.py',
    'search_g1/test_qprobe_model.py',
    'verl/trainer/config/ppo_trainer.yaml',
    'reproduction/common8/sampling_capture.py',
    'reproduction/common8/test_continuous_sampling.py',
    'reproduction/common8/generation_behavior.py',
    'reproduction/common8/formal_attempt_audit.py',
    'reproduction/common8/safety_calibration_evidence.py',
    'reproduction/common8/test_zero_reference_monitor_schema.py',
    'reproduction/common8/test_complete_outer_accounting.py',
    'reproduction/common8/recoverable_attempt.py',
    'reproduction/common8/runtime.py',
    'reproduction/common8/scheduler.py',
    'reproduction/common8/update_evidence.py',
    'reproduction/common8/worker_evidence.py',
    'search_g1/full_state_restore.py',
    'search_g1/async_rollout_runtime.py',
    'search_g1/test_async_rollout_runtime.py',
    'search_g1/rollout_phase_contract.py',
    'search_g1/logprob_memory.py',
    'search_g1/monitor_forward_reuse.py',
    'search_g1/numerics.py',
    'search_g1/optimizer_transaction.py',
    'search_g1/policy_safety.py',
    'search_g1/qprobe_checkpoint.py',
    'search_g1/qprobe_saved_checkpoint.py',
    'search_g1/qprobe_model.py',
    'search_g1/training_state.py',
    'verl/trainer/ppo/ray_trainer.py',
    'search_r1/llm_agent/generation.py',
    'verl/workers/fsdp_workers.py',
    'verl/workers/sharding_manager/fsdp_vllm.py',
    'verl/workers/rollout/vllm_rollout/vllm_rollout.py',
    'verl/utils/torch_functional.py',
    'verl/workers/actor/dp_actor.py',
    # Explicit question-only feature/head additions, separate from frozen G1
    # reward calibration. Every added/edited byte remains in the exact delta.
    'search_g1/qprobe_token_features.py',
    'search_g1/qprobe_token_model.py',
    'search_g1/qprobe_token_online.py',
    'search_g1/test_qprobe_token_model.py',
    'search_g1/test_qprobe_token_online.py',
    'search_g1/test_qprobe_token_runtime.py',
    'search_g1/qprobe_qwen_block_model.py',
    'search_g1/qprobe_qwen_block_online.py',
    'search_g1/test_qprobe_qwen_block_model.py',
    'search_g1/test_qprobe_qwen_block_online.py',
    'search_g1/test_qprobe_qwen_runtime.py',
    'search_g1/test_qprobe_saved_native_checkpoint.py',
    'search_g1/test_qprobe_checkpoint.py',
    'search_g1/qprobe_driver_gpu.py',
    'search_g1/test_qprobe_driver_gpu.py',
    'verl/trainer/main_ppo_eco.py',
    'reproduction/common8/qprobe_token_bind.py',
    'reproduction/common8/qprobe_qwen_bind.py',
    'reproduction/common8/test_qprobe_qwen_bind.py',
    'reproduction/common8/qprobe-native-block-defaults.json',
    'reproduction/common8/test_native_qprobe_configuration.py',
    'reproduction/a800_common/qprobe_token_warmup.py',
    'reproduction/a800_common/qprobe_qwen_block_warmup.py',
    'reproduction/a800_common/test_qprobe_qwen_block_warmup.py',
    'reproduction/a800_common/qprobe_token_feature_bank.py',
    'reproduction/a800_common/test_qprobe_token_feature_bank.py',
    'reproduction/a800_common/test_qprobe_native_initial_adoption.py',
    'reproduction/a800_common/qprobe_qwen_formal_ops.py',
    'reproduction/a800_common/test_qprobe_qwen_formal_ops.py',
))

# These implement original G1 calibration, hidden-feature extraction, binary
# head fitting, frozen readout and strict labels. New question-probe labels do
# not authorize changing them. Generation is versioned separately: new bounded
# rollout contexts do not claim to reproduce the historical calibration traces.
# Model/data/artifact bytes are checked separately
# by adopt_initial_probes._compatible and _original_artifacts.
FROZEN_G1_SOURCES = (
    'reproduction/common8/refit.py',
    'ts_probe_dump.py', 'ts_probe_train.py', 'probe_server.py',
    'search_g1/calibration_evidence.py', 'search_g1/signals.py',
    'search_g1/resource.py', 'search_r1/qa_protocol.py',
    'verl/utils/reward_score/qa_em.py',
    'reproduction/a800/prepare_run.py',
    'reproduction/a800/retriever_local.py',
    'reproduction/a800/recovery_binding.py',
    'reproduction/common8/adopt_initial_probes.py',
    'reproduction/common8/verify_data.py',
    'reproduction/common8/budget.json',
)
SCIENTIFIC_IDENTITY = (
    'initial_model', 'reference_model', 'initial_model_sha256',
    'model_revision', 'input_sha256', 'calibration_question_ids',
    'retrieval_asset_paths', 'retrieval_encoder',
)
INHERITED_STATE_FIELDS = (
    'actor_checkpoint', 'reference_checkpoint', 'resume_path',
    'inherited_policy_run_id', 'inherited_optimizer_run_id',
    'inherited_scheduler_run_id', 'full_state_restore',
)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def document_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _source_map(mapping):
    _check(type(mapping) is dict and mapping, 'QPROBE_ADOPTION_EMPTY_SOURCE_MAP')
    for name, sha in mapping.items():
        _check(type(name) is str and str(PurePosixPath(name)) == name
               and not PurePosixPath(name).is_absolute()
               and '..' not in PurePosixPath(name).parts
               and type(sha) is str and len(sha) == 64
               and all(c in '0123456789abcdef' for c in sha),
               'QPROBE_ADOPTION_BAD_SOURCE_BINDING: ' + str(name))
    return mapping


def _native_storage_path(value):
    path = Path(value)
    _check(path.is_absolute() and path == path.resolve() and
           str(path).startswith(('/data/', '/mnt/', '/backup/')),
           'QPROBE_ADOPTION_NATIVE_PATH_NOT_CANONICAL_REMOTE_STORAGE')
    return path


def _owned_native_file(value):
    path = _native_storage_path(value)
    _check(path.is_file() and not path.is_symlink() and path.stat().st_uid == os.getuid(),
           'QPROBE_ADOPTION_NATIVE_INITIAL_FILE_NOT_OWNED')
    return path


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _native_json(path, expected_sha):
    _check(type(expected_sha) is str and len(expected_sha) == 64 and
           all(c in '0123456789abcdef' for c in expected_sha) and
           path.stat().st_size <= 128 * 1024**2 and _file_sha(path) == expected_sha,
           'QPROBE_ADOPTION_NATIVE_INITIAL_METADATA_SHA_CHANGED')
    return json.loads(path.read_bytes())


def _native_file_identity(path):
    state = path.stat()
    return (str(path), state.st_dev, state.st_ino, state.st_size,
            state.st_mtime_ns, state.st_ctime_ns, state.st_uid)


def validate_native_initial_head(run, settings):
    """Accept only this experiment's owned logical0 native warmup adoption."""
    _check(settings.get('snapshot_path') is None and settings.get('restore_state') is None,
           'QPROBE_ADOPTION_NATIVE_INITIAL_MUST_NOT_RESTORE_AN_ACTOR_PAIR')
    checkpoint_dir = _native_storage_path(run.get('checkpoint_dir', ''))
    base = checkpoint_dir.parents[1]
    directory = base / 'artifacts/question-probe/native-qwen-warmup-adoption'
    _check(checkpoint_dir == base / 'checkpoints/search-g1-common8' and
           settings.get('output_dir') == str(base / 'artifacts/question-probe') and
           settings.get('initial_head_checkpoint') == str(directory / 'initial.pt') and
           settings.get('initial_head_ready') == str(directory / 'ready.json'),
           'QPROBE_ADOPTION_NATIVE_INITIAL_OUTSIDE_NEW_RUN_ARTIFACTS')
    head_path = _owned_native_file(directory / 'initial.pt')
    ready_path = _owned_native_file(directory / 'ready.json')
    ready = _native_json(ready_path, settings.get('initial_head_ready_sha256'))
    head_sha = settings.get('initial_head_checkpoint_sha256')
    receipt = ready.get('head_receipt', {})
    actual = ready.get('actual_warmup', {})
    expected_raw = NATIVE_WARMUP_BASE / 'warmup/ready.json'
    _check(type(head_sha) is str and len(head_sha) == 64 and all(c in '0123456789abcdef' for c in head_sha) and
           ready.get('stage') == 'QWEN_NATIVE_FORMAL_INITIAL_HEAD_ADOPTED' and
           ready.get('run_id') == run['run_id'] and ready.get('source_commit') == run['source_commit'] and
           receipt.get('path') == str(head_path) and receipt.get('actor_snapshot_id') == run['run_id'] + ':0' and
           receipt.get('checkpoint_sha', receipt.get('sha256')) == head_sha and
           ready.get('actual_head_adam_updates') == 200 and ready.get('completed_outer') == 0 and
           ready.get('warmup_questions') == 5120 and ready.get('no_head_refit') is True and
           ready.get('teacher_trajectories_reused') == 25600 and ready.get('formal_training_started') is False and
           ready.get('actual_saved_head_cpu_readback') is True and
           ready.get('teacher_replay_excluded') is True and ready.get('trajectories_regenerated') is False and
           ready.get('actor_optimizer_inherited') is False and ready.get('actor_checkpoint_inherited') is False and
           actual.get('ready_path') == str(expected_raw) and actual.get('source_commit') == NATIVE_WARMUP_SOURCE and
           actual.get('source_map_sha256') == NATIVE_WARMUP_SOURCE_MAP,
           'QPROBE_ADOPTION_NATIVE_INITIAL_NOT_THIS_FRESH_WARMUP_EXPERIMENT')
    raw_path = _owned_native_file(expected_raw)
    raw = _native_json(raw_path, actual.get('ready_sha256'))
    proof = raw.get('native_block_proof', {})
    _check(raw.get('stage') == 'QWEN_LAST_BLOCK_FRESH_HEAD_WARMUP10_COMPLETE' and
           raw.get('source_commit') == NATIVE_WARMUP_SOURCE and raw.get('source_map_sha256') == NATIVE_WARMUP_SOURCE_MAP and
           raw.get('head_sha256') == actual.get('head_sha256') == ready.get('raw_warmup_checkpoint_sha256') and
           raw.get('prepared_sha256') == actual.get('prepared_sha256') and
           raw.get('head_optimizer_updates') == raw.get('head_adam_steps') == 200 and
           raw.get('epoch_records') == 11 and raw.get('minibatch_records') == 200 and raw.get('questions') == 5120 and
           raw.get('trajectories_generated') == 0 and raw.get('formal_training_started') is False and
           raw.get('old_head_inherited') is False and raw.get('token_feature_spec_sha256') == ready.get('feature_spec_sha') and
           proof == ready.get('native_block_proof') and proof.get('official_block_copy_verified') is True and
           proof.get('native_class') == 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer' and
           proof.get('block_index') == 35 and proof.get('official_revision') == MODEL_REVISION and
           bool(proof.get('transformers_version')),
           'QPROBE_ADOPTION_NATIVE_WARMUP_COMPLETION_OR_COPY_PROOF_CHANGED')
    identity = _native_file_identity(head_path)
    cache_key = (run['run_id'], run['source_commit'], document_sha(settings), document_sha(ready),
                 document_sha(raw), identity)
    if cache_key not in _NATIVE_INITIAL_CACHE:
        from search_g1.qprobe_runtime import load_head_checkpoint
        from search_g1.qprobe_qwen_block_online import QwenBlockOnlineConfig
        head = load_head_checkpoint(head_path, settings, expected_actor_snapshot_id=run['run_id'] + ':0',
            expected_feature_spec_sha=ready['feature_spec_sha'], expected_checkpoint_sha=head_sha)
        provenance = head.initialization_provenance
        verified = head.verify_native_provenance()
        _check(head.config == QwenBlockOnlineConfig.from_mapping(settings) and
               head.feature_spec_sha == ready['feature_spec_sha'] and
               head.completed_outer == head.update_count == 0 and head._adam_steps() == 200 and
               head.replay_labels == [] and head.replay_features == [] and head.protocol is None and
               verified.get('accepted') is True and verified.get('teacher_replay_excluded') is True and
               provenance == ready.get('initialization_provenance') and
               provenance.get('warmup_source_commit') == NATIVE_WARMUP_SOURCE and
               provenance.get('warmup_ready') == str(expected_raw) and
               provenance.get('warmup_ready_sha256') == actual['ready_sha256'] and
               provenance.get('warmup_checkpoint_sha256') == raw['head_sha256'] and
               provenance.get('warmup_head_adam_steps') == 200 and provenance.get('no_refit') is True and
               provenance.get('teacher_replay_excluded') is True and
               provenance.get('formal_replay_groups_at_adoption') == 0 and
               head.initial_block_proof == raw.get('initial_block_proof') and
               head.initial_readout_proof == raw.get('initial_readout_proof') and
               head.native_implementation == raw.get('native_implementation') and
               _native_file_identity(head_path) == identity,
               'QPROBE_ADOPTION_NATIVE_INITIAL_HEAD_STATE_OR_LINEAGE_CHANGED')
        _NATIVE_INITIAL_CACHE[cache_key] = copy.deepcopy(verified)
    return copy.deepcopy(_NATIVE_INITIAL_CACHE[cache_key])


def validate_fresh_run(run, old):
    """Keep the special source route exclusive to this fresh official actor."""
    _check(old['run_id'] == ORIGINAL_RUN_ID
           and old['source_commit'] == ORIGINAL_SOURCE_COMMIT,
           'QPROBE_ADOPTION_WRONG_ORIGINAL_RUN')
    _check(run.get('initial_probe_adoption_source_base') == ORIGINAL_BASE,
           'QPROBE_ADOPTION_WRONG_ORIGINAL_NAMESPACE')
    for candidate in (run, old):
        _check(candidate.get('profile') == 'common8'
               and candidate.get('execution_backend') == 'ssh_local'
               and type(candidate.get('training_world_size')) is int
               and candidate['training_world_size'] == 4
               and type(candidate.get('completed_updates')) is int
               and candidate['completed_updates'] == 0
               and candidate.get('target_updates') == 200
               and all(candidate.get(name) is None for name in INHERITED_STATE_FIELDS),
               'QPROBE_ADOPTION_REQUIRES_FRESH_FOUR_RANK_ACTOR')
    _check(run['run_id'] != old['run_id']
           and run['execution_id'] != old['execution_id']
           and run.get('precision_profile') == PRECISION_PROFILE,
           'QPROBE_ADOPTION_TARGET_IDENTITY_OR_PRECISION_DIFFERS')
    for name in SCIENTIFIC_IDENTITY:
        _check(canonical(run[name]) == canonical(old[name]),
               'QPROBE_ADOPTION_INITIAL0_SCIENTIFIC_IDENTITY_CHANGED: ' + name)
    _check(run['model_revision'] == MODEL_REVISION
           and run['initial_model'] == run['reference_model']
           == '/data/search-g1/assets/qwen2.5-3b-instruct',
           'QPROBE_ADOPTION_NOT_THE_OFFICIAL_INITIAL_MODEL')
    settings = run.get('question_probe')
    _check(type(settings) is dict and run.get('mixed_first_selection') is False,
           'QPROBE_ADOPTION_NOT_QUESTION_PROBE_SELECTION')
    for name, value in dict(candidate_questions=5120, selected_questions=512,
                            responses_per_question=5, exploration_questions=0,
                            bootstrap_max_groups_per_outer=5120).items():
        _check(type(settings.get(name)) is int and settings[name] == value,
               'QPROBE_ADOPTION_QUESTION_BUDGET_CHANGED: ' + name)
    if settings.get('head_kind') == NATIVE_HEAD_KIND:
        validate_native_initial_head(run, settings)
    else:
        _check(all(settings.get(name) is None for name in
                   ('initial_head_checkpoint', 'snapshot_path')),
               'QPROBE_ADOPTION_MUST_NOT_INHERIT_A_QUESTION_HEAD')
    teacher = settings.get('teacher_warmup_cache')
    _check(type(teacher) is str and Path(teacher).is_absolute()
           and str(Path(teacher)).startswith(('/data/', '/mnt/', '/backup/'))
           and type(run.get('retriever_version')) is str
           and bool(run['retriever_version']),
           'QPROBE_ADOPTION_UNBOUND_TEACHER_OR_RETRIEVER')


def validate_source_delta(delta, previous, current):
    """Reject unknown edits, omitted differences and changed calibration bytes."""
    previous, current = _source_map(previous), _source_map(current)
    fields = {'schema_version', 'role', 'original_run_id', 'original_source_commit',
              'original_source_map_sha256', 'approved_base_commit',
              'approved_base_source_sha256', 'approved_base_map_sha256',
              'target_precision_profile', 'target_source_map_sha256_without_delta',
              'changes', 'unchanged_frozen_g1_source_sha256'}
    _check(type(delta) is dict and set(delta) == fields
           and type(delta['schema_version']) is int and delta['schema_version'] == 1
           and delta['role'] == ROLE
           and delta['original_run_id'] == ORIGINAL_RUN_ID
           and delta['original_source_commit'] == ORIGINAL_SOURCE_COMMIT
           and delta['approved_base_commit'] == APPROVED_BASE_COMMIT
           and delta['target_precision_profile'] == PRECISION_PROFILE,
           'QPROBE_ADOPTION_UNRECOGNIZED_DELTA')
    _check(document_sha(previous) == ORIGINAL_SOURCE_MAP_SHA256
           == delta['original_source_map_sha256'],
           'QPROBE_ADOPTION_ORIGINAL_SOURCE_MAP_DIFFERS')
    baseline = _source_map(delta['approved_base_source_sha256'])
    _check(document_sha(baseline) == APPROVED_BASE_MAP_SHA256
           == delta['approved_base_map_sha256'],
           'QPROBE_ADOPTION_APPROVED_BASELINE_DIFFERS')
    _check(DELTA_PATH not in previous and DELTA_PATH not in baseline,
           'QPROBE_ADOPTION_DELTA_REPLACED_AN_ORIGINAL_FILE')
    target = {name: sha for name, sha in current.items() if name != DELTA_PATH}
    _check(document_sha(target) == delta['target_source_map_sha256_without_delta'],
           'QPROBE_ADOPTION_TARGET_SOURCE_MAP_DIFFERS')
    _check(set(baseline) <= set(target), 'QPROBE_ADOPTION_BASELINE_SOURCE_REMOVED')
    edits = {name for name in set(baseline) | set(target)
             if baseline.get(name) != target.get(name)}
    _check(edits <= INTEGRATION_EDITS,
           'QPROBE_ADOPTION_UNAPPROVED_INTEGRATION_EDIT: ' + ', '.join(sorted(edits-INTEGRATION_EDITS)))
    frozen = {name: previous.get(name) for name in FROZEN_G1_SOURCES}
    _check(all(type(sha) is str and baseline.get(name) == target.get(name) == sha
               for name, sha in frozen.items())
           and delta['unchanged_frozen_g1_source_sha256'] == frozen,
           'QPROBE_ADOPTION_FROZEN_G1_CALIBRATION_OR_SCORER_CHANGED')
    expected = {name: dict(old_sha256=previous.get(name), new_sha256=target.get(name))
                for name in sorted(set(previous) | set(target))
                if previous.get(name) != target.get(name)}
    _check(delta['changes'] == expected and expected,
           'QPROBE_ADOPTION_CHANGED_SOURCE_PAIRS_DIFFER')
    return expected


def validate(delta, previous, current, run, old):
    validate_fresh_run(run, old)
    _check(previous == old['source_sha256'] and current == run['source_sha256'],
           'QPROBE_ADOPTION_NOT_COMPLETE_BOUND_MANIFESTS')
    return validate_source_delta(delta, previous, current)


def git_source_map(repo, commit):
    names = subprocess.check_output(
        ['git', '-C', str(repo), 'ls-tree', '-r', '--name-only', commit], text=True).splitlines()
    versions = ('verl/version/version', 'verl/single_controller/version/version')
    result = {}
    for name in names:
        if (name.endswith(('.py', '.sh', '.yaml', '.yml', '.json')) or name in versions) and not \
                {'.git', '__pycache__', 'evidence'}.intersection(PurePosixPath(name).parts):
            raw = subprocess.check_output(['git', '-C', str(repo), 'show', commit + ':' + name])
            result[name] = hashlib.sha256(raw).hexdigest()
    return result


def generate(repo):
    """Inventory final edited files; no model load, remote access or run claims."""
    from reproduction.common8.source_files import runtime_files
    repo = Path(repo).resolve()
    previous = git_source_map(repo, ORIGINAL_SOURCE_COMMIT)
    baseline = git_source_map(repo, APPROVED_BASE_COMMIT)
    target = {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in runtime_files(repo) if str(path.relative_to(repo)) != DELTA_PATH}
    delta = dict(schema_version=1, role=ROLE, original_run_id=ORIGINAL_RUN_ID,
                 original_source_commit=ORIGINAL_SOURCE_COMMIT,
                 original_source_map_sha256=document_sha(previous),
                 approved_base_commit=APPROVED_BASE_COMMIT,
                 approved_base_source_sha256=baseline,
                 approved_base_map_sha256=document_sha(baseline),
                 target_precision_profile=PRECISION_PROFILE,
                 target_source_map_sha256_without_delta=document_sha(target),
                 unchanged_frozen_g1_source_sha256={name: previous.get(name) for name in FROZEN_G1_SOURCES},
                 changes={name: dict(old_sha256=previous.get(name), new_sha256=target.get(name))
                          for name in sorted(set(previous) | set(target))
                          if previous.get(name) != target.get(name)})
    validate_source_delta(delta, previous, target)
    return delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    delta = generate(args.repo)
    output = args.repo.resolve() / DELTA_PATH
    output.write_bytes(json.dumps(delta, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n')
    print(json.dumps(dict(path=str(output), role=ROLE, changed_sources=len(delta['changes']),
                          target_source_map_sha256_without_delta=delta['target_source_map_sha256_without_delta'],
                          document_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                          calibration_generated=False, training_started=False)))


if __name__ == '__main__':
    main()
