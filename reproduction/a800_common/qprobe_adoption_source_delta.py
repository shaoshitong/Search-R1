"""Exact source provenance for reuse of the original frozen G1 initial0 heads.

This is a source-inventory tool, not calibration or training. The complete
approved question-probe baseline is pinned independently of the new document.
Only the explicitly named integration edits may differ from that baseline.
The document excludes only its own byte hash, avoiding a self-referential map.
"""
import argparse
import hashlib
import json
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
