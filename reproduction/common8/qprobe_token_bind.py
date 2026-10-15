"""Continue the completed token-head warmup into a prospective fresh actor run.

This CPU-only step authenticates the original teacher labels and the actual
ten-epoch head artifact. It never samples, fits, or loads an actor checkpoint.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from reproduction.common8.qprobe_configuration import arguments
from search_g1.qprobe_features import COMMON_MANIFEST_SHA, OFFICIAL_REVISION, sha_file
from search_g1.qprobe_token_features import TokenFeatureCache, remote_path

WARMUP = Path('/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-494eafd18b004cb980121e46b188db44/warmup')
READY_SHA = '496e30af7c9c4afb164507f90d14e3f6bfdc7e15e1f6c6f3a4a7c0d5f4198f5f'
HEAD_SHA = '59294d69348917a0ebbfefe8f3a96911187799bb3d5e15d196187b91227dce5c'
TEACHER = Path('/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v4-sigma3-fullouter')
TEACHER_SHA = '88998de7a30455b9135b02cc3c82251fbc1ba1765708244538b6a66c1371e871'
TEACHER_COMPLETE_SHA = '07f4e2e3937c814637b783468ff1062af7ba701e1ac1e136da2dd56f838b45b1'
TOKEN_SPEC_SHA = 'fcc9d61db156ddbb4e96dcd6c9437797b978dbbfc9774145a28e0c761de1a9e7'


def recipe(cache_dir, *, max_tokens=244):
    return dict(head_kind='token_binary_v1', loss_mode='token_binary_std_v1',
        ranking='binary_mixed_probability', candidate_questions=5120,
        selected_questions=512, responses_per_question=5, exploration_questions=0,
        seed=20261013, head_seed=1729, cache_dir=str(cache_dir),
        max_tokens=max_tokens, feature_dim=2048, head_hidden_size=128, attention_heads=4,
        head_learning_rate=3e-4, head_batch_size=256, head_epochs_per_outer=2,
        bootstrap_head_epochs=10, bootstrap_max_groups_per_outer=5120,
        mean_loss_weight=0.0, std_loss_weight=0.3, count_aux_loss_weight=0.0,
        target_statistic='sample_std', replay_window_outer=16, replay_half_life_outer=4,
        teacher_warmup_cache=str(TEACHER))


def _json(path):
    path = Path(path)
    if path.is_symlink() or path.resolve() != path or path.stat().st_size > 128 * 1024**2:
        raise ValueError('TOKEN_BIND_UNSAFE_METADATA_PATH')
    return json.loads(path.read_bytes())


def _once(path, value):
    path = Path(path)
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fresh_run(base):
    from reproduction.common8.formal_contract import assert_formal_run
    from reproduction.common8.source_files import runtime_files
    run = _json(base / 'logs/run-manifest.json')
    assert_formal_run(run)
    if (run.get('training_world_size') != 4 or run.get('actor_micro_batch_per_rank') != 4 or
            run.get('full_state_restore') or run.get('question_probe') is not None):
        raise ValueError('TOKEN_BIND_REQUIRES_PRISTINE_FOUR_RANK_MICRO4_ACTOR')
    for name in ('training-launch-claim.json', 'pipeline-launch-claim.json',
                 'qprobe-bind-claim.json', 'actual-attempts.jsonl', 'actual-updates.jsonl',
                 'resolved-training-config.json', 'policy-safety-calibration.json'):
        if (base / 'logs' / name).exists():
            raise ValueError('TOKEN_BIND_RUN_ALREADY_CLAIMED: ' + name)
    repo = base / 'repo'
    actual_source = {str(p.relative_to(repo)): sha_file(p) for p in runtime_files(repo)}
    if run['source_sha256'] != actual_source:
        raise ValueError('TOKEN_BIND_PREPARED_SOURCE_CHANGED')
    return run


def qualified_warmup(run, cache, retriever_version, resolved):
    """Keep the old label-feature identity distinct from the new token identity."""
    from search_g1.qprobe_warmup import TeacherCache
    from search_g1.qprobe_runtime import strict_scoring_sources
    if (sha_file(WARMUP / 'ready.json') != READY_SHA or
            sha_file(WARMUP / 'fresh-head.pt') != HEAD_SHA):
        raise ValueError('TOKEN_BIND_ACTUAL_WARMUP_ARTIFACT_CHANGED')
    ready = _json(WARMUP / 'ready.json')
    if (ready.get('stage') != 'TOKEN_BINARY_FRESH_HEAD_WARMUP10_COMPLETE' or
            ready['source_commit'] != 'f3807ac6642761fcf06a7772255879b1cf291105' or
            ready['head_sha256'] != HEAD_SHA or ready['head_adam_steps'] != 200 or
            ready['head_optimizer_updates'] != 200 or ready['questions'] != 5120 or
            ready['epoch_records'] != 11 or ready['minibatch_records'] != 200 or
            ready['token_feature_spec_sha256'] != cache.spec_sha or
            ready['trajectories_generated'] != 0 or ready['old_head_inherited'] is not False or
            ready['formal_training_started'] is not False):
        raise ValueError('TOKEN_BIND_WARMUP_NOT_THE_COMPLETED_FRESH_BINARY_FIT')
    if sha_file(WARMUP / 'warmup-metrics.json') != ready['metrics_sha256']:
        raise ValueError('TOKEN_BIND_WARMUP_METRICS_SHA_CHANGED')
    if sha_file(TEACHER / 'identity.json') != TEACHER_SHA or sha_file(TEACHER / 'complete.json') != TEACHER_COMPLETE_SHA:
        raise ValueError('TOKEN_BIND_ORIGINAL_TEACHER_IDENTITY_CHANGED')
    identity = _json(TEACHER / 'identity.json')
    if (identity['feature_spec_sha'] != ready['original_teacher_feature_spec_sha256'] or
            identity['model_revision'] != run['model_revision'] or
            identity['model_files_sha256'] != run['initial_model_sha256'] or
            identity['train_parquet_sha256'] != run['input_sha256']['train.parquet'] or
            identity['data_manifest_sha256'] != COMMON_MANIFEST_SHA or
            identity['retriever_version'] != retriever_version or
            identity['sampling_config'] != resolved['actor_rollout_ref']['rollout'] or
            identity['agent_config'] != {key: resolved['data'][key] for key in (
                'max_start_length', 'max_prompt_length', 'max_response_length', 'max_obs_length')} or
            identity['max_turns'] != resolved['max_turns'] or identity['training_world_size'] != 4):
        raise ValueError('TOKEN_BIND_TEACHER_MODEL_DATA_OR_GENERATION_RECIPE_CHANGED')
    if identity['scorer_sources'] != strict_scoring_sources():
        raise ValueError('TOKEN_BIND_STRICT_SCORER_CHANGED')
    for name, digest in dict(identity['scorer_sources'], **identity['rollout_sources']).items():
        if run['source_sha256'].get(name) != digest:
            raise ValueError('TOKEN_BIND_TEACHER_SCIENTIFIC_SOURCE_CHANGED: ' + name)
    # The immutable completed cache is read-only: no constructor writer path.
    teacher = TeacherCache.__new__(TeacherCache)
    teacher.root, teacher.identity = TEACHER, identity
    with (TEACHER / 'writer.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        proof, groups, _ = teacher.verify_complete()
    if proof != _json(TEACHER / 'complete.json'):
        raise ValueError('TOKEN_BIND_TEACHER_COMPLETE_PROOF_CHANGED')
    labels = [dict(group, complete=True, selection_source='bootstrap_all_candidate', **proof['protocol'])
              for group in groups]
    qids = [label['qid'] for label in labels]
    if len(qids) != len(set(qids)) or len(qids) != 5120:
        raise ValueError('TOKEN_BIND_WARMUP_LABEL_COUNT')
    for label in labels:
        if label['source_outer'] != 0 or cache.index[label['qid']]['question_hash'] != label['question_hash']:
            raise ValueError('TOKEN_BIND_LABEL_FEATURE_MISMATCH')
    return ready, labels, qids


def bind(base, cache_dir, *, retriever_version, infra_rollout):
    """Bind actual warmup state without re-fitting or inheriting an actor state."""
    from reproduction.common8 import configuration
    from search_g1.qprobe_token_online import OnlineTokenBinaryQProbe, TokenOnlineQProbeConfig
    from search_g1.rollout_phase_contract import validate as validate_rollout
    base, cache_dir = remote_path(base), remote_path(cache_dir)
    run = _fresh_run(base)
    cache = TokenFeatureCache(cache_dir, expected_spec_sha=TOKEN_SPEC_SHA)
    cache.verify_all()
    if (cache.manifest['count'] != 169028 or cache.spec['global_max_tokens'] != 244 or
            cache.spec['train_parquet_sha256'] != run['input_sha256']['train.parquet'] or
            cache.spec['encoder_revision'] != OFFICIAL_REVISION):
        raise ValueError('TOKEN_BIND_CACHE_NOT_FULL_CANONICAL_TRAIN')
    _, config, _ = configuration.resolved(base)
    from omegaconf import OmegaConf
    resolved = OmegaConf.to_container(config, resolve=True)
    ready, labels, qids = qualified_warmup(run, cache, retriever_version, resolved)
    settings = recipe(cache_dir)
    directory = base / 'artifacts/question-probe/token-warmup-adoption'
    directory.mkdir(parents=True, exist_ok=False)
    _once(base / 'logs/qprobe-bind-claim.json', dict(source_commit=run['source_commit'],
        run_id=run['run_id'], actual_warmup_sha256=HEAD_SHA, claimed_unix=time.time(), no_automatic_retry=True))
    provenance = dict(actor_snapshot_id=run['run_id'] + ':0', warmup_ready=str(WARMUP / 'ready.json'),
        warmup_ready_sha256=READY_SHA, warmup_source_commit=ready['source_commit'],
        teacher_identity_sha256=TEACHER_SHA, teacher_complete_sha256=TEACHER_COMPLETE_SHA)
    head = OnlineTokenBinaryQProbe.adopt_warmup(WARMUP / 'fresh-head.pt', labels, qids,
        expected_checkpoint_sha=HEAD_SHA, expected_feature_spec_sha=cache.spec_sha,
        provenance=provenance, config=TokenOnlineQProbeConfig.from_mapping(settings), feature_cache=cache)
    head_path = directory / 'initial.pt'
    head_receipt = head.save(head_path, actor_snapshot_id=run['run_id'] + ':0')
    adopted = dict(stage='TOKEN_BINARY_FORMAL_INITIAL_HEAD_ADOPTED', run_id=run['run_id'],
        source_commit=run['source_commit'], head_receipt=head_receipt,
        raw_warmup_checkpoint_sha256=HEAD_SHA, feature_spec_sha=cache.spec_sha,
        actual_head_adam_updates=200, completed_outer=0, warmup_questions=5120,
        teacher_trajectories_reused=25600, trajectories_regenerated=False,
        no_head_refit=True, actor_optimizer_inherited=False, actor_checkpoint_inherited=False,
        initialization_provenance=copy.deepcopy(head.initialization_provenance), checked_unix=time.time())
    ready_path = directory / 'ready.json'
    _once(ready_path, adopted)
    settings.update(output_dir=str(base / 'artifacts/question-probe'),
        initial_head_checkpoint=str(head_path), initial_head_checkpoint_sha256=sha_file(head_path),
        initial_head_ready=str(ready_path), initial_head_ready_sha256=sha_file(ready_path))
    proposed = dict(run, question_probe=settings, retriever_version=retriever_version,
                    mixed_first_selection=False, infra_rollout=validate_rollout(infra_rollout, 4))
    arguments(proposed, base)
    original = (base / 'logs/run-manifest.json').read_bytes()
    if _json(base / 'logs/run-manifest.json') != run:
        raise ValueError('TOKEN_BIND_MANIFEST_CHANGED_DURING_ADOPTION')
    backup = base / 'preparation/question-probe-pre-bind.json'
    with backup.open('xb') as stream:
        stream.write(original); stream.flush(); os.fsync(stream.fileno())
    from reproduction.a800.owned_processes import publish
    publish(base / 'logs/run-manifest.json', proposed)
    _once(base / 'logs/qprobe-bound.json', dict(run_id=run['run_id'], source_commit=run['source_commit'],
        question_probe=settings, initialized_from_this_experiment_warmup=True,
        head_warmup_adam_updates_excluded_from_formal_outers=200, checked_unix=time.time()))
    return proposed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--retriever-version', required=True)
    parser.add_argument('--infra-rollout-json', required=True, type=Path)
    args = parser.parse_args()
    run = bind(args.base, args.cache_dir, retriever_version=args.retriever_version,
               infra_rollout=_json(remote_path(args.infra_rollout_json)))
    print(json.dumps(dict(run_id=run['run_id'], question_probe=run['question_probe'],
                         formal_training_started=False), allow_nan=False))


if __name__ == '__main__':
    main()
