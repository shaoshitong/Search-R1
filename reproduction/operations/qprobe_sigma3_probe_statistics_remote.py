"""One frozen-prefix CPU analysis of existing sigma3 probe evidence.

Read stdin JSON; emit PROBE_STATS_JSON plus scalars/hashes only. This program
never trains, generates rollouts, loads an actor, reads QA/token files, or writes.
The cold prediction is derived from published zero initialization, not a log.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import re
import signal
import stat
import sys
import time

sys.dont_write_bytecode = True
EXPECTED_BASE = '/data/shaoshitong/Search-G1-a800-fresh200-86cf721183014e729036924e8978a6b9'
EXPECTED_RUN = 'f84749ca43be42bd91f2bc63d73347dd'
EXPECTED_EXECUTION = 'ffdc8bb4690d4b8793335d9a718e3dcb'
EXPECTED_SOURCE = 'c200f9ebcaba6fd08039f24df8e7ffc55cf19c40'
EXPECTED_HEAD_SHA = '8a34948bbb115213be405b192f4b40d04bfc25a7f35f7ce1d02d389c845ae641'
EXPECTED_READY_SHA = '0fb27f558bd7e4c7b36e5c52a74df44da257e8715cdd33553e19455839fd5fa8'
TOTAL_LIMIT = 256 * 1024 * 1024
JSON_TOTAL_LIMIT = 128 * 1024 * 1024
JSON_FILE_LIMIT = 8 * 1024 * 1024
HEAD_LIMIT = 128 * 1024 * 1024
OUTPUT_LIMIT = 128 * 1024
MAX_COMPLETED_STEPS = 50
PREFIX = 'PROBE_STATS_JSON:'
started = time.monotonic()
spent = 0
json_spent = 0
evidence_hashes = {}


class Unknown(Exception):
    pass


def need(value, reason):
    if not value:
        raise Unknown(reason)


def check():
    need(time.monotonic() - started < 120, 'cpu_analysis_deadline')


def expired(signum, frame):
    raise Unknown('cpu_analysis_deadline')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def valid_hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def snapshot(path, limit, label, json_file=False):
    """Owned canonical immutable file, read twice from the same descriptor."""
    global spent, json_spent
    check()
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink(),
         'file_path_not_canonical')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid(), 'file_owner_or_type')
        size = before.st_size
        need(0 < size <= limit, 'file_size_limit')
        need(spent + 2 * size <= TOTAL_LIMIT, 'aggregate_read_limit')
        need(not json_file or json_spent + 2 * size <= JSON_TOTAL_LIMIT, 'json_read_limit')
        copies = []
        for _ in range(2):
            offset = 0
            parts = []
            while offset < size:
                check()
                part = os.pread(fd, min(1024 * 1024, size - offset), offset)
                need(bool(part), 'file_truncated')
                parts.append(part)
                offset += len(part)
                spent += len(part)
                if json_file:
                    json_spent += len(part)
            copies.append(b''.join(parts))
        after = os.fstat(fd)
        current = path.stat(follow_symlinks=False)
        signature = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                   value.st_mtime_ns, value.st_ctime_ns, value.st_uid)
        need(signature(before) == signature(after) == signature(current) and
             copies[0] == copies[1], 'immutable_file_changed')
        digest = sha(copies[0])
        evidence_hashes[label] = dict(sha256=digest, bytes=size,
                                      twice_read_same_owned_file=True)
        return copies[0], digest
    finally:
        os.close(fd)


def document(path, label):
    raw, digest = snapshot(path, JSON_FILE_LIMIT, label, json_file=True)
    value = json.loads(raw)
    need(type(value) is dict, 'json_document_object')
    return value, digest


def close_scalar(left, right, reason):
    need(finite(left) and finite(right) and math.isclose(left, right, rel_tol=1e-5, abs_tol=5e-6), reason)


def scalar_losses(record):
    names = ('replay_weighted_loss', 'sigma3_ce', 'count_ce', 'std_mse',
             'normalized_sample_std_mse', 'sigma3_ce_contribution',
             'std_mse_contribution', 'count_ce_contribution',
             'replay_groups', 'bootstrap_groups', 'head_optimizer_updates',
             'completed_outer', 'update_count')
    return {name: record[name] for name in names if name in record and finite(record[name])}


def statistics(probabilities, counts, np):
    """Uniform question weights; no answer text, identifiers, or per-row output."""
    p = np.asarray(probabilities, dtype=np.float64)
    k = np.asarray(counts, dtype=np.int64)
    n = len(k)
    need(n > 0 and p.shape == (n, 6) and np.isfinite(p).all() and
         (p >= 0).all() and (p <= 1).all() and
         np.allclose(p.sum(1), 1, rtol=0, atol=1e-5), 'count_probability_schema')
    need(((k >= 0) & (k <= 5)).all(), 'correct_count_range')
    mixed = (k > 0) & (k < 5)
    pm = p[:, 1:5].sum(1)
    guess = pm >= 0.5
    tp = int((guess & mixed).sum())
    tn = int((~guess & ~mixed).sum())
    fp = int((guess & ~mixed).sum())
    fn = int((~guess & mixed).sum())
    ratio = lambda a, b: float(a / b) if b else None
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    stds = np.sqrt(np.arange(6) * (5 - np.arange(6)) / 20)
    actual_std = stds[k]
    predicted_std = p @ stds
    raw_mse = float(np.square(predicted_std - actual_std).mean())
    normalized_mse = raw_mse / 0.3
    sigma = np.minimum(k, 5 - k)
    sigma_p = np.stack((p[:, 0] + p[:, 5], p[:, 1] + p[:, 4], p[:, 2] + p[:, 3]), axis=1)
    ce6 = float(-np.log(np.maximum(p[np.arange(n), k], 1e-12)).mean())
    ce3 = float(-np.log(np.maximum(sigma_p[np.arange(n), sigma], 1e-12)).mean())
    correlation = None
    if np.ptp(predicted_std) > 0 and np.ptp(actual_std) > 0:
        correlation = float(np.corrcoef(predicted_std, actual_std)[0, 1])
    return dict(questions=n, correct_count_histogram=np.bincount(k, minlength=6).tolist(),
        true_mixed_fraction=float(mixed.mean()), all_wrong_fraction=float((k == 0).mean()),
        all_correct_fraction=float((k == 5).mean()), binary_threshold=0.5,
        binary_confusion=dict(tp=tp, tn=tn, fp=fp, fn=fn),
        binary_accuracy=float((guess == mixed).mean()),
        binary_balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
        binary_precision=ratio(tp, tp + fp), binary_recall=recall,
        binary_specificity=specificity, binary_f1=ratio(2 * tp, 2 * tp + fp + fn),
        binary_predicted_mixed_fraction=float(guess.mean()), binary_brier=float(np.square(pm - mixed).mean()),
        binary_majority_baseline=max(float(mixed.mean()), float((~mixed).mean())),
        six_accuracy=float((p.argmax(1) == k).mean()),
        six_majority_baseline=float(np.bincount(k, minlength=6).max() / n),
        sigma3_accuracy=float((sigma_p.argmax(1) == sigma).mean()),
        sigma3_ce=ce3, six_class_ce=ce6, normalized_sample_std_mse=normalized_mse,
        sample_std_mse=raw_mse, sample_std_mae=float(np.abs(predicted_std - actual_std).mean()),
        actual_sample_std_mean=float(actual_std.mean()), predicted_sample_std_mean=float(predicted_std.mean()),
        predicted_std_actual_std_pearson=correlation,
        sigma3_ce_contribution=ce3, std_mse_contribution=3 * normalized_mse,
        count_ce_contribution=0.1 * ce6, total_sigma3_loss=ce3 + 3 * normalized_mse + 0.1 * ce6,
        question_weights='uniform', probability_log_floor=1e-12,
        argmax_tie_rule='lowest_index', undefined_ratio_value=None)


def main(result):
    raw_request = sys.stdin.buffer.read(65537)
    need(len(raw_request) <= 65536, 'stdin_limit')
    request = json.loads(raw_request)
    need(type(request) is dict, 'request_object')
    binding = dict(base=request.get('base'), run=request.get('run'),
                   execution=request.get('execution'), source=request.get('source'))
    need(binding == dict(base=EXPECTED_BASE, run=EXPECTED_RUN,
                         execution=EXPECTED_EXECUTION, source=EXPECTED_SOURCE), 'fixed_run_binding')
    result.update(run_id=EXPECTED_RUN, execution_id=EXPECTED_EXECUTION, source_commit=EXPECTED_SOURCE)
    result['binding_sha256'] = sha(canonical(binding))
    steps = request.get('completed_steps')
    need(type(steps) is list and 0 < len(steps) <= MAX_COMPLETED_STEPS, 'frozen_complete_prefix_required')
    need(all(type(row) is dict and type(row.get('step')) is int and row['step'] == index and
             type(row.get('attempt_id')) is str and re.fullmatch('[0-9a-f]{32}', row['attempt_id'])
             for index, row in enumerate(steps, 1)), 'frozen_complete_prefix_schema')
    need(len({row['attempt_id'] for row in steps}) == len(steps), 'duplicate_attempt_in_frozen_prefix')
    base = Path(EXPECTED_BASE)
    need(base.resolve() == base and base.stat().st_uid == os.getuid(), 'base_owner_or_path')
    manifest, _ = document(base / 'logs/run-manifest.json', 'run_manifest')
    need((manifest['run_id'], manifest['execution_id'], manifest['source_commit']) ==
         (EXPECTED_RUN, EXPECTED_EXECUTION, EXPECTED_SOURCE) and
         manifest['execution_backend'] == 'ssh_local' and manifest['completed_updates'] == 0 and
         manifest['training_world_size'] == 4 and manifest.get('full_state_restore') is None, 'manifest_identity')
    source_map = manifest['source_sha256']
    need(type(source_map) is dict, 'source_map_schema')
    if request.get('source_map_sha256') is not None:
        need(sha(canonical(source_map)) == request['source_map_sha256'], 'source_map_sha')
    module_path = base / 'repo/search_g1/qprobe_model.py'
    source_raw, source_sha = snapshot(module_path, 2 * 1024 * 1024, 'qprobe_model_source')
    need(valid_hash(source_map.get('search_g1/qprobe_model.py')) and
         source_sha == source_map['search_g1/qprobe_model.py'], 'qprobe_model_source_sha')
    sys.path.insert(0, str(base / 'repo'))
    module = importlib.import_module('search_g1.qprobe_model')
    need(Path(module.__file__).resolve() == module_path, 'qprobe_model_import_path')
    import numpy as np
    import torch
    selection_path = base / 'repo/search_g1/qprobe_selection.py'
    _, selection_sha = snapshot(selection_path, 2 * 1024 * 1024, 'qprobe_selection_source')
    need(selection_sha == source_map.get('search_g1/qprobe_selection.py'), 'qprobe_selection_source_sha')
    selection_module = importlib.import_module('search_g1.qprobe_selection')
    need(Path(selection_module.__file__).resolve() == selection_path, 'qprobe_selection_import_path')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    need(os.environ['CUDA_VISIBLE_DEVICES'] == '', 'cuda_visibility')
    expected_loss = dict(loss_mode='sigma3_std_aux6_v1', std_loss_weight=3.0,
                         count_aux_loss_weight=0.1, mean_loss_weight=0.0)
    need(all(manifest['question_probe'].get(key) == value for key, value in expected_loss.items()), 'manifest_loss_identity')
    ready, ready_sha = document(base / 'artifacts/question-probe/teacher-warmup-ready.json', 'teacher_warmup_ready')
    need(ready_sha == request.get('expected_warmup_ready_sha256', EXPECTED_READY_SHA) == EXPECTED_READY_SHA,
         'teacher_warmup_ready_sha')
    head = ready['head_checkpoint']
    need(ready['status'] == 'FROZEN_TEACHER_WARMUP_HEAD_READY' and ready['fitted_questions'] == 5120 and
         ready['completed_training_outers'] == 0 and ready['target_training_outers'] == 200 and
         ready['head_metrics']['bootstrap_groups'] == 5120 and head['completed_outer'] == 0 and
         head['actor_snapshot_id'] == EXPECTED_RUN + ':0', 'teacher_warmup_ready_identity')
    head_path = base / 'artifacts/question-probe/teacher-warmup-head/initial.pt'
    need(head['path'] == str(head_path) and type(head['bytes']) is int and 0 < head['bytes'] <= HEAD_LIMIT and
         head['sha256'] == head['checkpoint_sha'] == request.get('expected_initial_head_sha256', EXPECTED_HEAD_SHA) == EXPECTED_HEAD_SHA,
         'immutable_initial_head_binding')
    checkpoint_raw, checkpoint_sha = snapshot(head_path, HEAD_LIMIT, 'immutable_initial_head')
    need(len(checkpoint_raw) == head['bytes'] and checkpoint_sha == EXPECTED_HEAD_SHA, 'immutable_initial_head_sha_or_size')
    state = torch.load(io.BytesIO(checkpoint_raw), map_location='cpu', weights_only=True)
    del checkpoint_raw
    need(type(state) is dict and state['schema_version'] == module.SCHEMA_VERSION and
         state['completed_outer'] == state['update_count'] == 0 and not state['completed_attempts'] and
         state['actor_snapshot_id'] == EXPECTED_RUN + ':0' and state['feature_spec_sha'] == head['feature_spec_sha'],
         'initial_head_state_identity')
    config = module.QProbeConfig.from_mapping(state['config'])
    need(config == module.QProbeConfig.from_mapping(manifest['question_probe']) and
         state['loss_identity'] == head['loss_identity'] == ready['head_metrics']['loss_identity'] == config.loss_identity,
         'initial_head_configuration')
    batches_per_epoch = math.ceil(5120 / config.head_batch_size)
    need(ready['head_metrics']['head_optimizer_updates'] == batches_per_epoch * config.head_epochs_per_outer,
         'warmup_epoch_batch_update_count')
    result['warmup_training_schedule'] = dict(
        groups=5120, responses_per_group=5, trajectories=25600,
        extra_warmup_outer_stages=ready['extra_warmup_outers'],
        teacher_generation_chunks=10, groups_per_generation_chunk=512,
        head_epochs=config.head_epochs_per_outer, head_batch_size=config.head_batch_size,
        head_batches_per_epoch=batches_per_epoch,
        successful_head_optimizer_updates=ready['head_metrics']['head_optimizer_updates'],
        head_learning_rate=config.head_learning_rate, hidden_size=config.hidden_size,
        actor_optimizer_updates_during_warmup=0,
        actor_complete_outers_during_warmup=ready['completed_training_outers'],
        head_per_minibatch_loss_history='not_persisted_by_training_source',
        initial_or_epoch_boundary_loss_history='not_persisted_except_final_fit_summary',
        evidence='SHA_verified_saved_initial_head_config_and_matching_warmup_ready_summary')
    features = state['replay_features']
    labels = state['replay_labels']
    need(isinstance(features, torch.Tensor) and features.device.type == 'cpu' and features.dtype == torch.float32 and
         features.shape == (5120, state['feature_dim']) and type(state['feature_dim']) is int and state['feature_dim'] == 2048 and
         type(labels) is list and len(labels) == 5120, 'initial_replay_shape')
    counts = []
    identities = set()
    for label in labels:
        check()
        need(type(label) is dict and label.get('complete') is True and label.get('source_outer') == 0 and
             label.get('selection_source') == 'bootstrap_all_candidate', 'bootstrap_label_provenance')
        em = label.get('strict_em')
        need(type(em) is list and len(em) == 5 and all(type(value) is int and value in (0, 1) for value in em),
             'bootstrap_strict_label_schema')
        need(type(label.get('correct_count')) is int and label['correct_count'] == sum(em) and
             type(label.get('qid')) is str and bool(label['qid']) and valid_hash(label.get('question_hash')),
             'bootstrap_label_count_or_identity')
        identities.add(label['qid'])
        need(all(label.get(key) == state['protocol'][key] for key in
                 ('sampling_contract_sha', 'strict_scoring_contract_sha', 'retriever_version')), 'bootstrap_label_protocol')
        counts.append(sum(em))
    need(len(identities) == 5120, 'bootstrap_question_uniqueness')
    model = module.CountMomentProbe(state['feature_dim'], config.hidden_size).cpu()
    need(int(torch.count_nonzero(model.net[-1].weight)) == 0 and int(torch.count_nonzero(model.net[-1].bias)) == 0,
         'published_cold_zero_initialization')
    cold = statistics(np.full((5120, 6), 1 / 6, dtype=np.float64), counts, np)
    cold.update(provenance='derived_not_logged', derivation='published final-layer zero weight/bias implies six equal logits',
                predicted_mixed_probability=2 / 3, historical_cold_predictions_logged=False,
                predicted_std_actual_std_pearson=None, correlation_reason='undefined_constant_cold_predictor')
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    version_digest = hashlib.sha256(state['feature_spec_sha'].encode())
    for key, value in sorted(model.state_dict().items()):
        need(value.device.type == 'cpu' and torch.isfinite(value).all().item(), 'head_parameter_schema')
        version_digest.update(key.encode())
        version_digest.update(value.detach().numpy().tobytes())
    version = 'qprobe:0:0:' + version_digest.hexdigest()[:16]
    need(version == head['version'] == ready['head_metrics']['version'], 'immutable_head_version')
    with torch.inference_mode():
        prediction = model(features)
        target = torch.tensor(counts, dtype=torch.long)
        objective, parts = module.count_moment_loss(prediction, target, config, torch.ones(5120))
    fitted = statistics(prediction['probabilities'].numpy(), counts, np)
    fitted.update(provenance='immutable_warmup_head_cpu_inference_on_its_stored_5120_training_groups',
                  interpretation='in_sample_after_fitting_not_held_out_or_prefit_accuracy')
    recomputed = dict(replay_weighted_loss=float(objective), **{key: float(value) for key, value in parts.items()})
    for key, value in recomputed.items():
        close_scalar(value, ready['head_metrics'][key], 'saved_warmup_loss_recompute_mismatch')
    need(cold['correct_count_histogram'] == fitted['correct_count_histogram'] and
         cold['true_mixed_fraction'] == fitted['true_mixed_fraction'], 'same_bootstrap_labels')
    def offline_top512(probabilities):
        moments = selection_module.count_predictions(probabilities)
        scores = moments['expected_std']
        tie_seed = str(config.seed) + ':1'
        def rank_key(index):
            label = labels[index]
            tie = sha((tie_seed + '\0' + label['qid'] + '\0' + label['question_hash']).encode())
            return (-float(scores[index]), tie)
        selected = sorted(range(5120), key=rank_key)[:512]
        actual = np.asarray(counts, dtype=np.int64)[selected]
        return dict(questions=512, true_mixed_fraction=float(((actual > 0) & (actual < 5)).mean()),
            actual_sample_std_mean=float(np.sqrt(actual * (5 - actual) / 20).mean()),
            ranking='published_count_predictions_expected_sample_std_then_identity_hash',
            tie_seed_config_seed_plus_outer1=True, all_scores_tied=bool(np.ptp(scores) == 0),
            scope='in_sample_offline_Top512_from_same_initial_5120_teacher_labels; not formal rollout or generalization')
    result['in_sample_offline_Top512'] = dict(
        cold_uniform=offline_top512(np.full((5120, 6), 1 / 6, dtype=np.float64)),
        warmup_fitted=offline_top512(prediction['probabilities'].numpy()))
    result['initial_5120_same_labels'] = dict(cold_uniform=cold, warmup_fitted=fitted,
        warmup_saved_losses=scalar_losses(ready['head_metrics']), warmup_recomputed_losses=recomputed,
        saved_losses_recomputed_match=True, labels_identical_before_after=True,
        fresh_actor_or_optimizer_loaded=False, head_training_performed=False,
        actor_four_rank_live_verification='unknown_not_performed_by_cpu_statistics')
    del prediction, state, features, labels, model
    result['formal_selected_512'] = []
    for frozen in steps:
        check()
        step, attempt = frozen['step'], frozen['attempt_id']
        event, _ = document(base / 'artifacts/question-probe/completed' / (attempt + '.json'), 'committed_step_' + str(step))
        directory = base / 'artifacts/question-probe/attempts' / attempt
        selection, _ = document(directory / 'selection.json', 'selection_step_' + str(step))
        audit, _ = document(directory / 'rollout-audit.json', 'rollout_audit_step_' + str(step))
        need(event['stage'] == 'question_probe_outer_committed' and event['outer'] == step and event['attempt_id'] == attempt and
             event['successful_actor_snapshot_id'] == EXPECTED_RUN + ':' + str(step) and
             event['head_update']['completed_outer'] == step, 'committed_outer_identity')
        need(selection['stage'] == 'question_probe_selection_frozen' and selection['step'] == step and
             selection['attempt_id'] == attempt and selection['prediction_before_label'] is True and
             selection['post_rollout_em_filter'] is False and selection['exploration_questions'] == 0 and
             selection['attempted_candidates'] == 5120 and selection['selected_groups'] == 512 and
             selection['head_version'] == event['prediction_version'] and
             selection['feature_spec_sha'] == head['feature_spec_sha'], 'prefit_selection_identity')
        need(audit['selection_mode'] == 'question_probe' and audit['step'] == step and audit['attempt_id'] == attempt and
             audit['source_commit'] == EXPECTED_SOURCE and audit['selected_groups'] == 512 and
             audit['attempted_candidates'] == 5120 and audit['responses_per_group'] == 5 and
             audit['selected_random_exploration'] == 0 and audit['question_probe_version'] == selection['head_version'] and
             audit['policy_snapshot_id'] == selection['policy_snapshot_id'] and
             audit['prequential'] == event['prequential'], 'rollout_audit_identity')
        candidates = selection['questions']
        groups = audit['groups']
        need(type(candidates) is list and len(candidates) == 5120 and type(groups) is list and len(groups) == 512,
             'formal_metadata_counts')
        by_ordinal = {}
        for row in candidates:
            ordinal = row['ordinal']
            need(type(ordinal) is int and 0 <= ordinal < 5120 and ordinal not in by_ordinal and
                 type(row['selected']) is bool, 'candidate_ordinal_schema')
            if row['selected']:
                by_ordinal[ordinal] = row
        need(len(by_ordinal) == 512, 'selected_candidate_count')
        seen = set()
        probabilities = []
        selected_counts = []
        for group in groups:
            ordinal = group['ordinal']
            need(type(ordinal) is int and ordinal in by_ordinal and ordinal not in seen, 'selected_label_join')
            seen.add(ordinal)
            row = by_ordinal[ordinal]
            em = group['strict_correctness']
            need(type(em) is list and len(em) == 5 and all(type(value) is int and value in (0, 1) for value in em),
                 'formal_strict_label_schema')
            need(group['question_id'] == row['qid'] and group['content_sha256'] == row['content_sha256'] and
                 group['selected'] is True and group['source'] == row['selection_source'] == 'predicted_diversity',
                 'selected_question_binding')
            probabilities.append(row['probabilities'])
            selected_counts.append(sum(em))
        measured = statistics(probabilities, selected_counts, np)
        prequential = event['prequential']['all']
        need(prequential['questions'] == 512, 'prequential_question_count')
        for saved_key, calculated_key in dict(actual_mixed_fraction='true_mixed_fraction',
                count_nll='six_class_ce', std_mae='sample_std_mae', mixed_brier='binary_brier',
                actual_sample_std_mean='actual_sample_std_mean', all_wrong_fraction='all_wrong_fraction',
                all_correct_fraction='all_correct_fraction').items():
            close_scalar(prequential[saved_key], measured[calculated_key], 'saved_prequential_recompute_mismatch')
        postfit = event['head_update']
        need(postfit['loss_mode'] == config.loss_mode and postfit['loss_identity'] == config.loss_identity,
             'formal_head_loss_identity')
        result['formal_selected_512'].append(dict(step=step, prefit=measured,
            prefit_matches_existing_prequential=True, postfit_replay_loss=scalar_losses(postfit),
            scope='only selected 512 labeled questions; unselected 4608 have predictions but no outcomes',
            complete_outer_evidence='frozen caller_verified_complete_prefix_and_matching_committed_head_event',
            rank_ledger_readback='unknown_not_performed_by_cpu_statistics'))
    result['frozen_complete_step_count'] = len(steps)
    result['unrequested_new_steps_scanned'] = False
    result['status'] = 'observed'


if __name__ == '__main__':
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(120)
    result = dict(role='qprobe_sigma3_existing_immutable_head_and_prefit_statistics', status='unknown',
        cpu_only=True, read_only=True, binary_threshold=0.5,
        no_actor_load=True, no_training=True, no_new_rollout=True,
        no_qa_or_token_files=True, no_question_identifiers_or_vectors_output=True)
    try:
        main(result)
    except Exception as failure:
        reason = str(failure) if isinstance(failure, Unknown) else 'unexpected_exception'
        if not re.fullmatch('[a-z0-9_]{1,96}', reason):
            reason = 'sanitized_failure'
        result.update(status='unknown', failure_type=type(failure).__name__, reason=reason)
    finally:
        signal.alarm(0)
        result.update(aggregate_bytes_read=spent, json_bytes_read=json_spent,
            elapsed_seconds=time.monotonic() - started, evidence_hashes=evidence_hashes,
            limits=dict(total_bytes=TOTAL_LIMIT, json_total_bytes=JSON_TOTAL_LIMIT,
                        head_file_bytes=HEAD_LIMIT, json_file_bytes=JSON_FILE_LIMIT,
                        completed_steps=MAX_COMPLETED_STEPS, cpu_seconds=120, output_bytes=OUTPUT_LIMIT))
    encoded = json.dumps(result, allow_nan=False)
    if len((PREFIX + encoded + '\n').encode()) > OUTPUT_LIMIT:
        encoded = json.dumps(dict(role=result['role'], status='unknown', reason='output_limit',
                                  aggregate_bytes_read=spent, cpu_only=True, read_only=True), allow_nan=False)
    print(PREFIX + encoded, flush=True)
