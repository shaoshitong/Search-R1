"""Frozen Prefix20 E4 scoring of an identity-selected training-only pool.

This does not mine validation/final, generate responses, fit a probe or choose a
hard stratum. A separately prepared SHA-bound plan selects question identities
without consulting scores/EM. Labels only verify authentic own trajectories and
annotate remote diagnostic output. Inference reuses the qualified frozen-G5
evaluator; subsequent hard-versus-random CPU plans are separate artifacts.
"""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import time
import traceback

from . import evaluate_frozen_validation as e

ROLE = 'training_hard_mining_candidates'
STATUS = 'E4_PREFIX20_TRAIN_ONLY_CANDIDATE_SCORING_COMPLETE'
E4_SHA = 'e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'


def exclusion_identities(audits):
    """Read only precomputed small identity audits, never final trajectories."""
    qids, hashes = set(), set()
    if not audits:
        raise ValueError('Explicit matched-fit/diagnostic/selection/final identity exclusions required')
    for audit in audits:
        if audit.get('role') != 'training_mining_exclusions' or not audit.get('groups'):
            raise ValueError('Only explicit training-mining exclusion identity audits may be consumed')
        for row in audit['groups']:
            if (set(row) != {'qid', 'question_hash'} or not isinstance(row['qid'], str) or
                    not isinstance(row['question_hash'], str) or not row['qid'] or not row['question_hash']):
                raise ValueError('Exclusion audit must contain only qid/question_hash, no EM/raw trajectories')
            qids.add(row['qid']); hashes.add(row['question_hash'])
    return qids, hashes


def validate_candidate_plan(plan, canonical_train, exclusion_audits, expected_questions):
    """No score or label participates in candidate identity selection."""
    if type(expected_questions) is not int or not 1 <= expected_questions <= 32768:
        raise ValueError('Explicit actual candidate budget1..32768 required; never pad')
    summary = plan['summary']; groups = plan['groups']
    required = dict(questions=expected_questions, trajectories=5*expected_questions,
        question_selection_uses_labels=False, question_selection_uses_scores=False,
        validation_or_final_questions_used=False, unseen_in_initial169028_warmup_claim=False,
        unseen_in_previous_finetuning_claim=False, excluded_current_matched6528_fit=True)
    if plan.get('role') != ROLE or any(summary.get(key) != value for key, value in required.items()):
        raise ValueError('Require declared training-only identity plan and honest prior-exposure provenance')
    if len(groups) != expected_questions:
        raise ValueError('Actual candidate count differs from explicit budget')
    if summary['ordered_qid_sha256'] != e.digest([g['qid'] for g in groups]):
        raise ValueError('Precommitted candidate qid order changed')
    canonical = {g['qid']: g for g in canonical_train}
    if len(canonical) != len(canonical_train):
        raise ValueError('Canonical training qids repeat')
    forbidden_qids, forbidden_hashes = exclusion_identities(exclusion_audits)
    seen_qids, seen_hashes, seen_tids = set(), set(), set()
    for group in groups:
        if set(group) != {'qid', 'question_hash', 'trajectory_ids', 'labels', 'split'}:
            raise ValueError('Candidate group schema differs; no score/hidden label selection fields')
        qid, qhash, ids = group['qid'], group['question_hash'], group['trajectory_ids']
        if (group['split'] != 'train' or qid not in canonical or qid in seen_qids or qhash in seen_hashes or
                qid in forbidden_qids or qhash in forbidden_hashes or len(ids) != 5 or len(set(ids)) != 5 or
                seen_tids.intersection(ids) or len(group['labels']) != 5 or
                any(type(y) is not int or y not in (0, 1) for y in group['labels'])):
            raise ValueError('Candidate is duplicated, forbidden, non-train, or not authentic own G5')
        source = canonical[qid]
        if any(group[key] != source[key] for key in ('qid', 'question_hash', 'trajectory_ids', 'labels')):
            raise ValueError('Candidate differs from canonical training own IDs/hash/EM')
        seen_qids.add(qid); seen_hashes.add(qhash); seen_tids.update(ids)
    return dict(questions=len(groups), trajectories=len(seen_tids),
        ordered_qid_sha256=e.digest([g['qid'] for g in groups]),
        ordered_own_trajectory_ids_sha256=e.digest([g['trajectory_ids'] for g in groups]),
        exclusions=dict(qids=len(forbidden_qids), question_hashes=len(forbidden_hashes)),
        candidate_identity_selection_used_scores=False, candidate_identity_selection_used_labels=False,
        unseen_in_initial169028_warmup_claim=False, unseen_in_previous_finetuning_claim=False,
        excluded_current_matched6528_fit=True)


def candidate_reader(manifests, groups, references):
    from .packed_features import FeatureReader
    if not isinstance(manifests, list) or not manifests or any(not isinstance(p, str) for p in manifests):
        raise ValueError('Canonical flat native manifest path list required')
    if len(manifests) != len(set(manifests)):
        raise ValueError('Repeated native feature manifest')
    if manifests != [row['path'] for row in references]:
        raise ValueError('Prepared native manifest SHA references differ from canonical path list')
    for reference in references:
        path = e.checked(reference); manifest = e.read(path)
        if (manifest['status'] != 'LOSSLESS_NATIVE_FEATURES_COMPLETE' or manifest['feature_layer'] != 26 or
                manifest['prefix_tokens'] != 20 or manifest['dimension'] != 3584 or
                manifest['question_max_tokens'] != 273):
            raise ValueError('Only native L26 question+Prefix20 training features are supported')
        for filename in ('index.jsonl', 'prototype.bin'):
            if e.checked(manifest['files'][filename]) != path.parent/filename:
                raise ValueError('Native feature identity/prototype binding differs')
    reader = FeatureReader(manifests)
    for group in groups:
        for tid in group['trajectory_ids']:
            if tid not in reader.records or reader.records[tid][1]['qid'] != group['qid']:
                reader.close(); raise ValueError('Candidate native feature belongs to wrong own question')
    return reader


def run(config_path, config_sha):
    if (platform.system() != 'Linux' or not os.environ.get('SLURM_JOB_ID') or
            not os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise RuntimeError('Candidate features/checkpoints may only load inside remote Slurm GPU job')
    config_path = e.remote(config_path)
    if e.proof(config_path)['sha256'] != config_sha:
        raise ValueError('Published immutable scorer config SHA differs')
    config = e.read(config_path); base = e.remote(config['base'])
    if not base.is_dir(): raise ValueError('Prepared isolated candidate scoring root required')
    with (base/'scoring.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return locked_run(config_path, config, config_sha)


def locked_run(config_path, config, config_sha):
    import torch
    from . import probe_ablation_train as u
    from .model import Probe, native_buffers, restore_buffers
    if (config['role'] != ROLE or config['probe_optimizer_updates_authorized'] is not False or
            config['actor_optimizer_updates_authorized'] is not False or
            config.get('validation_complete') is not None or
            type(config['micro_groups']) is not int or not 1 <= config['micro_groups'] <= 8):
        raise ValueError('Training-only bounded frozen scoring configuration required')
    if (int(os.environ.get('WORLD_SIZE', '1')) != 1 or torch.cuda.device_count() != 1 or
            os.environ.get('FLASH_ATTENTION_DETERMINISTIC') != '1' or
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise ValueError('Exactly one GPU with deterministic startup flags required')
    repo = Path(__file__).resolve().parents[3]
    required = {'reproduction/operations/qwen7b_warmup/'+name for name in
        ('score_training_candidates.py', 'evaluate_frozen_validation.py', 'model.py',
         'packed_features.py', 'probe_ablation_train.py', 'probe_ablation_loss.py')}
    if not required <= set(config['code_hashes']): raise ValueError('Missing immutable scorer source bindings')
    for name, wanted in config['code_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or e.proof(repo/name)['sha256'] != wanted:
            raise ValueError('Published scorer source changed: '+name)
    plan_path = e.checked(config['candidate_plan']); plan = e.read(plan_path)
    canonical_path = e.checked(plan['source_bank']); canonical = e.read(canonical_path)
    # Only canonical['train'] is joined; no evaluation split supplies a label/score.
    if any(reference['bytes'] > 64*1024**2 for reference in plan['exclusion_refs']):
        raise ValueError('Only small exclusion identity audits, never raw final trajectories, may be loaded')
    audits = [e.read(e.checked(reference)) for reference in plan['exclusion_refs']]
    acceptance = validate_candidate_plan(plan, canonical['train'], audits, config['questions'])
    del canonical, audits
    descriptor = config['checkpoint']
    if (descriptor['name'] != 'E4' or descriptor['steps'] != 14418 or descriptor['sha256'] != E4_SHA or
            any(plan['source_checkpoint'][key] != descriptor[key] for key in ('path', 'bytes', 'sha256'))):
        raise ValueError('Require exactly the frozen sealed balanced E4 checkpoint')
    checkpoint = e.checked(descriptor); receipt = e.read(e.checked(descriptor['source_receipt']))
    source_config = e.read(e.checked(descriptor['source_config']))
    model_dir = e.remote(config['model_dir']); model_config = e.checked(config['model_config'])
    if model_dir/'config.json' != model_config or model_dir != e.remote(descriptor['model_dir']):
        raise ValueError('Official model metadata path differs')
    features_path = e.checked(plan['features']); manifests = e.read(features_path)
    manifest_refs = e.read(e.checked(config['feature_manifest_refs']))
    reader = candidate_reader(manifests, plan['groups'], manifest_refs)
    base = e.remote(config['base']); output = base/'scoring-output'; output.mkdir(exist_ok=True)
    binding = dict(config=e.proof(config_path), candidate_plan=config['candidate_plan'],
        checkpoint=descriptor, role=ROLE, source_bank=plan['source_bank'], features=plan['features'],
        feature_manifest_refs=config['feature_manifest_refs'],
        exclusion_refs=plan['exclusion_refs'], plan_acceptance=acceptance,
        actor_optimizer_updates=0, probe_optimizer_updates=0, hard_strata_selected=False)
    try:
        if (output/'intent.json').exists():
            if e.read(output/'intent.json') != binding:
                raise ValueError('Previous scoring intent has different immutable sources')
        else: e.atomic(output/'intent.json', binding)
        if (output/'complete.json').exists():
            prior = e.read(output/'complete.json')
            if prior['status'] != STATUS or prior['binding'] != binding:
                raise ValueError('Existing candidate scoring acceptance differs')
            e.checked(prior['predictions']); return prior
        os.environ['QWEN7B_QUESTION_MAX_TOKENS'] = '273'; os.environ['SEARCH_G1_B_PREFIX_TOKENS'] = '20'
        device = torch.device('cuda', 0); torch.cuda.set_device(device)
        torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model = Probe(str(model_dir), 273, 20).eval().requires_grad_(False).to(device)
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        e.validate_checkpoint_metadata(descriptor, receipt, source_config, state)
        model.load_state_dict(state['ema'], strict=True); restore_buffers(model, state['ema_buffers'])
        parameter_sha = u.model_hash(model); buffer_sha = u.tensor_hash(native_buffers(model))
        if any(row['ema_sha256'] != parameter_sha for row in state['rank_states']):
            raise ValueError('Loaded actual E4 EMA hash differs from all four original ranks')
        predictions = output/'predictions.jsonl.gz'
        diagnostics = e.evaluate(model, reader, plan['groups'], device, predictions,
            output/'progress.json', 'E4-training-candidates', micro_groups=config['micro_groups'])
        if (u.model_hash(model) != parameter_sha or
                any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters())):
            raise ValueError('Frozen scorer changed a parameter or computed parameter gradients')
        restore_buffers(model, state['ema_buffers'])
        if u.tensor_hash(native_buffers(model)) != buffer_sha:
            raise ValueError('Original E4 native buffers did not restore exactly')
        if (e.proof(checkpoint) != dict(path=str(checkpoint), bytes=descriptor['bytes'], sha256=descriptor['sha256']) or
                e.proof(plan_path) != config['candidate_plan']):
            raise ValueError('Source checkpoint or fixed candidate plan changed while scoring')
        report = dict(status=STATUS, binding=binding, questions=len(plan['groups']), trajectories=5*len(plan['groups']),
            predictions=e.proof(predictions), diagnostics=diagnostics,
            diagnostic_scope='TRAINING_MINING_ONLY_NOT_VALIDATION_OR_GENERALIZATION',
            score='1-prod(sigmoid(z))-prod(1-sigmoid(z))_stable_logspace',
            feature_layer=26, native_block=27, prefix_tokens=20, question_max_tokens=273,
            ema_parameter_sha256=parameter_sha, ema_native_buffers_sha256=buffer_sha,
            frozen_parameters_unchanged=True, optimizer_constructed=False,
            probe_optimizer_updates=0, actor_optimizer_updates=0, hard_strata_selected=False,
            validation_or_final_scored=False, question_identity_selection_used_scores=False,
            question_identity_selection_used_labels=False, completed_epoch_time=time.time())
        e.atomic(output/'complete.json', report)
        print(e.json.dumps(dict(status=STATUS, questions=report['questions'], trajectories=report['trajectories'],
            predictions=report['predictions'], elapsed_seconds=diagnostics['timing']['elapsed_seconds'])), flush=True)
        return report
    finally: reader.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True); args = parser.parse_args()
    try: run(args.config, args.config_sha256)
    except Exception as error:
        if platform.system() == 'Linux' and os.environ.get('SLURM_JOB_ID'):
            e.atomic(args.config.parent.parent/'failure.json', dict(status='TRAIN_ONLY_CANDIDATE_SCORING_FAILED',
                error=repr(error), traceback=traceback.format_exc(), observed_epoch=time.time()))
        raise


if __name__ == '__main__': main()
