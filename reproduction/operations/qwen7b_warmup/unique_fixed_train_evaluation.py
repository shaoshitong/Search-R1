"""Matched raw/EMA fit measurements on one fixed, label-blind training G5 pool.

The question identities were seen by the source E1 training. Their particular
G5 combinations are sampled independently of EM and are held fixed afterwards.
This pool measures fitting, not independent generalization or online batches.
No final confirmation pool is read and no feature bank is reopened here.
"""
import random
import time
from pathlib import Path

from . import probe_ablation_train as u
from .mixed_direct_evaluation import live_state_hashes, score_reader


def make_fixed_training_groups(groups, source_e1_seen_qids, seed=2026102201,
                               count=2048):
    """Select questions and one own G5 with no class-conditioned eligibility."""
    lookup = {g['qid']: g for g in groups}
    seen = set(source_e1_seen_qids)
    if len(lookup) != len(groups) or not seen <= set(lookup) or len(seen) < count:
        raise ValueError('Source E1 seen training identities must bind the fixed pool')
    rng = random.Random(seed)
    # Sorting removes set/process/hash randomization from the fixed selection.
    eligible = sorted(seen, key=str)
    selected = rng.sample(eligible, count)
    result = []
    for qid in selected:
        group = lookup[qid]
        ids, labels = group['trajectory_ids'], group['labels']
        if len(ids) != 10 or len(set(ids)) != 10 or len(labels) != 10:
            raise ValueError('Fixed training evaluation requires authentic own G10')
        indices = sorted(rng.sample(range(10), 5))
        result.append(dict(qid=qid,
            trajectory_ids=[ids[i] for i in indices],
            labels=[labels[i] for i in indices]))
    proof = dict(role='fixed_E1_seen_training_question_pool_not_independent',
        questions=count, trajectories=5*count, seed=seed,
        source_training_questions=len(groups), source_E1_seen_questions=len(seen),
        source_E1_seen_qids_sha256=u.digest(eligible),
        ordered_qids_sha256=u.digest(selected), groups_sha256=u.digest(result),
        question_choice_label_blind=True, own_G5_choice_label_blind=True,
        fixed_across_E1_E2_E3_and_raw_EMA=True,
        G5_previously_trained_combination_claimed=False,
        source_EM_labels_used_for_sampling=False,
        class_balancing_or_hard_replay_used=False,
        evaluation_question_overlap_allowed=False)
    return result, proof


def evaluate_fixed_training(base, c, model, ema, native, head, device, reader,
                            groups, poolproof, receipt, epoch):
    """Use the already open training reader, retaining all accepted live state."""
    base = Path(base)
    folder = base/'fixed-training-evaluations'/('epoch-%02d' % epoch)
    done = folder/'complete.json'
    if (len(groups) != 2048 or poolproof['questions'] != len(groups) or
            poolproof['groups_sha256'] != u.digest(groups)):
        raise ValueError('Matched fixed2048 training pool changed')
    if done.exists():
        result = u.read(done)
        if result['checkpoint'] != receipt or result['pool_proof'] != poolproof:
            raise ValueError('Fixed train evaluation belongs to another state or pool')
        for value in result['metrics'].values():
            u.checked(value['predictions'])
        return result
    folder.mkdir(parents=True, exist_ok=True)
    rng = u.rng_state(device)
    buffers = (u.buffers(model), u.buffers(ema))
    modes = [[(module, module.training) for module in probe.modules()]
             for probe in (model, ema)]
    before = live_state_hashes(model, ema, native, head, device)
    metrics = {}
    try:
        for label, probe in (('raw', model), ('EMA', ema)):
            metrics[label] = score_reader(probe, reader, groups, device,
                folder/(label+'-fixed_training2048.jsonl.gz'),
                base/'fixed-training-evaluation-progress.json',
                label+'-fixed_training2048', micro_groups=4)
    finally:
        u.restore_rng(rng, device)
        u.restore_buffers(model, buffers[0])
        u.restore_buffers(ema, buffers[1])
        for values in modes:
            for module, training in values:
                module.training = training
    after = live_state_hashes(model, ema, native, head, device)
    if before != after:
        raise ValueError('Fixed training evaluation changed accepted live state')
    result = dict(status='FIXED_E1_SEEN_TRAINING2048_RAW_EMA_EVALUATION_COMPLETE',
        epoch=epoch, cursor=receipt['cursor'], checkpoint=receipt,
        config_sha256=u.sha(base/'config.json'), pool_proof=poolproof,
        metrics=metrics, training_question_fit_measurement=True,
        independent_validation=False, online_training_batches=False,
        models_two_Adam_RNG_native_buffers_unchanged=True,
        raw_training_and_EMA_evaluation_distinct=True,
        labels_used_to_select=False, threshold_fitted=False,
        final4096_scored=False, trajectory_accuracy_produced=False,
        feature_reader_reused=True, feature_bank_reopened=False,
        completed_epoch_time=time.time())
    u.atomic(done, result)
    return result
