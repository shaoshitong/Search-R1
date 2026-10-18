"""Explicit two-arm selection-probe policy; candidate/actor science stays separate."""
import random
from collections import Counter
from .contract import WORLD

FROZEN = 'warmup_frozen'
BALANCED = 'balanced_online_ema'
COMPLETE = 'BALANCED_CURRENT_OUTER_PROBE_COMPLETE'


def mode(config):
    return config.get('selection_probe_experiment')


def validate(config):
    arm = mode(config)
    if arm not in (FROZEN, BALANCED):
        raise ValueError('Explicit frozen or balanced-EMA experiment required')
    expected = dict(probe_learning_rate=5e-6, probe_group_loss_weight=5. if arm == BALANCED else 1.,
                    probe_ema_rate=.99 if arm == BALANCED else None)
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('AB learning rate/loss/EMA differs from human protocol')
    slot = list(range(8)) if WORLD == 8 and arm == BALANCED else ([0, 1, 2, 3] if arm == FROZEN else [4, 5, 6, 7])
    if config['physical_gpu_indices'] != slot or config.get('world_size', WORLD) != WORLD:
        raise ValueError('AB arm must own its assigned four-card slot')
    return config


def updates(outer):
    from .periodic import is_probe_update
    return 50 if is_probe_update(outer) else 10


def validate_receipt(config, receipt, outer):
    """Actual writes may be absent only for an explicitly recorded missing class."""
    arm = mode(config)
    expected_outer = 0 if arm == FROZEN else outer
    counts = receipt['actual_adam_steps_per_rank']
    if (receipt['completed_outer'] != expected_outer or len(counts) != WORLD
            or len(set(counts)) != 1 or type(counts[0]) is not int or counts[0] < 0):
        raise ValueError('AB actual checkpoint boundary differs')
    if arm == FROZEN and counts != [0]*WORLD:
        raise ValueError('Frozen arm cannot have online Adam writes')
    if arm == BALANCED and receipt.get('ema_updates') != counts[0]:
        raise ValueError('EMA advances once per actual Adam write')
    return dict(last_update_outer=expected_outer, actual_adam_per_rank=counts[0])


def balanced_plan(labels, seed, outer):
    """One fixed-size resampled pass; keep every authentic G5 together."""
    from .probe_plan import _natural_plan
    n = len(labels)
    if n not in (512, 2560) or any(type(x) is not int or x not in (0, 1) for x in labels):
        raise ValueError('Expected current original512/2560 binary group labels')
    classes = [[i for i, value in enumerate(labels) if value == cls] for cls in (0, 1)]
    rng = random.Random(seed + outer)
    before = rng.getstate()
    if any(not rows for rows in classes):
        return [], dict(status='SKIPPED_MISSING_CLASS', original_counts=list(map(len, classes)),
                        sampled_indices=[], rng_before=before, rng_after=rng.getstate())
    sampled = [rng.choice(rows) for rows in classes for _ in range(n//2)]
    rng.shuffle(sampled)
    plan = _natural_plan(seed, outer, n, n//512*10)
    for batch in plan:
        batch['question_indices'] = [sampled[i] for i in batch['question_indices']]
    frequencies = Counter(sampled)
    return plan, dict(status='BALANCED_RESAMPLED_ONE_PASS', original_counts=list(map(len, classes)),
        sampled_counts=[n//2, n//2], sampled_indices=sampled,
        group_multiplicities={str(i): frequencies[i] for i in range(n)},
        unique_questions=len(frequencies), repeated_fraction=1-len(frequencies)/n,
        rng_before=before, rng_after=rng.getstate())


def ema_step(ema, raw, rate=.99):
    """Update all floating state; synchronize integer state without interpolation."""
    import torch
    with torch.no_grad():
        target, source = ema.state_dict(), raw.state_dict()
        if target.keys() != source.keys():
            raise ValueError('EMA and online architectures differ')
        for key, value in target.items():
            if value.is_floating_point():
                value.mul_(rate).add_(source[key], alpha=1-rate)
            else:
                value.copy_(source[key])

        # Nonpersistent native RoPE buffers do not appear in state_dict.
        persistent = set(target)
        raw_buffers = dict(raw.named_buffers())
        for key, value in ema.named_buffers():
            if key in persistent:
                continue
            if value.is_floating_point():
                value.mul_(rate).add_(raw_buffers[key], alpha=1-rate)
            else:
                value.copy_(raw_buffers[key])
