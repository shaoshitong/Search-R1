"""Four or eight physical ranks preserve the reference eight-rank objective.

The native sampling journal supplies old policy probabilities, including the
scored prefixes. Grounding flip/boundary supplies the unchanged soft reward;
the correctness prefix probe is used neither as reward nor as an old policy.
Every completed phase processes twenty opportunities and advances one outer scheduler step.
Ordinary guards retain actual partial Adam writes and skip rejected opportunities.
HF exports retain FP32 master values, rather than quantizing resume state to BF16.
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import uuid

from .contract import SAMPLING_CONTRACT, WORLD
from .actor_safety import (REFERENCE_MICRO, REFERENCE_MICROS_PER_VIRTUAL_MINI,
    POLICY_SAFEGUARDS, FrozenPolicyMonitor, frozen_monitor_ids, opportunity_result, attempt_result)

REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
MILESTONES = (5, 50, 100, 150, 200)
VIRTUAL_WORLD = 8
VIRTUAL_MINI = 16
VIRTUAL_PER_RANK = VIRTUAL_WORLD // WORLD
OBJECTIVE = 'mean_of_eight_virtual_ranks_each_mean_of_eight_micro2_masked_token_means'
ZERO_ANCHOR_MODE = 'zero_anchor_all_correct_only'
ADVANTAGE_EPSILON = 1e-6
GROUNDING_ARCHIVE = ('/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012/'
    'artifacts/round_0-attempt-22c55c34f646454f9c1f158ebdeb59c2/version-manifest.json')
GROUNDING_ARCHIVE_SHA256 = '924e5487d19cf667dee56ae0a76eba7143cd6eb6325f1024989bac5d05efbb61'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def remote(path):
    path = Path(path).resolve()
    if not path.is_absolute() or not str(path).startswith(('/mnt/', '/dataset/', '/data/')):
        raise ValueError('Actor weights, optimizer, logs and trajectories must remain remote')
    return path


def atomic_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(str(path))
    partial = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    with partial.open('xb') as handle:
        handle.write(canonical(value) + b'\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(partial, path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def append(path, value):
    with Path(path).open('ab') as handle:
        handle.write(canonical(value) + b'\n'); handle.flush(); os.fsync(handle.fileno())


@dataclass(frozen=True)
class ActorBudget:
    # Explicitly requested global micro; a local forward never crosses a
    # reference virtual-rank16 boundary. Receipts expose any effective cap.
    global_micro: int = 16
    global_mini: int = 128
    global_trajectories: int = 2560
    world: int = WORLD
    ppo_epochs: int = 1

    def __post_init__(self):
        if (self.world != WORLD or self.global_mini != 128 or self.global_trajectories != 2560
                or self.ppo_epochs != 1 or type(self.global_micro) is not int
                or self.global_micro < WORLD or self.global_mini % self.global_micro
                or self.global_micro % WORLD):
            raise ValueError('Keep the configured physical/8virtual ranks, mini128, one pass,20opportunities; micro must divide mini128')

    @property
    def effective_global_micro(self):
        return min(self.global_micro, WORLD * VIRTUAL_MINI)

    @property
    def local_micro(self):
        return self.effective_global_micro // self.world

    @property
    def accumulation(self):
        return self.global_mini // self.effective_global_micro


def learning_rate(completed_outer):
    if type(completed_outer) is not int or not 0 <= completed_outer <= 200:
        raise ValueError('Completed outer is outside fresh200')
    return 1e-6 * min(completed_outer / 57, 1.)


def oom_retry_plan(*, outer, failed_micro, latest_complete_outer, rollout_actor_version,
                   latest_actor_version, partial_writes):
    """Actual OOM controller input; unsaved partial writes are never stitched in.

    Call only after the failed native process group has exited. Retain its rank
    ledgers and explicitly account for discarded *unsaved* writes. Reload the
    preceding complete state under a NEW ID; this is recovery with lost updates,
    never a claim that the failed process's real Adam writes were rolled back.
    """
    budget = ActorBudget(failed_micro)
    if (type(outer) is not int or not 1 <= outer <= 200 or latest_complete_outer != outer - 1
            or rollout_actor_version != latest_actor_version or len(partial_writes) != WORLD
            or any(type(n) is not int or not 0 <= n <= 20 for n in partial_writes)):
        raise ValueError('OOM recovery must use the same rollout and latest complete actor boundary')
    if budget.local_micro == 1:
        return dict(status='OOM_AT_MINIMUM_MICRO', retry=False,
            observed_unsaved_partial_writes=partial_writes, abandoned_partial_writes=partial_writes,
            partial_updates_occurred=any(partial_writes), automatic_rollback_performed=False)
    next_budget = ActorBudget(budget.effective_global_micro // 2)
    return dict(status='RETRY_SAME_CACHED_OUTER_FROM_LAST_COMPLETE', retry=True,
        restore_completed_outer=latest_complete_outer, new_attempt_id=uuid.uuid4().hex,
        actor_global_micro_batch_size=next_budget.global_micro,
        actor_local_micro_batch_size=next_budget.local_micro,
        gradient_accumulation=next_budget.accumulation,
        abandoned_partial_writes=partial_writes, stitch_partial_writes=False,
        observed_unsaved_partial_writes=partial_writes,
        partial_updates_occurred=any(partial_writes), lost_unsaved_updates_on_reload=partial_writes,
        automatic_rollback_performed=False, failed_requested_global_micro=failed_micro,
        failed_effective_global_micro=budget.effective_global_micro,
        rollout_actor_version=rollout_actor_version, bitwise_continuation_claimed=False)


def collect_oom_retry(output, config_path, outer, rollout_path, attempt_id, *, phase_exit_code):
    """Parent CPU API after torchrun and all its owned ranks have actually exited.

    Supply the exact32hex ID set in that launch's config.actor_attempt_id. Only
    that attempt's files are read; stale failures from prior retries are ignored.
    A peer NCCL error does not hide another rank's typed actual CUDA OOM. The
    result distinguishes ledger-confirmed writes from uncommitted optimizer work.
    It performs no process signalling, deployment, GPU call or state mutation.
    """
    output, config_path, rollout_path = map(Path, (output, config_path, rollout_path))
    config, rollout = read(config_path), read(rollout_path)
    if (not isinstance(attempt_id, str) or not re.fullmatch('[0-9a-f]{32}', attempt_id)
            or config.get('actor_attempt_id') != attempt_id or not isinstance(phase_exit_code, int)
            or phase_exit_code == 0 or rollout['run_id'] != config['run_id'] or rollout['outer'] != outer):
        raise ValueError('Require the actual failed phase exit and its exact launch attempt ID')
    directory = output / 'outers' / ('outer-%04d-attempt-%s' % (outer, attempt_id))
    latest = read(output / 'latest.json') if (output / 'latest.json').exists() else None
    if latest is None:
        if outer != 1: raise ValueError('Missing preceding complete actor for OOM recovery')
        completed, version = 0, config['run_id'] + '-official0'
    else:
        if latest['completed_outer'] == outer:
            # A launcher failure after state publication must not run the20 steps again.
            if (latest.get('run_id') != config['run_id'] or latest.get('binding') != _state_binding(config)
                    or not valid_actual_steps(latest.get('actual_adam_steps_per_rank'), outer)
                    or latest.get('scheduler_completed_outer') != outer
                    or latest.get('original_rollout_sha256') != sha(rollout_path)):
                raise ValueError('Published actor completion does not match this failed launch input')
            return dict(status='ACTOR_ALREADY_COMMITTED', retry=False,
                        completed_outer=outer, actor_version=latest['actor_version'])
        restore_metadata(config, latest, outer)
        completed, version = latest['completed_outer'], latest['actor_version']
    completed_steps = latest['actual_adam_steps_per_rank'][0] if latest else 0
    failures, ledgers, counts, missing = [], [], [], []
    for rank in range(WORLD):
        ledger = directory / ('rank-%d-updates.jsonl' % rank)
        records = []
        if ledger.exists():
            lines = ledger.read_text().splitlines()
            for line_index, line in enumerate(lines):
                try: records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A killed process can leave only its final uncommitted line incomplete.
                    if line_index != len(lines) - 1: raise
            for row in records:
                if row.get('rank') != rank or row.get('outer') != outer:
                    raise ValueError('Failed-attempt ledger belongs to another rank or outer')
            writes = [row for row in records if row['event'] == 'ACTUAL_ADAM_WRITE']
            for index, row in enumerate(writes):
                if (row['rank'] != rank or row['outer'] != outer or type(row['opportunity']) is not int or not 0 <= row['opportunity'] < 20
                        or (index and row['opportunity'] <= writes[index - 1]['opportunity'])
                        or row['writes_this_attempt'] != index + 1
                        or row['actual_adam_steps'] != completed_steps + index + 1):
                    raise ValueError('Failed-attempt ledger counters are not sequential actual writes')
            count = len(writes)
            ledgers.append(dict(rank=rank, path=str(ledger), sha256=sha(ledger), confirmed_writes=count))
        else:
            count = 0; missing.append(rank)
        counts.append(count)
        paths = [directory / ('rank-%d-failure.json' % rank),
                 output / ('native-failure-outer-%04d-attempt-%s-rank-%d.json' % (outer, attempt_id, rank))]
        for path in paths:
            if not path.exists(): continue
            failure = read(path)
            if (failure.get('rank') != rank or failure.get('outer') != outer
                    or failure.get('attempt_id') != attempt_id or failure.get('run_id') != config['run_id']
                    or Path(failure['cached_rollout_path']).resolve() != rollout_path.resolve()):
                raise ValueError('Failed native actor evidence belongs to another launch')
            failures.append(dict(path=str(path), sha256=sha(path), value=failure))
    actual_oom = any(item['value'].get('actual_cuda_oom') is True for item in failures)
    if not actual_oom:
        return dict(status='NON_OOM_FAILURE', retry=False, failures=failures, ledgers=ledgers,
                    observed_committed_partial_writes=counts, ranks_without_ledger=missing)
    plan = oom_retry_plan(outer=outer, failed_micro=int(config.get('actor_global_micro_batch_size', 16)),
        latest_complete_outer=completed, rollout_actor_version=rollout['actor_version'],
        latest_actor_version=version, partial_writes=counts)
    plan.update(attempt_id=attempt_id, failed_attempt_dir=str(directory), failures=failures,
        ledgers=ledgers, ranks_without_ledger=missing, observed_committed_partial_writes=counts,
        uncommitted_or_interrupted_optimizer_work_count_unknown=True,
        cached_rollout_sha256=sha(rollout_path), latest_completed_outer=completed)
    return plan


def _tokens(value, field):
    if not isinstance(value, list) or any(type(token) is not int or token < 0 for token in value):
        raise ValueError('Invalid actual token IDs: ' + field)
    return value


def validate_record(row, actor_version):
    """Authenticate token/LP/causal identity, without a new numeric risk threshold."""
    if row.get('actor_version') != actor_version or row.get('policy_snapshot_id') != actor_version:
        raise ValueError('Continued trajectory uses a different actor snapshot')
    if row.get('sampling_contract') != SAMPLING_CONTRACT:
        raise ValueError('Selected prefix must retain the truthful chunked native sampling contract')
    prompt = _tokens(row['initial_prompt_token_ids'], 'initial_prompt')
    response = _tokens(row['response_token_ids'], 'response')
    mask, probabilities = row['response_info_mask'], row['response_log_probs']
    if (not prompt or len(response) != len(mask) or len(response) != len(probabilities)
            or any(type(v) is not int or v not in (0, 1) for v in mask)
            or row.get('strict_em') not in (0, 1)
            or type(row.get('strict_em')) is not int):
        raise ValueError('Incomplete actual response, policy mask, probabilities or own strictEM')
    if any(not isinstance(p, (int, float)) or not math.isfinite(p) for p in probabilities):
        raise ValueError('Actual selected native token log-probability is nonfinite')
    rebuilt, rebuilt_mask, rebuilt_lp, chunks = [], [], [], []
    for ordinal, turn in enumerate(row['behavior_turns']):
        request = _tokens(turn['actual_request_token_ids'], 'actual_request')
        if request != prompt + rebuilt or turn['turn_index'] != ordinal:
            raise ValueError('A logical action did not use its true causal context')
        raw, lp = [], []
        native_chunks = turn['actual_sampling_chunks']
        if not native_chunks:
            raise ValueError('A retained policy action lacks native sampling chunks')
        for chunk in native_chunks:
            if chunk['actual_request_token_ids'] != request + raw:
                raise ValueError('Continuation did not continue the same actual prefix/context')
            ids = _tokens(chunk['raw_generated_token_ids'], 'native_chunk')
            values = chunk['selected_logprobs']
            sampling = chunk['sampling']
            if (len(ids) != len(values) or len(ids) > sampling['max_tokens']
                    or sampling['temperature'] != 1. or sampling['top_p'] != 1.
                    or sampling['top_k'] != -1 or sampling.get('n', 1) != 1):
                raise ValueError('Native selected probabilities or sampler recipe differ')
            raw.extend(ids); lp.extend(values); chunks.append(chunk)
        n = turn['kept_token_count']
        if (type(n) is not int or not 0 <= n <= len(raw) or n > 500
                or raw != turn['raw_generated_token_ids'] or lp != turn['selected_logprobs']
                or turn['prefix_cut']['kept_token_ids'] != raw[:n]):
            raise ValueError('Logical action is not the recorded native kept prefix')
        observation = _tokens(turn['observation_token_ids'], 'observation')
        correction = _tokens(turn['correction_token_ids'], 'correction')
        if observation and correction:
            raise ValueError('Environment event cannot be both retrieval and correction')
        environment = observation + correction
        rebuilt.extend(raw[:n] + environment)
        rebuilt_mask.extend([1] * n + [0] * len(environment))
        rebuilt_lp.extend(lp[:n] + [0.] * len(environment))
    if (response != rebuilt or mask != rebuilt_mask or probabilities != rebuilt_lp
            or row['sampling_journal'] != chunks):
        raise ValueError('PPO input changed policy tokens, observations, oldLP or sampling lineage')
    if 'ground_truth' not in row.get('reward_model', {}) or not row.get('data_source'):
        raise ValueError('Original gold and data_source must feed the unchanged grounding reward')
    return row


def load_rollout(path, config, outer, actor_version):
    path = remote(path); complete = read(path)
    expected = dict(run_id=config['run_id'], outer=outer, actor_version=actor_version,
                    questions=512, trajectories=2560)
    if (complete.get('phase') not in ('continue', 'full')
            or any(complete.get(key) != value for key, value in expected.items())):
        raise ValueError('Completed selected rollout identity/count differs')
    rows = []
    for item in complete['shards']:
        shard = remote(item['path'])
        if shard.stat().st_size != item['bytes'] or sha(shard) != item['sha256']:
            raise ValueError('Committed native rollout shard differs')
        opener = gzip.open if shard.suffix == '.gz' else open
        with opener(shard, 'rt', encoding='utf8') as handle:
            local = [validate_record(json.loads(line), actor_version) for line in handle]
        if len(local) != item['count']:
            raise ValueError('Native shard trajectory count differs')
        rows.extend(local)
    groups, seen = defaultdict(list), set()
    for row in rows:
        if row['trajectory_id'] in seen:
            raise ValueError('Duplicate selected trajectory identity')
        seen.add(row['trajectory_id']); groups[str(row['qid'])].append(row)
    if len(rows) != 2560 or len(groups) != 512:
        raise ValueError('Selected actor batch is not512 completeG5')
    for siblings in groups.values():
        if ({row['response_index'] for row in siblings} != set(range(5)) or len(siblings) != 5
                or len({row['candidate_ordinal'] for row in siblings}) != 1
                or len({canonical(row['reward_model']) for row in siblings}) != 1
                or len({row['data_source'] for row in siblings}) != 1
                or len({canonical(row['initial_prompt_token_ids']) for row in siblings}) != 1):
            raise ValueError('Original selected G5 identity changed')
    if len({row['candidate_ordinal'] for row in rows}) != 512:
        raise ValueError('Selected question candidate ordinals repeat')
    selection_path = remote(complete['actor_selection_path'])
    selection_sha = complete['actor_selection_sha256']
    if (re.fullmatch('[0-9a-f]{64}', selection_sha) is None
            or selection_sha != complete['selection_sha256'] or sha(selection_path) != selection_sha):
        raise ValueError('Actual actor selection path/SHA differs from the completed own rollout')
    validate_actor_selection(complete, rows, read(selection_path))
    rows.sort(key=lambda row: (row['candidate_ordinal'], row['response_index']))
    return rows, complete


def validate_actor_selection(complete, rows, selection):
    """Bind PPO to the actual final512 G5, including EM-selected update rounds."""
    if (any(selection.get(key) != complete[key] for key in ('run_id', 'outer', 'actor_version'))
            or selection.get('prefix_scan_sha256') != complete['prefix_scan_sha256']
            or selection.get('selected_questions') != 512 or len(selection.get('selected', [])) != 512):
        raise ValueError('Actual actor selection is not this completed actor/prefix512 boundary')
    chosen = {}
    qids, ordinals = set(), set()
    for group in selection['selected']:
        if (group['qid'] in qids or group['candidate_ordinal'] in ordinals
                or len(group['trajectory_ids']) != 5 or len(set(group['trajectory_ids'])) != 5
                or any(tid in chosen for tid in group['trajectory_ids'])):
            raise ValueError('Actual actor selection repeats or omits an original own G5 identity')
        qids.add(group['qid']); ordinals.add(group['candidate_ordinal'])
        chosen.update({tid: (str(group['qid']), group['candidate_ordinal']) for tid in group['trajectory_ids']})
    lineage = ('full_pool_origin_selection_sha256', 'full_pool_complete_sha256',
               'full_pool_own_em_sha256', 'predicted_selection_sha256') if complete.get('projected_from_full_pool') else ()
    if any(not isinstance(complete.get(key), str) or re.fullmatch('[0-9a-f]{64}', complete[key]) is None
           for key in lineage):
        raise ValueError('Projected actor selection must retain the truthful full-pool and predicted-selection lineage')
    observed = set()
    for row in rows:
        if (row['trajectory_id'] in observed
                or chosen.get(row['trajectory_id']) != (str(row['qid']), row['candidate_ordinal'])
                or row.get('selection_sha256') != complete['actor_selection_sha256']
                or row.get('prefix_scan_sha256') != complete['prefix_scan_sha256']
                or any(row.get(key) != complete[key] for key in lineage)):
            raise ValueError('Actor row does not belong to the actual final selection and native prefix')
        observed.add(row['trajectory_id'])
    if observed != set(chosen):
        raise ValueError('Actual final actor selection does not have exactly its own complete512G5')
    return True


def tensor_batch(rows, pad_token_id):
    """No text round-trip: native prompts/policy/observations form the PPO inputs."""
    import torch
    pwidth = max(len(row['initial_prompt_token_ids']) for row in rows)
    # A real first-token EOS has zero retained policy tokens. Keep its G5 row;
    # the single storage pad below has attention/lossmask0 and is never authored.
    rwidth = max(1, max(len(row['response_token_ids']) for row in rows))
    prompts = torch.full((len(rows), pwidth), pad_token_id, dtype=torch.long)
    responses = torch.full((len(rows), rwidth), pad_token_id, dtype=torch.long)
    pa, ra, policy = torch.zeros_like(prompts), torch.zeros_like(responses), torch.zeros_like(responses)
    old = torch.zeros((len(rows), rwidth), dtype=torch.float32)
    for i, row in enumerate(rows):
        p, r = row['initial_prompt_token_ids'], row['response_token_ids']
        prompts[i, -len(p):] = torch.tensor(p); pa[i, -len(p):] = 1
        responses[i, :len(r)] = torch.tensor(r); ra[i, :len(r)] = 1
        policy[i, :len(r)] = torch.tensor(row['response_info_mask'])
        old[i, :len(r)] = torch.tensor(row['response_log_probs'], dtype=torch.float32)
    attention = torch.cat((pa, ra), dim=1)
    return dict(prompts=prompts, responses=responses, input_ids=torch.cat((prompts, responses), dim=1),
        attention_mask=attention, position_ids=(attention.cumsum(-1) - 1) * attention,
        info_mask=torch.cat((pa, policy), dim=1), loss_mask=policy,
        old_log_probs=old, sampling_log_probs=old.clone(), sampling_mask=policy.bool())


def to_dataproto(rows, tokenizer):
    import numpy as np
    from verl import DataProto
    turns = np.empty(len(rows), dtype=object); turns[:] = [row['behavior_turns'] for row in rows]
    return DataProto.from_dict(tensor_batch(rows, tokenizer.pad_token_id), non_tensors={
        'uid': [str(row['qid']) for row in rows], 'index': [str(row['qid']) for row in rows],
        'reward_model': [row['reward_model'] for row in rows],
        'data_source': [row['data_source'] for row in rows],
        'behavior_trajectory_id': [row['trajectory_id'] for row in rows], 'behavior_turns': turns})


REWARD_ENV = dict(ECO_TRUST_VARIANT='balanced', ECO_TRUST_SIGNAL='probe', ECO_PROBE_GATE='1',
    ECO_BAL_GROUND='0.6', ECO_BAL_NOSEARCH_BONUS='0.5', ECO_W_COST='0.02',
    ECO_WRONG_GROUND_SCALE='0', ECO_WRONG_FLOOR='0.05', ECO_PROBE_BASE='0',
    ECO_PROBE_ALPHA_COST='0', ECO_USE_CF='0', ECO_NO_SELF='1', ECO_PROBE_MAX_WAIT='300')


def advantage_contract(config):
    """The human-selected zero anchor is a new, explicit scientific contract."""
    if config.get('nonmixed_advantage_mode') != ZERO_ANCHOR_MODE:
        raise ValueError('Explicit zero_anchor_all_correct_only advantage contract is required')
    return dict(mode=ZERO_ANCHOR_MODE, classification='actual_native_strict_em_G5',
        anchored_group='all_five_correct', all_wrong_and_mixed='original_G5_normalization',
        anchor_value=0., anchor_in_statistics_only=True, synthetic_actor_trajectory=False,
        standard_deviation_correction=1, epsilon=ADVANTAGE_EPSILON,
        raw_grounding_rewards_modified=False)


def normalize_reward_group(rewards, *, zero_anchor=False):
    """Return only real advantages; an optional0 affects statistics, not data.

    This small formula helper also accepts the user's three-reward illustration.
    The actor caller below still requires the actual complete five trajectories.
    """
    import torch
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.ndim != 1 or values.numel() < 2 or not bool(torch.isfinite(values).all()):
        raise ValueError('Group rewards must be a finite one-dimensional sample with at least two values')
    if type(zero_anchor) is not bool:
        raise ValueError('Zero-anchor selection must be an explicit Boolean')
    raw_mean, raw_std = values.mean(), values.std(correction=1)
    statistics = torch.cat((values.new_zeros(1), values)) if zero_anchor else values
    mean, std = statistics.mean(), statistics.std(correction=1)
    if not bool(torch.isfinite(torch.stack((raw_mean, raw_std, mean, std))).all()):
        raise FloatingPointError('Actual reward-group normalization statistics are nonfinite')
    advantages = (values - mean) / (std + ADVANTAGE_EPSILON)
    return advantages, dict(raw_reward_count=values.numel(), raw_reward_mean=float(raw_mean),
        raw_reward_sample_std=float(raw_std), normalization_count=statistics.numel(),
        normalization_mean=float(mean), normalization_sample_std=float(std),
        virtual_zero_in_statistics=zero_anchor, virtual_zero_in_actor_batch=False,
        advantage_count=values.numel(), advantage_mean=float(advantages.mean()),
        advantage_min=float(advantages.min()), advantage_max=float(advantages.max()))


def group_advantages(rows, scores, *, mode='original_g5', return_diagnostics=False):
    import torch
    scores = torch.as_tensor(scores, dtype=torch.float32)
    if scores.ndim != 1 or len(rows) != scores.numel() or not bool(torch.isfinite(scores).all()):
        raise ValueError('Actual rows and finite scalar grounding rewards must align')
    if mode not in ('original_g5', ZERO_ANCHOR_MODE):
        raise ValueError('Unknown GRPO advantage normalization mode')
    advantages = torch.empty_like(scores)
    grouped = defaultdict(list)
    for i, row in enumerate(rows): grouped[str(row['qid'])].append(i)
    diagnostics = []
    for qid, ids in grouped.items():
        if len(ids) != 5: raise ValueError('GRPO advantages require own completeG5 rewards')
        if mode == ZERO_ANCHOR_MODE:
            own_em = [rows[i].get('strict_em') for i in ids]
            if (any(type(value) is not int or value not in (0, 1) for value in own_em)
                    or any(type(rows[i].get('response_index')) is not int for i in ids)
                    or {rows[i].get('response_index') for i in ids} != set(range(5))):
                raise ValueError('Zero-anchor classification requires the actual complete native G5 strictEM labels')
            correct = sum(own_em)
            kind = 'all_correct' if correct == 5 else 'all_wrong' if correct == 0 else 'mixed'
        else:
            correct, kind = None, 'original_g5'
        local, statistics = normalize_reward_group(scores[ids], zero_anchor=(kind == 'all_correct'))
        advantages[ids] = local
        diagnostics.append(dict(qid_sha256=hashlib.sha256(qid.encode()).hexdigest(),
            actual_correct_count=correct, group_kind=kind, **statistics))
    if not return_diagnostics:
        return advantages
    counts = {kind: sum(row['group_kind'] == kind for row in diagnostics)
              for kind in ('all_correct', 'all_wrong', 'mixed', 'original_g5')}
    summary = dict(mode=mode, actual_groups=len(grouped), actual_trajectories=len(rows),
        actual_group_counts=counts,
        virtual_zero_statistics_groups=sum(row['virtual_zero_in_statistics'] for row in diagnostics),
        synthetic_actor_trajectories=0, standard_deviation_correction=1, epsilon=ADVANTAGE_EPSILON,
        raw_rewards_modified=False, per_type_statistics={})
    for kind, count in counts.items():
        if not count: continue
        selected = [row for row in diagnostics if row['group_kind'] == kind]
        summary['per_type_statistics'][kind] = {}
        for key in ('raw_reward_mean', 'raw_reward_sample_std', 'normalization_mean',
                    'normalization_sample_std', 'advantage_mean'):
            values = [row[key] for row in selected]
            summary['per_type_statistics'][kind][key] = dict(min=min(values),
                mean=math.fsum(values) / len(values), max=max(values))
    return advantages, dict(summary=summary, groups=diagnostics)


def compute_soft_rewards(rows, tokenizer, config, rank=0):
    """One original EcoRewardManager call; all new prefix-head values stay separate."""
    import torch
    os.environ.update(REWARD_ENV)
    os.environ['PROBE_URL'] = ('http://127.0.0.1:%d/judge_probe' % (int(config['grounding_probe_base_port']) + rank)
        if 'grounding_probe_base_port' in config else config.get('grounding_probe_url', 'http://127.0.0.1:18012/judge_probe'))
    from verl.trainer.main_ppo_eco import EcoRewardManager
    manager = EcoRewardManager(tokenizer, num_examine=0, reward_mode='eco')
    manager.expected_probe_version = config['grounding_probe_version']
    data = to_dataproto(rows, tokenizer)
    scores = manager(data).sum(-1).float()
    if manager.last_probe_version != config['grounding_probe_version']:
        raise ValueError('Grounding reward probe version differs')
    return scores, dict(version=manager.last_probe_version, **_grounding_receipt(config),
        rank=rank, probe_url=os.environ['PROBE_URL'],
        reward_environment=REWARD_ENV, score_mean=float(scores.mean()), score_min=float(scores.min()),
        score_max=float(scores.max()), prefix_correctness_head_used_for_reward=False)


def rank_plan(seed, outer, world=WORLD):
    """Each physical rank owns its adjacent original virtual16 minibatches."""
    if world != WORLD or not 1 <= outer <= 200: raise ValueError('Require configured physical ranks and outer1..200')
    order = list(range(2560)); random.Random(seed + outer).shuffle(order)
    width = 128 // world
    return [[order[step * 128 + rank * width:step * 128 + (rank + 1) * width]
             for step in range(20)] for rank in range(world)]


def virtual_minibatches(ids):
    """Preserve two virtual ranks and their eight reference micro2 means each."""
    if len(ids) != VIRTUAL_MINI * VIRTUAL_PER_RANK:
        raise ValueError('A physical minibatch must contain two complete virtual16 groups')
    return [ids[start:start + VIRTUAL_MINI] for start in range(0, len(ids), VIRTUAL_MINI)]


def masked_policy_objective(old, current, advantages, mask, *, loss_fn):
    """A legal all-empty policy micro contributes zero, while joining backward."""
    if not bool(mask.bool().any()):
        zero = current.masked_fill(~mask.bool(), 0.).sum() * 0.
        return zero, zero.detach(), zero.detach()
    return loss_fn(old, current, advantages, mask, cliprange=.2, dual_clip=3.)


def micro_loss_weight(micro_policy_tokens, local_minibatch_policy_tokens):
    """A split reference micro2 retains its parent two-row token denominator."""
    if (type(micro_policy_tokens) is not int or type(local_minibatch_policy_tokens) is not int
            or not 0 <= micro_policy_tokens <= local_minibatch_policy_tokens):
        raise ValueError('Actual authored-token mass differs from its parent minibatch')
    return micro_policy_tokens / local_minibatch_policy_tokens if local_minibatch_policy_tokens else 0.


def reference_policy_objective(old, current, advantages, mask, *, virtual_offset,
                               virtual_pair_token_masses, loss_fn):
    """Large forward, original micro2 means; OOM micro1 keeps pair denominators.

    The caller sums these losses over both virtual16 blocks. FSDP's four-rank
    average then yields (1/64) sum of all64 reference micro2 masked means.
    """
    if (type(virtual_offset) is not int or virtual_offset < 0 or
            virtual_offset + len(current) > VIRTUAL_MINI or
            len(virtual_pair_token_masses) != REFERENCE_MICROS_PER_VIRTUAL_MINI):
        raise ValueError('A physical forward must retain its original virtual16/micro2 identity')
    loss = current.masked_fill(~mask.bool(), 0.).sum() * 0.
    metrics = []
    offset = 0
    while offset < len(current):
        pair = (virtual_offset + offset) // REFERENCE_MICRO
        end = min(len(current), offset + REFERENCE_MICRO - (virtual_offset + offset) % REFERENCE_MICRO)
        part = slice(offset, end)
        value, clipfrac, kl = masked_policy_objective(old[part], current[part], advantages[part], mask[part],
                                                     loss_fn=loss_fn)
        mass = int(mask[part].sum())
        weight = micro_loss_weight(mass, virtual_pair_token_masses[pair]) / (
            REFERENCE_MICROS_PER_VIRTUAL_MINI * VIRTUAL_PER_RANK)
        loss = loss + value * weight
        metrics.append(dict(loss=float(value.detach()), clip_fraction=float(clipfrac.detach()),
            old_policy_kl=float(kl.detach()), actual_policy_token_mass=mass,
            reference_micro2_index=pair, reference_micro2_policy_token_mass=virtual_pair_token_masses[pair],
            gradient_weight=weight))
        offset = end
    return loss, metrics


def valid_actual_steps(values, completed_outer):
    """One to twenty real writes per completed20-opportunity outer, equal on ranks."""
    return (isinstance(values, list) and len(values) == WORLD and
            all(type(value) is int and completed_outer <= value <= 20 * completed_outer for value in values) and
            len(set(values)) == 1)


def _rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state())


def _set_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu']); torch.cuda.set_rng_state(state['torch_cuda'])



def _grounding_attempt(config, latest, outer):
    """Use the exact adopted evaluator bound by this attempt's launcher config."""
    from .grounding import checked_state, validate_actor_transition
    if (config.get('grounding_refresh_enabled') is not True or
            not isinstance(config.get('grounding_state'), dict) or
            type(outer) is not int or not 1 <= outer <= 200):
        raise ValueError('Fresh actor requires its actual adopted dynamic grounding state')
    supplied = config['grounding_state']
    state = checked_state(config, supplied)
    if (state != supplied or state['round'] != min((outer - 1) // 50 * 50, 150) or
            state['version'] != config['grounding_probe_version'] or
            state['manifest_path'] != config['grounding_current_manifest']):
        raise ValueError('Actor attempt grounding version, manifest or scheduled round differs')
    transition = validate_actor_transition(config, latest, outer)
    # Preserve the complete exact checked state; do not reconstruct a lossy subset.
    return json.loads(canonical(state)), transition


def _grounding_receipt(config):
    state = config['grounding_state']
    return dict(grounding_state=state, grounding_probe_version=state['version'],
        grounding_current_manifest=state['manifest_path'], grounding_manifest=state['manifest_path'],
        grounding_manifest_sha256=state['manifest_sha256'])

def _state_binding(config):
    # Micro is intentionally mutable after a real OOM; the scientific budget is not.
    return dict(run_id=config['run_id'], model_revision=config.get('model_revision', REVISION),
        source_commit=config.get('source_commit'), model_files_sha256=config.get('model_files_sha256'),
        canonical_train=config.get('canonical_train'), canonical_train_sha256=config.get('canonical_train_sha256'),
        source_question_pool_sha256=config.get('source_question_pool_sha256'),
        common_manifest_sha256=config.get('common_manifest_sha256'),
        composite_manifest_sha256=config.get('composite_manifest_sha256'),
        official_model_dir=str(remote(config['official_model_dir'])), world_size=WORLD,
        seed=int(config.get('seed', 1729)),
        global_mini=128, ppo_epochs=1, trajectories=2560, updates_per_outer=20,
        learning_rate=1e-6, outer_warmup=57, clip_ratio=.2, dual_clip=3., grad_clip=1.,
        loss_reduction=OBJECTIVE, reference_virtual_world=VIRTUAL_WORLD,
        reference_micro=REFERENCE_MICRO, reference_micros_per_virtual_mini=REFERENCE_MICROS_PER_VIRTUAL_MINI,
        policy_safeguards=POLICY_SAFEGUARDS, policy_guard_accounting='20opportunities_at_least1write_no_rollback',
        reference_virtual_local_mini=VIRTUAL_MINI, virtual_blocks_per_physical_rank=VIRTUAL_PER_RANK,
        entropy_coefficient=0., reference_kl_coefficient=0., reference_kl_type='low_var_kl',
        advantage_normalization=advantage_contract(config),
        grounding_lifecycle=config.get('grounding_state', {}).get('lifecycle'),
        reward_environment=REWARD_ENV)


def restore_metadata(config, latest, outer):
    expected = outer - 1
    if (latest['run_id'] != config['run_id'] or latest['completed_outer'] != expected
            or latest['binding'] != _state_binding(config)
            or not valid_actual_steps(latest['actual_adam_steps_per_rank'], expected)
            or latest['scheduler_completed_outer'] != expected):
        raise ValueError('Restore only the immediately previous complete actor/Adam/scheduler boundary')


def optimizer_layout(optimizer):
    """Rank-local FP32 flat-parameter ordering needed by ordinary Adam load_state_dict."""
    return [[dict(shape=list(parameter.shape), numel=parameter.numel(), dtype=str(parameter.dtype))
             for parameter in group['params']] for group in optimizer.param_groups]


def linked_probe_receipt(value):
    """Capture the actual source probe version now, rather than a mutable latest path."""
    if value is None: return None
    if isinstance(value, str):
        path = remote(value)
        return dict(path=str(path), sha256=sha(path), value=read(path))
    if isinstance(value, dict): return value
    raise ValueError('Prefix probe receipt must be an actual small JSON path or proof')


def _build_actor(model_dir, budget, device):
    import functools
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from verl.utils.rope_precision import cast_preserving_qwen_rope
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    policy = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation='flash_attention_2', trust_remote_code=False)
    # Explicitly pass reset RoPE positions into HF's varlen dispatch. Relying on
    # a version-specific default attention forward could join packed examples.
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    apply_monkey_patch(policy.config, verbose=False)
    cast_preserving_qwen_rope(policy, torch.float32)
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    wrapped = FSDP(policy, device_id=device, use_orig_params=False,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                       buffer_dtype=torch.float32), sync_module_states=True,
        forward_prefetch=False)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-6, betas=(.9, .999), weight_decay=.01)
    # Scheduler construction at0 establishes the original completed-outer clock.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda completed: min(completed / 57, 1.))
    actor = DataParallelPPOActor(OmegaConf.create(dict(compute_dtype='bfloat16', allow_tf32=None,
        use_remove_padding=True, ulysses_sequence_parallel_size=1, entropy_coeff=0.,
        ppo_mini_batch_size=16, ppo_micro_batch_size=budget.local_micro, ppo_epochs=1)), wrapped, optimizer)
    return actor, wrapped, optimizer, scheduler, tokenizer


def _save_complete(config, output, attempt_dir, outer, wrapped, optimizer, scheduler, tokenizer,
                   rollout_path, actor_version, budget, reward_proof, ranks):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    from search_g1.training_state import atomic_save, cpu_tree
    rank = dist.get_rank()
    state = dict(schema_version=1, run_id=config['run_id'], completed_outer=outer, rank=rank,
        actual_adam_steps=ranks[rank]['actual_adam_steps'], scheduler_completed_outer=outer,
        binding=_state_binding(config), **_grounding_receipt(config), optimizer=cpu_tree(optimizer.state_dict()),
        optimizer_parameter_layout=optimizer_layout(optimizer),
        scheduler=cpu_tree(scheduler.state_dict()), rng=_rng(),
        model_master_dtype='torch.float32', optimizer_layout='FSDP_FULL_SHARD_same_Qwen2DecoderLayer_wrap_world%d' % WORLD,
        original_rollout_path=str(rollout_path), original_rollout_sha256=sha(rollout_path))
    state_proof = atomic_save(attempt_dir / ('rank-%d.pt' % rank), state)
    proofs = [None] * WORLD; dist.all_gather_object(proofs, state_proof)
    hfdir = attempt_dir / 'huggingface'
    with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        full = wrapped.state_dict()
    if rank == 0:
        if any(value.is_floating_point() and value.dtype != torch.float32 for value in full.values()):
            raise ValueError('HF resume export must preserve the actual FP32 master model')
        wrapped.module.save_pretrained(hfdir, state_dict=full, safe_serialization=True, max_shard_size='4GB')
        tokenizer.save_pretrained(hfdir)
        hf = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(hfdir.iterdir()) if p.is_file()}
        driver_state = config.get('driver_state', {})
        pool_path = remote(config['pool_state_path']) if config.get('pool_state_path') else None
        pool_proof = dict(path=str(pool_path), sha256=sha(pool_path), value=read(pool_path)) if pool_path else None
        driver_proof = atomic_save(attempt_dir / 'driver.pt', dict(run_id=config['run_id'],
            completed_outer=outer, actor_version=actor_version, driver_state=driver_state,
            **_grounding_receipt(config),
            rng=_rng(),
            rng_scope='actor_phase_rank0; controller candidate RNG/order lives in candidate_pool_state',
            candidate_pool_state=pool_proof, prefix_probe_receipt=linked_probe_receipt(config.get('prefix_probe_receipt'))))
        receipt = dict(schema_version=1, role='prefix_selection200_actor_complete', run_id=config['run_id'],
            status='complete', complete_outer=True, **_grounding_receipt(config),
            source_commit=config.get('source_commit'), completed_outer=outer, actor_version=actor_version,
            model_dir=str(hfdir), actor_model_dir=str(hfdir), hf_model_dir=str(hfdir), model_master_dtype='torch.float32',
            rollout_compute_dtype='bfloat16', state_dir=str(attempt_dir), state_path=str(attempt_dir), rank_states=proofs,
            hf_files=hf, driver_state=driver_proof,
            actual_adam_steps_per_rank=[row['actual_adam_steps'] for row in ranks],
            new_adam_steps_per_rank=[row['new_adam_steps'] for row in ranks], scheduler_completed_outer=outer,
            opportunities_processed=20, full_budget_complete=all(row['full_budget_complete'] for row in ranks),
            skipped_update_opportunities=[row['skipped_update_opportunities'] for row in ranks],
            written_with_post_guard=[row['written_with_post_guard'] for row in ranks],
            automatic_rollback_performed=False,
            learning_rate_used=learning_rate(outer - 1), next_learning_rate=learning_rate(outer),
            requested_global_micro=budget.global_micro,
            actual_global_micro=budget.effective_global_micro, actual_local_micro=budget.local_micro,
            gradient_accumulation=budget.accumulation, global_mini=128, ppo_epochs=1,
            original_rollout_path=str(rollout_path), original_rollout_sha256=sha(rollout_path),
            sampling_actor_version=read(rollout_path)['actor_version'], reward=reward_proof,
            rank_receipts=ranks, binding=_state_binding(config), milestone=outer in MILESTONES,
            saved_utc=utc(), independent_readback_performed=False, joint_outer_complete=False)
        atomic_json(attempt_dir / 'complete.json', receipt)
        atomic_json(output / 'latest.json', receipt, replace=True)
    dist.barrier()


def train(config_path, outer, rollout_path, output_path):
    import numpy as np
    import torch
    import torch.distributed as dist
    from verl.trainer.ppo.core_algos import compute_policy_loss
    from search_g1.gradient_clipping import clip_global_l2
    config = read(remote(config_path)); output = remote(output_path); rollout_path = remote(rollout_path)
    normalization_contract = advantage_contract(config)
    if config['model_revision'] != REVISION or not 1 <= outer <= 200:
        raise ValueError('Fresh official actor200 revision/budget differs')
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world != WORLD: raise ValueError('Actor phase requires the configured physical rank count')
    torch.cuda.set_device(local); dist.init_process_group('nccl')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    budget = ActorBudget(int(config.get('actor_global_micro_batch_size', 16)))
    seed = int(config.get('seed', 1729))
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    output.mkdir(parents=True, exist_ok=True)
    latest = read(output / 'latest.json') if (output / 'latest.json').exists() else None
    if outer == 1:
        if latest is not None: raise ValueError('Fresh official outer1 cannot inherit an actor checkpoint')
        model_dir = str(remote(config['official_model_dir'])); old_version = config['run_id'] + '-official0'
    else:
        if latest is None: raise ValueError('Missing this run latest complete actor state')
        restore_metadata(config, latest, outer)
        model_dir = str(remote(latest['model_dir'])); old_version = latest['actor_version']
    grounding_state, grounding_transition = _grounding_attempt(config, latest, outer)
    config = dict(config, grounding_state=grounding_state)
    grounding_digests = [None] * WORLD
    grounding_digest = hashlib.sha256(canonical(grounding_state)).hexdigest()
    dist.all_gather_object(grounding_digests, grounding_digest)
    if any(digest != grounding_digest for digest in grounding_digests):
        raise ValueError('All four actor ranks must adopt the same exact grounding attempt state')
    supplied_attempt = config.get('actor_attempt_id')
    if supplied_attempt is not None and (not isinstance(supplied_attempt, str)
            or re.fullmatch('[0-9a-f]{32}', supplied_attempt) is None):
        raise ValueError('actor_attempt_id must be a new32hex native launch identity')
    attempt = [(supplied_attempt or uuid.uuid4().hex) if rank == 0 else None]
    dist.broadcast_object_list(attempt, src=0)
    attempt_dir = output / 'outers' / ('outer-%04d-attempt-%s' % (outer, attempt[0]))
    if rank == 0:
        attempt_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(attempt_dir / 'attempt.json', dict(run_id=config['run_id'], outer=outer,
            attempt_id=attempt[0], cached_rollout_path=str(rollout_path), actor_version=old_version,
            config_path=str(config_path), config_sha256=sha(config_path),
            **_grounding_receipt(config), grounding_transition=grounding_transition,
            requested_actor_global_micro_batch_size=budget.global_micro,
            actor_global_micro_batch_size=budget.effective_global_micro,
            virtual_local_mini=VIRTUAL_MINI, virtual_blocks_per_physical_rank=VIRTUAL_PER_RANK,
            created_utc=utc()))
    dist.barrier()
    # Attempt identity exists even if loading/initializing the model encounters OOM.
    rows, rollout = load_rollout(rollout_path, config, outer, old_version)
    actor, wrapped, optimizer, scheduler, tokenizer = _build_actor(model_dir, budget, local)
    writes_before = latest['actual_adam_steps_per_rank'][rank] if latest else 0
    if latest:
        item = latest['rank_states'][rank]; path = remote(item['path'])
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Latest complete rank state differs')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['rank'] != rank or saved['binding'] != _state_binding(config) or saved['actual_adam_steps'] != writes_before:
            raise ValueError('Rank-local actor restore binding/count differs')
        if (saved['grounding_state'] != latest['grounding_state'] or
                saved['grounding_probe_version'] != latest['grounding_probe_version']):
            raise ValueError('Restored rank checkpoint lost its actual previous reward evaluator')
        if saved['optimizer_parameter_layout'] != optimizer_layout(optimizer):
            raise ValueError('FSDP rank-local parameter shard ordering/shape differs from the complete state')
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler']); _set_rng(saved['rng'])
    actual_steps = {int(value['step'].item()) for value in optimizer.state.values() if 'step' in value}
    expected_steps = {writes_before} if writes_before else set()
    if (actual_steps != expected_steps or any(p.dtype != torch.float32 for p in wrapped.parameters())
            or any(value.dtype != torch.float32 for state in optimizer.state.values()
                   for key, value in state.items() if key in ('exp_avg', 'exp_avg_sq'))):
        raise ValueError('Actual FP32 master/Adam state or restored step differs')
    if scheduler.last_epoch != outer - 1 or any(group['lr'] != learning_rate(outer - 1) for group in optimizer.param_groups):
        raise ValueError('Actual outer scheduler did not restore its true completed clock')
    ledger = attempt_dir / ('rank-%d-updates.jsonl' % rank)
    append(ledger, dict(event='ACTUAL_RESTORE', rank=rank, outer=outer, restored_completed_outer=outer - 1,
        actual_adam_before=writes_before, restored_scheduler=scheduler.last_epoch,
        actual_learning_rates=[part['lr'] for part in optimizer.param_groups],
        actual_model_master_dtypes=sorted({str(p.dtype) for p in wrapped.parameters()}),
        restored_grounding_state=latest['grounding_state'] if latest else None,
        restored_grounding_probe_version=latest['grounding_probe_version'] if latest else None,
        actor_version=old_version, **_grounding_receipt(config), grounding_transition=grounding_transition,
        requested_global_micro=budget.global_micro,
        actual_global_micro=budget.effective_global_micro, utc=utc()))
    reward_ids = list(range(rank, len(rows), WORLD))
    local_scores, local_reward_proof = compute_soft_rewards([rows[i] for i in reward_ids], tokenizer, config, rank)
    reward_parts = [None] * WORLD
    dist.all_gather_object(reward_parts, dict(ids=reward_ids, scores=local_scores.tolist(), proof=local_reward_proof))
    scores = [None] * len(rows)
    for part in reward_parts:
        for i, score in zip(part['ids'], part['scores']): scores[i] = score
    if any(score is None for score in scores): raise ValueError('Distributed grounding reward omitted an own trajectory')
    # Classify with the real ownEM from this exact retained rollout, never the
    # grounding soft reward. The artificial0 exists only in all-correct stats.
    advantage_values, normalization = group_advantages(rows, scores,
        mode=normalization_contract['mode'], return_diagnostics=True)
    normalization.update(run_id=config['run_id'], outer=outer, actor_version=old_version,
        contract=normalization_contract, original_rollout_sha256=sha(rollout_path),
        unmodified_raw_rewards_sha256=hashlib.sha256(canonical(scores)).hexdigest())
    normalization_path = attempt_dir / 'advantage-statistics.json'
    normalization_bytes = canonical(normalization) + b'\n'
    if rank == 0:
        atomic_json(normalization_path, normalization)
    advantages = advantage_values.tolist()
    reward_proof = dict(version=config['grounding_probe_version'], **_grounding_receipt(config), ranks=[part['proof'] for part in reward_parts],
        trajectories=len(rows), score_mean=math.fsum(scores) / len(scores),
        prefix_correctness_head_used_for_reward=False, actual_reward_world_size=WORLD,
        advantage_contract=normalization_contract, advantage_summary=normalization['summary'],
        advantage_statistics=dict(path=str(normalization_path), bytes=len(normalization_bytes),
            sha256=hashlib.sha256(normalization_bytes).hexdigest()),
        unmodified_raw_rewards_sha256=normalization['unmodified_raw_rewards_sha256'])
    plan = rank_plan(seed, outer)[rank]
    monitor_ids = frozen_monitor_ids(plan, rows)
    frozen = tensor_batch([rows[i] for i in monitor_ids], tokenizer.pad_token_id)
    frozen['advantages'] = torch.tensor([advantages[i] for i in monitor_ids], dtype=torch.float32)[:, None].expand(
        -1, frozen['responses'].shape[1]).clone()
    monitor = FrozenPolicyMonitor(actor, frozen, device=local, local_micro=budget.local_micro,
        identity=dict(run_id=config['run_id'], outer=outer, attempt_id=attempt[0], rank=rank,
            restored_actor_version=old_version, restored_actual_adam_steps=writes_before,
            original_rollout_sha256=sha(rollout_path),
            trajectory_ids=[rows[i]['trajectory_id'] for i in monitor_ids], selection_indices=monitor_ids))
    writes, opportunities = 0, []
    from search_g1.numerics import require_finite
    from search_g1.policy_safety import RecoverableGuard
    try:
        wrapped.train()
        for step, ids in enumerate(plan):
            optimizer.zero_grad(set_to_none=True); metrics = []; numerical_guard = None; norm = None
            before = writes_before + writes
            pre = monitor.observe(stage='before_minibatch', opportunity=step, actual_adam_steps=before)
            append(ledger, dict(event='POLICY_MONITOR', rank=rank, outer=outer, **pre, utc=utc()))
            if pre['allowed']:
                try:
                    for block, virtual_ids in enumerate(virtual_minibatches(ids)):
                        pair_masses = [sum(sum(rows[i]['response_info_mask']) for i in virtual_ids[pair:pair + REFERENCE_MICRO])
                                       for pair in range(0, VIRTUAL_MINI, REFERENCE_MICRO)]
                        for offset in range(0, len(virtual_ids), budget.local_micro):
                            micro_ids = virtual_ids[offset:offset + budget.local_micro]
                            batch = tensor_batch([rows[i] for i in micro_ids], tokenizer.pad_token_id)
                            response_width = batch['responses'].shape[1]
                            batch['advantages'] = torch.tensor([advantages[i] for i in micro_ids], dtype=torch.float32)[:, None].expand(-1, response_width)
                            batch = {key: value.to(local) for key, value in batch.items()}
                            _, current = actor._forward_micro_batch(batch, temperature=1., compute_entropy=False)
                            authored = batch['loss_mask'].bool()
                            require_finite(dict(new=torch.where(authored, current, 0.),
                                old=torch.where(authored, batch['old_log_probs'], 0.),
                                advantages=torch.where(authored, batch['advantages'], 0.)),
                                stage='before_dual_clip_exp', recoverable=True)
                            loss, parts = reference_policy_objective(batch['old_log_probs'], current,
                                batch['advantages'], batch['loss_mask'], virtual_offset=offset,
                                virtual_pair_token_masses=pair_masses, loss_fn=compute_policy_loss)
                            require_finite(dict(loss=loss), stage='before_backward', recoverable=True)
                            loss.backward()
                            metrics.extend(dict(part, virtual_rank=rank * VIRTUAL_PER_RANK + block) for part in parts)
                    norm = clip_global_l2(wrapped, 1., diagnostic=False, recoverable=True)
                    optimizer.step(); writes += 1
                    steps = {int(value['step'].item()) for value in optimizer.state.values() if 'step' in value}
                    if steps != {writes_before + writes}: raise ValueError('Actual Adam step counters differ')
                    append(ledger, dict(event='ACTUAL_ADAM_WRITE', rank=rank, outer=outer, opportunity=step,
                        actual_adam_steps=writes_before + writes, writes_this_attempt=writes,
                        actual_learning_rate=optimizer.param_groups[0]['lr'], grad_norm=float(norm),
                        local_trajectories=len(ids), global_mini=128, local_micro=budget.local_micro,
                        requested_global_micro=budget.global_micro, actual_global_micro=budget.effective_global_micro,
                        gradient_accumulation=budget.accumulation, objective=OBJECTIVE, metrics=metrics, utc=utc()))
                except RecoverableGuard as rejection:
                    numerical_guard = dict(stage=rejection.stage, reasons=rejection.reasons,
                                           diagnostic=rejection.diagnostic)
                    optimizer.zero_grad(set_to_none=True)
            # Every opportunity has a real after observation, including a skip.
            # Only its adjacent unchanged-state next-before call can share it.
            post = monitor.observe(stage='after_minibatch', opportunity=step, actual_adam_steps=writes_before + writes)
            append(ledger, dict(event='POLICY_MONITOR', rank=rank, outer=outer, **post, utc=utc()))
            disposition = opportunity_result(opportunity=step, before=before, after=writes_before + writes,
                pre=pre, post=post, numerical_guard=numerical_guard)
            opportunities.append(disposition)
            append(ledger, dict(event='OPPORTUNITY', rank=rank, outer=outer, **disposition, utc=utc()))
        optimizer.zero_grad(set_to_none=True)
        result = attempt_result(opportunities, writes_before=writes_before)
        rank_receipt = dict(result, rank=rank, run_id=config['run_id'], outer=outer, attempt_id=attempt[0],
            **_grounding_receipt(config), grounding_transition=grounding_transition,
            new_adam_steps=writes, actual_adam_steps=writes_before + writes, opportunities=20,
            next_learning_rate=learning_rate(outer),
            scheduler_completed_outer=scheduler.last_epoch + int(result['complete_outer']),
            actual_monitor_forward_calls=monitor.actual_forward_calls,
            fresh_monitor_observations=monitor.fresh_observations, reused_monitor_observations=monitor.reused_observations,
            ledger_path=str(ledger), ledger_sha256=sha(ledger))
        ranks = [None] * WORLD; dist.all_gather_object(ranks, rank_receipt)
        shared = ('new_adam_steps', 'actual_adam_steps', 'opportunities', 'opportunity_statuses', 'complete_outer',
                  'grounding_state', 'grounding_probe_version')
        if any(any(row[key] != rank_receipt[key] for key in shared) for row in ranks):
            raise ValueError('All four ranks must agree on actual writes and all20 opportunity dispositions')
        if not result['complete_outer']:
            guarded = dict(result, run_id=config['run_id'], outer=outer, attempt_id=attempt[0],
                binding=_state_binding(config), **_grounding_receipt(config), grounding_transition=grounding_transition,
                ranks=ranks, scheduler_completed_outer=scheduler.last_epoch,
                cached_rollout_path=str(rollout_path), cached_rollout_sha256=sha(rollout_path),
                actor_version=old_version, utc=utc())
            if rank == 0: atomic_json(attempt_dir / 'guard-result.json', guarded)
            dist.barrier()
            return guarded
        scheduler.step()
        _save_complete(config, output, attempt_dir, outer, wrapped, optimizer, scheduler, tokenizer,
            rollout_path, config['run_id'] + '-actor%04d' % outer, budget, reward_proof, ranks)
        return dict(result, run_id=config['run_id'], outer=outer, attempt_id=attempt[0],
                    **_grounding_receipt(config), grounding_transition=grounding_transition,
                    actual_adam_steps_per_rank=[row['actual_adam_steps'] for row in ranks])
    except BaseException as error:
        # A true OOM is a failed attempt, never a numeric skip or a completed outer.
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError)
        atomic_json(attempt_dir / ('rank-%d-failure.json' % rank), dict(rank=rank, outer=outer,
            run_id=config['run_id'], attempt_id=attempt[0],
            error_type=type(error).__name__, actual_cuda_oom=is_oom, writes_this_attempt=writes,
            restore_completed_outer=outer - 1, observed_unsaved_partial_writes=writes,
            abandoned_unsaved_partial_writes=writes, partial_updates_occurred=bool(writes),
            automatic_rollback_performed=False, complete_outer_count_advanced=False,
            cached_rollout_path=str(rollout_path), actor_version=old_version,
            requested_actor_global_micro_batch_size=budget.global_micro,
            actor_global_micro_batch_size=budget.effective_global_micro, complete_outer=False, utc=utc()))
        raise
    finally:
        dist.destroy_process_group()


def prune_previous(output, keep_actor_version, joint_complete_path):
    """Only explicit jointcomplete permits deleting this run's derived old actor latest."""
    output = remote(output); joint = read(remote(joint_complete_path)); latest = read(output / 'latest.json')
    if (joint.get('run_id') != latest['run_id'] or joint.get('outer') != latest['completed_outer']
            or joint.get('actor_version') != latest['actor_version']
            or keep_actor_version != latest['actor_version'] or not joint.get('joint_outer_complete')):
        raise ValueError('Actor/probe/pool must jointly commit before pruning a previous latest')
    deleted = []
    for directory in sorted((output / 'outers').iterdir()):
        proof_path = directory / 'complete.json'
        if not proof_path.exists(): continue  # Failed-attempt evidence is retained.
        proof = read(proof_path)
        if (proof['run_id'] != latest['run_id'] or proof['milestone']
                or proof['actor_version'] == keep_actor_version): continue
        for name in ('huggingface', 'driver.pt') + tuple('rank-%d.pt' % r for r in range(WORLD)):
            target = directory / name
            if target.is_dir(): shutil.rmtree(target)
            elif target.is_file(): target.unlink()
        deleted.append(dict(state_dir=str(directory), completed_outer=proof['completed_outer'],
                            saved_files_sha256=dict(hf=proof['hf_files'], rank_states=proof['rank_states'])))
    atomic_json(output / ('prune-after-outer-%04d.json' % latest['completed_outer']),
                dict(joint_complete_sha256=sha(joint_complete_path), deleted=deleted, utc=utc()))
    return deleted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--outer', type=int, required=True)
    parser.add_argument('--rollout', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        result = train(args.config, args.outer, args.rollout, args.output)
        print('SEARCH_G1_NATIVE_ACTOR_RESULT=' + json.dumps(result), flush=True)
    except BaseException as error:
        import torch
        # Also record real initialization OOMs, before a synchronized attempt
        # directory/ledger exists. The controller may retry after this group exits.
        config = read(remote(args.config))
        attempt_id = config.get('actor_attempt_id', 'unspecified-' + uuid.uuid4().hex)
        failure = dict(event='NATIVE_ACTOR_PHASE_FAILURE', outer=args.outer,
            run_id=config['run_id'], attempt_id=attempt_id,
            rank=int(os.environ.get('RANK', '-1')), error_type=type(error).__name__,
            actual_cuda_oom=isinstance(error, torch.cuda.OutOfMemoryError),
            cached_rollout_path=str(args.rollout), complete_outer=False, utc=utc())
        output = remote(args.output); output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / ('native-failure-outer-%04d-attempt-%s-rank-%s.json' %
            (args.outer, attempt_id, failure['rank'])), failure)
        print('SEARCH_G1_NATIVE_ACTOR_FAILURE=' + json.dumps(failure), flush=True)
        raise


if __name__ == '__main__': main()
