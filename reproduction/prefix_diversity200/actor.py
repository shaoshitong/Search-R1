"""Eight-rank, phase-local Search-G1 actor updates on the selected *same* G5.

The native sampling journal supplies old policy probabilities, including the
scored prefixes. Grounding flip/boundary supplies the unchanged soft reward;
the correctness prefix probe is used neither as reward nor as an old policy.
Every successful phase writes twenty Adam updates and one outer scheduler step.
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

REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
WORLD = 8
MILESTONES = (5, 50, 100, 150, 200)
SAMPLING_CONTRACT = 'chunked_native_action500_prefix20_v1'
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
    if not path.is_absolute() or not str(path).startswith('/mnt/'):
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
    global_micro: int = 128
    global_mini: int = 128
    global_trajectories: int = 2560
    world: int = 8
    ppo_epochs: int = 1

    def __post_init__(self):
        if (self.world != WORLD or self.global_mini != 128 or self.global_trajectories != 2560
                or self.ppo_epochs != 1 or type(self.global_micro) is not int
                or self.global_micro < WORLD or self.global_mini % self.global_micro
                or self.global_micro % WORLD):
            raise ValueError('Keep8ranks, mini128, one pass,20writes; micro must divide mini128')

    @property
    def local_micro(self):
        return self.global_micro // self.world

    @property
    def accumulation(self):
        return self.global_mini // self.global_micro


def learning_rate(completed_outer):
    if type(completed_outer) is not int or not 0 <= completed_outer <= 200:
        raise ValueError('Completed outer is outside fresh200')
    return 1e-6 * min(completed_outer / 57, 1.)


def oom_retry_plan(*, outer, failed_micro, latest_complete_outer, rollout_actor_version,
                   latest_actor_version, partial_writes):
    """Actual OOM controller input; abandoned partial writes are never stitched in.

    Call only after the failed native process group has exited. Retain its rank
    ledgers, mark partial writes abandoned, and retry this exact cached rollout
    from this run's immediately preceding committed actor state under a NEW ID.
    """
    budget = ActorBudget(failed_micro)
    if (type(outer) is not int or not 1 <= outer <= 200 or latest_complete_outer != outer - 1
            or rollout_actor_version != latest_actor_version or len(partial_writes) != WORLD
            or any(type(n) is not int or not 0 <= n <= 20 for n in partial_writes)):
        raise ValueError('OOM recovery must use the same rollout and latest complete actor boundary')
    if budget.local_micro == 1:
        return dict(status='OOM_AT_MINIMUM_MICRO', retry=False, abandoned_partial_writes=partial_writes)
    next_budget = ActorBudget(failed_micro // 2)
    return dict(status='RETRY_SAME_CACHED_OUTER_FROM_LAST_COMPLETE', retry=True,
        restore_completed_outer=latest_complete_outer, new_attempt_id=uuid.uuid4().hex,
        actor_global_micro_batch_size=next_budget.global_micro,
        actor_local_micro_batch_size=next_budget.local_micro,
        gradient_accumulation=next_budget.accumulation,
        abandoned_partial_writes=partial_writes, stitch_partial_writes=False,
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
                    or latest.get('actual_adam_steps_per_rank') != [20 * outer] * WORLD
                    or latest.get('scheduler_completed_outer') != outer
                    or latest.get('original_rollout_sha256') != sha(rollout_path)):
                raise ValueError('Published actor completion does not match this failed launch input')
            return dict(status='ACTOR_ALREADY_COMMITTED', retry=False,
                        completed_outer=outer, actor_version=latest['actor_version'])
        restore_metadata(config, latest, outer)
        completed, version = latest['completed_outer'], latest['actor_version']
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
                if (row['rank'] != rank or row['outer'] != outer or row['opportunity'] != index
                        or row['writes_this_attempt'] != index + 1
                        or row['actual_adam_steps'] != 20 * completed + index + 1):
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
    plan = oom_retry_plan(outer=outer, failed_micro=int(config.get('actor_global_micro_batch_size', 128)),
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
    rows.sort(key=lambda row: (row['candidate_ordinal'], row['response_index']))
    return rows, complete


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


def group_advantages(rows, scores):
    import torch
    scores = torch.as_tensor(scores, dtype=torch.float32)
    advantages = torch.empty_like(scores)
    grouped = defaultdict(list)
    for i, row in enumerate(rows): grouped[str(row['qid'])].append(i)
    for ids in grouped.values():
        if len(ids) != 5: raise ValueError('GRPO advantages require own completeG5 rewards')
        values = scores[ids]
        advantages[ids] = (values - values.mean()) / (values.std(correction=1) + 1e-6)
    return advantages


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
    return scores, dict(version=manager.last_probe_version, rank=rank, probe_url=os.environ['PROBE_URL'],
        reward_environment=REWARD_ENV, score_mean=float(scores.mean()), score_min=float(scores.min()),
        score_max=float(scores.max()), prefix_correctness_head_used_for_reward=False)


def rank_plan(seed, outer, world=WORLD):
    """One natural randomized2560 trajectory pass, exactly20x16 rows/rank."""
    if world != WORLD or not 1 <= outer <= 200: raise ValueError('Require eight ranks and outer1..200')
    order = list(range(2560)); random.Random(seed + outer).shuffle(order)
    return [[order[step * 128 + rank * 16:step * 128 + (rank + 1) * 16]
             for step in range(20)] for rank in range(world)]


def masked_policy_objective(old, current, advantages, mask, *, loss_fn):
    """A legal all-empty policy micro contributes zero, while joining backward."""
    if not bool(mask.bool().any()):
        zero = current.masked_fill(~mask.bool(), 0.).sum() * 0.
        return zero, zero.detach(), zero.detach()
    return loss_fn(old, current, advantages, mask, cliprange=.2, dual_clip=3.)


def logical_pair_objective(old, current, advantages, mask, offset, pair_masses, *, loss_fn):
    """Original eight local two-row means, independent of physical forward size."""
    combined = current.masked_fill(~mask.bool(), 0.).sum() * 0.
    parts = []
    for first in range(0, len(current)):
        absolute = offset + first
        if first and (absolute // 2)==((absolute-1)//2): continue
        last = min(len(current), first + (2 - absolute % 2))
        mass = int(mask[first:last].sum().item()); pair = absolute // 2
        weight = mass / (8 * pair_masses[pair]) if pair_masses[pair] else 0.
        loss, clipfrac, kl = masked_policy_objective(old[first:last],current[first:last],
            advantages[first:last],mask[first:last],loss_fn=loss_fn)
        combined = combined + loss * weight
        parts.append(dict(logical_pair=pair,policy_tokens=mass,pair_policy_tokens=pair_masses[pair],
            gradient_weight=weight,loss=float(loss.detach()),clip_fraction=float(clipfrac.detach()),old_policy_kl=float(kl.detach())))
    return combined, parts


def frozen_policy_monitor(rows, ids, advantages, tokenizer):
    import torch
    ordered = sorted(ids,key=lambda i:sum(rows[i]['response_info_mask']))
    indices = [ordered[int(x)] for x in torch.linspace(0,len(ordered)-1,4).long().tolist()]
    batch = tensor_batch([rows[i] for i in indices],tokenizer.pad_token_id)
    batch['advantages'] = torch.tensor([advantages[i] for i in indices],dtype=torch.float32)[:,None].expand_as(batch['loss_mask'])
    weights = torch.zeros_like(batch['loss_mask'],dtype=torch.float64)
    for first in range(0,4,2):
        mass = batch['loss_mask'][first:first+2].sum().double().clamp_min(1)
        weights[first:first+2] = batch['loss_mask'][first:first+2].double()/mass
    identity = {name:hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for name,value in batch.items()}
    return batch, weights, dict(selection_indices=indices,tensor_sha256=identity,scope='four length-stratified trajectories per rank fixed through the outer')


def policy_monitor(actor, wrapped, frozen, weights, identity, local_micro, local, stage, writes, ledger, config):
    import torch
    from search_g1.policy_safety import monitor
    mode=wrapped.training;values=[]
    try:
        wrapped.eval()
        with torch.no_grad():
            for first in range(0,4,min(local_micro,4)):
                last=min(first+local_micro,4)
                batch={name:value[first:last].to(local) for name,value in frozen.items()}
                _, current=actor._forward_micro_batch(batch,temperature=1.,compute_entropy=False)
                mask=batch['loss_mask'].bool()
                values.append(torch.where(mask,current.double()-batch['old_log_probs'].double(),0.))
    finally:wrapped.train(mode)
    safety=config['policy_monitor']
    report=monitor(torch.cat(values),frozen['advantages'].to(local),frozen['loss_mask'].to(local),weights.to(local),
        c=3.,kl_log_bound=safety['kl_log_bound'],target_kl=safety['target_old_policy_kl'],
        negative_tail_limit=safety['negative_tail_limit'],group=wrapped.process_group,
        execution_role='production',finite_thresholds_diagnostic_only=True)
    append(ledger,dict(event='POLICY_MONITOR',stage=stage,optimizer_writes=writes,input_identity=identity,report=report,utc=utc()))
    if report['invalid_count']:
        raise FloatingPointError('Actual nonfinite authored policy monitor values')
    return report


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


def _state_binding(config):
    # Micro is intentionally mutable after a real OOM; the scientific budget is not.
    return dict(run_id=config['run_id'], model_revision=REVISION,
        official_model_dir=str(remote(config['official_model_dir'])), world_size=8,
        seed=int(config.get('seed', 1729)),
        global_mini=128, ppo_epochs=1, trajectories=2560, updates_per_outer=20,
        learning_rate=1e-6, outer_warmup=57, clip_ratio=.2, dual_clip=3., grad_clip=1.,
        loss_reduction='mean_of_eight_rank_eight_logical_two_row_policy_means',
        entropy_coefficient=0., reference_kl_coefficient=0., reference_kl_type='low_var_kl',
        grounding_probe_version=config.get('grounding_initial_probe_version',config['grounding_probe_version']),
        grounding_refit_rounds=[0,50,100,150], reward_environment=REWARD_ENV)


def restore_metadata(config, latest, outer):
    expected = outer - 1
    if (latest['run_id'] != config['run_id'] or latest['completed_outer'] != expected
            or latest['binding'] != _state_binding(config)
            or latest['actual_adam_steps_per_rank'] != [20 * expected] * WORLD
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


def current_driver_metadata(config, outer):
    """Bind a new save to this execution base, never inherited resume aliases."""
    from .pool import state_sha
    base = remote(config['base'])
    root = base / 'outers' / ('outer-%06d' % outer)
    pool_path = base / 'question-pool/state.json'
    pool = read(pool_path)
    reservation = read(root / 'pool-reservation.json')
    if (reservation['outer'] != outer or pool != reservation['after']
            or state_sha(pool) != reservation['after_sha256']
            or pool['next_outer'] != outer + 1
            or pool['consumed_candidates'] != 5120 * outer):
        raise ValueError('Current actor save lost its actual reserved candidate boundary')
    pool_proof = dict(path=str(pool_path), sha256=sha(pool_path), value=pool)
    if pool_proof['sha256'] != hashlib.sha256(canonical(pool) + b'\n').hexdigest():
        raise ValueError('Candidate state changed while capturing the driver receipt')
    probe = linked_probe_receipt(str(base / 'probe/latest.json'))
    score_path = root / 'probe-score/manifest.json'
    score = read(score_path)
    before = probe['value']
    if (score['run_id'] != config['run_id'] or score['outer'] != outer
            or before['run_id'] != config['run_id']
            or before['completed_outer'] != outer - 1
            or before['actual_adam_steps_per_rank'] != [20 * (outer - 1)] * WORLD
            or before['sha256'] != score['probe_before_checkpoint_sha256']):
        raise ValueError('Current actor save lost its actual selection-time probe boundary')
    return dict(candidate_pool_state=pool_proof, prefix_probe_receipt=probe,
        driver_metadata_binding=dict(source='current_execution_base', outer=outer,
            run_id=config['run_id'], score_manifest_sha256=sha(score_path)))


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
        actual_adam_steps=20 * outer, scheduler_completed_outer=outer,
        binding=_state_binding(config), optimizer=cpu_tree(optimizer.state_dict()),
        optimizer_parameter_layout=optimizer_layout(optimizer),
        scheduler=cpu_tree(scheduler.state_dict()), rng=_rng(),
        model_master_dtype='torch.float32', optimizer_layout='FSDP_FULL_SHARD_same_Qwen2DecoderLayer_wrap_world8',
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
        metadata = current_driver_metadata(config, outer)
        driver_proof = atomic_save(attempt_dir / 'driver.pt', dict(run_id=config['run_id'],
            completed_outer=outer, actor_version=actor_version, driver_state=driver_state,
            grounding_probe_version=config['grounding_probe_version'], rng=_rng(),
            rng_scope='actor_phase_rank0; controller candidate RNG/order lives in candidate_pool_state',
            **metadata))
        receipt = dict(schema_version=1, role='prefix_selection200_actor_complete', run_id=config['run_id'],
            source_commit=config.get('source_commit'), completed_outer=outer, actor_version=actor_version,
            model_dir=str(hfdir), actor_model_dir=str(hfdir), hf_model_dir=str(hfdir), model_master_dtype='torch.float32',
            rollout_compute_dtype='bfloat16', state_dir=str(attempt_dir), state_path=str(attempt_dir), rank_states=proofs,
            hf_files=hf, driver_state=driver_proof, actual_adam_steps_per_rank=[20 * outer] * WORLD,
            new_adam_steps_per_rank=[20] * WORLD, scheduler_completed_outer=outer,
            learning_rate_used=learning_rate(outer - 1), next_learning_rate=learning_rate(outer),
            actual_global_micro=budget.global_micro, actual_local_micro=budget.local_micro,
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
    if config['model_revision'] != REVISION or not 1 <= outer <= 200:
        raise ValueError('Fresh official actor200 revision/budget differs')
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world != WORLD: raise ValueError('Actor phase requires exactly8 ranks')
    torch.cuda.set_device(local); dist.init_process_group('nccl')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    budget = ActorBudget(int(config.get('actor_global_micro_batch_size', 128)))
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
            actor_global_micro_batch_size=budget.global_micro, created_utc=utc()))
    dist.barrier()
    # Attempt identity exists even if loading/initializing the model encounters OOM.
    rows, rollout = load_rollout(rollout_path, config, outer, old_version)
    actor, wrapped, optimizer, scheduler, tokenizer = _build_actor(model_dir, budget, local)
    if latest:
        item = latest['rank_states'][rank]; path = remote(item['path'])
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Latest complete rank state differs')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['rank'] != rank or saved['binding'] != _state_binding(config) or saved['actual_adam_steps'] != 20 * (outer - 1):
            raise ValueError('Rank-local actor restore binding/count differs')
        if saved['optimizer_parameter_layout'] != optimizer_layout(optimizer):
            raise ValueError('FSDP rank-local parameter shard ordering/shape differs from the complete state')
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler']); _set_rng(saved['rng'])
    actual_steps = {int(value['step'].item()) for value in optimizer.state.values() if 'step' in value}
    expected_steps = {20 * (outer - 1)} if outer > 1 else set()
    if (actual_steps != expected_steps or any(p.dtype != torch.float32 for p in wrapped.parameters())
            or any(value.dtype != torch.float32 for state in optimizer.state.values()
                   for key, value in state.items() if key in ('exp_avg', 'exp_avg_sq'))):
        raise ValueError('Actual FP32 master/Adam state or restored step differs')
    if scheduler.last_epoch != outer - 1 or any(group['lr'] != learning_rate(outer - 1) for group in optimizer.param_groups):
        raise ValueError('Actual outer scheduler did not restore its true completed clock')
    ledger = attempt_dir / ('rank-%d-updates.jsonl' % rank)
    append(ledger, dict(event='ACTUAL_RESTORE', rank=rank, outer=outer, restored_completed_outer=outer - 1,
        actual_adam_before=20 * (outer - 1), restored_scheduler=scheduler.last_epoch,
        actual_learning_rates=[part['lr'] for part in optimizer.param_groups],
        actual_model_master_dtypes=sorted({str(p.dtype) for p in wrapped.parameters()}),
        actor_version=old_version, actual_global_micro=budget.global_micro, utc=utc()))
    reward_ids = list(range(rank, len(rows), WORLD))
    local_scores, local_reward_proof = compute_soft_rewards([rows[i] for i in reward_ids], tokenizer, config, rank)
    reward_parts = [None] * WORLD
    dist.all_gather_object(reward_parts, dict(ids=reward_ids, scores=local_scores.tolist(), proof=local_reward_proof))
    scores = [None] * len(rows)
    for part in reward_parts:
        for i, score in zip(part['ids'], part['scores']): scores[i] = score
    if any(score is None for score in scores): raise ValueError('Distributed grounding reward omitted an own trajectory')
    # Original group normalization happens after the8 reward shards are reunited.
    advantages = group_advantages(rows, scores).tolist()
    reward_proof = dict(version=config['grounding_probe_version'], ranks=[part['proof'] for part in reward_parts],
        trajectories=len(rows), score_mean=math.fsum(scores) / len(scores),
        prefix_correctness_head_used_for_reward=False, actual_reward_world_size=WORLD)
    plan = rank_plan(seed, outer)[rank]
    frozen, monitor_weights, monitor_identity = frozen_policy_monitor(rows,reward_ids,advantages,tokenizer)
    writes = 0
    try:
        wrapped.train()
        for step, ids in enumerate(plan):
            optimizer.zero_grad(set_to_none=True); metrics = []
            pair_masses = [sum(sum(rows[i]['response_info_mask']) for i in ids[first:first+2]) for first in range(0,16,2)]
            policy_monitor(actor,wrapped,frozen,monitor_weights,monitor_identity,budget.local_micro,local,'before_minibatch',writes,ledger,config)
            for offset in range(0, len(ids), budget.local_micro):
                micro_ids = ids[offset:offset + budget.local_micro]
                batch = tensor_batch([rows[i] for i in micro_ids], tokenizer.pad_token_id)
                response_width = batch['responses'].shape[1]
                batch['advantages'] = torch.tensor([advantages[i] for i in micro_ids], dtype=torch.float32)[:, None].expand(-1, response_width)
                batch = {key: value.to(local) for key, value in batch.items()}
                _, current = actor._forward_micro_batch(batch, temperature=1., compute_entropy=False)
                loss, parts = logical_pair_objective(batch['old_log_probs'],current,
                    batch['advantages'],batch['loss_mask'],offset,pair_masses,loss_fn=compute_policy_loss)
                finite = torch.tensor(int(torch.isfinite(loss)),dtype=torch.int32,device=local)
                dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item():raise FloatingPointError('Nonfinite actual actor loss')
                loss.backward();metrics.extend(parts)
            norm = clip_global_l2(wrapped, 1., diagnostic=False)
            optimizer.step(); writes += 1
            policy_monitor(actor,wrapped,frozen,monitor_weights,monitor_identity,budget.local_micro,local,'after_minibatch',writes,ledger,config)
            steps = {int(value['step'].item()) for value in optimizer.state.values() if 'step' in value}
            if steps != {20 * (outer - 1) + writes}: raise ValueError('Actual Adam step counters differ')
            append(ledger, dict(event='ACTUAL_ADAM_WRITE', rank=rank, outer=outer, opportunity=step,
                actual_adam_steps=20 * (outer - 1) + writes, writes_this_attempt=writes,
                actual_learning_rate=optimizer.param_groups[0]['lr'], grad_norm=float(norm),
                local_trajectories=len(ids), global_mini=128, local_micro=budget.local_micro,
                gradient_accumulation=budget.accumulation,
                objective='mean_of_eight_rank_eight_logical_two_row_policy_means', metrics=metrics, utc=utc()))
        if writes != 20: raise ValueError('Incomplete outer budget cannot advance scheduler/save latest')
        scheduler.step()
        rank_receipt = dict(rank=rank, outer=outer, new_adam_steps=writes, actual_adam_steps=20 * outer,
            opportunities=20, scheduler_completed_outer=scheduler.last_epoch,
            next_learning_rate=optimizer.param_groups[0]['lr'], ledger_path=str(ledger), ledger_sha256=sha(ledger))
        ranks = [None] * WORLD; dist.all_gather_object(ranks, rank_receipt)
        if any(row['new_adam_steps'] != 20 or row['scheduler_completed_outer'] != outer for row in ranks):
            raise ValueError('All eight ranks must complete20 writes before actor completion')
        _save_complete(config, output, attempt_dir, outer, wrapped, optimizer, scheduler, tokenizer,
            rollout_path, config['run_id'] + '-actor%04d' % outer, budget, reward_proof, ranks)
    except BaseException as error:
        # A true OOM is a failed attempt, never a numeric skip or a completed outer.
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError)
        atomic_json(attempt_dir / ('rank-%d-failure.json' % rank), dict(rank=rank, outer=outer,
            run_id=config['run_id'], attempt_id=attempt[0],
            error_type=type(error).__name__, actual_cuda_oom=is_oom, writes_this_attempt=writes,
            restore_completed_outer=outer - 1, abandoned_unsaved_partial_writes=writes,
            cached_rollout_path=str(rollout_path), actor_version=old_version,
            actor_global_micro_batch_size=budget.global_micro, complete_outer=False, utc=utc()))
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
        train(args.config, args.outer, args.rollout, args.output)
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
