"""Native 4 x TP1 prefix screening and continuation of the SAME G5 samples.

Train and greedy QA share the canonical new interaction protocol.
A logical action has a 500-token budget;
native calls can pause it at the fifth retained policy token. The first
complete action actually observed in the cumulative native stream is executed.
Length quota is a pause, never an invented EOS or environment correction.

The prefix scan is label-free. Gold is retained only in private continuation
states and completed PPO records. No model, trajectory or tensor is moved to
the management computer by this module.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from reproduction.prefix20_online.pool import atomic, canonical, digest, immutable, read
from reproduction.prefix20_online.rollout import prompt_tokens, strict_label, utc
from .contract import (G, PREFIX_TOKENS, QMAX, SAMPLING_CONTRACT, WIDTH, WORLD,
    PERIODIC_CANDIDATES, SELECTED_QUESTIONS)
from .candidate_schedule import candidate_count, schedule_metadata
from reproduction.prefix_selection200.vllm_ports import bounded_vllm_port_namespace
from .interaction_protocol import (ACTUAL_COMPUTE_DTYPE, ACTION_TOKENS, BOUNDARY, BUDGET,
    COMPUTE_DTYPE, CONTRACT as INTERACTION_CONTRACT, MAX_OBSERVATION_TOKENS, MAX_SEARCH_TURNS,
    MAX_START_TOKENS, append_chunk, finish_action, first_action_cut, native_snapshot,
    registered_sampling, start_action, stream_action_present, terminal_token_ids)


REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
CONTRACT = SAMPLING_CONTRACT


def _configure_one_visible_gpu(rank):
    """Resolve the four admitted GPU identities before importing CUDA libraries."""
    visible = [value for value in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if value]
    if len(visible) != WORLD or len(set(visible)) != WORLD or not 0 <= rank < WORLD:
        raise ValueError('Prefix5 rollout needs exactly four actually assigned visible GPUs')
    import csv
    import io
    inventory = list(csv.reader(io.StringIO(subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
        text=True, timeout=30))))
    by_uuid, by_index = {}, {}
    for row in inventory:
        if len(row) != 2 or not row[0].strip().isdigit() or not row[1].strip().startswith('GPU-'):
            raise ValueError('Cannot resolve assigned GPU UUIDs from actual NVML inventory')
        index, gpu_uuid = int(row[0].strip()), row[1].strip()
        if gpu_uuid in by_uuid or index in by_index:
            raise ValueError('Actual GPU inventory identity repeats')
        by_uuid[gpu_uuid], by_index[index] = index, gpu_uuid
    assigned = []
    for value in visible:
        if value.isdigit() and int(value) in by_index:
            assigned.append((int(value), by_index[int(value)]))
        elif value in by_uuid:
            assigned.append((by_uuid[value], value))
        else:
            raise ValueError('An assigned visible GPU is absent from actual NVML inventory')
    if len({value[1] for value in assigned}) != WORLD:
        raise ValueError('Assigned visible devices resolve to duplicate physical GPUs')
    physical_index, gpu_uuid = assigned[rank]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(physical_index)
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        os.environ.pop(key, None)
    return dict(physical_gpu_index=physical_index, gpu_uuid=gpu_uuid,
                originally_assigned_device=visible[rank], cuda_device_order='PCI_BUS_ID')


def root_dir(config, outer):
    return Path(config['base']) / 'outers' / ('outer-%06d' % outer)


def question_count(config, phase, outer=None):
    if phase not in ('prefix', 'continue'):
        raise ValueError('Unknown native rollout phase')
    if outer is None:
        if config.get('candidate_schedule') is not None:
            raise ValueError('Scheduled rollout count requires its actual outer')
        expected = PERIODIC_CANDIDATES
    else:
        expected = candidate_count(config, outer)
    if phase == 'prefix':
        requested = config.get('rollout_candidate_questions', SELECTED_QUESTIONS if config.get('diagnostic_only') else expected)
        allowed = {expected, SELECTED_QUESTIONS} if config.get('candidate_schedule') is None else {SELECTED_QUESTIONS if config.get('diagnostic_only') else expected}
    else:
        requested = config.get('rollout_continuation_questions', SELECTED_QUESTIONS)
        allowed = {SELECTED_QUESTIONS, PERIODIC_CANDIDATES}
        if config.get('candidate_schedule') is not None:
            from reproduction.prefix_selection200.periodic import is_probe_update
            allowed = {SELECTED_QUESTIONS, PERIODIC_CANDIDATES} if is_probe_update(outer) else {SELECTED_QUESTIONS}
    if type(requested) is not int or requested not in allowed:
        raise ValueError('Native question count differs from scheduled prefix/Top512/periodic-full-pool scope')
    return requested


def phase_output(config,outer,phase):
    name='prefix' if phase=='prefix' else config.get('rollout_full_subdirectory','full')
    if name not in ('prefix','full','full-pool'):
        raise ValueError('Unknown native artifact scope')
    return root_dir(config,outer)/name


def binding(config, outer):
    value = dict(config)
    if value.get('interaction_protocol', INTERACTION_CONTRACT) != INTERACTION_CONTRACT:
        raise ValueError('Configured rollout interaction protocol differs from canonical train/QA protocol')
    if (value.get('world_size', WORLD) != WORLD or
            value.get('prefix_policy_tokens', PREFIX_TOKENS) != PREFIX_TOKENS):
        raise ValueError('Native rollout requires the bound four-rank prefix20 recipe')
    path = root_dir(config, outer) / 'actor-binding.json'
    if path.exists():
        value.update(read(path))
    if not value.get('actor_model_dir') or not value.get('actor_version'):
        raise ValueError('An explicit immutable current actor path/version is required')
    if value.get('action_boundary', BOUNDARY) != BOUNDARY:
        raise ValueError('Wrong training stream action-boundary contract')
    return value


def seed_for(run_id, outer, trajectory_id, turn, chunk, seed=1729, sampling_attempt_id=None):
    """Independent, recorded native request streams, including resumed chunks."""
    value = [run_id, int(outer), trajectory_id, int(turn), int(chunk), int(seed)]
    if sampling_attempt_id is not None:
        if not isinstance(sampling_attempt_id, str) or not sampling_attempt_id:
            raise ValueError('A resampling attempt requires its explicit nonempty identity')
        value.append(sampling_attempt_id)
    return int.from_bytes(hashlib.sha256(canonical(value)).digest()[:8], 'big') % (2 ** 31 - 1)


def initial_state(config, outer, group, response_index, tokenizer):
    prompt = prompt_tokens(tokenizer, group['prompt'], MAX_START_TOKENS)
    if not prompt:
        raise ValueError('Actual official initial request is empty')
    ordinal = int(group['candidate_ordinal'])
    tid = '%s%s-outer%06d-c%04d-g%d' % (config['run_id'],
        '-diagnostic' if config.get('diagnostic_only') else '',outer,ordinal,response_index)
    return dict(schema_version=1, run_id=config['run_id'], outer=outer,
                actor_version=config['actor_version'], trajectory_id=tid,
                candidate_ordinal=ordinal, response_index=response_index,
                group=group, prompt=prompt, response=[], info_mask=[], log_probs=[],
                turns=[], ordinal=0, chunks_started=0, action=None,
                retrieval_calls=0, terminal=False, terminal_reason=None,
                sampling_contract=CONTRACT, action_boundary=BOUNDARY, interaction_protocol=INTERACTION_CONTRACT,
                sampling_attempt_id=config.get('sampling_attempt_id'))


def retained_length(state):
    count = sum(state['info_mask'])
    if state['action'] is not None:
        count += len(state['action']['raw_generated_token_ids'])
    return count


def next_request(state, phase, config):
    if state['terminal']:
        return None
    action = state['action']
    remaining = ACTION_TOKENS - len(action['raw_generated_token_ids'])
    if phase == 'prefix':
        remaining = min(remaining, PREFIX_TOKENS - retained_length(state))
    if remaining <= 0:
        return None
    chunk = state['chunks_started']
    request = action['actual_request_token_ids'] + action['raw_generated_token_ids']
    rid = '%s-t%d-chunk%d' % (state['trajectory_id'], state['ordinal'], chunk)
    seed = seed_for(state['run_id'], state['outer'], state['trajectory_id'],
                    state['ordinal'], chunk, config.get('seed', 1729),
                    sampling_attempt_id=config.get('sampling_attempt_id', state.get('sampling_attempt_id')))
    state['chunks_started'] += 1
    return dict(request_id=rid, actual_request_token_ids=request, max_tokens=remaining, seed=seed,
                logical_turn_index=state['ordinal'], native_chunk_index=chunk,
                logical_action_budget=ACTION_TOKENS, phase=phase,
                sampling_attempt_id=config.get('sampling_attempt_id', state.get('sampling_attempt_id')))


def prefix_record(state):
    """Actual initial request and first20 kept positions, including tool turns."""
    segments, ids, positions = [], [], []
    turns = list(state['turns'])
    if state['action'] is not None:
        a = state['action']
        turns.append(dict(a, kept_token_count=len(a['raw_generated_token_ids'])))
    for turn in turns:
        take = min(turn['kept_token_count'], PREFIX_TOKENS - len(ids))
        if take <= 0:
            break
        request = turn['actual_request_token_ids']
        policy = turn['raw_generated_token_ids'][:take]
        segments.append(dict(turn_index=turn['turn_index'], policy_start=len(ids), policy_length=take,
                             request_length=len(request), position_start=len(request),
                             context_tokens=request + policy))
        ids.extend(policy)
        positions.extend(range(len(request), len(request) + take))
    if len(ids) != PREFIX_TOKENS and not state['terminal']:
        raise ValueError('Prefix phase ended before5 retained tokens without a natural terminal')
    group = state['group']
    row = dict(schema_version=1, run_id=state['run_id'], outer=state['outer'], split='train',
               trajectory_id=state['trajectory_id'], qid=group['qid'],
               question_hash=group['question_hash'], candidate_ordinal=state['candidate_ordinal'],
               response_index=state['response_index'], actor_version=state['actor_version'],
               initial_prompt_token_ids=state['prompt'], prefix_segments=segments,
               policy_token_ids=ids, policy_position_ids=positions,
               terminal=state['terminal'], terminal_reason=state['terminal_reason'],
               retained_policy_length=len(ids), future_information_included=False)
    if 'normalized_question_hash' in group:
        row['normalized_question_hash'] = group['normalized_question_hash']
    row['prefix_sha256'] = hashlib.sha256(canonical(row)).hexdigest()
    return row


def validate_selection(selection, prefix_receipt, expected_questions=512):
    if (selection.get('run_id') != prefix_receipt['run_id'] or selection.get('outer') != prefix_receipt['outer']
            or selection.get('actor_version') != prefix_receipt['actor_version']
            or selection.get('prefix_scan_sha256') != prefix_receipt['scan_sha256']):
        raise ValueError('Top512 selection is not bound to this actor/prefix scan')
    groups = selection['selected']
    if len(groups) != expected_questions:
        raise ValueError('Top512 selection must retain exactly512 original completeG5 groups')
    qids, tids, ordinals = set(), set(), set()
    for group in groups:
        if (group['qid'] in qids or group['candidate_ordinal'] in ordinals or
                len(group['trajectory_ids']) != G or len(set(group['trajectory_ids'])) != G):
            raise ValueError('Selected group identity repeats or is incomplete')
        if tids.intersection(group['trajectory_ids']):
            raise ValueError('Selected trajectory occurs in more than one group')
        qids.add(group['qid']); ordinals.add(group['candidate_ordinal']); tids.update(group['trajectory_ids'])
    return groups


def file_item(path, count=None):
    result = dict(path=str(path), sha256=digest(path), bytes=Path(path).stat().st_size)
    if count is not None:
        result['count'] = count
    return result


def validate_rank_receipt(config_path, config, outer, phase, rank, path):
    """Admit current receipts or explicitly imported same-run continuation.

    An old config digest never becomes the new digest. The recovery whitelist
    must bind the original source config and the complete original receipt;
    both original and imported bytes, actor identity and all raw outputs are
    independently read here. Partial/failed artifacts are never accepted.
    """
    path=Path(path);saved=read(path);current_sha=digest(config_path)
    expected=dict(run_id=config['run_id'],outer=outer,phase=phase,rank=rank,
                  actor_version=config['actor_version'],trajectories=question_count(config,phase,outer)*G//WORLD,
                  interaction_protocol=INTERACTION_CONTRACT,actual_compute_dtype=ACTUAL_COMPUTE_DTYPE,
                  sampling_attempt_id=config.get('sampling_attempt_id'))
    if any(saved.get(key)!=value for key,value in expected.items()):
        raise ValueError('Existing rank receipt has a different actual run/actor/outer/phase/rank/protocol')
    reuse=None
    if saved.get('config_sha256')!=current_sha:
        allowed=config.get('continue_reuse')
        if phase!='continue' or not isinstance(allowed,dict) or allowed.get('outer')!=outer:
            raise ValueError('Old-config rank receipt has no exact continuation reuse authorization')
        source_base=Path(allowed['source_base'])
        if not source_base.is_absolute():
            raise ValueError('Continuation receipt source must be an absolute isolated base')
        source_config=source_base/'preparation'/'launch-config.json'
        source_sha=allowed['source_config_sha256']
        if (saved.get('config_sha256')!=source_sha or digest(source_config)!=source_sha
                or read(source_config).get('run_id')!=config['run_id']):
            raise ValueError('Continuation source config differs from the exact approved source')
        expected_sha=allowed['imported_receipts_sha256'].get(str(rank))
        source_receipt=root_dir(dict(base=str(source_base)),outer)/'full'/('rank-%d-complete.json'%rank)
        if not expected_sha or digest(path)!=expected_sha or digest(source_receipt)!=expected_sha:
            raise ValueError('Imported/original continuation receipt differs from its full SHA whitelist')
        reuse=dict(rank=rank,source_base=str(source_base),source_config_sha256=source_sha,
                   original_receipt_path=str(source_receipt),imported_receipt_path=str(path),
                   full_receipt_sha256=expected_sha,current_config_sha256=current_sha)
    for key in ('states','prefix') if phase=='prefix' else ('full','own_em'):
        item=saved[key];raw=Path(item['path'])
        if raw.stat().st_size!=item['bytes'] or digest(raw)!=item['sha256']:
            raise ValueError('Committed original native rank output changed: '+key)
    return saved,reuse


def iter_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf8') as stream:
        for line in stream:
            yield json.loads(line)


def _states(config, outer, rank, phase, tokenizer, selection_path):
    root = root_dir(config, outer)
    if phase == 'prefix':
        bundle = read(root / 'questions.json')
        if bundle.get('outer', outer) != outer:
            raise ValueError('Questions belong to another outer')
        groups = [dict(row) for row in bundle['rows']]
        if len(groups) != question_count(config,phase,outer) or len({r['qid'] for r in groups}) != question_count(config,phase,outer):
            raise ValueError('Candidates must cover every distinct question in the current scheduled question pool')
        normalized = [r.get('normalized_question_hash', r['question_hash']) for r in groups]
        if len(set(normalized)) != question_count(config,phase,outer):
            raise ValueError('Candidate question contents repeat')
        for i, group in enumerate(groups):
            group.setdefault('candidate_ordinal', i)
            if group['candidate_ordinal'] != i:
                raise ValueError('Candidate ordinal must preserve input order')
        return [initial_state(config, outer, row, j, tokenizer) for row in groups[rank::WORLD] for j in range(G)]
    receipt = read(root / 'prefix' / 'complete.json')
    if (receipt.get('interaction_protocol') != INTERACTION_CONTRACT
            or receipt.get('actual_compute_dtype') != ACTUAL_COMPUTE_DTYPE):
        raise ValueError('Selected continuation prefix scan used a different interaction protocol/dtype')
    groups = validate_selection(read(selection_path), receipt,question_count(config,phase,outer))
    assigned = groups[rank::WORLD]
    if len(assigned) != question_count(config,phase,outer)//WORLD:
        raise ValueError('Continuation requires equal completeG5 question shards across four GPUs')
    wanted = {tid: group for group in assigned for tid in group['trajectory_ids']}
    states = {}
    owners = {int(group['candidate_ordinal']) % WORLD for group in assigned}
    for owner in owners:
        shard = receipt['ranks'][owner]['states']
        if digest(shard['path']) != shard['sha256']:
            raise ValueError('Committed prefix states changed before continuation')
        for state in iter_rows(shard['path']):
            tid = state['trajectory_id']
            if tid not in wanted:
                continue
            group = wanted[tid]
            if (state.get('sampling_attempt_id') != config.get('sampling_attempt_id')
                    or state.get('interaction_protocol') != INTERACTION_CONTRACT
                    or state['actor_version'] != config['actor_version'] or state['group']['qid'] != group['qid']
                    or state['candidate_ordinal'] != group['candidate_ordinal']
                    or state['group']['question_hash'] != group['question_hash']):
                raise ValueError('Selected original prefix/state identity differs')
            actual = prefix_record(state)
            if actual['prefix_sha256'] != state['prefix_sha256']:
                raise ValueError('Persisted prefix state does not reproduce its score-safe prefix')
            if tid in states:
                raise ValueError('Selected prefix trajectory repeats across committed state shards')
            state['prefix_scan_sha256'] = receipt['scan_sha256']
            state['selection_sha256'] = digest(selection_path)
            states[tid] = state
    if set(states) != set(wanted):
        raise ValueError('A selected original trajectory prefix is missing')
    return [states[tid] for group in assigned for tid in group['trajectory_ids']]


def discard_hidden_lookahead(retained, actual):
    """Retain the pre-flush snapshot, never the sixth sampled token or its LP."""
    ids, lps = retained['raw_generated_token_ids'], retained['selected_logprobs']
    raw, values = actual['raw_generated_token_ids'], actual['selected_logprobs']
    extra = len(raw) - len(ids)
    if (not 0 <= extra <= 1 or raw[:len(ids)] != ids or values[:len(lps)] != lps
            or len(ids) != len(lps) or len(raw) != len(values)):
        raise ValueError('Native hidden flush changed its retained prefix snapshot')
    snapshot = dict(retained, raw_generated_token_ids=list(ids), selected_logprobs=list(lps))
    return snapshot, dict(captured_after_last_retained_token=True,
        discarded_sampled_lookahead_tokens=extra, lookahead_hidden_not_passed_to_probe=True,
        lookahead_token_not_passed_to_ppo_or_continuation=True)


def _build_llm(config, model, seed, free, total):
    """Use actual available VRAM; retry only an actual initialization OOM."""
    from vllm import LLM
    requested = config.get('rollout_memory_utilization', 'auto')
    reserve = float(config.get('rollout_free_reserve_gib', 2)) * 1024 ** 3
    cap = float(config.get('rollout_utilization_cap', .95))
    utilization = min(cap, (free - reserve) / total) if requested == 'auto' else float(requested)
    if not 0 < utilization <= 1:
        raise ValueError('Actual free VRAM cannot supply configured vLLM reservation')
    limits = dict(max_num_seqs=int(config.get('rollout_max_num_seqs', 1024)),
                  max_num_batched_tokens=int(config.get('rollout_max_batched_tokens', 32768)))
    # Constructor OOM has not sampled a prefix or written output; lowering this
    # process's cache/concurrency does not repeat any already committed sample.
    attempts = []
    while True:
        try:
            llm = LLM(model=model, tokenizer=model, trust_remote_code=False, dtype=COMPUTE_DTYPE,
                      tensor_parallel_size=1, max_model_len=BUDGET['total_tokens'], seed=seed,
                      gpu_memory_utilization=utilization, swap_space=0,
                      enforce_eager=bool(config.get('_capture_actor_hidden') or config.get('rollout_enforce_eager', False)),
                      enable_chunked_prefill=False, enable_prefix_caching=False,
                      disable_async_output_proc=True,
                      disable_log_stats=True, **limits)
            return llm, utilization, limits, attempts
        except Exception as error:
            actual_oom = 'out of memory' in str(error).lower()
            attempts.append(dict(utilization=utilization, limits=dict(limits), error=repr(error), actual_oom=actual_oom))
            if not actual_oom or not bool(config.get('rollout_retry_initialization_oom', True)) or len(attempts) >= 3:
                raise
            import gc
            import torch
            gc.collect(); torch.cuda.empty_cache()
            utilization *= .85
            limits['max_num_seqs'] = max(32, limits['max_num_seqs'] // 2)
            limits['max_num_batched_tokens'] = max(4096, limits['max_num_batched_tokens'] // 2)


def worker(config_path, outer, phase, rank, selection_path=None):
    config = binding(read(config_path), outer)
    root = root_dir(config, outer)
    output = phase_output(config,outer,phase)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / ('rank-%d-complete.json' % rank)
    if receipt_path.exists():
        saved,_reuse=validate_rank_receipt(config_path,config,outer,phase,rank,receipt_path)
        return saved
    device_binding = _configure_one_visible_gpu(rank)
    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from search_g1.async_rollout_runtime import AsyncRolloutConfig, BoundedRetriever, execute_one_action
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    model = str(Path(config['actor_model_dir']).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    selection_path = Path(selection_path or root / 'probe-score' / 'selection.json')
    states = _states(config, outer, rank, phase, tokenizer, selection_path)
    free, total = torch.cuda.mem_get_info()
    seed = seed_for(config['run_id'], outer, 'engine', 0, rank, config.get('seed', 1729),
                    sampling_attempt_id=config.get('sampling_attempt_id'))
    capture_hidden = phase == 'prefix' and config.get('actor_hidden_native', False)
    with bounded_vllm_port_namespace(config, rank, outer=outer, phase=phase) as port_namespace:
        print('VLLM_PORT_NAMESPACE=' + json.dumps(port_namespace.receipt(), sort_keys=True), flush=True)
        llm, utilization, limits, oom_attempts = _build_llm(dict(config, _capture_actor_hidden=capture_hidden), model, seed, free, total)
    engine = llm.llm_engine
    if str(engine.model_config.dtype) != ACTUAL_COMPUTE_DTYPE:
        raise ValueError('Actual rollout engine must use the shared BF16 train/QA protocol')
    hidden_capture = hidden_writer = None
    if capture_hidden:
        from .actor_hidden import ActorHiddenCapture, NativeFeatureWriter
        hidden_capture = ActorHiddenCapture(engine, config['actor_version'])
        hidden_writer = NativeFeatureWriter(output, rank, [row['trajectory_id'] for row in states])
    hidden_flushes = {}
    generation_path = Path(model) / 'generation_config.json'
    generation = read(generation_path) if generation_path.exists() else {}
    stop, terminal_ids = terminal_token_ids(generation, tokenizer)
    recipe = GenerationConfig(max_turns=MAX_SEARCH_TURNS, max_start_length=MAX_START_TOKENS,
                              max_prompt_length=BUDGET['total_tokens'], max_response_length=ACTION_TOKENS,
                              max_obs_length=MAX_OBSERVATION_TOKENS, num_gpus=WORLD,
                              search_url=config['search_url'], topk=3)
    manager = LLMGenerationManager(tokenizer, None, recipe)
    retriever = BoundedRetriever(manager, AsyncRolloutConfig(agent_workers=limits['max_num_seqs'],
        max_inflight_trajectories=limits['max_num_seqs'], retrieval_concurrency=int(config.get('retrieval_threads_per_engine', 16)),
        tool_timeout=float(config.get('retrieval_timeout', 60)), tool_retries=2, engine_step_budget=8))
    budget = dict(BUDGET)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    waiting, ready, pending, tools = deque(states), deque(), {}, {}
    completed, active, peak, requests, sampled = 0, 0, 0, 0, 0
    started = time.monotonic()
    progress_path = output / ('rank-%d-progress.json' % rank)
    names = ('states.jsonl.gz', 'prefix.jsonl.gz') if phase == 'prefix' else ('full.jsonl.gz', 'own-em.jsonl')
    paths = [output / ('rank-%d-%s' % (rank, name)) for name in names]
    if any(path.exists() for path in paths):
        raise ValueError('Uncommitted phase shard exists: preserve and reconcile it before retrying')
    pool = ThreadPoolExecutor(max_workers=retriever.config.retrieval_concurrency, thread_name_prefix='prefix-selection-tools')

    def progress(stage):
        atomic(progress_path, dict(stage=stage, phase=phase, outer=outer, rank=rank, utc=utc(),
            actor_version=config['actor_version'], completed_trajectories=completed, expected_trajectories=len(states),
            active_trajectories=active, pending_generation=len(pending), pending_retrieval=len(tools),
            peak_inflight_trajectories=peak, generation_requests=requests, actual_sampled_tokens=sampled,
            elapsed_seconds=time.monotonic()-started, device_binding=device_binding,
            actual_engine_enforce_eager=bool(engine.model_config.enforce_eager),
            actual_engine_enable_prefix_caching=bool(engine.cache_config.enable_prefix_caching),
            actual_engine_chunked_prefill_enabled=bool(engine.scheduler_config.chunked_prefill_enabled),
            gpu_memory_utilization=utilization, scheduler_limits=limits, initialization_oom_attempts=oom_attempts))

    def save(state, first_stream, second_stream):
        nonlocal completed, active
        if phase == 'prefix':
            row = prefix_record(state)
            if hidden_capture is not None:
                native_features = hidden_capture.finish(row)
                row.update(hidden_writer.write(row, native_features), feature_rank=rank)
            state['prefix_sha256'] = row['prefix_sha256']
            first_stream.write(canonical(state).decode() + '\n')
            second_stream.write(canonical(row).decode() + '\n')
        else:
            group = state['group']
            label = strict_label(tokenizer, state['response'], state['info_mask'], group['reward_model']['ground_truth'])
            row = dict(schema_version=1, run_id=config['run_id'], outer=outer, actor_version=config['actor_version'],
                policy_snapshot_id=config['actor_version'], trajectory_id=state['trajectory_id'], qid=group['qid'],
                question_hash=group['question_hash'], candidate_ordinal=state['candidate_ordinal'],
                response_index=state['response_index'], initial_prompt_token_ids=state['prompt'],
                response_token_ids=state['response'], response_info_mask=state['info_mask'],
                response_log_probs=state['log_probs'], behavior_turns=state['turns'],
                sampling_journal=[chunk for turn in state['turns'] for chunk in turn['actual_sampling_chunks']],
                sampling_contract=CONTRACT, action_boundary=BOUNDARY, interaction_protocol=INTERACTION_CONTRACT, strict_em=label,
                reward_model=group['reward_model'], data_source=group['data_source'],
                extra_info=group.get('extra_info', {}), prefix_sha256=state['prefix_sha256'],
                prefix_scan_sha256=state['prefix_scan_sha256'], selection_sha256=state['selection_sha256'],
                retrieval_calls=state['retrieval_calls'], terminal_reason=state['terminal_reason'])
            first_stream.write(canonical(row).decode() + '\n')
            second_stream.write(canonical({key: row[key] for key in ('trajectory_id', 'qid', 'strict_em',
                'actor_version', 'prefix_sha256', 'prefix_scan_sha256', 'selection_sha256')}).decode() + '\n')
        completed += 1; active -= 1
        if completed % 100 == 0:
            first_stream.flush(); second_stream.flush(); progress('NATIVE_PHASE_ADVANCING')

    def schedule(state, streams):
        if state['terminal'] or (phase == 'prefix' and retained_length(state) >= PREFIX_TOKENS):
            save(state, *streams)
        else:
            ready.append(state)

    def environment_done(state, cut, result, streams):
        finish_action(state, cut, result, tokenizer, budget)
        schedule(state, streams)

    try:
        opener2 = gzip.open if phase == 'prefix' else open
        with gzip.open(paths[0], 'xt', encoding='utf8', compresslevel=1) as first_stream, \
             opener2(paths[1], 'xt', encoding='utf8') as second_stream:
            streams = first_stream, second_stream
            progress('NATIVE_ENGINE_READY')
            while completed < len(states):
                while waiting and active < limits['max_num_seqs']:
                    state = waiting.popleft(); active += 1; schedule(state, streams)
                peak = max(peak, active)
                while ready:
                    state = ready.popleft(); start_action(state, budget)
                    request = next_request(state, phase, config)
                    if request is None:
                        raise ValueError('Unfinished logical action has no available native quota')
                    params = SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1,
                        max_tokens=request['max_tokens'] + int(capture_hidden), stop=[], stop_token_ids=stop, ignore_eos=False,
                        logprobs=1, detokenize=False, seed=request['seed'])
                    engine.add_request(request['request_id'], {'prompt_token_ids': request['actual_request_token_ids']}, params)
                    sampling = registered_sampling(engine, request, params)
                    if hidden_capture is not None:
                        hidden_capture.register(state, request)
                    pending[request['request_id']] = state, request, sampling
                    requests += 1
                if engine.has_unfinished_requests():
                    for result in engine.step():
                        rid = str(result.request_id)
                        if rid not in pending:
                            continue
                        state, request, sampling = pending[rid]
                        if len(result.outputs) != 1 or list(result.prompt_token_ids) != request['actual_request_token_ids']:
                            raise ValueError('Actual native output request identity changed')
                        native = result.outputs[0]; raw = list(native.token_ids)
                        cumulative = state['action']['raw_generated_token_ids'] + raw
                        seen_action = stream_action_present(cumulative, decode)
                        # Processing the final retained token requires one more
                        # native forward than sampling it. Freeze the boundary
                        # snapshot, run one decode step, and discard its sampled
                        # lookahead. That token never enters the state or head.
                        flushing = rid in hidden_flushes
                        if capture_hidden and not flushing and not result.finished and (
                                seen_action or len(raw) >= request['max_tokens']):
                            hidden_flushes[rid] = native_snapshot(native)
                            continue
                        aborted = bool(not result.finished and (seen_action or flushing))
                        if not flushing and not result.finished and not seen_action:
                            continue
                        actual_snapshot = native_snapshot(native)
                        snapshot = actual_snapshot
                        if flushing:
                            snapshot, flush_receipt = discard_hidden_lookahead(hidden_flushes.pop(rid), actual_snapshot)
                            sampling = dict(sampling, actor_hidden_flush=flush_receipt)
                        if aborted:
                            engine.abort_request(rid)
                        if hidden_capture is not None:
                            hidden_capture.forget_request(rid)
                        pending.pop(rid); sampled += len(actual_snapshot['raw_generated_token_ids'])
                        cut = append_chunk(state, request, snapshot['raw_generated_token_ids'], snapshot['selected_logprobs'], sampling,
                            finish_reason=snapshot['finish_reason'],
                            stop_reason=snapshot['stop_reason'], terminal_ids=terminal_ids, decode=decode,
                            action_boundary_abort=aborted)
                        if cut is None:
                            schedule(state, streams)
                        elif cut['native_terminal']:
                            environment_done(state, cut, ('', True, int(cut['action'] is not None), 0), streams)
                        elif cut['action'] == 'search' and not state['action']['final_turn']:
                            future = pool.submit(execute_one_action, manager, cut['execution_text'], True, retriever)
                            tools[future] = state, cut
                        else:
                            environment_done(state, cut, execute_one_action(manager, cut['execution_text'],
                                             not state['action']['final_turn'], retriever), streams)
                done = [future for future in tools if future.done()]
                for future in done:
                    state, cut = tools.pop(future)
                    environment_done(state, cut, future.result(), streams)
                if not ready and not engine.has_unfinished_requests() and tools and not done:
                    time.sleep(.001)
            if waiting or ready or pending or tools:
                raise ValueError('Completed native phase leaves incomplete own trajectories')
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        if hidden_capture is not None:
            hidden_capture.close()
    feature_path = hidden_writer.finish() if hidden_writer is not None else None
    result = dict(schema_version=1, phase=phase, outer=outer, rank=rank, run_id=config['run_id'],
        actor_version=config['actor_version'], actor_model_dir=model, config_sha256=digest(config_path),
        sampling_contract=CONTRACT, action_boundary=BOUNDARY, interaction_protocol=INTERACTION_CONTRACT,
        actual_compute_dtype=str(engine.model_config.dtype), sampling_attempt_id=config.get('sampling_attempt_id'),
        vllm_port_namespace=port_namespace.receipt(),
        native_output_commit='synchronous_token_ids_and_selected_logprobs_snapshot_before_action_abort',
        trajectories=completed, questions=completed//G, device_binding=device_binding,
        pad_token_id=int(tokenizer.pad_token_id), eos_token_id=int(tokenizer.eos_token_id),
        native_terminal_ids=terminal_ids, gpu_memory_utilization=utilization,
        actual_engine_enforce_eager=bool(engine.model_config.enforce_eager),
        actual_engine_enable_prefix_caching=bool(engine.cache_config.enable_prefix_caching),
        actual_engine_chunked_prefill_enabled=bool(engine.scheduler_config.chunked_prefill_enabled),
        initial_free_gpu_bytes=free, initial_total_gpu_bytes=total, initialization_oom_attempts=oom_attempts,
        scheduler_limits=limits, engine_seed=seed, actual_sampled_tokens=sampled, generation_requests=requests,
        elapsed_seconds=time.monotonic()-started, completed_utc=utc())
    for key, path in zip(('states', 'prefix') if phase == 'prefix' else ('full', 'own_em'), paths):
        result[key] = file_item(path, completed)
    if feature_path is not None:
        result.update(actor_hidden=file_item(feature_path, completed),
            actor_hidden_layer=27, actor_hidden_boundary='post_block_hidden_plus_residual',
            actor_hidden_forward_calls=hidden_capture.calls, actor_hidden_captured_rows=hidden_capture.captured_rows,
            actor_hidden_forward_and_capture_seconds=hidden_capture.elapsed_seconds,
            independent_encoder_forward=False, native_hidden_flush=True)
    if phase == 'continue':
        result.update(selection_sha256=digest(selection_path), prefix_scan_sha256=read(root/'prefix'/'complete.json')['scan_sha256'])
    immutable(receipt_path, result); progress('NATIVE_RANK_PHASE_COMPLETE')
    return result


def pack_prefix(config, outer, ranks):
    """CPU pack true score inputs, never cached labels or future suffix states."""
    import numpy as np
    if len(ranks) != WORLD or [row['rank'] for row in ranks] != list(range(WORLD)):
        raise ValueError('Prefix scan requires exactly four ordered native rank receipts')
    root = root_dir(config, outer); out = root/'prefix'/'scan'
    out.mkdir(parents=True, exist_ok=False)
    rows, contexts, policies, positions, offsets = [], [], [], [], [0]
    seen, groups = set(), {}
    for rank in ranks:
        for original in iter_rows(rank['prefix']['path']):
            tid, qid = original['trajectory_id'], original['qid']
            if tid in seen or original['actor_version'] != config['actor_version']:
                raise ValueError('Duplicate trajectory or changed prefix actor')
            if 'strict_em' in original or 'reward_model' in original:
                raise ValueError('Score-safe prefix input contains forbidden future labels/gold')
            seen.add(tid); groups.setdefault(qid, []).append(original)
            prompt = original['initial_prompt_token_ids']
            if not 1 <= len(prompt) <= QMAX:
                raise ValueError('Native question must fit the unchanged273-token head layout')
            initial = dict(context_offset=len(contexts), context_length=len(prompt), request_length=len(prompt))
            contexts.extend(prompt)
            segments = []
            for source in original['prefix_segments']:
                segment = dict(source); tokens = segment.pop('context_tokens')
                segment.update(context_offset=len(contexts), context_length=len(tokens),
                               context_sha256=hashlib.sha256(canonical(tokens)).hexdigest())
                contexts.extend(tokens); segments.append(segment)
            ids, pos = original['policy_token_ids'], original['policy_position_ids']
            if (len(ids) != len(pos) or not 0 <= len(ids) <= PREFIX_TOKENS
                    or (len(ids) != PREFIX_TOKENS and not original['terminal'])):
                raise ValueError('Prefix scan includes a future token or an unfinished short prefix')
            row = {key: original[key] for key in ('trajectory_id', 'qid', 'question_hash', 'candidate_ordinal',
                'response_index', 'actor_version', 'prefix_sha256', 'terminal', 'terminal_reason')}
            row.update(row=len(rows), split='train', length=len(ids), prefix_length=len(ids),
                policy_offset=len(policies), initial_context=initial, segments=segments,
                segment_bounds=[{key:s[key] for key in ('turn_index','policy_start','policy_length','position_start')} for s in segments],
                outer=outer, policy_prefix_sha256=hashlib.sha256(canonical(ids)).hexdigest(),
                position_ids_sha256=hashlib.sha256(canonical(pos)).hexdigest())
            if config.get('actor_hidden_native'):
                row.update({key:original[key] for key in ('feature_rank','feature_row','feature_file',
                    'feature_actor_version','feature_layer','feature_boundary','feature_dtype')})
            if 'normalized_question_hash' in original:
                row['normalized_question_hash'] = original['normalized_question_hash']
            rows.append(row); policies.extend(ids); positions.extend(pos); offsets.append(len(policies))
    if len(rows) != question_count(config,'prefix',outer)*5 or len(groups) != question_count(config,'prefix',outer):
        raise ValueError('Prefix scan requires all current candidate completeG5 groups')
    for siblings in groups.values():
        if len(siblings) != G or {r['response_index'] for r in siblings} != set(range(G)) or len({r['question_hash'] for r in siblings}) != 1:
            raise ValueError('Prefix scan group membership differs')
    arrays = dict(packed_context_ids=np.asarray(contexts,dtype=np.int64),
                  packed_policy_ids=np.asarray(policies,dtype=np.int64),
                  packed_policy_positions=np.asarray(positions,dtype=np.int64),
                  policy_offsets=np.asarray(offsets,dtype=np.int64))
    for name, value in arrays.items():
        np.save(out/(name+'.npy'), value, allow_pickle=False)
    with (out/'index.jsonl').open('x',encoding='utf8') as stream:
        for row in rows:
            stream.write(canonical(row).decode()+'\n')
        stream.flush(); os.fsync(stream.fileno())
    source_sha = hashlib.sha256(canonical([rank['prefix'] for rank in ranks])).hexdigest()
    qmax = max(r['initial_context']['request_length'] for r in rows)
    proof = dict(schema_version=3, role='prefix_selection200_score_safe_actual_token_scan',
        representation='initial_request_and_retained_policy20_layer27_postblock_v1',
        encoder_revision=REVISION, actor_hidden_native=bool(config.get('actor_hidden_native')),
        actor_hidden_shards=[r['actor_hidden'] for r in ranks] if config.get('actor_hidden_native') else None,
        hidden_feature_actor_version=config['actor_version'] if config.get('actor_hidden_native') else None,
        independent_encoder_forward=not config.get('actor_hidden_native', False),
        run_id=config['run_id'], outer=outer, actor_version=config['actor_version'],
        actor_model_dir=config['actor_model_dir'], policy_actor_model_dir=config['actor_model_dir'],
        policy_actor_files_sha256=config.get('actor_model_files_sha256'),
        model_dir=config['actor_model_dir'] if config.get('actor_hidden_native') else config.get('encoder_model_dir',config.get('model_dir')),
        model_files_sha256=config.get('model_files_sha256'), count=len(rows), split_counts=dict(train=len(rows),test=0),
        encoder_files_sha256=config.get('actor_model_files_sha256') if config.get('actor_hidden_native') else config.get('model_files_sha256'),
        pad_token_id=ranks[0]['pad_token_id'], question_max_tokens=qmax,
        head_max_tokens=WIDTH if config.get('actor_hidden_native') else PREFIX_TOKENS,
        max_policy_prefix_tokens=PREFIX_TOKENS,
        encoder_global_max_tokens=max([qmax]+[s['context_length'] for row in rows for s in row['segments']]),
        sources_sha256=source_sha, prefix_window='multi_turn_actual_kept_policy_tokens',
        prefix_source='same_trajectory_native_prefix_then_continue', labels_present=False,
        sampling_contract=CONTRACT, action_boundary=BOUNDARY, interaction_protocol=INTERACTION_CONTRACT,
        **schedule_metadata(config, outer),
        files={name:dict(sha256=digest(out/name),bytes=(out/name).stat().st_size) for name in
               tuple(name+'.npy' for name in arrays)+('index.jsonl',)})
    immutable(out/'scan-complete.json',proof)
    return out/'scan-complete.json'


def aggregate(config_path, outer, phase, selection_path=None, worker_exit_codes=None):
    config = binding(read(config_path), outer); root=root_dir(config,outer)
    output=phase_output(config,outer,phase)
    target=output/'complete.json'
    if target.exists():
        value=read(target)
        if (value['actor_version']!=config['actor_version'] or value['config_sha256']!=digest(config_path)
                or value.get('interaction_protocol')!=INTERACTION_CONTRACT
                or value.get('actual_compute_dtype')!=ACTUAL_COMPUTE_DTYPE
                or value.get('sampling_attempt_id')!=config.get('sampling_attempt_id')):
            raise ValueError('Existing phase receipt belongs to a different actor/config')
        return value
    checked=[validate_rank_receipt(config_path,config,outer,phase,rank,output/('rank-%d-complete.json'%rank))
             for rank in range(WORLD)]
    ranks=[item[0] for item in checked]
    expected=question_count(config,phase,outer)*G//WORLD
    for rank,value in enumerate(ranks):
        if value['rank']!=rank or value['phase']!=phase or value['outer']!=outer or value['trajectories']!=expected or value['actor_version']!=config['actor_version']:
            raise ValueError('Native phase rank completion identity/count differs')
        for key in ('states','prefix') if phase=='prefix' else ('full','own_em'):
            item=value[key]
            if digest(item['path'])!=item['sha256'] or Path(item['path']).stat().st_size!=item['bytes']:
                raise ValueError('Native committed phase output differs')
    value=dict(schema_version=1,run_id=config['run_id'],outer=outer,phase=phase,actor_version=config['actor_version'],
        actor_model_dir=config['actor_model_dir'],config_sha256=digest(config_path),ranks=ranks,
        model_revision=REVISION,pad_token_id=ranks[0]['pad_token_id'],eos_token_id=ranks[0]['eos_token_id'],
        candidate_questions_sha256=digest(root/'questions.json'),head_prefix_tokens=PREFIX_TOKENS,
        trajectories=sum(r['trajectories'] for r in ranks),questions=sum(r['questions'] for r in ranks),
        sampling_contract=CONTRACT,action_boundary=BOUNDARY,interaction_protocol=INTERACTION_CONTRACT,
        **schedule_metadata(config, outer),
        actual_compute_dtype=ACTUAL_COMPUTE_DTYPE, sampling_attempt_id=config.get('sampling_attempt_id'),
        actual_sampled_tokens=sum(r['actual_sampled_tokens'] for r in ranks),
        native_generation_requests=sum(r['generation_requests'] for r in ranks),completed_utc=utc())
    value['reused_original_rank_receipts']=[item[1] for item in checked if item[1] is not None]
    if worker_exit_codes is not None:
        value['owned_native_worker_exit_codes']=worker_exit_codes
    if phase=='prefix':
        scan=pack_prefix(config,outer,ranks)
        value.update(scan_path=str(scan.parent),scan_sha256=digest(scan),
                     shards=[r['prefix'] for r in ranks],state_shards=[r['states'] for r in ranks])
    else:
        selection_path=Path(selection_path or root/'probe-score'/'selection.json')
        prefix=read(root/'prefix'/'complete.json')
        groups=validate_selection(read(selection_path),prefix,question_count(config,phase,outer))
        wanted={tid:group['qid'] for group in groups for tid in group['trajectory_ids']}
        seen=set(); all_labels=[]
        for rank in ranks:
            for row in iter_rows(rank['own_em']['path']):
                tid=row['trajectory_id']
                if (tid in seen or wanted.get(tid)!=row['qid'] or row['strict_em'] not in (0,1)
                    or row['actor_version']!=config['actor_version'] or row['selection_sha256']!=digest(selection_path)):
                    raise ValueError('Completed selected ownEM identity or label differs')
                seen.add(tid);all_labels.append(row)
        if seen!=set(wanted):
            raise ValueError('A selected original G5 trajectory did not finish')
        own=output/'own-em.jsonl'
        with own.open('x',encoding='utf8') as stream:
            for row in all_labels:stream.write(canonical(row).decode()+'\n')
            stream.flush();os.fsync(stream.fileno())
        labels={row['trajectory_id']:row['strict_em'] for row in all_labels}
        mixed=sum(0<sum(labels[t] for t in group['trajectory_ids'])<G for group in groups)
        value.update(own_em=file_item(own,len(all_labels)),shards=[r['full'] for r in ranks],
            selection_sha256=digest(selection_path),prefix_scan_sha256=prefix['scan_sha256'],
            actor_selection_path=str(selection_path.resolve()),actor_selection_sha256=digest(selection_path),
            actual_selected_mixed_questions=mixed,actual_selected_mixed_fraction=mixed/question_count(config,phase,outer),
            all_candidate_mixed_accuracy_available=question_count(config,phase,outer)==candidate_count(config,outer),
            selected_strict_correct=sum(row['strict_em'] for row in all_labels))
    immutable(target,value)
    return value


def launch_phase(config_path, outer, phase, selection_path=None):
    """Own exactly4 native workers; phase completion includes their actual exit."""
    config=binding(read(config_path),outer);output=phase_output(config,outer,phase)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():
        return aggregate(config_path,outer,phase,selection_path)
    processes=[];streams=[]
    try:
        for rank in range(WORLD):
            existing=output/('rank-%d-complete.json'%rank)
            if existing.exists():
                validate_rank_receipt(config_path,config,outer,phase,rank,existing)
                continue
            stream=(output/('rank-%d.log'%rank)).open('ab');streams.append(stream)
            command=[sys.executable,'-m','reproduction.actor50_prefix20_fresh100.rollout','--config',str(config_path),
                     '--outer',str(outer),'--phase',phase,'--rank',str(rank)]
            if selection_path:command.extend(['--selection',str(selection_path)])
            processes.append((rank,subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)))
        while any(p.poll() is None for _,p in processes):
            failed={rank:p.returncode for rank,p in processes if p.poll() not in (None,0)}
            if failed:raise RuntimeError('Native phase worker failure: '+json.dumps(failed))
            time.sleep(1)
        codes={rank:p.returncode for rank,p in processes}
        if any(code!=0 for code in codes.values()):raise RuntimeError('Native phase worker failure: '+json.dumps(codes))
    finally:
        for _,p in processes:
            if p.poll() is None:p.terminate()
        deadline=time.monotonic()+60
        for _,p in processes:
            if p.poll() is None:
                try:p.wait(timeout=max(.01,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for stream in streams:stream.close()
    result=aggregate(config_path,outer,phase,selection_path,
                     worker_exit_codes={str(rank):p.returncode for rank,p in processes})
    # The immutable phase receipt is written after every worker actually exits.
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--outer',required=True,type=int)
    parser.add_argument('--phase',choices=('prefix','continue'),required=True)
    parser.add_argument('--rank',type=int);parser.add_argument('--selection')
    args=parser.parse_args()
    if args.outer<1 or (args.rank is not None and not 0<=args.rank<WORLD):parser.error('Invalid actual outer/rank')
    function=launch_phase if args.rank is None else worker
    if args.rank is None:result=function(Path(args.config).resolve(),args.outer,args.phase,args.selection)
    else:result=function(Path(args.config).resolve(),args.outer,args.phase,args.rank,args.selection)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
