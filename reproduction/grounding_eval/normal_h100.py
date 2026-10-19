"""NQ-only normal evaluation for historical Easy/OPSD HF exports.

Run this FILE in a fresh interpreter, with the historical runtime repo supplied.
Easy retains authored token prefixes; OPSD uses its original retokenizing
SearchRollout. Neither adapter creates an optimizer or imports a trainer object.
GPU initialization is deliberately inside main, after identity/lock validation.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'wb') as stream:
        stream.write(canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(str(path.parent), os.O_DIRECTORY); os.fsync(fd); os.close(fd)


def plain(value):
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def response_spans(tokenizer, tokens, mask):
    spans, active = [], []
    for token, valid in zip(tokens, mask):
        if valid:
            active.append(int(token))
        elif active:
            spans.append(tokenizer.decode(active, skip_special_tokens=False, clean_up_tokenization_spaces=False)); active = []
    if active:
        spans.append(tokenizer.decode(active, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    return spans


def token_accounting(row):
    """Actual sampled output includes discarded suffixes; inputs count repeats."""
    generated = inputs = observations = corrections = kept = 0
    for turn in row['turns']:
        raw = turn['raw_response_token_ids']
        count = sum(turn['raw_response_attention_mask']) if 'raw_response_attention_mask' in turn else len(raw)
        input_count = len(turn['generation_context_token_ids'])
        output_count = len(turn['postprocessed_action_token_ids'])
        environment_count = len(turn['retained_observation_token_ids'])
        search = turn.get('actual_search', bool(turn.get('valid_search') and turn.get('normal_search_round')))
        turn.update(actual_input_tokens=input_count, actual_generated_tokens=count,
                    retained_model_tokens=output_count, retained_environment_tokens=environment_count)
        generated += count; inputs += input_count; kept += output_count
        if search:
            observations += environment_count
        else:
            corrections += environment_count
    initial = len(row['initial_prompt_token_ids'])
    row.update(model_generated_tokens=generated, observation_tokens=observations,
        environment_correction_tokens=corrections, retained_model_tokens=kept,
        initial_prompt_tokens=initial, trajectory_total_tokens=initial+kept+observations+corrections,
        cumulative_model_input_tokens=inputs, api_cache_tokens=None,
        api_cache_tokens_null_reason='local_vllm_does_not_report_per_request_cache_tokens')
    return row


def gpu_free(index):
    import subprocess
    result = subprocess.run(['nvidia-smi', '-i', str(index), '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


class LocalVLLM:
    """TP1 inference-only worker for the historical OPSD controller."""
    def __init__(self, llm, tokenizer, sampling):
        self.llm, self.tokenizer, self.sampling = llm, tokenizer, sampling
        self.last = []

    def decode(self, requests):
        # Sorting is unnecessary: vLLM performs its own bounded continuous batch.
        self.last = self.llm.generate(prompt_token_ids=requests, sampling_params=self.sampling,
                                     use_tqdm=False)
        return self.last

    def generate_sequences(self, batch):
        import torch
        from verl import DataProto
        requests = [ids[mask.bool()].tolist() for ids, mask in
                    zip(batch.batch['input_ids'], batch.batch['attention_mask'])]
        outputs = self.decode(requests)
        rows = [list(o.outputs[0].token_ids) for o in outputs]
        width = max(1, max(map(len, rows)))
        responses = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(responses)
        for i, row in enumerate(rows):
            responses[i, :len(row)] = torch.tensor(row); attention[i, :len(row)] = 1
        return DataProto.from_dict(dict(responses=responses, attention_mask=attention))


def easy_normal(manager, worker, prompts, tokenizer, config, raw_cut):
    """Same old Easy raw-prefix/500 action/500 observation/four+final loop.

    Provenance checking unrelated to inference-only LP transport is omitted;
    actual vLLM request/response/finish, raw cut, retained observations are saved.
    Context overflow rejects rather than silently switching to a new protocol.
    """
    import torch
    pad = tokenizer.pad_token_id
    states = [dict(prompt=list(p[-config.max_start_length:]), response=[], mask=[],
                   turns=[], done=False, actual_calls=0) for p in prompts]
    stops = set(worker.sampling.stop_token_ids or []) | {tokenizer.eos_token_id}
    for ordinal in range(config.max_turns + 1):
        indices = [i for i, state in enumerate(states) if not state['done']]
        if not indices:
            break
        final = ordinal == config.max_turns
        requests = [states[i]['prompt'] + states[i]['response'] for i in indices]
        if any(len(x) > config.max_prompt_length for x in requests):
            raise ValueError('Historical Easy rolling_window_would_change_actor_context')
        outputs = worker.decode(requests)
        texts, cuts = [''] * len(states), {}
        for i, output in zip(indices, outputs):
            item = output.outputs[0]
            cut = raw_cut(list(item.token_ids), lambda ids: tokenizer.decode(ids,
                skip_special_tokens=False, clean_up_tokenization_spaces=False),
                eos_token_id=tokenizer.eos_token_id, pad_token_id=pad,
                special_token_ids=tokenizer.all_special_ids,
                executor_decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False), effective_stop_token_ids=sorted(stops))
            texts[i], cuts[i] = cut['execution_text'], cut
        active = torch.tensor([i in indices for i in range(len(states))], dtype=torch.bool)
        observations, dones, valid, searches = manager.execute_predictions(texts,
            tokenizer.pad_token, active, do_search=not final)
        obs_tensor = None if final else manager._process_next_obs(observations)
        for i, output, request in zip(indices, outputs, requests):
            item, state, cut = output.outputs[0], states[i], cuts[i]
            obs = [] if final else [int(x) for x in obs_tensor[i].tolist() if int(x) != pad]
            kept = cut['kept_token_ids']
            if len(state['response']) + len(kept) + len(obs) > config.max_prompt_length:
                raise ValueError('Historical Easy response_prefix_budget_would_drop_actual_tokens')
            state['turns'].append(dict(turn_index=ordinal, generation_context_token_ids=request,
                raw_response_token_ids=list(item.token_ids), postprocessed_action_token_ids=kept,
                retained_observation_token_ids=obs, observation_text_before_truncation=observations[i],
                actual_search=bool(searches[i] and not final), final_only=final,
                finish_reason=item.finish_reason, stop_reason=item.stop_reason, prefix_cut=cut))
            state['response'].extend(kept + obs)
            state['mask'].extend([1] * len(kept) + [0] * len(obs))
            state['actual_calls'] += int(bool(searches[i] and not final))
            state['done'] = bool(dones[i]) or final
    return [dict(model_spans=response_spans(tokenizer, s['response'], s['mask']),
        seen_docs=[tokenizer.decode(t['retained_observation_token_ids'], skip_special_tokens=False,
            clean_up_tokenization_spaces=False) for t in s['turns'] if t['actual_search']],
        actual_retrieval_calls=s['actual_calls'], turns=s['turns'],
        response_token_ids=s['response'], model_mask=s['mask'], initial_prompt_token_ids=s['prompt']) for s in states]


def opsd_normal(manager, worker, prompts, tokenizer, config, engine_class):
    """Use the historical OPSD SearchRollout unchanged, capture_raw enabled."""
    import torch
    from verl import DataProto
    width = max(map(len, prompts)); pad = tokenizer.pad_token_id
    ids = torch.full((len(prompts), width), pad, dtype=torch.long)
    for i, row in enumerate(prompts):
        ids[i, -len(row):] = torch.tensor(row)
    mask = (ids != pad).long()
    batch = DataProto.from_dict(dict(input_ids=ids, attention_mask=mask,
        position_ids=manager.tensor_fn.create_position_ids(mask)), meta_info=dict(do_sample=False,
        recompute_log_prob=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=pad, validate=True))
    engine = engine_class(manager, context_alignment='report', capture_raw=True)
    output, owners = engine.run(batch)
    if list(owners) != list(range(len(prompts))):
        raise ValueError('Historical OPSD unexpectedly changed source order')
    rows = []
    for state in engine.last_states:
        rows.append(dict(model_spans=response_spans(tokenizer, state.response, state.model_mask),
            seen_docs=[tokenizer.decode(t['retained_observation_token_ids'], skip_special_tokens=False,
                clean_up_tokenization_spaces=False) for t in state.records
                if t['valid_search'] and t['normal_search_round']],
            actual_retrieval_calls=sum(int(t['valid_search'] and t['normal_search_round']) for t in state.records),
            turns=state.records, response_token_ids=state.response, model_mask=state.model_mask,
            initial_prompt_token_ids=state.prompt))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['easy', 'opsd'], required=True)
    p.add_argument('--runtime-repo', required=True)
    p.add_argument('--source-config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--retrieval-url', required=True)
    p.add_argument('--identity-file', help='Historical Easy final-result.json question_ids, required for Easy')
    p.add_argument('--gpu-index', type=int, required=True)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--max-num-seqs', type=int, default=128)
    p.add_argument('--max-num-batched-tokens', type=int, default=8192)
    p.add_argument('--gpu-memory-utilization', type=float, default=.25)
    p.add_argument('--min-start-free-gib', type=float, default=48.)
    p.add_argument('--min-running-free-gib', type=float, default=12.)
    p.add_argument('--limit', type=int, default=3610)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    lease = open(out / '.normal.lock', 'a+'); fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.method == 'easy' and not args.identity_file:
        raise ValueError('Easy requires original final-result occurrence identity')
    if not (0 < args.batch_size <= 1024 and 0 < args.limit <= 3610):
        raise ValueError('Invalid batch or NQ limit')
    if gpu_free(args.gpu_index) < args.min_start_free_gib * 1024:
        raise RuntimeError('Insufficient fresh free memory before model load')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    runtime = str(Path(args.runtime_repo).resolve())
    sys.path.insert(0, runtime)
    import pandas as pd
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    historical_generation = importlib.import_module('search_r1.llm_agent.generation')
    if not str(Path(historical_generation.__file__).resolve()).startswith(runtime + '/'):
        raise ValueError('Historical runtime import shadowed by another checkout')
    config_doc = json.loads(Path(args.source_config).read_text())
    config = OmegaConf.create(config_doc.get('resolved_config', config_doc))
    expected = dict(max_turns=int(config.max_turns), max_start_length=int(config.data.max_start_length),
        max_prompt_length=int(config.data.max_prompt_length), max_response_length=int(config.data.max_response_length),
        max_obs_length=int(config.data.max_obs_length), num_gpus=1, no_think_rl=False,
        search_url=args.retrieval_url, topk=int(config.retriever.topk))
    if expected['max_turns'] != 4 or expected['max_response_length'] != 500 or expected['topk'] != 3:
        raise ValueError('Historical normal protocol differs from reviewed adapter')
    generation = GenerationConfig(**{k: v for k, v in expected.items() if k in GenerationConfig.__annotations__})
    generation.force_final_answer = bool(config.get('force_final_answer', True))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    frame = pd.read_parquet(args.data)
    indices = [i for i, value in enumerate(frame['data_source']) if value == 'nq']
    if len(indices) != 3610:
        raise ValueError('Expected fixed 3610 NQ occurrences in original final parquet')
    original_ids = json.loads(Path(args.identity_file).read_text()) if args.identity_file else None
    source_files = dict(generation=sha(historical_generation.__file__))
    if args.method == 'easy':
        trace = importlib.import_module('reproduction.common8.behavior_trace')
        if not str(Path(trace.__file__).resolve()).startswith(runtime + '/'):
            raise ValueError('Historical Easy trace imported from another checkout')
        source_files['behavior_trace'] = sha(trace.__file__)
    else:
        rollout = importlib.import_module('search_r1.baselines.rollout')
        if not str(Path(rollout.__file__).resolve()).startswith(runtime + '/'):
            raise ValueError('Historical OPSD rollout imported from another checkout')
        source_files['rollout'] = sha(rollout.__file__)
    normal_config = dict(method=args.method, run_id=args.run_id, checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_files={x.name: sha(x) for x in Path(args.checkpoint).iterdir() if x.is_file()
            and x.name.endswith(('.json', '.safetensors', '.txt')) and not x.name.startswith('backup-')},
        data_sha256=sha(args.data), source_config_sha256=sha(args.source_config),
        normal_protocol=expected, runtime_sources_sha256=source_files,
        greedy=True, dtype='bfloat16', adapter_sha256=sha(__file__),
        inference_batch=args.batch_size, max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens, gpu_memory_utilization=args.gpu_memory_utilization,
        nq_occurrences=indices, limit=args.limit)
    binding_hash = hashlib.sha256(canonical(normal_config)).hexdigest()
    binding_path = out / 'normal-binding.json'
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != normal_config:
            raise ValueError('Existing normal binding mismatch; use a new output directory')
    else:
        atomic(binding_path, normal_config)
    gc = json.loads((Path(args.checkpoint) / 'generation_config.json').read_text())
    stops = gc.get('eos_token_id', tokenizer.eos_token_id)
    stops = stops if isinstance(stops, list) else [stops]
    sampling = SamplingParams(temperature=0., top_p=1., top_k=-1, max_tokens=500,
        stop_token_ids=stops, skip_special_tokens=False, spaces_between_special_tokens=False,
        detokenize=False, ignore_eos=False)
    llm = LLM(model=args.checkpoint, tokenizer=args.checkpoint, tensor_parallel_size=1,
        dtype='bfloat16', trust_remote_code=False, max_model_len=max(8192, generation.max_prompt_length + 500),
        gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True,
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
        disable_log_stats=True)
    worker = LocalVLLM(llm, tokenizer, sampling)
    manager = LLMGenerationManager(tokenizer, worker, generation, is_validation=True)
    # No injected text or docs enters this normal rerun. Original prompts only.
    completed = 0; started = time.time(); total = min(args.limit, 3610)
    for start in range(0, total, args.batch_size):
        selected = indices[start:min(total, start + args.batch_size)]
        stem = 'batch-%06d-%06d' % (start, start + len(selected))
        raw_path, norm_path = out / 'raw' / (stem + '.json'), out / 'normalized' / (stem + '.json')
        if raw_path.exists() and norm_path.exists():
            saved = json.loads(norm_path.read_text())
            if saved['binding_sha256'] != binding_hash or saved['raw_sha256'] != sha(raw_path):
                raise ValueError('Completed batch integrity mismatch')
            completed += len(selected); continue
        if raw_path.exists() and not norm_path.exists():
            saved = json.loads(raw_path.read_text())
            if saved['binding_sha256'] != binding_hash or len(saved['rows']) != len(selected):
                raise ValueError('Incomplete raw batch identity mismatch')
            raw_sha = sha(raw_path)
            atomic(norm_path, dict(binding_sha256=binding_hash, raw_sha256=raw_sha,
                rows=[dict(row, source_shard_sha256=raw_sha) for row in saved['rows']]))
            completed += len(selected); continue
        if norm_path.exists():
            raise ValueError('Normalized batch exists without its original raw source')
        if gpu_free(args.gpu_index) < args.min_running_free_gib * 1024:
            raise RuntimeError('Shared GPU free memory below running reserve; completed batches retained')
        prompts, identities = [], []
        for ordinal in selected:
            row = frame.iloc[ordinal]; chat = plain(row[config.data.prompt_key])
            text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False) if tokenizer.chat_template else chat[0]['content']
            prompt = tokenizer.encode(text, add_special_tokens=False)
            if len(prompt) > generation.max_prompt_length:
                raise ValueError('Original prompt exceeds historical error-truncation budget')
            prompts.append(prompt)
            gold = plain(row['reward_model'])['ground_truth']['target']
            gold = [gold] if isinstance(gold, str) else gold
            prompt_text = '\n'.join(turn['content'] for turn in chat)
            match = re.search(r'Question:\s*(.*)', prompt_text, re.S)
            question = match.group(1).strip() if match else prompt_text
            extra = plain(row.get('extra_info', {}))
            qid = str(extra.get('id', extra.get('index', hashlib.sha256(canonical([chat, gold])).hexdigest())))
            if original_ids is not None:
                if original_ids['data_sources'][ordinal] != 'nq':
                    raise ValueError('Historical final occurrence source mismatch')
                qid = str(original_ids['question_ids'][ordinal])
            identities.append(dict(data_source='nq', evaluation_ordinal=ordinal, qid=qid,
                question=question, gold=gold, original_prompt=chat, source_run_id=args.run_id))
        t0 = time.time()
        if args.method == 'easy':
            generated = easy_normal(manager, worker, prompts, tokenizer, generation, trace.raw_prefix_to_first_closing_tag)
        else:
            generated = opsd_normal(manager, worker, prompts, tokenizer, generation, rollout.SearchRollout)
        raw = [token_accounting(dict(identity, **value)) for identity, value in zip(identities, generated)]
        atomic(raw_path, dict(binding_sha256=binding_hash, rows=raw, completed_at=time.time()))
        raw_sha = sha(raw_path)
        normalized = [dict(row, source_shard_sha256=raw_sha) for row in raw]
        atomic(norm_path, dict(binding_sha256=binding_hash, raw_sha256=raw_sha, rows=normalized))
        completed += len(selected)
        progress = dict(state='RUNNING', method=args.method, pid=os.getpid(), completed=completed,
            target=total, updated_at=time.time(), latest_batch_seconds=time.time()-t0,
            total_seconds=time.time()-started, binding_sha256=binding_hash,
            unique_occurrences=completed, raw_bytes=sum(p.stat().st_size for p in (out/'raw').glob('*.json')))
        atomic(out/'progress.json', progress); print(json.dumps(progress), flush=True)
    combined = out/'normal.jsonl'; temporary = out/'normal.jsonl.tmp'
    with open(temporary, 'wb') as stream:
        for path in sorted((out/'normalized').glob('*.json')):
            for row in json.loads(path.read_text())['rows']:
                stream.write(canonical(row)+b'\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, combined)
    atomic(out/'complete.json', dict(state='COMPLETE', completed=completed, target=total,
        binding_sha256=binding_hash, completed_at=time.time(), elapsed_seconds=time.time()-started,
        normal_jsonl_sha256=sha(combined),
        normalized_shards={str(p.relative_to(out)):sha(p) for p in sorted((out/'normalized').glob('*.json'))}))


if __name__ == '__main__':
    main()
