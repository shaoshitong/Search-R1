"""NQ-only normal evaluation for historical Easy/OPSD HF exports.

Run this FILE in a fresh interpreter, with the historical runtime repo supplied.
Easy retains authored token prefixes; OPSD uses its original retokenizing
LLMGenerationManager validation loop. Neither adapter creates an optimizer.
GPU initialization is deliberately inside main, after identity/lock validation.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import struct
import subprocess
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
    row.update(model_generated_tokens=kept, raw_generated_tokens=generated, observation_tokens=observations,
        environment_correction_tokens=corrections, retained_model_tokens=kept,
        initial_prompt_tokens=initial, trajectory_total_tokens=initial+kept+observations+corrections,
        cumulative_model_input_tokens=inputs, api_cache_tokens=None,
        api_cache_tokens_null_reason='local_vllm_does_not_report_per_request_cache_tokens')
    return row


def seen_document(text):
    """Remove only actual boundary information wrappers, retain cropped body."""
    text = re.sub(r'^\s*<information>', '', text, count=1)
    return re.sub(r'</information>\s*$', '', text, count=1)


def gpu_free(index):
    import subprocess
    result = subprocess.run(['nvidia-smi', '-i', str(index), '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


def gpu_uuid(index):
    import subprocess
    return subprocess.run(['nvidia-smi', '-i', str(index), '--query-gpu=uuid',
        '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.strip()


def validate_gpu_snapshot(index, expected_uuid, rows, processes, minimum_free_gib):
    selected = [r for r in rows if r[1].strip() == expected_uuid]
    if len(selected) != 1 or int(selected[0][0]) != index:
        raise RuntimeError('Physical numeric index does not match frozen UUID')
    row = selected[0]
    if int(row[2]) > 512 or int(row[4]) != 0 or int(row[3]) < minimum_free_gib * 1024:
        raise RuntimeError('Dedicated GPU memory/utilization admission failed')
    if any(r and r[0].strip() == expected_uuid for r in processes):
        raise RuntimeError('GPU already has a compute process')


def admit_gpu(index, expected_uuid, directory, minimum_free_gib):
    """Same UUID OFD lease namespace as standalone training and CF workers."""
    if not re.fullmatch(r'GPU-[0-9a-f-]+', expected_uuid):
        raise ValueError('An explicit physical GPU UUID is required')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / (expected_uuid + '.lock'),
                 os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.fcntl(fd, fcntl.F_OFD_SETLK,
                    struct.pack('hhqqi4x', fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0))
        rows = list(csv.reader(io.StringIO(subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True))))
        processes = list(csv.reader(io.StringIO(subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)))
        )
        validate_gpu_snapshot(index, expected_uuid, rows, processes, minimum_free_gib)
        for process in Path('/proc').iterdir():
            if not process.name.isdigit() or int(process.name) == os.getpid():
                continue
            try:
                if (process / 'comm').read_text().strip() == 'slurmd':
                    raise RuntimeError('Scheduler node requires explicit Slurm allocation adapter')
                command = (process / 'cmdline').read_bytes().split(b'\0')
                if any(x.startswith((b'reproduction.prefix_selection200.',
                                     b'reproduction.prefix20_online.')) for x in command):
                    raise RuntimeError('A training/bootstrap controller may reserve the GPU')
            except (PermissionError, FileNotFoundError, ProcessLookupError):
                pass
        return fd, rows
    except BaseException:
        os.close(fd)
        raise


class LocalVLLM:
    """TP1 inference-only worker for the historical OPSD controller."""
    def __init__(self, llm, tokenizer, sampling):
        self.llm, self.tokenizer, self.sampling = llm, tokenizer, sampling
        self.last = []

    def decode(self, requests):
        # Sorting is unnecessary: vLLM performs its own bounded continuous batch.
        self.last_requests = [list(row) for row in requests]
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
        seen_docs=[seen_document(tokenizer.decode(t['retained_observation_token_ids'], skip_special_tokens=False,
            clean_up_tokenization_spaces=False)) for t in s['turns'] if t['actual_search']],
        actual_retrieval_calls=s['actual_calls'], turns=s['turns'],
        response_token_ids=s['response'], model_mask=s['mask'], initial_prompt_token_ids=s['prompt']) for s in states]


def opsd_normal(manager, worker, prompts, tokenizer, config):
    """Wrap the REAL historical _validate -> run_llm_loop, without changing it.

    SearchRollout is used by training, not this validation path. Capture the
    exact requests, original retokenized actions and cropped observations via
    transparent wrappers around the historical manager's methods.
    """
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
    captures = [[] for _ in prompts]
    original_post = manager._postprocess_responses
    original_execute = manager.execute_predictions
    original_observations = manager._process_next_obs
    current = {}

    def postprocess(responses):
        value = original_post(responses)
        current['actions'] = value[0]
        return value

    def execute(predictions, pad_token, active_mask=None, do_search=True):
        result = original_execute(predictions, pad_token, active_mask, do_search=do_search)
        active_indices = [i for i, flag in enumerate(active_mask.tolist()) if flag]
        if len(active_indices) != len(worker.last):
            raise ValueError('Historical validation active row capture mismatch')
        current['active'] = active_indices
        for local, i in enumerate(active_indices):
            item = worker.last[local].outputs[0]
            action = [int(v) for v in current['actions'][local].tolist() if int(v) != pad]
            captures[i].append(dict(turn_index=len(captures[i]),
                generation_context_token_ids=worker.last_requests[local],
                raw_response_token_ids=list(item.token_ids), postprocessed_action_token_ids=action,
                retained_observation_token_ids=[], observation_text_before_truncation=result[0][i],
                actual_search=bool(result[3][i] and do_search), final_only=not do_search,
                finish_reason=item.finish_reason, stop_reason=item.stop_reason))
        return result

    def observations(texts):
        value = original_observations(texts)
        for i in current['active']:
            captures[i][-1]['retained_observation_token_ids'] = [int(v) for v in value[i].tolist() if int(v) != pad]
        return value

    manager._postprocess_responses = postprocess
    manager.execute_predictions = execute
    manager._process_next_obs = observations
    try:
        manager.timing_raw = {}
        output = manager.run_llm_loop(batch, ids[:, -config.max_start_length:].clone())
    finally:
        manager._postprocess_responses = original_post
        manager.execute_predictions = original_execute
        manager._process_next_obs = original_observations
    rows = []
    response_width = output.batch['responses'].shape[1]
    for i, turns in enumerate(captures):
        response = output.batch['responses'][i].tolist()
        active = output.batch['attention_mask'][i, -response_width:].bool()
        model = output.batch['info_mask'][i, -response_width:].bool()
        model_mask = (active & model).long().tolist()
        rows.append(dict(model_spans=response_spans(tokenizer, response, model_mask),
            seen_docs=[seen_document(tokenizer.decode(t['retained_observation_token_ids'], skip_special_tokens=False,
                clean_up_tokenization_spaces=False)) for t in turns if t['actual_search']],
            actual_retrieval_calls=sum(int(t['actual_search']) for t in turns),
            turns=turns, response_token_ids=response, model_mask=model_mask,
            initial_prompt_token_ids=turns[0]['generation_context_token_ids']))
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
    p.add_argument('--gpu-uuid', required=True,
                   help='Frozen physical UUID; numeric CUDA mapping is verified before initialization')
    p.add_argument('--gpu-lock-directory', required=True,
                   help='One common task directory shared by all normal/CF workers')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--max-batch-size', type=int,
                   help='Execution-only growth cap after successful batches; frozen in binding')
    p.add_argument('--max-num-seqs', type=int, default=128)
    p.add_argument('--max-num-batched-tokens', type=int, default=8192)
    p.add_argument('--gpu-memory-utilization', type=float, default=.25)
    p.add_argument('--min-start-free-gib', type=float, default=48.)
    p.add_argument('--min-running-free-gib', type=float, default=12.)
    p.add_argument('--limit', type=int, default=3610)
    p.add_argument('--retrieval-provenance', help='Explicit source/spec/topology caveats for this normal rerun')
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    lease = open(out / '.normal.lock', 'a+'); fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_fd, admission_rows = admit_gpu(args.gpu_index, args.gpu_uuid,
                                     args.gpu_lock_directory, args.min_start_free_gib)
    physical_uuid = args.gpu_uuid
    # Keep this file descriptor/OFD lease live for the entire evaluation.
    gpu_lease = os.fdopen(gpu_fd, 'r+')
    gpu_lease.seek(0); gpu_lease.truncate()
    json.dump(dict(pid=os.getpid(), output=str(out), physical_uuid=physical_uuid,
                   started_at=time.time(), argv=sys.argv), gpu_lease)
    gpu_lease.flush(); os.fsync(gpu_lease.fileno())
    atomic(out / 'gpu-admission.json', dict(physical_uuid=physical_uuid,
        physical_index=args.gpu_index, inventory=admission_rows, pid=os.getpid(),
        lock_kind='OFD', gpu_lock_directory=str(Path(args.gpu_lock_directory).resolve()),
        cuda_visible_devices=str(args.gpu_index), checked_at=time.time()))
    if args.method == 'easy' and not args.identity_file:
        raise ValueError('Easy requires original final-result occurrence identity')
    maximum_batch = args.max_batch_size or args.batch_size
    if not (0 < args.batch_size <= maximum_batch <= args.max_num_seqs <= 1024 and 0 < args.limit <= 3610):
        raise ValueError('Invalid batch or NQ limit')
    if gpu_free(args.gpu_index) < args.min_start_free_gib * 1024:
        raise RuntimeError('Insufficient fresh free memory before model load')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    runtime = str(Path(args.runtime_repo).resolve())
    sys.path.insert(0, runtime)
    import pandas as pd
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    import torch
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
        search_url=args.retrieval_url, topk=int(config.retriever.topk),
        force_final_answer=bool(config.get('force_final_answer', True)))
    if expected['max_turns'] != 4 or expected['max_response_length'] != 500 or expected['topk'] != 3:
        raise ValueError('Historical normal protocol differs from reviewed adapter')
    generation = GenerationConfig(**{k: v for k, v in expected.items() if k in GenerationConfig.__annotations__})
    generation.force_final_answer = bool(config.get('force_final_answer', True))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    data_sha256 = sha(args.data)
    if data_sha256 != '55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1':
        raise ValueError('Final parquet differs from frozen common split SHA256')
    frame = pd.read_parquet(args.data)
    if len(frame) != 51713:
        raise ValueError('Original final parquet must contain exactly 51713 rows')
    indices = [i for i, value in enumerate(frame['data_source']) if value == 'nq']
    if len(indices) != 3610:
        raise ValueError('Expected fixed 3610 NQ occurrences in original final parquet')
    original_ids = json.loads(Path(args.identity_file).read_text()) if args.identity_file else None
    if original_ids is not None:
        if (original_ids['run_id'] != args.run_id or original_ids['validation_sha256'] != data_sha256 or
            len(original_ids['question_ids']) != 51713 or len(original_ids['data_sources']) != 51713):
            raise ValueError('Historical final identity run/data SHA/count mismatch')
    source_files = dict(generation=sha(historical_generation.__file__))
    if args.method == 'easy':
        trace = importlib.import_module('reproduction.common8.behavior_trace')
        if not str(Path(trace.__file__).resolve()).startswith(runtime + '/'):
            raise ValueError('Historical Easy trace imported from another checkout')
        source_files['behavior_trace'] = sha(trace.__file__)
    normal_config = dict(method=args.method, run_id=args.run_id, checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_files={x.name: sha(x) for x in Path(args.checkpoint).iterdir() if x.is_file()
            and x.name.endswith(('.json', '.safetensors', '.txt')) and not x.name.startswith('backup-')},
        data_path=str(Path(args.data).resolve()), data_sha256=data_sha256,
        source_config_sha256=sha(args.source_config),
        normal_protocol=expected, runtime_sources_sha256=source_files,
        greedy=True, dtype='bfloat16', adapter_sha256=sha(__file__),
        inference_batch=args.batch_size, max_num_seqs=args.max_num_seqs,
        maximum_inference_batch=maximum_batch,
        max_num_batched_tokens=args.max_num_batched_tokens, gpu_memory_utilization=args.gpu_memory_utilization,
        nq_occurrences=indices, limit=args.limit, physical_gpu_uuid=physical_uuid,
        retrieval_provenance=json.loads(Path(args.retrieval_provenance).read_text())
            if args.retrieval_provenance else None)
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
    start = 0; batch_size = args.batch_size
    while start < total:
        selected = indices[start:min(total, start + batch_size)]
        stem = 'batch-%06d-%06d' % (start, start + len(selected))
        raw_path, norm_path = out / 'raw' / (stem + '.json'), out / 'normalized' / (stem + '.json')
        if raw_path.exists() and norm_path.exists():
            saved = json.loads(norm_path.read_text())
            if saved['binding_sha256'] != binding_hash or saved['raw_sha256'] != sha(raw_path):
                raise ValueError('Completed batch integrity mismatch')
            completed += len(selected); start += len(selected)
            batch_size = min(maximum_batch, batch_size * 2); continue
        if raw_path.exists() and not norm_path.exists():
            saved = json.loads(raw_path.read_text())
            if saved['binding_sha256'] != binding_hash or len(saved['rows']) != len(selected):
                raise ValueError('Incomplete raw batch identity mismatch')
            raw_sha = sha(raw_path)
            atomic(norm_path, dict(binding_sha256=binding_hash, raw_sha256=raw_sha,
                rows=[dict(row, source_shard_sha256=raw_sha) for row in saved['rows']]))
            completed += len(selected); start += len(selected)
            batch_size = min(maximum_batch, batch_size * 2); continue
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
            matches = list(re.finditer(r'Question:\s*', prompt_text))
            if not matches:
                raise ValueError('Unknown original common prompt question schema')
            question = prompt_text[matches[-1].end():].strip()
            extra = plain(row.get('extra_info', {}))
            if not isinstance(extra, dict) or 'index' not in extra:
                raise ValueError('Original canonical extra_info.index missing')
            qid = str(extra['index'])
            historical_qid = None
            if original_ids is not None:
                if original_ids['data_sources'][ordinal] != 'nq':
                    raise ValueError('Historical final occurrence source mismatch')
                historical_qid = str(original_ids['question_ids'][ordinal])
                expected_historical_qid = hashlib.sha256(re.sub(r'\s+', ' ', question).strip().casefold().encode()).hexdigest()
                if historical_qid != expected_historical_qid:
                    raise ValueError('Historical final identity prompt/order mismatch')
                if original_ids.get('rows'):
                    saved_row = original_ids['rows'][ordinal]
                    if (saved_row['ordinal'] != ordinal or saved_row['question_id'] != historical_qid or
                        saved_row['data_source'] != 'nq'):
                        raise ValueError('Historical final journal row occurrence mismatch')
            identities.append(dict(data_source='nq', evaluation_ordinal=ordinal, qid=qid,
                historical_qid=historical_qid, question=question, gold=gold,
                original_prompt=chat, source_run_id=args.run_id,
                normal_binding_sha256=binding_hash,
                checkpoint_files_sha256=hashlib.sha256(canonical(normal_config['checkpoint_files'])).hexdigest(),
                checkpoint_id=hashlib.sha256(canonical(normal_config['checkpoint_files'])).hexdigest()))
        t0 = time.time()
        try:
            if args.method == 'easy':
                generated = easy_normal(manager, worker, prompts, tokenizer, generation, trace.raw_prefix_to_first_closing_tag)
            else:
                generated = opsd_normal(manager, worker, prompts, tokenizer, generation)
        except Exception as error:
            atomic(out/'normal-error.json', dict(state='INFERENCE_ERROR', completed=completed,
                offset=start, actual_batch_size=len(selected), error_type=type(error).__name__,
                error=str(error), updated_at=time.time(), binding_sha256=binding_hash,
                completed_batches_retained=True, broken_engine_reused=False))
            raise
        raw = [token_accounting(dict(identity, **value)) for identity, value in zip(identities, generated)]
        atomic(raw_path, dict(binding_sha256=binding_hash, rows=raw, completed_at=time.time()))
        raw_sha = sha(raw_path)
        normalized = [dict(row, source_shard_sha256=raw_sha) for row in raw]
        atomic(norm_path, dict(binding_sha256=binding_hash, raw_sha256=raw_sha, rows=normalized))
        completed += len(selected)
        progress = dict(state='RUNNING', method=args.method, pid=os.getpid(), completed=completed,
            target=total, updated_at=time.time(), latest_batch_seconds=time.time()-t0,
            total_seconds=time.time()-started, binding_sha256=binding_hash,
            actual_batch_size=len(selected), next_batch_size=min(maximum_batch, batch_size * 2),
            maximum_batch_size=maximum_batch, physical_gpu_uuid=physical_uuid,
            gpu_allocated_bytes=torch.cuda.memory_allocated(), gpu_reserved_bytes=torch.cuda.memory_reserved(),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_free_mib=gpu_free(args.gpu_index), errors=0,
            unique_occurrences=completed, raw_bytes=sum(p.stat().st_size for p in (out/'raw').glob('*.json')))
        atomic(out/'progress.json', progress); print(json.dumps(progress), flush=True)
        start += len(selected); batch_size = min(maximum_batch, batch_size * 2)
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
