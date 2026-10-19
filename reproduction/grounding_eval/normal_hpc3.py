"""NQ-only normal rollout for preserved TOP-D/OPD, isolated from training.

Run with the historical TOP-D Python environment and an explicit JSON job.
The old generator remains unmodified. Observations are captured AFTER its token
truncation, and its policy mask separates generated actions from observations.
This module never submits jobs or discovers alternate checkpoints.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

EXPECTED_DATA_SHA = '55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1'


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        f.write(canonical(data) + b'\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def question_from_prompt(chat):
    texts = [x['content'] for x in chat if x['role'] == 'user']
    if len(texts) != 1 or '\nQuestion:' not in texts[0]:
        raise ValueError('Unexpected historical question prompt')
    question = texts[0].rsplit('\nQuestion:', 1)[1].strip()
    if not question:
        raise ValueError('Empty question')
    return question


def load_job(path):
    job = json.loads(Path(path).read_text())
    required = ('method', 'runtime_root', 'legacy_source', 'checkpoint_path', 'data_path', 'output_dir')
    if any(not job.get(k) for k in required):
        raise ValueError('Explicit job paths and method are mandatory')
    if job['method'] not in ('TOP-D alpha=0.1', 'OPD alpha=1'):
        raise ValueError('HPC3 runner only accepts the preserved TOP-D/OPD identities')
    for k in required[1:]:
        if not Path(job[k]).is_absolute():
            raise ValueError('Only absolute remote paths: ' + k)
    if '/checkpoints/' in str(Path(job['output_dir']).resolve()):
        raise ValueError('Evaluation cannot write inside checkpoints')
    job.setdefault('batch_size', 64)
    job.setdefault('gpu_memory_utilization', .60)
    job.setdefault('max_num_batched_tokens', 16384)
    job.setdefault('max_num_seqs', 64)
    job.setdefault('retriever_port', 18119)
    job.setdefault('retriever_threads', 16)
    job.setdefault('start_retriever', True)
    if not 1 <= job['batch_size'] <= job['max_num_seqs'] <= 256:
        raise ValueError('Unsafe batch bounds')
    if not .2 <= job['gpu_memory_utilization'] <= .8:
        raise ValueError('Reserve memory for the E5 encoder')
    return job


def checkpoint_inventory(path):
    root = Path(path)
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    names = set(index['weight_map'].values())
    names.update(('config.json', 'model.safetensors.index.json', 'tokenizer.json',
                  'tokenizer_config.json', 'generation_config.json', 'special_tokens_map.json',
                  'added_tokens.json', 'merges.txt', 'vocab.json'))
    files = {}
    for name in sorted(names):
        p = root / name
        if p.is_symlink() or not p.is_file() or p.stat().st_size <= 0:
            raise ValueError('Missing/nonregular checkpoint file: ' + str(p))
        files[name] = {'size': p.stat().st_size, 'sha256': sha(p)}
    return files


def start_retriever(job, out):
    """Exact original CPU-FAISS/GPU-E5 recipe, task-owned port and child handle."""
    import requests
    runtime = Path(job['runtime_root'])
    config = json.loads(Path(job['retriever_config']).read_text())
    cmd = list(config['retriever_command'])
    if '--faiss_gpu' in cmd:
        cmd.remove('--faiss_gpu')
    cmd[cmd.index('--port') + 1] = str(job['retriever_port'])
    cmd[cmd.index('--threads') + 1] = str(job['retriever_threads'])
    env = dict(os.environ)
    env['PYTHONPATH'] = str(job['legacy_source']) + os.pathsep + str(runtime / 'retriever-gpu-lib')
    env['OMP_NUM_THREADS'] = str(job['retriever_threads'])
    env['MKL_NUM_THREADS'] = str(job['retriever_threads'])
    log = open(out / 'retriever.log', 'ab')
    child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    atomic(out / 'retriever-child.json', {'pid': child.pid, 'command': cmd, 'started': time.time()})
    url = 'http://127.0.0.1:%d' % job['retriever_port']
    for _ in range(600):
        if child.poll() is not None:
            raise RuntimeError('Owned retriever exited: ' + str(child.returncode))
        try:
            if requests.get(url + '/docs', timeout=2).status_code == 200:
                return child, log, url + '/retrieve'
        except requests.RequestException:
            pass
        time.sleep(1)
    child.terminate(); child.wait(timeout=30); log.close()
    raise TimeoutError('Owned retriever did not become ready')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args(argv)
    job = load_job(args.job)
    out = Path(job['output_dir']); out.mkdir(parents=True, exist_ok=True)
    lock = open(out / 'normal.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic(out / 'worker.json', {'pid': os.getpid(), 'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'command': sys.argv,
        'started': time.time(), 'job_hash': digest(job)})
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('A real Slurm GPU allocation is required')
    if sha(job['data_path']) != EXPECTED_DATA_SHA:
        raise ValueError('Unexpected common final parquet SHA')
    # Before adding the historical source path, bind this runner to its own bytes.
    own_sha = sha(__file__)
    sys.path.insert(0, job['legacy_source'])
    import torch
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from verl.utils.torch_functional import get_eos_mask
    from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
    from verl.trainer.main_ppo import decode_model_response_spans
    raw = pq.read_table(job['data_path']).to_pylist()
    indexes = [i for i, row in enumerate(raw) if row['data_source'] == 'nq']
    if len(raw) != 51713 or len(indexes) != 3610:
        raise ValueError('Expected exact 51713 final occurrences, including 3610 NQ')
    selected = indexes[:job.get('max_rows', len(indexes))]
    if not selected:
        raise ValueError('No selected NQ occurrences')
    model_files = checkpoint_inventory(job['checkpoint_path'])
    binding = {'job': job, 'job_hash': digest(job), 'runner_sha256': own_sha,
        'checkpoint_files': model_files, 'checkpoint_hash': digest(model_files),
        'data_sha256': EXPECTED_DATA_SHA, 'evaluation_ordinals': selected,
        'legacy_source_sha256': {str(p.relative_to(job['legacy_source'])): sha(p) for p in
            [Path(job['legacy_source']) / name for name in ('search_r1/llm_agent/generation.py',
                'verl/utils/dataset/rl_dataset.py', 'search_r1/qa_protocol.py', 'verl/trainer/main_ppo.py',
                'verl/utils/torch_functional.py')]},
        'normal_protocol': {'dtype': 'bfloat16', 'seed': 42, 'greedy': True,
        'max_start_length': 2048, 'max_prompt_length': 4096, 'max_response_length': 500,
        'max_obs_length': 500, 'max_turns': 4, 'force_final_answer': True, 'topk': 3},
        'partial_pilot': len(selected) != 3610}
    binding_path = out / 'normal-binding.json'
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise ValueError('Resume configuration, source, checkpoint, or data changed')
    else:
        atomic(binding_path, binding)
    batch_root = out / 'normal-batches'; batch_root.mkdir(exist_ok=True)
    existing = []
    for p in sorted(batch_root.glob('batch-*.json')):
        blob = json.loads(p.read_text())
        if digest(blob['records']) != blob['records_sha256'] or blob['binding_hash'] != digest(binding):
            raise ValueError('Corrupt/foreign normal shard')
        existing.extend(blob['records'])
    if [r['evaluation_ordinal'] for r in existing] != selected[:len(existing)]:
        raise ValueError('Normal resume has duplicate, missing, or foreign occurrences')
    offset = len(existing)
    if offset == len(selected):
        publish_normal(out, batch_root, binding, selected)
        return
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(job['checkpoint_path'], local_files_only=True)
    if tok.padding_side != 'right':
        raise ValueError('Historical checkpoint tokenizer must retain right padding')
    dataset = RLHFDataset(job['data_path'], tok, prompt_key='prompt', max_prompt_length=4096,
        filter_prompts=True, return_raw_chat=False, truncation='error',
        cache_dir=str(out / 'dataset-cache'))
    if len(dataset) != 51713:
        raise ValueError('Historical dataset filtering changed occurrences')
    child = None; log = None
    try:
        search_url = job.get('search_url')
        if job['start_retriever']:
            if 'retriever_config' not in job:
                raise ValueError('Explicit original retriever configuration required')
            child, log, search_url = start_retriever(job, out)
        if not search_url:
            raise ValueError('No retrieval endpoint')
        engine = LLM(model=job['checkpoint_path'], tokenizer=job['checkpoint_path'], dtype='bfloat16',
            tensor_parallel_size=1, gpu_memory_utilization=job['gpu_memory_utilization'],
            enforce_eager=True, max_model_len=4596,
            max_num_batched_tokens=job['max_num_batched_tokens'], max_num_seqs=job['max_num_seqs'],
            enable_prefix_caching=False, seed=42)
        params = SamplingParams(n=1, best_of=1, top_p=1., top_k=-1, min_p=0., temperature=0.,
            max_tokens=500, ignore_eos=False, detokenize=False)
        class Adapter:
            def generate_sequences(self, p):
                ids = p.batch['input_ids']; att = p.batch['attention_mask']; pos = p.batch['position_ids']
                prompts = [x[m.bool()].tolist() for x, m in zip(ids, att)]
                requests = engine.generate(prompt_token_ids=prompts, sampling_params=params, use_tqdm=False)
                tokens = [list(x.outputs[0].token_ids) for x in requests]
                if len(prompts) != len(manager.active_indices):
                    raise RuntimeError('Request-to-occurrence mapping mismatch')
                for local, original in enumerate(manager.active_indices):
                    manager.llm_requests[original].append({'input_tokens': len(prompts[local]),
                        'output_tokens': len(tokens[local]),
                        'finish_reason': requests[local].outputs[0].finish_reason,
                        'stop_reason': requests[local].outputs[0].stop_reason})
                if len(tokens) != len(prompts):
                    raise RuntimeError('Inference response count mismatch')
                response = torch.full((len(tokens), 500), tok.pad_token_id, dtype=torch.long)
                for i, t in enumerate(tokens):
                    response[i, :len(t)] = torch.tensor(t, dtype=torch.long)
                seq = torch.cat([ids, response], -1)
                rp = pos[:, -1:] + torch.arange(1, 501).unsqueeze(0)
                mask = get_eos_mask(response_id=response, eos_token=p.meta_info['eos_token_id'], dtype=att.dtype)
                return DataProto.from_dict({'prompts': ids, 'responses': response, 'input_ids': seq,
                    'attention_mask': torch.cat([att, mask], -1), 'position_ids': torch.cat([pos, rp], -1)})
        class CaptureManager(LLMGenerationManager):
            def reset_capture(self, count):
                self.seen = [[] for _ in range(count)]
                self.observation_ids = [[] for _ in range(count)]
                self.calls = [0] * count
                self.active_indices = list(range(count))
                self.llm_requests = [[] for _ in range(count)]
            def execute_predictions(self, predictions, pad_token, active_mask=None, do_search=True):
                actions, _ = self.postprocess_predictions(predictions)
                if do_search:
                    for i, (action, active) in enumerate(zip(actions, active_mask)):
                        if active and action == 'search':
                            self.calls[i] += 1
                result = super().execute_predictions(predictions, pad_token, active_mask, do_search)
                self.active_indices = [i for i, (active, done) in enumerate(zip(active_mask, result[1]))
                    if active and not done]
                return result
            def _process_next_obs(self, next_obs):
                ids = super()._process_next_obs(next_obs)
                for i, text in enumerate(next_obs):
                    if not text.startswith('\n\n<information>'):
                        continue
                    # Exact returned IDs are what _update_rolling_state receives.
                    values = ids[i].tolist()
                    while values and values[-1] == tok.pad_token_id:
                        values.pop()
                    decoded = tok.decode(values, skip_special_tokens=False)
                    body = decoded.split('<information>', 1)[-1]
                    if '</information>' in body:
                        body = body.split('</information>', 1)[0]
                    self.seen[i].append(body)
                    self.observation_ids[i].append(values)
                return ids
        manager = CaptureManager(tok, Adapter(), GenerationConfig(max_turns=4, max_start_length=2048,
            max_prompt_length=4096, max_response_length=500, max_obs_length=500, num_gpus=1,
            search_url=search_url, topk=3, force_final_answer=True), is_validation=True)
        started = time.time()
        batch_size = job['batch_size']
        while offset < len(selected):
            chosen = selected[offset:offset + batch_size]
            bd = collate_fn([dataset[i] for i in chosen])
            batch = DataProto.from_single_dict(bd)
            gen = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            gen.meta_info = {'eos_token_id': tok.eos_token_id, 'pad_token_id': tok.pad_token_id,
                'recompute_log_prob': False, 'do_sample': False, 'validate': True}
            manager.reset_capture(len(chosen)); manager.timing_raw = {}
            before = time.time()
            try:
                result = manager.run_llm_loop(gen_batch=gen,
                    initial_input_ids=gen.batch['input_ids'][:, -2048:].clone())
            except Exception as error:
                atomic(out / 'normal-error.json', {'status': 'INFERENCE_ERROR', 'offset': offset,
                    'batch_size': batch_size, 'error_type': type(error).__name__, 'error': str(error),
                    'time': time.time(), 'binding_hash': digest(binding)})
                raise  # A broken vLLM engine is not silently reused after OOM.
            batch = batch.union(result); records = []
            for i, ordinal in enumerate(chosen):
                row = raw[ordinal]; ground = row['reward_model']['ground_truth']['target']
                gold = [ground] if isinstance(ground, str) else list(ground)
                width = batch.batch['responses'].shape[1]
                mask = batch.batch['attention_mask'][i, -width:].bool() & batch.batch['info_mask'][i, -width:].bool()
                spans = decode_model_response_spans(tok, batch.batch['responses'][i], mask)
                record = {'qid': str(row['extra_info']['index']), 'data_source': 'nq',
                    'evaluation_ordinal': ordinal, 'question': question_from_prompt(row['prompt']), 'gold': gold,
                    'model_spans': spans, 'seen_docs': manager.seen[i],
                    'actual_retrieval_calls': manager.calls[i], 'source_run_id': job['method'],
                    'checkpoint_hash': binding['checkpoint_hash'], 'normal_config_hash': digest(binding),
                    'seen_observation_token_ids': manager.observation_ids[i],
                    'response_token_ids': batch.batch['responses'][i].tolist(),
                    'model_token_mask': mask.tolist(),
                    'llm_requests': manager.llm_requests[i],
                    'model_generated_tokens': int(mask.sum()),
                    'inference_generated_tokens': sum(t['output_tokens'] for t in manager.llm_requests[i]),
                    'observation_tokens': sum(len(t) for t in manager.observation_ids[i]),
                    'initial_prompt_tokens': manager.llm_requests[i][0]['input_tokens'],
                    'cumulative_input_tokens': sum(t['input_tokens'] for t in manager.llm_requests[i])}
                record['raw_generated_tokens'] = record['inference_generated_tokens']
                record['cumulative_model_input_tokens'] = record['cumulative_input_tokens']
                record['trajectory_total_tokens'] = (record['initial_prompt_tokens'] +
                    record['model_generated_tokens'] + record['observation_tokens'])
                record['generated_tokens_discarded_by_action_parser'] = (
                    record['inference_generated_tokens'] - record['model_generated_tokens'])
                record['source_record_hash'] = digest(record)
                records.append(record)
            shard = batch_root / ('batch-%06d.json' % offset)
            if shard.exists():
                raise FileExistsError(shard)
            atomic(shard, {'binding_hash': digest(binding), 'records': records,
                'records_sha256': digest(records), 'elapsed_seconds': time.time() - before})
            offset += len(records)
            atomic(out / 'normal-progress.json', {'completed_unique': offset, 'total': len(selected),
                'output_bytes': sum(p.stat().st_size for p in batch_root.glob('batch-*.json')),
                'latest_success': time.time(), 'seconds': time.time() - started,
                'batch_size': batch_size, 'errors': 0, 'pid': os.getpid(),
                'gpu_allocated_bytes': torch.cuda.memory_allocated(), 'gpu_reserved_bytes': torch.cuda.memory_reserved()})
            print('NORMAL_PROGRESS', offset, len(selected), time.time(), flush=True)
        publish_normal(out, batch_root, binding, selected)
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=10)
            atomic(out / 'retriever-exit.json', {'pid': child.pid, 'returncode': child.returncode,
                'all_owned_children_reaped': True, 'time': time.time()})
        if log is not None:
            log.close()


def publish_normal(out, batch_root, binding, selected):
    target = out / 'normal.jsonl'; tmp = out / 'normal.jsonl.tmp'
    ordinals = []; token_totals = {}
    with open(tmp, 'wb') as f:
        for p in sorted(batch_root.glob('batch-*.json')):
            blob = json.loads(p.read_text())
            if digest(blob['records']) != blob['records_sha256'] or blob['binding_hash'] != digest(binding):
                raise ValueError('Invalid normal shard at final readback')
            shard_hash = sha(p)
            for row in blob['records']:
                ordinals.append(row['evaluation_ordinal'])
                for key in ('model_generated_tokens', 'inference_generated_tokens', 'observation_tokens',
                        'initial_prompt_tokens', 'cumulative_input_tokens', 'trajectory_total_tokens',
                        'raw_generated_tokens', 'cumulative_model_input_tokens'):
                    token_totals[key] = token_totals.get(key, 0) + row.get(key, 0)
                f.write(canonical(dict(row, source_shard_sha256=shard_hash)) + b'\n')
        f.flush(); os.fsync(f.fileno())
    if ordinals != selected:
        raise ValueError('Final occurrence coverage mismatch')
    os.replace(tmp, target)
    fd = os.open(out, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    atomic(out / 'normal-complete.json', {'status': 'NORMAL_NQ_COMPLETE' if len(selected) == 3610 else 'NORMAL_NQ_PILOT',
        'rows': len(selected), 'normal_sha256': sha(target), 'binding_hash': digest(binding),
        'checkpoint_hash': binding['checkpoint_hash'], 'token_totals': token_totals, 'time': time.time()})


if __name__ == '__main__':
    main()
