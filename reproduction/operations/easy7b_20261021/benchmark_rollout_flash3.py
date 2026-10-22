"""Independent one-H100 replay from a sealed HF checkpoint; no actor update.

Each backend runs in a fresh process. Input IDs and KV token capacity match.
This measures vLLM generation, not the complete retrieval/selection outer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def progress(root, phase, **value):
    record = dict(epoch=time.time(), phase=phase, **value)
    with (root / 'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')
        stream.flush()
    print(json.dumps(record, sort_keys=True), flush=True)


def inputs(args):
    import torch
    # Trusted, owned completed production output. Never written or moved.
    record = torch.load(args.candidate_file, map_location='cpu', weights_only=False)
    batch = record['output'].batch
    parent = record['parents'].batch
    def rows(b, indices, cap):
        result = []
        for i in indices:
            ids = b['input_ids'][i][b['attention_mask'][i].bool()].tolist()
            result.append(ids[:cap])
        return result
    prompts = rows(batch, range(0, min(len(batch), args.batch * 5), 5), 3584)
    short = rows(parent, range(min(8, len(parent))), 2048)
    value = dict(prompts=prompts, short_prompts=short,
                 candidate_file=args.candidate_file, checkpoint=args.checkpoint,
                 prompt_lengths=[len(p) for p in prompts],
                 prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest())
    save(Path(args.root) / 'inputs.json', value)
    progress(Path(args.root), 'ACTUAL_SAVED_INPUTS_READY',
             questions=len(prompts), prompt_tokens=sum(map(len, prompts)),
             prompt_sha256=value['prompt_sha256'])


def engine(args):
    root = Path(args.root)
    backend = args.backend
    os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS' if backend == 'xformers' else 'FLASH_ATTN'
    os.environ['VLLM_USE_V1'] = '0'
    os.environ['VLLM_NO_USAGE_STATS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    audit = None
    if backend == 'fa3':
        from reproduction.easy2560_joint4.flash3_runtime import binding
        from search_g1.vllm_flash3 import install_vllm_flash3
        audit = install_vllm_flash3(dict(infra_flash_attention_3=binding(args.source_base)))
    from vllm import LLM, SamplingParams
    payload = json.loads((root / 'inputs.json').read_text())
    block = 16 if backend == 'xformers' else 256
    progress(root, 'LOADING_LATEST_CHECKPOINT', backend=backend,
             checkpoint=args.checkpoint, cuda_device=torch.cuda.get_device_name())
    start = time.monotonic()
    llm = LLM(model=args.checkpoint, tokenizer=args.checkpoint,
              tensor_parallel_size=1, dtype='bfloat16', enforce_eager=True,
              trust_remote_code=False, max_model_len=4096,
              max_num_seqs=args.batch, max_num_batched_tokens=8192,
              block_size=block, num_gpu_blocks_override=131072 // block,
              gpu_memory_utilization=.65, swap_space=0,
              enable_prefix_caching=False, disable_log_stats=True, seed=317)
    init_seconds = time.monotonic() - start
    progress(root, 'MODEL_LOADED', backend=backend, init_seconds=init_seconds)
    # Include saved long histories as well as original question prompts. The
    # same fixed appended token IDs are then scored by every backend.
    check_inputs = payload['short_prompts'] + payload['prompts'][:8]
    check = llm.generate(prompt_token_ids=check_inputs,
                        sampling_params=SamplingParams(temperature=0., max_tokens=32,
                                                       logprobs=1, ignore_eos=True),
                        use_tqdm=False)
    check_rows = [dict(token_ids=r.outputs[0].token_ids,
                       token_logprobs=[float(step[token].logprob) for step, token in
                                       zip(r.outputs[0].logprobs, r.outputs[0].token_ids)])
                  for r in check]
    if backend == 'xformers':
        teacher = [p + r['token_ids'] for p, r in zip(check_inputs, check_rows)]
        save(root / 'teacher-inputs.json', teacher)
    teacher = json.loads((root / 'teacher-inputs.json').read_text())
    forced = llm.generate(prompt_token_ids=teacher,
                          sampling_params=SamplingParams(temperature=0., max_tokens=1,
                                                         prompt_logprobs=1), use_tqdm=False)
    forced_rows = []
    if not (len(check_rows) == len(check_inputs) == len(teacher) == len(forced)):
        raise RuntimeError('Incomplete numerical comparison batch')
    for result, ids, original in zip(forced, teacher, check_inputs):
        start_index = len(original)
        forced_rows.append([float(result.prompt_logprobs[i][ids[i]].logprob)
                            for i in range(start_index, len(ids))])
    timings = []
    params = SamplingParams(temperature=0., max_tokens=args.output_tokens,
                            logprobs=1, ignore_eos=True, detokenize=False)
    for repeat in range(args.warmup + args.repeats):
        torch.cuda.synchronize()
        began = time.monotonic()
        result = llm.generate(prompt_token_ids=payload['prompts'],
                              sampling_params=params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - began
        generated = sum(len(row.outputs[0].token_ids) for row in result)
        if repeat >= args.warmup:
            timings.append(dict(seconds=elapsed, generated_tokens=generated,
                                generated_tokens_per_second=generated / elapsed))
        progress(root, 'WARMUP' if repeat < args.warmup else 'MEASURED_GENERATION',
                 backend=backend, repeat=repeat, seconds=elapsed, generated_tokens=generated)
    report = dict(backend=backend, checkpoint=args.checkpoint, init_seconds=init_seconds,
                  prompt_sha256=payload['prompt_sha256'], prompt_lengths=payload['prompt_lengths'],
                  block_size=block, kv_token_capacity=131072,
                  warmup=args.warmup, repeats=args.repeats, timings=timings,
                  median_seconds=statistics.median(r['seconds'] for r in timings),
                  greedy_check=check_rows, teacher_forced_logprobs=forced_rows,
                  numerical_prompt_lengths=[len(p) for p in check_inputs],
                  fa3=audit.report() if audit else None,
                  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  attention_backend=os.environ['VLLM_ATTENTION_BACKEND'],
                  training_modified=False, optimizer_loaded=False,
                  qualification_scope='Same saved inputs and CP50 HF weights, TP1 eager generation only; not full outer')
    save(root / (backend + '.json'), report)
    progress(root, 'BACKEND_COMPLETE', backend=backend, median_seconds=report['median_seconds'],
             actual_fa3_calls=audit.calls if audit else None)


def controller(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    inputs(args)
    common = [sys.executable, '-B', str(Path(__file__).resolve()),
              '--root', args.root, '--checkpoint', args.checkpoint,
              '--candidate-file', args.candidate_file, '--source-base', args.source_base,
              '--batch', str(args.batch), '--output-tokens', str(args.output_tokens),
              '--warmup', str(args.warmup), '--repeats', str(args.repeats)]
    for backend in ('xformers', 'fa2', 'fa3'):
        with (root / (backend + '.log')).open('w') as stream:
            result = subprocess.run(common + ['--backend', backend], stdout=stream,
                                    stderr=subprocess.STDOUT)
        progress(root, 'BACKEND_EXIT', backend=backend, exit_code=result.returncode)
        if result.returncode:
            save(root / 'failure.json', dict(backend=backend, exit_code=result.returncode,
                 log=str(root / (backend + '.log')), training_modified=False))
            raise SystemExit(result.returncode)
    baseline = json.loads((root / 'xformers.json').read_text())
    candidate = json.loads((root / 'fa3.json').read_text())
    a, b = baseline['teacher_forced_logprobs'], candidate['teacher_forced_logprobs']
    if (len(a) != len(b) or not a or any(len(x) != len(y) or not x for x, y in zip(a, b))):
        raise RuntimeError('Missing teacher-forced rows or tokens')
    pairs = [(x, y) for xs, ys in zip(a, b)
             for x, y in zip(xs, ys)]
    diffs = sorted(abs(x - y) for x, y in pairs)
    result = dict(status='COMPLETED', checkpoint=args.checkpoint,
                  baseline_median_seconds=baseline['median_seconds'],
                  fa3_median_seconds=candidate['median_seconds'],
                  generation_speedup=baseline['median_seconds'] / candidate['median_seconds'],
                  teacher_forced_token_count=len(pairs),
                  logprob_max_abs=max(diffs), logprob_mean_abs=statistics.mean(diffs),
                  logprob_p99_abs=diffs[min(len(diffs)-1, int(len(diffs)*.99))],
                  greedy_sequences_equal=sum(a['token_ids'] == b['token_ids'] for a,b in
                      zip(baseline['greedy_check'], candidate['greedy_check'])),
                  greedy_sequences=len(candidate['greedy_check']), fa3=candidate['fa3'],
                  numerical_prompt_lengths=candidate['numerical_prompt_lengths'],
                  actual_fa3_prefill_and_decode=bool(candidate['fa3']['calls']['prefill'] and
                                                    candidate['fa3']['calls']['decode']),
                  training_modified=False, full_outer_speedup_proven=False,
                  actual_full_state_training_resume=False)
    save(root / 'result.json', result)
    progress(root, 'ALL_BACKENDS_COMPLETE', **result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--candidate-file', required=True)
    p.add_argument('--source-base', required=True)
    p.add_argument('--backend', choices=['xformers', 'fa2', 'fa3'])
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--output-tokens', type=int, default=128)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    engine(args) if args.backend else controller(args)


if __name__ == '__main__':
    main()
