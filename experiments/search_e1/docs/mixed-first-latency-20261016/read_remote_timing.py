"""Read-only remote scalar extraction. No rollout text, tokens, tensors or writes."""
import datetime
import hashlib
import json
import pathlib
import re

BASES = [
    '/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011',
    '/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012',
]
RUN = '15bb15365346401da1d919e9cde8f813'
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
SCALAR = re.compile(r'([A-Za-z_][A-Za-z0-9_./-]*)\s*:\s*(' + NUMBER + r')')
KEEP_PREFIXES = ('timing_s/', 'timing_per_token_ms/', 'selection/', 'response_length/', 'prompt_length/')
KEEP_KEYS = {'actor/optimizer_steps_this_update', 'actor/skipped_update_opportunities',
             'actor/update_opportunities_this_attempt', 'critic/score/mean'}

def identity(path):
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

result = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              read_only=True, no_raw_rollout_export=True, runs=[])
for base_name in BASES:
    base = pathlib.Path(base_name)
    if not base.exists():
        result['runs'].append(dict(base=str(base), exists=False))
        continue
    run_path = base / 'logs/run-manifest.json'
    run = json.loads(run_path.read_text())
    assert run['run_id'] == RUN
    item = dict(base=str(base), run_manifest=identity(run_path),
                run={k:run.get(k) for k in ('run_id','source_commit','model_revision',
                     'actor_micro_batch_per_rank','vllm_gpu_memory_utilization')},
                metric_logs=[], selection=[], updates=[])
    log = base / 'logs/search-g1-common8-training.log'
    if log.exists():
        h = hashlib.sha256()
        rows = []
        with log.open('rb') as stream:
            for raw in stream:
                h.update(raw)
                if b'timing_s/' not in raw:
                    continue
                text = re.sub(r'\x1b\[[0-9;]*m', '', raw.decode(errors='replace'))
                found = SCALAR.findall(text)
                values = {k:float(v) for k,v in found}
                step = values.get('step')
                if step is None:
                    continue
                metrics = {k:v for k,v in values.items()
                           if k.startswith(KEEP_PREFIXES) or k in KEEP_KEYS}
                if metrics:
                    rows.append(dict(step=int(step),metrics=metrics,
                                     line_sha256=hashlib.sha256(raw).hexdigest()))
        stat = log.stat()
        item['metric_logs'].append(dict(path=str(log),bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,sha256=h.hexdigest(),rows=rows))
    for path in sorted((base / 'artifacts/mixed-selection').glob('*.json')):
        value = json.loads(path.read_text())
        compact = {k:value.get(k) for k in (
            'step','attempted_candidates','selected_mixed','selected_all_correct_fallback',
            'raw_generated_trajectories','raw_generated_policy_tokens',
            'raw_actual_retrieval_calls','candidate_fill_seconds','candidate_generation_timings')}
        compact['source'] = identity(path)
        item['selection'].append(compact)
    updates = base / 'logs/actual-updates.jsonl'
    if updates.exists():
        for raw in updates.read_text().splitlines():
            value = json.loads(raw)
            assert value['run_id'] == RUN
            item['updates'].append({k:value.get(k) for k in
                ('step','attempt_id','optimizer_writes_before','optimizer_writes_after',
                 'actual_optimizer_writes','opportunities_processed')})
        item['updates_source'] = identity(updates)
    item['source_files'] = []
    for relative in ('verl/trainer/ppo/ray_trainer.py','search_g1/mixed_runtime.py',
                     'search_r1/llm_agent/generation.py'):
        source = base / 'repo' / relative
        if source.exists():
            item['source_files'].append(identity(source))
    result['runs'].append(item)
print(json.dumps(result, separators=(',',':')))
