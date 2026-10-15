"""Remote read-only extraction of Q-probe phase scalars; no model/raw data reads."""
import datetime
import hashlib
import json
import pathlib
import re
import socket

BASES = {'/mnt/data/shaoshitong/search-g1-qprobe3-binary10-c5fc096-20261015': 'f77c3245dcfe49a88a5a99b1dec161e6'}

NUM = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
PATTERN = re.compile(r'([A-Za-z_][A-Za-z0-9_./-]*)\s*:\s*('+NUM+r')')
PREFIXES = ('timing_s/','infra/rollout/','selection/','question_probe/',
            'response_length/','prompt_length/')
KEYS = {'actor/optimizer_steps_this_update','actor/skipped_update_opportunities'}

def identity(path):
    s=path.stat()
    return dict(path=str(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

result = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              node=socket.gethostname(),read_only=True,no_raw_rollout_export=True,runs=[])
for name,expected_run in BASES.items():
    base=pathlib.Path(name)
    run_path=base/'logs/run-manifest.json'
    run=json.loads(run_path.read_text());assert run['run_id']==expected_run
    record=dict(base=name,run_manifest=identity(run_path),
                run={k:run.get(k) for k in ('run_id','source_commit','training_world_size',
                     'training_physical_indices','execution_id','execution_host')},
                metric_logs=[],selection=[],head_events=[],updates=[])
    for relative in ('logs/trainer.log','logs/search-g1-common8-training.log'):
        log=base/relative
        if not log.exists():continue
        h=hashlib.sha256();rows=[]
        with log.open('rb') as f:
            for raw in f:
                h.update(raw)
                if b'timing_s/' not in raw:continue
                text=re.sub(r'\x1b\[[0-9;]*m','',raw.decode(errors='replace'))
                v={k:float(x) for k,x in PATTERN.findall(text)}
                if 'step' not in v:continue
                metrics={k:x for k,x in v.items() if k.startswith(PREFIXES) or k in KEYS}
                rows.append(dict(step=int(v['step']),metrics=metrics,
                                 line_sha256=hashlib.sha256(raw).hexdigest()))
        record['metric_logs'].append(dict(path=str(log),bytes=log.stat().st_size,
                    sha256=h.hexdigest(),rows=rows))
    for path in sorted((base/'artifacts').glob('**/rollout-audit.json')):
        value=json.loads(path.read_text())
        if value.get('selection_mode')!='question_probe':continue
        small={k:value.get(k) for k in ('step','attempt_id','attempted_candidates',
            'selected_groups','selected_mixed','selected_all_wrong','selected_all_correct',
            'raw_generated_trajectories','raw_generated_policy_tokens',
            'raw_actual_retrieval_calls','candidate_fill_seconds','candidate_generation_timings')}
        small['source']=identity(path);record['selection'].append(small)
    updates=base/'logs/actual-updates.jsonl'
    if updates.exists():
        for raw in updates.read_text().splitlines():
            v=json.loads(raw);assert v['run_id']==expected_run
            record['updates'].append({k:v.get(k) for k in ('step','attempt_id',
                'optimizer_writes_before','optimizer_writes_after','actual_optimizer_writes',
                'opportunities_processed')})
        record['updates_source']=identity(updates)
    record['source_files']=[]
    for relative in ('verl/trainer/ppo/ray_trainer.py','search_g1/qprobe_runtime.py',
                     'search_g1/async_rollout_runtime.py','search_g1/qprobe_token_online.py'):
        p=base/'repo'/relative
        if p.exists():record['source_files'].append(identity(p))
    result['runs'].append(record)
print(json.dumps(result,separators=(',',':')))
