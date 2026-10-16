"""One bounded direct HPC3 query; only scalar/JSON progress comes back locally."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from .direct_control import run


REMOTE = r'''
import json, subprocess, time
from pathlib import Path
b = Path(BASE)
o = dict(epoch=time.time(), job=JOB, base=str(b))
o['read_warnings'] = []
def read(p):
    try: return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return None
def rows(p):
    if not p.is_file(): return []
    result = []
    for number, line in enumerate(p.read_text().splitlines(), 1):
        try: result.append(json.loads(line))
        except json.JSONDecodeError:
            o['read_warnings'].append(dict(path=str(p), line=number, status='JSONL_READ_INCOMPLETE'))
    return result
o['queue'] = subprocess.check_output(['squeue','-h','-j',str(JOB),'-o','%i|%T|%P|%R|%S'], text=True).strip()
o['accounting'] = subprocess.check_output(['sacct','-nP','-X','-j',str(JOB),'--format=JobID,State,ExitCode,NodeList'], text=True).strip()
r = read(b/'logs/run-manifest.json')
if r: o['run'] = {k:r.get(k) for k in ('run_id','source_commit','question_probe_refit_outers','extra_checkpoint_steps')}
o['updates'] = [{k:r.get(k) for k in ('step','actual_optimizer_writes','full_budget_complete','update_status')} for r in rows(b/'logs/actual-updates.jsonl')[-6:]]
o['ranks'] = {}
for rank in range(4):
    rs = rows(b/'logs'/('optimizer-rank-'+str(rank)+'.jsonl'))
    writes = [r for r in rs if r.get('kind') == 'OPTIMIZER_WRITE']
    budgets = [r for r in rs if r.get('kind') == 'OUTER_BUDGET']
    monitor = [r for r in rs if r.get('kind') == 'POLICY_MONITOR']
    scheduler = rows(b/'logs'/('scheduler-rank-'+str(rank)+'.jsonl'))
    o['ranks'][str(rank)] = dict(allocation_writes=len(writes), last_write=writes[-1].get('write_count') if writes else None,
        budgets=[{k:r.get(k) for k in ('logical_update','optimizer_writes','cumulative_writes','full_budget_complete')} for r in budgets[-3:]],
        last_monitor_allowed=monitor[-1].get('allowed') if monitor else None,
        scheduler={k:scheduler[-1].get(k) for k in ('kind','completed_outer_after','scheduler_advances','actual_optimizer_writes','writes_after')} if scheduler else None)
ps = sorted((b/'artifacts/question-probe/attempts').glob('*'), key=lambda p:p.stat().st_mtime)
if ps:
    p = ps[-1]; o['latest_attempt'] = p.name; o['attempt_files'] = [x.name for x in p.iterdir() if x.is_file()]
    for name in ('selection.json','rollout-audit.json'):
        r = read(p/name)
        if r:
            o[name] = {k:r[k] for k in ('step','selected_groups','raw_generated_trajectories','candidate_fill_seconds','online_features','head_version','question_probe_version') if k in r}
            if name == 'rollout-audit.json':
                groups = r['groups']
                valid = len(groups) == 512 and all(len(g['strict_correctness']) == 5 and
                    all(type(v) is int and v in (0,1) for v in g['strict_correctness']) for g in groups)
                if not valid:
                    o['read_warnings'].append(dict(path=str(p/name), status='INCOMPLETE_STRICT_G5_LABELS'))
                else:
                    counts = [sum(g['strict_correctness']) for g in groups]
                    mixed = sum(0 < n < 5 for n in counts)
                    o[name].update(selected_mixed=mixed, selected_mixed_fraction=mixed/512,
                        selected_all_wrong=counts.count(0), selected_all_correct=counts.count(5),
                        quality_scope='actual selected strict-EM G5 labels; not held-out QA EM')
ps = sorted((b/'artifacts/question-probe/completed').glob('*.json'), key=lambda p:p.stat().st_mtime)
o['probe_commits'] = [{k:r.get(k) for k in ('stage','outer','head_update')} for p in ps[-3:] if (r := read(p))]
o['pairs'] = {}
for outer in (25,50,75,100,125,150,175,200):
    for kind in ('pending-probe-pairs','completed-pairs'):
        p = b/'artifacts/question-probe'/kind/('global_step_'+str(outer))/'source-receipt.json'; r = read(p)
        if r: o['pairs'][str(outer)+':'+kind] = {k:r.get(k) for k in ('status','step','execution_phase','world_size','scope')}
o['refits'] = {}
for p in sorted((b/'artifacts/question-probe/periodic-refits').glob('*/latest-state.json')):
    r = read(p)
    if r:
        chunks = [read(f) for f in p.parent.glob('chunk-*/real-rollout-audit.json')]
        valid = [c for c in chunks if c and c.get('outer') == int(p.parent.name)
                 and c.get('questions') == 512 and c.get('trajectories') == 2560]
        r = dict(r, completed_label_chunks=len(valid), collected_questions=512*len(valid),
                 collected_trajectories=2560*len(valid))
        o['refits'][p.parent.name] = r
o['errors'] = []
for name in ('formal-'+str(JOB)+'.log','search-g1-common8-training.log'):
    p = b/'logs'/name
    if not p.is_file(): continue
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-30000)); tail=f.read().decode(errors='replace')
    o['errors'] += [line[:500] for line in tail.splitlines() if any(x in line for x in ('Traceback','Error:','AssertionError'))][-4:]
print(json.dumps(o))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--job', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    address = subprocess.check_output(['ipconfig','getifaddr','en0'], text=True).strip()
    code = 'BASE='+repr(args.base)+'\nJOB='+repr(args.job)+'\n'+REMOTE
    result = run('/data/user/jxu957/experiments/topd-20261008/environment/topd-py310/bin/python -c '+shlex.quote(code),
                 account='mou176', source_address=address)
    if result.returncode:
        raise RuntimeError('Direct control failed: '+result.stderr.decode(errors='replace'))
    records = []
    for line in result.stdout.decode().splitlines():
        try: records.append(json.loads(line))
        except json.JSONDecodeError: pass
    if len(records) != 1 or records[0].get('job') != args.job:
        raise RuntimeError('Missing or ambiguous bounded progress response')
    record = records[0]; record['control_checked_unix'] = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'live-sample-latest.json').write_text(json.dumps(record, indent=2)+'\n')
    with (args.output/'live-samples.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
    print(json.dumps(dict(epoch=record['epoch'], queue=record['queue'],
        completed=[r['step'] for r in record['updates'] if r.get('full_budget_complete')],
        rank_adam={k:v['last_write'] for k,v in record['ranks'].items()},
        latest_attempt=record.get('latest_attempt'), selection=record.get('selection.json'),
        rollout=record.get('rollout-audit.json'), refits=record['refits'],
        errors=record['errors'], read_warnings=record['read_warnings'])))


if __name__ == '__main__':
    main()
