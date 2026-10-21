"""Read-only phase evidence and one CPU-only /proc sample inside own live job."""
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

BASE=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021')
assert os.getuid()==14090 and (BASE/'job.id').read_text().strip()=='745371'
result=dict(epoch=time.time(),job='745371',read_only=True)
result['selection']=[]
for path in sorted((BASE/'artifacts/mixed-selection').glob('*.json')):
    row=json.loads(path.read_text());hist={str(k):0 for k in range(6)}
    for group in row.get('groups',[]):hist[str(sum(group['strict_correctness']))]+=1
    result['selection'].append(dict(path=str(path),bytes=path.stat().st_size,mtime=path.stat().st_mtime,
        **{key:row.get(key)for key in ['step','attempted_candidates','responses_per_group','selected_groups',
        'selected_mixed','selected_all_correct_fallback','raw_generated_trajectories','raw_actual_retrieval_calls',
        'candidate_fill_seconds','candidate_generation_timings','source_consumed','policy_snapshot_id']},
        candidate_correct_count_histogram=hist))
result['ledgers']={}
for rank in range(4):
    for role in ['optimizer','scheduler']:
        path=BASE/f'logs/{role}-rank-{rank}.jsonl'
        if not path.exists():continue
        with path.open('rb')as stream:stream.seek(max(0,path.stat().st_size-18000));raw=stream.read().decode(errors='replace')
        rows=[]
        for line in raw.splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            rows.append({key:row.get(key)for key in ['kind','rank','logical_update','attempt_index','optimizer_writes','optimizer_writes_before','optimizer_writes_after','writes_before','writes_after','minibatch_index','allowed','opportunities_processed','actual_optimizer_writes','full_budget_complete']if key in row})
        result['ledgers'][path.name]=dict(bytes=path.stat().st_size,last_rows=rows[-4:])
for name in ['logs/probe-server.log','logs/trainer.log']:
    path=BASE/name
    if path.exists():
        with path.open('rb')as stream:stream.seek(max(0,path.stat().st_size-4000));raw=stream.read().decode(errors='replace')
        result[name]=dict(bytes=path.stat().st_size,mtime=path.stat().st_mtime,tail=raw)
workers=json.loads((BASE/'logs/full-state-restored-workers.json').read_text())['ranks'];pids=[row['process_identity']['pid']for row in workers]
log=(BASE/'logs/search-g1-common8-training.log').read_text();matches=re.findall(r'main_task pid=(\d+)',log)
if matches:pids.append(int(matches[-1]))
code="""import json,os,socket,time
from pathlib import Path
assert os.getuid()==14090 and os.environ.get('SLURM_JOB_ID')=='745371'
pids=PID_LIST
rows=[]
def sample(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return {'pid':pid,'missing':True}
 assert p.stat().st_uid==14090
 fields=(p/'stat').read_text().rsplit(')',1)[1].split()
 io={k:int(v) for k,v in (line.split(':')for line in (p/'io').read_text().splitlines())}
 return dict(pid=pid,state=fields[0],ticks=int(fields[11])+int(fields[12]),rss_pages=int(fields[21]),wchan=(p/'wchan').read_text().strip(),io=io)
before={pid:sample(pid)for pid in pids};time.sleep(1)
for pid in pids:
 after=sample(pid);old=before[pid]
 after['cpu_ticks_delta_1s']=after.get('ticks',0)-old.get('ticks',0)
 after['read_bytes_delta_1s']=after.get('io',{}).get('read_bytes',0)-old.get('io',{}).get('read_bytes',0)
 after['write_bytes_delta_1s']=after.get('io',{}).get('write_bytes',0)-old.get('io',{}).get('write_bytes',0)
 rows.append(after)
print(json.dumps(dict(host=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],gpu_allocation_requested=False,processes=rows)))
""".replace('PID_LIST',repr(pids))
observed=subprocess.run(['srun','--jobid=745371','--overlap','--ntasks=1','--cpus-per-task=1','--gres=none','python3','-B','-c',code],capture_output=True,text=True,timeout=30)
result['cpu_only_process_sample']=dict(returncode=observed.returncode,stdout=observed.stdout,stderr=observed.stderr)
print(json.dumps(result))
