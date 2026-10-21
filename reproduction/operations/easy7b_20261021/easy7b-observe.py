import time,json,os,subprocess
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021');job=(b/'job.id').read_text().strip();d=dict(epoch=time.time(),job=job)
for key,cmd in [('queue',['squeue','-j',job,'-h','-o','%i|%T|%N|%R|%S']),('accounting',['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode,NodeList,Elapsed'])]:
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=8);d[key]=dict(rc=x.returncode,stdout=x.stdout,stderr=x.stderr)
for n in ['logs/run-manifest.json','logs/pipeline-status.json','logs/easy2560-configuration-prepared.json','logs/cleanup-status.json','logs/process-ledger.json','logs/actual-updates.jsonl','preparation/rolling-checkpoints/latest.json']:
 p=b/n
 if not p.exists():continue
 if n.endswith('jsonl'):
  try:d[n]=[json.loads(line)for line in p.read_text().splitlines() if line][-2:]
  except Exception as e:d[n]=repr(e)
 else:
  z=json.loads(p.read_text())
  if n.endswith('run-manifest.json'):z={k:z.get(k)for k in ['run_id','source_commit','initial_model','reference_model','model_revision','training_world_size','actor_micro_batch_per_rank','all_correct_virtual_zero','initial_reward_probe_source_base','initial_probe_version']}
  d[n]=z
logs=[*b.glob('logs/training-*.log'),*b.glob('logs/*trainer*.log'),*b.glob('logs/search-g1-common8-training.log'),*b.glob('logs/*initial-calibration*.log'),*b.glob('logs/*retriever*.log'),*b.glob('logs/*retrieval-stage*.log'),*b.glob('artifacts/round_0-*/calibration.log')]
d['logs']=[]
for p in logs:
 s=p.stat()
 with p.open('rb')as f:f.seek(max(0,s.st_size-6500));tail=f.read().decode(errors='replace')
 d['logs'].append(dict(path=str(p.relative_to(b)),bytes=s.st_size,mtime=s.st_mtime,tail=tail[-3000:]))
d['calibration_dumps']=[dict(path=str(p.relative_to(b)),bytes=p.stat().st_size)for p in b.glob('artifacts/round_0-*/dumps/*')if p.is_file()]
for c in [b/'control/storage-current',Path('/tmp/easy7b-storage-current-20261021-21930')]:
 for p in c.glob('result-*.json'):
  d[str(p)]=json.loads(p.read_text())
g=b/('logs/gpu-samples-'+d.get('logs/run-manifest.json',{}).get('run_id','')+'.jsonl')
if g.is_file():
 with g.open('rb') as f:f.seek(max(0,g.stat().st_size-20000));lines=f.read().decode(errors='replace').splitlines()
 d['latest_gpu_samples']=[json.loads(z) for z in lines[-2:] if z]
pool=b/'artifacts/mixed-candidate-pools'
d['candidate_progress']=[]
if pool.is_dir():
 for a in sorted(pool.iterdir(),key=lambda a:a.stat().st_mtime)[-2:]:
  item=dict(pool=a.name)
  q=a/'progress.json'
  if q.is_file():
   z=json.loads(q.read_text());item.update(step=z.get('step'),completed_candidates=len(z.get('complete_ordinals',[])),mtime=q.stat().st_mtime)
  item['files']=[dict(name=q.name,bytes=q.stat().st_size,mtime=q.stat().st_mtime)for q in a.iterdir()if q.is_file() and q.name.endswith('.pt')]
  d['candidate_progress'].append(item)
c=b/'control/storage-breakdown'
for name in ('claim.json','result.json'):
 q=c/name
 if q.is_file():d['storage_breakdown_'+name]=json.loads(q.read_text())
q=c/'usage.tsv'
if q.is_file():
 rows=[]
 for line in q.read_text().splitlines():
  try:n,name=line.split('\t',1);rows.append(dict(bytes=int(n),path=name))
  except ValueError:pass
 d['storage_breakdown_partial']=sorted(rows,key=lambda r:r['bytes'],reverse=True)[:15]
p=b/'logs/search-g1-common8-training.log'
if p.is_file():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2000000));s=f.read().decode(errors='replace')
 d['actor_cache_log_lines']=[z for z in s.splitlines()if any(k in z for k in ('# GPU blocks','before init cache','after init cache','Maximum concurrency','gpu_memory_utilization','cache block'))][-20:]
d['optimizer_progress']=[]
for rank in range(4):
 p=b/('logs/optimizer-rank-'+str(rank)+'.jsonl')
 if not p.is_file():continue
 with p.open('rb')as f:f.seek(max(0,p.stat().st_size-35000));lines=f.read().decode(errors='replace').splitlines()
 rows=[]
 for line in lines:
  try:z=json.loads(line)
  except ValueError:continue
  context=z.get('context',{})
  rows.append(dict(kind=z.get('kind'),stage=z.get('stage'),optimizer_writes=z.get('optimizer_writes'),cumulative_writes=z.get('cumulative_writes'),logical_update=context.get('logical_update'),minibatch=context.get('minibatch_index')))
 d['optimizer_progress'].append(dict(rank=rank,bytes=p.stat().st_size,mtime=p.stat().st_mtime,last=rows[-2:]))
print(json.dumps(d))
