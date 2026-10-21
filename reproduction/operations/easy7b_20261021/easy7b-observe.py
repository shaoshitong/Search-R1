import time,json,os,subprocess
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021');job=(b/'job.id').read_text().strip();d=dict(epoch=time.time(),job=job)
for key,cmd in [('queue',['squeue','-j',job,'-h','-o','%i|%T|%N|%R|%S']),('accounting',['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode,NodeList,Elapsed'])]:
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=8);d[key]=dict(rc=x.returncode,stdout=x.stdout,stderr=x.stderr)
for n in ['logs/run-manifest.json','logs/pipeline-stage.json','logs/easy2560-configuration-prepared.json','logs/cleanup-status.json','logs/actual-updates.jsonl','preparation/rolling-checkpoints/latest.json']:
 p=b/n
 if not p.exists():continue
 if n.endswith('jsonl'):
  try:d[n]=[json.loads(line)for line in p.read_text().splitlines() if line][-2:]
  except Exception as e:d[n]=repr(e)
 else:
  z=json.loads(p.read_text())
  if n.endswith('run-manifest.json'):z={k:z.get(k)for k in ['run_id','source_commit','initial_model','reference_model','model_revision','training_world_size','actor_micro_batch_per_rank','all_correct_virtual_zero','initial_reward_probe_source_base','initial_probe_version']}
  d[n]=z
logs=[*b.glob('logs/training-*.log'),*b.glob('logs/*initial-calibration*.log'),*b.glob('artifacts/round_0-*/calibration.log')]
d['logs']=[]
for p in logs:
 s=p.stat()
 with p.open('rb')as f:f.seek(max(0,s.st_size-6500));tail=f.read().decode(errors='replace')
 d['logs'].append(dict(path=str(p.relative_to(b)),bytes=s.st_size,mtime=s.st_mtime,tail=tail[-3000:]))
d['calibration_dumps']=[dict(path=str(p.relative_to(b)),bytes=p.stat().st_size)for p in b.glob('artifacts/round_0-*/dumps/*')if p.is_file()]
for c in [b/'control/storage-current',Path('/tmp/easy7b-storage-current-20261021-21930')]:
 for p in c.glob('result-*.json'):
  d[str(p)]=json.loads(p.read_text())
print(json.dumps(d))
