from pathlib import Path
import hashlib,json,subprocess,time
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');base=A/'online-packed-full169028-Z-seed19137-plan2026102302-P20-20261023-v1';out=A/'fullZ-seed2-749935-terminal-CPU-coordinator-20261023-v1';accept=A/'full169028-Z-seed2-terminal-CPU-acceptance-20261023-v1/epoch10/accepted.json'
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def identity(c):
 p=Path('/proc/%d'%c['pid'])
 try:return p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==c['start_ticks']
 except(FileNotFoundError,ProcessLookupError):return False
r=dict(time=time.time(),job_id='749935',Slurm_mutated=False,CP_tensor_or_final_read=False);assert str(read(base/'root-submission.json')['job_id'])=='749935';r['accounting']=subprocess.check_output(['sacct','-X','-n','-P','-j','749935','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],universal_newlines=True,timeout=15).strip()
r['training']={}
for name in('progress-rank-0.json','progress-rank-1.json','complete.json'):
 if(base/name).is_file():
  v=read(base/name);r['training'][name]=v if name=='complete.json'else{k:v.get(k)for k in('cursor','epoch','native_gradient_L2','head_gradient_L2','alignment','time')}
r['training_failures']={p.name:read(p)for p in base.glob('failure-rank-*.json')};r['coordinator']={}
for name in('launch.json','controller.json','progress.json','active-child.json','complete.json','failure.json'):
 if(out/name).is_file():r['coordinator'][name]=read(out/name)
c=r['coordinator']['controller.json'];assert c['pid']==2764625 and c['start_ticks']=='1165095'and c['source']['sha256']=='1cdd6073d27f82f6b3f356bfae8ce48f0dc0bc6131b31068a14141dc2ca55037';r['coordinator_actual_identity_live']=identity(c)
if'active-child.json'in r['coordinator']:r['CPU_child_actual_identity_live']=identity(r['coordinator']['active-child.json'])
if accept.is_file():r['accepted']=read(accept);r['accepted_ref']=proof(accept)
if(out/'failure.json').is_file():r['coordinator_log_tail']=(out/'worker.log').read_text()[-2500:]
p=A/'full169028-Z-seed2-terminal-CPU-acceptance-20261023-v1/progress.json'
if p.is_file():r['CPU_validator_progress']=read(p)
print(json.dumps(r))
