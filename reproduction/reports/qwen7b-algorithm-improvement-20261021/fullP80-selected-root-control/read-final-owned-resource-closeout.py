"""Final owned handles/locks and bounded metadata proof; never stop another task."""
import fcntl,hashlib,json,pathlib,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
C=A/'fullP80-selected-root-control-20261023-v1'
AUDIT=A/'full169028-L15P80-MSE-viewed-permutation-audit-20261023-v1'
FINAL=A/'frozen-final4096-full169028-L15P80-MSE-20261023-v2'
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr[-800:])
def proof(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
roots=[('seed1',A/'full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3'),('seed2',A/'full169028-L15P80-MSE-ownG5-physical-seed19137-plan2026102302-20261023-v3'),('audit',AUDIT),('final',FINAL)]
locks={};metadata={}
for name,base in roots:
 for p in sorted(base.glob('*.lock')):
  with p.open('r')as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True;fcntl.flock(f,fcntl.LOCK_UN)
   except BlockingIOError:free=False
  locks[str(p)]=free
 for fn in ['source-publication.json','complete.json']:
  p=base/fn
  if p.is_file():metadata[name+'/'+fn]=proof(p)
 for p in sorted(base.glob('failure*.json')):metadata[name+'/'+p.name]=proof(p)
jobs='750189,750190,750334,750474,750476,750481'
out=dict(time=time.time(),owned_jobs=jobs,accounting=command(['sacct','-X','-n','-P','-j',jobs,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList,AllocTRES%100']),queue=command(['squeue','-h','-j',jobs,'-o','%i|%j|%T|%R']),task_locks_free=locks,metadata=metadata,mutation=False,checkpoint_tensor_loaded=False,model_or_rollout_data_returned=False)
print(json.dumps(out))
