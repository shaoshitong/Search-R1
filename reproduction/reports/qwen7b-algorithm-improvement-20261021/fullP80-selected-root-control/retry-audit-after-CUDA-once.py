"""Root recovery: retry the failed owned audit once, unchanged inputs/code, exclude ACD1-13."""
import datetime,fcntl,hashlib,json,os,pathlib,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
C=A/'fullP80-selected-root-control-20261023-v1';B=A/'full169028-L15P80-MSE-viewed-permutation-audit-20261023-v1'
def read(p):return json.loads(p.read_bytes())
def proof(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 if r.returncode:raise RuntimeError(r.stdout[-500:]+r.stderr[-1000:])
 return r.stdout.strip()
def once(p,d):
 raw=(json.dumps(d,sort_keys=True,separators=(',',':'))+'\n').encode()
 if p.exists():assert p.read_bytes()==raw
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
def main():
 with(C/'control.lock').open('a+')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  original=read(C/'audit-submission.json');assert original['job_id']=='750474'
  entry=B/'audit.sbatch';config=C/'audit-bound.json';assert proof(entry)==original['entry']and proof(config)==original['config']
  accounting=command(['sacct','-X','-n','-P','-j','750474','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'])
  assert any(x.split('|')[:3]==['750474','FAILED','1:0']and x.split('|')[4]=='ACD1-13'for x in accounting.splitlines()),accounting
  receipt=C/'audit-CUDA-retry-submission.json';intent=C/'audit-CUDA-retry-intent.json'
  if receipt.exists():print(json.dumps(read(receipt)));return
  comment='q7b-fullP80-audit-retry-'+original['config']['sha256'][:20];name=original['name']
  candidates=command(['squeue','-h','-u','mou176','-n',name,'-o','%i|%j|%u'])
  if intent.exists():
   previous=read(intent)
   if candidates:
    ids=[x.split('|')[0]for x in candidates.splitlines()];assert len(ids)==1
    job=ids[0];details=command(['scontrol','show','job',job,'-o'])
    assert 'Comment='+comment in details and 'Command='+str(entry) in details and 'UserId=mou176('in details
    result=dict(previous,job_id=job,status='RECONCILED_OWNED_CUDA_RETRY',submitted_now=False)
    once(receipt,result);print(json.dumps(result));return
   since=datetime.datetime.fromtimestamp(previous['time']-3600).strftime('%Y-%m-%dT%H:%M:%S')
   history=command(['sacct','-X','-n','-P','-u','mou176','-S',since,'--format=JobIDRaw,JobName%80,State,ExitCode'])
   print(json.dumps(dict(status='PRIOR_RETRY_OUTCOME_UNKNOWN_NO_RESUBMISSION',intent=previous,matching_accounting=[x for x in history.splitlines()if name in x])));return
  assert not candidates and not(B/'controller.json').exists()and not(B/'complete.json').exists()
  locks=[]
  for n in('audit-job.lock','audit-rank-0.lock','audit-rank-1.lock'):
   p=B/n
   if p.exists():
    f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  archive=B/'attempt-750474-CUDA-init-failed';archive.mkdir(exist_ok=True);failures=[]
  for rank in(0,1):
   p=B/('failure-rank-%d.json'%rank);dst=archive/p.name
   source=p if p.exists()else dst;d=read(source)
   assert 'CUDA unknown error'in d['traceback']and 'torch.cuda.set_device'in d['traceback']
   if p.exists():assert not dst.exists();p.rename(dst)
   failures.append(proof(dst))
  metadata=dict(time=time.time(),original_job_id='750474',original_accounting=accounting,original_submission=proof(C/'audit-submission.json'),entry=proof(entry),config=proof(config),comment=comment,name=name,exclude='ACD1-13',failure_archive=failures,model_or_scoring_changed=False,final4096_read=False)
  once(intent,metadata)
  for f in locks:f.close()
  job=command(['sbatch','--parsable','--exclude=ACD1-13','--comment='+comment,str(entry),str(config),original['config']['sha256']]).split(';')[0];assert job.isdigit()
  result=dict(metadata,job_id=job,status='OWNED_CUDA_INIT_FAILURE_RETRY_SUBMITTED_ONCE',submitted_now=True)
  once(receipt,result);print(json.dumps(result))
main()
