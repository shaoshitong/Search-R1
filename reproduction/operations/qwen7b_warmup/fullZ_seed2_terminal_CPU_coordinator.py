"""Durable metadata-only wait for749935, then one immutable CPU acceptance.

No tensor imports or shared CPU mutex while training is running. The existing
validator acquires that shared mutex only after exact terminal completion.
"""
import argparse,fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path

def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
 p=Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def atomic(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.%d.tmp'%os.getpid());tmp.write_text(json.dumps(v,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def ticks(pid):
 try:return Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except(FileNotFoundError,ProcessLookupError):return None
def inspect_terminal(base,job):
 if str(read(base/'root-submission.json')['job_id'])!=job:raise ValueError('Actual Z job handle changed')
 reply=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 if reply.returncode:raise RuntimeError('Accounting query failed: '+reply.stderr[-500:])
 rows=[row.split('|')[:3]for row in reply.stdout.splitlines()if row.split('|')[0]==job];states=[row[1]for row in rows];ok=[job,'COMPLETED','0:0']in rows
 failed=[row for row in rows if row[1].split('+')[0]in('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED','REVOKED')or(row[1]=='COMPLETED'and row[2]!='0:0')]
 if failed:raise RuntimeError('Actual training terminal failure, no checkpoint opened: '+repr(failed))
 complete=read(base/'complete.json')if(base/'complete.json').is_file()else None
 if ok and complete is None:raise RuntimeError('Accounting successful but actual complete receipt missing; no checkpoint opened')
 if ok and complete!=dict(status='PACKED_VARIANCE_FULL_SWEEP_COMPLETE',epochs=10,cursor=13210,training_questions=169028,pilot_not_full169028=False,final4096_scored=False):raise ValueError('Actual complete budget differs')
 progress={}
 for rank in(0,1):
  p=base/('progress-rank-%d.json'%rank)
  if p.is_file():
   r=read(p);progress[str(rank)]={k:r.get(k)for k in('cursor','epoch','native_gradient_L2','head_gradient_L2','time')}
 return ok,dict(job_id=job,accounting=reply.stdout.strip(),actual_complete=complete,rank_progress=progress)
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);a=p.parse_args();assert proof(a.config)['sha256']==a.config_sha;cfg=read(a.config);out=Path(cfg['out']);base=Path(cfg['base']);job=str(cfg['job_id']);assert job=='749935';guard=(out/'coordinator.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 atomic(out/'controller.json',dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),source=proof(__file__),config=proof(a.config),job_id=job,time=time.time()))
 try:
  polls=0
  while True:
   ready,observed=inspect_terminal(base,job);polls+=1;atomic(out/'progress.json',dict(stage='TERMINAL_CONFIRMED_READY_FOR_CPU'if ready else'WAITING_ACTUAL_TERMINAL_NO_CP_OR_COMMON_LOCK_HELD',polls=polls,CP_or_tensor_opened=False,shared_CPU_mutex_held=False,time=time.time(),**observed))
   if ready:break
   time.sleep(30)
  entry=Path(cfg['entry']['path']);validator=Path(cfg['validator']['path']);assert proof(entry)==cfg['entry']and proof(validator)==cfg['validator'];dest=Path(cfg['accepted'])
  if not dest.exists():
   log=out/'epoch10-validator.log'
   with log.open('ab')as stream:
    child=subprocess.Popen(['bash',str(entry),'--epoch','10'],stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,close_fds=True)
    atomic(out/'active-child.json',dict(pid=child.pid,start_ticks=ticks(child.pid),entry=cfg['entry'],validator=cfg['validator'],log=str(log),job_id=job,time=time.time()));code=child.wait()
   if code:raise RuntimeError('Immutable CPU validator exited%d; see%s'%(code,log))
  accepted=read(dest)
  for k,v in dict(status='TERMINAL_FULL169028_PACKED_VARIANCE_CPU_SHA_MODEL_ADAM_RNG_PLAN_ACCEPTED',job_id=job,epoch=10,cursor=13210,init_seed=19137,train_plan_seed=2026102302,diagnostic_seed=2026102228,CPU_restore_only=True).items():
   if accepted.get(k)!=v:raise ValueError('Actual CPU acceptance differs: '+k)
  atomic(out/'complete.json',dict(status='FULL_Z_SEED2_E10_TERMINAL_CPU_ACCEPTANCE_COMPLETE',job_id=job,accepted=proof(dest),checkpoint=accepted['checkpoint'],CPU_restore_only=True,GPU_next_update_restore_tested=False,Slurm_mutated=False,final4096_read=False,time=time.time()))
 except BaseException:
  atomic(out/'failure.json',dict(job_id=job,time=time.time(),traceback=traceback.format_exc(),Slurm_mutated=False));raise
if __name__=='__main__':main()
