"""Two explicit full-P80 MSE jobs: metadata wait, then existing693 CPU entry.

One independent worker/config per actual job. The worker never imports model
packages or holds the common CPU mutex; the existing CPU child owns that mutex.
"""
import argparse,fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
JOBS={'750189':(9137,2026102227),'750190':(19137,2026102302)}
def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
 p=Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def atomic(p,v):
 p=Path(p);q=p.with_name(p.name+'.%d.tmp'%os.getpid());q.write_text(json.dumps(v,sort_keys=True,allow_nan=False)+'\n');q.replace(p)
def ticks(pid):
 try:return Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except(FileNotFoundError,ProcessLookupError):return None
def validate_complete(value,job):
 init,plan=JOBS[job]
 required=dict(status='FULL169028_P80_OWNG5_MSE_TRAINING_COMPLETE',epochs=10,cursor=13210,batch_attempts=13210,actual_Adam_updates=13210,EMA_updates=13210,margin_satisfied_batches=0,training_questions=169028,final4096_scored=False)
 if any(value.get(k)!=v for k,v in required.items()):raise ValueError('Actual full MSE10E/13210 completion differs')
 required_binding=dict(init_seed=init,train_plan_seed=plan,diagnostic_seed=2026102228,training_questions=169028,training_scope='full169028_fresh_official_actor_not_pilot_resume',fixed_m=5,layer=15,prefix=80,objective='mse')
 if any(value.get('binding',{}).get(k)!=v for k,v in required_binding.items()):raise ValueError('Actual complete seed/geometry/fixture family differs')
def inspect_terminal(base,job):
 if str(read(base/'root-submission.json')['job_id'])!=job:raise ValueError('Actual full MSE job handle changed')
 reply=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 if reply.returncode:raise RuntimeError('Accounting query failed: '+reply.stderr[-500:])
 rows=[r.split('|')[:3]for r in reply.stdout.splitlines()if r.split('|')[0]==job];ok=[job,'COMPLETED','0:0']in rows
 failed=[r for r in rows if r[1].split('+')[0]in('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED','REVOKED')or(r[1]=='COMPLETED'and r[2]!='0:0')]
 if failed:raise RuntimeError('Actual training terminal failure; checkpoint not opened: '+repr(failed))
 complete=read(base/'complete.json')if(base/'complete.json').is_file()else None
 if ok and complete is None:raise RuntimeError('Actual successful complete receipt missing; no CPU child launched')
 if ok:validate_complete(complete,job)
 progress={}
 for rank in(0,1):
  p=base/('progress-rank-%d.json'%rank)
  if p.is_file():r=read(p);progress[str(rank)]={k:r.get(k)for k in('cursor','actual_Adam_updates','epoch','time')}
 return ok,dict(job_id=job,accounting=reply.stdout.strip(),actual_complete=complete,rank_progress=progress)
def accepted_metadata(value,cfg,complete):
 job=str(cfg['job_id']);init,plan=JOBS[job]
 required=dict(status='TERMINAL_FULL169028_P80_MSE_CPU_SHA_MODEL_ADAM_RNG_PLAN_PHYSICAL_GATE_ACCEPTED',job_id=job,epoch=10,batch_attempts=13210,actual_Adam_updates=13210,EMA_updates=13210,epoch_complete=True,next_epoch=11,within_next_epoch_question_position=0,CPU_only=True,final4096_read=False)
 if any(value.get(k)!=v for k,v in required.items()):raise ValueError('Actual E10 CPU acceptance differs')
 if value.get('binding')!=complete['binding']or any(value.get('explicit_seed_recipe',{}).get(k)!=v for k,v in dict(init_seed=init,train_plan_seed=plan,diagnostic_seed=2026102228).items()):raise ValueError('Actual accepted seed/config differs')
 if value.get('checkpoint',{}).get('path')!=str(Path(cfg['base'])/'checkpoint-epoch-10.pt'):raise ValueError('Different checkpoint acceptance cannot be reused')
 refs=value.get('accepted_input',{})
 if refs.get('fixture')!=complete['binding']['fixture']or refs.get('physical_source_contract')!=complete['binding']['physical_source_contract']or not isinstance(refs.get('P20_alignment'),dict):raise ValueError('Exact accepted_input physical fixture/P20 refs required')
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);a=p.parse_args();assert proof(a.config)['sha256']==a.config_sha;cfg=read(a.config);job=str(cfg['job_id']);assert job in JOBS;out=Path(cfg['out']);base=Path(cfg['base']);init,plan=JOBS[job];assert base.name=='full169028-L15P80-MSE-ownG5-physical-seed%d-plan%d-20261023-v3'%(init,plan)
 guard=(out/'coordinator.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);atomic(out/'controller.json',dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),source=proof(__file__),config=proof(a.config),job_id=job,time=time.time()))
 try:
  polls=0
  while True:
   ready,observed=inspect_terminal(base,job);polls+=1;atomic(out/'progress.json',dict(stage='TERMINAL_CONFIRMED_READY_FOR_CPU'if ready else'WAITING_ACTUAL_TERMINAL_NO_CP_OR_COMMON_LOCK_HELD',polls=polls,CP_or_tensor_opened=False,shared_CPU_mutex_held=False,time=time.time(),**observed))
   if ready:break
   time.sleep(30)
  entry=Path(cfg['entry']['path']);validator=Path(cfg['validator']['path']);assert proof(entry)==cfg['entry']and proof(validator)==cfg['validator'];dest=Path(cfg['accepted']);reused=dest.exists()
  if not reused:
   log=out/'epoch10-validator.log'
   with log.open('ab')as stream:
    child=subprocess.Popen(['bash',str(entry),'--epoch','10'],stdout=stream,stderr=stream,stdin=subprocess.DEVNULL,close_fds=True);atomic(out/'active-child.json',dict(pid=child.pid,start_ticks=ticks(child.pid),entry=cfg['entry'],validator=cfg['validator'],log=str(log),job_id=job,time=time.time()));code=child.wait()
   if code:raise RuntimeError('Immutable693 CPU validator exited%d; see%s'%(code,log))
  accepted=read(dest);accepted_metadata(accepted,cfg,observed['actual_complete'])
  if accepted['source_complete']!=proof(base/'complete.json')or accepted['source_publication']!=proof(base/'source-publication.json'):raise ValueError('Accepted source/complete proof changed')
  atomic(out/'complete.json',dict(status='FULL_P80_MSE_E10_TERMINAL_CPU_ACCEPTANCE_COMPLETE',job_id=job,accepted=proof(dest),checkpoint=accepted['checkpoint'],existing_exact_CPU_acceptance_reused=reused,CPU_only=True,GPU_next_update_restore_tested=False,Slurm_mutated=False,final4096_read=False,time=time.time()))
 except BaseException:
  atomic(out/'failure.json',dict(job_id=job,time=time.time(),traceback=traceback.format_exc(),Slurm_mutated=False));raise
if __name__=='__main__':main()
