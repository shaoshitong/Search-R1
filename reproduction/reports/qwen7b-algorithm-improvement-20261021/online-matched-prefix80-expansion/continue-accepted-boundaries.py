"""Root-only accepted-state handoff; held successor has no job dependency."""
import pathlib,json,subprocess,time,fcntl,hashlib,re,shutil
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');S=A/'online-mixed-layer-prefix-20261022-2gpu-v1';stage=json.loads((A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1/staged.json').read_bytes());guard=(S/'prefix80-submit.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);out={'time':time.time(),'jobs':{}}
def read(p):return json.loads(p.read_bytes())
def atomic(p,v):
 q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,sort_keys=True)+'\n');q.replace(p)
def run(cmd):
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25);assert p.returncode==0,p.stdout+p.stderr;return p.stdout.strip()
def proof(p):
 raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
for layer in(18,15):
 b=S/('layer%d-prefix80'%layer);done=b/'walltime-handoff-complete.json'
 if done.exists():out['jobs'][str(layer)]=read(done);continue
 accepted=b/'walltime-boundary-CPU-accepted.json'
 if not accepted.exists():out['jobs'][str(layer)]={'status':'WAIT_CPU_ACCEPTANCE'};continue
 transaction=b/'walltime-handoff-transaction.json'
 if transaction.exists():
  tx=read(transaction);fresh_held=read(b/'walltime-boundary-held.json');fresh_cp=read(accepted)
  if tx['held']['epoch']!=fresh_held['epoch']:
   assert not tx.get('old_writer_cancel_requested') and not tx.get('root_identity_moved')
   assert fresh_cp['checkpoint']==fresh_held['checkpoint']and fresh_cp['cursor']==fresh_held['cursor']and fresh_held['epoch']==2
   shutil.copy2(str(transaction),str(b/'walltime-handoff-transaction-epoch1-historical.json'))
   tx.update(held=fresh_held,CPU_acceptance=proof(accepted),boundary_upgrade_time=time.time());atomic(transaction,tx)
 else:
  held=read(b/'walltime-boundary-held.json');cp=read(accepted);old=read(b/'root-submission.json');assert cp['checkpoint']==held['checkpoint']and cp['cursor']==held['cursor']and cp['job_id']==old['job_id']==held['job_id']
  assert cp['status']=='PUBLISHED_EPOCH_BOUNDARY_CPU_SHA_RAW_EMA_ADAM_TWO_RANK_RNG_ACCEPTED'
  tx={'held':held,'CPU_acceptance':proof(accepted),'old_root_receipt':old,'time':time.time()};atomic(transaction,tx)
 held=tx['held'];old=tx['old_root_receipt'];oldjob=old['job_id'];epoch=held['epoch'];cursor=held['cursor'];name='q7b-P80-L%d-cont'%layer;entry=b/'continuation90.sbatch'
 template=(b/'training.sbatch').read_text();oldpub=read(S/'depth-expansion-source-publication.json')
 text=template.replace(old['name'],name).replace(oldpub['code_root'],stage['code_root']).replace('qwen7b_warmup.online_mixed_depth_expansion','qwen7b_warmup.online_mixed_depth_resume');text=re.sub(r'(?m)^#SBATCH --time=\S+$','#SBATCH --time=01:30:00',text)
 assert stage['code_root']in text and stage['resume_module']in text and'--time=01:30:00'in text
 if entry.exists():assert entry.read_text()==text
 else:entry.write_text(text)
 run(['bash','-n',str(entry)])
 if 'new_job_id'not in tx:
  queued=run(['squeue','-h','-u','mou176','-n',name,'-o','%i']).splitlines()
  if queued:
   assert len(queued)==1;job=queued[0];details=run(['scontrol','show','job',job,'-o']);assert'Command='+str(entry)in details and'UserId=mou176('in details
  else:job=run(['sbatch','--parsable','--hold','--exclude=ACD1-13',str(entry)]).split(';')[0];assert job.isdigit()
  tx.update(new_job_id=job,new_entry=proof(entry),new_runtime_publication=stage['publication'],stage='SUCCESSOR_HELD_OLD_WRITER_NOT_RELEASED');atomic(transaction,tx)
 job=tx['new_job_id'];ev=b/('evaluation-epoch-%02d.json'%epoch)
 if not tx.get('old_writer_cancel_requested'):
  assert not ev.exists()and read(b/'latest.json')==held['checkpoint']
  assert all(read(b/('progress-rank-%d.json'%rank))['cursor']==cursor for rank in(0,1))
  prior=run(['sacct','-X','-n','-P','-j',oldjob,'--format=JobIDRaw,State,ExitCode'])
  terminal=any(row.split('|')[0]==oldjob and row.split('|')[1].startswith(('CANCELLED','TIMEOUT'))for row in prior.splitlines())
  tx['old_writer_cancel_requested']=time.time();tx['old_already_terminal_before_handoff']=terminal;atomic(transaction,tx)
  if not terminal:run(['scancel','--signal=KILL','--full',oldjob])
 state=run(['sacct','-X','-n','-P','-j',oldjob,'--format=JobIDRaw,State,ExitCode'])
 if not any(row.split('|')[0]==oldjob and row.split('|')[1].startswith(('CANCELLED','TIMEOUT'))for row in state.splitlines()):out['jobs'][str(layer)]={'status':'WAIT_OLD_JOB_TERMINATION','job_id':oldjob,'successor':job,'accounting':state};continue
 locks=[]
 try:
  for rank in(0,1):
   f=(b/('writer-rank-%d.lock'%rank)).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 except BlockingIOError:
  for f in locks:f.close()
  out['jobs'][str(layer)]={'status':'WAIT_OLD_WRITER_LOCK_RELEASE','successor':job};continue
 for f in locks:f.close()
 archive=b/('attempts/complete-epoch-%d-handoff-%s'%(epoch,oldjob));archive.mkdir(parents=True,exist_ok=True)
 if not tx.get('root_identity_moved'):
  for filename in ['root-submission.json','root-submission-intent.json','controller.json','progress-rank-0.json','progress-rank-1.json','walltime-boundary-held.json']:
   p=b/filename
   if p.exists():shutil.move(str(p),str(archive/filename))
  for p in b.glob('epoch-%02d-*-predictions.jsonl.gz'%epoch):shutil.move(str(p),str(archive/p.name))
  receipt={'job_id':job,'name':name,'entry':proof(entry),'runtime_commit':'74178e17320d2a3d44eca00766c3061330958bee','source_publication':stage['publication'],'time':time.time(),'resources':{'partition':'acd_u','GPUs':2,'CPUs':12,'memory_GiB':128,'time':'01:30:00'},'exclude':'ACD1-13','previous_job_id':oldjob,'resume_checkpoint':held['checkpoint'],'expected_initial_cursor':cursor,'completed_epoch':epoch,'phase':'COMPLETE_EPOCH_EVALUATION_PENDING_THEN_NEXT_EPOCH','no_dependencies':True,'discarded_optimizer_updates':0}
  atomic(b/'root-submission.json',receipt);tx['root_identity_moved']=True;atomic(transaction,tx)
 run(['scontrol','release',job]);tx.update(stage='ACCEPTED_BOUNDARY_SUCCESSOR_RELEASED',time=time.time(),old_accounting=state,no_partial_epoch_discarded=True,goal_paused=False);atomic(transaction,tx);atomic(done,tx);out['jobs'][str(layer)]={'status':tx['stage'],'old_job_id':oldjob,'new_job_id':job,'resume_cursor':cursor,'checkpoint':held['checkpoint'],'discarded_optimizer_updates':0}
print(json.dumps(out))
