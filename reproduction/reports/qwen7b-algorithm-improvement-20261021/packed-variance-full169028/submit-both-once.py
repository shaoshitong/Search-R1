"""Root-owned one-time submission control; execute remotely only by Root.

This file is prepared for Root and is never called by the stage controller.
An existing receipt is returned, not replaced; uncertain prior intent requires
reconciliation, never a blind second submission. No other job is modified.
"""
import fcntl,hashlib,json,os,pathlib,re,subprocess,time
BASE=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1')

def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def once(p,v):
 raw=(json.dumps(v,sort_keys=True,separators=(',',':'))+'\n').encode()
 if p.exists():assert p.read_bytes()==raw,'Preserve existing submission artifact'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 if r.returncode:raise RuntimeError(r.stderr[-1400:])
 return r.stdout.strip()
def main(loss):
 B=BASE/loss;JOB='q7b-full-var-'+loss
 guard=(B/'root-submit-control.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ready=read(B/'ready.json');assert ready['status']=='FULL169028_PACKED_VARIANCE_READY_FIXTURE_REQUIRED_NOT_SUBMITTED'and ready['loss']==loss;fixture=read(BASE/'fixture/fixture-complete.json');assert fixture['status']=='PACKED_VARIANCE_OWN_G10_POLICY20_FIXTURE_READY'and fixture['training_questions']==169028 and fixture['validation_questions']==2048 and fixture['trajectories']==1710760
 assert ready['training']['EMA_decay']==.9995 and ready['training']['epochs']==10 and ready['training']['pilot_checkpoint_resumed']is False;assert proof(ready['descriptor']['path'])==ready['descriptor']
 assert ready['resources']==dict(partition='acd_u',qos='normal',GPUs=2,CPUs=12,memory_GiB=128,time='08:00:00')
 assert ready['training']['global_batch']==128 and ready['training']['accumulation']==8 and ready['training']['updates']==13210
 for ref in(ready['entry'],ready['source_publication']):assert proof(ref['path'])==ref
 for key in('train_groups','validation_groups','fixed_validation_G5','fixed_train_G5'):
  ref=fixture[key];assert proof(ref['path'])==ref
 for ref in fixture['token_shards']:assert proof(ref['path'])==ref
 receipt=B/'root-submission.json';intent=B/'root-submission-intent.json'
 if receipt.exists():
  prior=read(receipt);assert prior['name']==JOB
  snapshot=command(['sacct','-X','-n','-P','-j',prior['job_id'],'--format=JobIDRaw,State,ExitCode'])
  print(json.dumps(dict(status='EXISTING_PACKED_VARIANCE_SUBMISSION_RECONCILED_NO_NEW_SUBMIT',receipt=prior,sacct=snapshot)));return
 same=command(['squeue','-h','-u','mou176','-n',JOB,'-o','%i|%T|%R'])
 assert not same,'Actual same-name job found; reconcile identity, never duplicate: '+same
 assert not intent.exists(),'Prior intent without receipt: inspect same-name squeue/sacct before any retry'
 assert not any((B/n).exists()for n in('controller.json','latest.json','progress-rank-0.json','progress-rank-1.json','complete.json')),'Existing run artifacts require reconciliation'
 lock=(B/'training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
 once(intent,dict(name=JOB,time=time.time(),entry=ready['entry'],ready=proof(B/'ready.json'),publication_commit=ready['runtime_commit'],source_controller=ready['root_submit_controller']))
 answer=command(['sbatch','--hold','--parsable',str(B/'training.sbatch')])
 if not re.fullmatch(r'[0-9]+(?:;[^ ]+)?',answer):raise RuntimeError('Unknown submission result; preserve intent and inspect same-name jobs: '+answer)
 job=answer.split(';')[0];identity=command(['scontrol','show','job','-o',job])
 for field in('JobName='+JOB,'UserId=mou176','Account=mou176','QOS=normal','Partition=acd_u','NumCPUs=12','MinMemoryNode=128G','TimeLimit=08:00:00','Dependency=(null)','Command='+str(B/'training.sbatch')):assert field in identity,field
 assert'gres/gpu=2'in identity or'TresPerNode=gres/gpu:2'in identity
 result=dict(job_id=job,layer=21,prefix=20,M_range=[3,10],loss=loss,epochs=10,EMA_decay=.9995,name=JOB,publication_commit=ready['runtime_commit'],source_controller=ready['root_submit_controller'],entry=ready['entry'],implementation_ready=proof(B/'ready.json'),slurm_identity_before_release=identity,time=time.time(),state='SUBMITTED_NOT_PROGRESS_VERIFIED')
 once(receipt,result);command(['scontrol','release',job]);print(json.dumps(dict(status='PACKED_VARIANCE_2GPU_JOB_RELEASED_NOT_PROGRESS_VERIFIED',receipt=result)))
if __name__=='__main__':
 for loss in('mse','mae'):main(loss)
