"""Root-only fresh second-seed submission with persistent outcome tracking."""
import pathlib,json,subprocess,time,hashlib,fcntl,re
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-Z-seed19137-plan2026102302-P20-20261023-v1')
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 if p.returncode:raise RuntimeError(p.stdout[-800:]+p.stderr[-1000:])
 return p.stdout.strip()
lock=(b/'root-submit.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ready=read(b/'ready.json');entry=b/'training.sbatch'
assert ready['status']=='FRESH_SECOND_TRAINING_SEED_Z_REPLICA_STAGED_NOT_SUBMITTED'and ready['entry']==proof(entry)and ready['source_commit']=='e5e5678ec6f42669509b493f08fab0dbea4994a8'
assert proof(entry)['sha256']=='e7e097ccbd429a0344c1ca9503ce91ec122ca5f7d4056ff934495b4194c3d662'and proof(b/'source-publication.json')['sha256']=='4b75af0e51d4745e83dbfbfe03945d0814c7351874785e2b8fe4520ad603047c'
t=ready['training'];assert t['init_seed']==19137 and t['train_plan_seed']==2026102302 and t['diagnostic_seed']==2026102228 and t['epochs']==10 and t['start_mode']=='fresh'
receipt=b/'root-submission.json';intent=b/'root-submit-intent.json'
if receipt.exists():print(json.dumps(read(receipt)))
else:
 name=re.search(r'^#SBATCH --job-name=(.+)$',entry.read_text(),re.M).group(1);rows=run(['squeue','-h','-u','mou176','-n',name,'-o','%i']).splitlines()
 if rows:
  assert len(rows)==1;job=rows[0];details=run(['scontrol','show','job',job,'-o']);assert 'Command='+str(entry)in details and 'UserId=mou176('in details
 else:
  if intent.exists():raise RuntimeError('Previous submit outcome unknown; inspect same name/accounting before retrying.')
  assert not(b/'latest.json').exists()and not(b/'controller.json').exists(),'Fresh replica must not silently resume another run.'
  intent.write_text(json.dumps({'time':time.time(),'entry':proof(entry),'name':name},sort_keys=True)+'\n')
  job=run(['sbatch','--parsable','--exclude=ACD1-13',str(entry)]).split(';')[0];assert job.isdigit()
 result={'time':time.time(),'job_id':job,'name':name,'entry':proof(entry),'ready':proof(b/'ready.json'),'source_commit':ready['source_commit'],'initialization_seed':19137,'train_plan_seed':2026102302,'fresh':True,'resume_checkpoint':None,'GPUs':2,'partition':'acd_u','time_limit':'08:00:00','epochs':10,'global_batch':128,'LR':5e-6,'EMA_decay':.9995,'no_dependencies':True,'final4096_read':False}
 receipt.write_text(json.dumps(result,sort_keys=True)+'\n');print(json.dumps(result))
