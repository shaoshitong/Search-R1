"""Root-only second seed of the completed CURRENT64 configuration."""
import pathlib,json,subprocess,time,hashlib,fcntl,re
s=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-current64-seed19137-plan2026102302-L15P80-8192-20261023-v1');b=s/'current64'
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 if p.returncode:raise RuntimeError(p.stdout[-800:]+p.stderr[-1000:])
 return p.stdout.strip()
lock=(b/'root-submit.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);entry=b/'training.sbatch';assert proof(entry)['sha256']=='bfed2ae69a56f4f3550da8858036d7248c3faa9aa59d9ebe8e09cb0fd096c825'
pub=read(s/'source-publication.json');assert pub['commit']=='34579cbfb9729afb21c88b540b7ded7d3db823f4'and proof(s/'source-publication.json')['sha256']=='f01a5a4111ac0c4c21637658b07936893fb23b65bacfaf5ebdebb5904495b7a6'
ready=read(b/'ready.json');assert ready['entry']==proof(entry)
receipt=b/'root-submission.json';intent=b/'root-submit-intent.json'
if receipt.exists():print(json.dumps(read(receipt)))
else:
 name=re.search(r'^#SBATCH --job-name=(.+)$',entry.read_text(),re.M).group(1);rows=run(['squeue','-h','-u','mou176','-n',name,'-o','%i']).splitlines()
 if rows:
  assert len(rows)==1;job=rows[0];details=run(['scontrol','show','job',job,'-o']);assert 'UserId=mou176('in details and 'Command='+str(entry)in details
 else:
  if intent.exists():raise RuntimeError('Prior submission outcome unknown; inspect same job name/accounting before retrying.')
  assert not(b/'latest.json').exists()and not(b/'controller.json').exists()
  intent.write_text(json.dumps(dict(time=time.time(),entry=proof(entry),name=name),sort_keys=True)+'\n');job=run(['sbatch','--parsable','--exclude=ACD1-13',str(entry)]).split(';')[0];assert job.isdigit()
 result=dict(time=time.time(),job_id=job,name=name,entry=proof(entry),ready=proof(b/'ready.json'),source_commit=pub['commit'],init_seed=19137,train_plan_seed=2026102302,GPUs=2,partition='acd_u',epochs=10,global_batch=64,batch_attempt_budget=1280,fresh=True,resume_checkpoint=None,primary_metric='Top512_real_Mixed',replication_epoch=5,replication_state='raw',final4096_read=False,no_dependencies=True)
 receipt.write_text(json.dumps(result,sort_keys=True)+'\n');print(json.dumps(result))
