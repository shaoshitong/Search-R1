"""Root-only four accepted controls; pureTopK128 intentionally remains unrun."""
import pathlib,json,subprocess,time,hashlib,fcntl,re
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-matched-L15P80-8192-20261023-v1');out={'time':time.time(),'jobs':{},'topk128_not_submitted':'Superseded by human requested matchedglobal64 cached-windowpair.'}
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def run(cmd):
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 if p.returncode:raise RuntimeError(p.stdout[-800:]+p.stderr[-1000:])
 return p.stdout.strip()
accepted=read(b/'fixture/accepted.json');assert accepted['status']=='MATCHED_PACKED_REAL_PREFIX80_FIXTURE_INDEPENDENT_CPU_ACCEPTED'and accepted['fixture']==proof(b/'fixture/fixture-complete.json')
pub=read(b/'source-publication.json');assert pub['commit']=='0436c2c1e3c1b38a2abcbc252bf3524eff0e6deb'
for cell in('mse-Mrandom','mse-M5','mixed-bce-Mrandom','mixed-bce-M5'):
 d=b/cell;lock=(d/'root-submit.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);entry=d/'training.sbatch';ready=read(d/'ready.json');assert ready['entry']==proof(entry)
 receipt=d/'root-submission.json'
 if receipt.exists():out['jobs'][cell]=read(receipt);continue
 name=re.search(r'^#SBATCH --job-name=(.+)$',entry.read_text(),re.M).group(1)
 rows=run(['squeue','-h','-u','mou176','-n',name,'-o','%i']).splitlines();intent=d/'root-submit-intent.json'
 if rows:
  assert len(rows)==1;job=rows[0];details=run(['scontrol','show','job',job,'-o']);assert'Command='+str(entry)in details and'UserId=mou176('in details
 else:
  if intent.exists():raise RuntimeError('Previoussubmitoutcomeunknown; inspectaccountingbeforeanotherwrite: '+cell)
  intent.write_text(json.dumps({'time':time.time(),'name':name,'entry':proof(entry),'source_commit':pub['commit'],'fixture_acceptance':proof(b/'fixture/accepted.json')},sort_keys=True)+'\n')
  job=run(['sbatch','--parsable','--exclude=ACD1-13',str(entry)]).split(';')[0];assert job.isdigit()
 result={'time':time.time(),'job_id':job,'name':name,'cell':cell,'entry':proof(entry),'source_commit':pub['commit'],'fixture_acceptance':proof(b/'fixture/accepted.json'),'no_dependencies':True,'exclude':'ACD1-13','partition':'acd_u','GPUs':2,'CPUs':12,'memory_GiB':128,'time_limit':'02:00:00','epochs':10,'max_batch_attempts':640,'global_batch':128,'layer':15,'prefix':80,'EMA_decay':.9995,'primary_metric':'developmentTop512_then_frozen_unseen_confirmation','final4096_read':False}
 receipt.write_text(json.dumps(result,sort_keys=True)+'\n');out['jobs'][cell]=result
print(json.dumps(out))
