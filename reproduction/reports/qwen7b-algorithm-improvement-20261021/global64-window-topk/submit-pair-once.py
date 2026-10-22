"""Root-only submission with durable intent; never retry an unknown outcome."""
import pathlib,json,subprocess,time,hashlib,fcntl,re
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-global64-window2018-L15P80-8192-20261023-v1')
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def run(cmd):
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 if p.returncode:raise RuntimeError(p.stdout[-800:]+p.stderr[-1000:])
 return p.stdout.strip()
staged=read(b/'staged.json');assert staged['status']=='MATCHED_GLOBAL64_WINDOW_PAIR_STAGED_ACCEPTED_FIXTURE_NOT_SUBMITTED'
pub=read(b/'source-publication.json');assert pub['branch']=='codex/q7b-global64-detached-window-topk-20261023'and staged['publication']==proof(b/'source-publication.json')
reuse=read(b/'fixture-reuse.json');assert reuse['status']=='EXISTING_AUTHENTIC_POLICY80_FIXTURE_FRESH_SHA_REUSED'and staged['fixture_reuse']==proof(b/'fixture-reuse.json')
out={'time':time.time(),'jobs':{}}
for cell,capacity in [('current64',0),('window2018',2018)]:
 d=b/cell;lock=(d/'root-submit.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);entry=d/'training.sbatch';ready=read(d/'ready.json')
 assert ready['entry']==proof(entry)and ready['runtime_commit']==pub['commit']and ready['cache_window_capacity']==capacity
 receipt=d/'root-submission.json'
 if receipt.exists():out['jobs'][cell]=read(receipt);continue
 name=re.search(r'^#SBATCH --job-name=(.+)$',entry.read_text(),re.M).group(1);rows=run(['squeue','-h','-u','mou176','-n',name,'-o','%i']).splitlines();intent=d/'root-submit-intent.json'
 if rows:
  assert len(rows)==1;job=rows[0];details=run(['scontrol','show','job',job,'-o']);assert 'Command='+str(entry)in details and 'UserId=mou176('in details
 else:
  if intent.exists():raise RuntimeError('Previous submit outcome unknown; inspect accounting before any retry: '+cell)
  intent.write_text(json.dumps({'time':time.time(),'name':name,'entry':proof(entry),'source_commit':pub['commit']},sort_keys=True)+'\n')
  job=run(['sbatch','--parsable','--exclude=ACD1-13',str(entry)]).split(';')[0];assert job.isdigit()
 result={'time':time.time(),'job_id':job,'name':name,'cell':cell,'entry':proof(entry),'ready':proof(d/'ready.json'),'source_commit':pub['commit'],'fixture_reuse':proof(b/'fixture-reuse.json'),'no_dependencies':True,'exclude':'ACD1-13','partition':'acd_u','GPUs':2,'CPUs':12,'memory_GiB':128,'time_limit':'02:00:00','epochs':10,'batch_attempt_budget':1280,'global_batch':64,'cache_window_capacity':capacity,'historical_groups_at_capacity':1954 if capacity else 0,'target_top_k':500,'reference_population':2018,'layer':15,'prefix':80,'EMA_decay':.9995,'final4096_read':False}
 receipt.write_text(json.dumps(result,sort_keys=True)+'\n');out['jobs'][cell]=result
print(json.dumps(out))
