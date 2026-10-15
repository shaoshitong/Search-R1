"""Restore measured faster baseline without losing any accepted evaluation result."""
import argparse, fcntl, hashlib, json, os, pathlib, signal, time
OUT=pathlib.Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-final200-four-gpu-20261015')
JOB=721542
CONFIG_COMMIT=os.environ['COSEARCH_PERFORMANCE_PUBLICATION_COMMIT']
CONFIG_SHA='109c9b8fdbaba6a08c43dde5210d4aca19079cb4c0347e8b9d1da08178a598a0'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,d):
 t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w') as f: json.dump(d,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 t.replace(p)
def read(n):return json.loads((OUT/n).read_text())
def prepare():
 assert os.environ.get('SLURM_JOB_ID')==str(JOB)
 assert not (OUT/'complete.json').exists()
 config=OUT/'performance-configs/concurrency128.json'
 assert sha(config)==CONFIG_SHA
 assert read('progress.json')['question_concurrency']==256
 candidates=[]
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=os.getuid():continue
   args=proc.joinpath('cmdline').read_bytes().split(b'\0')
   if str(OUT/'tools/evaluate.py').encode() not in args:continue
   if 'job_'+str(JOB) not in proc.joinpath('cgroup').read_text():continue
   stat=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
   parent=int(stat[1]);tick=stat[19]
   parent_args=pathlib.Path('/proc',str(parent),'cmdline').read_bytes().split(b'\0')
   assert str(OUT/'tools/run.py').encode() in parent_args
   candidates.append((int(proc.name),parent,tick))
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 assert len(candidates)==1,candidates
 pid,parent,tick=candidates[0]
 intent=dict(epoch=time.time(),status='MEASURED_CONCURRENCY256_TO128_RESTORE',
  job=JOB,evaluator_pid=pid,evaluator_start_ticks=tick,coordinator_pid=parent,
  producer_binding_sha256=sha(OUT/'binding.json'),manifest_sha256=sha(OUT/'source-manifest.json'),
  configuration_commit=CONFIG_COMMIT,configuration_sha256=CONFIG_SHA,
  accepted_rows=len(list((OUT/'rows').glob('*.json'))),baseline_progress=read('progress.json'),
  previous_wait_id=read('repair-wait.json')['wait_id'],model_and_protocol_unchanged=True)
 prior=read('performance-handoff-intent.json')
 assert read('repair-wait.json')['status']=='VERIFIED_REPAIR_ACCEPTED_EXECUTING'
 history=OUT/'performance-handoff-history';history.mkdir(exist_ok=True)
 atomic(history/'concurrency256-handoff-intent.json',prior)
 atomic(history/'concurrency256-handoff-approved.json',read('performance-handoff-approved.json'))
 atomic(history/'concurrency256-acceptance.json',read('performance-256-acceptance.json'))
 temporary=OUT/'performance.json.tmp'
 with temporary.open('wb') as stream:
  stream.write(config.read_bytes());stream.flush();os.fsync(stream.fileno())
 temporary.replace(OUT/'performance.json')
 assert sha(OUT/'performance.json')==CONFIG_SHA
 atomic(OUT/'performance-handoff-intent.json',intent)
 assert pathlib.Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]==tick
 os.kill(pid,signal.SIGTERM)
 print(json.dumps(intent))
def approve():
 intent=read('performance-handoff-intent.json');w=read('repair-wait.json')
 assert w['job']==JOB and w['pid']==intent['coordinator_pid']
 assert w['status']=='OWNED_FAILED_SERVICES_STOPPED_AWAITING_VERIFIED_REPAIR'
 assert w['wait_id']!=intent['previous_wait_id'] and w['deadline_epoch']>time.time()+120
 assert sha(OUT/'source-manifest.json')==intent['manifest_sha256']
 assert sha(OUT/'binding.json')==intent['producer_binding_sha256']
 assert sha(OUT/'performance.json')==CONFIG_SHA
 assert read('cleanup-'+str(JOB)+'.json')['all_owned_children_reaped']
 assert len(list((OUT/'rows').glob('*.json')))>=intent['accepted_rows']
 with (OUT/'eval.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Release our test lock before authorizing the coordinator.
 history=OUT/'performance-handoff-history';history.mkdir(exist_ok=True)
 for n in ['repair-wait.json','failure-'+str(JOB)+'.json','cleanup-'+str(JOB)+'.json']:
  atomic(history/('concurrency256-end-'+n),read(n))
 command=dict(epoch=time.time(),job=JOB,wait_id=w['wait_id'],
  manifest_sha256=intent['manifest_sha256'],configuration_commit=CONFIG_COMMIT,
  reason='Restore measured faster 128 concurrency; preserve all accepted rows and immutable protocol binding')
 atomic(OUT/'repair-command.json',command)
 atomic(OUT/'performance-handoff-approved.json',dict(command=command,accepted_rows=intent['accepted_rows']))
 print(json.dumps(command))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','approve']);a=p.parse_args()
 (prepare if a.mode=='prepare' else approve)()
