"""Controlled performance-only handoff for the owned CoSearch evaluation."""
import argparse, fcntl, hashlib, json, os, pathlib, signal, time
OUT=pathlib.Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-final200-four-gpu-20261015')
JOB=721542
CONFIG_COMMIT='a989c01d39c60aa5fb2bc767cf0c7da0da638fce'
CONFIG_SHA='139275bc73f7d37aedbf8971116c19f8eebce18105a8fecd00509642bafb3a86'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,d):
 t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w') as f: json.dump(d,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 t.replace(p)
def read(n):return json.loads((OUT/n).read_text())
def prepare():
 assert os.environ.get('SLURM_JOB_ID')==str(JOB)
 assert not (OUT/'complete.json').exists()
 config=OUT/'performance-configs/concurrency256.json'
 assert sha(config)==CONFIG_SHA
 assert read('progress.json')['question_concurrency']==128
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
 intent=dict(epoch=time.time(),status='CONTROLLED_CONCURRENCY128_TO256_HANDOFF',
  job=JOB,evaluator_pid=pid,evaluator_start_ticks=tick,coordinator_pid=parent,
  producer_binding_sha256=sha(OUT/'binding.json'),manifest_sha256=sha(OUT/'source-manifest.json'),
  configuration_commit=CONFIG_COMMIT,configuration_sha256=CONFIG_SHA,
  accepted_rows=len(list((OUT/'rows').glob('*.json'))),baseline_progress=read('progress.json'),
  previous_wait_id=read('repair-wait.json')['wait_id'],model_and_protocol_unchanged=True)
 assert not (OUT/'performance-handoff-intent.json').exists()
 atomic(OUT/'performance.json',json.loads(config.read_text()))
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
  atomic(history/('baseline-'+n),read(n))
 command=dict(epoch=time.time(),job=JOB,wait_id=w['wait_id'],
  manifest_sha256=intent['manifest_sha256'],configuration_commit=CONFIG_COMMIT,
  reason='Measured higher concurrency trial; preserve all accepted rows and immutable protocol binding')
 atomic(OUT/'repair-command.json',command)
 atomic(OUT/'performance-handoff-approved.json',dict(command=command,accepted_rows=intent['accepted_rows']))
 print(json.dumps(command))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','approve']);a=p.parse_args()
 (prepare if a.mode=='prepare' else approve)()
