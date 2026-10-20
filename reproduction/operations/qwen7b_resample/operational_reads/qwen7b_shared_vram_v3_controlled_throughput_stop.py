"""Stop only evidenced v3 after two saved batches for the wider-VRAM continuation."""
import fcntl,hashlib,json,os,socket,subprocess,time
from pathlib import Path
BASE=Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
UNIT='sst-q7b-g5-sharedvram-v3-70188b8e50fd.service'
PID=1259750
SOURCE_SHA256=globals().get('SOURCE_SHA256')
assert isinstance(SOURCE_SHA256,str)and len(SOURCE_SHA256)==64
RUN='70188b8e50fd4b45bc420ce58014f46e'
assert os.geteuid()==0 and socket.gethostname().split('.')[0]=='h100-5'
def read(p):return json.loads(p.read_text())if p.exists()else None
def atomic(p,v):
 q=p.with_name(p.name+'.tmp-'+str(os.getpid()))
 with q.open('x')as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def unit():
 r=subprocess.run(['systemctl','show',UNIT]+['--property='+x for x in ('Id','MainPID','ControlGroup','ActiveState','SubState','Result','ExecMainStatus')],text=True,capture_output=True,timeout=20)
 return dict(x.split('=',1)for x in r.stdout.splitlines()if'='in x)
def birth(pid):
 try:return Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except(FileNotFoundError,ProcessLookupError):return None
def owned(group):
 out={}
 for p in Path('/proc').glob('[0-9]*'):
  try:
   groups=[x.split(':',2)[2]for x in(p/'cgroup').read_text().splitlines()]
   if any(x==group or x.startswith(group+'/')for x in groups):
    value=birth(int(p.name))
    if value is not None:out[p.name]=value
  except(FileNotFoundError,ProcessLookupError,PermissionError):pass
 return out
root=BASE/'preparation/shared-vram-v3-controlled-throughput-stop-v1';root.mkdir(parents=True,exist_ok=True)
with(root/'control.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 intent=root/'intent.json';receipt=root/'receipt.json'
 if intent.exists():
  print(json.dumps(dict(status='EXISTING_STOP_INTENT_NEVER_REPLAYED',intent=read(intent),receipt=read(receipt),actual_unit=unit())));raise SystemExit(0)
 current=unit();assert current['Id']==UNIT and current['MainPID']==str(PID)and current['ActiveState']=='active'
 group=current['ControlGroup'];assert group.endswith('/'+UNIT)
 attempts=list((BASE/'pool/jobs'/('job-shared5-'+str(PID))).glob('attempt-*'));assert len(attempts)==1
 job=attempts[0];progress=read(job/'progress.json')
 assert progress['run_id']==RUN and progress['completed_own_batches']>=2,'Wait for two real saved whole G5 batches; no stop intent created'
 known=owned(group);assert str(PID)in known
 proposed=dict(status='CONTROLLED_THROUGHPUT_CONTINUATION_STOP_INTENT',unit=UNIT,main_pid=PID,run_id=RUN,
  source_sha256=SOURCE_SHA256,
  cause='Use newly available free VRAM with two TP2 engines and full original local retrieval; preserve every sealed batch and unsealed partial',
  job_directory=str(job),progress=progress,actual_unit=current,original_owned_births=known,foreign_processes_signalled=False,epoch=time.time())
 with intent.open('x')as f:json.dump(proposed,f,indent=2);f.flush();os.fsync(f.fileno())
 try:
  r=subprocess.run(['systemctl','stop',UNIT],text=True,capture_output=True,timeout=75)
  result=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 except subprocess.TimeoutExpired as error:result=dict(error_type=type(error).__name__,error=str(error),outcome_unknown=True)
 current=unit();remaining={pid:t for pid,t in known.items()if birth(int(pid))==t}
 result.update(status='CONTROLLED_V3_STOP_OBSERVED'if current.get('MainPID')=='0'and not remaining else'CONTROLLED_V3_STOP_REQUIRES_RECONCILIATION',
  intent=str(intent),intent_sha256=hashlib.sha256(intent.read_bytes()).hexdigest(),actual_unit=current,
  original_births_remaining=remaining,cgroup_remaining=owned(group),job_directory=str(job),epoch=time.time(),foreign_processes_signalled=False)
 atomic(receipt,result);print(json.dumps(result))
