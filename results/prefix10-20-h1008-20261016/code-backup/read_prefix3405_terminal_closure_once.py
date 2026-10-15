"""Read-only terminal closure; source logs and full hashes remain remote."""
import datetime,json,sys
from pathlib import Path
repo=Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix10-20-h1008-20261016');sys.path.insert(0,str(repo))
from reproduction.h1008_qprobe.management_transport import remote
work=Path(__file__).parent;control=json.loads((work/'prefix10-20-current-control.json').read_text());mgmt=json.loads((work/'qprobe3-management-current.json').read_text());tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
script='''import datetime,json,hashlib,os,socket,subprocess,time
from pathlib import Path
base=Path(BASE);expected_run=RUN;job='3405'
assert socket.gethostname().split('.')[0]=='h100-8'
def read(p):return json.loads(Path(p).read_text())
def command(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=30);return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def identity(pid):
 p=Path('/proc')/str(pid);raw=(p/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(v[19]),sid=int(v[3]),uid=p.stat().st_uid,state=v[0])
def same(old,new):return all(old[k]==new[k] for k in ('pid','start_ticks','sid','uid'))
def observation():
 native=command(['scontrol','show','job',job,'-o']);queue=command(['squeue','-h','-o','%i|%T'])
 gpu=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'])
 assert queue['returncode']==0 and gpu['returncode']==0
 ledger=read(base/'logs/process-ledger.json');assert ledger['run_id']==expected_run
 births=[r for task in ledger['jobs'].values() for r in task['ledger'].values()];present=[]
 for birth in births:
  try:now=identity(birth['pid'])
  except (FileNotFoundError,ProcessLookupError):continue
  if same(birth,now):present.append(now)
 cgroups=[str(p) for pattern in ('**/job_'+job,'**/job'+job) for p in Path('/sys/fs/cgroup').glob(pattern)]
 apps=[]
 for line in gpu['stdout'].splitlines():
  if not line.strip():continue
  uuid,pid=[v.strip() for v in line.split(',')];pid=int(pid)
  try:now=identity(pid)
  except (FileNotFoundError,ProcessLookupError):now=None
  apps.append(dict(gpu_uuid=uuid,pid=pid,owned=any(r['pid']==pid and (now is None or same(r,now)) for r in births)))
 cleanup=read(base/'logs/cleanup-status.json') if (base/'logs/cleanup-status.json').exists() else None
 complete=read(base/'logs/pipeline-complete.json') if (base/'logs/pipeline-complete.json').exists() else None
 fields={}
 if native['returncode']==0:
  for piece in native['stdout'].split():
   if '=' in piece:k,v=piece.split('=',1);fields[k]=v
 terminal=fields.get('JobId')==job and fields.get('NodeList')=='h100-8' and fields.get('JobState') in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE')
 absent=job not in {line.split('|')[0] for line in queue['stdout'].splitlines()}
 return dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=job,run_id=expected_run,native=native,native_fields=fields,actual_native_terminal=terminal,actual_native_exit0=terminal and fields.get('JobState')=='COMPLETED' and fields.get('ExitCode')=='0:0',queue_absent=absent,original_owned_birth_count=len(births),present_owned_births=present,active_owned_births=[r for r in present if r['state']!='Z'],zombie_owned_births=[r for r in present if r['state']=='Z'],owned_cgroups=cgroups,gpu_applications=apps,cleanup=cleanup,pipeline_complete=complete,owned_resources_released=absent and not present and not cgroups and not any(r['owned'] for r in apps) and bool(cleanup and cleanup.get('run_id')==expected_run and cleanup.get('cleaned') is True))
first=observation();second=observation();audit=base/'diagnostics'/('native-terminal-closure-'+TAG);audit.mkdir(parents=True,exist_ok=False)
for name,value in [('observation1.json',first),('observation2.json',second)]:
 with (audit/name).open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
records=[]
if first['actual_native_terminal'] and second['actual_native_terminal']:
 paths=set(p for p in (base/'logs').rglob('*') if p.is_file() and not p.is_symlink())
 for prefix in (10,20):
  group=base/'groups'/('prefix-'+str(prefix))
  paths.update(p for p in (group/'train').glob('*') if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.log'))
  paths.update(p for pattern in ('progress-rank-*.json','feature-failure-rank-*.json','features-complete.json') for p in (group/'features').glob(pattern) if p.is_file() and not p.is_symlink())
 for p in sorted(paths):
  before=p.stat();h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
  after=p.stat();assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),str(p)
  records.append(dict(relative_path=str(p.relative_to(base)),inode=after.st_ino,device=after.st_dev,bytes=after.st_size,mtime_ns=after.st_mtime_ns,sha256=h.hexdigest()))
proof=dict(role='actual_native_terminal_original_logs_full_sha',base=str(base),job_id=job,run_id=expected_run,observations=[first,second],original_log_files=records,full_file_sha_performed=bool(records),original_logs_copied_to_mac=False,model_or_raw_trajectory_read=False)
with (audit/'RESULT.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
resultsha=hashlib.sha256((audit/'RESULT.json').read_bytes()).hexdigest()
summary={k:second[k] for k in ('job_id','run_id','native_fields','actual_native_terminal','actual_native_exit0','queue_absent','original_owned_birth_count','present_owned_births','active_owned_births','zombie_owned_births','owned_cgroups','gpu_applications','cleanup','pipeline_complete','owned_resources_released')}
summary.update(actual_two_observations=True,first_actual_exit0=first['actual_native_exit0'],first_resources_released=first['owned_resources_released'],remote_original_log_files_hashed=len(records),remote_original_log_bytes_hashed=sum(r['bytes'] for r in records),remote_proof=str(audit/'RESULT.json'),remote_proof_sha256=resultsha,original_logs_copied_to_mac=False)
print(json.dumps(summary))
'''.replace('BASE',repr(control['base'])).replace('RUN',repr(control['run_id'])).replace('TAG',repr(tag))
evidence=work/('prefix3405-terminal-closure-'+tag+'.json');out=remote(script,evidence,mgmt['socket']);print(out);print('TRANSPORT_EVIDENCE='+str(evidence))
