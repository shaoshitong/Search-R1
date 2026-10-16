import hashlib,json,os,re,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
UNIT='sst-tree-nested-branch-fresh200-778b2b43.service'
PID='503255';INVOCATION='a313bda919834d24b4c5d491698d39ce'
RUN=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:778b2b43dad90fc0c1c5d95d77c2326286011b73')
EXPECTED='/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/runs/'+RUN.name
os.nice(15)
if str(RUN.resolve(strict=True))!=EXPECTED:raise ValueError('Original retained18 run identity differs')
def systemd():
 raw=subprocess.check_output(['systemctl','show',UNIT,'--property=MainPID,InvocationID,ControlGroup,CPUUsageNSec,MemoryCurrent,MemoryMax,TasksCurrent,ActiveState,SubState,Result,ExecMainStatus'],text=True)
 d=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
 if d.get('MainPID')!=PID or d.get('InvocationID')!=INVOCATION or not d.get('ControlGroup','').endswith('/'+UNIT):raise ValueError('Original unit/PID/Invocation/cgroup differs')
 return d
initial=systemd();CG=initial['ControlGroup'];cg=Path('/sys/fs/cgroup')/CG.lstrip('/')
def text(path,limit=10000):
 try:
  with path.open('rb') as f:return f.read(limit).decode('utf8','replace')
 except OSError as e:return {'unavailable_errno':e.errno}
def kv(value):
 if not isinstance(value,str):return value
 out={}
 for line in value.splitlines():
  k,v=line.split(None,1);out[k]=int(v) if v.isdigit() else v
 return out
def own(pid):
 value=text(Path('/proc')/str(pid)/'cgroup',2000)
 return isinstance(value,str) and any(line.split(':',2)[-1]==CG or line.split(':',2)[-1].startswith(CG+'/') for line in value.splitlines())
def pid_snapshot(pid):
 p=Path('/proc')/str(pid)
 try:
  if not own(pid):return {'pid':pid,'scope_changed':True}
  raw=(p/'stat').read_text();parts=raw[raw.rfind(')')+2:].split();status=text(p/'status',8000)
  mem={}
  if isinstance(status,str):
   for line in status.splitlines():
    if line.startswith(('VmRSS:','VmSwap:','Threads:')):
     k,v=line.split(':',1);mem[k]=v.strip()
  io=kv(text(p/'io',3000));io={k.rstrip(':'):v for k,v in io.items()} if isinstance(io,dict) else io
  return dict(pid=pid,comm=raw[raw.find('(')+1:raw.rfind(')')][:96],state=parts[0],ppid=int(parts[1]),cpu_seconds=(int(parts[11])+int(parts[12]))/os.sysconf('SC_CLK_TCK'),minor_faults=int(parts[7]),major_faults=int(parts[9]),threads=int(parts[17]),start_ticks=int(parts[19]),wchan=text(p/'wchan',128),memory=mem,io=io)
 except (OSError,ValueError,IndexError) as e:return {'pid':pid,'unavailable':type(e).__name__}
def statfile(path):
 try:
  s=path.stat();return {'path':str(path.relative_to(RUN)),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
 except FileNotFoundError:return {'path':str(path.relative_to(RUN)),'exists':False}
def snapshot():
 unit=systemd();pids=set();groups=0
 for root,dirs,files in os.walk(cg):
  groups+=1
  if groups>64:raise ValueError('Own cgroup subtree exceeds lightweight scope')
  if 'cgroup.procs' in files:pids.update(int(v) for v in (Path(root)/'cgroup.procs').read_text().split())
 pids=sorted(pid for pid in pids if own(pid))
 if int(PID) not in pids or len(pids)>256:raise ValueError('Own bounded process set differs')
 ps=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid=,ppid=,pcpu=,etimes=,stat=,wchan:32=,comm:64='],capture_output=True,text=True)
 stats={name:kv(text(cg/name)) if name in ('cpu.stat','memory.events','memory.stat') else text(cg/name) for name in ['cpu.stat','cpu.max','cpu.pressure','memory.current','memory.max','memory.high','memory.swap.current','memory.swap.max','memory.events','memory.stat','memory.pressure','io.stat','io.pressure','pids.current','pids.max']}
 files=[statfile(RUN/name) for name in ['stage.log','retriever.log','process-resources.jsonl','resource-samples.jsonl']]
 rankdir=RUN/'output/tree-rank-audit'
 ranks={str(step):[statfile(p) for p in sorted(rankdir.glob('step-%06d-attempt-*-rank-*.json'%step))] for step in [180,181]}
 controlpath=RUN/'output/training-rollout-cost/attempt-000181.json';control=statfile(controlpath)
 if control.get('exists') is not False:
  if control['bytes']>20000:raise ValueError('Controller scalar metadata budget differs')
  d=json.loads(controlpath.read_bytes());control.update({k:d.get(k) for k in ['status','candidate_outer','attempt_id','start_epoch','end_epoch','wall_seconds','trainer_process_cpu_seconds','code_commit','config_sha256']})
 with (RUN/'stage.log').open('rb') as f:f.seek(max(0,(RUN/'stage.log').stat().st_size-4096));tail=f.read(4096).decode('utf8','replace')
 tail=re.sub(r'(?i)(bearer\s+|(?:token|password|secret|api_key)\s*[=:]\s*)[^\s,;]+',r'\1[REDACTED]',tail)
 return dict(observed_utc=datetime.now(timezone.utc).isoformat(),epoch=time.time(),systemd=unit,cgroup_path=str(cg),own_process_count=len(pids),ps_columns='pid ppid pcpu etimes stat wchan comm-only(no argv/env)',ps_exit=ps.returncode,ps_stdout=ps.stdout,ps_stderr=ps.stderr,processes=[pid_snapshot(pid) for pid in pids],cgroup=stats,files=files,rank_file_stats=ranks,controller181=control,stage_tail_utf8=tail)
a=snapshot();time.sleep(12);b=snapshot()
result=dict(schema='own_outer181_cpu_io_two_observations_v1',unit=UNIT,original_PID=int(PID),InvocationID=INVOCATION,run=str(RUN),resolved_run=EXPECTED,snapshots=[a,b],interval_seconds=b['epoch']-a['epoch'],query_helper_invocations=1,remote_files_written=False,process_attach_or_training_mutation=False,other_units_or_tasks_queried=False,command_scope='comm-only; no argv, environment or credentials read',large_resource_files_stat_only=True,rank_files_stat_only=True)
encoded=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)
if len(encoded.encode())>95000:raise ValueError('Small diagnostic output budget exceeded')
print(encoded)
