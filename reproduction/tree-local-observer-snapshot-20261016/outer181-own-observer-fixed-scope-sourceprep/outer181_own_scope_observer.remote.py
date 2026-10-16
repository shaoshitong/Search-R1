"""Own original unit only; one invocation, two15s-separated frames; no attach."""
import hashlib,json,os,re,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
UNIT='sst-tree-nested-branch-fresh200-778b2b43.service'
MAIN=503255;CONTROLLER=506917;INVOCATION='a313bda919834d24b4c5d491698d39ce'
RUN=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:778b2b43dad90fc0c1c5d95d77c2326286011b73')
EXPECTED='/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/runs/'+RUN.name
os.nice(15);HZ=os.sysconf('SC_CLK_TCK')
if str(RUN.resolve(strict=True))!=EXPECTED:raise ValueError('Original retained18 run identity differs')
def readtext(path,limit=12000):
 try:
  with path.open('rb') as stream:return stream.read(limit).decode('utf8','replace')
 except OSError as error:return {'unavailable_errno':error.errno}
def parse(value):
 if not isinstance(value,str):return value
 result={}
 for line in value.splitlines():
  key,value=line.split(None,1);result[key]=int(value) if value.isdigit() else value
 return result
def systemd():
 raw=subprocess.check_output(['systemctl','show',UNIT,'--property=MainPID,InvocationID,ControlGroup,CPUUsageNSec,MemoryCurrent,MemoryMax,TasksCurrent,ActiveState,SubState,Result,ExecMainStatus'],text=True)
 value=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
 if value.get('MainPID')!=str(MAIN) or value.get('InvocationID')!=INVOCATION or not value.get('ControlGroup','').endswith('/'+UNIT):raise ValueError('Exact original unit/PID/Invocation identity differs')
 return value
first_unit=systemd();CG=first_unit['ControlGroup'];ROOT=Path('/sys/fs/cgroup')/CG.lstrip('/')
def cgroup_membership(pid):return readtext(Path('/proc')/str(pid)/'cgroup',4000)
def identifies_unit(value):
 return isinstance(value,str) and any(UNIT in line.split(':',2)[-1].split('/') for line in value.splitlines())
main_membership=cgroup_membership(MAIN)
if not identifies_unit(main_membership):raise ValueError('MainPID /proc cgroup does not prove exact own unit membership')
def members():
 limit_text=(ROOT/'pids.max').read_text().strip()
 if limit_text!='4096':raise ValueError('Original own4096 task boundary differs')
 pids=set();groups=[]
 for folder,subdirs,files in os.walk(ROOT):
  path=Path(folder)
  if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Own cgroup subtree escaped')
  if 'cgroup.procs' in files:
   values={int(value) for value in (path/'cgroup.procs').read_text().split() if int(value)>0};pids.update(values)
   groups.append(dict(path=str(path.relative_to(ROOT)),processes=len(values)))
 if len(pids)>4096:raise ValueError('Actual unique process count exceeds own4096 task bound')
 return sorted(pids),groups
# Membership comes from the complete own unit subtree, never from a limited ps
# selection. MainPID may live in a child cgroup and is independently proved above.
def procstat(pid,details=False):
 root=Path('/proc')/str(pid)
 try:
  raw=(root/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
  result=dict(pid=pid,comm=raw[raw.find('(')+1:raw.rfind(')')][:96],state=parts[0],ppid=int(parts[1]),cpu_ticks=int(parts[11])+int(parts[12]),cpu_seconds=(int(parts[11])+int(parts[12]))/HZ,start_ticks=int(parts[19]),minor_faults=int(parts[7]),major_faults=int(parts[9]))
  if details:
   result.update(proc_cgroup=cgroup_membership(pid),wchan=readtext(root/'wchan',128),threads=int(parts[17]),io=parse(readtext(root/'io',3000)))
   if isinstance(result['io'],dict):result['io']={key.rstrip(':'):value for key,value in result['io'].items()}
   status=readtext(root/'status',8000);result['memory']={}
   if isinstance(status,str):
    for line in status.splitlines():
     if line.startswith(('VmRSS:','VmSwap:','Threads:')):
      key,value=line.split(':',1);result['memory'][key]=value.strip()
  return result
 except (OSError,ValueError,IndexError) as error:return dict(pid=pid,unavailable=type(error).__name__)
initial_pids,initial_groups=members();candidates=[procstat(pid) for pid in initial_pids]
ranked=sorted([value for value in candidates if 'cpu_ticks' in value],key=lambda value:(-value['cpu_ticks'],value['pid']))[:20]
selected={value['pid'] for value in ranked};selected.add(MAIN)
controller_membership=cgroup_membership(CONTROLLER)
controller_owned=CONTROLLER in initial_pids or identifies_unit(controller_membership)
if controller_owned:selected.add(CONTROLLER)
selected=sorted(selected)
def statfile(path):
 try:
  value=path.stat();return dict(path=str(path.relative_to(RUN)),bytes=value.st_size,mtime_ns=value.st_mtime_ns)
 except FileNotFoundError:return dict(path=str(path.relative_to(RUN)),exists=False)
def snapshot():
 unit=systemd();membership=cgroup_membership(MAIN)
 if not identifies_unit(membership):raise ValueError('MainPID exact own cgroup proof changed')
 pids,groups=members();owned=set(pids);owned.add(MAIN)
 if identifies_unit(cgroup_membership(CONTROLLER)):owned.add(CONTROLLER)
 active=[pid for pid in selected if pid in owned]
 ps=subprocess.run(['ps','-p',','.join(map(str,active)),'-o','pid=,ppid=,pcpu=,etimes=,stat=,wchan:32=,comm:64='],capture_output=True,text=True)
 fields=['cpu.stat','cpu.max','cpu.pressure','memory.current','memory.max','memory.high','memory.swap.current','memory.swap.max','memory.events','memory.stat','memory.pressure','io.stat','io.pressure','pids.current','pids.max']
 cgroup={name:parse(readtext(ROOT/name)) if name in ('cpu.stat','memory.events','memory.stat') else readtext(ROOT/name) for name in fields}
 files=[statfile(RUN/name) for name in ['stage.log','retriever.log','process-resources.jsonl','resource-samples.jsonl']]
 rankdir=RUN/'output/tree-rank-audit';ranks={str(step):[statfile(path) for path in sorted(rankdir.glob('step-%06d-attempt-*-rank-*.json'%step))] for step in [180,181]}
 path=RUN/'output/training-rollout-cost/attempt-000181.json';controller=statfile(path)
 if controller.get('exists') is not False:
  if controller['bytes']>20000:raise ValueError('Controller scalar-only metadata budget differs')
  data=json.loads(path.read_bytes());controller.update({key:data.get(key) for key in ['status','candidate_outer','attempt_id','start_epoch','end_epoch','wall_seconds','trainer_process_cpu_seconds','code_commit','config_sha256']})
 with (RUN/'stage.log').open('rb') as stream:stream.seek(max(0,(RUN/'stage.log').stat().st_size-4096));tail=stream.read(4096).decode('utf8','replace')
 tail=re.sub(r'(?i)(bearer\s+|(?:token|password|secret|api_key)\s*[=:]\s*)[^\s,;]+',r'\1[REDACTED]',tail)
 return dict(observed_utc=datetime.now(timezone.utc).isoformat(),epoch=time.time(),systemd=unit,main_proc_cgroup=membership,cgroup_path=str(ROOT),own_unique_process_count=len(pids),own_task_count=cgroup['pids.current'],own_task_limit=4096,own_subcgroups=groups,selected_pids=active,selected_processes=[procstat(pid,True) for pid in active],ps_columns='pid ppid pcpu etimes stat wchan comm-only(no argv/env)',ps_exit=ps.returncode,ps_stdout=ps.stdout,ps_stderr=ps.stderr,cgroup=cgroup,files=files,rank_file_stats=ranks,controller181=controller,stage_tail_utf8=tail)
a=snapshot();time.sleep(15);b=snapshot()
result=dict(schema='outer181_own_scope_observer_two_frames_v2',unit=UNIT,original_PID=MAIN,InvocationID=INVOCATION,run=str(RUN),resolved_run=EXPECTED,selection=dict(main_pid_proc_cgroup=main_membership,controller_pid=CONTROLLER,controller_owned=controller_owned,controller_proc_cgroup=controller_membership,top20_actual_own_cumulative_cpu=ranked,selected_pids=selected,initial_unique_own_process_count=len(initial_pids),initial_own_subcgroups=initial_groups),frames=[a,b],interval_seconds=b['epoch']-a['epoch'],clock_ticks_per_second=HZ,remote_files_written=False,process_attach_or_training_mutation=False,other_units_or_tasks_queried=False,own_task_thread_count_not_process_count=True,ps_scope='Selected own MainPID/controller and own top20 cumulative CPU only; no all-machine ps; comm-only without argv/env.',large_resource_and_rank_files_stat_only=True)
raw=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)
if len(raw.encode())>95000:raise ValueError('Small own diagnostic result budget exceeded')
print(raw)
