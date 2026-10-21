"""One read-only Slurm/allocation observation; never submits or starts a watcher.

Only bounded stdlib metadata is returned. Main PID I/O delta is evidence of
loading, not optimizer progress; the two are explicitly reported separately.
"""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,time,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3]
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-v1'
ASSET_ROOT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
JOB_NAME='q7b-mixed-direct169028';MODULE='reproduction.operations.qwen7b_warmup.joint_mixed_direct_pretrain'
PROC_SOURCE=r'''import json,os,pathlib,time
v=__VALUES__
out=dict(epoch=time.time(),host=os.uname().nodename,allocation_job=os.environ.get('SLURM_JOB_ID'),CPU=1,GPU=0)
if os.environ.get('SLURM_JOB_ID')!=v['job']:raise ValueError('Read-only step allocated to a different job')
pid=v.get('pid');out['controller_pid']=pid
if not isinstance(pid,int)or pid<=1:out.update(status='NO_CONTROLLER_PID_YET',process_exists=False)
else:
 root=pathlib.Path('/proc')/str(pid)
 try:
  def stat():
   x=(root/'stat').read_text().rsplit(')',1)[1].split();return dict(state=x[0],user_ticks=int(x[11]),system_ticks=int(x[12]),birth_ticks=int(x[19]))
  before=stat();args=(root/'cmdline').read_bytes().decode(errors='replace').split(chr(0));cg=(root/'cgroup').read_text();import re
  command_ok=('-m'in args and args[args.index('-m')+1]==v['module']and'--base'in args and args[args.index('--base')+1]==v['base'])
  cgroup_ok=bool(re.search(r'(?:^|[/_])job[_/]'+re.escape(v['job'])+r'(?:[/_.]|$)',cg,re.M))
  out.update(process_exists=True,command_identity_matches=command_ok,cgroup_job_matches=cgroup_ok,**before)
  if command_ok and cgroup_ok:
   io={}
   for line in(root/'io').read_text().splitlines():
    k,x=line.split(':',1);io[k]=int(x.strip())
   status=(root/'status').read_text().splitlines();out.update(io=io,status='OWNED_MAIN_PROCESS_OBSERVED',status_fields=[x for x in status if x.startswith(('VmRSS:','VmSize:','Threads:'))],cpu_clock_ticks_per_second=os.sysconf('SC_CLK_TCK'),cgroup_paths=cg.strip().splitlines())
   # Diagnostic only: never open/read these targets or inspect foreign roots.
   out.update(open_whitelisted_fds=[],fd_readlink_races=0,fd_diagnostic_errors=0,fd_scan_truncated=False)
   try:
    for number,fd in enumerate((root/'fd').iterdir()):
     if number>=256:out['fd_scan_truncated']=True;break
     try:link=os.readlink(str(fd))
     except FileNotFoundError:out['fd_readlink_races']+=1;continue
     except OSError:out['fd_diagnostic_errors']+=1;continue
     path=pathlib.PurePosixPath(link)
     if path.is_absolute()and'..'not in path.parts and str(path).startswith(v['asset_root'].rstrip('/')+'/')and path.suffix in('.bin','.jsonl','.safetensors'):
      if len(out['open_whitelisted_fds'])<16:out['open_whitelisted_fds'].append(dict(fd=fd.name,path=str(path)))
      else:out['fd_scan_truncated']=True
   except Exception as e:out['fd_diagnostic_error']=type(e).__name__+': '+str(e)[:160]
   after=stat();out['same_birth_through_observation']=after['birth_ticks']==before['birth_ticks']
   if not out['same_birth_through_observation']:out['status']='PROCESS_IDENTITY_RACED_NO_DELTA_ACCEPTED'
  else:out['status']='CONTROLLER_PID_NOT_VERIFIED_AS_OWN_MAIN'
 except FileNotFoundError:out.update(status='CONTROLLER_PROCESS_NOT_PRESENT',process_exists=False)
 except Exception as e:out.update(status='PROCESS_OBSERVATION_FAILED',error=type(e).__name__+': '+str(e)[:300])
print('MIXED_DIRECT_PROC_OBSERVATION='+json.dumps(out,sort_keys=True))
'''
REMOTE_SOURCE=r'''import hashlib,json,os,pathlib,pwd,re,subprocess,time
v=__VALUES__;b=pathlib.Path(v['base']);out=dict(epoch=time.time(),base=str(b),read_only=True,watcher_started=False,model_or_tensor_loaded=False)
def small(path,limit=98304):
 for attempt in range(2):
  try:
   with pathlib.Path(path).open('rb')as f:raw=f.read(limit+1)
   if len(raw)>limit:return {'metadata_error':'bounded metadata exceeds limit','bytes_read':len(raw)}
   return json.loads(raw)
  except FileNotFoundError:
   if attempt==0:time.sleep(.03)
  except Exception as e:return {'metadata_error':type(e).__name__+': '+str(e)[:200]}
 return None
def command(args,timeout=8):
 try:
  r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout);return dict(returncode=r.returncode,stdout=r.stdout[-12000:],stderr=r.stderr[-1200:])
 except subprocess.TimeoutExpired:return dict(returncode=None,timeout=True,command=args[0])
submitted=small(b/'submission-result.json');out['submission']=submitted
if not submitted or 'job_id'not in submitted:out['status']='NO_ACTUAL_SUBMISSION_RESULT_YET';print(json.dumps(out));raise SystemExit(0)
job=str(submitted['job_id'])
if not re.fullmatch(r'[0-9]+',job):raise ValueError('Exact numeric own submission ID required')
out['job_id']=job;control=command(['scontrol','show','job','-o',job]);out['scontrol_observation']=control
out['main_sacct']=command(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed'])
if control['returncode']!=0:out['status']='LIVE_SCONTROL_UNAVAILABLE_USE_TERMINAL_MAIN_SACCT';print(json.dumps(out));raise SystemExit(0)
fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/]*)=([^\s]*)',control['stdout']));current=pwd.getpwuid(os.getuid());expected_user='%s(%d)'%(current.pw_name,os.getuid())
expected=dict(JobId=job,JobName=v['job_name'],UserId=expected_user,Account='mou176',Command=str(b/'training.sbatch'),WorkDir=str(b));checks={k:fields.get(k)==x for k,x in expected.items()};out['job_identity']=dict(expected=expected,actual={k:fields.get(k)for k in expected},checks=checks,verified=all(checks.values()));out['slurm_state']=fields.get('JobState');out['reason']=fields.get('Reason');out['start_time']=fields.get('StartTime');out['nodes']=fields.get('NodeList')
if not all(checks.values()):out['status']='OWN_JOB_IDENTITY_MISMATCH_NO_PROCESS_STEP';print(json.dumps(out));raise SystemExit(0)
ref=submitted.get('config',{});config_path=b/'config.json'
try:
 assert config_path.stat().st_size<=98304,'Actual config must be bounded metadata';raw=config_path.read_bytes();out['config_binding_verified']=(ref.get('path')==str(config_path)and len(raw)==ref.get('bytes')and hashlib.sha256(raw).hexdigest()==ref.get('sha256'))
except FileNotFoundError:out['config_binding_verified']=False
if not out['config_binding_verified']:out['status']='CONFIG_IDENTITY_MISMATCH_NO_PROCESS_STEP';print(json.dumps(out));raise SystemExit(0)
for name in('controller.json','progress.json','evaluation-progress.json','update-intent.json','latest.json','first-update-replay.json','first-cp-independent-audit.json','durable-next-update-replay.json','failure.json','interrupted.json','complete.json','mixed-checkpoint-transaction.json'):
 x=small(b/name)
 if x is not None:out[name[:-5]]=x
log=b/'logs'/('train-'+job+'.log')
try:
 with log.open('rb')as stream:stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-4096));out['log']=dict(path=str(log),bytes=size,tail=stream.read(4096).decode(errors='replace'))
except FileNotFoundError:out['log']=dict(path=str(log),absent=True)
if fields.get('JobState')=='PENDING':out['status']='PENDING_NOT_TRAINING';out['queue_start_estimate']=command(['squeue','--start','-h','-j',job,'-o','%i|%S|%R']);print(json.dumps(out));raise SystemExit(0)
if fields.get('JobState')!='RUNNING':out['status']='TERMINAL_OR_TRANSITIONING_NO_PROCESS_STEP';print(json.dumps(out));raise SystemExit(0)
controller=out.get('controller',{});pid=controller.get('pid')
if str(controller.get('job_id'))!=job or controller.get('config_sha256')!=ref.get('sha256'):out['status']='RUNNING_CONTROLLER_NOT_YET_BOUND_NO_PROCESS_STEP';print(json.dumps(out));raise SystemExit(0)
proc=v['proc_source'].replace('__VALUES__',repr(dict(job=job,pid=pid,module=v['module'],base=v['base'],asset_root=v['asset_root'])))
step=command(['srun','--jobid='+job,'--overlap','--gres=gpu:0','--ntasks=1','--cpus-per-task=1','--mem=256M','env','CUDA_VISIBLE_DEVICES=','python3','-c',proc],timeout=18);out['allocation_step']=dict(returncode=step['returncode'],timeout=step.get('timeout',False),stderr=step.get('stderr',''))
if step['returncode']==0:
 markers=[line for line in step['stdout'].splitlines()if line.startswith('MIXED_DIRECT_PROC_OBSERVATION=')]
 if markers:out['process']=json.loads(markers[-1].split('=',1)[1])
 else:out['allocation_step']['missing_marker']=True
# All manifest-list work is small metadata on the login side; no tensor reads.
# Any readlink close-race or optional mapping failure remains diagnostic only.
if out.get('process',{}).get('status')=='OWNED_MAIN_PROCESS_OBSERVED':
 try:
  config=json.loads(raw);feature_ref=config['features'];feature_path=pathlib.Path(feature_ref['path'])
  assert feature_path.is_absolute()and str(feature_path).startswith(v['asset_root'].rstrip('/')+'/')and 0<feature_ref['bytes']<=4*1024**2
  with feature_path.open('rb')as stream:feature_raw=stream.read(4*1024**2+1)
  assert len(feature_raw)==feature_ref['bytes']and hashlib.sha256(feature_raw).hexdigest()==feature_ref['sha256'],'Feature descriptor list SHA/bytes differs'
  manifests=json.loads(feature_raw);assert isinstance(manifests,list)and len(manifests)==5356,'Expected frozen5356 manifest list'
  by_parent={}
  for ordinal,manifest in enumerate(manifests,1):by_parent.setdefault(str(pathlib.PurePosixPath(manifest['path']).parent),[]).append(ordinal)
  matches=[]
  for fd in out['process'].get('open_whitelisted_fds',[]):
   ordinals=by_parent.get(str(pathlib.PurePosixPath(fd['path']).parent),[])
   if ordinals:matches.append(dict(fd=fd['fd'],path=fd['path'],manifest_ordinals=ordinals))
  ordinals=sorted({i for match in matches for i in match['manifest_ordinals']})
  out['manifest_position']=dict(status='OPEN_FD_MANIFEST_PARENT_MATCH'if ordinals else'NO_OPEN_FD_FEATURE_MANIFEST_MATCH',manifest_total=len(manifests),current_manifest_ordinal=ordinals[0]if len(ordinals)==1 and not out['process'].get('fd_scan_truncated')else None,matched_manifest_ordinals=ordinals,matched_open_files=matches,features=feature_ref,completion_or_completed_byte_claim=False,ordinal_semantics='one-based frozen list position of currently open whitelisted file parent; not a completion count')
 except Exception as e:out['manifest_position']=dict(status='OPTIONAL_FD_MANIFEST_DIAGNOSTIC_UNAVAILABLE',current_manifest_ordinal=None,diagnostic_error=type(e).__name__+': '+str(e)[:220],training_failure_inferred=False)
progress=out.get('progress',{});cursor=progress.get('cursor');out['actual_accepted_updates_reported']=cursor if type(cursor)is int else None
out['healthy_training_qualification_complete']=(out.get('process',{}).get('status')=='OWNED_MAIN_PROCESS_OBSERVED'and out.get('process',{}).get('same_birth_through_observation')is True and type(cursor)is int and cursor>=6 and out.get('first-update-replay',{}).get('status')=='PASSED'and out.get('durable-next-update-replay',{}).get('status')=='PASSED'and out.get('first-cp-independent-audit',{}).get('status')=='DIRECT_MIXED_CP1_INDEPENDENT_CPU_FULL_RAW_EMA_TWO_ADAM_RNG_SAMPLER_SOURCE_PASSED')
out['status']='ACCEPTED_UPDATE_METADATA_PRESENT'if type(cursor)is int and cursor>0 else'LOADING_OR_INITIAL_EVALUATION_NO_ACCEPTED_UPDATE_YET'
print(json.dumps(out,sort_keys=True))
'''


def delta(current,previous):
 a=current.get('process',{});b=previous.get('process',{})
 same=bool(a.get('status')=='OWNED_MAIN_PROCESS_OBSERVED'and b.get('status')=='OWNED_MAIN_PROCESS_OBSERVED'and a.get('same_birth_through_observation')and b.get('same_birth_through_observation')and current.get('job_id')==previous.get('job_id')and a.get('host')==b.get('host')and a.get('controller_pid')==b.get('controller_pid')and a.get('birth_ticks')==b.get('birth_ticks'))
 out=dict(same_verified_process_identity=same,pid_change_is_observation_only=True)
 if same:
  seconds=a['epoch']-b['epoch'];out['seconds']=seconds;out['io_delta']={k:a['io'][k]-b['io'].get(k,0)for k in a.get('io',{})};out['CPU_user_ticks_delta']=a['user_ticks']-b['user_ticks'];out['CPU_system_ticks_delta']=a['system_ticks']-b['system_ticks'];out['main_read_MiB_per_second']=out['io_delta'].get('rchar',0)/(1024**2*seconds)if seconds>0 else None;out['I_O_growth_is_optimizer_progress_claim']=False
 for key in('progress','latest'):
  x=current.get(key,{}).get('cursor');y=previous.get(key,{}).get('cursor')
  if type(x)is int and type(y)is int:out[key+'_cursor_delta']=x-y
 return out


def remote_source():
 values=dict(base=BASE,job_name=JOB_NAME,module=MODULE,proc_source=PROC_SOURCE,asset_root=ASSET_ROOT)
 return REMOTE_SOURCE.replace('__VALUES__',repr(values),1)


def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,default=R.parents[1]);p.add_argument('--report-dir',type=Path,default=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-official-live-observations');p.add_argument('--previous',type=Path);a=p.parse_args();control=a.control_root.resolve();report=a.report_dir.resolve();report.mkdir(parents=True,exist_ok=True);last=a.previous or report/'latest.json';previous=json.loads(last.read_text())if last.exists()else None
 spec=importlib.util.spec_from_file_location('official_mixed_progress_private_control',control/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();source=remote_source();compile(source,'read-only-mixed-progress','exec');packed=base64.b64encode(zlib.compress(source.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
 if len(command.encode())>=16000:raise ValueError('Bounded metadata observation exceeds transport size')
 r=ctl.run(command,account='mou176',source_address=ip,timeout=55)
 if r.returncode:raise RuntimeError('Read-only observation failed; no retry/submit/process action: '+r.stderr.decode(errors='replace')[-1200:])
 lines=[line for line in r.stdout.decode().splitlines()if line.startswith('{')]
 if not lines:raise ValueError('Missing read-only metadata receipt')
 result=json.loads(lines[-1]);result['observer_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 if previous:result['since_previous']=delta(result,previous)
 raw=json.dumps(result,indent=2,allow_nan=False)+'\n';stamp='%d'%(time.time()*1000);path=report/('observation-'+stamp+'.json');path.write_text(raw);temp=report/'latest.json.tmp';temp.write_text(raw);temp.replace(report/'latest.json');print(json.dumps(dict(report=str(path),**result),allow_nan=False))


if __name__=='__main__':main()
