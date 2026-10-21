"""Immutable fast continuation controller. Root alone may retire/submit.

Stage and prepare cannot signal a process or submit GPU. Retire only sends USR1
inside the verified own old allocation; successful terminal plus exact highest
accepted state is mandatory before the new single job may start.
"""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/';BRANCH='codex/mixed-direct-fast-continuation-20261022';A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021';MAIN=A+'/joint-mixed-direct-official-169028-v1';BASE=A+'/joint-mixed-direct-official-169028-fast-v1';SOURCE_JOB='746083';JOB='q7b-mixed-direct169028-fast'
WORKER=P+'joint_mixed_direct_fast_continuation.py';HELPER=P+'mixed_direct_fast_handoff.py';FAST_READER=P+'replay_feature_reader_fast.py';FAST_UPDATE=P+'direct_update_fast.py'
WORKER_ORIGIN='WAITING_FOR_QUALIFIED_EVENT_WORKER';READER_ORIGIN='c2fdc9598ad55fea11bc5a5d2298ba4560807c24';UPDATE_ORIGIN='WAITING_FOR_MICRO32_UPDATER';HELPER_ORIGIN='94bb18ef17cae80f8500e43fd666d747132be6eb'
SOURCE_CONFIG=dict(path=MAIN+'/config.json',bytes=9615,sha256='2a7f039f4172f5284e3b8282e8ca6200e566bd92bb3013e9802306eca84e7393');SOURCE_PUBLICATION=dict(path=MAIN+'/source-publication.json',bytes=6458,sha256='cdae3137aec68c9521afe2bb11e60fb67fc5922b7db00bb076b0e2f787bfce80');PUB_BACKUP='46a583ba73e75a9049f942c66b0b666fe72a7446';PUB_PATH='reproduction/reports/qwen7b-algorithm-improvement-20261021/joint-mixed-direct-official-deployment/actual-source-publication.json';OUT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-fast-continuation'
COMMON='''import pathlib,json,hashlib,os,fcntl,subprocess,time,re,base64,io,gzip,tarfile,stat,sys
b=pathlib.Path(v['base']);old=pathlib.Path(v['main']);b.mkdir(parents=True,exist_ok=True);code=b/('code-'+v['archive_sha256'][:12]);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);assert p.stat().st_size<4*1024**2 and p.suffix!='.pt','Metadata/code only';raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(ref):assert proof(ref['path'])==ref;return read(ref['path'])
def immutable(p,value):
 raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw,'Immutable continuation control differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def atomic(p,value):
 t=p.with_name(p.name+'.partial');t.write_text(json.dumps(value,sort_keys=True)+chr(10));os.replace(str(t),str(p))
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert r.returncode==0,r.stderr[-1000:];return r.stdout.strip()
def active(name):return command(['squeue','-h','-u','mou176','-n',name,'-o','%i|%T|%R'])
def publication():
 pub=read(b/'source-publication.json');assert pub['archive_sha256']==v['archive_sha256']and pub['code_root']==str(code)and pub['files']==v['manifest']['files']and pub['file_origins']==v['manifest']['origins']
 for name,item in pub['files'].items():assert proof(code/name)==dict(path=str(code/name),**item)
 return pub
def handoff_module():
 import importlib.util
 p=code/v['helper'];spec=importlib.util.spec_from_file_location('immutable_fast_handoff',str(p));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def finalize():
 pub=publication();h=handoff_module();result=h.pin_terminal_source(b,old)
 if result['status']!='OFFICIAL_DIRECT_MIXED_HIGHEST_ACCEPTED_SOURCE_GRACEFULLY_RETIRED_AND_PINNED':return result
 c=json.loads((old/'config.json').read_bytes());assert proof(old/'config.json')==v['source_config'];original=json.loads(json.dumps(c));c['base']=str(b);c['mode']='fast_continuation';c['architecture']['micro_questions']=32;c['code_hashes']={n:x['sha256']for n,x in pub['files'].items()};c['reader_integrity_mode']='inherited_full_seal_with_fresh_metadata_and_each_row_SHA';c['prefetch_batches']=8;c['parallel_workers']=8;c['continuation']=dict(source_config=result['source_config'],source_main_config=result['source_main_config'],source_checkpoint=result['pinned_checkpoint'],source_checkpoint_original=result['checkpoint'],source_handoff_receipt=proof(b/'source-handoff.json'))
 # Explicitly prove every unlisted original leaf is unchanged.
 compare=json.loads(json.dumps(c));compare['base']=original['base'];compare['mode']=original['mode'];compare['architecture']['micro_questions']=original['architecture']['micro_questions'];compare['code_hashes']=original['code_hashes']
 for key in('reader_integrity_mode','prefetch_batches','parallel_workers','continuation'):compare.pop(key)
 assert compare==original,'Only transport batching and explicit provenance may change'
 immutable(b/'config.json',c);(b/'logs').mkdir(exist_ok=True);text=v['entry'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code));p=b/'training.sbatch'
 if p.exists():assert p.read_text()==text
 else:p.write_text(text)
 subprocess.check_call(['bash','-n',str(p)]);ready=dict(status='OFFICIAL_MIXED_FAST_EXACT_CONTINUATION_READY_NOT_SUBMITTED',config=proof(b/'config.json'),entry=proof(p),publication=proof(b/'source-publication.json'),source_handoff=proof(b/'source-handoff.json'),source_checkpoint=result['pinned_checkpoint'],source_checkpoint_original=result['checkpoint'],source_cursor=result['cursor'],remaining_updates=21132-result['cursor'],source_job_terminal=result['source_main_terminal'],all_source_nine_hashes_unchanged=True,checkpoint_payload_read_on_CPU=False,actual_GPU_restore_and_next_update_replay_required=True,normal_one_H100_48h=True,GPU_submitted=False);immutable(b/'preparation.json',ready);return ready
'''
STAGE='''raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha'];p=code/('upload-%04d.part'%v['part'])
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(part=v['part'],accepted=True)))
'''
EXTRACT='''raw=b''.join((code/('upload-%04d.part'%i)).read_bytes()for i in range(v['parts']));assert len(raw)==v['archive_bytes']and hashlib.sha256(raw).hexdigest()==v['archive_sha256'];members={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as tar:
 for member in tar:
  p=pathlib.PurePosixPath(member.name);assert member.isfile()and not p.is_absolute()and'..'not in p.parts and member.name in v['manifest']['files']and member.name not in members;members[member.name]=tar.extractfile(member).read()
assert set(members)==set(v['manifest']['files'])
for name,data in members.items():
 item=v['manifest']['files'][name];assert len(data)==item['bytes']and hashlib.sha256(data).hexdigest()==item['sha256'];p=code/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data);p.chmod(0o444)
oldpub=checked(v['source_publication']);assert len(oldpub['files'])==21
for name,item in oldpub['files'].items():assert v['manifest']['files'][name]==item and v['manifest']['origins'][name]==oldpub['file_origins'][name]
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');script='import pathlib;root=pathlib.Path('+repr(str(code))+');names='+repr(sorted(members))+';[compile((root/n).read_bytes(),str(root/n),"exec")for n in names if n.endswith(".py")];print("FAST_SOURCE_COMPILE_PASSED")';r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',script],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=35);assert r.returncode==0 and'FAST_SOURCE_COMPILE_PASSED'in r.stdout,r.stderr[-1000:];result=dict(archive_sha256=v['archive_sha256'],code_root=str(code),files=v['manifest']['files'],file_origins=v['manifest']['origins'],controller=v['controller'],original_source_publication=v['source_publication'],original21_sources_unchanged=True,new_runtime_files=[v['worker'],v['fast_reader'],v['fast_update']],new_control_helper=v['helper']);immutable(b/'source-publication.json',result);publication();print(json.dumps(dict(status='IMMUTABLE_FAST_CONTINUATION_SOURCE_STAGED_NO_HANDOFF',publication=proof(b/'source-publication.json'),files=len(members),old_writer_signaled=False,GPU_submitted=False)))
'''
PREPARE='''publication();h=handoff_module();c,p=h.source_metadata(old);assert not active(v['job'])and not(b/'submission-intent.json').exists(),'Existing new identity/intent requires reconciliation';p=read(old/'progress.json');assert p['cursor']==p['native_steps']==p['head_steps']and p['cursor']>=128,'Actual old native/head updates required'
if(b/'source-handoff.json').exists():print(json.dumps(finalize()))
else:print(json.dumps(dict(status='FAST_SOURCE_READY_WAITING_FOR_ROOT_AUTHORIZED_EXACT_HANDOFF',old_job_id=v['source_job'],old_active=active('q7b-mixed-direct169028'),actual_old_cursor=p['cursor'],source_publication=proof(b/'source-publication.json'),source_config=v['source_config'],old_writer_signaled=False,GPU_submitted=False)))
'''
SIGNAL='''import os,json,pathlib,signal,hashlib,socket
main=pathlib.Path(V['main']);p=pathlib.Path('/proc')/str(V['pid']);assert os.environ.get('SLURM_JOB_ID')==V['job']and p.stat().st_uid==os.getuid();raw=(p/'cmdline').read_bytes();argv=[x.decode()for x in raw.split(bytes([0]))if x];env=(p/'environ').read_bytes().split(bytes([0]));assert('SLURM_JOB_ID='+V['job']).encode()in env and V['module']in argv and '--base'in argv and argv[argv.index('--base')+1]==str(main);birth=(p/'stat').read_text().rsplit(')',1)[1].split()[19];record=dict(pid=V['pid'],birth=birth,uid=p.stat().st_uid,node=socket.gethostname(),cmdline_sha256=hashlib.sha256(raw).hexdigest(),job_id=V['job'],module_and_base_exact=True,signal='SIGUSR1',checkpoint_payload_read=False);assert(p/'stat').read_text().rsplit(')',1)[1].split()[19]==birth;os.kill(V['pid'],signal.SIGUSR1);print(json.dumps(record))
'''
RETIRE='''assert v['handoff_authorized']is True;publication();h=handoff_module();h.source_metadata(old);assert not active(v['job'])and not(b/'submission-intent.json').exists();control=(b/'retire-control.lock').open('a+');fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(b/'source-handoff.json').exists():print(json.dumps(finalize()));raise SystemExit(0)
intent=b/'retirement-intent.json'
if not intent.exists():
 profile=checked(v['profile']);assert v['profile']['path'].startswith('/data/')and profile['status']=='INDEPENDENT_CP128_SAME_RANDOM_BATCH_FULLSTEP_PROFILE_COMPLETE'and profile['source_config']['sha256']==v['source_config']['sha256']and profile['questions']==32 and profile['rows']==320 and profile['formal_training_update_count']==0 and profile['main_state_or_pinned_checkpoint_written']is False and profile['external_or_final_outcomes_read']is False and profile['final_state_restored_to_CP128']is True,'Actual completed same-batch full-step profile required';identity=command(['scontrol','show','job','-o',v['source_job']])
 for item in('JobId='+v['source_job'],'JobState=RUNNING','JobName=q7b-mixed-direct169028','UserId=mou176','Account=mou176','QOS=normal','Partition=acd_u','Command='+str(old/'training.sbatch'),'WorkDir='+str(old)):assert item in identity,item
 owner=read(old/'controller.json');p=read(old/'progress.json');assert owner['job_id']==v['source_job']and p['cursor']==p['native_steps']==p['head_steps']>=128 and not(old/'failure.json').exists();guard=(old/'training.lock').open('a+')
 try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise AssertionError('Old RUNNING writer training lock unexpectedly free')
 guard.close();immutable(intent,dict(controller=v['controller'],old_job=v['source_job'],old_identity=identity,old_controller=proof(old/'controller.json'),old_observed_cursor=p['cursor'],requested_signal='SIGUSR1',profile=v['profile'],epoch=time.time()));signal_source='V='+repr(dict(main=str(old),job=v['source_job'],pid=owner['pid'],module='reproduction.operations.qwen7b_warmup.joint_mixed_direct_pretrain'))+chr(10)+v['signal_body'];encoded=base64.b64encode(signal_source.encode()).decode();inner='import base64;exec(base64.b64decode('+repr(encoded)+'))';r=subprocess.run(['srun','--overlap','--jobid='+v['source_job'],'--gres=gpu:0','--cpus-per-task=1','--mem=128M','--time=00:01:00','python3','-c',inner],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40)
 if r.returncode:raise RuntimeError('Signal result uncertain: preserve intent and inspect old PID/state; never blind resend: '+r.stderr[-1200:])
 signal_receipt=json.loads(r.stdout.strip().splitlines()[-1]);assert signal_receipt['pid']==owner['pid']and signal_receipt['module_and_base_exact']is True;immutable(b/'retirement-signal.json',signal_receipt)
else:assert(b/'retirement-signal.json').exists(),'Prior unknown signal intent requires explicit reconciliation; do not resend'
# Launch exactly one bounded metadata-only terminal observer. It never submits.
watchlock=(b/'retire-watch.lock').open('a+')
try:fcntl.flock(watchlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:print(json.dumps(dict(status='GRACEFUL_RETIREMENT_ALREADY_IN_PROGRESS',signal=proof(b/'retirement-signal.json'))));raise SystemExit(0)
watchlock.close();path=b/('retire-observer-'+v['controller']['commit'][:12]+'.py');text=v['watch_source']
if path.exists():assert path.read_text()==text
else:path.write_text(text);path.chmod(0o444)
log=(b/'retire-observer.log').open('ab');child=subprocess.Popen(['python3','-u',str(path)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps(dict(status='OLD_OWN_PYTHON_SIGUSR1_SENT_BOUNDED_EXACT_RETIREMENT_OBSERVER_STARTED',pid=child.pid,signal=proof(b/'retirement-signal.json'),observer=proof(path),GPU_submitted=False)))
'''
WATCH='''with(b/'retire-watch.lock').open('a+')as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for attempt in range(121):
  try:
   result=finalize();atomic(b/'retirement-status.json',dict(attempt=attempt,epoch=time.time(),result=result));print(json.dumps(result),flush=True)
   if result['status']=='OFFICIAL_MIXED_FAST_EXACT_CONTINUATION_READY_NOT_SUBMITTED':break
  except Exception as e:atomic(b/'retirement-failure.json',dict(type=type(e).__name__,message=str(e),epoch=time.time()));raise
  if attempt==120:raise RuntimeError('Finite retirement observer expired; preserve source/new namespace')
  time.sleep(10)
'''
ENTRY='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-mixed-direct169028-fast
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --signal=USR1@900
#SBATCH --output=@@BASE@@/logs/train-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.joint_mixed_direct_fast_continuation --base @@BASE@@
'''
SUBMIT='''publication();assert v['submit_authorized']is True;guard=(b/'submit-control.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not active(v['job'])and not active('q7b-mixed-direct169028'),'Any active old/new writer blocks submit';assert not(b/'submission-intent.json').exists(),'Unknown/existing submit intent: query same name before retry';ready=finalize();assert ready['status']=='OFFICIAL_MIXED_FAST_EXACT_CONTINUATION_READY_NOT_SUBMITTED'
for key in('config','entry','publication','source_handoff'):assert proof(ready[key]['path'])==ready[key]
for root in(old,b):
 lock=(root/'training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
for name in('latest.json','progress.json','update-intent.json','failure.json','complete.json'):assert not(b/name).exists(),'Existing target state needs exact recovery, never fresh submission'
immutable(b/'submission-intent.json',dict(ready=ready,controller=v['controller'],epoch=time.time()));r=subprocess.run(['sbatch','--hold','--parsable','--account=mou176',str(b/'training.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
if r.returncode or not re.fullmatch(r'[0-9]+(?:;[^ ]+)?',r.stdout.strip()):raise RuntimeError('Unknown/rejected submit; preserve intent and query actual identity: '+r.stdout[-300:]+r.stderr[-1000:])
job=r.stdout.strip().split(';')[0];identity=command(['scontrol','show','job','-o',job])
for item in('JobName='+v['job'],'Account=mou176','QOS=normal','Partition=acd_u','TimeLimit=2-00:00:00','NumCPUs=8','MinMemoryNode=128G','Dependency=(null)','Command='+str(b/'training.sbatch'),'WorkDir='+str(b)):assert item in identity,item
assert re.search(r'(?:^|[ ,])gres/gpu=1(?:[ ,]|$)',identity)or re.search(r'TresPerNode=gres/gpu:1(?:[ ,]|$)',identity);immutable(b/'submission-identity.json',dict(job_id=job,job=identity,ready=ready,epoch=time.time()));command(['scontrol','release',job]);result=dict(status='EXACT_NINE_STATE_FAST_CONTINUATION_JOB_RELEASED_NOT_PROGRESS_VERIFIED',job_id=job,identity=proof(b/'submission-identity.json'),config=ready['config'],source_cursor=ready['source_cursor']);immutable(b/'submission-result.json',result);print(json.dumps(result))
'''
STATUS='''out=dict(epoch=time.time(),old_active=active('q7b-mixed-direct169028'),new_active=active(v['job']))
for name in('preparation.json','retirement-intent.json','retirement-signal.json','retirement-status.json','retirement-failure.json','source-handoff.json','submission-result.json','controller.json','progress.json','latest.json','failure.json','complete.json'):
 try:out[name[:-5]]=read(b/name)
 except FileNotFoundError:pass
print(json.dumps(out))
'''
def git(commit,name):return subprocess.check_output(['git','show',commit+':'+name],cwd=R)
def bundle():
 origins={WORKER:WORKER_ORIGIN,FAST_READER:READER_ORIGIN,FAST_UPDATE:UPDATE_ORIGIN,HELPER:HELPER_ORIGIN}
 if any(len(origin)!=40 for origin in origins.values()):raise ValueError('Final published source origins required before staging')
 data=git(PUB_BACKUP,PUB_PATH);assert len(data)==SOURCE_PUBLICATION['bytes']and hashlib.sha256(data).hexdigest()==SOURCE_PUBLICATION['sha256'];pub=json.loads(data);contents={n:git(pub['file_origins'][n],n)for n in pub['files']}
 for n,raw in contents.items():assert dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==pub['files'][n]
 for n,origin in origins.items():contents[n]=git(origin,n)
 origins=dict(pub['file_origins'],**origins);manifest=dict(origins=origins,files={n:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())for n,raw in contents.items()});buf=io.BytesIO()
 with gzip.GzipFile(fileobj=buf,mode='wb',mtime=0,filename='')as gz:
  with tarfile.open(fileobj=gz,mode='w')as tar:
   for n,raw in sorted(contents.items()):item=tarfile.TarInfo(n);item.size=len(raw);item.mode=0o444;item.mtime=0;tar.addfile(item,io.BytesIO(raw))
 return buf.getvalue(),manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('check','stage','prepare','retire','submit','status'),required=True);p.add_argument('--handoff-authorized',action='store_true');p.add_argument('--submit-authorized',action='store_true');p.add_argument('--profile-receipt',type=Path);a=p.parse_args()
 if a.action=='retire'and not a.handoff_authorized:p.error('Root explicit retirement authorization required')
 if a.action=='submit'and not a.submit_authorized:p.error('Root explicit single-submit authorization required')
 if a.action=='retire'and a.profile_receipt is None:p.error('Accepted actual full-step profile descriptor JSON required')
 raw,manifest=bundle();own=Path(__file__);rel=str(own.relative_to(R));commit=subprocess.check_output(['git','log','-1','--format=%H','refs/heads/'+BRANCH,'--',rel],cwd=R).decode().strip();assert own.read_bytes()==git(commit,rel);remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R).decode().split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',commit,remote],cwd=R);value=dict(base=BASE,main=MAIN,job=JOB,source_job=SOURCE_JOB,source_config=SOURCE_CONFIG,source_publication=SOURCE_PUBLICATION,manifest=manifest,archive_sha256=hashlib.sha256(raw).hexdigest(),worker=WORKER,fast_reader=FAST_READER,fast_update=FAST_UPDATE,helper=HELPER,controller=dict(commit=commit,path=rel,sha256=hashlib.sha256(own.read_bytes()).hexdigest()))
 if a.action=='check':print(json.dumps(dict(status='EXACT_ORIGIN_FAST_CONTINUATION_STATIC_CHECK_PASSED',files=len(manifest['files']),archive_bytes=len(raw),archive_sha256=value['archive_sha256'],manifest=manifest)));return
 spec=importlib.util.spec_from_file_location('private_fast_continuation_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(body,**kwargs):
  v=dict(value,**kwargs);source='v='+repr(v)+'\n'+COMMON+body;compile(source,'fast_continuation_controller','exec');packed=base64.b64encode(zlib.compress(source.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;result=ctl.run(command,account='mou176',source_address=ip,timeout=55)
  if result.returncode:raise RuntimeError(result.stdout[-3000:]+result.stderr[-2000:])
  return json.loads(result.stdout.splitlines()[-1])
 if a.action=='stage':
  exists=call("print(json.dumps(dict(exists=(b/'source-publication.json').exists())))")
  if exists['exists']:result=call("publication();print(json.dumps(dict(status='EXISTING_IMMUTABLE_FAST_SOURCE_REVERIFIED',publication=proof(b/'source-publication.json'))))")
  else:
   chunks=[raw[i:i+4096]for i in range(0,len(raw),4096)]
   for i,data in enumerate(chunks):call(STAGE,part=i,data=base64.b64encode(data).decode(),sha=hashlib.sha256(data).hexdigest())
   result=call(EXTRACT,parts=len(chunks),archive_bytes=len(raw))
 elif a.action=='retire':
  profile=json.loads(a.profile_receipt.read_bytes());assert set(profile)=={'path','bytes','sha256'};watch_value=dict(value,entry=ENTRY);watch_source='v='+repr(watch_value)+'\n'+COMMON+WATCH;result=call(RETIRE,handoff_authorized=True,profile=profile,signal_body=SIGNAL,watch_source=watch_source,entry=ENTRY)
 else:result=call(dict(prepare=PREPARE,submit=SUBMIT,status=STATUS)[a.action],entry=ENTRY,submit_authorized=a.submit_authorized)
 OUT.mkdir(parents=True,exist_ok=True);(OUT/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
for script in(COMMON,STAGE,EXTRACT,PREPARE,SIGNAL,RETIRE,WATCH,SUBMIT,STATUS):compile(script,'fast_continuation_static','exec')
if __name__=='__main__':main()
