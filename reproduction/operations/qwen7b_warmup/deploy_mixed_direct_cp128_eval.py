"""Independent frozen official Mixed CP128 evaluation; root alone submits GPU.

Original 21 runtime files are loaded from their accepted exact Git origins.
The only additional runtime file is the separate snapshot evaluator.
"""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/';BRANCH='codex/mixed-direct-cp128-eval-20261022'
A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021';BASE=A+'/mixed-direct-official-cp128-eval-v1';MAIN=A+'/joint-mixed-direct-official-169028-v1';JOB='q7b-mixed-direct-cp128-eval'
MAIN_CONFIG=dict(path=MAIN+'/config.json',bytes=9615,sha256='2a7f039f4172f5284e3b8282e8ca6200e566bd92bb3013e9802306eca84e7393')
MAIN_PUBLICATION=dict(path=MAIN+'/source-publication.json',bytes=6458,sha256='cdae3137aec68c9521afe2bb11e60fb67fc5922b7db00bb076b0e2f787bfce80')
PUBLICATION_BACKUP='46a583ba73e75a9049f942c66b0b666fe72a7446';PUBLICATION_PATH='reproduction/reports/qwen7b-algorithm-improvement-20261021/joint-mixed-direct-official-deployment/actual-source-publication.json'
WORKER_ORIGIN='c0f7a82b6c28b2c8b5a353905c77b8b8eb1cb5ac';WORKER=P+'evaluate_mixed_direct_checkpoint.py';OUT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp128-eval'
COMMON='''import pathlib,json,hashlib,os,fcntl,subprocess,time,re,base64,io,gzip,tarfile,stat
b=pathlib.Path(v['base']);b.mkdir(parents=True,exist_ok=True);code=b/('eval-code-'+v['archive_sha256'][:12]);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);assert p.stat().st_size<4*1024**2,'Controller metadata/code only; never checkpoint payload';raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(ref):
 assert proof(ref['path'])==ref,'Declared small input SHA/bytes differ';return read(ref['path'])
def immutable(p,value):
 raw=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw,'Immutable snapshot control differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert r.returncode==0,r.stderr[-1000:];return r.stdout.strip()
def active():return command(['squeue','-h','-u','mou176','-n',v['job'],'-o','%i|%T|%R'])
def publication():
 p=read(b/'source-publication.json');assert p['archive_sha256']==v['archive_sha256']and p['code_root']==str(code)and p['file_origins']==v['manifest']['origins']and p['files']==v['manifest']['files']
 for name,item in p['files'].items():assert proof(code/name)==dict(path=str(code/name),**item)
 assert p['original21_runtime_files_unchanged']is True;return p
'''
STAGE='''raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha'];p=code/('upload-%04d.part'%v['part'])
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(part=v['part'],accepted=True)))
'''
EXTRACT='''raw=b''.join((code/('upload-%04d.part'%i)).read_bytes()for i in range(v['parts']));assert len(raw)==v['archive_bytes']and hashlib.sha256(raw).hexdigest()==v['archive_sha256'];members={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as tar:
 for member in tar:
  path=pathlib.PurePosixPath(member.name);assert member.isfile()and not path.is_absolute()and'..'not in path.parts and member.name in v['manifest']['files'];assert member.name not in members;members[member.name]=tar.extractfile(member).read()
assert set(members)==set(v['manifest']['files'])
for name,data in members.items():
 item=v['manifest']['files'][name];assert len(data)==item['bytes']and hashlib.sha256(data).hexdigest()==item['sha256'];p=code/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data);p.chmod(0o444)
mainpub=checked(v['main_publication']);assert mainpub['files']=={n:item for n,item in v['manifest']['files'].items()if n!=v['worker']}and mainpub['file_origins']=={n:origin for n,origin in v['manifest']['origins'].items()if n!=v['worker']}
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');script='import pathlib;root=pathlib.Path('+repr(str(code))+');names='+repr(sorted(members))+';[compile((root/n).read_bytes(),str(root/n),"exec")for n in names if n.endswith(".py")];print("SNAPSHOT_SOURCE_COMPILE_PASSED")';r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',script],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=35);assert r.returncode==0 and'SNAPSHOT_SOURCE_COMPILE_PASSED'in r.stdout,r.stderr[-1200:]
result=dict(archive_sha256=v['archive_sha256'],code_root=str(code),file_origins=v['manifest']['origins'],files=v['manifest']['files'],controller=v['controller'],original_main_publication=v['main_publication'],original21_runtime_files_unchanged=True,new_runtime_files=[v['worker']],no_training_worker_modified=True);immutable(b/'source-publication.json',result);publication();print(json.dumps(dict(status='INDEPENDENT_CP128_EVALUATION_EXACT_RUNTIME_STAGED',publication=proof(b/'source-publication.json'),files=len(members))))
'''
PREPARE='''publication();assert not active()and not(b/'submission-intent.json').exists(),'Existing job/unknown intent must be reconciled';guard=(b/'evaluation.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);guard.close()
pinpath=b/'snapshot-pin.json'
if not pinpath.exists():print(json.dumps(dict(status='WAITING_FOR_REAL_ACCEPTED_CP128_IMMUTABLE_PIN',GPU_submitted=False)));raise SystemExit(0)
pinref=proof(pinpath);pin=read(pinpath);assert pin['status']=='MAIN_CP128_FULL_ACCEPTED_STATE_IMMUTABLY_HARDLINKED_NO_PAYLOAD_READ'and pin['source_main_config']==v['main_config']and pin['checkpoint_payload_read_or_rehashed_by_pin']is False and pin['producer_full_CPU_reload_acceptance_inherited']is True
c=checked(pin['source_config']);maincfg=checked(v['main_config']);assert c==maincfg and c['base']==v['main']and c['initialization_source']=='official_initial'and c['code_hashes']=={n:x['sha256']for n,x in checked(v['main_publication'])['files'].items()}
cp=pin['checkpoint'];original=pin['source_checkpoint_original'];assert cp['cursor']==cp['native_steps']==cp['head_steps']==128 and cp['config_sha256']==v['main_config']['sha256']and cp['full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed']is True and cp['allocation_job_id']=='746083'and{k:x for k,x in cp.items()if k!='path'}=={k:x for k,x in original.items()if k!='path'}
p=pathlib.Path(cp['path']);s=p.stat();saved=pin['pinned_stat'];assert p.parent==b/'inputs'and not p.is_symlink()and stat.S_ISREG(s.st_mode)and s.st_size==cp['bytes']and s.st_uid==os.getuid()and(s.st_dev,s.st_ino,s.st_uid,s.st_size,s.st_mtime_ns)==tuple(saved[k]for k in('device','inode','uid','bytes','mtime_ns'))
config=dict(mode='read_only_checkpoint_snapshot',source_config=pin['source_config'],source_main_config=v['main_config'],checkpoint=cp,source_checkpoint_original=original,code_hashes={v['worker']:v['manifest']['files'][v['worker']]['sha256']},snapshot_pin=pinref,source_publication=proof(b/'source-publication.json'),checkpoint_payload_SHA_deferred_to_owned_GPU=True,source_config_base_unchanged=True,final4096_read=False,actor_updates=0,training_updates=0);immutable(b/'evaluation-config.json',config);(b/'logs').mkdir(exist_ok=True);text=v['entry'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code));entry=b/'evaluation.sbatch'
if entry.exists():assert entry.read_text()==text
else:entry.write_text(text)
subprocess.check_call(['bash','-n',str(entry)]);ready=dict(status='OFFICIAL_CP128_SNAPSHOT_EVALUATION_READY_NOT_SUBMITTED',evaluation_config=proof(b/'evaluation-config.json'),entry=proof(entry),publication=proof(b/'source-publication.json'),snapshot_pin=pinref,original_main_config=v['main_config'],checkpoint=cp,checkpoint_payload_read_or_rehashed_by_controller=False,GPU_submitted=False);immutable(b/'preparation.json',ready);print(json.dumps(ready))
'''
ENTRY='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-mixed-direct-cp128-eval
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=@@BASE@@/logs/evaluate-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.evaluate_mixed_direct_checkpoint --base @@BASE@@
'''
SUBMIT='''publication();control=(b/'submit-control.lock').open('a+');fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not active(),'Same-name actual job, never duplicate';assert not(b/'submission-intent.json').exists(),'Unknown/prior submit intent, inspect actual scheduler before any retry';ready=read(b/'preparation.json');assert ready['status']=='OFFICIAL_CP128_SNAPSHOT_EVALUATION_READY_NOT_SUBMITTED'
for key in('evaluation_config','entry','publication','snapshot_pin','original_main_config'):assert proof(ready[key]['path'])==ready[key]
lock=(b/'evaluation.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
for name in('controller.json','restore-complete.json','failure.json','complete.json'):assert not(b/name).exists(),'Existing evaluation attempt, preserve and inspect'
immutable(b/'submission-intent.json',dict(epoch=time.time(),ready=ready,controller=v['controller']));r=subprocess.run(['sbatch','--hold','--parsable','--account=mou176',str(b/'evaluation.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
if r.returncode or not re.fullmatch(r'[0-9]+(?:;[^ ]+)?',r.stdout.strip()):raise RuntimeError('Submission unknown/rejected; intent preserved, query same name before retry: '+r.stdout[-300:]+r.stderr[-1000:])
job=r.stdout.strip().split(';')[0];identity=command(['scontrol','show','job','-o',job])
for item in('JobName='+v['job'],'Account=mou176','QOS=normal','Partition=acd_u','TimeLimit=01:00:00','NumCPUs=8','MinMemoryNode=128G','Dependency=(null)','Command='+str(b/'evaluation.sbatch'),'WorkDir='+str(b)):assert item in identity,item
assert re.search(r'(?:^|[ ,])gres/gpu=1(?:[ ,]|$)',identity)or re.search(r'TresPerNode=gres/gpu:1(?:[ ,]|$)',identity),'Exactly1 GPU required';immutable(b/'submission-identity.json',dict(job_id=job,job=identity,ready=ready,epoch=time.time()));command(['scontrol','release',job]);result=dict(status='READONLY_CP128_EVALUATION_JOB_RELEASED_NOT_PROGRESS_VERIFIED',job_id=job,identity=proof(b/'submission-identity.json'),evaluation_config=ready['evaluation_config']);immutable(b/'submission-result.json',result);print(json.dumps(result))
'''
STATUS='''out=dict(epoch=time.time(),active_same_name_jobs=active())
for name in('snapshot-pin.json','preparation.json','submission-intent.json','submission-result.json','controller.json','restore-complete.json','evaluation-progress.json','failure.json','complete.json'):
 try:out[name[:-5]]=read(b/name)
 except FileNotFoundError:pass
if(b/'submission-result.json').exists():out['main_sacct']=command(['sacct','-X','-n','-P','-j',read(b/'submission-result.json')['job_id'],'--format=JobIDRaw,State,ExitCode,Elapsed'])
print(json.dumps(out))
'''
def git(commit,name):return subprocess.check_output(['git','show',commit+':'+name],cwd=R)
def bundle():
 if len(WORKER_ORIGIN)!=40:raise ValueError('Final published and qualified event worker required')
 rawpub=git(PUBLICATION_BACKUP,PUBLICATION_PATH);assert len(rawpub)==MAIN_PUBLICATION['bytes']and hashlib.sha256(rawpub).hexdigest()==MAIN_PUBLICATION['sha256'];pub=json.loads(rawpub);assert len(pub['files'])==21
 contents={name:git(pub['file_origins'][name],name)for name in pub['files']}
 for name,raw in contents.items():assert dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==pub['files'][name]
 contents[WORKER]=git(WORKER_ORIGIN,WORKER);origins=dict(pub['file_origins'],**{WORKER:WORKER_ORIGIN});manifest=dict(origins=origins,files={n:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())for n,raw in contents.items()});buffer=io.BytesIO()
 with gzip.GzipFile(fileobj=buffer,mode='wb',mtime=0,filename='')as gz:
  with tarfile.open(fileobj=gz,mode='w')as tar:
   for name,raw in sorted(contents.items()):item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o444;item.mtime=0;tar.addfile(item,io.BytesIO(raw))
 return buffer.getvalue(),manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('check','stage','prepare','submit','status'),required=True);p.add_argument('--submit-authorized',action='store_true');a=p.parse_args()
 if a.action=='submit'and not a.submit_authorized:p.error('Root actual-submit authorization required')
 raw,manifest=bundle();own=Path(__file__);rel=str(own.relative_to(R));commit=subprocess.check_output(['git','log','-1','--format=%H','refs/heads/'+BRANCH,'--',rel],cwd=R).decode().strip();assert own.read_bytes()==git(commit,rel),'Controller must equal its published source';remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R).decode().split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',commit,remote],cwd=R);archive_sha=hashlib.sha256(raw).hexdigest()
 if a.action=='check':print(json.dumps(dict(status='EXACT21_PLUS_NEW_SNAPSHOT_EVALUATOR_SOURCE_CHECKED',files=len(manifest['files']),archive_bytes=len(raw),archive_sha256=archive_sha,manifest=manifest)));return
 spec=importlib.util.spec_from_file_location('private_snapshot_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(body,**kwargs):
  value=dict(base=BASE,main=MAIN,job=JOB,archive_sha256=archive_sha,manifest=manifest,worker=WORKER,main_publication=MAIN_PUBLICATION,main_config=MAIN_CONFIG,controller=dict(commit=commit,path=rel,sha256=hashlib.sha256(own.read_bytes()).hexdigest()));value.update(kwargs);source='v='+repr(value)+'\n'+COMMON+body;compile(source,'snapshot_eval_controller','exec');packed=base64.b64encode(zlib.compress(source.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;result=ctl.run(command,account='mou176',source_address=ip,timeout=55)
  if result.returncode:raise RuntimeError(result.stdout[-4000:]+result.stderr[-2000:])
  return json.loads(result.stdout.splitlines()[-1])
 if a.action=='stage':
  prior=call("print(json.dumps(dict(exists=(b/'source-publication.json').exists())))")
  if prior['exists']:result=call("publication();print(json.dumps(dict(status='EXISTING_EXACT_SNAPSHOT_RUNTIME_REVERIFIED',publication=proof(b/'source-publication.json'))))")
  else:
   chunks=[raw[i:i+4096]for i in range(0,len(raw),4096)]
   for i,data in enumerate(chunks):call(STAGE,part=i,data=base64.b64encode(data).decode(),sha=hashlib.sha256(data).hexdigest())
   result=call(EXTRACT,parts=len(chunks),archive_bytes=len(raw))
 else:result=call(dict(prepare=PREPARE,submit=SUBMIT,status=STATUS)[a.action],**(dict(entry=ENTRY)if a.action=='prepare'else{}))
 OUT.mkdir(parents=True,exist_ok=True);(OUT/('actual-eval-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
for script in(COMMON,STAGE,EXTRACT,PREPARE,SUBMIT,STATUS):compile(script,'snapshot_eval_static','exec')
if __name__=='__main__':main()
