"""Code-only stage and guarded finite actual native-GPU qualification."""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile,zlib
from pathlib import Path
from .deploy_confirmation_materialization import import_closure
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];PKG='reproduction/operations/qwen7b_warmup/';A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021';BASE=A+'/joint-balanced-native-E4-qualification-v2';OUT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/joint-balanced-native'
SEEDS=(PKG+'joint_balanced_native_train.py',PKG+'joint_balanced_native_update.py',PKG+'runtime.sh');SAMPLER='c98ddcd7861b7023f3efa0d82d3c1fbea0664a13'
COMMON='''import pathlib,json,hashlib,base64,subprocess,fcntl,os,time,re,tarfile,io
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);lock=(b/'deployment.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def immutable(p,x):
 if p.exists():assert read(p)==x,'Immutable metadata differs'
 else:
  with p.open('x')as f:f.write(json.dumps(x,sort_keys=True)+'\\n');f.flush();os.fsync(f.fileno())
def checked(ref):
 assert proof(ref['path'])=={k:ref[k]for k in ('path','bytes','sha256')},'Metadata SHA differs';return read(ref['path'])
def active():
 r=subprocess.run(['squeue','-h','-u','mou176','-n','q7b-native-joint-qual','-o','%i|%j|%T|%R'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=12);assert r.returncode==0,r.stderr[-500:];return r.stdout.strip().splitlines()
'''
EXTRACT='''raw=b''.join((code/('upload-%04d.part'%i)).read_bytes()for i in range(v['parts']));assert len(raw)==v['bytes']and hashlib.sha256(raw).hexdigest()==v['sha256'];files={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as archive:
 for item in archive:
  name=item.name;p=pathlib.PurePosixPath(name);assert item.isfile()and not p.is_absolute()and'..'not in p.parts;data=archive.extractfile(item).read();dst=code/name;dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():assert dst.read_bytes()==data
  else:dst.write_bytes(data);dst.chmod(0o444)
  files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
immutable(b/'source-publication.json',dict(commit=v['commit'],code_root=str(code),files=files,sampler_commit=v['sampler'],file_origins={n:(v['sampler']if n.endswith('balanced_hard_g10_sampler.py')else v['commit'])for n in files},controller=v['controller']));print(json.dumps(dict(status='NATIVE_JOINT_CODE_STAGED',publication=proof(b/'source-publication.json'),files=len(files))))
'''
PREPARE='''assert not active()and not(b/'submission-intent.json').exists(),'Existing job/intent must be reconciled';old=read(pathlib.Path(v['A'])/'hard-vs-random-E4-477-v1/targeted/config.json');pub=read(b/'source-publication.json');assert pub['commit']==v['commit'];groups_ref=dict(path=v['A']+'/peer-threebank58432-internal5120-plan-v1/merged-bank/groups.json',bytes=43507411,sha256='ff78875397f3af73c4908be9bf330dda125427a7921988fe1ee8042f1d563216');features=dict(path=v['A']+'/peer-threebank58432-internal5120-plan-v1/features-layer26.json',bytes=540578,sha256='e1d17ea44fd56cad4e40ae571b2eb0625389b582b70a6056e4176444fd9cf560');groups=checked(groups_ref)['train'];assert len(groups)==58432;checked(features)
source_bank=checked(old['source_bank']);diag=source_bank['diagnostic'];val=read(pathlib.Path(v['A'])/'validation-g10-20261021/materialized/bank/groups.json')['validation'];final=read(pathlib.Path(v['A'])/'training-extraG5-matched6528-20261021/final-exclusion-plan.json');finalrows=final.get('questions',final.get('groups'));assert finalrows and len(finalrows)==4096;identities={r['qid']:dict(qid=r['qid'],question_hash=r['question_hash'])for r in diag+val+finalrows};assert len(diag)==512 and len(val)==2048 and len(identities)==6656;blocked=set(identities);hashes={r['question_hash']for r in identities.values()};assert all(r['qid']not in blocked and r['question_hash']not in hashes for r in groups)
exclusion=b/'evaluation-identity-exclusions.json';immutable(exclusion,dict(role='joint_training_fixed_evaluation_identity_exclusions',groups=sorted(identities.values(),key=lambda r:r['qid']),scope_counts=dict(diagnostic=512,selection=2048,final=4096),final_labels_or_rollouts_read=False));source=old['source_checkpoint'];assert source['sha256']=='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8';assert pathlib.Path(source['path']).stat().st_size==source['bytes'];checked(old['source_receipt']);checked(old['source_config']);checked(old['model_config']);c=dict(base=str(b),mode='qualification',groups=groups_ref,features=features,exclusions=proof(exclusion),expected_exclusion_questions=6656,fit_questions=58432,steps_per_epoch=1826,update_budget=7,save_every=1,source_checkpoint=source,source_receipt=old['source_receipt'],source_config=old['source_config'],source_bank=old['source_bank'],model_dir=old['model_dir'],model_config=old['model_config'],lr=5e-6,ema_decay=.99,loss_kwargs=dict(trajectory_weight=1.,binary_mixed_weight=1.),micro_trajectories=5,sampler_source=dict(commit=v['sampler'],path=str(code/v['pkg']/'balanced_hard_g10_sampler.py'),sha256=pub['files'][v['pkg']+'balanced_hard_g10_sampler.py']['sha256']),code_hashes={n:r['sha256']for n,r in pub['files'].items()},actor_updates=0,postblock_cache_used=False,external_evaluation_performed=False,full124194_training_claim=False);immutable(b/'config.json',c);(b/'logs').mkdir(exist_ok=True)
script='''+repr('''#!/usr/bin/env bash
#SBATCH --job-name=q7b-native-joint-qual
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --signal=USR1@180
#SBATCH --output=@@BASE@@/logs/train-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.joint_balanced_native_train --base @@BASE@@
''')+'''.replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code));p=b/'training.sbatch'
if p.exists():assert p.read_text()==script
else:p.write_text(script)
subprocess.check_call(['bash','-n',str(p)]);ready=dict(status='NATIVE_BALANCED_JOINT_REAL_GPU_QUALIFICATION_METADATA_READY',config=proof(b/'config.json'),entry=proof(p),publication=proof(b/'source-publication.json'),source_E4_full_payload_SHA_deferred_to_GPU=True,fit_questions=58432,real_accepted_updates_budget=7,full124194_main_training_claim=False,GPU_submitted=False);immutable(b/'preparation.json',ready);print(json.dumps(ready))
'''
SUBMIT='''assert not active(),'Own job already active';assert not(b/'submission-intent.json').exists(),'Prior unknown submission intent, inspect before retry';ready=read(b/'preparation.json')
for key in ('config','entry','publication'):assert proof(ready[key]['path'])==ready[key]
f=(b/'training.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);f.close();immutable(b/'submission-intent.json',dict(commit=v['commit'],ready=ready,epoch=time.time()));r=subprocess.run(['sbatch','--hold','--parsable','--account=mou176',str(b/'training.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
if r.returncode or not re.fullmatch(r'[0-9]+(?:;[^ ]+)?',r.stdout.strip()):raise RuntimeError('Uncertain sbatch; retained intent, inspect same-name before retry '+r.stdout[-500:]+r.stderr[-1000:])
job=r.stdout.strip().split(';')[0];raw=subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True)
for item in ('JobName=q7b-native-joint-qual','Account=mou176','QOS=normal','Partition=acd_u','TimeLimit=00:20:00','NumCPUs=8','MinMemoryNode=128G','Dependency=(null)','Command='+str(b/'training.sbatch')):assert item in raw,item
identity=dict(job_id=job,job=raw,commit=v['commit'],ready=ready,epoch=time.time());immutable(b/'submission-identity.json',identity);subprocess.check_call(['scontrol','release',job]);result=dict(status='NATIVE_JOINT_QUALIFICATION_JOB_RELEASED',job_id=job,identity=proof(b/'submission-identity.json'),config=ready['config']);immutable(b/'submission-result.json',result);print(json.dumps(result))
'''
STATUS='''out=dict(epoch=time.time(),active_same_name_jobs=active())
for name in ('submission-intent.json','submission-identity.json','submission-result.json','controller.json','reader-selection.json','progress.json','first-update-replay.json','first-cp-independent-audit.json','durable-next-update-replay.json','latest.json','failure.json','interrupted.json','complete.json'):
 if(b/name).exists():
  try:out[name[:-5]]=read(b/name)
  except FileNotFoundError:pass
print(json.dumps(out))
'''
def bundle(commit):
 paths=[str(p.relative_to(R))for folder in ('reproduction','search_g1','search_r1','verl')for p in(R/folder).rglob('*.py')if'__pycache__'not in p.parts]+[PKG+'runtime.sh'];names,_,missing=import_closure(paths,lambda n:(R/n).read_bytes(),seeds=SEEDS,resource_files=())
 if missing:raise ValueError('Missing source closure '+str(missing))
 out=io.BytesIO()
 with gzip.GzipFile(fileobj=out,mode='wb',mtime=0,filename='')as gz:
  with tarfile.open(fileobj=gz,mode='w')as archive:
   for name in names:
    origin=SAMPLER if name==PKG+'balanced_hard_g10_sampler.py'else commit;raw=subprocess.check_output(['git','show',origin+':'+name],cwd=R);assert raw==(R/name).read_bytes(),'Unpublished runtime '+name;info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o444;info.mtime=0;archive.addfile(info,io.BytesIO(raw))
 return out.getvalue()
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','submit','status'),required=True);a=p.parse_args();own=Path(__file__);commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(own.relative_to(R))],cwd=R).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',commit,remote],cwd=R);assert own.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(own.relative_to(R))],cwd=R);spec=importlib.util.spec_from_file_location('native_joint_private',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(body,**kw):
  script='v='+repr(dict(base=BASE,A=A,pkg=PKG,commit=commit,sampler=SAMPLER,controller=dict(path=str(own.relative_to(R)),sha256=hashlib.sha256(own.read_bytes()).hexdigest()),**kw))+'\n'+COMMON+body;compile(script,'native-joint-controller','exec');packed=base64.b64encode(zlib.compress(script.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(cmd.encode())<16000;r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError(r.stdout[-4500:]+r.stderr[-1000:])
  return json.loads(r.stdout.splitlines()[-1])
 if a.action=='stage':
  raw=bundle(commit)
  for i,start in enumerate(range(0,len(raw),4096)):
   chunk=raw[start:start+4096];call("raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha'];p=code/('upload-%04d.part'%v['i'])\nif p.exists():assert p.read_bytes()==raw\nelse:p.write_bytes(raw)\nprint(json.dumps(dict(part=v['i'],accepted=True)))",data=base64.b64encode(chunk).decode(),sha=hashlib.sha256(chunk).hexdigest(),i=i)
  result=call(EXTRACT,parts=(len(raw)+4095)//4096,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 else:result=call({'prepare':PREPARE,'submit':SUBMIT,'status':STATUS}[a.action])
 OUT.mkdir(exist_ok=True,parents=True);(OUT/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
