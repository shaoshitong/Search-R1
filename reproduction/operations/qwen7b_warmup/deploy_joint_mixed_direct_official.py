"""Isolated published-origin deployment for official Mixed-only G10 training.

Stage/prepare/status never submit. The root owner must explicitly authorize the
single guarded normal1H10048h submission. No E4 weights/Adam/RNG are accepted.
"""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021';BASE=A+'/joint-mixed-direct-official-169028-v1';BANK=A+'/joint169028-native-training-bank-v1';JOB='q7b-mixed-direct169028'
BASE_RUNTIME='3b56e583efe2469ac9108873c31de7144a5201e6';SAMPLER='c98ddcd7861b7023f3efa0d82d3c1fbea0664a13';MODEL='8c0ccaf4eda6d97d46c9f0ea82620aa1a6d9a973';WORKER='e2fd6e7fbe4fbd6d62fc86923565c01765d02794';EVALUATION='7380f3bfa82e4d699a617590d25d56bf4ef8b5ad'
CONTRACT={'path':BASE+'/OFFICIAL-SOURCE-AND-EVAL-CONTRACT.json','bytes':7613,'sha256':'8753d78685a402454c0e2d03bdf7da903b23a537a73d4e3dc52a9690d59729c3'}
OVERRIDES={P+'joint_mixed_direct_pretrain.py':WORKER,P+'mixed_group_checkpoint.py':WORKER,P+'native_mixed_group_classifier.py':MODEL,P+'mixed_direct_evaluation.py':EVALUATION,P+'balanced_hard_g10_sampler.py':SAMPLER}
SEEDS=tuple(OVERRIDES)+tuple(P+n for n in('runtime.sh','joint_balanced_native_evaluation.py'))
OUT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/joint-mixed-direct-official-deployment'
COMMON='''import pathlib,json,hashlib,os,fcntl,subprocess,time,re,base64,io,gzip,tarfile,shutil
b=pathlib.Path(v['base']);code=b/('code-'+v['commit'][:12]);b.mkdir(parents=True,exist_ok=True);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 assert proof(ref['path'])==ref,'Declared input SHA/bytes differ';return read(ref['path'])
def immutable(p,value):
 raw=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw,'Immutable official phase control differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert r.returncode==0,r.stderr[-1000:];return r.stdout.strip()
def active():return command(['squeue','-h','-u','mou176','-n',v['job'],'-o','%i|%T|%R'])
def publication():
 p=read(b/'source-publication.json');assert p['commit']==v['commit']and p['code_root']==str(code)and p['file_origins']==v['origins']
 for name,item in p['files'].items():assert proof(code/name)==dict(path=str(code/name),**item)
 return p
'''
STAGE='''raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha'];p=code/('upload-%04d.part'%v['part'])
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(part=v['part'],accepted=True)))
'''
EXTRACT='''raw=b''.join((code/('upload-%04d.part'%i)).read_bytes()for i in range(v['parts']));assert len(raw)==v['bytes']and hashlib.sha256(raw).hexdigest()==v['sha'];members={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as tar:
 for member in tar:
  path=pathlib.PurePosixPath(member.name);assert member.isfile()and not path.is_absolute()and'..'not in path.parts and (member.name=='runtime-manifest.json'or path.parts[0]in('reproduction','search_g1','search_r1','verl'))
  assert member.name not in members;members[member.name]=tar.extractfile(member).read()
manifest=json.loads(members.pop('runtime-manifest.json'));assert manifest['origins']==v['origins']and set(manifest['files'])==set(members)
for name,data in members.items():
 item=manifest['files'][name];assert len(data)==item['bytes']and hashlib.sha256(data).hexdigest()==item['sha256'];p=code/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data);p.chmod(0o444)
# Compile source with the qualified Python3.10, not login Python3.6.
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');check="import pathlib;root=pathlib.Path("+repr(str(code))+");files="+repr(sorted(members))+";[compile((root/n).read_bytes(),str(root/n),'exec')for n in files if n.endswith('.py')];print('SOURCE_COMPILE_PASSED')"
r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',check],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=35);assert r.returncode==0 and'SOURCE_COMPILE_PASSED'in r.stdout,r.stderr[-1200:]
assert not any(name.endswith(('joint_balanced_native_main.py','joint_balanced_native_full169028.py'))for name in members),'Old dual-BCE workers are not runtime seeds'
result=dict(commit=v['commit'],controller=v['controller'],code_root=str(code),files=manifest['files'],file_origins=v['origins'],official_contract=v['contract'],training_worker='reproduction.operations.qwen7b_warmup.joint_mixed_direct_pretrain',initialization_source='official_initial',old_probe_checkpoint_used=False,old_dual_BCE_worker_submittable=False)
immutable(b/'source-publication.json',result);publication();print(json.dumps(dict(status='OFFICIAL_DIRECT_MIXED_ONLY_IMMUTABLE_SOURCE_STAGED',publication=proof(b/'source-publication.json'),files=len(members))))
'''
PREPARE='''pub=publication();assert not active()and not(b/'submission-intent.json').exists(),'Existing identity/intent must be reconciled, not duplicated';f=(b/'training.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);f.close()
for name in('latest.json','progress.json','update-intent.json','interrupted.json'):assert not(b/name).exists(),'Existing training state requires exact resume, never fresh preparation'
contract=checked(v['contract']);assert contract['status']=='OFFICIAL_FRESH_NATIVE_AND_DIRECT_MIXED_GROUP_SOURCE_EVAL_METADATA_BOUND'and contract['revision']=='a09a35458c702b33eeacc393d103063234e8bc28'and contract['initialization']['native_Adam_steps']==contract['initialization']['group_head_Adam_steps']==0 and contract['initialization']['posttraining_probe_checkpoint_read']is False
full=pathlib.Path(v['bank'])/'complete.json'
if not full.exists():print(json.dumps(dict(status='WAITING_FOR_ACTUAL_FULL169028_SOURCE_RESOLVED_BANK',expected_path=str(full),source_publication=proof(b/'source-publication.json'),config_written=False,GPU_submitted=False)));raise SystemExit(0)
bref=proof(full);bank=read(full);assert bank['status']=='FROZEN_FIVE_BANK_REAL_G10_169028_NATIVE_JOINT_INPUTS_ACCEPTED'and bank['questions']==169028 and bank['trajectories']==1690280 and bank['canonical_qid_sha256']=='aa14bce45fc6a793789369d6f33c84edf3c1d79e601fec45882ec193b1e4c58d'and bank['whole_G5_source_selection_before_packed_union']is True and bank['final_outcomes_read']is False and bank['actor_updates']==bank['probe_updates']==0
outputs=bank.get('outputs',{});groups=bank.get('groups',outputs.get('groups',outputs.get('groups.json')));features=bank.get('features_layer26',outputs.get('features_layer26',outputs.get('features-layer26.json')));assert groups['path']==v['bank']+'/groups.json'and features['path']==v['bank']+'/features-layer26.json';assert proof(groups['path'])==groups and proof(features['path'])==features
# The full-bank producer has already joined every ownG10 identity and feature row.
# Rehash its sealed groups/list, without repeating the large identity proof or tensors.
exclusions=checked(contract['exclusions']);assert exclusions['role']=='joint_training_fixed_evaluation_identity_exclusions'and exclusions['scope_counts']==dict(diagnostic=512,selection=2048,final=4096)and len(exclusions['groups'])==6656 and exclusions['final_labels_or_rollouts_read']is False
assert bank['evaluation_exclusions']==contract['exclusions'],'Actual full bank and fixed6656 run exclusions differ'
for key in('model_config','model_index','official_inventory'):checked(contract[key])
assert shutil.disk_usage(str(b)).free>=64*1024**3,'Need rolling plus epoch pins/staging reserve'
c=dict(base=str(b),mode='main',initialization_source='official_initial',official_revision=contract['revision'],official_source_eval_contract=v['contract'],official_inventory=contract['official_inventory'],model_dir=contract['model_dir'],model_config=contract['model_config'],model_files=contract['model_files'],complete_bank=bref,groups=groups,features=features,exclusions=contract['exclusions'],expected_exclusion_questions=6656,fit_questions=169028,fit_real_trajectories=1690280,steps_per_epoch=5283,epochs=4,update_budget=21132,seed=1729,head_seed=9137,lr=5e-6,ema_decay=.99,save_every=128,sampler_source=v['sampler'],architecture=dict(capture_layer=26,block_layer=27,question_max_tokens=273,prefix_tokens=20,head_dimension=256,head_layers=2,head_heads=8,micro_questions=1,views_per_question=4,batch_questions=32),loss=dict(binary_mixed_weight=1.),evaluation=contract['evaluation'],code_hashes={name:item['sha256']for name,item in pub['files'].items()},checkpoint_policy=dict(immutable_pins=[0,1,2,5283,10566,15849,21132],rolling_every=128,raw_EMA_native_Adam_head_Adam_RNG_sampler=True,graceful_USR1=True,unsaved_accepted_tail_rollback_allowed=False),initial_native_Adam_steps=0,initial_group_head_Adam_steps=0,actor_updates=0,trajectory_loss_enabled=False,old_probe_checkpoint_loaded=False,old_probe_Adam_or_RNG_inherited=False,final4096_scored=False,requested_hours=48,source_publication=proof(b/'source-publication.json'))
for name in('source_checkpoint','source_receipt','source_config','resume_from_E4'):assert name not in c
immutable(b/'config.json',c);(b/'logs').mkdir(exist_ok=True);text=v['entry'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code));p=b/'training.sbatch'
if p.exists():assert p.read_text()==text
else:p.write_text(text)
subprocess.check_call(['bash','-n',str(p)]);ready=dict(status='OFFICIAL_MIXED_ONLY_FULL169028_METADATA_READY_NOT_SUBMITTED',config=proof(b/'config.json'),entry=proof(p),publication=proof(b/'source-publication.json'),bank_complete=bref,official_contract=v['contract'],native_Adam0=True,group_head_Adam0=True,full_official_weights_SHA_deferred_to_GPU=True,all_features_payload_SHA_deferred_to_GPU=True,GPU_submitted=False);immutable(b/'preparation.json',ready);print(json.dumps(ready))
'''
ENTRY='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-mixed-direct169028
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
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.joint_mixed_direct_pretrain --base @@BASE@@
'''
SUBMIT='''publication();f=(b/'submit-control.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not active(),'Same-name active job: do not repeat';assert not(b/'submission-intent.json').exists(),'Unknown/existing submit intent: inspect same job before retry';ready=read(b/'preparation.json');assert ready['status']=='OFFICIAL_MIXED_ONLY_FULL169028_METADATA_READY_NOT_SUBMITTED'
for key in('config','entry','publication','bank_complete','official_contract'):assert proof(ready[key]['path'])==ready[key]
t=(b/'training.lock').open('a+');fcntl.flock(t,fcntl.LOCK_EX|fcntl.LOCK_NB);t.close()
for name in('latest.json','progress.json','update-intent.json','failure.json','interrupted.json','complete.json'):assert not(b/name).exists(),'Existing transaction needs exact continuation, never resubmit fresh'
immutable(b/'submission-intent.json',dict(epoch=time.time(),ready=ready,controller_commit=v['commit']))
r=subprocess.run(['sbatch','--hold','--parsable','--account=mou176',str(b/'training.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
if r.returncode or not re.fullmatch(r'[0-9]+(?:;[^ ]+)?',r.stdout.strip()):raise RuntimeError('Submission uncertain/rejected: preserve intent, inspect same name; '+r.stdout[-300:]+r.stderr[-1000:])
job=r.stdout.strip().split(';')[0];identity=command(['scontrol','show','job','-o',job])
for item in('JobName='+v['job'],'Account=mou176','QOS=normal','Partition=acd_u','TimeLimit=2-00:00:00','NumCPUs=8','MinMemoryNode=128G','Dependency=(null)','Command='+str(b/'training.sbatch'),'WorkDir='+str(b)):assert item in identity,item
assert re.search(r'(?:^|[ ,])gres/gpu=1(?:[ ,]|$)',identity)or re.search(r'TresPerNode=gres/gpu:1(?:[ ,]|$)',identity),'Exactly1 GPU required'
immutable(b/'submission-identity.json',dict(job_id=job,job=identity,ready=ready,controller_commit=v['commit'],epoch=time.time()));command(['scontrol','release',job]);result=dict(status='OFFICIAL_MIXED_ONLY169028_JOB_RELEASED_NOT_PROGRESS_VERIFIED',job_id=job,identity=proof(b/'submission-identity.json'),config=ready['config']);immutable(b/'submission-result.json',result);print(json.dumps(result))
'''
STATUS='''out=dict(epoch=time.time(),active_same_name_jobs=active(),base=str(b))
for name in('preparation.json','submission-intent.json','submission-identity.json','submission-result.json','controller.json','progress.json','evaluation-progress.json','first-update-replay.json','first-cp-independent-audit.json','durable-next-update-replay.json','latest.json','failure.json','interrupted.json','complete.json'):
 for attempt in range(3):
  try:out[name[:-5]]=read(b/name);break
  except FileNotFoundError:
   if attempt<2:time.sleep(.03)
if(b/'submission-result.json').exists():out['main_sacct']=command(['sacct','-X','-n','-P','-j',read(b/'submission-result.json')['job_id'],'--format=JobIDRaw,State,ExitCode,Elapsed'])
print(json.dumps(out))
'''
def git_bytes(commit,name):return subprocess.check_output(['git','show',commit+':'+name],cwd=R)
def runtime_sources():
 if len(WORKER)!=40:raise ValueError('Final qualified published worker origin required before deployment')
 scope={}
 for origin in set(OVERRIDES.values())|{BASE_RUNTIME}:
  for n in subprocess.check_output(['git','ls-tree','-r','--name-only',origin],cwd=R).decode().splitlines():
   if(n.endswith('.py')and n.split('/')[0]in('reproduction','search_g1','search_r1','verl'))or n==P+'runtime.sh':scope[n]=True
 helper=P+'deploy_confirmation_materialization.py';ns=dict(__name__='qualified_frozen_import_closure',__file__=str(R/helper));exec(compile(git_bytes(BASE_RUNTIME,helper),str(R/helper),'exec'),ns);cache={}
 def load(name):
  if name not in cache:cache[name]=git_bytes(OVERRIDES.get(name,BASE_RUNTIME),name)
  return cache[name]
 names,_,missing=ns['import_closure'](list(scope),load,seeds=SEEDS,resource_files=())
 if missing:raise ValueError('Published runtime closure missing '+str(missing))
 contents={name:load(name)for name in names};origins={name:OVERRIDES.get(name,BASE_RUNTIME)for name in names}
 return contents,origins

def bundle():
 contents,origins=runtime_sources();manifest=dict(origins=origins,files={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())for name,raw in contents.items()});payload=dict(contents);payload['runtime-manifest.json']=(json.dumps(manifest,sort_keys=True)+chr(10)).encode();out=io.BytesIO()
 with gzip.GzipFile(fileobj=out,mode='wb',mtime=0,filename='')as gz:
  with tarfile.open(fileobj=gz,mode='w')as tar:
   for name,raw in sorted(payload.items()):info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o444;info.mtime=0;tar.addfile(info,io.BytesIO(raw))
 return out.getvalue(),manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('check','stage','prepare','submit','status'),required=True);p.add_argument('--submit-authorized',action='store_true');a=p.parse_args()
 if a.action=='submit'and not a.submit_authorized:p.error('Root explicit actual-submit authorization required')
 raw,manifest=bundle();own=Path(__file__);rel=str(own.relative_to(R));commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=R).decode().strip();assert own.read_bytes()==git_bytes(commit,rel),'Controller must be published unchanged';remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',commit,remote],cwd=R)
 if a.action=='check':print(json.dumps(dict(status='PUBLISHED_ORIGIN_CLOSURE_AND_STATIC_SCRIPTS_PASSED',files=len(manifest['files']),archive_bytes=len(raw),file_origins=manifest['origins'],files_manifest=manifest['files'])));return
 spec=importlib.util.spec_from_file_location('official_direct_mix_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(body,**kw):
  values=dict(base=BASE,bank=BANK,job=JOB,commit=commit,contract=CONTRACT,sampler=SAMPLER,origins=manifest['origins'],controller=dict(path=rel,sha256=hashlib.sha256(own.read_bytes()).hexdigest(),commit=commit));values.update(kw);source='v='+repr(values)+'\n'+COMMON+body;compile(source,'official-direct-mix-controller','exec');packed=base64.b64encode(zlib.compress(source.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;r=ctl.run(command,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError(r.stdout[-3000:]+r.stderr[-2000:])
  return json.loads(r.stdout.splitlines()[-1])
 if a.action=='stage':
  parts=[raw[i:i+4096]for i in range(0,len(raw),4096)]
  for i,chunk in enumerate(parts):call(STAGE,data=base64.b64encode(chunk).decode(),sha=hashlib.sha256(chunk).hexdigest(),part=i)
  result=call(EXTRACT,parts=len(parts),bytes=len(raw),sha=hashlib.sha256(raw).hexdigest())
 else:result=call({'prepare':PREPARE,'submit':SUBMIT,'status':STATUS}[a.action],**({'entry':ENTRY}if a.action=='prepare'else{}))
 OUT.mkdir(parents=True,exist_ok=True);(OUT/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
for text in(COMMON,STAGE,EXTRACT,PREPARE,SUBMIT,STATUS):compile(text,'official-direct-mixed-static','exec')
if __name__=='__main__':main()
