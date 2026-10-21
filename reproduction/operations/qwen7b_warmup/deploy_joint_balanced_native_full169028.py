"""Immutable169028 native main entry; data-incomplete means WAIT, never a GPU.

The qualification namespace is immutable and never reused. This controller only
stages code / reads metadata until the final full169028 bank is actually sealed.
"""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile,zlib
from pathlib import Path
from .deploy_confirmation_materialization import import_closure
from .deploy_joint_balanced_native import COMMON as QUAL_COMMON,EXTRACT as BASE_EXTRACT
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];PKG='reproduction/operations/qwen7b_warmup/'
A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021';BASE=A+'/joint-balanced-native-E4-169028-v1';BANK=A+'/joint169028-native-training-bank-v1'
OUT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/joint-balanced-native-full169028'
ROOT_BRANCH='refs/heads/codex/q7b-native-joint-root-20261021'
ROOT='ec7347b038d5ccfa7b413f3e24e35e430e516145'
WORKER_ORIGIN='e0559dbbc793a02aacffbaaf345b8c8938df44be'
BASE_RUNTIME='3b56e583efe2469ac9108873c31de7144a5201e6'
ROOT_FILES=tuple(PKG+n for n in('joint_balanced_native_full169028.py','joint_balanced_native_full169028_evaluation.py','full169028_protocol.py'))
JOB='q7b-native-joint169028';SAMPLER='c98ddcd7861b7023f3efa0d82d3c1fbea0664a13'
SEEDS=tuple(PKG+n for n in('joint_balanced_native_full169028.py','joint_balanced_native_full169028_evaluation.py','full169028_protocol.py','joint_balanced_native_update.py','native_joint_rolling_checkpoint.py','runtime.sh'))
COMMON=QUAL_COMMON.replace('q7b-native-joint-qual',JOB)
EXTRACT=BASE_EXTRACT.replace("else v['commit']","else (v['runtime_commit']if n in v['root_files']else v['base_runtime_commit'])").replace("sampler_commit=v['sampler'],","sampler_commit=v['sampler'],runtime_source_commit=v['runtime_commit'],qualified_runtime_base_commit=v['base_runtime_commit'],runtime_source_branch=v['runtime_branch'],worker_origin_commit=v['worker_origin'],")
PREPARE='''import shutil
assert not active()and not(b/'submission-intent.json').exists(),'Existing job/intent must be reconciled, never duplicate'
f=(b/'training.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);f.close()
full=pathlib.Path(v['bank'])/'complete.json'
if not full.exists():
 print(json.dumps(dict(status='WAITING_FOR_ACTUAL_FULL169028_NATIVE_TRAINING_BANK',expected_path=str(full),config_written=False,GPU_submitted=False)));raise SystemExit(0)
complete_ref=proof(full);bank=read(full);assert bank['status']=='FROZEN_FIVE_BANK_REAL_G10_169028_NATIVE_JOINT_INPUTS_ACCEPTED'and bank['questions']==169028 and bank['trajectories']==1690280 and bank['canonical_qid_sha256']=='aa14bce45fc6a793789369d6f33c84edf3c1d79e601fec45882ec193b1e4c58d'and bank['whole_G5_source_selection_before_packed_union']is True and bank['actor_updates']==bank['probe_updates']==0 and bank['final_outcomes_read']is False,'Actual accepted full169028 same-HF source-resolved bank required';gr=bank.get('groups',bank.get('outputs',{}).get('groups'));fr=bank.get('features_layer26',bank.get('features',bank.get('outputs',{}).get('features_layer26',bank.get('outputs',{}).get('features'))))
assert isinstance(gr,dict)and isinstance(fr,dict),'Typed final groups/features descriptors required'
assert gr['path']==v['bank']+'/groups.json'and fr['path']==v['bank']+'/features-layer26.json','Actual frozen final bank filenames differ'
groups=checked(gr)['train'];features=checked(fr);assert bank.get('questions')==len(groups)==169028 and bank.get('trajectories')==1690280
assert isinstance(features,list)and features and all(isinstance(x,dict)and set(('path','bytes','sha256'))<=set(x)for x in features)
qids=set();tids=set();hashes=set();counts=[0]*11
for g in groups:
 assert g['qid']not in qids and isinstance(g.get('question_hash'),str)and g['question_hash'];qids.add(g['qid']);hashes.add(g['question_hash']);own=g['trajectory_ids'];ys=g['labels'];assert len(own)==len(set(own))==len(ys)==10 and not tids.intersection(own)and all(type(y)is int and y in(0,1)for y in ys);tids.update(own);counts[sum(ys)]+=1
assert len(tids)==1690280
old=read(pathlib.Path(v['A'])/'hard-vs-random-E4-477-v1/targeted/config.json');pub=read(b/'source-publication.json');assert pub['commit']==v['commit']
assert pub['runtime_source_commit']==v['runtime_commit']and pub['sampler_commit']==v['sampler']and pub['qualified_runtime_base_commit']==v['base_runtime_commit']
baseline_publication_ref=proof(pathlib.Path(v['A'])/'joint-balanced-native-E4-124194-v1/source-publication.json');baseline_publication=checked(baseline_publication_ref);assert baseline_publication['commit']==v['base_runtime_commit']and baseline_publication['sampler_commit']==v['sampler'],'Actual qualified124194 runtime publication required'
for name,ref in pub['files'].items():
 if name not in v['root_files']:assert baseline_publication['files'].get(name)==ref,'Unchanged helper bytes must match the qualified running124194 source: '+name
source_bank=checked(old['source_bank']);diag=source_bank['diagnostic'];valroot=pathlib.Path(v['A'])/'validation-fixedG5-hf-replay-20261021';valcomplete=proof(valroot/'capture/complete.json');vc=checked(valcomplete)
assert vc['status']=='NEW_VALIDATION2048_FIXED_G5_QUALIFIED_HF_REPLAY_LAYER26_COMPLETE'and vc['source_complete']['sha256']=='5ea5197a841d7bd0b226a82350a2436a8ae3d860143a2ea39fe59b7ecb6837d5'and vc['fixed_G5_sha256']=='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7'
val=checked(vc['source_groups'])['validation'];audit=proof(valroot/'independent-reader-audit.json');assert audit['sha256']=='7c37920375dc13eff72cf0fef1b99153a03bed3113bd9e5c22646d60c99a8b6f'
finalref=proof(pathlib.Path(v['A'])/'training-extraG5-matched6528-20261021/final-exclusion-plan.json');final=checked(finalref);finalrows=final.get('questions',final.get('groups'));assert len(diag)==512 and len(val)==2048 and len(finalrows)==4096
identities={r['qid']:dict(qid=r['qid'],question_hash=r['question_hash'])for r in diag+val+finalrows};assert len(identities)==6656 and not qids.intersection(identities)and not hashes.intersection(r['question_hash']for r in identities.values())
exclusion=b/'evaluation-identity-exclusions.json';immutable(exclusion,dict(role='joint_training_fixed_evaluation_identity_exclusions',groups=sorted(identities.values(),key=lambda r:str(r['qid'])),scope_counts=dict(diagnostic=512,selection=2048,final=4096),final_labels_or_rollouts_read=False,source_refs=dict(diagnostic=old['source_bank'],selection=vc['source_groups'],final_identity_only=finalref)))
dfeatures=proof(pathlib.Path(v['A'])/'peer-newseed8192-g10-train-v1/inputs/diagnostic512-features-HF.json');dr=checked(dfeatures);assert isinstance(dr,list)and dr
r2ref=proof(pathlib.Path(v['A']).parent/'replay-matched9088-bank-v2-20261021/prepared/bank-complete.json');assert r2ref['sha256']=='c593363dfaca0edf3f142e1fa5f93bd7f461432ce7b0f1e186101039ac163dc4';r2=checked(r2ref);allowed=checked(r2['features_by_capture_layer']['26']);assert all(x in allowed for x in dr),'Only diagnostic512 HF manifests from accepted R2 bank'
source=old['source_checkpoint'];assert source['sha256']=='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'and source['steps']==14418 and pathlib.Path(source['path']).stat().st_size==source['bytes']
for key in('source_receipt','source_config','model_config'):checked(old[key])
assert shutil.disk_usage(str(b)).free>=64*1024**3,'Need64GiB free for pins/rolling/staging; no per-step immutable main saves'
c=dict(base=str(b),mode='main',complete_bank=complete_ref,groups=gr,features=fr,exclusions=proof(exclusion),expected_exclusion_questions=6656,fit_questions=169028,fit_real_trajectories=1690280,batch_questions=32,steps_per_epoch=5283,update_budget=21132,epochs=4,matched_evaluation_updates=[3882,7764,11646,15528],save_every=128,checkpoint_policy=dict(immutable_pins=[0,1,3882,5283,7764,10566,11646,15528,15849,21132],atomic_rolling_full_state_every=128,graceful_USR1=True,unsaved_accepted_tail_rollback_allowed=False),source_checkpoint=source,source_receipt=old['source_receipt'],source_config=old['source_config'],source_bank=old['source_bank'],model_dir=old['model_dir'],model_config=old['model_config'],lr=5e-6,ema_decay=.99,loss_kwargs=dict(trajectory_weight=1.,binary_mixed_weight=1.),micro_trajectories=5,sampler_source=dict(commit=v['sampler'],path=str(code/v['pkg']/'balanced_hard_g10_sampler.py'),sha256=pub['files'][v['pkg']+'balanced_hard_g10_sampler.py']['sha256']),code_hashes={n:r['sha256']for n,r in pub['files'].items()},evaluation=dict(diagnostic_groups=old['source_bank'],diagnostic_features=dfeatures,diagnostic_HF_bank=r2ref,validation_complete=valcomplete,validation_reader_audit=audit),actor_updates=0,postblock_cache_used=False,old_Count5120_is_training=True,old_Count5120_unseen_evaluation_claim=False,untouched_final_evaluation_performed=False,canonical_duplicate_question_hashes=len(groups)-len(hashes),source_world_size=4,current_world_size=1,four_rank_bitexact_resume_claim=False,requested_hours=48,qualified_runtime_base_commit=v['base_runtime_commit'],runtime_source_commit=v['runtime_commit'],worker_origin_commit=v['worker_origin'],reference_cohort_questions=124194,qualified124194_runtime_publication=baseline_publication_ref,sampler_epoch_semantics='ceil(cohort/32) accepted updates, class-balanced draws; not each question exactly once')
bank_exclusion=checked(bank['evaluation_exclusions']);assert bank_exclusion['role']=='joint_training_fixed_evaluation_identity_exclusions'and {r['qid']:r['question_hash']for r in bank_exclusion['groups']}=={q:r['question_hash']for q,r in identities.items()},'Full bank and run use identical true6656 exclusions';immutable(b/'config.json',c);(b/'logs').mkdir(exist_ok=True)
script=v['entry'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code));p=b/'training.sbatch'
if p.exists():assert p.read_text()==script
else:p.write_text(script)
subprocess.check_call(['bash','-n',str(p)]);ready=dict(status='FULL169028_NATIVE_BALANCED_HARD_JOINT_MAIN_METADATA_READY',config=proof(b/'config.json'),entry=proof(p),publication=proof(b/'source-publication.json'),actual_bank_complete=complete_ref,questions=169028,trajectories=1690280,canonical_duplicate_question_hashes=len(groups)-len(hashes),K10_histogram=counts,source_E4_full_payload_SHA_and_tensor_restore_deferred_to_allocation=True,all_native_feature_payload_SHA_deferred_to_allocation=True,GPU_submitted=False);immutable(b/'preparation.json',ready);print(json.dumps(ready))
'''
ENTRY='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-native-joint169028
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
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
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.joint_balanced_native_full169028 --base @@BASE@@
'''
SUBMIT='''assert not active(),'Own main job already active';assert not(b/'submission-intent.json').exists(),'Unknown prior submit intent: reconcile same-name job before retry';ready=read(b/'preparation.json');assert ready['status']=='FULL169028_NATIVE_BALANCED_HARD_JOINT_MAIN_METADATA_READY'
for key in('config','entry','publication'):assert proof(ready[key]['path'])==ready[key]
assert proof(ready['actual_bank_complete']['path'])==ready['actual_bank_complete'];f=(b/'training.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);f.close();immutable(b/'submission-intent.json',dict(commit=v['commit'],ready=ready,epoch=time.time()))
r=subprocess.run(['sbatch','--hold','--parsable','--account=mou176',str(b/'training.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
if r.returncode or not re.fullmatch(r'[0-9]+(?:;[^ ]+)?',r.stdout.strip()):raise RuntimeError('Uncertain sbatch; retained intent, inspect same-name '+r.stdout[-500:]+r.stderr[-1000:])
job=r.stdout.strip().split(';')[0];raw=subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True)
for item in('JobName=q7b-native-joint169028','Account=mou176','QOS=normal','Partition=acd_u','TimeLimit=2-00:00:00','NumCPUs=8','MinMemoryNode=128G','Dependency=(null)','Command='+str(b/'training.sbatch'),'WorkDir='+str(b)):assert item in raw,item
assert re.search(r'(?:^|[ ,])gres/gpu=1(?:[ ,]|$)',raw)or re.search(r'TresPerNode=gres/gpu:1(?:[ ,]|$)',raw),'Exactly one GPU allocation required'
identity=dict(job_id=job,job=raw,commit=v['commit'],ready=ready,epoch=time.time());immutable(b/'submission-identity.json',identity);subprocess.check_call(['scontrol','release',job]);result=dict(status='FULL169028_NATIVE_JOINT_MAIN_JOB_RELEASED',job_id=job,identity=proof(b/'submission-identity.json'),config=ready['config']);immutable(b/'submission-result.json',result);print(json.dumps(result))
'''
STATUS='''out=dict(epoch=time.time(),active_same_name_jobs=active())
for name in('preparation.json','submission-intent.json','submission-identity.json','submission-result.json','controller.json','reader-selection.json','progress.json','evaluation-progress.json','first-update-replay.json','first-cp-independent-audit.json','durable-next-update-replay.json','latest.json','failure.json','interrupted.json','complete.json'):
 try:
  if(b/name).exists():out[name[:-5]]=read(b/name)
 except FileNotFoundError:pass
print(json.dumps(out))
'''
def runtime_sources():
 """Read the immutable root publication, never the shared working tree."""
 paths=set(subprocess.check_output(['git','ls-tree','-r','--name-only',ROOT],cwd=R).decode().splitlines())|set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASE_RUNTIME],cwd=R).decode().splitlines())|set(subprocess.check_output(['git','ls-tree','-r','--name-only',SAMPLER],cwd=R).decode().splitlines());paths=[n for n in paths if(n.endswith('.py')and n.split('/')[0]in('reproduction','search_g1','search_r1','verl'))or n==PKG+'runtime.sh'];cache={}
 def load(name):
  if name not in cache:cache[name]=subprocess.check_output(['git','show',(SAMPLER if name==PKG+'balanced_hard_g10_sampler.py'else ROOT if name in ROOT_FILES else BASE_RUNTIME)+':'+name],cwd=R)
  return cache[name]
 names,_,missing=import_closure(paths,load,seeds=SEEDS,resource_files=())
 if missing:raise ValueError('Missing published runtime closure '+str(missing))
 for name in('joint_balanced_native_full169028.py','joint_balanced_native_full169028_evaluation.py','full169028_protocol.py'):
  if load(PKG+name)!=subprocess.check_output(['git','show',WORKER_ORIGIN+':'+PKG+name],cwd=R):raise ValueError('Declared worker origin changed '+name)
 return {name:load(name)for name in names}

def bundle(commit):
 contents=runtime_sources();out=io.BytesIO()
 with gzip.GzipFile(fileobj=out,mode='wb',mtime=0,filename='')as gz:
  with tarfile.open(fileobj=gz,mode='w')as archive:
   for name,raw in sorted(contents.items()):
    info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o444;info.mtime=0;archive.addfile(info,io.BytesIO(raw))
 return out.getvalue()

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','submit','status'),required=True);p.add_argument('--submit-authorized',action='store_true');a=p.parse_args();
 if a.action=='submit'and not a.submit_authorized:p.error('Explicit --submit-authorized is required after the separate GPU GO')
 own=Path(__file__);commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(own.relative_to(R))],cwd=R).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',commit,remote],cwd=R);subprocess.check_call(['git','merge-base','--is-ancestor',BASE_RUNTIME,remote],cwd=R);assert own.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(own.relative_to(R))],cwd=R)
 root_remote=subprocess.check_output(['git','ls-remote','origin',ROOT_BRANCH],cwd=R).decode().split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',ROOT,root_remote],cwd=R);spec=importlib.util.spec_from_file_location('native_joint_full_private',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(body,**kw):
  script='v='+repr(dict(base=BASE,A=A,bank=BANK,pkg=PKG,commit=commit,sampler=SAMPLER,base_runtime_commit=BASE_RUNTIME,root_files=ROOT_FILES,runtime_commit=ROOT,runtime_branch=ROOT_BRANCH,worker_origin=WORKER_ORIGIN,controller=dict(path=str(own.relative_to(R)),sha256=hashlib.sha256(own.read_bytes()).hexdigest()),**kw))+'\n'+COMMON+body;compile(script,'native-joint-main-controller','exec');packed=base64.b64encode(zlib.compress(script.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(cmd.encode())<16000;r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError(r.stdout[-4500:]+r.stderr[-1000:])
  return json.loads(r.stdout.splitlines()[-1])
 if a.action=='stage':
  raw=bundle(commit)
  for i,start in enumerate(range(0,len(raw),4096)):
   chunk=raw[start:start+4096];call("raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha'];p=code/('upload-%04d.part'%v['i'])\nif p.exists():assert p.read_bytes()==raw\nelse:p.write_bytes(raw)\nprint(json.dumps(dict(part=v['i'],accepted=True)))",data=base64.b64encode(chunk).decode(),sha=hashlib.sha256(chunk).hexdigest(),i=i)
  result=call(EXTRACT,parts=(len(raw)+4095)//4096,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 else:result=call({'prepare':PREPARE,'submit':SUBMIT,'status':STATUS}[a.action],**({'entry':ENTRY}if a.action=='prepare'else{}))
 OUT.mkdir(exist_ok=True,parents=True);(OUT/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
