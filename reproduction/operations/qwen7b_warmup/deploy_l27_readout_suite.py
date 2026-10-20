"""Published4KiB code staging and explicit guarded L27 readout suite actions.

Prepare performs CPU imports and the actual metadata alias audit ONLY: no E1
checkpoint tensors or official model tensors are loaded on the login node.
GPU variants independently inherit E1 block Adam10464 and create head Adam0.
No action executes on import or --self-test. No activity modifies old sources.
"""
import argparse
import base64
import copy
import gzip
import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

PARENT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
TARGET=PARENT+'/algorithm-improvement-20261021/l27-readout128-v1'
Q1=PARENT+'/replay-layer-qualification-20261021'
SOURCE_SHA='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'
CODE_FILES=tuple('reproduction/operations/qwen7b_warmup/'+name for name in (
    'l27_readout_train.py','test_l27_readout_train.py','probe_ablation_train.py',
    'probe_ablation_loss.py','probe_ablation_model.py','test_probe_ablation_model.py',
    'model.py','packed_features.py','native_capture.py','download_model.py',
    'runtime.sh','deploy_l27_readout_suite.py'))


def variant_specs(alias_verified):
    result={
        'last-prefix':('last_valid_token','prefix_only'),
        'mean-prefix':('mean','prefix_only'),
        'mean-question-prefix':('mean','question_prefix'),
        'attention-prefix':('learned_attention','prefix_only'),
        'attention-question-prefix':('learned_attention','question_prefix'),
        'flatten':('flatten','question_prefix')}
    if not alias_verified:result['last-question-prefix']=('last_valid_token','question_prefix')
    return result


def make_config(reference,name,spec,hashes,commit,model_dir,model_config,model_inventory):
    if (reference['source_checkpoint']['sha256']!=SOURCE_SHA or reference['update_budget']!=128
            or reference['global_groups']!=51 or reference['micro_groups']!=2
            or reference['recipe']['loss_kwargs']!=dict(trajectory_weight=1.,three_event_weight=1.)
            or model_config['sha256']!=reference['model_config']['sha256']):
        raise ValueError('Reuse the exact accepted E1/event/6528/B51/128 reference')
    config=copy.deepcopy(reference)
    config.update(model_dir=model_dir,model_config=copy.deepcopy(model_config),
        model_inventory=copy.deepcopy(model_inventory),code_hashes=copy.deepcopy(hashes),
        source_commit=commit,save_every=16,
        readout=dict(kind=spec[0],scope=spec[1],block_layer=27,capture_layer=26,head_seed=1729),
        recipe=dict(id='l27-'+name,loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.)),
        source_block_adam_expected=10464,new_head_adam_expected=0,
        whole_model_lossless_resume=False,screening_only=True)
    return config


def render_sbatch(name,base,code_root):
    if name not in variant_specs(False):raise ValueError('Unknown readout variant')
    for value in (base,code_root):
        p=Path(value)
        if p.parent!=Path(TARGET) or any(c.isspace() for c in str(p)) or any(c in str(p) for c in '\r\n;`$'):
            raise ValueError('Safe independent suite child paths required')
    return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-l27r128-@@NAME@@
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=@@BASE@@/logs/readout-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE_Q@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.l27_readout_train --base @@BASE_Q@@
'''.replace('@@NAME@@',name).replace('@@BASE@@',str(base)).replace('@@BASE_Q@@',shlex.quote(str(base))).replace('@@CODE_Q@@',shlex.quote(str(code_root)))


COMMON=r'''
import pathlib,json,hashlib,base64,os,fcntl,time,subprocess,gzip,tarfile,io,sys,copy,shlex
from pathlib import Path
sys.dont_write_bytecode=True
b=pathlib.Path(inputs['base']);parent=pathlib.Path(inputs['parent']);q1=pathlib.Path(inputs['q1'])
assert str(b)==inputs['target'] and b.is_absolute() and b.resolve()==b
code=b/('code-'+inputs['commit'][:12]);uploads=b/('upload-'+inputs['bundle_sha256'][:16])
def read(p):return json.loads(pathlib.Path(p).read_text())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def atomic(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 with tmp.open('wb') as f:f.write(canonical(v)+b'\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def verify(item):
 p=pathlib.Path(item['path']);assert p.is_absolute() and sha(p)==item['sha256'],str(p)
 if 'bytes' in item:assert p.stat().st_size==item['bytes']
 return p
def lock():
 h=(b/'controller.lock').open('a+');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);return h
def publication():
 pub=read(b/'source-publication.json');root=pathlib.Path(pub['code_root'])
 assert root.parent==b and root.name=='code-'+pub['commit'][:12]
 for name,item in pub['files'].items():assert (root/name).stat().st_size==item['bytes'] and sha(root/name)==item['sha256'],name
 return pub,root
def command(args,timeout=25):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout)
 assert p.returncode==0,(args,p.stdout[-1000:],p.stderr[-2000:]);return p.stdout.strip()
def verify_prepared():
 pub,root=publication();done=read(b/'preparation.json')
 verify(done['source_inputs']);verify(done['alias_audit']);verify(done['native_index_audit'])
 assert done['commit']==pub['commit'] and done['code_root']==str(root)
 for item in done['variants'].values():verify(item['config']);verify(item['sbatch'])
 for item in done['aliases'].values():verify(item['config']);verify(item['proof'])
 return done,pub,root
'''

INIT=r'''
b.mkdir(parents=True,exist_ok=True);guard=lock()
if (b/'preparation.json').exists():
 done,pub,root=verify_prepared()
 print(json.dumps(dict(status='EXISTING_PREPARED_READOUT_SUITE_VERIFIED',preparation=done)))
else:
 for name in ['logs','inputs']:(b/name).mkdir(exist_ok=True)
 controller=dict(commit=inputs['commit'],bundle_sha256=inputs['bundle_sha256'],bundle_bytes=inputs['bundle_bytes'],chunks=inputs['chunks'],files=inputs['files'])
 path=b/'upload-controller.json'
 if path.exists():assert read(path)==controller,'Different immutable preparation version exists'
 else:atomic(path,controller)
 uploads.mkdir(exist_ok=True)
 print(json.dumps(dict(status='READOUT_CODE_UPLOAD_BOUND',commit=inputs['commit'],chunks=inputs['chunks'])))
'''

CHUNK=r'''
guard=lock();assert not (b/'preparation.json').exists();controller=read(b/'upload-controller.json')
assert controller['commit']==inputs['commit'] and controller['bundle_sha256']==inputs['bundle_sha256']
index=inputs['chunk_index'];assert 0<=index<inputs['chunks']
raw=base64.b64decode(inputs['chunk_data']);assert len(raw)<=4096 and hashlib.sha256(raw).hexdigest()==inputs['chunk_sha256']
part=uploads/('part-%05d'%index)
if part.exists():assert part.read_bytes()==raw
else:
 with part.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status='READOUT_CODE_CHUNK_SHA_PASSED',index=index,bytes=len(raw))))
'''

EXTRACT=r'''
guard=lock();parts=sorted(uploads.glob('part-*'));assert len(parts)==inputs['chunks']
raw=b''.join(p.read_bytes() for p in parts)
assert len(raw)==inputs['bundle_bytes'] and hashlib.sha256(raw).hexdigest()==inputs['bundle_sha256']
if not code.exists():
 stage=code.with_name(code.name+'.staging');stage.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
  members=archive.getmembers();assert {m.name for m in members}==set(inputs['files'])
  for member in members:
   rel=pathlib.Path(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
   payload=archive.extractfile(member).read();assert hashlib.sha256(payload).hexdigest()==inputs['files'][member.name]['sha256']
   dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():assert dest.read_bytes()==payload
   else:
    with dest.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
   dest.chmod(0o444)
 assert {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}==set(inputs['files'])
 stage.rename(code)
pub=dict(commit=inputs['commit'],remote_sha=inputs['commit'],code_root=str(code),files=inputs['files'],bundle_sha256=inputs['bundle_sha256'],other_active_sources_unchanged=True)
path=b/'source-publication.json'
if path.exists():assert read(path)==pub
else:atomic(path,pub)
publication();print(json.dumps(dict(status='IMMUTABLE_READOUT_RUNTIME_READY',code_root=str(code),files=len(inputs['files']))))
'''

PREPARE=r'''
guard=lock();pub,root=publication();reference_suite=parent/'algorithm-improvement-20261021/matched-loss128-v2'
accepted=read(reference_suite/'preparation.json');assert accepted['status']=='MATCHED_LOSS128_SUITE_PREPARED_ACTUAL_CPU_TESTS_PASSED'
reference_ref=accepted['variants']['event']['config'];reference=read(verify(reference_ref))
assert reference['source_checkpoint']['sha256']==SOURCE_SHA and reference['update_budget']==128 and reference['global_groups']==51 and reference['micro_groups']==2
assert reference['recipe']['loss_kwargs']==dict(trajectory_weight=1.,three_event_weight=1.)
keys=['source_checkpoint','source_config','source_bank','model_config','bank','features','selection_bank','selection_features','plan']
for key in keys:verify(reference[key])
plan=read(reference['plan']['path']);flat=[qid for batch in plan['batches'] for qid in batch]
assert len(plan['batches'])==128 and all(len(batch)==51 for batch in plan['batches']) and len(flat)==len(set(flat))==6528 and set(flat)==set(plan['cohort_qids'])
oldpub=read(q1/'source-publication.json');assert oldpub['code_root']==str(q1/'code-18c46affb9c6')
for name,item in oldpub['files'].items():assert sha(pathlib.Path(oldpub['code_root'])/name)==item['sha256']
oldprep=read(q1/'preparation.json');selected=read(verify(oldprep['selection']));complete=read(q1/'qualification-output/complete.json')
assert complete['status']=='BOUNDED_HF_REPLAY_CAPTURE_AND_NATIVE_COMPARISON_COMPLETE' and complete['source_selection']==fp(oldprep['selection']['path'])
view=q1/'official-snapshot-view';assert oldprep['model_view']==str(view)
for name,item in selected['model_files'].items():
 assert complete['model_files'][name]['sha256']==item['sha256'] and (view/name).stat().st_size==item['bytes']
 if not name.endswith('.safetensors'):assert sha(view/name)==item['sha256']
model_config=fp(view/'config.json');inventory=fp(view/'official-inventory.json')
assert model_config['sha256']==reference['model_config']['sha256']
hashes={name:item['sha256'] for name,item in pub['files'].items()}
canonical_config=make_config(reference,'last-prefix',('last_valid_token','prefix_only'),hashes,pub['commit'],str(view),model_config,inventory)
base=b/'last-prefix';base.mkdir(exist_ok=True);(base/'logs').mkdir(exist_ok=True)
path=base/'config.json'
if path.exists():assert read(path)==canonical_config
else:atomic(path,canonical_config)
source=dict(reference_config=reference_ref,reference_preparation=fp(reference_suite/'preparation.json'),
 reference_source_checkpoint=reference['source_checkpoint'],source_plan=reference['plan'],
 q1_source_publication=fp(q1/'source-publication.json'),q1_model_selection=oldprep['selection'],
 q1_complete=fp(q1/'qualification-output/complete.json'),model_dir=str(view),model_config=model_config,model_inventory=inventory,
 checkpoint_tensors_loaded_on_login=False,model_tensors_loaded_on_login=False)
path=b/'prepared-inputs.json'
if path.exists():assert read(path)==source
else:atomic(path,source)
print(json.dumps(dict(status='READOUT128_SOURCE_PLAN_AND_Q1_MODEL_BOUND',fit_questions=6528,updates=128,block_expected_steps=10464,new_head_expected_steps=0)))
'''

AUDIT_ALIAS=r'''
guard=lock();pub,root=publication();source=read(b/'prepared-inputs.json');config=read(b/'last-prefix/config.json')
def proc_ticks(pid):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
controller=b/'cpu-alias-controller.json'
if controller.exists():
 previous=read(controller)
 if previous['status'].startswith('ACTUAL_METADATA_ALIAS_AUDIT_RUNNING'):
  same_process=previous.get('proc_start_ticks') is not None and proc_ticks(previous['pid'])==previous['proc_start_ticks']
  assert not same_process,'Existing CPU alias audit is still active; do not duplicate'
# Authenticate index metadata before relying on policy_length for scope alias.
manifests=[]
for key in ['features','selection_features']:manifests.extend(read(verify(config[key])))
entries=[]
for path in sorted(set(manifests)):
 item=read(path)
 assert item['status']=='LOSSLESS_NATIVE_FEATURES_COMPLETE' and item['feature_layer']==26 and item['prefix_tokens']==20
 index=item['files']['index.jsonl'];verify(index)
 assert pathlib.Path(index['path']).resolve()==pathlib.Path(path).parent/'index.jsonl'
 entries.append(dict(manifest=fp(path),index=index))
audit=dict(status='NATIVE_L26_INDEX_METADATA_SHA_PASSED',manifests=len(entries),entries=entries,
           checkpoint_tensors_loaded=False,feature_tensor_files_read=False)
atomic(b/'native-index-audit.json',audit)
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
runtime=str(root/'reproduction/operations/qwen7b_warmup/runtime.sh')
imports=subprocess.run(['bash',runtime,'-c','import reproduction.operations.qwen7b_warmup.l27_readout_train; import reproduction.operations.qwen7b_warmup.probe_ablation_model; print("READOUT_CPU_IMPORT_OK")'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=45)
(b/'cpu-import.log').write_text(imports.stdout);assert imports.returncode==0,imports.stdout
cmd=['bash',runtime,'-u','-m','reproduction.operations.qwen7b_warmup.l27_readout_train','--base',str(b/'last-prefix'),'--alias-check-only']
with (b/'cpu-alias-audit.log').open('w') as log:
 process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 current=dict(pid=process.pid,proc_start_ticks=proc_ticks(process.pid),status='ACTUAL_METADATA_ALIAS_AUDIT_RUNNING',checkpoint_tensors_loaded=False,started=time.time())
 atomic(controller,current)
 try:result=process.wait(timeout=180)
 except subprocess.TimeoutExpired:
  current['status']='ACTUAL_METADATA_ALIAS_AUDIT_RUNNING_WAIT_TIMEOUT';atomic(controller,current);raise
atomic(b/'cpu-alias-controller.json',dict(pid=process.pid,status='ACTUAL_METADATA_ALIAS_AUDIT_FINISHED',returncode=result,finished=time.time()))
assert result==0,(b/'cpu-alias-audit.log').read_text()[-3000:]
alias=read(b/'last-prefix/last-scope-alias-check.json')
assert set(alias['roles'])=={'fit','diagnostic512','selection2048'} and alias['real_duplicate_GPU_execution_claimed'] is False
assert alias['roles']['fit']['trajectories']==6528*5 and alias['roles']['diagnostic512']['trajectories']==512*5 and alias['roles']['selection2048']['trajectories']==2048*5
assert alias['eligible']==all(own['empty_prefix']==0 for own in alias['roles'].values())
print(json.dumps(dict(status='ACTUAL_L27_READOUT_CPU_IMPORT_AND_ALIAS_AUDIT_COMPLETE',alias=alias,native_index_manifests=len(entries),checkpoint_tensors_loaded_on_login=False,model_tensors_loaded_on_login=False)))
'''

FINALIZE=r'''
guard=lock();pub,root=publication();source=read(b/'prepared-inputs.json');reference=read(verify(source['reference_config']))
alias_path=b/'last-prefix/last-scope-alias-check.json';alias=read(alias_path)
assert read(b/'cpu-alias-controller.json')['returncode']==0
variants={};hashes={name:item['sha256'] for name,item in pub['files'].items()}
for name,spec in variant_specs(alias['eligible']).items():
 base=b/name;base.mkdir(exist_ok=True);(base/'logs').mkdir(exist_ok=True)
 config=make_config(reference,name,spec,hashes,pub['commit'],source['model_dir'],source['model_config'],source['model_inventory'])
 path=base/'config.json'
 if path.exists():assert read(path)==config
 else:atomic(path,config)
 script=render_sbatch(name,str(base),str(root));path=base/'fit.sbatch'
 if path.exists():assert path.read_text()==script
 else:path.write_text(script)
 variants[name]=dict(base=str(base),job_name='q7b-l27r128-'+name,config=fp(base/'config.json'),sbatch=fp(path))
aliases={}
if alias['eligible']:
 alias_config=make_config(reference,'last-question-prefix',('last_valid_token','question_prefix'),hashes,pub['commit'],source['model_dir'],source['model_config'],source['model_inventory'])
 path=b/'inputs/alias-last-question-prefix-config.json'
 if path.exists():assert read(path)==alias_config
 else:atomic(path,alias_config)
 aliases['last-question-prefix']=dict(alias_of='last-prefix',proof=fp(alias_path),config=fp(path),mathematical_alias=True,duplicate_GPU_execution_claimed=False)
done=dict(status='L27_READOUT128_SUITE_PREPARED_ACTUAL_METADATA_ALIAS_AUDIT_PASSED',commit=pub['commit'],code_root=str(root),
 source_inputs=fp(b/'prepared-inputs.json'),alias_audit=fp(alias_path),native_index_audit=fp(b/'native-index-audit.json'),
 variants=variants,aliases=aliases,gpu_variants=len(variants),alias_eligible=alias['eligible'],
 source_checkpoint=source['reference_source_checkpoint'],source_plan=source['source_plan'],fit_questions=6528,fit_updates=128,
 source_block_expected_steps=10464,new_head_expected_steps=0,whole_model_lossless_resume=False,
 checkpoint_tensors_loaded_on_login=False,model_tensors_loaded_on_login=False,
 no_dependency=True,no_automatic_submission=True)
path=b/'preparation.json'
if path.exists():assert read(path)==done
else:atomic(path,done)
print(json.dumps(done))
'''

SUBMIT=r'''
guard=lock();done,pub,root=verify_prepared();name=inputs['variant']
if name in done['aliases']:
 print(json.dumps(dict(status='VERIFIED_SCOPE_ALIAS_NO_DUPLICATE_GPU_SUBMITTED',variant=name,alias=done['aliases'][name])))
else:
 assert name in done['variants'];item=done['variants'][name];base=pathlib.Path(item['base'])
 if (base/'job.id').exists():
  job=(base/'job.id').read_text().strip();assert job.isdigit()
  print(json.dumps(dict(status='EXISTING_READOUT_JOB_PRESERVED',variant=name,job_id=job,squeue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%R']),sacct=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,JobName,State,ExitCode']))))
 else:
  assert not (base/'submission-intent.json').exists(),'Unknown prior submit outcome: reconcile before another attempt'
  assert not (base/'latest.json').exists() and not (base/'progress.json').exists(),'Prior training state exists; inspect without a fresh submit'
  assert not command(['squeue','-h','-u','mou176','-n',item['job_name'],'-o','%i|%T|%R']),'Unknown same-name job: do not duplicate'
  intent=dict(status='READOUT_SUBMIT_INTENT',variant=name,job_name=item['job_name'],config=item['config'],script=item['sbatch'],source_publication=fp(b/'source-publication.json'),
    requested_resources=dict(partition='acd_u',qos='normal',gpus=1,cpus=8,memory='64G',time='01:00:00'),dependency=None,epoch_time=time.time())
  atomic(base/'submission-intent.json',intent)
  try:r=subprocess.run(['sbatch','--hold','--parsable',item['sbatch']['path']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
  except subprocess.TimeoutExpired:
   intent['status']='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPEAT';atomic(base/'submission-intent.json',intent);raise
  if r.returncode:
   intent.update(status='SBATCH_REJECTED',stderr=r.stderr[-2000:]);atomic(base/'submission-intent.json',intent);raise RuntimeError(r.stderr[-2000:])
  job=r.stdout.strip().split(';')[0];assert job.isdigit(),'Unknown sbatch reply: reconcile same-name job'
  with (base/'job.id').open('x') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
  submitted=dict(intent,status='READOUT_JOB_HELD',job_id=job);atomic(base/'submission.json',submitted)
  state=command(['squeue','-h','-j',job,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state
  command(['scontrol','release',job]);submitted.update(status='READOUT_JOB_RELEASED',epoch_time=time.time(),queue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%R']))
  atomic(base/'submission.json',submitted);print(json.dumps(submitted))
'''

STATUS=r'''
result=dict(base=str(b),exists=b.exists(),observed_epoch=time.time())
if (b/'preparation.json').exists():
 done,pub,root=verify_prepared();result.update(commit=pub['commit'],code_root=str(root),gpu_variants=done['gpu_variants'],aliases=done['aliases'],variants={})
 ids=[]
 for name,item in done['variants'].items():
  base=pathlib.Path(item['base']);own=dict(base=str(base),job_name=item['job_name'])
  if (base/'job.id').exists():
   job=(base/'job.id').read_text().strip();assert job.isdigit();own['job_id']=job;ids.append(job)
  for filename in ['progress.json','failure.json','restore-acceptance.json']:
   if (base/filename).exists():
    value=read(base/filename);own[filename]={k:value[k] for k in ['cursor','block_steps','head_steps','loss','message','epoch_time'] if k in value}
  if (base/'complete.json').exists():
   value=read(base/'complete.json');assert value['config_sha256']==item['config']['sha256']
   own['complete']=dict(status=value['status'],cursor=value['cursor'],block_steps=value['block_steps'],head_steps=value['head_steps'],metrics=value['metrics'])
  own['unknown_submission_outcome']=(base/'submission-intent.json').exists() and not (base/'job.id').exists()
  result['variants'][name]=own
 result['squeue']=command(['squeue','-h','-u','mou176','-o','%i|%j|%T|%R'])
 result['sacct']=command(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobID,JobName,State,ExitCode,Elapsed']) if ids else ''
elif (b/'cpu-alias-controller.json').exists():result['cpu_alias']=read(b/'cpu-alias-controller.json')
print(json.dumps(result))
'''


def remote_source(inputs,action):
    functions=''
    if action=='prepare':functions=inspect.getsource(make_config)
    if action=='finalize':functions='\n'.join(inspect.getsource(f) for f in (variant_specs,make_config,render_sbatch))
    prefix='inputs='+repr(inputs)+'\nSOURCE_SHA='+repr(SOURCE_SHA)+'\nTARGET='+repr(TARGET)+'\n'
    return prefix+COMMON+'\n'+functions+'\n'+dict(init=INIT,chunk=CHUNK,extract=EXTRACT,prepare=PREPARE,alias=AUDIT_ALIAS,finalize=FINALIZE,submit=SUBMIT,status=STATUS)[action]


def self_test():
    root=TARGET+'/code-0123456789ab';sizes={}
    for name in variant_specs(False):
        script=render_sbatch(name,TARGET+'/'+name,root)
        subprocess.run(['bash','-n'],input=script,text=True,check=True)
        assert '#SBATCH --gres=gpu:1\n' in script and '--time=01:00:00' in script and '--dependency' not in script
    exports='\n'.join(line for line in script.splitlines() if line.startswith('export '))
    subprocess.run(['bash','-euc',exports+'\ntest "$QWEN7B_CODE_ROOT" = '+shlex.quote(root)],check=True)
    assert len(variant_specs(True))==6 and len(variant_specs(False))==7
    values=dict(base=TARGET,parent=PARENT,q1=Q1,target=TARGET,commit='a'*40,
                bundle_sha256='b'*64,bundle_bytes=100000,chunks=25,
                files={name:dict(bytes=10000,sha256='c'*64) for name in CODE_FILES},variant='flatten')
    for action in ['init','chunk','extract','prepare','alias','finalize','submit','status']:
        own=dict(values)
        if action=='chunk':own.update(chunk_index=0,chunk_data=base64.b64encode(b'x'*4096).decode(),chunk_sha256='d'*64)
        code=remote_source(own,action);compile(code,'remote-'+action,'exec')
        if len(code.encode())>=16000:raise AssertionError('16KB control command bound exceeded: '+action)
        sizes[action]=len(code.encode())
    reference=dict(source_checkpoint=dict(path='/source/E1',sha256=SOURCE_SHA),update_budget=128,
                   global_groups=51,micro_groups=2,recipe=dict(loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.)),
                   plan=dict(path='/source/plan',sha256='e'*64),model_config=dict(path='/source/config',sha256='f'*64))
    for name,spec in variant_specs(False).items():
        config=make_config(reference,name,spec,{},'a'*40,'/model',dict(path='/model/config.json',sha256='f'*64),dict(path='/model/official-inventory.json',sha256='0'*64))
        assert config['plan']==reference['plan'] and config['source_checkpoint']==reference['source_checkpoint']
        assert config['readout']['head_seed']==1729 and config['new_head_adam_expected']==0
    result=dict(status='LOCAL_L27_DEPLOYMENT_CHECKS_PASSED',safe_sbatch=True,code_files=len(CODE_FILES),
                raw_code_chunk_max_bytes=4096,remote_command_bytes=sizes,alias_true_variants=6,
                empty_prefix_fallback_variants=7,remote_actions_executed=False)
    print(json.dumps(result));return result


def published_bundle(repo):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]
    if remote!=commit:raise RuntimeError('Publish the exact current source before prepare')
    stream=io.BytesIO();files={}
    with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name in CODE_FILES:
                raw=(repo/name).read_bytes()
                if raw!=subprocess.check_output(['git','show',commit+':'+name],cwd=repo):raise RuntimeError('Unpublished runtime file '+name)
                files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                item=tarfile.TarInfo(name);item.mode=0o444;item.size=len(raw);item.mtime=0;archive.addfile(item,io.BytesIO(raw))
    return commit,stream.getvalue(),files


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control-root',type=Path)
    parser.add_argument('--action',choices=['prepare','status','submit'],default='status')
    parser.add_argument('--variant',choices=tuple(variant_specs(False)))
    parser.add_argument('--submit-authorized',action='store_true');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    if args.control_root is None:parser.error('--control-root required')
    if args.action=='submit' and (not args.submit_authorized or not args.variant):parser.error('Submit requires --variant and --submit-authorized')
    repo=Path(__file__).resolve().parents[3]
    commit,bundle,files=published_bundle(repo) if args.action=='prepare' else ('',b'',{})
    chunks=[bundle[i:i+4096] for i in range(0,len(bundle),4096)]
    inputs=dict(base=TARGET,parent=PARENT,q1=Q1,target=TARGET,commit=commit,
                bundle_sha256=hashlib.sha256(bundle).hexdigest(),bundle_bytes=len(bundle),chunks=len(chunks),files=files,variant=args.variant)
    spec=importlib.util.spec_from_file_location('l27_deployment_control',args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(action,extra=None):
        own=dict(inputs);own.update(extra or {});source=remote_source(own,action)
        if len(source.encode())>=16000:raise RuntimeError('Bounded16KB control command exceeded')
        result=control.run('python3 -c '+shlex.quote(source),account='mou176',source_address=address,timeout=195 if action=='alias' else 55)
        rows=[json.loads(line) for line in result.stdout.decode(errors='replace').splitlines() if line.startswith('{')]
        if result.returncode or not rows:raise RuntimeError('Readout action unconfirmed; inspect actual state before retry: '+result.stderr.decode(errors='replace')[-1500:]+' '+result.stdout.decode(errors='replace')[-1500:])
        print(json.dumps(rows[-1]),flush=True);return rows[-1]
    if args.action=='prepare':
        result=call('init')
        if result['status']!='EXISTING_PREPARED_READOUT_SUITE_VERIFIED':
            for i,chunk in enumerate(chunks):call('chunk',dict(chunk_index=i,chunk_data=base64.b64encode(chunk).decode(),chunk_sha256=hashlib.sha256(chunk).hexdigest()))
            for action in ['extract','prepare','alias','finalize']:result=call(action)
    else:result=call(args.action)
    output=args.control_root/'outputs/qwen7b-warmup-20261019'/('l27-readout128-'+args.action+('-'+args.variant if args.action=='submit' else '')+'.json')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
