"""Immutable actual-G10 pilot preparation; explicit guarded submit, never auto-run.

128 original training questions /16 per update /8 updates is a SMALL pilot,
not a budget-matched replacement for the6528/B51 baseline. Both variants start
from the same sealed E1 raw/EMA/fullAdam10464; no actor updates are performed.
Only the existing512 validation questions are scored, for screening only.
"""
import argparse
import ast
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
TARGET=PARENT+'/algorithm-improvement-20261021/g10-pilot128-qualification-v1'
G10_COMPLETE=PARENT+'/algorithm-improvement-20261021/training-extraG5-128-20261021/materialized/bank/complete.json'
G10_COMPLETE_SHA='88dbf2a24c883c807fdc1ea81604a812c76868c5fb8168add948d3aa7da71962'
G10_GROUPS_SHA='ead20f2e70fdd5de91eed0378b763385319086f51e104304fb7096109443109e'
SOURCE_SHA='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'
PACKAGE='reproduction.operations.qwen7b_warmup'
VARIANTS=('random','per-question-balanced')
SEEDS=tuple('reproduction/operations/qwen7b_warmup/'+n for n in
    ('g10_qualification_entry.py','probe_g10_train.py','test_probe_g10_train.py','runtime.sh','deploy_g10_pilot_suite.py'))


def source_closure(repo):
    """All local imports, including imports inside functions, by parsed AST."""
    result=set();pending=list(SEEDS)
    while pending:
        name=pending.pop()
        if name in result:continue
        path=repo/name
        if not path.is_file():raise ValueError('Missing local runtime dependency '+name)
        result.add(name)
        if path.suffix!='.py':continue
        owner=name[:-3].split('/');package=owner[:-1]
        for node in ast.walk(ast.parse(path.read_text(),filename=name)):
            modules=[]
            if isinstance(node,ast.ImportFrom):
                if node.level:
                    prefix=package[:len(package)-node.level+1]
                    if node.module:modules.append('.'.join(prefix+[node.module]))
                    else:modules.extend('.'.join(prefix+[v.name]) for v in node.names)
                elif node.module and node.module.startswith('reproduction.'):
                    modules.append(node.module)
            elif isinstance(node,ast.Import):
                modules.extend(v.name for v in node.names if v.name.startswith('reproduction.'))
            for module in modules:
                relative=module.replace('.','/')+'.py'
                if not (repo/relative).is_file():raise ValueError('Unresolved local import '+module)
                pending.append(relative)
    return tuple(sorted(result))


def fixed_existing_g5(rows):
    """Read qid/IDs only. Existing five own trajectories are all retained."""
    groups=[];seen=set();tids=set()
    for row in rows:
        qid=row['qid'];own=list(row['trajectory_ids'])
        if not isinstance(qid,str) or qid in seen or len(own)!=5 or len(set(own))!=5 or tids.intersection(own):
            raise ValueError('Distinct actual existing G5 validation identities required')
        if any(not isinstance(t,str) or not t for t in own):raise ValueError('Nonempty trajectory IDs required')
        seen.add(qid);tids.update(own);groups.append(dict(qid=qid,trajectory_ids=own))
    digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(groups=groups,summary=dict(role='validation',selection='all-five-existing-own-trajectories',
        questions=len(groups),trajectories=5*len(groups),labels_consulted=False,
        original_fixed5_preserved=True,qid_sha256=digest([g['qid'] for g in groups]),
        selected_g5_sha256=digest(groups),screening_only=True,final_pool_read=False))


def make_config(reference,name,hashes,commit,complete,plan,evaluation):
    if name not in VARIANTS or reference['source_checkpoint']['sha256']!=SOURCE_SHA:
        raise ValueError('Pinned E1 source and declared pilot grouping required')
    config={k:copy.deepcopy(reference[k]) for k in
        ('source_checkpoint','source_config','source_bank','model_config','model_dir')}
    config.update(g10_complete=copy.deepcopy(complete),plan=copy.deepcopy(plan),
        update_budget=8,global_questions=16,smaller_question_budget_acknowledged=True,
        micro_trajectories=5,save_every=1,max_virtual_views=64,max_inter_view_pairs=4096,
        grouping=dict(mode=name,views_per_question=4,seed=1729,on_unbalance='report'),
        recipe=dict(id='g10-pilot-'+name,loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.)),
        evaluations=[copy.deepcopy(evaluation)],code_hashes=copy.deepcopy(hashes),source_commit=commit,
        fit_questions=128,fit_real_trajectories=1280,source_world_size=4,current_world_size=1,
        source_adam_steps=10464,actor_updates=0,screening_only=True,
        equal_budget_to_B51_6528_baseline_claim=False,four_rank_bitexact_resume_claim=False,
        original_question_total_weight=1.,first_actual_update_replay_required=True)
    return config


def render_sbatch(name,base,code_root):
    if name not in VARIANTS:raise ValueError('Unknown pilot variant')
    for value in (base,code_root):
        p=Path(value)
        if p.parent!=Path(TARGET) or any(c.isspace() for c in str(p)) or any(c in str(p) for c in '\r\n;`$'):
            raise ValueError('Independent safe pilot child path required')
    return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-g10p128-@@NAME@@
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
#SBATCH --output=@@BASE@@/logs/train-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE_Q@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.g10_qualification_entry --base @@BASE_Q@@
'''.replace('@@NAME@@',name).replace('@@BASE@@',base).replace('@@BASE_Q@@',shlex.quote(base)).replace('@@CODE_Q@@',shlex.quote(code_root))


COMMON=r'''
import pathlib,json,hashlib,base64,os,fcntl,time,subprocess,gzip,tarfile,io,sys,copy,shlex
from pathlib import Path
sys.dont_write_bytecode=True
b=pathlib.Path(inputs['base']);parent=pathlib.Path(inputs['parent'])
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
 assert done['commit']==pub['commit'] and done['code_root']==str(root)
 for key in ['g10_complete','source_plan','fixed_g5','cpu_qualification']:verify(done[key])
 for item in done['variants'].values():verify(item['config']);verify(item['sbatch'])
 return done,pub,root
'''

INIT=r'''
assert sha(inputs['g10_complete'])==G10_COMPLETE_SHA,'Actual sealed G10 complete SHA differs'
accepted=read(inputs['g10_complete'])
assert accepted['status']=='PILOT_ACTUAL_ORIGINAL5_PLUS_EXTRA5_G10_SHA_OWN_EM_VERIFIED' and accepted['questions']==128,'Actual complete128 G10 merge required before prepare'
b.mkdir(parents=True,exist_ok=True);guard=lock()
if (b/'preparation.json').exists():
 done,pub,root=verify_prepared();print(json.dumps(dict(status='EXISTING_G10_PILOT_PREPARATION_VERIFIED',preparation=done)))
else:
 for name in ['logs','inputs']:(b/name).mkdir(exist_ok=True)
 controller={k:inputs[k] for k in ['commit','bundle_sha256','bundle_bytes','chunks','files']}
 path=b/'upload-controller.json'
 if path.exists():assert read(path)==controller,'Different immutable preparation already exists'
 else:atomic(path,controller)
 uploads.mkdir(exist_ok=True);print(json.dumps(dict(status='G10_PILOT_UPLOAD_BOUND',chunks=inputs['chunks'])))
'''

CHUNK=r'''
guard=lock();assert not (b/'preparation.json').exists();old=read(b/'upload-controller.json')
assert old['commit']==inputs['commit'] and old['bundle_sha256']==inputs['bundle_sha256']
index=inputs['chunk_index'];assert 0<=index<inputs['chunks']
raw=base64.b64decode(inputs['chunk_data']);assert len(raw)<=4096 and hashlib.sha256(raw).hexdigest()==inputs['chunk_sha256']
part=uploads/('part-%05d'%index)
if part.exists():assert part.read_bytes()==raw
else:
 with part.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status='G10_PILOT_CODE_CHUNK_SHA_PASSED',index=index,bytes=len(raw))))
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
pub=dict(commit=inputs['commit'],remote_sha=inputs['commit'],code_root=str(code),files=inputs['files'],bundle_sha256=inputs['bundle_sha256'],active_sources_unchanged=True)
path=b/'source-publication.json'
if path.exists():assert read(path)==pub
else:atomic(path,pub)
publication();print(json.dumps(dict(status='IMMUTABLE_G10_PILOT_RUNTIME_READY',code_root=str(code),files=len(inputs['files']))))
'''

PREPARE=r'''
guard=lock();pub,root=publication()
reference_suite=parent/'algorithm-improvement-20261021/matched-loss128-v2'
accepted=read(reference_suite/'preparation.json');assert accepted['status']=='MATCHED_LOSS128_SUITE_PREPARED_ACTUAL_CPU_TESTS_PASSED'
reference=read(verify(accepted['variants']['event']['config']))
assert reference['source_checkpoint']['sha256']==SOURCE_SHA and reference['recipe']['loss_kwargs']==dict(trajectory_weight=1.,three_event_weight=1.)
for key in ['source_checkpoint','source_config','source_bank','model_config','bank','features']:verify(reference[key])
assert pathlib.Path(reference['model_dir'])/'config.json'==pathlib.Path(reference['model_config']['path'])
complete=read(inputs['g10_complete']);g10ref=fp(inputs['g10_complete']);assert g10ref['sha256']==G10_COMPLETE_SHA and complete['groups']['sha256']==G10_GROUPS_SHA
assert complete['status']=='PILOT_ACTUAL_ORIGINAL5_PLUS_EXTRA5_G10_SHA_OWN_EM_VERIFIED' and complete['questions']==128
assert complete['trajectories']==1280 and complete['extra_real_trajectories']==640 and not complete['synthetic_EM'] and complete['actor_updates']==0
assert complete['original_fixed5_preserved'] and not complete['validation_or_final_questions_generated']
groups=read(verify(complete['groups']))['train'];qids=[g['qid'] for g in groups]
assert len(groups)==128 and len(set(qids))==128 and hashlib.sha256(canonical(qids)).hexdigest()==complete['ordered_qid_sha256']
assert all(len(g['trajectory_ids'])==len(set(g['trajectory_ids']))==len(g['labels'])==10 and all(type(y) is int and y in (0,1) for y in g['labels']) for g in groups)
assert len({tid for g in groups for tid in g['trajectory_ids']})==1280
for key in ['original_G5_bank','original_feature_manifest','extra_G5_feature_sources']:verify(complete[key])
for ref in complete['extra_shards']:verify(ref)
diagnostic=read(reference['bank']['path'])['diagnostic'];assert len(diagnostic)==512
assert not set(qids)&{g['qid'] for g in diagnostic} and not {g['question_hash'] for g in groups}&{g['question_hash'] for g in diagnostic}
fixed=fixed_existing_g5(diagnostic);fixedpath=b/'inputs/diagnostic512-fixed-g5.json'
if fixedpath.exists():assert read(fixedpath)==fixed
else:atomic(fixedpath,fixed)
plan=dict(cohort_qids=qids,batches=[qids[i:i+16] for i in range(0,128,16)],batch_epochs=[0]*8,
 ordered_qid_sha256=complete['ordered_qid_sha256'],original_questions=128,global_original_questions=16,
 updates=8,smaller_question_budget_acknowledged=True,screening_only=True,labels_used_for_plan=False,
 rule='sealed accepted G10 order, contiguous16-question disjoint batches, all128 exactly once')
planpath=b/'inputs/plan.json'
if planpath.exists():assert read(planpath)==plan
else:atomic(planpath,plan)
evaluation=dict(name='diagnostic512',role='validation',bank=reference['bank'],split='diagnostic',
 features=reference['features'],fixed_g5=fp(fixedpath),screening_only=True,previously_observed=True,final_pool_read=False)
hashes={name:item['sha256'] for name,item in pub['files'].items()};variants={}
for name in VARIANTS:
 base=b/name;base.mkdir(exist_ok=True);(base/'logs').mkdir(exist_ok=True)
 config=make_config(reference,name,hashes,pub['commit'],g10ref,fp(planpath),evaluation)
 path=base/'config.json'
 if path.exists():assert read(path)==config
 else:atomic(path,config)
 script=render_sbatch(name,str(base),str(root));path=base/'fit.sbatch'
 if path.exists():assert path.read_text()==script
 else:path.write_text(script)
 variants[name]=dict(base=str(base),job_name='q7b-g10p128-'+name,config=fp(base/'config.json'),sbatch=fp(path))
source=dict(reference_preparation=fp(reference_suite/'preparation.json'),reference_event_config=accepted['variants']['event']['config'],
 g10_complete=g10ref,source_plan=fp(planpath),fixed_g5=fp(fixedpath),variants=variants,
 checkpoint_tensors_loaded_on_login=False,model_tensors_loaded_on_login=False)
path=b/'prepared-inputs.json'
if path.exists():assert read(path)==source
else:atomic(path,source)
print(json.dumps(dict(status='G10_PILOT_INPUTS_BOUND',questions=128,updates=8,original_questions_per_update=16,variants=list(variants))))
'''

CPU=r'''
guard=lock();pub,root=publication();source=read(b/'prepared-inputs.json')
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
runtime=str(root/'reproduction/operations/qwen7b_warmup/runtime.sh')
args=['bash',runtime,'-m','unittest','reproduction.operations.qwen7b_warmup.test_probe_g10_train']
result=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=120)
(b/'actual-cpu-tests.log').write_text(result.stdout);assert result.returncode==0,result.stdout[-3000:]
proof=dict(status='ACTUAL_G10_PILOT_CPU_TESTS_PASSED_GPU_QUALIFICATION_NOT_YET_RUN',code_commit=pub['commit'],
 log=fp(b/'actual-cpu-tests.log'),checkpoint_tensors_loaded_on_login=False,model_tensors_loaded_on_login=False,
 actual_GPU_next_update_replay_passed=False,completed_real_GPU_updates=0)
atomic(b/'cpu-qualification.json',proof)
done=dict(status='G10_PILOT128_PREPARED_ACTUAL_CPU_TESTS_PASSED_GPU_NOT_YET_QUALIFIED',commit=pub['commit'],code_root=str(root),
 g10_complete=source['g10_complete'],source_plan=source['source_plan'],fixed_g5=source['fixed_g5'],
 cpu_qualification=fp(b/'cpu-qualification.json'),variants=source['variants'],fit_questions=128,fit_real_trajectories=1280,
 global_original_questions=16,updates=8,views_per_question=4,on_unbalance='report',source_adam_steps=10464,
 source_raw_ema_adam_preserved=True,source_world_size=4,current_world_size=1,four_rank_bitexact_resume_claim=False,
 smaller_question_budget_acknowledged=True,equal_budget_to_B51_6528_baseline_claim=False,
 evaluation_role='previously viewed512 diagnostic screening only',final_pool_read=False,
 actor_updates=0,save_every=1,no_dependency=True,no_automatic_submission=True)
path=b/'preparation.json'
if path.exists():assert read(path)==done
else:atomic(path,done)
print(json.dumps(done))
'''

SUBMIT=r'''
guard=lock();done,pub,root=verify_prepared();name=inputs['variant'];assert name in done['variants']
item=done['variants'][name];base=pathlib.Path(item['base'])
if (base/'job.id').exists():
 job=(base/'job.id').read_text().strip();assert job.isdigit()
 print(json.dumps(dict(status='EXISTING_G10_PILOT_JOB_PRESERVED',variant=name,job_id=job,squeue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%R']),sacct=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,JobName,State,ExitCode']))))
else:
 assert not (base/'submission-intent.json').exists(),'Unknown prior submit outcome: reconcile without repeating'
 assert not (base/'latest.json').exists() and not (base/'progress.json').exists(),'Prior training state: inspect without fresh submit'
 assert not command(['squeue','-h','-u','mou176','-n',item['job_name'],'-o','%i|%T|%R']),'Unknown same-name queued job'
 intent=dict(status='G10_PILOT_SUBMIT_INTENT',variant=name,job_name=item['job_name'],config=item['config'],script=item['sbatch'],
  requested_resources=dict(partition='acd_u',qos='normal',gpus=1,cpus=8,memory='64G',time='01:00:00'),dependency=None,time=time.time())
 atomic(base/'submission-intent.json',intent)
 try:r=subprocess.run(['sbatch','--hold','--parsable',item['sbatch']['path']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 except subprocess.TimeoutExpired:
  intent['status']='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPEAT';atomic(base/'submission-intent.json',intent);raise
 if r.returncode:
  intent.update(status='SBATCH_REJECTED',stderr=r.stderr[-2000:]);atomic(base/'submission-intent.json',intent);raise RuntimeError(r.stderr[-2000:])
 job=r.stdout.strip().split(';')[0];assert job.isdigit(),'Unknown sbatch reply; reconcile same-name job'
 with (base/'job.id').open('x') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
 submitted=dict(intent,status='G10_PILOT_JOB_HELD',job_id=job);atomic(base/'submission.json',submitted)
 state=command(['squeue','-h','-j',job,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state
 command(['scontrol','release',job]);submitted.update(status='G10_PILOT_JOB_RELEASED',time=time.time(),queue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%R']))
 atomic(base/'submission.json',submitted);print(json.dumps(submitted))
'''

STATUS=r'''
result=dict(base=str(b),exists=b.exists(),observed_epoch=time.time(),actual_g10_complete_exists=pathlib.Path(inputs['g10_complete']).exists())
if (b/'preparation.json').exists():
 done,pub,root=verify_prepared();result.update(commit=pub['commit'],code_root=str(root),variants={});ids=[]
 for name,item in done['variants'].items():
  base=pathlib.Path(item['base']);own=dict(base=str(base),job_name=item['job_name'])
  if (base/'job.id').exists():
   job=(base/'job.id').read_text().strip();assert job.isdigit();own['job_id']=job;ids.append(job)
  for filename in ['progress.json','failure.json','next-update-replay.json','interrupted.json','gpu-memory.json']:
   if (base/filename).exists():
    value=read(base/filename);own[filename]={k:value[k] for k in ['status','cursor','steps','time','loss','message','original_questions','unique_real_trajectories','virtual_views','grouping','recomputed_logits_bit_identical','mixed_balanced_accuracy','mixed_recall','nonmixed_recall','source_steps','candidate_steps','peak_allocated_bytes','peak_reserved_bytes','device','exit_normally'] if k in value}
  if (base/'complete.json').exists():
   value=read(base/'complete.json');assert value['config_sha256']==item['config']['sha256']
   own['complete']=dict(status=value['status'],cursor=value['cursor'],steps=value['steps'],metrics=value['metrics'])
  own['unknown_submission_outcome']=(base/'submission-intent.json').exists() and not (base/'job.id').exists()
  result['variants'][name]=own
 result['squeue']=command(['squeue','-h','-u','mou176','-o','%i|%j|%T|%R'])
 result['sacct']=command(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobID,JobName,State,ExitCode,Elapsed']) if ids else ''
print(json.dumps(result))
'''


def remote_source(inputs,action):
    functions=''
    if action=='prepare':functions='\n'.join(inspect.getsource(f) for f in (fixed_existing_g5,make_config,render_sbatch))
    prefix='inputs='+repr(inputs)+'\nSOURCE_SHA='+repr(SOURCE_SHA)+'\nTARGET='+repr(TARGET)+'\nVARIANTS='+repr(VARIANTS)+'\nG10_COMPLETE_SHA='+repr(G10_COMPLETE_SHA)+'\nG10_GROUPS_SHA='+repr(G10_GROUPS_SHA)+'\n'
    return prefix+COMMON+'\n'+functions+'\n'+dict(init=INIT,chunk=CHUNK,extract=EXTRACT,prepare=PREPARE,cpu=CPU,submit=SUBMIT,status=STATUS)[action]


def published_bundle(repo):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]
    if remote!=commit:raise RuntimeError('Publish the exact source before prepare')
    stream=io.BytesIO();files={}
    with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name in source_closure(repo):
                raw=(repo/name).read_bytes()
                if raw!=subprocess.check_output(['git','show',commit+':'+name],cwd=repo):raise RuntimeError('Unpublished runtime dependency '+name)
                files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                member=tarfile.TarInfo(name);member.mode=0o444;member.size=len(raw);member.mtime=0;archive.addfile(member,io.BytesIO(raw))
    return commit,stream.getvalue(),files


def self_test():
    repo=Path(__file__).resolve().parents[3];closure=source_closure(repo)
    assert {Path(n).name for n in closure}>={'probe_g10_train.py','model.py','native_capture.py','download_model.py','runtime.sh'}
    for name in VARIANTS:
        script=render_sbatch(name,TARGET+'/'+name,TARGET+'/code-'+'a'*12)
        subprocess.run(['bash','-n'],input=script,text=True,check=True)
        assert '--dependency' not in script and '#SBATCH --gres=gpu:1\n' in script and '--time=01:00:00' in script
    class NoLabels(dict):
        def __getitem__(self,key):
            if key not in ('qid','trajectory_ids'):raise AssertionError('Label-aware fixed5 plan')
            return super().__getitem__(key)
    rows=[NoLabels(qid='q',trajectory_ids=['t'+str(i) for i in range(5)],labels=[0,1,1,0,0])]
    assert fixed_existing_g5(rows)['summary']['labels_consulted'] is False
    reference=dict(source_checkpoint=dict(path='/source/E1',sha256=SOURCE_SHA),source_config=dict(path='/source/config',sha256='1'*64),
        source_bank=dict(path='/source/bank',sha256='2'*64),model_config=dict(path='/model/config.json',sha256='3'*64),model_dir='/model')
    configs=[make_config(reference,n,{},'a'*40,{},dict(path='/plan',sha256='4'*64),{}) for n in VARIANTS]
    assert configs[0]['plan']==configs[1]['plan'] and configs[0]['source_checkpoint']==configs[1]['source_checkpoint']
    assert all(c['global_questions']==16 and c['update_budget']==8 and c['smaller_question_budget_acknowledged'] and c['save_every']==1 for c in configs)
    values=dict(base=TARGET,parent=PARENT,target=TARGET,g10_complete=G10_COMPLETE,commit='a'*40,bundle_sha256='b'*64,
        bundle_bytes=100000,chunks=25,files={n:dict(bytes=10000,sha256='c'*64) for n in closure},variant='random')
    sizes={}
    for action in ['init','chunk','extract','prepare','cpu','submit','status']:
        own=dict(values)
        if action=='chunk':own.update(chunk_index=0,chunk_data=base64.b64encode(b'x'*4096).decode(),chunk_sha256='d'*64)
        source=remote_source(own,action);compile(source,'remote-'+action,'exec')
        shell='python3 -c '+shlex.quote(source);sizes[action]=len(shell.encode())
        if sizes[action]>=16000:raise AssertionError('Control command exceeds16KB: '+action)
    result=dict(status='LOCAL_G10_PILOT_DEPLOYMENT_CHECKS_PASSED',code_files=len(closure),raw_chunk_bytes=4096,
        control_command_bytes=sizes,safe_sbatch=True,label_blind_existing_fixed5=True,remote_actions_executed=False)
    print(json.dumps(result));return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control-root',type=Path)
    parser.add_argument('--action',choices=['prepare','status','submit'],default='status')
    parser.add_argument('--variant',choices=VARIANTS);parser.add_argument('--submit-authorized',action='store_true')
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test();return
    if args.control_root is None:parser.error('--control-root required')
    if args.action=='submit' and (not args.variant or not args.submit_authorized):parser.error('Submit requires --variant and --submit-authorized')
    repo=Path(__file__).resolve().parents[3]
    commit,bundle,files=published_bundle(repo) if args.action=='prepare' else ('',b'',{})
    chunks=[bundle[i:i+4096] for i in range(0,len(bundle),4096)]
    inputs=dict(base=TARGET,parent=PARENT,target=TARGET,g10_complete=G10_COMPLETE,commit=commit,
        bundle_sha256=hashlib.sha256(bundle).hexdigest(),bundle_bytes=len(bundle),chunks=len(chunks),files=files,variant=args.variant)
    spec=importlib.util.spec_from_file_location('g10_pilot_control',args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(action,extra=None):
        own=dict(inputs);own.update(extra or {});command='python3 -c '+shlex.quote(remote_source(own,action))
        if len(command.encode())>=16000:raise RuntimeError('Bounded16KB control command exceeded')
        result=control.run(command,account='mou176',source_address=address,timeout=140 if action=='cpu' else 55)
        rows=[json.loads(line) for line in result.stdout.decode(errors='replace').splitlines() if line.startswith('{')]
        if result.returncode or not rows:raise RuntimeError('Pilot action unconfirmed; inspect actual state before retry: '+result.stderr.decode(errors='replace')[-1500:]+' '+result.stdout.decode(errors='replace')[-1500:])
        print(json.dumps(rows[-1]),flush=True);return rows[-1]
    if args.action=='prepare':
        result=call('init')
        if result['status']!='EXISTING_G10_PILOT_PREPARATION_VERIFIED':
            for i,chunk in enumerate(chunks):call('chunk',dict(chunk_index=i,chunk_data=base64.b64encode(chunk).decode(),chunk_sha256=hashlib.sha256(chunk).hexdigest()))
            for action in ['extract','prepare','cpu']:result=call(action)
    else:result=call(args.action)
    output=args.control_root/'outputs/qwen7b-warmup-20261019'/('g10-pilot128-'+args.action+('-'+args.variant if args.action=='submit' else '')+'.json')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
