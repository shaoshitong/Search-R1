"""Prepare/status ONLY for matched-loss128 followups; never submits a job.

The three accepted baseline budgets must finish before preparing a selected
group objective. All followups start at the SAME sealed E1 raw/EMA/Adam/RNG,
reuse the exact baseline source refs/plan/cohort128 updates, and have separate
code/run directories. The parent task owns explicit guarded submission.
"""
import argparse
import base64
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import shlex
import subprocess
import zlib
from .deploy_matched_loss_suite import NAMES,render_matched_sbatch

SOURCE_SHA='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'
ASSET_KEYS=('source_checkpoint','source_config','source_bank','model_config',
            'bank','features','selection_bank','selection_features','plan')
CRITICAL_FILES=tuple('reproduction/operations/qwen7b_warmup/'+name for name in
                    ('probe_ablation_train.py','probe_ablation_loss.py','model.py',
                     'packed_features.py','native_capture.py','download_model.py','runtime.sh'))


def recipe_specs(group_base,include_hard_negative=False):
    if group_base not in ('trajectory','joint','event'):raise ValueError('Unknown baseline group objective')
    group=dict(trajectory_weight=1.)
    if group_base=='joint':group['binary_mixed_weight']=1.
    if group_base=='event':group['three_event_weight']=1.
    extras=[('within1',dict(within_rank_weight=1.)),
            ('inter1',dict(inter_rank_weight=1.)),
            ('bothrank1',dict(within_rank_weight=1.,inter_rank_weight=1.)),
            ('mean1',dict(stat_mean_weight=1.)),
            ('var5',dict(stat_variance_weight=5.)),
            ('mean1-var5',dict(stat_mean_weight=1.,stat_variance_weight=5.)),
            ('bothrank1-mean1-var5',dict(within_rank_weight=1.,inter_rank_weight=1.,stat_mean_weight=1.,stat_variance_weight=5.))]
    if include_hard_negative:extras.append(('inter1-hard16',dict(inter_rank_weight=1.,inter_hard_negative_k=16)))
    return {name:dict(group,**weights) for name,weights in extras}


def job_name(group_base,name):
    if group_base not in ('trajectory','joint','event') or name not in recipe_specs(group_base,True):
        raise ValueError('Unknown followup recipe identity')
    return 'q7b-loss128f-'+group_base+'-'+name


def render_followup_sbatch(group_base,name,base,root):
    for value in (base,root):
        text=str(value)
        if (not text.startswith('/data/user/jxu957/experiments/') or
                not re.fullmatch(r'/[A-Za-z0-9_./-]+',text) or '..' in Path(text).parts):
            raise ValueError('Safe absolute remote shared-storage path required')
    label=job_name(group_base,name)
    script=render_matched_sbatch(group_base+'-'+name,base,root)
    return script.replace('#SBATCH --job-name=q7b-loss128-'+group_base+'-'+name,
                          '#SBATCH --job-name='+label)


def validate_reference_config(reference,group_base):
    expected=dict(trajectory_weight=1.)
    if group_base=='joint':expected['binary_mixed_weight']=1.
    if group_base=='event':expected['three_event_weight']=1.
    if group_base not in ('trajectory','joint','event'):raise ValueError('Unknown baseline')
    if (reference['recipe']['id']!=group_base or reference['recipe']['loss_kwargs']!=expected
            or reference['source_checkpoint']['sha256']!=SOURCE_SHA
            or reference['update_budget']!=128 or reference['global_groups']!=51
            or reference['micro_groups']!=2):
        raise ValueError('Reference is not the accepted matched-loss128-v2 baseline')
    for key in ASSET_KEYS:
        item=reference[key]
        if not Path(item['path']).is_absolute() or not re.fullmatch('[0-9a-f]{64}',item['sha256']):
            raise ValueError('Unbound reference asset '+key)


def make_variant_config(reference,group_base,name,hashes,commit,reference_config):
    validate_reference_config(reference,group_base)
    kwargs=recipe_specs(group_base,True)[name]
    for rel in CRITICAL_FILES:
        if hashes[rel]!=reference['code_hashes'][rel]:
            raise ValueError('Training runtime drift from accepted baseline: '+rel)
    config=copy.deepcopy(reference)
    config.update(recipe=dict(id=group_base+'-'+name,loss_kwargs=kwargs),
                  code_hashes=copy.deepcopy(hashes),source_commit=commit,
                  followup_reference_config=copy.deepcopy(reference_config),
                  followup_group_base=group_base,
                  followup_start='sealed-E1; NOT the completed128-step result',
                  screening_only=True,no_automatic_submission=True)
    return config


COMMON=r'''
import pathlib,base64,copy,hashlib,json,zlib,os,subprocess,time,fcntl,re
from pathlib import Path
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021'
reference_suite=A/'matched-loss128-v2'
suite=A/'matched-loss128-followup-v1'/group_base
root=A/('followup-repo-'+commit[:12])
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def verify(item):
 p=pathlib.Path(item['path']);assert p.is_absolute() and sha(p)==item['sha256'],'Asset/config SHA mismatch: '+str(p);return p
def atomic(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def command(args,timeout=30):
 result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout)
 assert result.returncode==0,result.stdout+result.stderr
 return result.stdout.strip()
def verify_publication(pub):
 for rel,value in pub['files'].items():assert sha(pathlib.Path(pub['code_root'])/rel)==value,'Published source drift: '+rel
'''

PREPARE=r'''
suite.mkdir(parents=True,exist_ok=True)
with (suite/'prepare.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 specs=recipe_specs(group_base,include_hard_negative)
 if (suite/'preparation.json').exists():
  done=read(suite/'preparation.json');pub=read(suite/'source-publication.json');verify_publication(pub)
  assert done['group_base']==group_base and set(done['variants'])==set(specs),'Existing immutable suite uses different recipes; choose a new suite version'
  verify(done['reference_config']);verify(done['plan'])
  for item in done['variants'].values():
   verify(item['config']);verify(item['sbatch'])
  print(json.dumps(dict(status='EXISTING_FOLLOWUP_SUITE_VERIFIED',preparation=done)))
 else:
  accepted=read(reference_suite/'preparation.json');old_pub=read(reference_suite/'source-publication.json');verify_publication(old_pub)
  assert accepted['status']=='MATCHED_LOSS128_SUITE_PREPARED_ACTUAL_CPU_TESTS_PASSED'
  baselines={};baseline_receipts={};common_assets=None
  for baseline in ['trajectory','joint','event']:
   item=accepted['variants'][baseline];config_path=verify(item['config']);config=read(config_path)
   validate_reference_config(config,baseline)
   own_assets={key:config[key] for key in ASSET_KEYS}
   if common_assets is None:common_assets=own_assets
   else:assert own_assets==common_assets,'Baseline source/plan bindings differ'
   complete_path=pathlib.Path(item['base'])/'complete.json';complete=read(complete_path)
   assert complete['status']=='MATCHED_PROBE_LOSS_ABLATION_COMPLETE' and complete['cursor']==128 and complete['steps']==10592,'All three real128-step baseline budgets must complete first'
   assert complete['config_sha256']==item['config']['sha256']
   checkpoint=complete['checkpoint'];assert checkpoint['cursor']==128 and checkpoint['steps']==10592 and checkpoint['config_sha256']==item['config']['sha256']
   baselines[baseline]=config;baseline_receipts[baseline]=ref(complete_path)
  reference=baselines[group_base];reference_ref=accepted['variants'][group_base]['config']
  for item in common_assets.values():verify(item)
  plan=read(reference['plan']['path']);flat=[q for batch in plan['batches'] for q in batch]
  assert len(plan['batches'])==128 and all(len(batch)==51 for batch in plan['batches'])
  assert len(flat)==len(set(flat))==6528 and set(flat)==set(plan['cohort_qids'])
  for rel in CRITICAL_FILES:assert hashes[rel]==reference['code_hashes'][rel],'Training runtime changed since baseline; do not silently compare'
  files=json.loads(zlib.decompress(base64.b64decode(packed)));assert set(files)==set(hashes)
  root.mkdir(parents=True,exist_ok=True)
  for rel,encoded in files.items():
   assert not pathlib.Path(rel).is_absolute() and '..' not in pathlib.Path(rel).parts
   raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==hashes[rel]
   dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():assert dest.read_bytes()==raw,'Unknown independent source conflict'
   else:dest.write_bytes(raw)
  for rel in ['reproduction','reproduction/operations','reproduction/operations/qwen7b_warmup','reproduction/prefix20_group_joint100']:(root/rel/'__init__.py').touch(exist_ok=True)
  pub=dict(commit=commit,remote_sha=commit,code_root=str(root),files=hashes,time=time.time());atomic(suite/'source-publication.json',pub)
  env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
  tested=subprocess.run(['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','unittest','-v','reproduction.operations.qwen7b_warmup.test_probe_ablation_loss','reproduction.operations.qwen7b_warmup.test_probe_ablation_train','reproduction.operations.qwen7b_warmup.test_deploy_followup_loss_suite'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=90)
  (suite/'actual-cpu-tests.log').write_text(tested.stdout);assert tested.returncode==0,tested.stdout
  variants={}
  for name in specs:
   base=suite/name;base.mkdir(exist_ok=True);(base/'logs').mkdir(exist_ok=True)
   config=make_variant_config(reference,group_base,name,hashes,commit,reference_ref)
   if (base/'config.json').exists():assert read(base/'config.json')==config,'Conflicting followup config; never overwrite'
   else:atomic(base/'config.json',config)
   script=render_followup_sbatch(group_base,name,base,root)
   target=base/'fit.sbatch'
   if target.exists():assert target.read_text()==script,'Conflicting followup sbatch; never overwrite'
   else:target.write_text(script)
   variants[name]=dict(base=str(base),job_name=job_name(group_base,name),config=ref(base/'config.json'),sbatch=ref(target))
  done=dict(status='FOLLOWUP_LOSS128_SUITE_PREPARED_CPU_CHECKS_PASSED',group_base=group_base,
       source_commit=commit,code_root=str(root),reference_suite=str(reference_suite),
       reference_config=reference_ref,reference_baseline_completions=baseline_receipts,
       source_checkpoint=reference['source_checkpoint'],source_adam_steps=10464,
       plan=reference['plan'],source_assets=common_assets,fit_updates_per_variant=128,
       fit_questions=6528,fit_trajectories=32640,global_groups=51,variants=variants,
       start_from_sealed_E1=True,continues128step_result=False,
       cpu_tests_log=ref(suite/'actual-cpu-tests.log'),actor_updates=0,
       no_dependency=True,no_automatic_submission=True,time=time.time())
  atomic(suite/'preparation.json',done);print(json.dumps(done))
'''

STATUS=r'''
if not (suite/'preparation.json').exists():
 print(json.dumps(dict(status='FOLLOWUP_NOT_PREPARED',group_base=group_base,suite=str(suite),time=time.time())))
else:
 done=read(suite/'preparation.json');pub=read(suite/'source-publication.json');verify_publication(pub)
 queue=command(['squeue','-h','-u','mou176','-o','%i|%j|%T|%R'])
 ids=[];variants={}
 for name,item in done['variants'].items():
  verify(item['config']);verify(item['sbatch']);base=pathlib.Path(item['base'])
  own=dict(base=str(base),job_name=item['job_name'],config_sha256=item['config']['sha256'],
           live_queue=[line for line in queue.splitlines() if line.split('|')[1]==item['job_name']],
           submission_reconciliation_required=(base/'submit-intent.json').exists() and not (base/'job.id').exists())
  if (base/'job.id').exists():
   job=(base/'job.id').read_text().strip();assert job.isdigit();own['job_id']=job;ids.append(job)
  if (base/'progress.json').exists():
   progress=read(base/'progress.json');own['progress']={k:progress.get(k) for k in ['cursor','steps','budget','loss','mixed_balanced_accuracy','mixed_recall','nonmixed_recall','epoch_time']}
  if (base/'failure.json').exists():own['failure']=read(base/'failure.json')
  if (base/'complete.json').exists():
   complete=read(base/'complete.json');assert complete['config_sha256']==item['config']['sha256']
   own['complete']=dict(status=complete['status'],cursor=complete['cursor'],steps=complete['steps'],epoch_time=complete['epoch_time'])
   own['metrics']={}
   for split,value in complete['metrics'].items():
    pool=value['pool'];selected=value['selected512']
    own['metrics'][split]=dict(mixed_balanced_accuracy=pool['mixed_balanced_accuracy'],mixed_recall=pool['mixed_recall'],nonmixed_recall=pool['nonmixed_recall'],nonmixed_false_positive_rate=pool['nonmixed_false_positive_rate'],mixed_roc_auc=pool['mixed_roc_auc'],selected512_mixed=selected['mixed_questions'],selected512_count=selected['questions'],selected512_mixed_fraction=selected['mixed_fraction'],selected512_K=selected['k_histogram'])
  variants[name]=own
 accounting=command(['sacct','-n','-P','-j',','.join(ids),'-o','JobID,JobName,State,ExitCode']) if ids else ''
 print(json.dumps(dict(status='FRESH_FOLLOWUP_SUITE_STATUS',group_base=group_base,suite=str(suite),variants=variants,accounting=accounting,source_commit=pub['commit'],time=time.time())))
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--control-root',type=Path,required=True)
    parser.add_argument('--action',choices=['prepare','status'],default='status')
    parser.add_argument('--group-base',choices=['trajectory','joint','event'],required=True)
    parser.add_argument('--include-hard-negative',action='store_true')
    args=parser.parse_args();repo=Path(__file__).resolve().parents[3]
    commit='';hashes={};packed='';chunks=[]
    if args.action=='prepare':
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]
        if remote!=commit:raise ValueError('Publish current source before preparing remote followups')
        payload={}
        names=list(dict.fromkeys(NAMES+['deploy_followup_loss_suite.py','test_deploy_followup_loss_suite.py']))
        for rel in ['reproduction/operations/qwen7b_warmup/'+name for name in names]+['reproduction/prefix20_group_joint100/loss.py']:
            raw=(repo/rel).read_bytes()
            if raw!=subprocess.check_output(['git','show',commit+':'+rel],cwd=repo):raise ValueError('Unpublished '+rel)
            payload[rel]=base64.b64encode(raw).decode();hashes[rel]=hashlib.sha256(raw).hexdigest()
        packed=base64.b64encode(zlib.compress(json.dumps(payload).encode(),9)).decode()
        chunks=[packed[i:i+16000] for i in range(0,len(packed),16000)]
    spec=importlib.util.spec_from_file_location('followup_control',args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    stage='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/source-staging/followup-'+commit
    if args.action=='prepare':
        for index,chunk in enumerate(chunks):
            upload='import pathlib,hashlib; p=pathlib.Path('+repr(stage)+')/'+repr('part-%04d'%index)+'; d='+repr(chunk)+'; p.parent.mkdir(parents=True,exist_ok=True); assert not p.exists() or p.read_text()==d; p.write_text(d); assert hashlib.sha256(p.read_text().encode()).hexdigest()=='+repr(hashlib.sha256(chunk.encode()).hexdigest())
            result=control.run('python3 -c '+shlex.quote(upload),account='mou176',source_address=address,timeout=35)
            if result.returncode:raise RuntimeError('Code-only followup staging incomplete; no submission: '+result.stderr.decode()[-1000:])
    expression="''.join(__import__('pathlib').Path(%r).joinpath('part-%%04d'%%i).read_text() for i in range(%d))"%(stage,len(chunks)) if args.action=='prepare' else "''"
    prefix='commit='+repr(commit)+'\ngroup_base='+repr(args.group_base)+'\ninclude_hard_negative='+repr(args.include_hard_negative)+'\nhashes='+repr(hashes)+'\npacked='+expression+'\nSOURCE_SHA='+repr(SOURCE_SHA)+'\nASSET_KEYS='+repr(ASSET_KEYS)+'\nCRITICAL_FILES='+repr(CRITICAL_FILES)+'\n'
    functions='\n'.join(inspect.getsource(f) for f in (render_matched_sbatch,recipe_specs,job_name,render_followup_sbatch,validate_reference_config,make_variant_config))
    code=prefix+COMMON+'\n'+functions+'\n'+(PREPARE if args.action=='prepare' else STATUS)
    if len(code.encode())>=64000:raise ValueError('Control script exceeds bounded code-only payload')
    result=control.run('python3 -c '+shlex.quote(code),account='mou176',source_address=address,timeout=150 if args.action=='prepare' else 45)
    values=[json.loads(line) for line in result.stdout.decode().splitlines() if line.startswith('{')]
    if result.returncode or not values:raise RuntimeError('Followup '+args.action+' unconfirmed; no submission performed: '+result.stderr.decode()[-1500:]+' '+result.stdout.decode()[-1500:])
    output=args.control_root/'outputs/qwen7b-warmup-20261019'/('followup-loss128-'+args.group_base+'-'+args.action+'.json')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(values[-1],indent=2)+'\n')
    print(json.dumps(values[-1]))


if __name__=='__main__':main()
