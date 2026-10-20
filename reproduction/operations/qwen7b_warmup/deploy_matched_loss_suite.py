"""Prepare/publish matched loss variants, then submit explicitly one at a time.

Code/config/small audits only cross the control connection; all assets stay HPC3.
"""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path

NAMES=['probe_ablation_train.py','test_probe_ablation_train.py','probe_ablation_loss.py','test_probe_ablation_loss.py','model.py','packed_features.py','native_capture.py','download_model.py','balanced_sampling.py','probe_ablation_model.py','test_probe_ablation_model.py','g10_grouping.py','test_g10_grouping.py','runtime.sh','deploy_matched_loss_suite.py']

def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--action',choices=['prepare','submit'],default='prepare');p.add_argument('--variant',choices=['trajectory','joint','event'],default='trajectory');a=p.parse_args();repo=Path(__file__).resolve().parents[3]
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();assert subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]==commit
 payload={};hashes={}
 for name in NAMES:
  rel='reproduction/operations/qwen7b_warmup/'+name;raw=(repo/rel).read_bytes();assert raw==subprocess.check_output(['git','show',commit+':'+rel],cwd=repo),'Unpublished '+rel
  payload[rel]=base64.b64encode(raw).decode();hashes[rel]=hashlib.sha256(raw).hexdigest()
 packed=base64.b64encode(zlib.compress(json.dumps(payload).encode(),9)).decode()
 spec=importlib.util.spec_from_file_location('control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 prefix='commit='+repr(commit)+'\npacked='+repr(packed)+'\nhashes='+repr(hashes)+'\nvariant='+repr(a.variant)+'\n'
 code=prefix+COMMON+(PREPARE if a.action=='prepare' else SUBMIT);assert len(code.encode())<64000
 r=m.run('python3 -c '+shlex.quote(code),account='mou176',source_address=address,timeout=150 if a.action=='prepare' else 45)
 values=[json.loads(x) for x in r.stdout.decode().splitlines() if x.startswith('{')]
 out=a.control_root/'outputs/qwen7b-warmup-20261019'/('matched-loss-'+a.action+('-'+a.variant if a.action=='submit' else '')+'.json')
 if r.returncode or not values:
  out.with_suffix('.failure.txt').write_text(r.stderr.decode()[-3000:]+'\n'+r.stdout.decode()[-3000:]);raise RuntimeError('Action unconfirmed; inspect actual suite/job before retry: '+r.stderr.decode()[-1000:]+' '+r.stdout.decode()[-1500:])
 out.write_text(json.dumps(values[-1],indent=2)+'\n');print(json.dumps(values[-1]))

COMMON=r'''
import pathlib,base64,hashlib,json,zlib,os,subprocess,time,fcntl
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';suite=A/'matched-loss128';root=A/('repo-'+commit[:12]);suite.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def atomic(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def command(args,timeout=30):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout);assert p.returncode==0,p.stdout+p.stderr;return p.stdout.strip()
'''
PREPARE=r'''
with (suite/'prepare.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (suite/'preparation.json').exists():
  done=read(suite/'preparation.json');pub=read(suite/'source-publication.json')
  for rel,digest in pub['files'].items():assert sha(pathlib.Path(pub['code_root'])/rel)==digest
  for item in done['variants'].values():assert sha(item['config']['path'])==item['config']['sha256']
  print(json.dumps(dict(status='EXISTING_MATCHED_LOSS_SUITE_VERIFIED',preparation=done)))
 else:
  root.mkdir(parents=True,exist_ok=True);files=json.loads(zlib.decompress(base64.b64decode(packed)))
  assert set(files)==set(hashes)
  for rel,encoded in files.items():
   assert not pathlib.Path(rel).is_absolute() and '..' not in pathlib.Path(rel).parts
   raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==hashes[rel];dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():assert dest.read_bytes()==raw,'Unknown source conflict'
   else:dest.write_bytes(raw)
  for rel in ['reproduction','reproduction/operations','reproduction/operations/qwen7b_warmup']:(root/rel/'__init__.py').touch(exist_ok=True)
  pub=dict(commit=commit,remote_sha=commit,code_root=str(root),files=hashes,time=time.time());atomic(suite/'source-publication.json',pub)
  env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
  test_run=subprocess.run(['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','unittest','-v','reproduction.operations.qwen7b_warmup.test_probe_ablation_loss','reproduction.operations.qwen7b_warmup.test_probe_ablation_train','reproduction.operations.qwen7b_warmup.test_probe_ablation_model','reproduction.operations.qwen7b_warmup.test_g10_grouping'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=90)
  assert test_run.returncode==0,test_run.stdout
  tests=test_run.stdout
  (suite/'actual-cpu-tests.log').write_text(tests)
  old=B/'mixed-only-ft-cp2-prefix20-20261020';bal=B/'balanced-rank1-from-epoch1-prefix20-20261020';c=read(bal/'config.json');source=read(old/'probe/epoch-0001.json')
  assert source['completed_epoch'] and source['cursor']==0 and source['steps']==10464 and source['sha256']=='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'
  sr=ref(pathlib.Path(source['path']));assert sr['sha256']==source['sha256']
  groups=read(bal/'bank/groups.json');held=pathlib.Path(c['selection_base']);assert len(groups['train'])==169028 and len(groups['diagnostic'])==512
  import sys;sys.path.insert(0,str(root))
  from reproduction.operations.qwen7b_warmup.balanced_sampling import plan,proof
  full=plan(groups['train'],1729,2,67218);assert len(full)==1318 and all(len(x)==51 for x in full)
  selected=full[:128];qids=[[groups['train'][i]['qid'] for i in batch] for batch in selected];cohort=[q for batch in qids for q in batch];assert len(cohort)==len(set(cohort))==6528
  forbidden={g['qid'] for g in groups['diagnostic']+read(held/'bank/groups.json')['selection_validation']};assert not set(cohort)&forbidden
  atomic(suite/'plan.json',dict(cohort_qids=cohort,batches=qids,source_seed=1729,source_epoch=2,screening_only=True,full_candidate_plan=proof(groups['train'],full)))
  assets=dict(source_checkpoint=sr,source_config=ref(old/'config.json'),source_bank=ref(old/'bank/groups.json'),model_config=ref(pathlib.Path(c['model_dir'])/'config.json'),bank=ref(bal/'bank/groups.json'),features=ref(B/'bank/features.json'),selection_bank=ref(held/'bank/groups.json'),selection_features=ref(held/'bank/features.json'),plan=ref(suite/'plan.json'))
  variants={}
  for name,group_kwargs in [('trajectory',{}),('joint',dict(binary_mixed_weight=1.)),('event',dict(three_event_weight=1.))]:
   base=suite/name;base.mkdir(exist_ok=True);(base/'logs').mkdir(exist_ok=True)
   config=dict(model_dir=c['model_dir'],update_budget=128,global_groups=51,micro_groups=2,save_every=16,recipe=dict(id=name,loss_kwargs=dict(trajectory_weight=1.,**group_kwargs)),code_hashes=hashes,source_commit=commit,screening_only=True,source_world_size=4,current_world_size=1,**assets)
   if (base/'config.json').exists():assert read(base/'config.json')==config,'Conflicting prepared config; do not overwrite'
   else:atomic(base/'config.json',config)
   script="""#!/usr/bin/env bash
#SBATCH --job-name=q7b-loss128-NAME
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=BASE/logs/fit-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=ROOT
export FLASH_ATTENTION_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun --kill-on-bad-exit=1 bash ROOT/reproduction/operations/qwen7b_warmup/runtime.sh -u -m reproduction.operations.qwen7b_warmup.probe_ablation_train --base BASE
""".replace('NAME',name).replace('BASE',str(base)).replace('ROOT',str(root))
   (base/'fit.sbatch').write_text(script);variants[name]=dict(base=str(base),config=ref(base/'config.json'),sbatch=ref(base/'fit.sbatch'))
  acceptance=dict(status='MATCHED_LOSS128_SUITE_PREPARED_ACTUAL_CPU_TESTS_PASSED',source_commit=commit,code_root=str(root),source_checkpoint=sr,source_adam_steps=10464,fit_questions=6528,fit_trajectories=32640,fit_updates_per_variant=128,global_groups=51,variants=variants,plan=ref(suite/'plan.json'),cpu_tests_log=ref(suite/'actual-cpu-tests.log'),hardware_topology_changed=True,no_four_rank_equivalence_claim=True,actor_updates=0,screening_only=True,time=time.time())
  atomic(suite/'preparation.json',acceptance);print(json.dumps(acceptance))
'''
SUBMIT=r'''
with (suite/'submission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 acceptance=read(suite/'preparation.json');pub=read(suite/'source-publication.json');base=pathlib.Path(acceptance['variants'][variant]['base'])
 for rel,digest in pub['files'].items():assert sha(pathlib.Path(pub['code_root'])/rel)==digest
 item=acceptance['variants'][variant]
 for key in ['config','sbatch']:assert sha(item[key]['path'])==item[key]['sha256']
 name='q7b-loss128-'+variant
 if (base/'job.id').exists():
  job=(base/'job.id').read_text().strip();state=command(['sacct','-n','-P','-j',job,'-o','JobID,JobName,State,ExitCode'])
  print(json.dumps(dict(status='EXISTING_MATCHED_JOB_NOT_RESUBMITTED',job_id=job,variant=variant,accounting=state,base=str(base),time=time.time())))
 else:
  existing=command(['squeue','-h','-u','mou176','-n',name,'-o','%i|%T|%R'])
  if existing:
   rows=existing.splitlines();assert len(rows)==1 and 'JobHeldUser' in rows[0],'Unknown same-name job: '+existing;job=rows[0].split('|')[0];assert job.isdigit()
  else:
   assert not (base/'submit-intent.json').exists(),'Unknown prior submission: reconcile sacct/name before retry'
   atomic(base/'submit-intent.json',dict(name=name,time=time.time(),source_commit=pub['commit'],config=item['config']))
   job=command(['sbatch','--hold','--parsable',str(base/'fit.sbatch')]).split(';')[0];assert job.isdigit()
  (base/'job.id').write_text(job+'\n');record=dict(status='MATCHED_LOSS128_JOB_HELD',job_id=job,variant=variant,base=str(base),source_commit=pub['commit'],config=item['config'],time=time.time(),gpus=1,partition='acd_u',no_dependency=True)
  atomic(base/'submission.json',record);state=command(['squeue','-h','-j',job,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state
  command(['scontrol','release',job]);record.update(status='MATCHED_LOSS128_JOB_RELEASED',time=time.time(),queue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%R']));atomic(base/'submission.json',record);print(json.dumps(record))
'''
if __name__=='__main__':main()
