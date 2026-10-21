"""Independent HPC3 execution of frozen donor-unsealed G5 inputs; no shared claim.

Sampling identity stays donor-exact to retain request seeds; execution identity,
code/config/host/receipts are explicitly HPC3-owned. No actor updates/hidden cache.
This standalone file imports the pinned old collector through its own runtime.
"""
import argparse
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
import uuid

B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021'
BASE=A/'hpc3-tail-newseedG5-20261021-v1'
ROLE='HPC3_OWNED_INDEPENDENT_DONOR_TAIL_REAL_G5_POTENTIAL_OVERLAP'
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
DONOR_RUN='70188b8e50fd4b45bc420ce58014f46e'
ATTEMPT='independent-seed20261021-g5-20261020'

def read(p):return json.loads(Path(p).read_bytes())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def proof(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))
def checked(ref):
 p=Path(ref['path'])
 if p.stat().st_size!=ref['bytes']or digest(p)!=ref['sha256']:raise ValueError('Bound metadata/source differs: '+str(p))
 return p
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 with tmp.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(str(tmp),str(p))
def once(p,raw):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=raw:raise ValueError('Immutable tail asset differs: '+str(p))
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def json_once(p,v):once(p,(json.dumps(v,indent=2,sort_keys=True)+'\n').encode())
def source_ordinals(item):
 shard=item['source_shard'];batch=item['source_batch'];n=item['questions']
 if shard not in (0,1)or type(batch)is not int or batch<0 or not 1<=n<=128:raise ValueError('Actual original2shard/128 partition required')
 return[shard+2*(batch*128+j)for j in range(n)]
def validate(c):
 if(c['role']!=ROLE or c['base']!=str(BASE)or c['run_id']!=DONOR_RUN or c['sampling_protocol_source_run_id']!=DONOR_RUN or c['sampling_attempt_id']!=ATTEMPT or c['seed']!=20261021 or c['model_revision']!=REVISION):raise ValueError('Frozen sampling protocol identity differs')
 if(c['execution_host_cluster']!='HPC3'or len(c['execution_run_id'])!=32 or c['execution_run_id']==DONOR_RUN or not c['potentialOverlap']or c['shared_donor_atomic_claim']):raise ValueError('Explicit independent execution and overlap provenance required')
 if(c['actor_optimizer_updates']!=0 or c['actor_updates_authorized']is not False or not c['warmup_only']or c['actor_hidden_native']or c['capture_prefix_tokens']!=20 or c['question_max_tokens']!=273 or c['workers']!=2):raise ValueError('Frozen official actor/P20/no-hidden/two-worker contract required')
 return c

def render_sbatch(base,code_root,oldroot,worker):
 if worker not in (0,1):raise ValueError('Exactly two independent2GPU workers')
 return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-tail-newseed-w@@WORKER@@
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=@@BASE@@/logs/worker-@@WORKER@@-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@OLD@@
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" @@ENTRY@@ --worker @@WORKER@@ --config @@CONFIG@@ --config-sha256 @@SHA@@
'''.replace('@@WORKER@@',str(worker)).replace('@@BASE@@',str(base)).replace('@@OLD@@',shlex.quote(str(oldroot))).replace('@@ENTRY@@',shlex.quote(str(code_root/'hpc3_tail_g5.py'))).replace('@@CONFIG@@',shlex.quote(str(base/'config.json')))

def prepare(plan_ref,assets_ref,source_commit,code_root):
 plan=read(checked(plan_ref));assets=read(checked(assets_ref));code_root=Path(code_root)
 if plan['role']!='HPC3_INDEPENDENT_TAIL_G5_PREPARATION_NO_DONOR_CLAIM'or plan['shared_donor_atomic_claim']or not plan['potentialOverlap']:raise ValueError('Frozen source unsealed snapshot required')
 if plan['source_identity']['source_run_id']!=assets['config_fields']['run_id']or plan['source_identity']['source_seed']!=1729:raise ValueError('Original canonical input provenance differs')
 if not all(x['exists']for x in assets['assets'].values())or assets['H100_NAS_path_shared_on_HPC3']:raise ValueError('Existing local assets and explicit nonshared source required')
 BASE.mkdir(parents=True,exist_ok=True);(BASE/'logs').mkdir(exist_ok=True)
 guard=(BASE/'prepare.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if(BASE/'prepared.json').exists():return read(BASE/'prepared.json')
 refs={name:proof(code_root/name)for name in ['hpc3_tail_g5.py','hpc3_tail_raw_native_vendor.py']}
 if(BASE/'execution-intent.json').exists():intent=read(BASE/'execution-intent.json')
 else:
  intent=dict(execution_run_id=uuid.uuid4().hex,source_tail_plan=plan_ref,source_commit=source_commit);json_once(BASE/'execution-intent.json',intent)
 if intent['source_tail_plan']!=plan_ref or intent['source_commit']!=source_commit:raise ValueError('Existing preparation belongs to another frozen cohort/source')
 c=read(checked(assets['original_config']));checked(assets['official_snapshot_complete']);checked(assets['collector_config'])
 old=read(assets['collector_config']['path']);oldroot=Path(old['code_root'])
 for rel,sha in old['code_files_sha256'].items():
  if digest(oldroot/rel)!=sha:raise ValueError('Qualified collector code differs')
 source_manifest=read(checked(assets['input_manifest']));byfile={str(Path(x['path'])):x for x in source_manifest['batches']if x['split']=='train'}
 exclusions=read(A/'joint-balanced-native-E4-qualification-v3/evaluation-identity-exclusions.json');assert len(exclusions['groups'])==6656
 blocked={g['qid']for g in exclusions['groups']};blockedhash={g['question_hash']for g in exclusions['groups']}
 # Only input JSON needed by frozen ordinals is read; no old rollout/tensor/gold from evaluations.
 ords={o for item in plan['remaining_batches']for o in source_ordinals(item)};paths={str(B/'inputs'/('shard-%02d'%(o%8))/('batch-%06d.json'%((o//8)//128)))for o in ords};lookup={}
 for p in sorted(paths):
  ref=byfile[p]
  if digest(p)!=ref['sha256']:raise ValueError('Original token input SHA differs')
  for row in read(p)['rows']:
   if row['source_ordinal']in ords:lookup[row['source_ordinal']]=row
 if set(lookup)!=ords:raise ValueError('Frozen canonical tail rows missing')
 manifests=[];seenq=set();seent=set();domains={};hashes=set()
 for index,item in enumerate(plan['remaining_batches']):
  rows=[lookup[o]for o in source_ordinals(item)]
  for row in rows:
   q,h=row['qid'],row['question_hash']
   if q in seenq or q in blocked or h in blockedhash:raise ValueError('Duplicate question or true evaluation identity in tail')
   if row['candidate_ordinal']!=row['source_ordinal']or row['split']!='train'or not row['initial_prompt_token_ids']:raise ValueError('Source ordinal/prompt identity differs')
   seenq.add(q);hashes.add(h);domain=json.loads(q)[0];domains[domain]=domains.get(domain,0)+1
  batch=dict(shard=item['source_shard'],split='train',rows=rows,model_revision=REVISION);raw=json.dumps(batch,indent=2).encode()
  if len(raw)!=item['input_bytes']or hashlib.sha256(raw).hexdigest()!=item['input_sha256']:raise ValueError('Reconstructed donor input bytes/SHA not identical')
  p=BASE/item['relative'];once(p,raw);manifests.append(dict(proof(p),relative=item['relative'],questions=len(rows),worker=index%2,source_shard=item['source_shard'],source_batch=item['source_batch']))
 c.update(plan['source_identity']);c.update(plan['actual_generation_settings']);c.update(base=str(BASE),role=ROLE,execution_run_id=intent['execution_run_id'],execution_host_cluster='HPC3',execution_namespace=str(BASE),execution_source_commit=source_commit,source_commit=source_commit,sampling_protocol_source_run_id=DONOR_RUN,sampling_protocol_source_config=plan['source_config'],sampling_protocol_source_manifest=plan['source_manifest'],potentialOverlap=True,shared_donor_atomic_claim=False,local_two_workers_disjoint=True,workers=2,train_questions=len(seenq),train_trajectories=5*len(seenq),diagnostic_questions=0,diagnostic_trajectories=0,actor_hidden_native=False,hidden_cached=False,actor_updates_authorized=False,actor_optimizer_updates=0,warmup_only=True,source_tail_plan=plan_ref,assets_metadata=assets_ref,qualified_runtime_root=str(oldroot),qualified_code_files_sha256=old['code_files_sha256'],execution_code=refs,model_snapshot=assets['official_snapshot_complete'],true_evaluation_exclusions=proof(A/'joint-balanced-native-E4-qualification-v3/evaluation-identity-exclusions.json'),frozen_main_cohort_questions=124194,frozen_main_cohort_expanded=False)
 validate(c);json_once(BASE/'inputs/manifest.json',dict(batches=manifests,questions=len(seenq),trajectories=5*len(seenq),domains=domains,train_text_hash_repeats=len(seenq)-len(hashes),potentialOverlap=True));json_once(BASE/'config.json',c);cfg=proof(BASE/'config.json')
 scripts=[]
 for worker in (0,1):
  p=BASE/('worker-%d.sbatch'%worker);once(p,render_sbatch(BASE,code_root,oldroot,worker).replace('@@SHA@@',cfg['sha256']).encode());scripts.append(proof(p))
 done=dict(status='HPC3_INDEPENDENT_TAIL_READY_GPU_NOT_SUBMITTED',config=cfg,inputs=proof(BASE/'inputs/manifest.json'),source_tail_plan=plan_ref,source_commit=source_commit,execution_run_id=c['execution_run_id'],questions=len(seenq),trajectories=5*len(seenq),domains=domains,workers=[dict(worker=w,questions=sum(x['questions']for x in manifests if x['worker']==w),batches=sum(x['worker']==w for x in manifests))for w in (0,1)],scripts=scripts,potentialOverlap=True,source_protocol_exact_input_SHA=True,donor_claim_performed=False,actor_updates=0,GPU_submitted=False,frozen_main_cohort_expanded=False,epoch=time.time());json_once(BASE/'prepared.json',done);return done

def run(config_ref,worker):
 c=validate(read(checked(config_ref)));base=Path(c['base']);control=base/'workers'/('worker-%d'%worker);control.mkdir(parents=True,exist_ok=True)
 guard=(control/'worker.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for ref in c['execution_code'].values():checked(ref)
 checked(c['model_snapshot'])
 for rel,sha in c['qualified_code_files_sha256'].items():
  if digest(Path(c['qualified_runtime_root'])/rel)!=sha:raise ValueError('Qualified runtime changed')
 assigned=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(assigned)==2 and len(set(assigned))==2
 from reproduction.operations.qwen7b_warmup import sampling
 if sampling._CACHED:raise ValueError('Tail collection requires fresh process')
 spec=importlib.util.spec_from_file_location('isolated_hpc3_raw_native',checked(c['execution_code']['hpc3_tail_raw_native_vendor.py']));raw=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw)
 port=27341+worker;env=dict(os.environ,CUDA_VISIBLE_DEVICES=assigned[1],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
 log=(control/'retriever.log').open('ab');command=[sys.executable,'-u','-m','reproduction.operations.qwen7b_warmup.retriever','--index',c['retrieval_index'],'--corpus',c['retrieval_corpus'],'--encoder',c['retrieval_encoder'],'--cache',str(control/'retriever'),'--data-cache',c['retrieval_data_cache'],'--run-id',c['execution_run_id'],'--port',str(port),'--gpu-shards','1','--batch-queries','256'];retriever=None;started=time.monotonic();accepted=[]
 atomic(control/'controller.json',dict(execution_run_id=c['execution_run_id'],sampling_protocol_source_run_id=DONOR_RUN,execution_host_cluster='HPC3',host=os.uname().nodename,pid=os.getpid(),birth_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],Slurm_job_id=os.environ['SLURM_JOB_ID'],source_commit=c['execution_source_commit'],config=config_ref,potentialOverlap=True,epoch=time.time()))
 try:
  retriever=subprocess.Popen(command,env=env,stdout=log,stderr=log)
  while True:
   if retriever.poll()is not None:raise RuntimeError('Own retriever exited during startup')
   try:
    with urllib.request.urlopen('http://127.0.0.1:%d/health'%port,timeout=2)as response:health=json.load(response)
    if (health.get('ok')or health.get('ready'))and health.get('run_id')==c['execution_run_id']:break
   except Exception:pass
   if time.monotonic()-started>1800:raise TimeoutError('Own retriever initialization exceeded30min')
   time.sleep(2)
  manifest=read(base/'inputs/manifest.json')
  for item in manifest['batches']:
   if item['worker']!=worker:continue
   inp=checked(item);lockpath=base/'batch-locks'/(item['sha256']+'.lock');lockpath.parent.mkdir(exist_ok=True);batchlock=lockpath.open('a+');fcntl.flock(batchlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   dest=base/'sampling'/inp.parent.name/'batches'/inp.stem;receipt=dest/'complete.json'
   if receipt.exists():
    done=read(receipt)
    if done['execution_run_id']!=c['execution_run_id']or done['input_sha256']!=item['sha256']:raise ValueError('Existing accepted batch identity differs')
    accepted.append(done);batchlock.close();continue
   batch=read(inp);outer=dest/'outers/outer-000001';outer.mkdir(parents=True,exist_ok=True);json_once(outer/'questions.json',dict(outer=1,rows=batch['rows']))
   cfg=dict(c,base=str(dest),world_size=1,actor_model_dir=c['model_dir'],actor_version='official7b-'+REVISION,prefix_policy_tokens=20,search_url='http://127.0.0.1:%d/retrieve'%port)
   path=dest/'config.json';json_once(path,cfg);prefix=raw.phase(path,'prefix');full=raw.phase(path,'continue')
   done=dict(run_id=DONOR_RUN,sampling_protocol_source_run_id=DONOR_RUN,execution_run_id=c['execution_run_id'],execution_host_cluster='HPC3',execution_host=os.uname().nodename,execution_job=os.environ['SLURM_JOB_ID'],execution_source_commit=c['execution_source_commit'],input_sha256=item['sha256'],batch=inp.stem,questions=len(batch['rows']),trajectories=5*len(batch['rows']),prefix=prefix,full=full,hidden_cached=False,model_revision=REVISION,actor_optimizer_updates=0,potentialOverlap=True,donor_original_receipt_claim=False,epoch=time.time());json_once(receipt,done);accepted.append(done)
   atomic(control/'progress.json',dict(status='HPC3_OWN_REAL_G5_SEALED',execution_run_id=c['execution_run_id'],worker=worker,completed_batches=len(accepted),questions=sum(x['questions']for x in accepted),trajectories=sum(x['trajectories']for x in accepted),latest_batch=str(receipt),elapsed=time.monotonic()-started,potentialOverlap=True,epoch=time.time()));batchlock.close()
  done=dict(status='HPC3_OWN_ASSIGNED_TAIL_REAL_G5_COMPLETE',execution_run_id=c['execution_run_id'],sampling_protocol_source_run_id=DONOR_RUN,worker=worker,questions=sum(x['questions']for x in accepted),trajectories=sum(x['trajectories']for x in accepted),batches=len(accepted),potentialOverlap=True,actor_updates=0,epoch=time.time());json_once(control/'complete.json',done)
 except BaseException as error:atomic(control/'failure.json',dict(type=type(error).__name__,error=str(error)[:500],epoch=time.time()));raise
 finally:
  if retriever is not None and retriever.poll()is None:
   retriever.terminate()
   try:retriever.wait(timeout=20)
   except subprocess.TimeoutExpired:retriever.kill();retriever.wait()
  log.close()
 return done

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--source-plan');p.add_argument('--source-plan-sha256');p.add_argument('--assets');p.add_argument('--assets-sha256');p.add_argument('--source-commit');p.add_argument('--code-root');p.add_argument('--worker',type=int,choices=(0,1));p.add_argument('--config');p.add_argument('--config-sha256');a=p.parse_args()
 if a.prepare:
  pref=dict(path=a.source_plan,bytes=Path(a.source_plan).stat().st_size,sha256=a.source_plan_sha256);aref=dict(path=a.assets,bytes=Path(a.assets).stat().st_size,sha256=a.assets_sha256);result=prepare(pref,aref,a.source_commit,Path(a.code_root))
 else:
  if a.worker is None or not a.config_sha256:p.error('Explicit owned GPU worker and config SHA required')
  result=run(dict(path=a.config,bytes=Path(a.config).stat().st_size,sha256=a.config_sha256),a.worker)
 print(json.dumps(result,sort_keys=True),flush=True)
if __name__=='__main__':
 def interrupted(signum,frame):raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 main()
