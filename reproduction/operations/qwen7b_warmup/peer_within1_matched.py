"""Matched NQ8192 G10 within1 fork: canonical-token AND actual-input mask.

Same E1 raw/EMA/Adam/RNG10464, plan, BF16 bank, B32x256 and fourG5/q as
peer743367. Only loss adds within1. Runtime preserves D8, with the published
48eb callback-only interface and this isolated callback; no active file edits.
"""
import argparse,copy,fcntl,gzip,hashlib,json,os,platform,shlex,subprocess,sys,time,traceback
from dataclasses import dataclass
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021'
CONTROL=A/'peer-newseed8192-g10-train-v1';TARGET=A/'peer8192-within1-matched256-v1'
D8='d8f74420aa09fa56e8ec59207c6d1453fa86f95e';CALLBACK='48eb944281d2e3d6fa5ad66e4d6f0f9eaf18aeec'
SOURCE_SHA='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'
PKG='reproduction/operations/qwen7b_warmup/'
@dataclass(frozen=True)
class Spec:
 questions:int=8192
 trajectories:int=81920
 updates:int=256
 global_questions:int=32
 source_steps:int=10464
 within_weight:float=1.
 learning_rate:float=5e-6

def read(p):return json.loads(Path(p).read_text())
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(r):
 p=Path(r['path'])
 if not p.is_absolute() or p.resolve()!=p or not str(p).startswith(str(B)+'/'):raise ValueError('Remote shared-storage input required')
 a=proof(p)
 if a['sha256']!=r['sha256'] or ('bytes' in r and a['bytes']!=r['bytes']):raise ValueError('Source SHA/bytes differ: '+str(p))
 return p
def atomic(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def immutable(p,v):
 if Path(p).exists():
  if read(p)!=v:raise ValueError('Immutable output differs: '+str(p))
 else:atomic(p,v)

def gated_pairs(labels,actual,causal):
 """Pure mask contract; BF16 numeric drift cannot distinguish equal tokens."""
 n=len(labels)
 if n!=10 or len(actual)!=10 or len(causal)!=10 or any(len(r)!=10 for r in actual) or any(type(y) is not int or y not in (0,1) for y in labels):raise ValueError('Own G10 masks/EM/signatures required')
 return [[bool(actual[i][j] and labels[i]==1 and labels[j]==0 and causal[i]!=causal[j]) for j in range(n)] for i in range(n)]

def matched_config(original,publication,signature_ref):
 sp=Spec()
 if (original['source_checkpoint']['sha256']!=SOURCE_SHA or original['source_adam_steps']!=sp.source_steps or original['update_budget']!=sp.updates or original['global_questions']!=sp.global_questions or original['fit_questions']!=sp.questions or original['fit_real_trajectories']!=sp.trajectories or original['grouping']!=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise') or original['recipe']['loss_kwargs']!=dict(trajectory_weight=1.,three_event_weight=1.) or original['optimizer_reset'] is not False):raise ValueError('Exact completed peer743367 protocol required')
 c=copy.deepcopy(original);c['recipe']['loss_kwargs']['within_rank_weight']=sp.within_weight
 c['source_commit']=publication['commit'];c['code_hashes']={n:r['sha256'] for n,r in publication['files'].items()};c['within_pair_inputs']=signature_ref
 audit=copy.deepcopy(c)
 for key in ('source_commit','code_hashes','within_pair_inputs'):audit.pop(key,None)
 baseline=copy.deepcopy(original)
 for key in ('source_commit','code_hashes'):baseline.pop(key,None)
 audit['recipe']['loss_kwargs'].pop('within_rank_weight')
 if audit!=baseline:raise ValueError('Additional unapproved protocol difference')
 return c

def signature_cache(original):
 from . import peer_newseed_train as fit
 from .peer_g10_learnability import causal_signature
 groups,manifests,accept=fit.accepted_peer_g10(original['g10_complete']);want={t:g['qid'] for g in groups for t in g['trajectory_ids']}
 refs=[];seen_paths=set()
 for r in accept['feature_shard_completes']:
  capture=read(checked(r));selection=read(checked(capture['source_selection']))
  if selection['question_plan']!=accept['identity_plan'] or selection['source_ingest']!=accept['raw_complete']:raise ValueError('Causal prefix sources do not match accepted fit bank')
  for ref in selection['prefix_files']:
   if ref['path'] in seen_paths:raise ValueError('Repeated projected prefix source')
   seen_paths.add(ref['path']);refs.append(ref)
 cache=TARGET/'causal-signatures-complete.json'
 if cache.exists():
  done=read(cache)
  if done['source_training_input']!=original['g10_complete'] or done['prefix_sources']!=refs:raise ValueError('Existing canonical prefix signature binding differs')
  signatures=read(checked(done['signatures']))
  if set(signatures)!=set(want):raise ValueError('Incomplete existing signature coverage')
  return proof(cache),groups
 signatures={};started=time.time()
 for ref in refs:
  p=checked(ref)
  with gzip.open(p,'rt') as f:
   for line in f:
    row=json.loads(line);tid=row['trajectory_id']
    if tid not in want or row['qid']!=want[tid] or tid in signatures:raise ValueError('Causal source own identity coverage differs')
    if any(k in row for k in ('strict_em','labels','response_token_ids','ground_truth')):raise ValueError('Projected prefix contains EM/future response payload')
    signatures[tid]=causal_signature(row)[0]
    if len(signatures)%2048==0:atomic(TARGET/'cpu-signature-progress.json',dict(trajectories=len(signatures),expected=81920,elapsed=time.time()-started,time=time.time()))
  checked(ref)
 if set(signatures)!=set(want):raise ValueError('Incomplete81920 canonical causal signatures')
 immutable(TARGET/'causal-signatures.json',signatures)
 done=dict(status='ALL_81920_OWN_PREFIX20_CANONICAL_CAUSAL_SIGNATURES_ACCEPTED',signatures=proof(TARGET/'causal-signatures.json'),prefix_sources=refs,source_training_input=original['g10_complete'],identity_plan=accept['identity_plan'],trajectories=len(signatures),question_plan_SHA=accept['ordered_qid_sha256'],signature_definition='peer_g10_learnability.causal_signature; full causal contexts/tokens/positions/masks/gather; no IDs or BF16 artifact',prefix_tokens=20,actor_updates=0,probe_updates=0,final_pool_read=False)
 immutable(cache,done);return proof(cache),groups

def render(base,code,sha):
 return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-peer8192-within1
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=@@BASE@@/logs/train-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.peer_within1_matched --action train --config-sha256 @@SHA@@
'''.replace('@@BASE@@',str(base)).replace('@@CODE@@',shlex.quote(str(code))).replace('@@SHA@@',sha)

def prepare():
 old=read(CONTROL/'config.json');pub=read(TARGET/'source-publication.json');root=Path(pub['code_root'])
 for n,r in pub['files'].items():checked(dict(path=str(root/n),**r))
 if pub['base_runtime_commit']!=D8 or pub['callback_interface_commit']!=CALLBACK:raise ValueError('Published minimal D8/callback-only runtime required')
 sig,groups=signature_cache(old);c=matched_config(old,pub,sig);immutable(TARGET/'config.json',c)
 audit=dict(status='MATCHED_PEER8192_CONFIG_ONLY_WITHIN1_AND_REQUIRED_CAUSAL_CALLBACK_BINDING_DIFFER',source_config=proof(CONTROL/'config.json'),config=proof(TARGET/'config.json'),same_plan=old['plan'],same_training_input=old['g10_complete'],same_source_checkpoint=old['source_checkpoint'],same_evaluations=old['evaluations'],same_B32_updates256_random4views_seed1729=True,changed_loss=dict(before=old['recipe']['loss_kwargs'],after=c['recipe']['loss_kwargs']),necessary_source_changes=['probe_g10_train.py callback-only interface48eb944','canonical-token intersected within1 callback and SHA-bound source signatures'],forward_loss_otherwise_D8_unchanged=True,source_Adam10464_and_all_four_RNG_preserved=True,optimizer_reset=False)
 immutable(TARGET/'matched-control-audit.json',audit)
 text=render(TARGET,root,proof(TARGET/'config.json')['sha256']);p=TARGET/'training.sbatch'
 if p.exists() and p.read_text()!=text:raise ValueError('Immutable within1 sbatch differs')
 if not p.exists():p.write_text(text)
 r=dict(status='MATCHED_REAL_G10_WITHIN1_PREPARED',config=proof(TARGET/'config.json'),sbatch=proof(p),matched_control_audit=proof(TARGET/'matched-control-audit.json'),causal_signatures=sig,questions=8192,trajectories=81920,updates=256,source_steps=10464,no_model_checkpoint_tensor_loaded=True)
 immutable(TARGET/'preparation.json',r);return r

def callback(signatures):
 from . import probe_g10_train as t
 def update(model,ema,optimizer,reader,lookup,qids,c,device,epoch):
  import torch
  selected=[lookup[q] for q in qids];actual=t.input_pair_mask(reader,selected,device)
  canonical=[gated_pairs(g['labels'],a,[signatures[x] for x in g['trajectory_ids']]) for g,a in zip(selected,actual.cpu().tolist())]
  mask=torch.tensor(canonical,dtype=torch.bool,device=device)
  if bool((mask&~actual).any()):raise ValueError('Causal intersection added an ineligible feature pair')
  grouping=dict(c['grouping']);grouping['epoch']=epoch
  r=t.bounded_question_update(model,ema,optimizer,reader,selected,device,grouping_kwargs=grouping,loss_kwargs=c['recipe']['loss_kwargs'],micro_trajectories=c['micro_trajectories'],distinguishable_pair_mask=mask,max_virtual_views=c['max_virtual_views'],max_inter_view_pairs=c['max_inter_view_pairs'])
  r.update(within_actual_feature_pairs=int(actual.sum()),within_canonical_and_feature_pairs=int(mask.sum()),within_same_causal_excluded_feature_pairs=int((actual&~mask).sum()),within_causal_signature_rule=True)
  return r
 return update

def audit_first():
 """Separate CPU subprocess inside the allocated job; no GPU/model updates."""
 import torch
 from . import probe_ablation_train as u
 deadline=time.time()+3600
 while not (TARGET/'progress.json').exists() or read(TARGET/'progress.json')['cursor']<1:
  if (TARGET/'failure.json').exists():raise ValueError('Trainer failed before first accepted update')
  if time.time()>deadline:raise ValueError('No first accepted CP within bounded audit wait')
  time.sleep(1)
 p=TARGET/'checkpoints/cursor-000001.pt';state=torch.load(p,map_location='cpu',mmap=True,weights_only=False);csha=proof(TARGET/'config.json')['sha256'];replay=read(TARGET/'next-update-replay.json')
 if state['cursor']!=1 or state['steps']!=10465 or state['config_sha256']!=csha or replay['status']!='PASSED' or replay['actual']!=replay['expected']:raise ValueError('Actual first completed replay/CP lineage differs')
 actual=dict(raw=u.model_mapping_hash(state['raw']),ema=u.model_mapping_hash(state['ema']),adam=u.optimizer_blob_hash(state['optimizer']),raw_buffers=u.tensor_hash(state['raw_buffers']),ema_buffers=u.tensor_hash(state['ema_buffers']),rng=u.rng_hash(state['rng']))
 if actual!=state['accepted_hashes'] or actual!=replay['actual'] or len(state['source_all_rank_states'])!=4:raise ValueError('Independent first CPU component reload differs')
 r=dict(status='FIRST_CP1_INDEPENDENT_FULL_SHA_AND_CPU_ALL_COMPONENT_RELOAD_PASSED',checkpoint=proof(p),cursor=1,steps=10465,config_sha256=csha,component_hashes=actual,source_all_four_RNG_preserved=True,CUDA_used=False,actor_updates=0,probe_extra_updates=0,time=time.time());immutable(TARGET/'first-cp-independent-audit.json',r);return r

def train(expected_sha):
 import torch
 from . import probe_g10_train as t,peer_newseed_train as fit
 c=read(TARGET/'config.json');pub=read(TARGET/'source-publication.json');root=Path(pub['code_root'])
 if proof(TARGET/'config.json')['sha256']!=expected_sha:raise ValueError('Actual within1 config SHA differs')
 for n,r in pub['files'].items():checked(dict(path=str(root/n),**r))
 old=read(CONTROL/'config.json')
 if matched_config(old,pub,c['within_pair_inputs'])!=c:raise ValueError('Matched original protocol changed')
 sig=read(checked(c['within_pair_inputs']));signatures=read(checked(sig['signatures']))
 if len(signatures)!=81920 or c['source_checkpoint']['sha256']!=SOURCE_SHA:raise ValueError('Full original source/causal signature coverage required')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');log=(TARGET/'first-cp-independent-audit.log').open('ab',0)
 child=subprocess.Popen([sys.executable,'-u','-m','reproduction.operations.qwen7b_warmup.peer_within1_matched','--action','audit-first'],stdout=log,stderr=subprocess.STDOUT,env=env)
 atomic(TARGET/'first-cp-audit-controller.json',dict(pid=child.pid,job_id=os.environ.get('SLURM_JOB_ID'),inside_allocated_job=True,time=time.time()))
 try:
  t.run(TARGET,bank_loader=fit.accepted_peer_g10,reader_factory=fit.replay_reader,update_callback=callback(signatures))
  if not (TARGET/'complete.json').exists():return dict(status='EXACT_SAVED_INTERRUPTED_TRAINING_NOT_COMPLETE')
  if child.wait(timeout=60)!=0 or not (TARGET/'first-cp-independent-audit.json').exists():raise ValueError('First independent CPU reload missing/failed')
  return dict(status='MATCHED_WITHIN1_FULL_BUDGET_AND_ORIGINAL_EVALUATIONS_COMPLETE',complete=proof(TARGET/'complete.json'),first_audit=proof(TARGET/'first-cp-independent-audit.json'))
 finally:
  if child.poll() is None:child.terminate();child.wait(timeout=30)

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','train','audit-first'),required=True);p.add_argument('--config-sha256');a=p.parse_args()
 if platform.system()!='Linux':p.error('Real actions are remote only')
 TARGET.mkdir(parents=True,exist_ok=True);(TARGET/'logs').mkdir(exist_ok=True)
 if a.action=='train' and (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES')):p.error('Training only inside owned GPU allocation')
 lockname='first-cp-audit.lock' if a.action=='audit-first' else 'within1.lock'
 with (TARGET/lockname).open('a+') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:r=prepare() if a.action=='prepare' else audit_first() if a.action=='audit-first' else train(a.config_sha256);print(json.dumps(r),flush=True)
  except BaseException as error:atomic(TARGET/('first-cp-audit-failure.json' if a.action=='audit-first' else 'failure.json'),dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),time=time.time(),job_id=os.environ.get('SLURM_JOB_ID')));raise
if __name__=='__main__':main()
