"""Two isolated E1/peer8192 matched256 ranking forks; never alters within1.

Inter is the existing product-view-weight, distinct-q, Q*(Q-1) objective.
Both reuses the accepted canonical-causal AND actual-feature within mask.
"""
import argparse,copy,fcntl,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path
from . import peer_within1_matched as w
B,A,CONTROL,PKG=w.B,w.A,w.CONTROL,w.PKG
REFERENCE='ebc452595fddfd5eb095b45091a642c1e02ea060'
SOURCE_CONFIG_SHA='16b2fd34e3e8907307b3866298b33216478fd77c2dc1b9ca80638e0929800130'
PLAN_SHA='0b2212b46e2046d097f7c1a90490406ba67babf81002f15b129a61efe4dba012'
SIGNATURE_SHA='5077b7b1244d2d8e20253c943028a9948e45732c3b600715836f0f4f40f93473'
RECIPES={'inter1':dict(inter_rank_weight=1.),'bothrank1':dict(within_rank_weight=1.,inter_rank_weight=1.)}
read,proof,checked,atomic,immutable=w.read,w.proof,w.checked,w.atomic,w.immutable

def target(recipe):
 if recipe not in RECIPES:raise ValueError('Only the two predeclared matched ranking recipes are authorized')
 return A/('peer8192-'+recipe+'-matched256-v1')

def matched_config(original,publication,recipe,signature_ref=None):
 target(recipe)
 if (original['source_checkpoint']['sha256']!=w.SOURCE_SHA or original['source_adam_steps']!=10464
  or original['update_budget']!=256 or original['global_questions']!=32 or original['fit_questions']!=8192
  or original['fit_real_trajectories']!=81920 or original['plan']['sha256']!=PLAN_SHA
  or original['grouping']!=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise')
  or original['recipe']['loss_kwargs']!=dict(trajectory_weight=1.,three_event_weight=1.)
  or original['optimizer_reset'] is not False or original['max_virtual_views']!=128
  or original['max_inter_view_pairs']!=4096):raise ValueError('Exact peer743367 E1/B32x256/128views/4096pair budget required')
 if (recipe=='bothrank1')!=(signature_ref is not None):raise ValueError('Only bothrank1 uses a real canonical signature reference')
 c=copy.deepcopy(original);c['recipe']['id']='peer8192-'+recipe+'-matched256'
 c['recipe']['loss_kwargs'].update(RECIPES[recipe]);c['source_commit']=publication['commit']
 c['code_hashes']={n:r['sha256'] for n,r in publication['files'].items()}
 if signature_ref is not None:c['within_pair_inputs']=signature_ref
 audit=copy.deepcopy(c);baseline=copy.deepcopy(original)
 for v in (audit,baseline):
  for key in ('source_commit','code_hashes','within_pair_inputs'):v.pop(key,None)
 audit['recipe']=copy.deepcopy(baseline['recipe'])
 if audit!=baseline:raise ValueError('Undeclared configuration leaf difference')
 return c

def actual_pair_budget(original):
 """CPU guard using real own labels/actual random views; no tensors loaded."""
 from . import peer_newseed_train as fit
 from .g10_grouping import build_training_views
 groups,_,_=fit.accepted_peer_g10(original['g10_complete']);lookup={g['qid']:g for g in groups}
 plan=read(checked(original['plan']));allq=[q for b in plan['batches'] for q in b]
 if (len(plan['batches'])!=256 or plan['batch_epochs']!=[0]*256 or len(allq)!=8192
  or len(set(allq))!=8192 or set(allq)!=set(lookup) or any(len(b)!=32 for b in plan['batches'])):raise ValueError('Actual inherited256 original batches differ')
 maximum=0
 for batch,epoch in zip(plan['batches'],plan['batch_epochs']):
  kw=dict(original['grouping']);kw['epoch']=epoch
  views=build_training_views([lookup[q] for q in batch],role='train',**kw)['groups']
  positives=sum(v['true_mixed'] for v in views);pairs=positives*(len(views)-positives)
  if len(views)!=128 or pairs>4096:raise ValueError('Actual128-view inter-pair memory budget exceeded')
  maximum=max(maximum,pairs)
 return dict(updates=256,questions=8192,real_trajectories=81920,views_per_update=128,
  maximum_positive_times_negative_views=maximum,max_inter_view_pairs=4096,
  nominal_ordered_distinct_question_pairs=32*31,inter_reduction='product view weights; exclude equal qid; divide Q*(Q-1), never eligible mean')

def signatures(original):
 p=w.TARGET/'causal-signatures-complete.json'
 if proof(p)['sha256']!=SIGNATURE_SHA:raise ValueError('Accepted within1 canonical-signature receipt changed')
 done=read(p)
 if (done['source_training_input']!=original['g10_complete'] or done['trajectories']!=81920
  or done['prefix_tokens']!=20 or done['actor_updates']!=0 or done['probe_updates']!=0):raise ValueError('Canonical signature lineage differs')
 values=read(checked(done['signatures']))
 if len(values)!=81920:raise ValueError('Complete81920 causal signatures required')
 return proof(p),values

def render(base,code,csha,recipe):
 text=w.render(base,code,csha).replace('q7b-peer8192-within1','q7b-peer8192-'+recipe)
 return text.replace('.peer_within1_matched --action train','.peer_rank_matched --recipe '+recipe+' --action train')

def prepare(recipe):
 base=target(recipe);old=read(CONTROL/'config.json');pub=read(base/'source-publication.json');root=Path(pub['code_root'])
 if proof(CONTROL/'config.json')['sha256']!=SOURCE_CONFIG_SHA:raise ValueError('Original peer control config SHA differs')
 if pub['matched_rank_reference_commit']!=REFERENCE:raise ValueError('Exact published within1 reference closure required')
 for n,r in pub['files'].items():checked(dict(path=str(root/n),**r))
 budget=actual_pair_budget(old);sig=signatures(old)[0] if recipe=='bothrank1' else None
 c=matched_config(old,pub,recipe,sig);immutable(base/'config.json',c)
 audit=dict(status='MATCHED_PEER8192_ONLY_EXPLICIT_RANK_RECIPE_AND_REQUIRED_BINDINGS_DIFFER',
  source_config=proof(CONTROL/'config.json'),config=proof(base/'config.json'),same_plan=old['plan'],same_training_input=old['g10_complete'],
  same_source_checkpoint=old['source_checkpoint'],same_evaluations=old['evaluations'],same_B32_updates256_random4views_seed1729=True,
  changed_loss=dict(before=old['recipe']['loss_kwargs'],after=c['recipe']['loss_kwargs']),
  source_Adam10464_and_all_four_RNG_preserved=True,optimizer_reset=False,pair_budget=budget,
  canonical_signature_read=recipe=='bothrank1',active_within1_modified=False)
 immutable(base/'matched-control-audit.json',audit);text=render(base,root,proof(base/'config.json')['sha256'],recipe);p=base/'training.sbatch'
 if p.exists() and p.read_text()!=text:raise ValueError('Immutable ranking sbatch differs')
 if not p.exists():p.write_text(text)
 result=dict(status='MATCHED_REAL_PEER8192_RANK_PREPARED',recipe=recipe,config=proof(base/'config.json'),sbatch=proof(p),
  matched_control_audit=proof(base/'matched-control-audit.json'),causal_signatures=sig,pair_budget=budget,no_model_checkpoint_tensor_loaded=True)
 immutable(base/'preparation.json',result);return result

def callback(recipe,causal=None):
 if recipe=='bothrank1':
  if causal is None:raise ValueError('Both ranking requires authentic causal signatures')
  return w.callback(causal)
 if recipe!='inter1' or causal is not None:raise ValueError('Inter-only callback must not use a fake within mask')
 from . import probe_g10_train as t
 def update(model,ema,optimizer,reader,lookup,qids,c,device,epoch):
  grouping=dict(c['grouping']);grouping['epoch']=epoch
  result=t.bounded_question_update(model,ema,optimizer,reader,[lookup[q] for q in qids],device,
   grouping_kwargs=grouping,loss_kwargs=c['recipe']['loss_kwargs'],micro_trajectories=c['micro_trajectories'],
   distinguishable_pair_mask=None,max_virtual_views=c['max_virtual_views'],max_inter_view_pairs=c['max_inter_view_pairs'])
  result.update(within_mask_constructed=False,canonical_signature_read=False,inter_distinct_original_questions=True)
  return result
 return update

def audit_first(recipe):
 import torch
 from . import probe_ablation_train as u
 base=target(recipe);deadline=time.time()+3600
 while not (base/'progress.json').exists() or read(base/'progress.json')['cursor']<1:
  if (base/'failure.json').exists():raise ValueError('Trainer failed before accepted CP1')
  if time.time()>deadline:raise ValueError('No accepted CP1 in bounded wait')
  time.sleep(1)
 p=base/'checkpoints/cursor-000001.pt';state=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
 csha=proof(base/'config.json')['sha256'];replay=read(base/'next-update-replay.json')
 if state['cursor']!=1 or state['steps']!=10465 or state['config_sha256']!=csha or replay['status']!='PASSED' or replay['actual']!=replay['expected']:raise ValueError('First fullstate/replay lineage differs')
 actual=dict(raw=u.model_mapping_hash(state['raw']),ema=u.model_mapping_hash(state['ema']),adam=u.optimizer_blob_hash(state['optimizer']),raw_buffers=u.tensor_hash(state['raw_buffers']),ema_buffers=u.tensor_hash(state['ema_buffers']),rng=u.rng_hash(state['rng']))
 if actual!=state['accepted_hashes'] or actual!=replay['actual'] or len(state['source_all_rank_states'])!=4:raise ValueError('Independent CPU reload six-component mismatch')
 result=dict(status='FIRST_CP1_INDEPENDENT_FULL_SHA_AND_CPU_ALL_COMPONENT_RELOAD_PASSED',recipe=recipe,checkpoint=proof(p),cursor=1,steps=10465,config_sha256=csha,component_hashes=actual,source_all_four_RNG_preserved=True,CUDA_used=False,actor_updates=0,probe_extra_updates=0,time=time.time())
 immutable(base/'first-cp-independent-audit.json',result);return result

def train(recipe,expected_sha):
 from . import probe_g10_train as t,peer_newseed_train as fit
 base=target(recipe);c=read(base/'config.json');pub=read(base/'source-publication.json');root=Path(pub['code_root']);old=read(CONTROL/'config.json')
 if proof(base/'config.json')['sha256']!=expected_sha or proof(CONTROL/'config.json')['sha256']!=SOURCE_CONFIG_SHA:raise ValueError('Actual config/source-control SHA differs')
 for n,r in pub['files'].items():checked(dict(path=str(root/n),**r))
 sig,causal=signatures(old) if recipe=='bothrank1' else (None,None)
 if matched_config(old,pub,recipe,sig)!=c:raise ValueError('Matched original protocol changed')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 log=(base/'first-cp-independent-audit.log').open('ab',0)
 child=subprocess.Popen([sys.executable,'-u','-m','reproduction.operations.qwen7b_warmup.peer_rank_matched','--recipe',recipe,'--action','audit-first'],stdout=log,stderr=subprocess.STDOUT,env=env)
 atomic(base/'first-cp-audit-controller.json',dict(pid=child.pid,job_id=os.environ.get('SLURM_JOB_ID'),inside_allocated_job=True,time=time.time()))
 try:
  t.run(base,bank_loader=fit.accepted_peer_g10,reader_factory=fit.replay_reader,update_callback=callback(recipe,causal))
  if not (base/'complete.json').exists():return dict(status='EXACT_SAVED_INTERRUPTED_TRAINING_NOT_COMPLETE')
  if child.wait(timeout=60)!=0 or not (base/'first-cp-independent-audit.json').exists():raise ValueError('Independent CP1 reload missing/failed')
  return dict(status='MATCHED_RANK_FULL256_AND_FIXED_HF_EVALUATIONS_COMPLETE',recipe=recipe,complete=proof(base/'complete.json'),first_audit=proof(base/'first-cp-independent-audit.json'))
 finally:
  if child.poll() is None:child.terminate();child.wait(timeout=30)

def main():
 p=argparse.ArgumentParser();p.add_argument('--recipe',choices=tuple(RECIPES),required=True);p.add_argument('--action',choices=('prepare','train','audit-first'),required=True);p.add_argument('--config-sha256');a=p.parse_args()
 if platform.system()!='Linux':p.error('Actual actions are remote only')
 base=target(a.recipe);base.mkdir(parents=True,exist_ok=True);(base/'logs').mkdir(exist_ok=True)
 if a.action=='train' and (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES')):p.error('Training requires an owned GPU allocation')
 with (base/('first-cp-audit.lock' if a.action=='audit-first' else 'rank-matched.lock')).open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:r=prepare(a.recipe) if a.action=='prepare' else audit_first(a.recipe) if a.action=='audit-first' else train(a.recipe,a.config_sha256);print(json.dumps(r),flush=True)
  except BaseException as e:atomic(base/('first-cp-audit-failure.json' if a.action=='audit-first' else 'failure.json'),dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=time.time(),job_id=os.environ.get('SLURM_JOB_ID')));raise
if __name__=='__main__':main()
