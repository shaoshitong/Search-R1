"""Finite frozen-E1 head-only training-cohort capacity diagnostic.

Not a production recipe selector. Only the new512518-parameter count head is
trained, on cached genuine masked P20 hidden states, inside one owned H100 job.
"""
import argparse,copy,fcntl,gzip,hashlib,importlib.util,json,math,os,random,shutil,signal,time,traceback
from pathlib import Path

B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
PRODUCTION=B/'algorithm-improvement-20261021/direct-hidden-count30016-train-v2'
TARGET=B/'algorithm-improvement-20261021/direct-count-head-capacity1536-v1'
CONFIG_SHA='3e910b12c0ff52eac70e61566c6135e2ec188bb47ec64ad79524abbd7a657526'
PKG='reproduction/operations/qwen7b_warmup/'
SOURCE_NAMES=('direct_hidden_count.py','g10_grouping.py','model.py','replay_feature_reader.py','probe_ablation_train.py')
LRS=(5e-6,1e-4,1e-3);NODES=(0,1,2,5,10,20);STOP=False
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):return json.loads(Path(p).read_bytes())
def checked(ref):
 p=Path(ref['path'])
 if p.resolve()!=p or not p.is_relative_to(B):raise ValueError('Owned immutable reference mismatch')
 actual=proof(p)
 if any(actual[k]!=ref[k]for k in ('bytes','sha256')):raise ValueError('Owned immutable reference mismatch')
 return p
def atomic(p,v):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.tmp')
 with tmp.open('wb')as f:f.write(canonical(v)+b'\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def immutable(p,v):
 if Path(p).exists():
  if read(p)!=v:raise ValueError('Immutable diagnostic metadata differs')
 else:atomic(p,v)
def split_questions(qids):
 if len(qids)!=8192 or len(set(qids))!=8192:raise ValueError('Original8192 distinct training qids required')
 ranked=sorted(qids,key=lambda q:(digest(['frozen-E1-head-capacity-v1',9137,q]),q))
 return ranked[:512],ranked[512:1536]
def fixed_view(views):
 # Only five source indices determine the designated primary G5; never labels.
 return min(views,key=lambda v:tuple(v['source_indices']))
def groups_for(c):
 bank=read(checked(c['g10_complete']));groups=read(checked(bank['outputs']['groups.json']))['train']
 return {g['qid']:g for g in groups},bank
def prepare():
 old=read(PRODUCTION/'config.json');pub=read(PRODUCTION/'source-publication.json')
 if proof(PRODUCTION/'config.json')['sha256']!=CONFIG_SHA or pub['commit']!='501cf80478b77f68c3e13a96d4659e3da0971b2e':raise ValueError('Exact original production E1 branch required')
 plan=read(checked(old['plan']));cohort=[q for batch in plan['batches'][:256]for q in batch];fit,held=split_questions(cohort)
 lookup,bank=groups_for(old)
 if len(lookup)!=30016 or not set(cohort)<=set(lookup):raise ValueError('Accepted training source differs')
 runtime=Path(pub['code_root']);sourcerefs={n:dict(path=str(runtime/(PKG+n)),**pub['files'][PKG+n])for n in SOURCE_NAMES}
 for ref in sourcerefs.values():checked(ref)
 selected=fit+held;split=dict(role='TRAINING_COHORT_ONLY_CAPACITY_DIAGNOSTIC',split_seed=9137,
  selection='SHA256 canonical [frozen-E1-head-capacity-v1,9137,qid]; sorted hash then qid; first512 fit, next1024 internal holdout',
  fit_qids=fit,holdout_qids=held,original8192_qids_sha256=digest(cohort),fit_qids_sha256=digest(fit),holdout_qids_sha256=digest(held),
  selected_identity_sha256=digest([{k:lookup[q][k]for k in ('qid','question_hash','trajectory_ids')}for q in selected]),
  identities_selected_without_EM=True,all_questions_from_original_warmup_seen_training_pool=True,
  existing_external_evaluation_or_final_not_used=True)
 immutable(TARGET/'split.json',split)
 orders=[]
 for epoch in range(1,21):
  order=list(fit);random.Random(int(digest(['same-head-forks-order',1729,epoch]),16)).shuffle(order)
  orders.append([order[i:i+32]for i in range(0,512,32)])
 immutable(TARGET/'orders.json',dict(epochs=orders,seed=1729))
 c=dict(kind='FROZEN_E1_HEAD_ONLY_THREE_LR_CAPACITY_DIAGNOSTIC',production_config=proof(PRODUCTION/'config.json'),production_publication=proof(PRODUCTION/'source-publication.json'),
  original_runtime=str(runtime),source_files=sourcerefs,source_checkpoint=old['source_checkpoint'],model_dir=old['model_dir'],model_config=old['model_config'],
  g10_complete=old['g10_complete'],source_plan=old['plan'],grouping=old['grouping'],split=proof(TARGET/'split.json'),orders=proof(TARGET/'orders.json'),
  source_old_Adam=10464,legacy_frozen=True,legacy_optimizer_created=False,new_head_init_seed=1729,new_head_Adam=0,
  feature_backend='same_original_frozen_HF_replay_to_native_probe_block27',prefix_tokens=20,question_max_tokens=273,
  native_micro=old['micro_trajectories'],fit_questions=512,internal_holdout_questions=1024,views_per_fit_question=4,view_weight='1/4',
  fixed_primary_evaluation_view='lexicographically smallest source_indices tuple among original4 real G5 views for each question; label-blind',
  learning_rates=list(LRS),epochs=20,questions_per_update=32,updates_per_epoch=16,updates_per_fork=320,EMA_decay=.99,evaluation_epochs=list(NODES),
  trajectory_BCE='frozen constant, recorded without backward',train_count_NLL_weight=1.,
  raw_and_EMA_reported_separately=True,holdout_primary_questions=1024,holdout_primary_top_questions=256,
  actor_updates=0,external_pool_access=False,final_pool_access=False,production_candidate_selection=False)
 own=read(TARGET/'source-publication.json');c['diagnostic_source_publication']=proof(TARGET/'source-publication.json')
 immutable(TARGET/'config.json',c)
 (TARGET/'logs').mkdir(exist_ok=True)
 code=Path(own['source']['path']);sha=proof(TARGET/'config.json')['sha256']
 script='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-headcap1536
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@180
#SBATCH --output=@@BASE@@/logs/diagnostic-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@RUNTIME@@
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u @@SOURCE@@ --action train --config-sha256 @@SHA@@
'''.replace('@@BASE@@',str(TARGET)).replace('@@RUNTIME@@',str(runtime)).replace('@@SOURCE@@',str(code)).replace('@@SHA@@',sha)
 sb=TARGET/'diagnostic.sbatch'
 if sb.exists()and sb.read_text()!=script:raise ValueError('Immutable sbatch differs')
 if not sb.exists():sb.write_text(script)
 result=dict(status='FROZEN_E1_HEAD_CAPACITY_METADATA_PREPARED_NO_GPU_SUBMISSION',config=proof(TARGET/'config.json'),split=c['split'],orders=c['orders'],sbatch=proof(sb),fit=512,holdout=1024,learning_rates=list(LRS),fit_Adam_updates_per_fork=320,
  checkpoint_tensors_loaded=False,actor_updates=0)
 immutable(TARGET/'preparation.json',result);return result

def train(expected_sha):
 import torch
 import torch.nn.functional as F
 from reproduction.operations.qwen7b_warmup import direct_hidden_count as d,probe_ablation_train as u,g10_grouping as g
 from reproduction.operations.qwen7b_warmup.model import Probe
 from reproduction.operations.qwen7b_warmup.replay_feature_reader import ReplayFeatureReader
 global STOP
 def stop(*_):
  global STOP
  STOP=True
 for s in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(s,stop)
 c=read(TARGET/'config.json')
 if proof(TARGET/'config.json')['sha256']!=expected_sha:raise ValueError('Exact typed config required')
 if (TARGET/'complete.json').exists():return read(TARGET/'complete.json')
 if (TARGET/'GPU-intent.json').exists():raise ValueError('Existing finite GPU attempt requires explicit reconciliation; no duplicate/reset')
 if not torch.cuda.is_available()or torch.cuda.device_count()!=1 or 'H100'not in torch.cuda.get_device_properties(0).name:raise ValueError('One owned H100 required')
 if shutil.disk_usage(TARGET).free<8*1024**3:raise ValueError('Remote cache/full checkpoints require >=8GiB free')
 atomic(TARGET/'GPU-intent.json',dict(job_id=os.environ.get('SLURM_JOB_ID'),pid=os.getpid(),epoch=time.time(),config_sha256=expected_sha,actor_updates=0))
 device=torch.device('cuda',0);torch.cuda.set_device(device);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
 os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20'
 for ref in c['source_files'].values():checked(ref)
 if proof(PRODUCTION/'config.json')!=c['production_config']:raise ValueError('Source production config changed')
 split=read(checked(c['split']));orders=read(checked(c['orders']))['epochs'];lookup,bank=groups_for(c)
 fit,held=split['fit_qids'],split['holdout_qids'];qids=fit+held;selected=[lookup[q]for q in qids]
 # Source first256 epochs are retained per qid; no new label-driven grouping.
 plan=read(checked(c['source_plan']));epochbyq={q:plan['batch_epochs'][i]for i,bs in enumerate(plan['batches'][:256])for q in bs}
 original_views={};primary={}
 for group in selected:
  views=g.build_training_views([group],role='train',epoch=epochbyq[group['qid']],**c['grouping'])['groups']
  if len(views)!=4:raise ValueError('Four original real G5 views required')
  original_views[group['qid']]=views;primary[group['qid']]=fixed_view(views)
 immutable(TARGET/'primary-G5-plan.json',dict(selection=c['fixed_primary_evaluation_view'],groups=[dict(qid=q,source_indices=primary[q]['source_indices'],trajectory_ids=primary[q]['trajectory_ids'])for q in qids],labels_used=False))
 source=torch.load(checked(c['source_checkpoint']),map_location='cpu',mmap=True,weights_only=False)
 if source['steps']!=10464 or source['epoch']!=1 or not source['completed_epoch']:raise ValueError('Original E1 required, no future production checkpoint')
 legacy=Probe(c['model_dir'],273,20).to(device);legacy.load_state_dict(source['raw'],strict=True);u.restore_buffers(legacy,source['raw_buffers']);legacy.eval().requires_grad_(False)
 archive=source['rank_states'];u.restore_rng(next(x['rng']for x in archive if x['rank']==0),device)
 legacy_initial=dict(raw=u.model_hash(legacy),buffers=u.tensor_hash(u.buffers(legacy)))
 if any(r['model_sha256']!=legacy_initial['raw']for r in archive):raise ValueError('Frozen E1 raw differs from archived original')
 head,ema,opt=d.initialize_head(seed=1729,device=device);init=copy.deepcopy(d.new_head_state(head,ema,opt));init_rng=u.rng_state(device)
 initial_raw=u.model_hash(head);initial_ema=u.model_hash(ema)
 if d.optimizer_steps(opt)!=0:raise ValueError('Fresh head Adam0 required')
 del source
 # Narrow reader source metadata selection. Only original1536 IDs are forwarded.
 allrefs=read(checked(bank['outputs']['features-layer26.json']));needed={t for x in selected for t in x['trajectory_ids']};refs=[]
 for ref in allrefs:
  m=read(checked(ref));index=Path(m['files']['index.jsonl']['path']);wanted=False
  with index.open()as f:
   for line in f:
    if json.loads(line)['trajectory_id']in needed:wanted=True;break
  if wanted:refs.append(ref)
 reader=ReplayFeatureReader(refs,26)
 if not needed<=set(reader.records):raise ValueError('All selected genuine trajectory IDs required')
 tids=[t for x in selected for t in x['trajectory_ids']];N=len(tids)
 H=torch.empty((N,20,3584),dtype=torch.bfloat16);M=torch.empty((N,20),dtype=torch.bool);Z=torch.empty(N,dtype=torch.float32)
 class Recorder:
  def __init__(self):self.hidden=None;self.mask=None
  def encode(self,hidden,mask):
   self.hidden=hidden.detach().clone();self.mask=mask.detach().clone();return head.encode(hidden,mask)
 recorder=Recorder();contexts=[];golden=None;cache_started=time.time()
 for start in range(0,N,c['native_micro']):
  ids=tids[start:start+c['native_micro']];batch=reader.get(ids,device)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):z,emb=d.native_hidden_forward(legacy,batch,recorder)
  if recorder.hidden.dtype!=torch.bfloat16:raise ValueError('Cache retains actual native BF16 dtype without quantization')
  H[start:start+len(ids)].copy_(recorder.hidden.cpu());M[start:start+len(ids)].copy_(recorder.mask.cpu());Z[start:start+len(ids)].copy_(z.cpu())
  if start==0:
   golden=dict(ids=ids,hidden=recorder.hidden.clone(),mask=recorder.mask.clone(),z=z.clone(),embedding=emb.clone(),positions=batch['position_ids'].clone(),attention=batch['attention_mask'].clone())
  contexts.append(dict(first=start,count=len(ids),ids_sha256=digest(ids),mask_positions_sha256=u.tensor_hash({k:batch[k]for k in ('attention_mask','position_ids')})))
  if start%640==0:atomic(TARGET/'progress.json',dict(stage='FROZEN_E1_NATIVE_P20_HIDDEN_CACHE_ADVANCING',trajectories=start+len(ids),total=N,epoch=time.time()))
  del batch,z,emb
 # Same IDs / mask / positions / native API, independently replayed, before fitting.
 batch=reader.get(golden['ids'],device)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):z,emb=d.native_hidden_forward(legacy,batch,recorder)
 checks=dict(ids=True,mask=torch.equal(batch['attention_mask'],golden['attention']),positions=torch.equal(batch['position_ids'],golden['positions']),
  hidden=torch.equal(recorder.hidden,golden['hidden']),prefix_mask=torch.equal(recorder.mask,golden['mask']),logits=torch.equal(z,golden['z']),embedding=torch.equal(emb,golden['embedding']),
  cached_hidden=torch.equal(H[:len(z)].to(device),golden['hidden']),cached_mask=torch.equal(M[:len(z)].to(device),golden['mask']))
 if not all(checks.values()):raise ValueError('Actual cached/native hidden golden mismatch')
 reader.close();del reader,batch,golden,recorder,z,emb
 if legacy_initial!=dict(raw=u.model_hash(legacy),buffers=u.tensor_hash(u.buffers(legacy))):raise ValueError('Frozen E1 changed during cache')
 path=TARGET/'cache/frozen-E1-P20-hidden.pt';path.parent.mkdir(exist_ok=True)
 torch.save(dict(hidden=H,mask=M,frozen_trajectory_logits=Z,trajectory_ids=tids,source_checkpoint=c['source_checkpoint'],context_bindings_sha256=digest(contexts)),path)
 cache=dict(status='FROZEN_E1_NATIVE_P20_CACHE_AND_SAME_API_GOLDEN_PASSED',file=proof(path),trajectories=N,questions=1536,
  context_bindings_sha256=digest(contexts),first_real_same_API_checks=checks,frozen_legacy_before=legacy_initial,seconds=time.time()-cache_started,
  initial_head_raw=initial_raw,initial_head_EMA=initial_ema,new_head_Adam=0,actor_updates=0)
 immutable(TARGET/'cache-complete.json',cache)
 H=H.to(device);M=M.to(device);Z=Z.to(device);byq={q:i for i,q in enumerate(qids)}
 def encoded(model,qs):
  ii=torch.tensor([byq[q]*10+j for q in qs for j in range(10)],device=device)
  with torch.autocast('cuda',dtype=torch.bfloat16):e=model.encode(H[ii],M[ii])
  return e.reshape(len(qs),10,128),Z[ii].reshape(len(qs),10)
 def hashes():return dict(raw=u.model_hash(head),ema=u.model_hash(ema),adam=u.optimizer_hash(opt),rng=u.rng_hash(u.rng_state(device)))
 def checkpoint(base,cursor,epoch):
  blob=dict(kind='HEAD_ONLY_CAPACITY_FULL_RAW_EMA_ADAM_RNG',head=d.new_head_state(head,ema,opt),rng=u.rng_state(device),cursor=cursor,epoch=epoch,
   learning_rate=opt.param_groups[0]['lr'],source_config=proof(TARGET/'config.json'),split=c['split'],orders=c['orders'],cache=proof(TARGET/'cache-complete.json'),
   frozen_legacy_hashes=legacy_initial,source_all_four_rank_states=archive,accepted_hashes=hashes(),data_cursor=dict(completed_updates=cursor,epoch=epoch,orders_sha256=c['orders']['sha256']))
  p=base/'checkpoints'/('cursor-%06d.pt'%cursor);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.tmp');torch.save(blob,tmp)
  with tmp.open('rb')as f:os.fsync(f.fileno())
  tmp.replace(p);ref=proof(p);loaded=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
  h=loaded['head'];accepted=dict(raw=u.model_mapping_hash(h['raw']),ema=u.model_mapping_hash(h['ema']),adam=u.optimizer_blob_hash(h['optimizer']),rng=u.rng_hash(loaded['rng']))
  if accepted!=blob['accepted_hashes']or h['steps']!=cursor:raise ValueError('Saved head raw/EMA/Adam/RNG CPU reload differs')
  receipt=dict(**ref,cursor=cursor,epoch=epoch,components=accepted,full_CPU_reload_passed=True,new_Adam_steps=cursor)
  atomic(base/'latest.json',receipt);return receipt
 def metrics(model,qs,all4):
  rows=[];sum_nll=0.;correct=0;total=0
  with torch.no_grad():
   for offset in range(0,len(qs),32):
    part=qs[offset:offset+32];em,z=encoded(model,part);items=[];indices=[]
    for i,q in enumerate(part):
     for v in original_views[q]if all4 else [primary[q]]:items.append(v);indices.append([i*10+j for j in v['source_indices']])
    idx=torch.tensor(indices,device=device);logits=model.count_logits(em.reshape(-1,128)[idx]);target=torch.tensor([sum(v['labels'])for v in items],device=device)
    losses=F.cross_entropy(logits,target,reduction='none');probs=logits.softmax(-1);scores=probs[:,1:5].sum(-1)
    sum_nll+=float(losses.sum());correct+=int((logits.argmax(-1)==target).sum());total+=len(items)
    for v,q in zip(items,scores.cpu().tolist()):rows.append(dict(qid=v['qid'],q=q,K=sum(v['labels'])))
  yy=[0<r['K']<5 for r in rows];ss=[r['q']for r in rows];pred=[x>=.5 for x in ss];pos=sum(yy);neg=len(yy)-pos
  tp=sum(a and b for a,b in zip(yy,pred));tn=sum(not a and not b for a,b in zip(yy,pred));tpr=tp/pos if pos else None;tnr=tn/neg if neg else None
  chosen=sorted(rows,key=lambda r:(-r['q'],r['qid']))[:256]if not all4 and len(qs)==1024 else []
  return dict(count_NLL=sum_nll/total,count_accuracy=correct/total,questions=len(qs),G5_views=total,related_views_not_independent_questions=all4,
   K_histogram=[sum(r['K']==k for r in rows)for k in range(6)],Mixed_fraction=pos/total,Mixed_BA=.5*(tpr+tnr)if pos and neg else None,
   Mixed_recall=tpr,nonMixed_recall=tnr,Mixed_AUC=u.roc_auc(ss,yy),predicted_Mixed_fraction=sum(pred)/total,
   Top256_of1024=dict(questions=len(chosen),Mixed_count=sum(0<r['K']<5 for r in chosen),Mixed_fraction=sum(0<r['K']<5 for r in chosen)/256)if chosen else None),rows
 def evaluate(base,epoch,cp):
  before=hashes();result={}
  for name,model in [('raw',head),('EMA',ema)]:
   result[name]={}
   for label,qs,all4 in [('fit_all4',fit,True),('fit_primaryG5',fit,False),('internal_holdout_primaryG5',held,False)]:
    m,rows=metrics(model,qs,all4);result[name][label]=m
    p=base/'evaluations'/('epoch-%02d'%epoch)/(name+'-'+label+'.jsonl.gz');p.parent.mkdir(exist_ok=True,parents=True)
    with gzip.open(p,'wt')as f:
     for row in rows:f.write(json.dumps(row)+'\n')
  if hashes()!=before:raise ValueError('Evaluation changed head/EMA/Adam/RNG')
  done=dict(epoch=epoch,cursor=epoch*16,checkpoint=cp,metrics=result,training_components_unchanged=True)
  immutable(base/'evaluations'/('epoch-%02d'%epoch)/'complete.json',done);return done
 results={};first_six={}
 for ordinal,lr in enumerate(LRS):
  base=TARGET/('lr-'+str(lr));base.mkdir(exist_ok=True)
  d.restore_new_head(init,head,ema,opt,device);u.restore_rng(init_rng,device);opt.param_groups[0]['lr']=lr
  if u.model_hash(head)!=initial_raw or u.model_hash(ema)!=initial_ema or d.optimizer_steps(opt)!=0:raise ValueError('Fork initial raw/EMA/Adam differs')
  result=[evaluate(base,0,checkpoint(base,0,0))];first=[]
  def update(qs):
   head.train();opt.zero_grad(set_to_none=True);em,z=encoded(head,qs);views=[v for q in qs for v in original_views[q]]
   loss=d.count_loss(z,em,[lookup[q]for q in qs],views,head);loss['count_NLL'].backward()
   if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in head.parameters()):raise ValueError('Nonfinite/missing head gradient')
   opt.step()
   with torch.no_grad():
    for e,p in zip(ema.parameters(),head.parameters()):e.mul_(.99).add_(p,alpha=.01)
   return dict(count_NLL=float(loss['count_NLL'].detach()),frozen_trajectory_BCE=float(loss['trajectory_loss'].detach()))
  # Real first whole update replay: same cached native IDs, complete optimizer/RNG.
  starting=copy.deepcopy(d.new_head_state(head,ema,opt));rng=u.rng_state(device);update(orders[0][0]);expected=hashes()
  d.restore_new_head(starting,head,ema,opt,device);u.restore_rng(rng,device);m=update(orders[0][0]);actual=hashes()
  if actual!=expected:raise ValueError('First real head full update replay differs')
  immutable(base/'first-update-replay.json',dict(status='PASSED',expected=expected,actual=actual,Adam_before=0,Adam_after=1,learning_rate=lr))
  cursor=1;cp=checkpoint(base,1,0);first.append(dict(cursor=cursor,new_Adam_steps=d.optimizer_steps(opt),**m))
  for ep,order in enumerate(orders,1):
   for bi,qs in enumerate(order):
    if ep==1 and bi==0:continue
    if STOP:
     cp=checkpoint(base,cursor,ep-1);atomic(TARGET/'interrupted.json',dict(lr=lr,cursor=cursor,checkpoint=cp,all_completed_forks=list(results)));return dict(status='INTERRUPTED_FULL_HEAD_STATE_SAVED')
    m=update(qs);cursor+=1
    if cursor<=6:first.append(dict(cursor=cursor,new_Adam_steps=d.optimizer_steps(opt),**m))
    atomic(TARGET/'progress.json',dict(stage='HEAD_ONLY_FIT_ADVANCING',learning_rate=lr,epoch=ep,cursor=cursor,Adam_steps=d.optimizer_steps(opt),count_NLL=m['count_NLL'],epoch_time=time.time()))
   cp=checkpoint(base,cursor,ep)
   if ep in NODES:result.append(evaluate(base,ep,cp))
  if cursor!=320 or d.optimizer_steps(opt)!=320:raise ValueError('Full20epoch/320 true updates required')
  immutable(base/'complete.json',dict(status='HEAD_ONLY_THREE_LR_ONE_FORK_COMPLETE',learning_rate=lr,checkpoint=cp,evaluations=result,first_six=first))
  results[str(lr)]=dict(checkpoint=cp,evaluations=result);first_six[str(lr)]=first
 frozen_after=dict(raw=u.model_hash(legacy),buffers=u.tensor_hash(u.buffers(legacy)))
 if frozen_after!=legacy_initial:raise ValueError('Frozen legacy raw/buffers changed')
 final=dict(status='FINITE_FROZEN_E1_HEAD_ONLY_THREE_LR_CAPACITY_COMPLETE',config=proof(TARGET/'config.json'),cache=proof(TARGET/'cache-complete.json'),
  original_frozen_legacy_before=legacy_initial,original_frozen_legacy_after=frozen_after,legacy_unchanged=True,legacy_optimizer_updates=0,actor_updates=0,
  same_initial_head_raw=initial_raw,same_initial_head_EMA=initial_ema,initial_new_Adam=0,first_actual_six_updates=first_six,results=results,
  max_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),max_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),external_payload_access=False,final_pool_access=False,
  internal_training_capacity_diagnostic_only=True)
 immutable(TARGET/'complete.json',final);return final

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','train','self-test'),required=True);p.add_argument('--config-sha256');a=p.parse_args()
 if a.action=='self-test':
  fit,held=split_questions(['qid%d'%i for i in range(8192)])
  assert len(fit)==512 and len(held)==1024 and not set(fit)&set(held)
  views=[dict(source_indices=list(x),labels=[0]*5)for x in [(1,2,3,4,5),(0,2,3,4,5),(0,1,2,3,4),(2,3,4,5,6)]]
  before=fixed_view(views)['source_indices']
  for v in views:v['labels']=[1]*5
  assert fixed_view(views)['source_indices']==before==[0,1,2,3,4]
  print('Two metadata-only split/fixed-G5 label-blind checks passed');return
 TARGET.mkdir(exist_ok=True)
 with (TARGET/('prepare.lock'if a.action=='prepare'else'diagnostic.lock')).open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:r=prepare()if a.action=='prepare'else train(a.config_sha256);print(json.dumps(r),flush=True)
  except BaseException as e:
   if a.action=='train':atomic(TARGET/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),epoch=time.time()))
   raise
if __name__=='__main__':main()
