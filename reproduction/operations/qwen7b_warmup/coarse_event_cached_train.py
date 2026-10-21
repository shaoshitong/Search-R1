"""Isolated shared-cache coarse-event forks; preparation never starts GPUs.

No native forward or legacy checkpoint load exists here. Both objectives start
from the actual baseline small CP0 and retain all four initial state hashes.
"""
import argparse,copy,fcntl,gzip,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from . import direct_count_large_capacity as shared

B=shared.B;A=B/'algorithm-improvement-20261021'
BASELINE=A/'direct-count-frozenE1-fit30016-hold5120-v2'
BASELINE_CONFIG=dict(path=str(BASELINE/'config.json'),bytes=7450,sha256='fb637c02812dd070cf392a58df75e0d219bbb10d1a2d24cd6e201f3638147767')
BASELINE_CODE='805123aebf8af83a7c1c44947f39c294f7aeb624'
MODES=('binary','three_event');NODES=(0,1,2,4,8,20);TOTAL=18760;STOP=False
proof,checked,read,atomic,immutable,digest=shared.proof,shared.checked,shared.read,shared.atomic,shared.immutable,shared.digest


def target(mode):
 if mode not in MODES:raise ValueError('Exactly one explicit prepared coarse objective')
 return A/('coarse-event-frozenE1-'+mode+'-fit30016-hold5120-v1')


def waiting_inputs(packet_path=None):
 required=[BASELINE/'cache/complete.json',BASELINE/'checkpoints/cursor-000000.pt',BASELINE/'evaluations/epoch-00/complete.json']
 missing=[str(p)for p in required if not p.is_file()]
 if packet_path is None or not Path(packet_path).is_file():missing.append('actual_family_resolved_input_packet')
 return dict(status='WAITING_FOR_VERIFIED_BASELINE_CACHE_CP0',missing=missing,prepared_inputs_passed=False,GPU_submitted=False)if missing else None


def configured(packet_path,mode):
 """Metadata/ZIP only; tensor restoration is an explicit later allocation gate."""
 wait=waiting_inputs(packet_path)
 if wait:return wait
 from . import prior_logit_event_loss as loss,direct_count_large_small_checkpoint as small
 p=read(packet_path)
 if p['role']!='FROZEN_E1_SHARED_CACHE_CP0_COARSE_FORK_INPUTS' or p['baseline_config']!=BASELINE_CONFIG:raise ValueError('Exact frozen-E1 shared input packet required')
 c0=read(checked(p['baseline_config']));pub=read(checked(p['baseline_publication']))
 if pub['commit']!=BASELINE_CODE or c0['head_LR']!=1e-4 or c0['EMA_decay']!=.99 or c0['updates_per_epoch']!=938 or c0['update_budget']!=TOTAL or c0['epochs']!=20 or c0['fit_views_per_question']!=4 or c0['fit_view_weight']!='1/4' or c0['encode_arithmetic']!='CUDA_BF16_autocast_then_float_embedding' or c0['classifier_arithmetic']!='FP32_outside_autocast':raise ValueError('Only objective may differ from current Count baseline')
 cache=read(checked(p['cache_complete']));epoch0=read(checked(p['baseline_epoch0_evaluation']))
 if p['cache_complete']['path']!=str(BASELINE/'cache/complete.json') or p['baseline_CP0']['path']!=str(BASELINE/'checkpoints/cursor-000000.pt') or p['baseline_epoch0_evaluation']['path']!=str(BASELINE/'evaluations/epoch-00/complete.json'):raise ValueError('Actual producer cache/CP0/node0 path differs')
 cp=p['baseline_CP0'];small.zip_guard(cp['path']);actual=proof(cp['path'])
 if any(actual[k]!=cp[k]for k in ('path','bytes','sha256')):raise ValueError('Actual small CP0 SHA changed')
 if cache['status']!='FROZEN_E1_LARGE_NATIVE_CACHE_COMPLETE' or cache['config_sha256']!=BASELINE_CONFIG['sha256'] or cache['frozen_legacy_before']!=cache['frozen_legacy_after'] or cache['legacy_updates']!=0 or cache['actor_updates']!=0 or cache['actual_native_hidden_dtype']!='bfloat16':raise ValueError('Complete frozen native cache provenance required')
 if epoch0['epoch']!=0 or epoch0['cursor']!=0 or not epoch0['training_components_unchanged'] or any(epoch0['checkpoint'][k]!=cp[k]for k in ('path','bytes','sha256')) or epoch0['checkpoint']['full_CPU_reload_passed']is not True or epoch0['checkpoint']['accepted_hashes']!=p['baseline_CP0_four_hashes']:raise ValueError('Actual saved CP0/four-state producer proof differs')
 if p['frozen_train_prior']!=loss.FULL938_PRIVATE:raise ValueError('Only original full938 real training-view prior')
 checked(p['frozen_train_prior']);checked(p['family_protocol'])
 for ref in cache['files'].values():
  file=Path(ref['path'])
  if file.resolve()!=file or not file.is_relative_to(BASELINE/'cache') or file.stat().st_size!=ref['bytes']:raise ValueError('Shared cache file missing/truncated; no new native forward')
 for key in ('cohort','orders','fit_view_epochs','source_plan'):checked(c0[key])
 for ref in c0['inputs'].values():checked(ref)
 c=dict(kind='SHARED_FROZEN_E1_CP0_COARSE_EVENT_HEAD_ONLY',mode=mode,baseline_config=p['baseline_config'],baseline_publication=p['baseline_publication'],baseline_CP0=cp,baseline_CP0_four_hashes=p['baseline_CP0_four_hashes'],baseline_epoch0_evaluation=p['baseline_epoch0_evaluation'],cache=p['cache_complete'],family_protocol=p['family_protocol'],frozen_train_prior=p['frozen_train_prior'],resolved_input_packet=proof(packet_path),source_E1_checkpoint=c0['source_checkpoint'],cohort=c0['cohort'],orders=c0['orders'],fit_view_epochs=c0['fit_view_epochs'],inputs=c0['inputs'],source_plan=c0['source_plan'],grouping=c0['grouping'],head_LR=1e-4,head_weight_decay=0.,EMA_decay=.99,fit_questions=30016,internal_holdout_questions=5120,questions_per_update=32,epochs=20,updates_per_epoch=938,update_budget=TOTAL,evaluation_epochs=list(NODES),fit_views_per_question=4,fit_view_weight='1/4',encode_arithmetic=c0['encode_arithmetic'],classifier_arithmetic=c0['classifier_arithmetic'],memory=c0['memory'],objective='CE(raw_event_logits + log(frozen_train_pi / tau), own_coarse_event)',desired_priors=[.5,.5]if mode=='binary'else[.25,.5,.25],inference_score='unadjusted eventsoftmax Mixed; tau-reference balanced decision score, not calibrated original-distribution Mixed probability',coarse_loss_cannot_identify_internal_K_split=True,class_resampling=False,class_weighting=False,gradient_clipping=False,legacy_or_actor_updates=0,native_forward_allowed=False,legacy_model_payload_load_allowed=False,external_pool_access=False,final_pool_access=False)
 return c


def render_entry(base,code,mode,config_sha):
 return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-coarse-@@MODE@@
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=@@BASE@@/logs/train-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.coarse_event_cached_train --action train --mode @@MODE@@ --config-sha256 @@SHA@@
'''.replace('@@BASE@@',str(base)).replace('@@CODE@@',str(code)).replace('@@MODE@@',mode).replace('@@SHA@@',config_sha)


def prepare(packet_path,mode):
 c=configured(packet_path,mode)
 if c.get('status')=='WAITING_FOR_VERIFIED_BASELINE_CACHE_CP0':return c
 base=target(mode);base.mkdir(parents=True,exist_ok=True);pub=read(base/'source-publication.json');code=Path(pub['code_root'])
 for ref in pub['files'].values():
  if 'path'in ref:checked(ref)
 c.update(implementation=proof(__file__),loss_implementation=proof(Path(__file__).with_name('prior_logit_event_loss.py')),publication=proof(base/'source-publication.json'))
 immutable(base/'config.json',c);(base/'logs').mkdir(exist_ok=True)
 script=render_entry(base,code,mode,proof(base/'config.json')['sha256']);path=base/'training.sbatch'
 if path.exists()and path.read_text()!=script:raise ValueError('Isolated immutable entry differs')
 if not path.exists():path.write_text(script)
 result=dict(status='COARSE_EVENT_SHARED_INPUT_METADATA_PREPARED_ALLOCATION_FOUR_STATE_RESTORE_PENDING',mode=mode,config=proof(base/'config.json'),entry=proof(path),baseline_CP0=c['baseline_CP0'],cache=c['cache'],CPU_tensor_payload_loaded=False,cache_payload_SHA_inherited_from_producer=True,fork_actual_four_state_restore_passed=False,GPU_submitted=False)
 immutable(base/'preparation.json',result);return result


def hashes(head,ema,opt,device):
 from . import probe_ablation_train as u
 return dict(raw=u.model_hash(head),ema=u.model_hash(ema),adam=u.optimizer_hash(opt),rng=u.rng_hash(u.rng_state(device)))


def checkpoint_blob(head,ema,opt,device,cursor,epoch,c,source0):
 from . import direct_hidden_count as d,probe_ablation_train as u
 return dict(kind='COARSE_EVENT_CACHED_HEAD_ONLY_FULL_STATE',mode=c['mode'],head=copy.deepcopy(d.new_head_state(head,ema,opt)),rng=u.rng_state(device),cursor=cursor,epoch=epoch,config=c['config_reference'],cache=c['cache'],baseline_CP0=c['baseline_CP0'],baseline_CP0_four_hashes=c['baseline_CP0_four_hashes'],source_all_four_rank_states=source0['source_all_four_rank_states'],frozen_legacy_hashes=source0['frozen_legacy_hashes'],initial_head_hashes=source0['initial_head_hashes'],accepted_hashes=hashes(head,ema,opt,device),data_cursor=dict(cursor=cursor,orders_sha256=c['orders']['sha256']),phase='accepted_update_complete',frozen_train_prior=c['frozen_train_prior'])


def verify_blob(state,c,source0,dimension=3584,embedding=128):
 from . import direct_count_large_small_checkpoint as small
 import torch
 required={'kind','mode','head','rng','cursor','epoch','config','cache','baseline_CP0','baseline_CP0_four_hashes','source_all_four_rank_states','frozen_legacy_hashes','initial_head_hashes','accepted_hashes','data_cursor','phase','frozen_train_prior'}
 if set(state)!=required:raise ValueError('Only small head state and historical RNG/digest archive allowed')
 if state['kind']!='COARSE_EVENT_CACHED_HEAD_ONLY_FULL_STATE' or state['mode']!=c['mode'] or state['config']!=c['config_reference'] or state['cache']!=c['cache'] or state['baseline_CP0']!=c['baseline_CP0'] or state['baseline_CP0_four_hashes']!=c['baseline_CP0_four_hashes'] or state['frozen_train_prior']!=c['frozen_train_prior'] or state['data_cursor']!=dict(cursor=state['cursor'],orders_sha256=c['orders']['sha256']) or state['phase']!='accepted_update_complete':raise ValueError('Exact objective/source/cache/data cursor checkpoint required')
 if not 0<=state['cursor']<=TOTAL or state['head']['steps']!=state['cursor'] or state['epoch']!=state['cursor']//938:raise ValueError('Accepted cursor/epoch/Adam mismatch')
 expected=small.shapes(dimension,embedding)
 for name in ('raw','ema'):
  if set(state['head'][name])!=set(expected)or any(tuple(t.shape)!=expected[k]or t.dtype!=torch.float32 for k,t in state['head'][name].items()):raise ValueError('Only seven new-head parameter tensors allowed')
 if set(state['head'])!={'raw','ema','optimizer','steps','accepted_hashes','new_parameters_initialized_once','inherited_legacy_optimizer_reset'}or state['head']['new_parameters_initialized_once']is not True or state['head']['inherited_legacy_optimizer_reset']is not False:raise ValueError('Head initialization/source scope differs')
 for item in state['source_all_four_rank_states']:
  if set(item)!={'rank','model_sha256','ema_sha256','optimizer_sha256','steps','rng'}:raise ValueError('Legacy model/Adam payload forbidden in archive')
 group=state['head']['optimizer']['param_groups']
 if len(group)!=1 or len(group[0]['params'])!=7 or group[0]['lr']!=1e-4 or group[0]['weight_decay']!=0:raise ValueError('Original small-head Adam hyperparameters differ')
 if [{k:v for k,v in g.items()if k!='params'}for g in group]!=[{k:v for k,v in g.items()if k!='params'}for g in source0['head']['optimizer']['param_groups']]:raise ValueError('Exact CP0 Adam hyperparameters must continue')
 slots=state['head']['optimizer']['state']
 if state['cursor']==0 and (slots or state['accepted_hashes']!=c['baseline_CP0_four_hashes']):raise ValueError('Actual CP0 four-state identity required')
 if state['cursor']>0 and set(slots)!=set(group[0]['params']):raise ValueError('Complete Adam state for every head parameter required')
 if small.archive_signature(state)!=small.archive_signature(source0)or state['frozen_legacy_hashes']!=source0['frozen_legacy_hashes']or state['initial_head_hashes']!=source0['initial_head_hashes']:raise ValueError('Frozen E1 historical four-rank archive/source changed')
 return small.component_hashes(state)


def restore_blob(state,head,ema,opt,device,c,source0,dimension=3584,embedding=128):
 from . import direct_hidden_count as d,probe_ablation_train as u
 before=verify_blob(state,c,source0,dimension,embedding);d.restore_new_head(state['head'],head,ema,opt,device);u.restore_rng(state['rng'],device)
 if hashes(head,ema,opt,device)!=before:raise ValueError('Actual new-head raw/EMA/Adam/RNG restore differs')
 return state['cursor']


def save_checkpoint(base,state,c,source0):
 import torch
 from . import direct_count_large_small_checkpoint as small
 folder=Path(base)/'checkpoints';folder.mkdir(exist_ok=True);path=folder/('cursor-%06d.pt'%state['cursor'])
 if path.exists():
  small.zip_guard(path);prior=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
  if verify_blob(prior,c,source0)!=state['accepted_hashes']:raise ValueError('Existing immutable checkpoint differs')
 else:
  tmp=path.with_suffix('.tmp');torch.save(state,tmp)
  with tmp.open('rb')as f:os.fsync(f.fileno())
  tmp.replace(path)
 small.zip_guard(path);ref=proof(path);loaded=torch.load(path,map_location='cpu',mmap=True,weights_only=False);values=verify_blob(loaded,c,source0)
 receipt=dict(**ref,cursor=state['cursor'],epoch=state['epoch'],accepted_hashes=values,full_CPU_reload_passed=True,new_head_Adam_steps=state['cursor'],mode=c['mode'])
 atomic(Path(base)/'latest.json',receipt);return receipt


def durable_resume_cursor(latest,progress=None,intent=None):
 cursor=latest['cursor'];values=[cursor]+[v.get('cursor',0)for v in (progress,intent)if v is not None]
 if max(values)!=cursor or (intent is not None and intent.get('phase')!='accepted_update_complete'):raise ValueError('Accepted or unresolved update newer than durable checkpoint; no silent rollback')
 return cursor


def audit_first(base,mode):
 """Independent allocation CPU step, only small CP0/CP1; no CUDA initialization."""
 import torch
 from . import direct_count_large_small_checkpoint as small
 if not os.environ.get('SLURM_JOB_ID')or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('Independent small-state audit must be CPU-only')
 base=Path(base);c=read(base/'config.json');c['config_reference']=proof(base/'config.json')
 if base!=target(mode)or c['mode']!=mode:raise ValueError('Own isolated typed mode required')
 small.zip_guard(c['baseline_CP0']['path']);source0=torch.load(small.checked(c['baseline_CP0']),map_location='cpu',mmap=True,weights_only=False);small.geometry(source0);small.component_hashes(source0)
 checks=[]
 for cursor in (0,1):
  path=base/'checkpoints'/('cursor-%06d.pt'%cursor);small.zip_guard(path);ref=proof(path);state=torch.load(path,map_location='cpu',mmap=True,weights_only=False);values=verify_blob(state,c,source0);small.restore_small(state);checks.append(dict(file=ref,cursor=cursor,components=values))
 replay=read(base/'first-update-replay.json')
 if checks[0]['components']!=c['baseline_CP0_four_hashes']or replay['expected']!=replay['actual']or checks[1]['components']!=replay['actual']:raise ValueError('Source CP0 identity/first whole update replay mismatch')
 result=dict(status='COARSE_EVENT_CP1_INDEPENDENT_SMALL_FOUR_STATE_CPU_RELOAD_PASSED',mode=mode,config=c['config_reference'],source_CP0=c['baseline_CP0'],CP0=checks[0],CP1=checks[1],baseline_CP0_four_hash_identity=True,source_archive_preserved=True,old_E1_payload_loaded=False,CUDA_initialized=False,GPU=0)
 immutable(base/'first-cp-independent-audit.json',result);return result


def objective_update(head,ema,opt,five_embeddings,frozen_z,own_labels,train_prior,mode,questions):
 """Shared CPU-testable/GPU update, no clipping/resampling/new hyperparameters."""
 import torch
 import torch.nn.functional as F
 from . import prior_logit_event_loss as loss
 head.train();opt.zero_grad(set_to_none=True);logits=head.count_logits(five_embeddings)
 objective=loss.question_weighted_loss(logits.reshape(questions,4,6),own_labels.reshape(questions,4,5),train_prior,mode);objective.backward()
 if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in head.parameters()):raise ValueError('Finite complete new-head gradient required')
 opt.step()
 with torch.no_grad():
  for e,p in zip(ema.parameters(),head.parameters()):e.mul_(.99).add_(p,alpha=.01)
 return dict(coarse_event_NLL=float(objective.detach()),frozen_trajectory_BCE=float(F.binary_cross_entropy_with_logits(frozen_z,own_labels.float()).detach()))


def train(mode,expected_sha):
 import torch
 import torch.nn.functional as F
 from . import direct_hidden_count as d,probe_ablation_train as u,g10_grouping as g,prior_logit_event_loss as loss,direct_count_large_small_checkpoint as small
 global STOP
 base=target(mode);config_ref=proof(base/'config.json');c=read(base/'config.json');c['config_reference']=config_ref
 if config_ref['sha256']!=expected_sha or c['mode']!=mode or proof(__file__)!=c['implementation']or proof(loss.__file__)!=c['loss_implementation']:raise ValueError('Published immutable typed source/config required')
 if(base/'complete.json').exists():return read(base/'complete.json')
 if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Own scheduled fork allocation required')
 if not torch.cuda.is_available()or torch.cuda.device_count()!=1 or 'H100'not in torch.cuda.get_device_properties(0).name:raise ValueError('One owned H100 allocation required')
 def stop(*_):
  global STOP
  STOP=True
 for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 atomic(base/'controller.json',dict(job_id=os.environ['SLURM_JOB_ID'],pid=os.getpid(),mode=mode,config=config_ref,started=time.time()))
 checked(c['family_protocol']);checked(c['resolved_input_packet'])
 device=torch.device('cuda',0);torch.cuda.set_device(device);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 # No legacy checkpoint, Probe, feature reader or native model is loaded here.
 cache=read(checked(c['cache']));cohort=read(checked(c['cohort']));fit=cohort['fit_qids'];held=cohort['holdout_qids'];qids=fit+held
 groups=read(checked(c['inputs']['merged_groups']))['train'];lookup={x['qid']:x for x in groups};fixed={x['qid']:x for x in read(checked(c['inputs']['internal_holdout_groups']))['internal_holdout']};orders=read(checked(c['orders']))['epochs'];epochs=read(checked(c['fit_view_epochs']))
 if len(fit)!=30016 or len(held)!=5120 or len(set(qids))!=35136 or len(orders)!=20 or any(len(o)!=938 or any(len(b)!=32 for b in o)or o!=orders[0]for o in orders):raise ValueError('Exact same30016/B32x938 order for20epochs required')
 tids=[t for q in qids for t in lookup[q]['trajectory_ids']]
 if len(tids)!=351360 or len(set(tids))!=351360 or cache['trajectory_ids_sha256']!=digest(tids):raise ValueError('Genuine same native cache order differs')
 for ref in cache['files'].values():checked(ref)  # Full streaming file SHA, only on this fork allocation.
 n=len(tids);files=cache['files'];H=torch.from_file(files['hidden']['path'],shared=False,size=n*20*3584,dtype=torch.bfloat16).reshape(n,20,3584);M=torch.from_file(files['mask']['path'],shared=False,size=n*20,dtype=torch.bool).reshape(n,20);Z=torch.from_file(files['logits']['path'],shared=False,size=n,dtype=torch.float32)
 free,_=torch.cuda.mem_get_info(device)
 if free<c['memory']['GPU_cache_bytes']+16*1024**3:raise ValueError('Complete cache plus16GiB reserve required, no truncation')
 H,M,Z=H.to(device),M.to(device),Z.to(device);byq={q:i for i,q in enumerate(qids)}
 views={q:g.build_training_views([lookup[q]],role='train',epoch=epochs[q],**c['grouping'])['groups']for q in fit};primary={q:shared.old.fixed_view(views[q])for q in fit}
 for q,vv in views.items():
  if len(vv)!=4 or any(v['weight_numerator']!=1 or v['weight_denominator']!=4 for v in vv):raise ValueError('Four real G5 views of weight.25 required')
  for v in vv:
   own=[lookup[q]['trajectory_ids'].index(t)for t in v['trajectory_ids']]
   if len(set(own))!=5 or [lookup[q]['labels'][j]for j in own]!=v['labels']:raise ValueError('Own real G5 trajectory EM binding differs')
 small.zip_guard(c['baseline_CP0']['path']);source0=torch.load(small.checked(c['baseline_CP0']),map_location='cpu',mmap=True,weights_only=False);small.geometry(source0);initial=small.component_hashes(source0)
 if source0['cursor']!=0 or source0['epoch']!=0 or source0['config']!=c['baseline_config']or source0['cache']!=c['cache']or initial!=c['baseline_CP0_four_hashes']or source0['frozen_legacy_hashes']!=cache['frozen_legacy_before']:raise ValueError('Actual baseline CP0 full initial four-state/cache provenance differs')
 head,ema,opt=d.initialize_head(device=device);d.restore_new_head(source0['head'],head,ema,opt,device);u.restore_rng(source0['rng'],device)
 if hashes(head,ema,opt,device)!=initial or d.optimizer_steps(opt)!=0:raise ValueError('Fork must restore actual baseline CP0 raw/EMA/Adam0/RNG exactly')
 pi=loss.frozen_full938_train_prior(mode,device=device,dtype=torch.float32)
 def encoded(model,qs):
  idx=torch.tensor([byq[q]*10+j for q in qs for j in range(10)],device=device)
  with torch.autocast('cuda',dtype=torch.bfloat16):emb=model.encode(H[idx],M[idx])
  return emb.reshape(len(qs),10,128),Z[idx].reshape(len(qs),10)
 def prepare_groups(qs,emb,z,all4):
  items=[];indices=[]
  for i,q in enumerate(qs):
   for v in views[q]if all4 else [primary[q]if q in primary else fixed[q]]:
    items.append((q,v));indices.append([i*10+lookup[q]['trajectory_ids'].index(t)for t in v['trajectory_ids']])
  idx=torch.tensor(indices,device=device);labels=torch.tensor([v['labels']for q,v in items],device=device,dtype=torch.long)
  return items,emb.reshape(-1,128)[idx],z.reshape(-1)[idx],labels
 def update(qs):
  head.train();emb,z=encoded(head,qs);items,five,zz,labels=prepare_groups(qs,emb,z,True)
  return objective_update(head,ema,opt,five,zz,labels,pi,mode,len(qs))
 def save(cursor):return save_checkpoint(base,checkpoint_blob(head,ema,opt,device,cursor,cursor//938,c,source0),c,source0)
 def score(model,qs,all4):
  rows=[];nll=0.;total=0
  with torch.no_grad():
   for start in range(0,len(qs),32):
    part=qs[start:start+32];emb,z=encoded(model,part);items,five,zz,labels=prepare_groups(part,emb,z,all4);logits=model.count_logits(five);out=loss.inference_outputs(logits,mode,zz.sigmoid());values=loss.prior_adjusted_view_losses(logits,labels,pi,mode);nll+=float(values.sum());total+=len(items)
    for(q,v),qvalue,kprobs,eventprobs,pj,lv in zip(items,out['Mixed_balanced_decision_score'].cpu().tolist(),out['unadjusted_six_count_probabilities'].cpu().tolist(),out['unadjusted_event_probabilities'].cpu().tolist(),out['trajectory_probabilities'].cpu().tolist(),values.cpu().tolist()):
     rows.append(dict(qid=q,q=qvalue,K=sum(v['labels']),domain=shared.domain(lookup[q]),trajectory_ids=v['trajectory_ids'],strict_em=v['labels'],unadjusted_six_count_probabilities=kprobs,unadjusted_event_probabilities=eventprobs,p_j=pj,coarse_event_NLL=lv,score_semantics=c['inference_score']))
  from .direct_count_large_screen import metrics
  result=dict(questions=len(qs),G5_views=total,coarse_event_NLL=nll/total,related_views_not_independent_questions=all4,score_semantics=c['inference_score'])
  if not all4:result.update(metrics(rows));result['by_domain']={name:metrics([r for r in rows if r['domain']==name])for name in ('NQ','HotpotQA')if any(r['domain']==name for r in rows)}
  return result,rows
 def evaluate(epoch,cp):
  folder=base/'evaluations'/('epoch-%02d'%epoch);folder.mkdir(parents=True,exist_ok=True);path=folder/'complete.json'
  if path.exists():
   prior=read(path)
   if prior['checkpoint']['sha256']!=cp['sha256']or prior['mode']!=mode or not prior['training_components_unchanged']:raise ValueError('Existing fixed evaluation differs')
   for ref in prior['predictions'].values():checked(ref)
   return prior
  before=hashes(head,ema,opt,device);saved_rng=u.rng_state(device);metrics={};outputs={}
  for name,model in (('raw',head),('EMA',ema)):
   metrics[name]={}
   for label,qs,all4 in (('fit_all4',fit,True),('fit_primaryG5',fit,False),('internal_holdout_new5',held,False)):
    value,rows=score(model,qs,all4);metrics[name][label]=value;p=folder/(name+'-'+label+'.jsonl.gz');tmp=p.with_suffix('.tmp')
    with gzip.open(tmp,'wt')as f:
     for row in rows:f.write(json.dumps(row)+'\n')
    tmp.replace(p);outputs[name+'-'+label]=proof(p)
  u.restore_rng(saved_rng,device)
  if hashes(head,ema,opt,device)!=before:raise ValueError('Evaluation changed full four-state snapshot')
  done=dict(mode=mode,epoch=epoch,cursor=epoch*938,checkpoint=cp,metrics=metrics,predictions=outputs,training_components_unchanged=True,score_semantics=c['inference_score'],count6_internal_split_identified=False,external_pool_access=False,final_pool_access=False)
  immutable(path,done);return done
 cursor=0
 if(base/'latest.json').exists():
  ref=read(base/'latest.json');small.zip_guard(ref['path']);state=torch.load(small.checked(ref),map_location='cpu',mmap=True,weights_only=False)
  durable_resume_cursor(ref,read(base/'progress.json')if(base/'progress.json').exists()else None,read(base/'update-intent.json')if(base/'update-intent.json').exists()else None)
  cursor=restore_blob(state,head,ema,opt,device,c,source0)
 else:
  if(base/'progress.json').exists()or(base/'update-intent.json').exists():raise ValueError('Prior update outcome exists without durable checkpoint')
  save(0)
 # Historical milestone missing can only be scored from that exact restored cursor.
 for ep in NODES:
  step=ep*938
  if step<=cursor and not(base/'evaluations'/('epoch-%02d'%ep)/'complete.json').exists():
   if step!=cursor:raise ValueError('Past evaluation missing; cannot score future weights as historical')
   evaluate(ep,save(cursor))
 if STOP:
  cp=save(cursor);atomic(base/'interrupted.json',dict(mode=mode,cursor=cursor,checkpoint=cp,evaluation_pending=[]));return dict(status='COARSE_EVENT_FULL_STATE_INTERRUPTED_SAVED',cursor=cursor)
 if cursor==0:
  state=copy.deepcopy(d.new_head_state(head,ema,opt));rng=u.rng_state(device);update(orders[0][0]);expected=hashes(head,ema,opt,device);d.restore_new_head(state,head,ema,opt,device);u.restore_rng(rng,device);value=update(orders[0][0]);actual=hashes(head,ema,opt,device)
  if actual!=expected:raise ValueError('First actual coarse objective full-state update replay differs')
  immutable(base/'first-update-replay.json',dict(status='PASSED',mode=mode,expected=expected,actual=actual,Adam_before=0,Adam_after=1));cursor=1;save(cursor)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  r=subprocess.run([sys.executable,'-m','reproduction.operations.qwen7b_warmup.coarse_event_cached_train','--action','audit-first','--mode',mode],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=60);(base/'first-cp-audit.log').write_text(r.stdout)
  if r.returncode:raise RuntimeError('Independent small CP1 reload failed: '+r.stdout[-1500:])
  atomic(base/'progress.json',dict(mode=mode,cursor=1,head_Adam=1,**value,time=time.time()))
 if not(base/'first-cp-independent-audit.json').exists():raise ValueError('Independent first CP gate missing')
 cp=read(base/'latest.json')
 for ep,order in enumerate(orders,1):
  if cursor>=ep*938:continue
  for bi,qs in enumerate(order):
   step=(ep-1)*938+bi+1
   if step<=cursor:continue
   if STOP:
    cp=save(cursor);atomic(base/'interrupted.json',dict(mode=mode,cursor=cursor,checkpoint=cp,evaluation_pending=[e for e in NODES if e*938==cursor and not(base/'evaluations'/('epoch-%02d'%e)/'complete.json').exists()]));return dict(status='COARSE_EVENT_FULL_STATE_INTERRUPTED_SAVED',cursor=cursor)
   atomic(base/'update-intent.json',dict(cursor=step,accepted_before=cursor,phase='started',mode=mode));value=update(qs);cursor=step;atomic(base/'update-intent.json',dict(cursor=cursor,phase='accepted_update_complete',mode=mode));atomic(base/'progress.json',dict(stage='COARSE_EVENT_SHARED_CACHE_HEAD_ADVANCING',mode=mode,cursor=cursor,head_Adam=d.optimizer_steps(opt),**value,time=time.time()))
   if cursor<=6 or cursor%128==0:save(cursor)
  if cursor<ep*938:continue
  cp=save(cursor)
  if ep in NODES:
   if STOP:atomic(base/'interrupted.json',dict(mode=mode,cursor=cursor,checkpoint=cp,evaluation_pending=[ep]));return dict(status='COARSE_EVENT_FULL_STATE_INTERRUPTED_SAVED',cursor=cursor)
   evaluate(ep,cp)
 if cursor!=TOTAL or d.optimizer_steps(opt)!=TOTAL:raise ValueError('20epoch18760 actual update budget incomplete')
 result=dict(status='COARSE_EVENT_SHARED_CACHE_HEAD20EPOCH_COMPLETE',mode=mode,config=config_ref,cache=c['cache'],baseline_CP0=c['baseline_CP0'],checkpoint=cp,evaluations=[proof(base/'evaluations'/('epoch-%02d'%e)/'complete.json')for e in NODES],head_Adam=TOTAL,legacy_or_actor_updates=0,native_forward_performed=False,external_pool_access=False,final_pool_access=False,score_semantics=c['inference_score'])
 immutable(base/'complete.json',result);return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','train','audit-first'),required=True);p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--packet');p.add_argument('--config-sha256');v=p.parse_args();base=target(v.mode)
 if v.action=='prepare' and waiting_inputs(v.packet):print(json.dumps(waiting_inputs(v.packet)));return
 base.mkdir(parents=True,exist_ok=True)
 with(base/('prepare.lock'if v.action=='prepare'else'audit-first.lock'if v.action=='audit-first'else'training.lock')).open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:result=prepare(v.packet,v.mode)if v.action=='prepare'else audit_first(base,v.mode)if v.action=='audit-first'else train(v.mode,v.config_sha256);print(json.dumps(result),flush=True)
  except BaseException as e:
   if v.action=='train':atomic(base/'failure.json',dict(mode=v.mode,type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),epoch=time.time()))
   raise
if __name__=='__main__':main()
