"""One isolated Count data-scaling experiment, same actual CP0 and frozen E1.

53312 fit/5120 internal heldout, 1666 batches x20, plus matched18760 node.
Old351360 postblock rows are reused; only new232960 are computed. CPU mmap
batch gathering keeps the83.77GB hidden bank off the80GB GPU.
"""
import argparse,copy,fcntl,gzip,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from . import direct_count_large_capacity as old
B=old.B;A=B/'algorithm-improvement-20261021';TARGET=A/'count53312-frozenE1-data-scaling-v1';BASELINE=old.TARGET
FIT=53312;HELD=5120;BATCHES=1666;TOTAL=33320;EPOCH_NODES=(0,1,2,4,8,20);STOP=False
proof,checked,read,atomic,immutable,digest=old.proof,old.checked,old.read,old.atomic,old.immutable,old.digest
BASELINE_CONFIG=dict(path=str(BASELINE/'config.json'),bytes=7450,sha256='fb637c02812dd070cf392a58df75e0d219bbb10d1a2d24cd6e201f3638147767')
CP0_SHA='acd1c4f4bc277a8cab83847f966cfbf2462b6391b6db1d9c52c9434447f562df'


def memory_plan():
 rows=(FIT+HELD)*10;extra=23296*10
 return dict(total_hidden_bytes=rows*20*3584*2,total_trajectories=rows,old_trajectories_recomputed=0,added_trajectories_native=extra,added_hidden_bytes=extra*20*3584*2,CPU_memory_request_bytes=128*1024**3,full_bank_CUDA_resident=False,CPU_mmap_segments=2,max_GPU_batch_trajectories=320,max_batch_hidden_bytes=320*20*3584*2,max_batch_mask_logit_bytes=320*24,remote_only=True)


def pick(packet,key,other=None):
 if key in packet:return packet[key]
 if other and other in packet:return packet[other]
 raise ValueError('Missing actual input descriptor '+key)


def prepare(packet_path):
 from .count53312_cache_layout import layout
 packet=read(packet_path)
 if packet['baseline_config']!=BASELINE_CONFIG or packet['baseline_CP0']['sha256']!=CP0_SHA:raise ValueError('Exact accepted Count baseline CP0/config required')
 c0=read(checked(packet['baseline_config']));pub=read(checked(packet['baseline_publication']));baseline_cache=pick(packet,'baseline_cache','cache_complete');checked(baseline_cache)
 if pub['commit']!='805123aebf8af83a7c1c44947f39c294f7aeb624'or c0['head_LR']!=1e-4 or c0['EMA_decay']!=.99 or c0['update_budget']!=18760 or c0['grouping']!={'mode':'random','on_unbalance':'raise','seed':1729,'views_per_question':4}:raise ValueError('Same frozen baseline head/grouping hyperparameters required')
 for key in ('baseline_CP0','Count20_complete','protocol'):checked(packet[key])
 oldcohort=read(checked(c0['cohort']));oldorders=read(checked(c0['orders']))['epochs'];oldviewepochs=read(checked(c0['fit_view_epochs']))
 plan_ref=pick(packet,'scale_plan','future1666_append_plan');plan=read(checked(plan_ref));batches=plan['batches'];fit=[q for b in batches for q in b]
 if len(batches)!=BATCHES or any(len(b)!=32 for b in batches)or len(fit)!=FIT or len(set(fit))!=FIT or any(order!=batches[:938]for order in oldorders):raise ValueError('Original938 batch prefix must be exact in all20 expanded epochs')
 inputs={k:packet[k]for k in ('merged_groups','merged_features_layer26','internal_holdout_groups','internal_holdout_selection','identity_exclusion_acceptance')}
 for ref in inputs.values():checked(ref)
 groups=read(inputs['merged_groups']['path'])['train'];lookup={x['qid']:x for x in groups};held=list(oldcohort['holdout_qids']);fixed=read(inputs['internal_holdout_groups']['path'])['internal_holdout'];byheld={r['qid']:r for r in fixed}
 if len(lookup)!=58432 or set(held)!=set(byheld):raise ValueError('Exact58432 merged questions and unchanged held5120 required')
 for q in held:
  if byheld[q]['trajectory_ids']!=lookup[q]['trajectory_ids'][5:]or byheld[q]['labels']!=lookup[q]['labels'][5:]:raise ValueError('Same committed real newseed5 heldout labels/IDs required')
 locations=layout(oldcohort,fit,groups,fixed)
 counts={name:sum(old.domain(lookup[q])==name for q in fit)for name in ('NQ','HotpotQA')}
 if counts!={'NQ':39884,'HotpotQA':13428}:raise ValueError('Exact requested expanded domain counts required')
 exclusion=read(inputs['identity_exclusion_acceptance']['path'])
 if any(v!=0 for v in exclusion['merged_fixed_evaluation_qid_and_hash_overlap'].values())or any(v!=0 for v in exclusion['holdout_fixed_evaluation_qid_and_hash_overlap'].values())or exclusion['labels_consulted_for_selection']or exclusion['scores_consulted_for_selection']:raise ValueError('Accepted identity-only exclusion must remain exact')
 viewepochs={q:plan['batch_epochs'][i]for i,b in enumerate(batches)for q in b}
 if any(viewepochs[q]!=oldviewepochs[q]for q in oldcohort['fit_qids']):raise ValueError('Historical real4-view epoch assignment changed')
 TARGET.mkdir(exist_ok=True,parents=True);immutable(TARGET/'cohort.json',dict(fit_qids=fit,holdout_qids=held,fit_questions=FIT,holdout_questions=HELD,fit_domain_counts=counts,fit_qids_sha256=digest(fit),held_qids_sha256=digest(held),holdout_role='Count-fit-disjoint_internal_selection; texts_seen_in_169028_upstream_warmup; not_untouched_external'))
 immutable(TARGET/'orders.json',dict(epochs=[copy.deepcopy(batches)for _ in range(20)],original938_prefix_per_epoch=True,selection_uses_labels=False,selection_uses_scores=False));immutable(TARGET/'fit-view-epochs.json',viewepochs);immutable(TARGET/'cache-layout.json',locations)
 nodes=[dict(name='epoch-%02d'%e,cursor=e*BATCHES,epoch=e)for e in EPOCH_NODES]+[dict(name='matched-update-18760',cursor=18760,epoch=None)];nodes.sort(key=lambda r:r['cursor'])
 c=dict(kind='COUNT53312_FROZEN_E1_ACTUAL_CP0_DATA_SCALING',baseline_config=packet['baseline_config'],baseline_publication=packet['baseline_publication'],baseline_cache=baseline_cache,baseline_CP0=packet['baseline_CP0'],baseline_CP0_four_hashes=packet['baseline_CP0_four_hashes'],baseline_Count20=packet['Count20_complete'],protocol=packet['protocol'],input_packet=proof(packet_path),source_checkpoint=c0['source_checkpoint'],model_dir=c0['model_dir'],model_config=c0['model_config'],inputs=inputs,source_plan=plan_ref,cohort=proof(TARGET/'cohort.json'),orders=proof(TARGET/'orders.json'),fit_view_epochs=proof(TARGET/'fit-view-epochs.json'),cache_layout=proof(TARGET/'cache-layout.json'),grouping=c0['grouping'],head_LR=1e-4,head_weight_decay=0.,EMA_decay=.99,head_seed=1729,actual_source_head_Adam=0,source_raw_EMA_Adam_RNG_fourhashes_required=True,fit_questions=FIT,internal_holdout_questions=HELD,epochs=20,questions_per_update=32,updates_per_epoch=BATCHES,update_budget=TOTAL,evaluation_nodes=nodes,fit_views_per_question=4,fit_view_weight='1/4',prefix_tokens=20,question_max_tokens=273,feature_layer=26,block_layer=27,encode_arithmetic=c0['encode_arithmetic'],classifier_arithmetic=c0['classifier_arithmetic'],loss='frozen trajectory BCE recorded; train six-way Count NLL1 only',frozen_legacy=True,legacy_optimizer_created=False,legacy_updates=0,actor_updates=0,memory=memory_plan(),external_pool_access=False,final_pool_access=False,implementation=proof(__file__),cache_implementation=proof(Path(__file__).with_name('count53312_incremental_cache.py')),layout_implementation=proof(Path(__file__).with_name('count53312_cache_layout.py')))
 old.checked_model_config(c['model_config']);immutable(TARGET/'config.json',c);(TARGET/'logs').mkdir(exist_ok=True)
 code=Path(read(TARGET/'source-publication.json')['code_root']);script=entry(code,proof(TARGET/'config.json')['sha256']);p=TARGET/'training.sbatch'
 if p.exists()and p.read_text()!=script:raise ValueError('Immutable scale entry differs')
 if not p.exists():p.write_text(script)
 result=dict(status='COUNT53312_ACTUAL_METADATA_AND_INCREMENTAL_CACHE_PLAN_PREPARED',config=proof(TARGET/'config.json'),entry=proof(p),actual_CP0_restore_pending_on_allocation=True,new_native_trajectory_budget=232960,old_native_rows_recomputed=0,memory=c['memory'],model_tensors_loaded=False,GPU_submitted=False);immutable(TARGET/'preparation.json',result);return result


def entry(code,sha):
 return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-count53312
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=@@BASE@@/logs/train-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.count53312_train --action train --config-sha256 @@SHA@@
'''.replace('@@BASE@@',str(TARGET)).replace('@@CODE@@',str(code)).replace('@@SHA@@',sha)


def hashes(head,ema,opt,device):
 from . import probe_ablation_train as u
 return dict(raw=u.model_hash(head),ema=u.model_hash(ema),adam=u.optimizer_hash(opt),rng=u.rng_hash(u.rng_state(device)))


def state_blob(head,ema,opt,device,cursor,c,source0,cache):
 from . import direct_hidden_count as d,probe_ablation_train as u
 return dict(kind='COUNT53312_SMALL_FULL_HEAD_STATE',head=copy.deepcopy(d.new_head_state(head,ema,opt)),rng=u.rng_state(device),cursor=cursor,epoch=cursor//BATCHES,config=c['config_reference'],cache=cache,baseline_CP0=c['baseline_CP0'],source_all_four_rank_states=source0['source_all_four_rank_states'],frozen_legacy_hashes=source0['frozen_legacy_hashes'],initial_head_hashes=source0['initial_head_hashes'],accepted_hashes=hashes(head,ema,opt,device),data_cursor=dict(cursor=cursor,orders_sha256=c['orders']['sha256']),phase='accepted_update_complete')


def verify_state(state,c,source0,cache,dimension=3584,embedding=128):
 import torch
 from . import direct_count_large_small_checkpoint as small
 required={'kind','head','rng','cursor','epoch','config','cache','baseline_CP0','source_all_four_rank_states','frozen_legacy_hashes','initial_head_hashes','accepted_hashes','data_cursor','phase'}
 if set(state)!=required or state['kind']!='COUNT53312_SMALL_FULL_HEAD_STATE'or state['config']!=c['config_reference']or state['cache']!=cache or state['baseline_CP0']!=c['baseline_CP0']or state['data_cursor']!=dict(cursor=state['cursor'],orders_sha256=c['orders']['sha256'])or state['phase']!='accepted_update_complete':raise ValueError('Exact scale state/source/cache/cursor binding required')
 cursor=state['cursor'];h=state['head'];expected=small.shapes(dimension,embedding)
 if not 0<=cursor<=TOTAL or h['steps']!=cursor or state['epoch']!=cursor//BATCHES:raise ValueError('Actual scale optimizer clock differs')
 if set(h)!={'raw','ema','optimizer','steps','accepted_hashes','new_parameters_initialized_once','inherited_legacy_optimizer_reset'}or not h['new_parameters_initialized_once']or h['inherited_legacy_optimizer_reset']:raise ValueError('Original head identity/Adam inheritance differs')
 for field in ('raw','ema'):
  if set(h[field])!=set(expected)or any(tuple(t.shape)!=expected[k]or t.dtype!=torch.float32 for k,t in h[field].items()):raise ValueError('Only original seven head FP32 parameter tensors accepted')
 if [{k:v for k,v in g.items()if k!='params'}for g in h['optimizer']['param_groups']]!=[{k:v for k,v in g.items()if k!='params'}for g in source0['head']['optimizer']['param_groups']]:raise ValueError('Original CP0 Adam hyperparameters changed')
 if cursor==0 and(h['optimizer']['state']or state['accepted_hashes']!=c['baseline_CP0_four_hashes']):raise ValueError('Actual source CP0 fourhash identity required')
 if cursor>0 and set(h['optimizer']['state'])!=set(h['optimizer']['param_groups'][0]['params']):raise ValueError('All seven real Adam moment slots required')
 if small.archive_signature(state)!=small.archive_signature(source0)or state['frozen_legacy_hashes']!=source0['frozen_legacy_hashes']or state['initial_head_hashes']!=source0['initial_head_hashes']:raise ValueError('Historical four-rank source archive changed')
 return small.component_hashes(state)


def restore(state,head,ema,opt,device,c,source0,cache,dimension=3584,embedding=128):
 from . import direct_hidden_count as d,probe_ablation_train as u
 expected=verify_state(state,c,source0,cache,dimension,embedding);d.restore_new_head(state['head'],head,ema,opt,device);u.restore_rng(state['rng'],device)
 if hashes(head,ema,opt,device)!=expected:raise ValueError('Actual head/EMA/Adam/RNG restoration differs')
 return state['cursor']


def save(base,state,c,source0,cache):
 import torch
 from . import direct_count_large_small_checkpoint as small
 base=Path(base);root=base/'checkpoints';root.mkdir(exist_ok=True);p=root/('cursor-%06d.pt'%state['cursor'])
 if p.exists():
  small.zip_guard(p);prior=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
  if verify_state(prior,c,source0,cache)!=state['accepted_hashes']:raise ValueError('Existing scale checkpoint differs')
 else:
  tmp=p.with_suffix('.tmp');torch.save(state,tmp)
  with tmp.open('rb')as f:os.fsync(f.fileno())
  tmp.replace(p)
 small.zip_guard(p);ref=proof(p);loaded=torch.load(p,map_location='cpu',mmap=True,weights_only=False);values=verify_state(loaded,c,source0,cache)
 result=dict(**ref,cursor=state['cursor'],epoch=state['epoch'],accepted_hashes=values,full_CPU_reload_passed=True,new_head_Adam_steps=state['cursor']);atomic(base/'latest.json',result);return result


def resume_cursor(latest,progress=None,intent=None):
 cursor=latest['cursor']
 if max([cursor]+[v.get('cursor',0)for v in (progress,intent)if v])!=cursor or(intent and intent['phase']!='accepted_update_complete'):raise ValueError('Highest accepted or unresolved update newer than full checkpoint; no rollback')
 return cursor


def audit_first():
 import torch
 from . import direct_count_large_small_checkpoint as small
 if not os.environ.get('SLURM_JOB_ID')or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or torch.cuda.is_initialized():raise ValueError('Own allocation CPU-only small-state independent audit required')
 c=read(TARGET/'config.json');c['config_reference']=proof(TARGET/'config.json');cache=proof(TARGET/'cache-complete.json');small.zip_guard(c['baseline_CP0']['path']);source0=torch.load(checked(c['baseline_CP0']),map_location='cpu',mmap=True,weights_only=False);small.geometry(source0);small.component_hashes(source0);checks=[]
 for cursor in (0,1):
  p=TARGET/'checkpoints'/('cursor-%06d.pt'%cursor);small.zip_guard(p);state=torch.load(p,map_location='cpu',mmap=True,weights_only=False);values=verify_state(state,c,source0,cache);small.restore_small(state);checks.append(dict(cursor=cursor,file=proof(p),components=values))
 replay=read(TARGET/'first-update-replay.json')
 if checks[0]['components']!=c['baseline_CP0_four_hashes']or replay['expected']!=replay['actual']or replay['actual']!=checks[1]['components']:raise ValueError('Source CP0 and actual first update fullstate replay differ')
 result=dict(status='COUNT53312_CP1_INDEPENDENT_SMALL_FOUR_STATE_CPU_RELOAD_PASSED',config=c['config_reference'],cache=cache,baseline_CP0=c['baseline_CP0'],CP0=checks[0],CP1=checks[1],source_archive_preserved=True,old_model_payload_loaded=False,CUDA_initialized=False,GPU=0);immutable(TARGET/'first-cp-independent-audit.json',result);return result


def train(config_sha):
 import torch
 import torch.nn.functional as F
 from . import direct_hidden_count as d,probe_ablation_train as u,g10_grouping as g,direct_count_large_small_checkpoint as small,count53312_incremental_cache as cachemod
 global STOP
 c=read(TARGET/'config.json');c['config_reference']=proof(TARGET/'config.json')
 if c['config_reference']['sha256']!=config_sha or c['implementation']!=proof(__file__)or c['cache_implementation']!=proof(cachemod.__file__):raise ValueError('Published immutable typed scale source/config required')
 if(TARGET/'complete.json').exists():return read(TARGET/'complete.json')
 if not os.environ.get('SLURM_JOB_ID')or not torch.cuda.is_available()or torch.cuda.device_count()!=1 or 'H100'not in torch.cuda.get_device_properties(0).name:raise ValueError('One owned scheduled H100 required')
 def stop(*_):
  global STOP
  STOP=True
 for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 atomic(TARGET/'controller.json',dict(job_id=os.environ['SLURM_JOB_ID'],pid=os.getpid(),config=c['config_reference'],started=time.time()));device=torch.device('cuda',0);torch.cuda.set_device(device);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20';old.checked_model_config(c['model_config'])
 cohort=read(checked(c['cohort']));fit=cohort['fit_qids'];held=cohort['holdout_qids'];lookup={r['qid']:r for r in read(checked(c['inputs']['merged_groups']))['train']};layout=read(checked(c['cache_layout']));orders=read(checked(c['orders']))['epochs'];ve=read(checked(c['fit_view_epochs']));fixed={r['qid']:r for r in read(checked(c['inputs']['internal_holdout_groups']))['internal_holdout']}
 views={q:g.build_training_views([lookup[q]],role='train',epoch=ve[q],**c['grouping'])['groups']for q in fit};primary={q:old.old.fixed_view(views[q])for q in fit}
 if len(orders)!=20 or any(len(x)!=BATCHES for x in orders)or any(len(v)!=4 for v in views.values()):raise ValueError('Exact20 x1666 batches and four real views required')
 small.zip_guard(c['baseline_CP0']['path']);source0=torch.load(checked(c['baseline_CP0']),map_location='cpu',mmap=True,weights_only=False);small.geometry(source0)
 if small.component_hashes(source0)!=c['baseline_CP0_four_hashes']or source0['cursor']!=0:raise ValueError('Actual source CP0 hash/clock differs')
 head,ema,opt=d.initialize_head(seed=1729,device=device);d.restore_new_head(source0['head'],head,ema,opt,device);u.restore_rng(source0['rng'],device)
 if hashes(head,ema,opt,device)!=c['baseline_CP0_four_hashes']:raise ValueError('Real baseline raw/EMA/Adam0/RNG identity failed')
 cache_result=cachemod.build(c,source0,head,lookup,layout,TARGET,device,lambda:STOP)
 if cache_result is None:atomic(TARGET/'interrupted.json',dict(phase='only_added_native_cache',head_updates=0,original_cache_modified=False));return dict(status='COUNT53312_CACHE_INTERRUPTED_SAVED')
 segments,cache=cache_result;locations=layout['qid_locations'];d.restore_new_head(source0['head'],head,ema,opt,device);u.restore_rng(source0['rng'],device)
 if hashes(head,ema,opt,device)!=c['baseline_CP0_four_hashes']:raise ValueError('Native cache preparation must not advance actual head CP0 clock/RNG')
 def encoded(model,qs):
  H,M,Z=cachemod.gather(segments,locations,qs,device)
  with torch.autocast('cuda',dtype=torch.bfloat16):emb=model.encode(H,M)
  return emb.reshape(len(qs),10,128),Z.reshape(len(qs),10)
 def update(qs):
  head.train();opt.zero_grad(set_to_none=True);emb,z=encoded(head,qs);value=d.count_loss(z,emb,[lookup[q]for q in qs],[v for q in qs for v in views[q]],head);value['count_NLL'].backward()
  if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in head.parameters()):raise ValueError('Finite complete count-head gradient required')
  opt.step()
  with torch.no_grad():
   for e,p in zip(ema.parameters(),head.parameters()):e.mul_(.99).add_(p,alpha=.01)
  return dict(count_NLL=float(value['count_NLL'].detach()),frozen_trajectory_BCE=float(value['trajectory_loss'].detach()))
 def score(model,qs,all4=False):
  rows=[];was=model.training;model.eval()
  with torch.no_grad():
   for start in range(0,len(qs),32):
    part=qs[start:start+32];emb,z=encoded(model,part);items=[];indices=[]
    for i,q in enumerate(part):
     chosen=views[q]if all4 else[primary[q]if q in primary else fixed[q]]
     for view in chosen:
      own=[lookup[q]['trajectory_ids'].index(t)for t in view['trajectory_ids']];items.append((q,view,own));indices.append([i*10+j for j in own])
    idx=torch.tensor(indices,device=device);logits=model.count_logits(emb.reshape(-1,128)[idx]);ys=torch.tensor([sum(v['labels'])for q,v,o in items],device=device);losses=F.cross_entropy(logits,ys,reduction='none');prob=logits.softmax(-1)
    for(q,v,own),p,loss_value,pred_k,p_j in zip(items,prob.cpu().tolist(),losses.cpu().tolist(),logits.argmax(-1).cpu().tolist(),z.reshape(-1)[idx].sigmoid().cpu().tolist()):rows.append(dict(qid=q,q=sum(p[1:5]),K=sum(v['labels']),domain=old.domain(lookup[q]),trajectory_ids=v['trajectory_ids'],strict_em=v['labels'],count_probabilities=p,count_NLL=loss_value,predicted_K=pred_k,p_j=p_j))
  model.train(was)
  from .direct_count_large_screen import metrics
  result=dict(questions=len(qs),G5_views=len(rows),count_NLL=sum(r['count_NLL']for r in rows)/len(rows),count_accuracy=sum(r['predicted_K']==r['K']for r in rows)/len(rows),related_views_not_independent_questions=all4)
  if not all4:result.update(metrics(rows));result['by_domain']={name:metrics([r for r in rows if r['domain']==name])for name in ('NQ','HotpotQA')if any(r['domain']==name for r in rows)}
  return result,rows
 def evaluate(node,cp):
  folder=TARGET/'evaluations'/node['name'];folder.mkdir(parents=True,exist_ok=True);done=folder/'complete.json'
  if done.exists():
   prior=read(done)
   if prior['checkpoint']!=cp or prior['cursor']!=node['cursor']:raise ValueError('Immutable evaluation state differs')
   for ref in prior['predictions'].values():checked(ref)
   return prior
  before=hashes(head,ema,opt,device);metrics={};outputs={}
  for name,model in [('raw',head),('EMA',ema)]:
   metrics[name]={}
   for label,qs,all4 in [('fit_all4',fit,True),('fit_primaryG5',fit,False),('internal_holdout_new5',held,False)]:
    result,rows=score(model,qs,all4);metrics[name][label]=result;p=folder/(name+'-'+label+'.jsonl.gz');tmp=p.with_suffix('.partial')
    with gzip.open(tmp,'wt')as f:
     for row in rows:f.write(json.dumps(row)+'\n')
    tmp.replace(p);outputs[name+'-'+label]=proof(p);atomic(TARGET/'evaluation-progress.json',dict(node=node['name'],cursor=node['cursor'],completed_output=name+'-'+label,rows=len(rows),time=time.time()))
  if hashes(head,ema,opt,device)!=before:raise ValueError('Read-only scale evaluation changed head state')
  result=dict(node=node['name'],epoch=node['epoch'],cursor=node['cursor'],checkpoint=cp,config=c['config_reference'],cache=cache,metrics=metrics,predictions=outputs,training_components_unchanged=True,heldout_is_not_external=True,external_pool_access=False,final_pool_access=False);immutable(done,result);return result
 def save_at(cursor):return save(TARGET,state_blob(head,ema,opt,device,cursor,c,source0,cache),c,source0,cache)
 nodes={v['cursor']:v for v in c['evaluation_nodes']};cursor=0
 if(TARGET/'latest.json').exists():
  latest=read(TARGET/'latest.json');cursor=resume_cursor(latest,read(TARGET/'progress.json')if(TARGET/'progress.json').exists()else None,read(TARGET/'update-intent.json')if(TARGET/'update-intent.json').exists()else None);small.zip_guard(latest['path']);state=torch.load(checked(latest),map_location='cpu',mmap=True,weights_only=False);restore(state,head,ema,opt,device,c,source0,cache)
 else:cp=save_at(0);evaluate(nodes[0],cp)
 for step,node in nodes.items():
  if step<cursor and not(TARGET/'evaluations'/node['name']/'complete.json').exists():raise ValueError('Missing past fixed node cannot be regenerated from later weights')
 if cursor in nodes and not(TARGET/'evaluations'/nodes[cursor]['name']/'complete.json').exists():evaluate(nodes[cursor],save_at(cursor))
 if cursor==0:
  zero=state_blob(head,ema,opt,device,0,c,source0,cache);qs=orders[0][0];atomic(TARGET/'update-intent.json',dict(cursor=1,phase='started'));update(qs);expected=hashes(head,ema,opt,device);restore(zero,head,ema,opt,device,c,source0,cache);value=update(qs)
  if hashes(head,ema,opt,device)!=expected:raise ValueError('First actual scale update fullstate replay differs')
  immutable(TARGET/'first-update-replay.json',dict(status='PASSED',expected=expected,actual=hashes(head,ema,opt,device),Adam_before=0,Adam_after=1,source_CP0_four_hashes=c['baseline_CP0_four_hashes'],first_batch_sha256=digest(qs),same_first_batch_as_baseline=True));cursor=1;atomic(TARGET/'update-intent.json',dict(cursor=1,phase='accepted_update_complete'));cp=save_at(1);atomic(TARGET/'progress.json',dict(stage='COUNT53312_HEAD_ADVANCING',cursor=1,head_Adam=1,**value,time=time.time()))
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  with(TARGET/'first-cp-independent-audit.log').open('ab')as log:r=subprocess.run([sys.executable,'-u','-m',__package__+'.count53312_train','--action','audit-first'],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=45)
  if r.returncode or not(TARGET/'first-cp-independent-audit.json').exists():raise ValueError('Independent allocation CPU CP1 audit failed; do not accept update2')
 if cursor>=1:checked(proof(TARGET/'first-cp-independent-audit.json'))
 for ep,order in enumerate(orders,1):
  for bi,qs in enumerate(order):
   step=(ep-1)*BATCHES+bi+1
   if step<=cursor:continue
   if STOP:cp=save_at(cursor);atomic(TARGET/'interrupted.json',dict(phase='head_training',cursor=cursor,checkpoint=cp));return dict(status='COUNT53312_FULL_HEAD_INTERRUPTED_SAVED',cursor=cursor)
   atomic(TARGET/'update-intent.json',dict(cursor=step,phase='started',batch_sha256=digest(qs),config_sha256=config_sha));value=update(qs);cursor=step;atomic(TARGET/'update-intent.json',dict(cursor=cursor,phase='accepted_update_complete'));atomic(TARGET/'progress.json',dict(stage='COUNT53312_HEAD_ADVANCING',epoch=ep,cursor=cursor,head_Adam=d.optimizer_steps(opt),**value,time=time.time()))
   if cursor<=6 or cursor%128==0 or cursor in nodes:cp=save_at(cursor)
   if cursor in nodes:
    if STOP:atomic(TARGET/'interrupted.json',dict(phase='fixed_evaluation_pending',cursor=cursor,checkpoint=cp,evaluation_pending=[nodes[cursor]['name']]));return dict(status='COUNT53312_FULL_HEAD_INTERRUPTED_SAVED',cursor=cursor)
    evaluate(nodes[cursor],cp)
 if cursor!=TOTAL or d.optimizer_steps(opt)!=TOTAL:raise ValueError('Exact expanded20epoch33320 update budget incomplete')
 cp=save_at(cursor);evaluations=[proof(TARGET/'evaluations'/v['name']/'complete.json')for v in c['evaluation_nodes']];done=dict(status='COUNT53312_FROZEN_E1_DATA_SCALING20EPOCH_COMPLETE',config=c['config_reference'],cache=cache,checkpoint=cp,evaluations=evaluations,head_Adam=TOTAL,source_CP0=c['baseline_CP0'],old_rows_recomputed=0,added_native_rows=232960,legacy_updates=0,actor_updates=0,full_bank_CUDA_resident=False,max_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),external_pool_access=False,final_pool_access=False);immutable(TARGET/'complete.json',done);return done


def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','train','audit-first'),required=True);p.add_argument('--packet');p.add_argument('--config-sha256');a=p.parse_args();TARGET.mkdir(parents=True,exist_ok=True)
 with(TARGET/('prepare.lock'if a.action=='prepare'else'audit-first.lock'if a.action=='audit-first'else'training.lock')).open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:result=prepare(a.packet)if a.action=='prepare'else audit_first()if a.action=='audit-first'else train(a.config_sha256);print(json.dumps(result),flush=True)
  except BaseException as e:
   if a.action=='train':atomic(TARGET/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),epoch=time.time()))
   raise
if __name__=='__main__':main()
