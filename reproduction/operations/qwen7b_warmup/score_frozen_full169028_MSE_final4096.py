"""Future two-training-seed final scorer for full169028 L15P80 ownG5 MSE.

Import/tests/staging never open final data. Both completed training seeds use the
same frozen epoch and primary raw/EMA state; no best-seed selection or ensemble.
"""
import argparse,ast,copy,fcntl,gc,gzip,hashlib,inspect,json,os,time,traceback
from pathlib import Path
from . import score_frozen_final4096 as old
B,A,REVISION=old.B,old.A,old.REVISION
BASE=A/'frozen-final4096-full169028-L15P80-MSE-20261023-v2'
PREVIOUS_BASES=old.PREVIOUS_BASES+(old.BASE,A/'packed-variance-frozen-final4096-20261023-v1',A/'frozen-final4096-architectures-20261023-v1')
STATUS='TWO_TRAINING_SEEDS_SAME_CONFIGURATION_FROZEN_BEFORE_FINAL_SCORES'

def valid_ref(ref):
 return isinstance(ref,dict)and set(('path','bytes','sha256'))<=set(ref)and isinstance(ref['bytes'],int)and ref['bytes']>0 and isinstance(ref['path'],str)and ref['path'].startswith(str(B)+'/')and isinstance(ref['sha256'],str)and len(ref['sha256'])==64 and all(c in'0123456789abcdef'for c in ref['sha256'])

def dispatch(configuration):
 family=configuration.get('family');geometry=(configuration.get('layer'),configuration.get('prefix'));architecture=configuration.get('architecture');objective=configuration.get('objective');positions=configuration.get('head_max_positions')
 if family=='native_direct'and geometry in((15,80),(18,80),(21,80))and architecture=='NativeMixedGroupProbe'and objective=='mixed_bce':return'native'
 if family=='packed_variance'and geometry==(21,20)and architecture=='one_question_M3to10_policy20_packed_variance_v1'and objective in('mse','mae','z_mse_anchor')and positions==1024:return'legacy_packed'
 if geometry==(15,80)and positions==2048:
  if architecture=='one_question_packed_matched_objective_v1'and((family=='packed_variance'and objective=='mse')or(family=='packed_direct'and objective=='mixed_bce')):return'packed'
  if family=='packed_topk'and architecture=='one_question_packed_global64_detached_window_topk_v1'and objective=='topk_ramp':return'packed'
 raise ValueError('Unsupported exact architecture/geometry/objective/head capacity')

def freeze_contract(value):
 if value.get('status')!=STATUS or value.get('final_labels_or_scores_used_for_selection')is not False or value.get('further_candidate_selection_on_final_allowed')is not False or value.get('two_original_2048_G5_plans_unchanged')is not True or value.get('primary_validation_questions')!=2048 or not isinstance(value.get('freeze_epoch'),(int,float)):
  raise ValueError('Explicit future-only two-seed freeze gate is not satisfied')
 if value.get('baseline')!=dict(checkpoint=old.E1,state_key='ema',layer=27,prefix=20):raise ValueError('Original accepted E1 EMA baseline required')
 config=value['configuration'];dispatch(config)
 if config.get('family')!='packed_variance'or config.get('objective')!='mse'or config.get('training_questions')!=169028 or config.get('training_scope')!='full169028_fresh_official_actor_not_pilot_resume'or config.get('training_budget_epochs')!=10:raise ValueError('This immutable scorer requires full169028 L15P80 ownG5 MSE configuration')
 if config.get('primary_metric')!='Top512_true_Mixed_fraction'or config.get('primary_state')not in('raw','EMA')or config.get('fixed_M')!=5 or not isinstance(config.get('epoch'),int)or not 1<=config['epoch']<=config.get('training_budget_epochs',0):raise ValueError('Same fixed epoch/state, ownG5 and primary metric required')
 expected_threshold=.08 if config['family']=='packed_variance'else.5
 if config.get('threshold')!=expected_threshold or config.get('threshold_fitted')is not False:raise ValueError('Family-specific threshold must be predeclared')
 second=config.get('secondary_states',[])
 if any(s not in('raw','EMA')or s==config['primary_state']for s in second)or len(set(second))!=len(second):raise ValueError('Only explicitly predeclared other state is secondary')
 replicas=value.get('replications',[])
 if len(replicas)!=2 or len({r.get('seed_id')for r in replicas})!=2:raise ValueError('Exactly two named training replications required, never best-seed selection')
 seeds=[]
 for r in replicas:
  pair=(r.get('init_seed'),r.get('train_plan_seed'))
  if not all(type(s)is int and s>=0 for s in pair):raise ValueError('Actual two-seed identities required')
  seeds.append(pair)
  if not isinstance(r.get('seed_id'),str)or not r['seed_id']or'::'in r['seed_id']or not isinstance(r.get('checkpoint_binding'),dict):raise ValueError('Named replication and exact checkpoint binding required')
  for name in('checkpoint','source_publication','training_complete','CPU_acceptance','viewed_validation_evaluation','viewed_G5_identity_proof'):
   if not valid_ref(r.get(name)):raise ValueError('Missing exact replication proof: '+name)
  if not isinstance(r.get('train_plan_sha256'),str)or len(r['train_plan_sha256'])!=64:raise ValueError('Actual checkpoint training plan SHA required')
 if len(set(seeds))!=2 or len({r['train_plan_sha256']for r in replicas})!=2:raise ValueError('Actual training seeds and plans must differ')
 if replicas[0]['viewed_G5_identity_proof']!=replicas[1]['viewed_G5_identity_proof']:raise ValueError('Both training replications must use the same fixed viewed G5')
 if value.get('epoch_state_selection_rule')!='seed1_viewed_Top512_then_same_epoch_state_for_both_seeds'or value.get('selection_seed_id')!=replicas[0]['seed_id']:raise ValueError('Freeze epoch/state from seed1 viewed validation, then apply to both seeds')
 if not valid_ref(value.get('scorer_publication')):raise ValueError('Exact immutable scorer publication required')
 return True

def full_checkpoint_metadata(blob,configuration):
 """Pure full-budget/epoch/physical-source admission; never opens any file."""
 actual=blob.get('binding',{});epoch=configuration['epoch'];clock=1321*epoch
 required=dict(architecture='one_question_packed_matched_objective_v1',objective='mse',fixed_m=5,layer=15,prefix=80,head_max_positions=2048,training_questions=169028,training_scope='full169028_fresh_official_actor_not_pilot_resume',updates_per_epoch=1321,configured_local_batch=64,configured_micro=8,configured_accumulation=8,tail_local_batch=34,tail_global_batch=68,EMA_decay=.9995,lr=5e-6)
 if any(actual.get(k)!=v for k,v in required.items()):raise ValueError('Exact full169028 MSE geometry, objective and budget required')
 if any(blob.get(k)!=clock for k in('cursor','batch_attempts','actual_Adam_updates','EMA_updates'))or blob.get('margin_satisfied_batches')!=0:raise ValueError('Full1321-per-epoch actual Adam/EMA clock required')
 if blob.get('epoch_complete')is not True or blob.get('next_epoch')!=epoch+1 or blob.get('within_next_epoch_question_position')!=0 or blob.get('execution_phase')!='epoch_complete_pre_evaluation':raise ValueError('Complete full-data epoch recovery boundary required')
 if not all(k in blob for k in('raw','EMA','Adam','RNG_by_rank'))or len(blob['RNG_by_rank'])!=2:raise ValueError('Complete raw/EMA/Adam/two-rank RNG fields required')
 if not valid_ref(actual.get('physical_source_contract'))or not valid_ref(actual.get('fixture')):raise ValueError('Actual full input and physical-source SHA proofs required')
 if any(blob.get(k)!=actual.get(k)for k in('init_seed','train_plan_seed','diagnostic_seed','initial_head_sha256'))or actual.get('diagnostic_seed')!=2026102228:raise ValueError('Explicit full training seed/head identities differ')
 return actual

def verify_full_input_evidence(binding,acceptance):
 """Read only already accepted training-input proofs before any final access."""
 physical_ref=binding['physical_source_contract'];physical=old.read(old.checked(physical_ref))
 if physical.get('status')!='ALL169028_OWN_G10_PHYSICAL_SEAL_EXECUTION_AND_OWN_EM_RESOLVED'or physical.get('all_selected_own_EM_matches_canonical')is not True or physical.get('whole_G5_not_mixed')is not True or physical.get('selection_used_EM')is not False:raise ValueError('Full physical source execution and own EM evidence differs')
 fixture=old.checked(binding['fixture']);accepted_ref=old.proof(fixture.parent/'accepted.json');accepted=old.read(old.checked(accepted_ref))
 if accepted.get('fixture')!=binding['fixture']or accepted.get('actual_physical_execution_and_seal_source_exact')is not True or accepted.get('physical_source_contract')!=physical_ref or accepted.get('fresh_full_P20_prefix_prompt_and_own_EM_exact')is not True:raise ValueError('Independent full training input/P20 acceptance differs')
 p20=accepted['P20_alignment'];old.checked(p20)
 if not all(find_ref(acceptance,r)for r in(binding['fixture'],physical_ref,p20)):raise ValueError('Actual frozen checkpoint CPU acceptance must bind full fixture, physical source and P20 proofs')
 return dict(fixture=binding['fixture'],physical_source_contract=physical_ref,P20_alignment=p20,input_acceptance=accepted_ref)

def checkpoint_metadata(blob,configuration):
 kind=dispatch(configuration)
 if blob.get('actor_updates',0)!=0 or blob.get('epoch')!=configuration['epoch']:raise ValueError('Frozen official-actor epoch differs')
 for state in[configuration['primary_state']]+configuration.get('secondary_states',[]):
  if state not in blob:raise ValueError('Exact raw/EMA checkpoint field missing')
 if kind=='native':
  if(blob.get('layer'),blob.get('prefix'))!=(configuration['layer'],configuration['prefix']):raise ValueError('Native checkpoint geometry differs')
  return {k:blob.get(k)for k in('layer','prefix','learning_rate')}
 if kind=='legacy_packed':
  actual={k:blob.get(k)for k in('architecture','layer','prefix','loss_name')}
  if(actual['architecture'],actual['layer'],actual['prefix'],actual['loss_name'])!=(configuration['architecture'],configuration['layer'],configuration['prefix'],configuration['objective']):raise ValueError('Legacy packed checkpoint architecture differs')
  actual.update({k:blob.get(k)for k in('EMA_decay','lr','training_questions','anchor_weight','normalization_epsilon')});return actual
 actual=full_checkpoint_metadata(blob,configuration)
 if any(actual.get(k)!=configuration[k]for k in('architecture','layer','prefix','head_max_positions','objective')):raise ValueError('Packed checkpoint binding differs')
 if configuration['family']=='packed_topk'and actual.get('cache_window_capacity')not in(0,2018):raise ValueError('Unknown training queue configuration')
 return actual

def find_ref(value,wanted):
 if isinstance(value,dict):return all(value.get(k)==wanted[k]for k in('path','bytes','sha256'))or any(find_ref(v,wanted)for v in value.values())
 if isinstance(value,list):return any(find_ref(v,wanted)for v in value)
 return False

def literal_seeds(source):
 """Accepted legacy full-variance source embeds both actual seeds literally."""
 tree=ast.parse(source);plans=[];initial=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SEED'for t in node.targets)and isinstance(node.value,ast.Constant)and type(node.value.value)is int:plans.append(node.value.value)
  if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name)and node.func.value.id=='torch'and node.func.attr=='manual_seed'and node.args and isinstance(node.args[0],ast.Constant)and type(node.args[0].value)is int:initial.append(node.args[0].value)
 if len(set(plans))!=1 or len(set(initial))!=1:raise ValueError('Legacy runtime requires one actual literal init and training-plan seed')
 return initial[0],plans[0]

def verify_replications(frozen):
 """Actual CP/source/complete/CPU proofs BEFORE opening any final journal."""
 import torch
 config=frozen['configuration'];bindings=[];actual=[]
 for r in frozen['replications']:
  checkpoint=old.checked(r['checkpoint']);pub=old.read(old.checked(r['source_publication']))
  for name,ref in pub['files'].items():old.checked(dict(ref,path=str(Path(pub['code_root'])/name)))
  done=old.read(old.checked(r['training_complete']));accept=old.read(old.checked(r['CPU_acceptance']));viewed=old.read(old.checked(r['viewed_validation_evaluation']));old.checked(r['viewed_G5_identity_proof'])
  if not str(done.get('status','')).endswith('COMPLETE')or done.get('epochs')!=config['training_budget_epochs']:raise ValueError('Both training replications must actually finish the declared same budget')
  if not str(accept.get('status','')).endswith('ACCEPTED')or not find_ref(accept,r['checkpoint']):raise ValueError('Each exact frozen checkpoint needs independent CPU acceptance')
  if viewed.get('epoch')!=config['epoch']:raise ValueError('View-only epoch evidence differs from frozen epoch')
  blob=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False);binding=checkpoint_metadata(blob,config);full_input_proof=verify_full_input_evidence(binding,accept)
  if blob.get('plan_sha256')!=r['train_plan_sha256']:raise ValueError('Declared training plan does not match actual checkpoint')
  if 'checkpoint_binding'in r and binding!=r['checkpoint_binding']:raise ValueError('Frozen binding is not exact actual checkpoint metadata')
  if 'init_seed'in blob and'train_plan_seed'in blob:seeds=(blob['init_seed'],blob['train_plan_seed'])
  elif 'init_seed'in binding and'train_plan_seed'in binding:seeds=(binding['init_seed'],binding['train_plan_seed'])
  elif dispatch(config)=='legacy_packed':
   relative=r['training_runtime_relative'];ref=pub['files'][relative];source=old.checked(dict(ref,path=str(Path(pub['code_root'])/relative))).read_text();seeds=literal_seeds(source)
  else:
   # Older native checkpoints need a separately produced, SHA-bound actual seed
   # qualification. A bare handwritten seed declaration cannot satisfy this.
   seedproof=old.read(old.checked(r['actual_seed_identity_proof']))
   if seedproof.get('status')!='ACTUAL_TRAINING_SEED_AND_PLAN_IDENTITY_VERIFIED'or seedproof.get('checkpoint')!=r['checkpoint']or seedproof.get('source_publication')!=r['source_publication']or seedproof.get('plan_sha256')!=blob['plan_sha256']:raise ValueError('Actual native seed qualification required')
   seeds=(seedproof['init_seed'],seedproof['train_plan_seed'])
  if seeds!=(r['init_seed'],r['train_plan_seed']):raise ValueError('Declared seed pair differs from actual CP/published runtime')
  actual.append(dict(seed_id=r['seed_id'],init_seed=seeds[0],train_plan_seed=seeds[1],plan_sha256=blob['plan_sha256'],checkpoint=r['checkpoint'],CPU_acceptance=r['CPU_acceptance'],full_input_proof=full_input_proof));bindings.append(binding);del blob;gc.collect()
 if len({(r['init_seed'],r['train_plan_seed'])for r in actual})!=2 or len({r['plan_sha256']for r in actual})!=2:raise ValueError('Actual two-seed replication gate failed')
 # Exact geometry/objective/state/budget are global; seed-specific fixture paths,
 # init head SHA and seed counters can legitimately differ between replications.
 common=('architecture','layer','prefix','head_max_positions','objective','fixed_m','global_batch','local_batch','micro','gradient_accumulation','EMA_decay','lr','training_questions','cache_window_capacity','reference_population','target_top_k','loss_name','learning_rate','anchor_weight','normalization_epsilon','fixture','physical_source_contract','training_scope','configured_local_batch','configured_micro','configured_accumulation','tail_local_batch','tail_global_batch','updates_per_epoch')
 if any(bindings[0].get(k)!=bindings[1].get(k)for k in common):raise ValueError('Two seeds do not share one candidate training configuration')
 return dict(status='TWO_ACTUAL_COMPLETED_SEEDS_SAME_CONFIGURATION_ACCEPTED_BEFORE_FINAL_ACCESS',replications=actual,primary_state=config['primary_state'],epoch=config['epoch'],best_seed_selection=False)

def compact(records,prefix):
 out={}
 for tid,r in records.items():
  policy=[]
  for segment in r['segments']:
   n=min(prefix-len(policy),segment['policy_length']);policy+=segment['context_tokens'][segment['request_length']:segment['request_length']+n]
   if len(policy)==prefix:break
  if not policy or len(policy)!=min(prefix,r['policy_tokens']):raise ValueError('Authentic retained prefix required; cannot extend Prefix20 into80')
  out[tid]=dict(qid=r['qid'],trajectory_id=tid,prompt=r['prompt'],policy_token_ids=policy)
 return out

def score_replica(replica,config,state,views,records,device,progress,rank):
 import torch
 from torch import nn
 from transformers import AutoModel,AutoTokenizer
 from .packed_variance_probe import PackedVarianceHead,PackedVarianceProbe
 from .online_packed_matched_objective import pack_group
 if dispatch(config)=='native':return old.score_variant(dict(checkpoint=replica['checkpoint'],state_key=state,layer=config['layer'],prefix=config['prefix'],name=replica['seed_id']+'-'+state),views,records,device,progress,rank)
 model_dir=B/'assets/qwen25-7b-instruct-a09a35458c70';tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True);separator=tokenizer.encode('\n### trajectory\n',add_special_tokens=False)
 if separator!=[198,14374,34682,198]:raise ValueError('Packed training separator changed')
 actor=AutoModel.from_pretrained(str(model_dir),local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to(device).eval().requires_grad_(False);layer=config['layer'];block=copy.deepcopy(actor.layers[layer]).float();rotary=copy.deepcopy(actor.rotary_emb).float();actor.layers=nn.ModuleList(list(actor.layers[:layer]));actor.norm=nn.Identity();torch.cuda.empty_cache();probe=PackedVarianceProbe(block,rotary,'cache_position'in inspect.signature(block.forward).parameters,PackedVarianceHead(max_positions=config['head_max_positions']));blob=torch.load(old.checked(replica['checkpoint']),map_location='cpu',mmap=True,weights_only=False);checkpoint_metadata(blob,config);probe.load_state_dict(blob[state],strict=True);del blob;gc.collect();probe.to(device).eval().requires_grad_(False);tokens=compact(records,config['prefix']);rows=[];started=time.time()
 with torch.inference_mode():
  for offset in range(0,len(views),8):
   part=views[offset:offset+8];packed=[pack_group(v,tokens,separator,config['prefix'])for v in part];width=max(len(r['input_ids'])for r in packed);ids=torch.full((len(part),width),151643,dtype=torch.long,device=device);mask=torch.zeros_like(ids,dtype=torch.bool);positions=torch.zeros_like(ids);types=torch.zeros_like(ids)
   for i,r in enumerate(packed):
    n=len(r['input_ids']);ids[i,:n]=torch.tensor(r['input_ids'],device=device);mask[i,:n]=torch.tensor(r['attention_mask'],device=device);positions[i,:n]=torch.tensor(r['position_ids'],device=device);types[i,:n]=torch.tensor(r['token_types'],device=device)
   features=actor(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True).last_hidden_state.detach();data=dict(features=features,attention_mask=mask,position_ids=positions,token_types=types)
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=probe(data).cpu().tolist()
   rows.extend(dict(v,logit=z)for v,z in zip(part,logits));old.atomic(progress,dict(stage='FROZEN_TWO_SEED_PACKED_OWNG5',seed_id=replica['seed_id'],state=state,rank=rank,questions_scored=len(rows),layer=layer,prefix=config['prefix'],actor_rows_per_group=1,probe_updates=0,actor_updates=0,seconds=time.time()-started,time=time.time()));del data,features
 del actor,probe;gc.collect();torch.cuda.empty_cache();return rows

def paired_summary(rows,configuration,repeats=2000):
 import numpy as np
 from .online_packed_matched_objective import metrics
 rows=sorted(rows,key=lambda r:r['qid']);assert len(rows)==2048 and len({r['qid']for r in rows})==2048;y=np.array([r['true_mixed']for r in rows],bool);base=np.array([r['baseline_logit']for r in rows]);variance=configuration['family']=='packed_variance';objective='mse'if variance else('topk_ramp'if configuration['family']=='packed_topk'else'mixed_bce');threshold=.32 if variance else.5
 def probability(z):return np.exp(-np.logaddexp(0,-z))
 def one(z,obj):return metrics([dict(r,score=float(p))for r,p in zip(rows,probability(z))],obj)
 baseline=one(base,'mixed_bce');models={name:one(np.array([r['replica_logits'][name]for r in rows]),objective)for name in rows[0]['replica_logits']};primary=configuration['primary_state'];primary_names=[name for name in models if name.endswith('::'+primary)];assert len(primary_names)==2
 def values(z,truth,threshold):
  score=probability(z);pred=score>=threshold;pos=int(truth.sum());neg=len(truth)-pos;ba=.5*(pred[truth].mean()+(~pred[~truth]).mean());top=truth[np.argsort(-z,kind='stable')[:512]].mean();from .online_packed_variance_scaling import ranks;auc=(ranks(z)[truth].sum()-pos*(pos-1)/2)/(pos*neg);return np.array([ba,top,auc])
 actual=np.mean([values(np.array([r['replica_logits'][name]for r in rows]),y,threshold)for name in primary_names],0)-values(base,y,.5);rng=np.random.default_rng(20261023);samples=[]
 for _ in range(repeats):
  idx=rng.integers(0,len(rows),len(rows));yy=y[idx]
  if not yy.any()or yy.all():continue
  mean=np.mean([values(np.array([r['replica_logits'][name]for r in rows])[idx],yy,threshold)for name in primary_names],0);samples.append(mean-values(base[idx],yy,.5))
 ci=np.quantile(np.asarray(samples),[.025,.975],axis=0)
 return dict(baseline_E1_EMA=baseline,individual_replication_and_state=models,primary_replication_mean={key:float(np.mean([models[name][key]for name in primary_names]))for key in('balanced_accuracy','Mixed_AUC')},primary_mean_Top512_Mixed_fraction=float(np.mean([models[name]['Top512']['Mixed_fraction']for name in primary_names])),primary_replicates=primary_names,not_an_ensemble=True,best_seed_selected=False,paired_primary_replication_mean_minus_E1=dict(observed=dict(zip(('Mixed_BA','Top512_Mixed_fraction','Mixed_AUC'),actual.tolist())),percentile95_question_bootstrap={key:ci[:,i].tolist()for i,key in enumerate(('Mixed_BA','Top512_Mixed_fraction','Mixed_AUC'))}),bootstrap=dict(repeats=len(samples),seed=20261023,unit='paired_question_same_indices_all_seeds',reselect_Top512=True,training_seed_CI_not_claimed=True),threshold_fitted=False,strict_EM_used_for_ranking=False,primary_metric='Top512_true_Mixed_fraction')

def run(path,sha):
 if os.uname().sysname!='Linux'or not os.environ.get('SLURM_JOB_ID')or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise RuntimeError('Owned remote two-GPU allocation required')
 path=old.remote(path)
 if old.proof(path)['sha256']!=sha:raise ValueError('Frozen descriptor changed')
 frozen=old.read(path);freeze_contract(frozen)
 if frozen['freeze_epoch']>time.time():raise ValueError('Future timestamp is not a freeze')
 if any((b/'final-use-started.json').exists()for b in PREVIOUS_BASES):raise RuntimeError('Prior final unblinding must preserve its original frozen protocol')
 pub=old.read(old.checked(frozen['scorer_publication']));code=Path(__file__).resolve().parents[3]
 if pub['code_root']!=str(code):raise ValueError('Wrong immutable scorer source')
 for name,ref in pub['files'].items():old.checked(dict(ref,path=str(code/name)))
 import torch
 import torch.distributed as dist
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2;device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);dist.init_process_group('nccl')
 gate_payload=[None]
 if rank==0:
  official_root=B/'assets/qwen25-7b-instruct-a09a35458c70';official=old.read(official_root/'complete.json')
  if official['revision']!=REVISION or official['status']!='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED':raise ValueError('Exact accepted frozen official actor required')
  for name,ref in official['files'].items():old.checked(dict(ref,path=str(official_root/name)))
  gate_payload[0]=verify_replications(frozen) # actual two seeds BEFORE final materialization
 dist.broadcast_object_list(gate_payload,src=0);gate=gate_payload[0];BASE.mkdir(exist_ok=True);lock=(BASE/('scoring-rank-%d.lock'%rank)).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if(BASE/'final-use-started.json').exists():
  if old.read(BASE/'final-use-started.json')['freeze']!=old.proof(path):raise ValueError('Already unblinded with another frozen configuration')
 elif rank==0:old.atomic(BASE/'final-use-started.json',dict(freeze=old.proof(path),two_seed_gate=gate,time=time.time(),further_tuning_allowed=False))
 dist.barrier()
 if old.proof(old.MATERIALIZATION)['sha256']!=old.MATERIALIZATION_SHA:raise ValueError('Original fixed final identities changed')
 materialized=old.read(old.MATERIALIZATION);allq=set();out={};sources={};config=frozen['configuration'];states=[config['primary_state']]+config.get('secondary_states',[])
 for pool in('confirmation2048','stability2048'):
  views,records,source=old.load_pool(materialized['pools'][pool],rank,world)
  if allq&{v['qid']for v in views}:raise ValueError('Final pools overlap')
  allq|={v['qid']for v in views}
  for v in views:m=sum(v['labels'])/5;v.update(M=5,target_variance=m*(1-m),target4variance=4*m*(1-m))
  baseline=old.score_variant(dict(frozen['baseline'],name='baseline'),views,records,device,BASE/('progress-rank-%d.json'%rank),rank);bs={v['qid']:v['logit']for v in baseline};scored={}
  for replica in frozen['replications']:
   for state in states:
    key=replica['seed_id']+'::'+state;scored[key]={v['qid']:v['logit']for v in score_replica(replica,config,state,views,records,device,BASE/('progress-rank-%d.json'%rank),rank)}
  paired=[dict(v,baseline_logit=bs[v['qid']],replica_logits={name:z[v['qid']]for name,z in scored.items()})for v in views];gathered=[None]*world;dist.all_gather_object(gathered,paired)
  if rank==0:
   rows=sorted([v for part in gathered for v in part],key=lambda v:v['qid']);target=BASE/(pool+'-paired-predictions.jsonl.gz');tmp=target.with_suffix('.tmp')
   with gzip.open(tmp,'wt')as f:
    for row in rows:f.write(json.dumps(row)+'\n')
   tmp.replace(target);out[pool]=paired_summary(rows,config);out[pool]['predictions']=old.proof(target);sources[pool]=source;old.atomic(BASE/(pool+'-complete.json'),dict(metrics=out[pool],freeze=old.proof(path),two_seed_gate=gate))
  del records;dist.barrier()
 if rank==0:old.atomic(BASE/'complete.json',dict(status='FROZEN_FINAL4096_TWO_TRAINING_SEEDS_ARCHITECTURE_MATCHED_COMPLETE',freeze=old.proof(path),two_seed_gate=gate,pools=out,sources=sources,questions=4096,own_G5_trajectories=20480,training_replications=2,primary_state=config['primary_state'],best_seed_selection=False,probe_updates=0,actor_updates=0,queue_used_for_inference=False,threshold_fitted=False,further_candidate_selection_on_final_allowed=False,time=time.time()))
 dist.destroy_process_group()

def future_descriptor():
 """Intentionally invalid until root freezes two real, accepted replications."""
 proof=dict(path='REMOTE_ACTUAL_PROOF_REQUIRED',bytes=None,sha256=None)
 replicas=[]
 for seed in('seed1','seed2'):
  r=dict(seed_id=seed,init_seed=None,train_plan_seed=None,train_plan_sha256=None,checkpoint_binding=None,training_runtime_relative='reproduction/operations/qwen7b_warmup/ACTUAL_PUBLISHED_TRAINING_RUNTIME.py')
  for name in('checkpoint','source_publication','training_complete','CPU_acceptance','viewed_validation_evaluation','viewed_G5_identity_proof'):r[name]=dict(proof)
  replicas.append(r)
 return dict(status='FUTURE_ONLY_NOT_FROZEN',required_status=STATUS,freeze_epoch=None,final_labels_or_scores_used_for_selection=False,further_candidate_selection_on_final_allowed=False,two_original_2048_G5_plans_unchanged=True,primary_validation_questions=2048,baseline=dict(checkpoint=old.E1,state_key='ema',layer=27,prefix=20),configuration=dict(family='packed_variance',architecture='one_question_packed_matched_objective_v1',objective='mse',layer=15,prefix=80,head_max_positions=2048,training_questions=169028,training_scope='full169028_fresh_official_actor_not_pilot_resume',fixed_M=5,epoch=None,training_budget_epochs=10,primary_state=None,secondary_states=[],primary_metric='Top512_true_Mixed_fraction',threshold=.08,threshold_fitted=False),replications=replicas,epoch_state_selection_rule='seed1_viewed_Top512_then_same_epoch_state_for_both_seeds',selection_seed_id='seed1',scorer_publication=dict(proof),final_scoring_started=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze-json',required=True);p.add_argument('--freeze-sha',required=True);a=p.parse_args()
 try:run(a.freeze_json,a.freeze_sha)
 except BaseException as e:
  if os.uname().sysname=='Linux'and BASE.exists():old.atomic(BASE/('failure-rank-%s.json'%os.environ.get('RANK','unknown')),dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()))
  raise
if __name__=='__main__':main()
