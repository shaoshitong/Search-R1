"""Fresh full169028 L15/P80 own-G5 variance MSE, copied from accepted92ec.

Same authentic G10 bank, copied official block, packed head, question budget,
EMA and optimizer. Objective and fixed/random M are the only factorial switches.
No class filtering/replay, per-trajectory target, final holdout, or cached hidden.
"""
import argparse,copy,fcntl,gzip,hashlib,inspect,json,math,os,random,time,traceback
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021'
TOKEN_FIXTURE=A/'online-mixed-layer-prefix-20261022-v1'
SEED=2026102227;DIAGNOSTIC_SEED=2026102228;ARCHITECTURE='one_question_packed_matched_objective_v1';FIXTURE_STATUS='FULL169028_AUTHENTIC_POLICY80_OWN_G10_FIXED2048_READY'
SOURCE_RUNTIME_COMMIT='92ec87b0c1a02a068517c8ed84aa4b8f5727599c'
TRAINING_QUESTIONS=169028
UPDATES_PER_EPOCH=1321

def read(p):return json.loads(Path(p).read_bytes())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);q=p.with_name(p.name+'.%d.tmp'%os.getpid());q.write_bytes(canonical(x)+b'\n');q.replace(p)
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for z in iter(lambda:f.read(8388608),b''):h.update(z)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 p=Path(ref['path']);assert proof(p)=={k:ref[k]for k in('path','bytes','sha256')};return p

def retained_policy(row):
 """Flatten real retained policy-token prefixes; never extend a Prefix20 bank."""
 tokens=[]
 for s in row['segments']:
  n=min(80-len(tokens),s['policy_length']);tokens+=s['context_tokens'][s['request_length']:s['request_length']+n]
  if len(tokens)==80:break
 assert len(tokens)==min(80,row['policy_tokens'])and 0<len(tokens)<=80
 return tokens

def prepare(args):
 raise RuntimeError('Full169028 authentic Policy80 fixture is independently produced and accepted; no pilot prepare or token fabrication')

def epoch_plan(groups,epoch,fixed_m,*,train_plan_seed):
 from .packed_variance_probe import view
 rows=[view(groups[q],seed=train_plan_seed,epoch=epoch,fixed_m=fixed_m or None)for q in sorted(groups)]
 random.Random(int(digest(['packed-question-order',train_plan_seed,epoch]),16)).shuffle(rows)
 assert len(rows)==len({r['qid']for r in rows})==169028
 return rows

def pack_group(v,records,separator,prefix):
 own=[records[t]for t in v['trajectory_ids']];prompt=own[0]['prompt'];assert prompt and all(r['prompt']==prompt for r in own)
 tokens=list(prompt);valid=[True]*len(prompt);types=[0]*len(prompt)
 for r in own:
  policy=r['policy_token_ids'][:prefix];assert 0<len(policy)<=prefix
  tokens+=separator+policy+[151643]*(prefix-len(policy));valid+=[True]*len(separator)+[True]*len(policy)+[False]*(prefix-len(policy));types+=[2]*len(separator)+[1]*prefix
 positions=[];counter=0
 for flag in valid:positions.append(counter if flag else 0);counter+=int(flag)
 assert counter<=2048
 return dict(input_ids=tokens,attention_mask=valid,position_ids=positions,token_types=types)

def metrics(rows,objective):
 import numpy as np
 from .online_packed_variance_scaling import ranks
 y=np.array([r['true_mixed']for r in rows],bool);s=np.array([r['score']for r in rows],float);v=np.array([r['target_variance']for r in rows]);threshold=.32 if objective=='mse'else.5;pred=s>=threshold
 tp=int((pred&y).sum());tn=int((~pred&~y).sum());pos=int(y.sum());neg=len(rows)-pos;order=sorted(rows,key=lambda r:(-r['score'],r['qid']));top=order[:512]
 auc=float((ranks(s)[y].sum()-pos*(pos-1)/2)/(pos*neg));rho=float(np.corrcoef(ranks(s),ranks(v))[0,1])if np.std(s)>0 and np.std(v)>0 else None;eps=np.finfo(float).eps;q=np.clip(s,eps,1-eps)
 result=dict(questions=len(rows),objective=objective,score_semantics={'mse':'4*predicted_population_variance','mixed_bce':'direct_Mixed_probability','topk_ramp':'sigmoid_ranking_score_not_calibrated_Mixed_probability'}[objective],Mixed_AUC=auc,variance_Spearman=rho,balanced_accuracy=.5*(tp/pos+tn/neg),Mixed_recall=tp/pos,nonMixed_false_positive_rate=1-tn/neg,confusion=dict(tp=tp,tn=tn,fp=neg-tn,fn=pos-tp),threshold=threshold,threshold_fitted=False,true_Mixed_fraction=float(y.mean()),Top512=dict(Mixed=sum(r['true_mixed']for r in top),questions=512,Mixed_fraction=sum(r['true_mixed']for r in top)/512),classwise={})
 if objective=='mse':result.update(variance_MSE=float(((s/4-v)**2).mean()),variance_MAE=float(np.abs(s/4-v).mean()),target4variance_MSE=float(((s-4*v)**2).mean()))
 elif objective=='mixed_bce':result['binary_Mixed_BCE']=float(-(y*np.log(q)+(1-y)*np.log1p(-q)).mean())
 for name,mask in(('all_wrong',np.array([sum(r['labels'])==0 for r in rows])),('Mixed',y),('all_correct',np.array([sum(r['labels'])==len(r['labels'])for r in rows]))):result['classwise'][name]=dict(questions=int(mask.sum()),predicted_Mixed=int(pred[mask].sum()),predicted_Mixed_fraction=float(pred[mask].mean())if mask.any()else None,mean_score=float(s[mask].mean())if mask.any()else None)
 return result

def seed_model_rng(init_seed):
 import numpy as np
 import torch
 if type(init_seed)is not int or not 0<=init_seed<2**32:raise ValueError('Explicit32-bit model initialization seed required')
 torch.manual_seed(init_seed);np.random.seed(init_seed);random.seed(init_seed)

def head_state_sha256(head):
 h=hashlib.sha256()
 for name,value in head.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def enforce_start_mode(base,start_mode):
 latest=Path(base)/'latest.json'
 if start_mode=='fresh'and latest.exists():raise ValueError('Fresh MSE-M5 seed replica cannot resume an existing checkpoint')
 if start_mode=='resume'and not latest.exists():raise ValueError('Exact resume requires this replica checkpoint')

def full_acceptance_alignment_ref(accepted):
 if accepted.get('fresh_full_P20_prefix_prompt_and_own_EM_exact')is not True:
  raise ValueError('Fresh full P20 prefix/prompt/ownEM alignment must pass before training')
 return accepted['P20_alignment']

def full_acceptance_physical_source_ref(accepted):
 if accepted.get('actual_physical_execution_and_seal_source_exact')is not True:
  raise ValueError('Actual physical execution/full SHA/own EM and complete seal resolution required')
 return accepted['physical_source_contract']

def validate_physical_source_contract(contract):
 if contract.get('status')!='ALL169028_OWN_G10_PHYSICAL_SEAL_EXECUTION_AND_OWN_EM_RESOLVED'or contract.get('all_selected_own_EM_matches_canonical')is not True or contract.get('whole_G5_not_mixed')is not True or contract.get('selection_used_EM')is not False:
  raise ValueError('Complete ownG10 physical source execution/seal and label contract is not accepted')

def actual_local_batch_layout(local_questions,world_size=2,micro=8):
 if not 1<=local_questions<=64 or world_size!=2 or micro!=8:raise ValueError('Two matched ranks with actual local1..64 questions required')
 return dict(actual_global_batch=local_questions*world_size,local_batch=local_questions,micro=micro,gradient_accumulation=(local_questions+micro-1)//micro,micro_question_counts=[min(micro,local_questions-i)for i in range(0,local_questions,micro)])

def local_mean_mse(logits,targets,actual_local_questions):
 # DDP averages equal-size rank means: tail34+34 therefore gives exact68 mean.
 return (__import__('torch').sigmoid(logits).float()-targets).square().sum()/actual_local_questions

def verify_resume_boundary(old,binding,groups,args,initial_head_sha256):
 if old.get('binding')!=binding or old.get('init_seed')!=args.init_seed or old.get('train_plan_seed')!=args.train_plan_seed or old.get('diagnostic_seed')!=DIAGNOSTIC_SEED or old.get('initial_head_sha256')!=initial_head_sha256:raise ValueError('Exact full experiment/model/fixture/seed binding required; no pilot resume')
 epoch=old.get('epoch');cursor=old.get('cursor')
 if type(epoch)is not int or not 1<=epoch<=args.epochs or cursor!=epoch*UPDATES_PER_EPOCH or old.get('batch_attempts')!=cursor or old.get('actual_Adam_updates')!=cursor or old.get('EMA_updates')!=cursor or old.get('margin_satisfied_batches')!=0 or len(old.get('RNG_by_rank',[]))!=2:raise ValueError('Complete epoch boundary/Adam/EMA/two-rank RNG counters required')
 if old.get('epoch_complete')is not True or old.get('next_epoch')!=epoch+1 or old.get('within_next_epoch_question_position')!=0:raise ValueError('Explicit saved complete data cursor required')
 if old.get('plan_sha256')!=digest(epoch_plan(groups,epoch,5,train_plan_seed=args.train_plan_seed)):raise ValueError('Exact saved full ownG5 plan differs')
 for rng in old['RNG_by_rank']:
  if set(rng)!={'python','numpy','torch','cuda'}:raise ValueError('All four saved RNG streams required per rank')
 for name in('raw','EMA','Adam'):
  if name not in old:raise ValueError('Complete raw/EMA/Adam checkpoint required')

def run(args):
 import numpy as np
 import torch
 import torch.distributed as dist
 from torch import nn
 from torch.nn.parallel import DistributedDataParallel as DDP
 from transformers import AutoModel,AutoTokenizer
 from .packed_variance_probe import PackedVarianceProbe,PackedVarianceHead
 base=args.base;base.mkdir(exist_ok=True,parents=True);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2;device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);dist.init_process_group('nccl');guard=(base/('writer-rank-%d.lock'%rank)).open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 fixture_base=args.fixture_base;fixture=read(fixture_base/'fixture-complete.json');assert fixture['status']==FIXTURE_STATUS and fixture['training_questions']==169028 and fixture['max_real_policy_prefix']==80;accepted=read(fixture_base/'accepted.json');assert str(accepted['status']).endswith('ACCEPTED')and accepted['fixture']==proof(fixture_base/'fixture-complete.json'),'Actual full fixture independent CPU acceptance required';physical_source_ref=full_acceptance_physical_source_ref(accepted);validate_physical_source_contract(read(checked(physical_source_ref)));checked(full_acceptance_alignment_ref(accepted));assert fixture['validation_groups']['sha256']=='50f1e3a6927d02c0f8cadf1feb5b4229b075b5d099fe31fd0de5c2ae3b149292'and fixture['fixed_train_G5']['sha256']=='e6e74d92160efb0740ba5c0316c4ee7686f06a499c90cafd5a545f325b62209b','Original fixed devG5 and trainB identity must remain unchanged';groups=read(checked(fixture['train_groups']));validation=read(checked(fixture['validation_groups']));assert len(groups)==169028 and len(validation)==2048 and not(set(groups)&set(validation));assert all(len(g['trajectory_ids'])==len(g['labels'])==10 for g in groups.values())and all(len(g['trajectory_ids'])==len(g['labels'])==5 for g in validation.values());records={}
 for i,ref in enumerate(fixture['token_shards']):
  if i%world!=rank:continue
  with gzip.open(checked(ref),'rt')as f:
   for line in f:r=json.loads(line);records[r['trajectory_id']]=r
 seed_model_rng(args.init_seed);enforce_start_mode(base,args.start_mode);model_dir=B/'assets/qwen25-7b-instruct-a09a35458c70';tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True);separator=tokenizer.encode('\n### trajectory\n',add_special_tokens=False)
 actor=AutoModel.from_pretrained(str(model_dir),local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to(device).eval().requires_grad_(False);block=copy.deepcopy(actor.layers[args.layer]).float().requires_grad_(True);rotary=copy.deepcopy(actor.rotary_emb).float();actor.layers=nn.ModuleList(list(actor.layers[:args.layer]));actor.norm=nn.Identity();torch.cuda.empty_cache()
 head=PackedVarianceHead(max_positions=2048);initial_head_sha256=head_state_sha256(head);
 probe=PackedVarianceProbe(block,rotary,'cache_position'in inspect.signature(block.forward).parameters,head).to(device);ema=copy.deepcopy(probe).eval().requires_grad_(False);optim=torch.optim.Adam(probe.parameters(),lr=args.lr);ddp=DDP(probe,device_ids=[device.index],broadcast_buffers=False);cursor=0;actual_Adam_updates=0;EMA_updates=0;margin_satisfied_batches=0;start_epoch=1
 binding=dict(architecture=ARCHITECTURE,objective=args.objective,fixed_m=args.fixed_m,layer=args.layer,capture_layer=args.layer-1,prefix=args.prefix,EMA_decay=args.ema_decay,lr=args.lr,global_batch=128,head_max_positions=2048,initial_head_sha256=initial_head_sha256,official_actor_revision='a09a35458c702b33eeacc393d103063234e8bc28',training_questions=169028,seed=args.train_plan_seed,init_seed=args.init_seed,train_plan_seed=args.train_plan_seed,diagnostic_seed=DIAGNOSTIC_SEED,seed_source='explicit_CLI',fixture=proof(fixture_base/'fixture-complete.json'),physical_source_contract=physical_source_ref,training_scope='full169028_fresh_official_actor_not_pilot_resume',configured_local_batch=64,configured_micro=8,configured_accumulation=8,tail_local_batch=34,tail_global_batch=68,updates_per_epoch=1321,sampling_independent_of_EM=True,class_weighting=False,question_prompt_occurrences_per_group=1,topk_margin=args.margin if args.objective=='topk_ramp'else None,topk_scope='actual_global_optimizer_batch_autograd_all_gather'if args.objective=='topk_ramp'else None,zero_ramp_loss_skips_Adam_and_EMA=True)
 if(base/'latest.json').exists():
  old=torch.load(checked(read(base/'latest.json')),map_location='cpu',weights_only=False);verify_resume_boundary(old,binding,groups,args,initial_head_sha256);assert old['init_seed']==args.init_seed and old['train_plan_seed']==args.train_plan_seed and old['diagnostic_seed']==DIAGNOSTIC_SEED and old['initial_head_sha256']==initial_head_sha256;assert old['binding']==binding;probe.load_state_dict(old['raw']);ema.load_state_dict(old['EMA']);optim.load_state_dict(old['Adam']);cursor=old['cursor'];actual_Adam_updates=old['actual_Adam_updates'];EMA_updates=old['EMA_updates'];margin_satisfied_batches=old['margin_satisfied_batches'];assert old['batch_attempts']==cursor and EMA_updates==actual_Adam_updates and actual_Adam_updates+margin_satisfied_batches==cursor;start_epoch=old['epoch']+1
  for state in optim.state.values():
   for k,value in state.items():
    if torch.is_tensor(value)and k!='step':state[k]=value.to(device)
  rng=old['RNG_by_rank'][rank];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'],device);del old
 qrank={q:i%world for role,gs in(('train',groups),('validation',validation))for i,q in enumerate(sorted(gs))}
 if rank==0:atomic(base/'controller.json',dict(binding,pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],world_size=world,GPU_name=torch.cuda.get_device_name(device),fresh_official_block_and_head=start_epoch==1,raw_updates_EMA=True,hidden_features_saved=False,final4096_read=False,time=time.time()))
 def capture(vs):
  start=time.monotonic();packed=[pack_group(v,records,separator,args.prefix)for v in vs];width=max(len(r['input_ids'])for r in packed);ids=torch.full((len(vs),width),151643,dtype=torch.long,device=device);mask=torch.zeros_like(ids,dtype=torch.bool);positions=torch.zeros_like(ids);types=torch.zeros_like(ids)
  for i,r in enumerate(packed):
   n=len(r['input_ids']);ids[i,:n]=torch.tensor(r['input_ids'],device=device);mask[i,:n]=torch.tensor(r['attention_mask'],device=device);positions[i,:n]=torch.tensor(r['position_ids'],device=device);types[i,:n]=torch.tensor(r['token_types'],device=device)
  with torch.no_grad():features=actor(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True).last_hidden_state.detach()
  return dict(features=features,attention_mask=mask,position_ids=positions,token_types=types),time.monotonic()-start
 def explicit(g):
  labels=g['labels'];m=len(labels);mean=sum(labels)/m;return dict(g,M=m,true_mixed=0<sum(labels)<m,target_variance=mean*(1-mean),target4variance=4*mean*(1-mean))
 sets={'fixed_train2048_G5':[explicit(g)for g in read(checked(fixture['fixed_train_G5']))],'fixed_validation2048_G5':[explicit(g)for g in validation.values()]}
 def evaluate(epoch):
  probe.eval();start=time.time();out={}
  for role,allviews in sets.items():
   own=[v for v in allviews if qrank[v['qid']]==rank];rows=[]
   for offset in range(0,len(own),8):
    vs=own[offset:offset+8];data,_=capture(vs)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):raw=torch.sigmoid(probe(data)).cpu().tolist();av=torch.sigmoid(ema(data)).cpu().tolist()
    rows.extend([dict(qid=v['qid'],trajectory_ids=v['trajectory_ids'],labels=v['labels'],M=v['M'],true_mixed=v['true_mixed'],target_variance=v['target_variance'],raw_score=x,EMA_score=y)for v,x,y in zip(vs,raw,av)])
   received=[None]*world;dist.all_gather_object(received,rows)
   if rank==0:
    allrows=[r for rs in received for r in rs];assert len(allrows)==2048 and len({r['qid']for r in allrows})==2048
    with gzip.open(base/('epoch-%02d-%s-predictions.jsonl.gz'%(epoch,role)),'wt')as f:
     for row in allrows:f.write(json.dumps(row)+'\n')
    out[role]={state:metrics([dict(r,score=r[state+'_score'])for r in allrows],args.objective)for state in('raw','EMA')}
   atomic(base/('evaluation-progress-rank-%d.json'%rank),dict(epoch=epoch,role=role,own_questions=len(own),seconds=time.time()-start))
  if rank==0:atomic(base/('evaluation-epoch-%02d.json'%epoch),dict(epoch=epoch,cursor=cursor,batch_attempts=cursor,actual_Adam_updates=actual_Adam_updates,EMA_updates=EMA_updates,margin_satisfied_batches=margin_satisfied_batches,binding=binding,metrics=out,seconds=time.time()-start,final4096_scored=False))
  dist.barrier();probe.train()
 if start_epoch==1:evaluate(0)
 elif not(base/('evaluation-epoch-%02d.json'%(start_epoch-1))).exists():evaluate(start_epoch-1)
 for epoch in range(start_epoch,args.epochs+1):
  fullplan=epoch_plan(groups,epoch,args.fixed_m,train_plan_seed=args.train_plan_seed);mine=[v for v in fullplan if qrank[v['qid']]==rank];assert len(mine)==84514;start=time.time()
  if rank==0:atomic(base/('sampling-plan-epoch-%02d.json'%epoch),dict(epoch=epoch,init_seed=args.init_seed,train_plan_seed=args.train_plan_seed,diagnostic_seed=DIAGNOSTIC_SEED,plan_sha256=digest(fullplan),rows=fullplan))
  for offset in range(0,len(mine),64):
   batch=mine[offset:offset+64];optim.zero_grad(set_to_none=True);total=0.;prefill=0.;actor_rows=0;local_logits=[];local_labels=[];loss_details={}
   for lo in range(0,len(batch),8):
    vs=batch[lo:lo+8];data,cost=capture(vs);prefill+=cost;actor_rows+=len(vs);target=torch.tensor([v['target4variance']if args.objective=='mse'else float(v['true_mixed'])for v in vs],device=device,dtype=torch.float32)
    with torch.autocast('cuda',dtype=torch.bfloat16):logit=ddp(data)
    if args.objective=='topk_ramp':local_logits.append(logit.float());local_labels.append(target)
    else:
     loss=local_mean_mse(logit,target,len(batch))if args.objective=='mse'else torch.nn.functional.binary_cross_entropy_with_logits(logit.float(),target,reduction='sum')/len(batch)
     loss.backward();total+=float(loss.detach())
   if args.objective=='topk_ramp':
    from torch.distributed.nn.functional import all_gather
    from .group_topk_losses import precision_at_k_ramp
    scores=torch.cat(all_gather(torch.cat(local_logits).contiguous())).float();labels_local=torch.cat(local_labels).contiguous();label_parts=[torch.empty_like(labels_local)for _ in range(world)];dist.all_gather(label_parts,labels_local);labels=torch.cat(label_parts).float();actual_global=len(batch)*world;assert scores.numel()==labels.numel()==actual_global;k=(actual_global+3)//4
    loss=precision_at_k_ramp(scores,labels,k,margin=args.margin);loss.backward();total=float(loss.detach());loss_details=dict(topk_global_questions=actual_global,topk_K=k,topk_margin=args.margin,topk_scope='actual_global_optimizer_batch_autograd_all_gather',global_true_Mixed=int(labels.sum()),zero_loss_margin_satisfied=total==0.,score_is_logit=True,global2048_theorem_claimed=False)
   ng=math.sqrt(sum(float(p.grad.detach().float().square().sum())for p in probe.block.parameters()if p.grad is not None));hg=math.sqrt(sum(float(p.grad.detach().float().square().sum())for p in probe.group_head.parameters()if p.grad is not None));assert math.isfinite(ng)and math.isfinite(hg)and((ng>0 and hg>0)or args.objective=='topk_ramp');torch.nn.utils.clip_grad_norm_(probe.parameters(),1.)
   # Backward still runs on every rank to complete DDP synchronization. A
   # satisfied global margin skips Adam AND EMA, preserving old moments without
   # allowing their momentum to move parameters after the objective is zero.
   update=not(args.objective=='topk_ramp'and total==0.)
   if update:
    optim.step();actual_Adam_updates+=1
    with torch.no_grad():
     for p,e in zip(probe.parameters(),ema.parameters()):e.mul_(args.ema_decay).add_(p,alpha=1-args.ema_decay)
    EMA_updates+=1
   else:margin_satisfied_batches+=1
   cursor+=1;assert actual_Adam_updates+margin_satisfied_batches==cursor and EMA_updates==actual_Adam_updates;row=dict(stage='FULL169028_P80_OWNG5_MSE_UPDATES',binding=binding,epoch=epoch,cursor=cursor,batch_attempts=cursor,actual_Adam_updates=actual_Adam_updates,EMA_updates=EMA_updates,margin_satisfied_batches=margin_satisfied_batches,Adam_and_EMA_step_performed=update,rank=rank,epoch_questions_seen=offset+len(batch),training_loss=total,native_gradient_L2=ng,head_gradient_L2=hg,**actual_local_batch_layout(len(batch),world,8),loss_details=loss_details,actor_rows=actor_rows,prefill_rows_equal_groups=actor_rows==len(batch),prefill_seconds=prefill,time=time.time());atomic(base/('progress-rank-%d.json'%rank),row)
   if rank==0:
    print(json.dumps(row),flush=True)
    with(base/'curve.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
  assert cursor==epoch*UPDATES_PER_EPOCH and actual_Adam_updates==EMA_updates==cursor and margin_satisfied_batches==0
  rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device));rngs=[None]*world;dist.all_gather_object(rngs,rng)
  if rank==0:
   path=base/('checkpoint-epoch-%02d.pt'%epoch);tmp=path.with_suffix('.tmp');torch.save(dict(init_seed=args.init_seed,train_plan_seed=args.train_plan_seed,diagnostic_seed=DIAGNOSTIC_SEED,initial_head_sha256=initial_head_sha256,raw=probe.state_dict(),EMA=ema.state_dict(),Adam=optim.state_dict(),RNG_by_rank=rngs,cursor=cursor,batch_attempts=cursor,actual_Adam_updates=actual_Adam_updates,EMA_updates=EMA_updates,margin_satisfied_batches=margin_satisfied_batches,epoch=epoch,epoch_complete=True,next_epoch=epoch+1,within_next_epoch_question_position=0,execution_phase='epoch_complete_pre_evaluation',plan_sha256=digest(fullplan),binding=binding,actor_updates=0),tmp);tmp.replace(path);atomic(base/'latest.json',proof(path));atomic(base/('coverage-epoch-%02d.json'%epoch),dict(epoch=epoch,questions=169028,unique_questions=169028,trajectory_visits=sum(v['M']for v in fullplan),M_counts={m:sum(v['M']==m for v in fullplan)for m in range(3,11)},Mixed=sum(v['true_mixed']for v in fullplan),init_seed=args.init_seed,train_plan_seed=args.train_plan_seed,diagnostic_seed=DIAGNOSTIC_SEED,plan_sha256=digest(fullplan),sampling_independent_of_EM=True,seconds=time.time()-start))
  dist.barrier();evaluate(epoch)
 if rank==0:atomic(base/'complete.json',dict(status='FULL169028_P80_OWNG5_MSE_TRAINING_COMPLETE',epochs=args.epochs,cursor=cursor,batch_attempts=cursor,actual_Adam_updates=actual_Adam_updates,EMA_updates=EMA_updates,margin_satisfied_batches=margin_satisfied_batches,training_questions=169028,binding=binding,final4096_scored=False))
 dist.destroy_process_group()

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('train',),required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--fixture-base',type=Path);p.add_argument('--objective',choices=('mse',),default='mse');p.add_argument('--margin',type=float,default=1.0);p.add_argument('--fixed-m',type=int,choices=(5,),default=5);p.add_argument('--layer',type=int,choices=(15,),default=15);p.add_argument('--prefix',type=int,choices=(80,),default=80);p.add_argument('--lr',type=float,default=5e-6);p.add_argument('--ema-decay',type=float,default=.9995);p.add_argument('--epochs',type=int,default=10);p.add_argument('--init-seed',type=int,default=9137);p.add_argument('--train-plan-seed',type=int,default=2026102227);p.add_argument('--start-mode',choices=('fresh','resume'),default='fresh');a=p.parse_args();assert a.lr>0 and 0<a.ema_decay<1 and a.epochs>0 and a.margin>0
 if a.objective=='topk_ramp'and a.fixed_m!=5:p.error('The declared fifth TopK-ramp arm uses fixed own-G5 (--fixed-m5)')
 if a.action=='train'and a.fixture_base is None:p.error('--fixture-base is required for train')
 assert 0<=a.init_seed<2**32 and 0<=a.train_plan_seed<2**32
 a.base.mkdir(exist_ok=True,parents=True)
 try:run(a)
 except BaseException:atomic(a.base/('failure-rank-%s.json'%os.environ.get('RANK','cpu')),dict(time=time.time(),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
