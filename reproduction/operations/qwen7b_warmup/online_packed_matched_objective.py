"""Matched packed-input pilot: direct Mixed BCE versus population-variance MSE.

Same authentic G10 bank, copied official block, packed head, question budget,
EMA and optimizer. Objective and fixed/random M are the only factorial switches.
No class filtering/replay, per-trajectory target, final holdout, or cached hidden.
"""
import argparse,copy,fcntl,gzip,hashlib,inspect,json,math,os,random,time,traceback
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021'
TOKEN_FIXTURE=A/'online-mixed-layer-prefix-20261022-v1'
SEED=2026102227;ARCHITECTURE='one_question_packed_matched_objective_v1';FIXTURE_STATUS='MATCHED_PACKED_REAL_POLICY80_G10_FIXED2048_READY'

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
 base=args.base;base.mkdir(exist_ok=True,parents=True);guard=(base/'prepare.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if(base/'fixture-complete.json').exists():
  result=read(base/'fixture-complete.json');assert result['status']==FIXTURE_STATUS;return result
 source=args.token_fixture_base;fixture=read(source/'fixture-complete.json');assert fixture['status']=='ONLINE_MIXED_MATCHED8192_G10_FIXED2048_TOKENS_READY'and fixture['training_questions']==8192
 train=read(source/'inputs/train-groups.json');validation=read(source/'inputs/validation-groups.json');fixed_train=read(source/'inputs/fixed-train2048.json');assert len(train)==8192 and len(validation)==2048 and len(fixed_train)==2048
 assert all(len(g['trajectory_ids'])==len(g['labels'])==10 for g in train.values())and all(len(g['trajectory_ids'])==len(g['labels'])==5 for g in validation.values())
 assert not(set(train)&set(validation))and not({g['question_hash']for g in train.values()}&{g['question_hash']for g in validation.values()})
 wanted={t:(role,q,y)for role,gs in(('train',train),('validation',validation))for q,g in gs.items()for t,y in zip(g['trajectory_ids'],g['labels'])};assert len(wanted)==92160
 inputs=base/'inputs';inputs.mkdir(exist_ok=True);seen=set();refs=[];start=time.time();outputs=[]
 for i,ref in enumerate(fixture['token_shards']):
  path=inputs/('tokens-rank-%d.jsonl.gz'%i)
  with gzip.open(checked(ref),'rt')as src,gzip.open(path,'wt',compresslevel=1)as dst:
   for line in src:
    r=json.loads(line);tid=r['trajectory_id'];role,q,y=wanted[tid];assert tid not in seen and r['qid']==q and r['label']==y and r['role']==role
    out=dict(qid=q,trajectory_id=tid,role=role,prompt=r['prompt'],policy_token_ids=retained_policy(r));dst.write(json.dumps(out,separators=(',',':'))+'\n');seen.add(tid)
  refs.append(ref);outputs.append(proof(path))
 assert seen==set(wanted)
 for name,value in(('train-groups.json',train),('validation-groups.json',validation),('fixed-train-G5.json',fixed_train)):atomic(inputs/name,value)
 result=dict(status=FIXTURE_STATUS,training_questions=8192,training_trajectories=81920,validation_questions=2048,validation_trajectories=10240,max_real_policy_prefix=80,real_policy_only_no_20_to_80_fabrication=True,token_shards=outputs,source_token_shards=refs,source_fixture=proof(source/'fixture-complete.json'),train_groups=proof(inputs/'train-groups.json'),validation_groups=proof(inputs/'validation-groups.json'),fixed_train_G5=proof(inputs/'fixed-train-G5.json'),fixed_G5_sha256=fixture['fixed_G5_sha256'],hidden_features_saved=False,final4096_read=False,seconds=time.time()-start)
 atomic(base/'fixture-complete.json',result);return result

def epoch_plan(groups,epoch,fixed_m):
 from .packed_variance_probe import view
 rows=[view(groups[q],seed=SEED,epoch=epoch,fixed_m=fixed_m or None)for q in sorted(groups)]
 random.Random(int(digest(['packed-question-order',SEED,epoch]),16)).shuffle(rows)
 assert len(rows)==len({r['qid']for r in rows})==8192
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

def run(args):
 import numpy as np
 import torch
 import torch.distributed as dist
 from torch import nn
 from torch.nn.parallel import DistributedDataParallel as DDP
 from transformers import AutoModel,AutoTokenizer
 from .packed_variance_probe import PackedVarianceProbe,PackedVarianceHead
 base=args.base;base.mkdir(exist_ok=True,parents=True);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2;device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);dist.init_process_group('nccl');guard=(base/('writer-rank-%d.lock'%rank)).open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 fixture_base=args.fixture_base;fixture=read(fixture_base/'fixture-complete.json');assert fixture['status']==FIXTURE_STATUS and fixture['training_questions']==8192 and fixture['max_real_policy_prefix']==80;groups=read(checked(fixture['train_groups']));validation=read(checked(fixture['validation_groups']));records={}
 for i,ref in enumerate(fixture['token_shards']):
  if i%world!=rank:continue
  with gzip.open(checked(ref),'rt')as f:
   for line in f:r=json.loads(line);records[r['trajectory_id']]=r
 torch.manual_seed(9137);np.random.seed(9137);random.seed(9137);model_dir=B/'assets/qwen25-7b-instruct-a09a35458c70';tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True);separator=tokenizer.encode('\n### trajectory\n',add_special_tokens=False)
 actor=AutoModel.from_pretrained(str(model_dir),local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to(device).eval().requires_grad_(False);block=copy.deepcopy(actor.layers[args.layer]).float().requires_grad_(True);rotary=copy.deepcopy(actor.rotary_emb).float();actor.layers=nn.ModuleList(list(actor.layers[:args.layer]));actor.norm=nn.Identity();torch.cuda.empty_cache()
 head=PackedVarianceHead(max_positions=2048);head_hash=hashlib.sha256()
 for name,value in head.state_dict().items():head_hash.update(name.encode());head_hash.update(value.detach().contiguous().numpy().tobytes())
 initial_head_sha256=head_hash.hexdigest();probe=PackedVarianceProbe(block,rotary,'cache_position'in inspect.signature(block.forward).parameters,head).to(device);ema=copy.deepcopy(probe).eval().requires_grad_(False);optim=torch.optim.Adam(probe.parameters(),lr=args.lr);ddp=DDP(probe,device_ids=[device.index],broadcast_buffers=False);cursor=0;actual_Adam_updates=0;EMA_updates=0;margin_satisfied_batches=0;start_epoch=1
 binding=dict(architecture=ARCHITECTURE,objective=args.objective,fixed_m=args.fixed_m,layer=args.layer,capture_layer=args.layer-1,prefix=args.prefix,EMA_decay=args.ema_decay,lr=args.lr,global_batch=128,head_max_positions=2048,initial_head_sha256=initial_head_sha256,official_actor_revision='a09a35458c702b33eeacc393d103063234e8bc28',training_questions=8192,seed=SEED,fixture=proof(fixture_base/'fixture-complete.json'),sampling_independent_of_EM=True,class_weighting=False,question_prompt_occurrences_per_group=1,topk_margin=args.margin if args.objective=='topk_ramp'else None,topk_scope='actual_global_optimizer_batch_autograd_all_gather'if args.objective=='topk_ramp'else None,zero_ramp_loss_skips_Adam_and_EMA=True)
 if(base/'latest.json').exists():
  old=torch.load(checked(read(base/'latest.json')),map_location='cpu',weights_only=False);assert old['binding']==binding;probe.load_state_dict(old['raw']);ema.load_state_dict(old['EMA']);optim.load_state_dict(old['Adam']);cursor=old['cursor'];actual_Adam_updates=old['actual_Adam_updates'];EMA_updates=old['EMA_updates'];margin_satisfied_batches=old['margin_satisfied_batches'];assert old['batch_attempts']==cursor and EMA_updates==actual_Adam_updates and actual_Adam_updates+margin_satisfied_batches==cursor;start_epoch=old['epoch']+1
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
  fullplan=epoch_plan(groups,epoch,args.fixed_m);mine=[v for v in fullplan if qrank[v['qid']]==rank];assert len(mine)==4096;start=time.time()
  if rank==0:atomic(base/('sampling-plan-epoch-%02d.json'%epoch),dict(epoch=epoch,plan_sha256=digest(fullplan),rows=fullplan))
  for offset in range(0,len(mine),64):
   batch=mine[offset:offset+64];optim.zero_grad(set_to_none=True);total=0.;prefill=0.;actor_rows=0;local_logits=[];local_labels=[];loss_details={}
   for lo in range(0,len(batch),8):
    vs=batch[lo:lo+8];data,cost=capture(vs);prefill+=cost;actor_rows+=len(vs);target=torch.tensor([v['target4variance']if args.objective=='mse'else float(v['true_mixed'])for v in vs],device=device,dtype=torch.float32)
    with torch.autocast('cuda',dtype=torch.bfloat16):logit=ddp(data)
    if args.objective=='topk_ramp':local_logits.append(logit.float());local_labels.append(target)
    else:
     loss=(torch.sigmoid(logit).float()-target).square().sum()/len(batch)if args.objective=='mse'else torch.nn.functional.binary_cross_entropy_with_logits(logit.float(),target,reduction='sum')/len(batch)
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
   cursor+=1;assert actual_Adam_updates+margin_satisfied_batches==cursor and EMA_updates==actual_Adam_updates;row=dict(stage='MATCHED_PACKED_OBJECTIVE_UPDATES',binding=binding,epoch=epoch,cursor=cursor,batch_attempts=cursor,actual_Adam_updates=actual_Adam_updates,EMA_updates=EMA_updates,margin_satisfied_batches=margin_satisfied_batches,Adam_and_EMA_step_performed=update,rank=rank,epoch_questions_seen=offset+len(batch),training_loss=total,native_gradient_L2=ng,head_gradient_L2=hg,actual_global_batch=128,local_batch=64,micro=8,gradient_accumulation=8,loss_details=loss_details,actor_rows=actor_rows,prefill_rows_equal_groups=actor_rows==len(batch),prefill_seconds=prefill,time=time.time());atomic(base/('progress-rank-%d.json'%rank),row)
   if rank==0:
    print(json.dumps(row),flush=True)
    with(base/'curve.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
  rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device));rngs=[None]*world;dist.all_gather_object(rngs,rng)
  if rank==0:
   path=base/('checkpoint-epoch-%02d.pt'%epoch);tmp=path.with_suffix('.tmp');torch.save(dict(raw=probe.state_dict(),EMA=ema.state_dict(),Adam=optim.state_dict(),RNG_by_rank=rngs,cursor=cursor,batch_attempts=cursor,actual_Adam_updates=actual_Adam_updates,EMA_updates=EMA_updates,margin_satisfied_batches=margin_satisfied_batches,epoch=epoch,plan_sha256=digest(fullplan),binding=binding,actor_updates=0),tmp);tmp.replace(path);atomic(base/'latest.json',proof(path));atomic(base/('coverage-epoch-%02d.json'%epoch),dict(epoch=epoch,questions=8192,unique_questions=8192,trajectory_visits=sum(v['M']for v in fullplan),M_counts={m:sum(v['M']==m for v in fullplan)for m in range(3,11)},Mixed=sum(v['true_mixed']for v in fullplan),plan_sha256=digest(fullplan),sampling_independent_of_EM=True,seconds=time.time()-start))
  dist.barrier();evaluate(epoch)
 if rank==0:atomic(base/'complete.json',dict(status='MATCHED_PACKED_OBJECTIVE_PILOT_COMPLETE',epochs=args.epochs,cursor=cursor,batch_attempts=cursor,actual_Adam_updates=actual_Adam_updates,EMA_updates=EMA_updates,margin_satisfied_batches=margin_satisfied_batches,training_questions=8192,binding=binding,final4096_scored=False))
 dist.destroy_process_group()

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','train'),required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--fixture-base',type=Path);p.add_argument('--token-fixture-base',type=Path,default=TOKEN_FIXTURE);p.add_argument('--objective',choices=('mse','mixed_bce','topk_ramp'),default='mse');p.add_argument('--margin',type=float,default=1.0);p.add_argument('--fixed-m',type=int,choices=(0,5),default=0);p.add_argument('--layer',type=int,choices=(15,21),default=21);p.add_argument('--prefix',type=int,choices=(20,80),default=20);p.add_argument('--lr',type=float,default=5e-6);p.add_argument('--ema-decay',type=float,default=.9995);p.add_argument('--epochs',type=int,default=10);a=p.parse_args();assert a.lr>0 and 0<a.ema_decay<1 and a.epochs>0 and a.margin>0
 if a.objective=='topk_ramp'and a.fixed_m!=5:p.error('The declared fifth TopK-ramp arm uses fixed own-G5 (--fixed-m5)')
 if a.action=='train'and a.fixture_base is None:p.error('--fixture-base is required for train')
 a.base.mkdir(exist_ok=True,parents=True)
 try:print(json.dumps(prepare(a)))if a.action=='prepare'else run(a)
 except BaseException:atomic(a.base/('failure-rank-%s.json'%os.environ.get('RANK','cpu')),dict(time=time.time(),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
