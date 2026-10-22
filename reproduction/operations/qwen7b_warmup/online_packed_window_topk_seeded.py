"""Explicit second-seed current-global64 TopK-ramp; existing runtime untouched.

Only current64 logits carry gradient. Cached labels/scores retain their actual
raw-probe update version; no stale-score unbiasedness claim is made. Existing
128-batch runtime and all active jobs remain untouched.
"""
import argparse,copy,fcntl,gzip,hashlib,inspect,json,math,os,random,time,traceback
from pathlib import Path
from .online_packed_matched_objective import B,A,TOKEN_FIXTURE,FIXTURE_STATUS,read,canonical,digest,atomic,proof,checked,prepare,pack_group,metrics
ARCHITECTURE='one_question_packed_global64_detached_window_topk_v1'
DIAGNOSTIC_SEED=2026102228

def seed_model_rng(init_seed):
 import numpy as np
 import torch
 if type(init_seed)is not int or not 0<=init_seed<2**32:raise ValueError('Explicit32-bit model initialization seed required')
 torch.manual_seed(init_seed);np.random.seed(init_seed);random.seed(init_seed)

def head_state_sha256(head):
 h=hashlib.sha256()
 for name,value in head.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def epoch_plan(groups,epoch,fixed_m,*,train_plan_seed):
 # Seed belongs to this function, never to an imported module global.
 from .packed_variance_probe import view
 rows=[view(groups[q],seed=train_plan_seed,epoch=epoch,fixed_m=fixed_m or None)for q in sorted(groups)]
 random.Random(int(digest(['packed-question-order',train_plan_seed,epoch]),16)).shuffle(rows)
 assert len(rows)==len({r['qid']for r in rows})==8192
 return rows

def enforce_start_mode(base,start_mode):
 latest=Path(base)/'latest.json'
 if start_mode=='fresh'and latest.exists():raise ValueError('Fresh CURRENT64 seed replica cannot resume an existing checkpoint')
 if start_mode=='resume'and not latest.exists():raise ValueError('Exact resume requires this replica checkpoint')

def run(args):
 import numpy as np
 import torch
 import torch.distributed as dist
 from torch import nn
 from torch.nn.parallel import DistributedDataParallel as DDP
 from transformers import AutoModel,AutoTokenizer
 from .packed_variance_probe import PackedVarianceProbe,PackedVarianceHead
 from .detached_score_window import DetachedScoreWindow
 from .group_topk_losses import precision_at_k_ramp
 from .score_window_audit import audit_score_window
 base=args.base;base.mkdir(exist_ok=True,parents=True);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2;device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);dist.init_process_group('nccl');guard=(base/('writer-rank-%d.lock'%rank)).open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 fixture_base=args.fixture_base;fixture=read(fixture_base/'fixture-complete.json');assert fixture['status']==FIXTURE_STATUS and fixture['training_questions']==8192 and fixture['max_real_policy_prefix']==80;groups=read(checked(fixture['train_groups']));validation=read(checked(fixture['validation_groups']));records={}
 for i,ref in enumerate(fixture['token_shards']):
  if i%world!=rank:continue
  with gzip.open(checked(ref),'rt')as f:
   for line in f:r=json.loads(line);records[r['trajectory_id']]=r
 seed_model_rng(args.init_seed);enforce_start_mode(base,args.start_mode);model_dir=B/'assets/qwen25-7b-instruct-a09a35458c70';tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True);separator=tokenizer.encode('\n### trajectory\n',add_special_tokens=False)
 actor=AutoModel.from_pretrained(str(model_dir),local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to(device).eval().requires_grad_(False);block=copy.deepcopy(actor.layers[args.layer]).float().requires_grad_(True);rotary=copy.deepcopy(actor.rotary_emb).float();actor.layers=nn.ModuleList(list(actor.layers[:args.layer]));actor.norm=nn.Identity();torch.cuda.empty_cache()
 head=PackedVarianceHead(max_positions=2048);initial_head_sha256=head_state_sha256(head);
 probe=PackedVarianceProbe(block,rotary,'cache_position'in inspect.signature(block.forward).parameters,head).to(device);ema=copy.deepcopy(probe).eval().requires_grad_(False);optim=torch.optim.Adam(probe.parameters(),lr=args.lr);ddp=DDP(probe,device_ids=[device.index],broadcast_buffers=False);cursor=0;actual_Adam_updates=0;EMA_updates=0;margin_satisfied_batches=0;cached_only_zero_current_gradient_batches=0;start_epoch=1;queue=DetachedScoreWindow(capacity=args.cache_window,target_k=500,reference_population=2018);last_window_entries=[]
 binding=dict(architecture=ARCHITECTURE,objective=args.objective,fixed_m=args.fixed_m,layer=args.layer,capture_layer=args.layer-1,prefix=args.prefix,EMA_decay=args.ema_decay,lr=args.lr,global_batch=64,local_batch=32,micro=8,gradient_accumulation=4,head_max_positions=2048,initial_head_sha256=initial_head_sha256,official_actor_revision='a09a35458c702b33eeacc393d103063234e8bc28',training_questions=8192,seed=args.train_plan_seed,init_seed=args.init_seed,train_plan_seed=args.train_plan_seed,diagnostic_seed=DIAGNOSTIC_SEED,seed_source='explicit_CLI',fixture=proof(fixture_base/'fixture-complete.json'),sampling_independent_of_EM=True,class_weighting=False,question_prompt_occurrences_per_group=1,topk_margin=args.margin if args.objective=='topk_ramp'else None,topk_scope='current_global64_autograd_plus_detached_historical_scores',zero_ramp_loss_skips_Adam_and_EMA=True,zero_current_gradient_skips_Adam_and_EMA=True,cache_window_capacity=args.cache_window,reference_population=2018,target_top_k=500,loss_gradient_scale='N_eff/current_global_batch',scaling_unbiased_claimed=False)
 if(base/'latest.json').exists():
  old=torch.load(checked(read(base/'latest.json')),map_location='cpu',weights_only=False);assert old['init_seed']==args.init_seed and old['train_plan_seed']==args.train_plan_seed and old['diagnostic_seed']==DIAGNOSTIC_SEED and old['initial_head_sha256']==initial_head_sha256;assert old['binding']==binding;probe.load_state_dict(old['raw']);ema.load_state_dict(old['EMA']);optim.load_state_dict(old['Adam']);cursor=old['cursor'];actual_Adam_updates=old['actual_Adam_updates'];EMA_updates=old['EMA_updates'];margin_satisfied_batches=old['margin_satisfied_batches'];cached_only_zero_current_gradient_batches=old['cached_only_zero_current_gradient_batches'];queue=DetachedScoreWindow.from_state_dict(old['score_window']);last_window_entries=old['last_window_entries'];assert digest(old['score_window'])==old['score_window_sha256']and old['score_model_version']==actual_Adam_updates and old['batch_attempts']==cursor and EMA_updates==actual_Adam_updates and actual_Adam_updates+margin_satisfied_batches+cached_only_zero_current_gradient_batches==cursor;start_epoch=old['epoch']+1
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
  if rank==0:atomic(base/('evaluation-epoch-%02d.json'%epoch),dict(epoch=epoch,cursor=cursor,batch_attempts=cursor,actual_Adam_updates=actual_Adam_updates,EMA_updates=EMA_updates,margin_satisfied_batches=margin_satisfied_batches,cached_only_zero_current_gradient_batches=cached_only_zero_current_gradient_batches,binding=binding,metrics=out,seconds=time.time()-start,final4096_scored=False))
  dist.barrier();probe.train()
 if start_epoch==1:evaluate(0)
 elif not(base/('evaluation-epoch-%02d.json'%(start_epoch-1))).exists():evaluate(start_epoch-1)
 def score_window(epoch):
  # Every rank enters the callback: it only receives view identities, never EM.
  if not last_window_entries:return
  probe.eval()
  def score_fn(identities):
   rows=[];own=[(i,v)for i,v in enumerate(identities)if qrank[v['qid']]==rank]
   for offset in range(0,len(own),8):
    part=own[offset:offset+8];data,_=capture([v for _,v in part])
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):values=probe(data).float().cpu().tolist()
    rows.extend((index,value)for(index,_),value in zip(part,values))
   gathered=[None]*world;dist.all_gather_object(gathered,rows);lookup={i:value for part in gathered for i,value in part};assert set(lookup)==set(range(len(identities)))
   return [lookup[i]for i in range(len(identities))]
  k=(len(last_window_entries)*500+2017)//2018;result=audit_score_window(last_window_entries,current_step=actual_Adam_updates,score_fn=score_fn,k=k,gamma=args.margin,loss_fn=precision_at_k_ramp)
  if rank==0:atomic(base/('score-window-audit-epoch-%02d.json'%epoch),dict(epoch=epoch,batch_attempts=cursor,actual_Adam_updates=actual_Adam_updates,cache_window_capacity=args.cache_window,audit=result,cache_and_parameters_modified=False,final4096_read=False))
  dist.barrier();probe.train()
 if start_epoch>1 and not(base/('score-window-audit-epoch-%02d.json'%(start_epoch-1))).exists():score_window(start_epoch-1)
 for epoch in range(start_epoch,args.epochs+1):
  fullplan=epoch_plan(groups,epoch,5,train_plan_seed=args.train_plan_seed);mine=[v for v in fullplan if qrank[v['qid']]==rank];assert len(mine)==4096;start=time.time()
  if rank==0:atomic(base/('sampling-plan-epoch-%02d.json'%epoch),dict(epoch=epoch,init_seed=args.init_seed,train_plan_seed=args.train_plan_seed,diagnostic_seed=DIAGNOSTIC_SEED,plan_sha256=digest(fullplan),rows=fullplan))
  for offset in range(0,len(mine),32):
   batch=mine[offset:offset+32];optim.zero_grad(set_to_none=True);prefill=0.;actor_rows=0;local_logits=[]
   for lo in range(0,len(batch),8):
    vs=batch[lo:lo+8];data,cost=capture(vs);prefill+=cost;actor_rows+=len(vs)
    with torch.autocast('cuda',dtype=torch.bfloat16):logit=ddp(data)
    local_logits.append(logit.float())
   from torch.distributed.nn.functional import all_gather
   current=torch.cat(all_gather(torch.cat(local_logits).contiguous())).float();current.retain_grad();actual_global=len(batch)*world;assert actual_global==64 and current.numel()==64
   local_entries=[dict(qid=v['qid'],trajectory_ids=v['trajectory_ids'],labels=v['labels'],true_mixed=bool(v['true_mixed']),M=5,scored_step=actual_Adam_updates,scored_batch_attempt=cursor+1,score_model='raw')for v in batch];entry_parts=[None]*world;dist.all_gather_object(entry_parts,local_entries);current_entries=[v for part in entry_parts for v in part];values=current.detach().cpu().tolist()
   for entry,value in zip(current_entries,values):entry['cached_score']=value
   window=queue.prepare(current_entries,model_version=actual_Adam_updates);history=torch.tensor(window['cached_scores'],device=device,dtype=torch.float32);scores=torch.cat((history,current));labels=torch.tensor([int(v['true_mixed'])for v in window['rows']],device=device,dtype=torch.float32);assert scores.numel()==labels.numel()==window['N_eff']and window['current_offset']==len(history)
   unscaled=precision_at_k_ramp(scores,labels,window['K'],margin=args.margin);scale=window['N_eff']/actual_global;scaled=unscaled*scale;scaled.backward();raw_loss=float(unscaled.detach());scaled_loss=float(scaled.detach());cg=current.grad;assert cg is not None and bool(torch.isfinite(cg).all());current_gradient_L2=float(cg.float().norm());ng=math.sqrt(sum(float(p.grad.detach().float().square().sum())for p in probe.block.parameters()if p.grad is not None));hg=math.sqrt(sum(float(p.grad.detach().float().square().sum())for p in probe.group_head.parameters()if p.grad is not None));assert math.isfinite(ng)and math.isfinite(hg)
   update=raw_loss!=0. and current_gradient_L2!=0.
   if update:
    torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);optim.step();actual_Adam_updates+=1
    with torch.no_grad():
     for p,e in zip(probe.parameters(),ema.parameters()):e.mul_(args.ema_decay).add_(p,alpha=1-args.ema_decay)
    EMA_updates+=1
   elif raw_loss==0.:margin_satisfied_batches+=1
   else:cached_only_zero_current_gradient_batches+=1
   queue.commit(post_update_step=actual_Adam_updates);last_window_entries=copy.deepcopy(window['rows']);cursor+=1;assert actual_Adam_updates+margin_satisfied_batches+cached_only_zero_current_gradient_batches==cursor and EMA_updates==actual_Adam_updates
   row=dict(stage='MATCHED_GLOBAL64_DETACHED_WINDOW_TOPK',binding=binding,epoch=epoch,cursor=cursor,batch_attempts=cursor,actual_Adam_updates=actual_Adam_updates,EMA_updates=EMA_updates,margin_satisfied_batches=margin_satisfied_batches,cached_only_zero_current_gradient_batches=cached_only_zero_current_gradient_batches,Adam_and_EMA_step_performed=update,rank=rank,epoch_questions_seen=offset+len(batch),unscaled_bound_loss=raw_loss,training_loss_scaled=scaled_loss,loss_gradient_scale=scale,scaling_unbiased_claimed=False,current_logit_gradient_L2=current_gradient_L2,native_gradient_L2=ng,head_gradient_L2=hg,N_eff=window['N_eff'],K=window['K'],current_global_questions=actual_global,detached_historical_questions=len(history),window_diagnostics=window['diagnostics'],local_batch=32,micro=8,gradient_accumulation=4,actor_rows=actor_rows,prefill_rows_equal_groups=actor_rows==len(batch),prefill_seconds=prefill,time=time.time());atomic(base/('progress-rank-%d.json'%rank),row)
   if rank==0:
    print(json.dumps(row),flush=True)
    with(base/'curve.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
  rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device));rngs=[None]*world;dist.all_gather_object(rngs,rng);state=queue.state_dict();assert state['pending']is None;state_hash=digest(state);state_hashes=[None]*world;dist.all_gather_object(state_hashes,state_hash);assert len(set(state_hashes))==1
  if rank==0:
   path=base/('checkpoint-epoch-%02d.pt'%epoch);tmp=path.with_suffix('.tmp');torch.save(dict(init_seed=args.init_seed,train_plan_seed=args.train_plan_seed,diagnostic_seed=DIAGNOSTIC_SEED,initial_head_sha256=initial_head_sha256,raw=probe.state_dict(),EMA=ema.state_dict(),Adam=optim.state_dict(),RNG_by_rank=rngs,cursor=cursor,batch_attempts=cursor,actual_Adam_updates=actual_Adam_updates,EMA_updates=EMA_updates,margin_satisfied_batches=margin_satisfied_batches,cached_only_zero_current_gradient_batches=cached_only_zero_current_gradient_batches,score_window=state,score_window_sha256=state_hash,score_model_version=actual_Adam_updates,last_window_entries=last_window_entries,epoch=epoch,plan_sha256=digest(fullplan),binding=binding,actor_updates=0),tmp);tmp.replace(path);atomic(base/'latest.json',proof(path));atomic(base/('coverage-epoch-%02d.json'%epoch),dict(epoch=epoch,questions=8192,unique_questions=8192,trajectory_visits=40960,init_seed=args.init_seed,train_plan_seed=args.train_plan_seed,diagnostic_seed=DIAGNOSTIC_SEED,plan_sha256=digest(fullplan),batch_attempts=cursor,actual_Adam_updates=actual_Adam_updates,EMA_updates=EMA_updates,margin_satisfied_batches=margin_satisfied_batches,cached_only_zero_current_gradient_batches=cached_only_zero_current_gradient_batches,score_window_sha256=state_hash,sampling_independent_of_EM=True,seconds=time.time()-start))
  dist.barrier();score_window(epoch);evaluate(epoch)
 if rank==0:atomic(base/'complete.json',dict(status='MATCHED_GLOBAL64_DETACHED_WINDOW_TOPK_COMPLETE',epochs=args.epochs,cursor=cursor,batch_attempts=cursor,actual_Adam_updates=actual_Adam_updates,EMA_updates=EMA_updates,margin_satisfied_batches=margin_satisfied_batches,cached_only_zero_current_gradient_batches=cached_only_zero_current_gradient_batches,training_questions=8192,binding=binding,final4096_scored=False))
 dist.destroy_process_group()

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','train'),required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--fixture-base',type=Path);p.add_argument('--token-fixture-base',type=Path,default=TOKEN_FIXTURE);p.add_argument('--cache-window',type=int,choices=(0,),required=True);p.add_argument('--layer',type=int,choices=(15,),default=15);p.add_argument('--prefix',type=int,choices=(80,),default=80);p.add_argument('--margin',type=float,default=1.0);p.add_argument('--lr',type=float,default=5e-6);p.add_argument('--ema-decay',type=float,default=.9995);p.add_argument('--epochs',type=int,default=10);p.add_argument('--init-seed',type=int,required=True);p.add_argument('--train-plan-seed',type=int,required=True);p.add_argument('--start-mode',choices=('fresh','resume'),default='fresh');a=p.parse_args();a.objective='topk_ramp';a.fixed_m=5;assert a.lr>0 and 0<a.ema_decay<1 and a.epochs>0 and a.margin>0
 if a.action=='train'and a.fixture_base is None:p.error('--fixture-base is required for train')
 assert 0<=a.init_seed<2**32 and 0<=a.train_plan_seed<2**32
 a.base.mkdir(exist_ok=True,parents=True)
 try:print(json.dumps(prepare(a)))if a.action=='prepare'else run(a)
 except BaseException:atomic(a.base/('failure-rank-%s.json'%os.environ.get('RANK','cpu')),dict(time=time.time(),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
