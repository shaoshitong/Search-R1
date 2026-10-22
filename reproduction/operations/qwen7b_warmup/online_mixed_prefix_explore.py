"""Four-GPU matched direct-Mixed layer/prefix pilot, authentic online causal prefill.

Independent from formal full169028 E1 continuation. Every pilot epoch visits8192
questions once, one class-feasible authentic G5 from own G10; no hard replay.
Frozen official actor supplies post(block_layer-1) online; gradients only train
one copied official block + direct Mixed head. Hidden features never persist.
All variants use the identical group seed, fixed training2048 and validation2048.
"""
import argparse,copy,fcntl,gzip,hashlib,inspect,itertools,json,math,os,random,time,traceback
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021';SUITE=A/'online-mixed-layer-prefix-20261022-v1'
REVISION='a09a35458c702b33eeacc393d103063234e8bc28';QMAX=273

def read(p):return json.loads(Path(p).read_text())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_name(p.name+'.%d.tmp'%os.getpid());tmp.write_bytes(canonical(v)+b'\n');tmp.replace(p)
def fp(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for z in iter(lambda:f.read(8388608),b''):h.update(z)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(r):
 p=Path(r['path']);a=fp(p)
 if any(a[k]!=r[k]for k in ('path','bytes','sha256')):raise ValueError('Source SHA differs '+str(p))
 return p

def cut(row,prefix):
 """Only actual retained policy, preserving each turn's real causal tool context."""
 left=prefix;parts=[];offset=0
 for s in row['segments']:
  n=min(left,s['policy_length'])
  if n:
   parts.append(dict(s,policy_start=offset,policy_length=n,context_tokens=s['context_tokens'][:s['request_length']+n]));offset+=n;left-=n
  if not left:break
 return dict(row,segments=parts,policy_tokens=offset,positions=row['positions'][:offset])

def combinations(labels,target):
 return [v for v in itertools.combinations(range(10),5)if (0<sum(labels[i]for i in v)<5)==target]

def epoch_plan(groups,epoch,seed=20261022):
 """One q occurrence; choose feasible globally balanced labels without duplicates."""
 rng=random.Random(int(digest(['online-mixed-pilot',seed,epoch]),16));eligible=[q for q,g in groups.items()if 0<sum(g['labels'])<10];rng.shuffle(eligible);positives=set(eligible[:min(len(groups)//2,len(eligible))]);out=[]
 for q in sorted(groups):
  g=groups[q];idx=rng.choice(combinations(g['labels'],q in positives));out.append(dict(qid=q,indices=list(idx),trajectory_ids=[g['trajectory_ids'][j]for j in idx],labels=[g['labels'][j]for j in idx],true_mixed=q in positives))
 rng.shuffle(out)
 return out,dict(questions=len(out),unique_questions=len({r['qid']for r in out}),Mixed=len(positives),nonMixed=len(out)-len(positives),seed=seed,epoch=epoch,plan_sha256=digest(out),hard_replay=False)

def prepare():
 """CPU token-only fixture. Reuse8192 existing real G10 and exact fixed2048 G5."""
 from .peer_long_prefix_diagnostic import reconstruct
 SUITE.mkdir(parents=True,exist_ok=True)
 if(SUITE/'fixture-complete.json').exists():return read(SUITE/'fixture-complete.json')
 t=read(A/'peer-newseed8192-g10-train-v1/training-input.json');ident=read(checked(t['identity_plan']));raw=read(checked(t['raw_complete']));all_train=read(checked(raw['outputs']['groups']))['train'];wantq={g['qid']for g in ident['groups']};train={g['qid']:g for g in all_train if g['qid']in wantq};assert len(train)==8192
 ev=read(checked(raw['outputs']['source-events']));sources=[b['full']for e in ev['events']for b in e['batches']]
 vp=A/'validation-g10-20261021/materialized/bank';vg=read(vp/'groups.json')['validation'];vlookup={g['qid']:g for g in vg};fixed=read(vp/'fixed-G5-selection.json');validation={}
 for g in fixed['groups']:
  original=vlookup[g['qid']];by=dict(zip(original['trajectory_ids'],original['labels']));validation[g['qid']]=dict(qid=g['qid'],question_hash=original['question_hash'],trajectory_ids=g['trajectory_ids'],labels=[by[t]for t in g['trajectory_ids']])
 assert len(validation)==2048 and len({g['qid']for g in train.values()}&set(validation))==0
 va=read(vp/'source-audit.json')
 for batches in va['events'].values():
  for batch in batches:sources.append(batch['phase_payloads']['full'])
 wanted={tid:(role,q,y)for role,groups in(('train',train),('validation',validation))for q,g in groups.items()for tid,y in zip(g['trajectory_ids'],g['labels'])};assert len(wanted)==92160
 (SUITE/'inputs').mkdir(exist_ok=True);streams=[gzip.open(SUITE/'inputs'/('tokens-rank-%d.jsonl.gz'%r),'wt',compresslevel=1)for r in range(4)];qidrank={q:i%4 for role,groups in(('train',train),('validation',validation))for i,q in enumerate(sorted(groups))};seen=set();proofs=[];start=time.time()
 try:
  for ref in sources:
   p=checked(ref)
   with gzip.open(p,'rt')as f:
    for line in f:
     v=json.loads(line);tid=v['trajectory_id']
     if tid not in wanted:continue
     role,q,y=wanted[tid]
     if tid in seen or v['qid']!=q or v['strict_em']!=y or v['actor_version']!='official7b-'+REVISION:raise ValueError('Wrong real source trajectory/EM')
     r=reconstruct(v,80);r={k:r[k]for k in('segments','positions','policy_tokens')};r.update(trajectory_id=tid,qid=q,label=y,role=role,prompt=v['initial_prompt_token_ids']);assert len(r['prompt'])<=QMAX
     streams[qidrank[q]].write(json.dumps(r,separators=(',',':'))+'\n');seen.add(tid)
   proofs.append(ref);atomic(SUITE/'prepare-progress.json',dict(stage='EXTRACTING_REAL_CAUSAL_PREFIX80_TOKEN_FIXTURE',trajectories=len(seen),expected=92160,source_files=len(proofs),seconds=time.time()-start))
 finally:
  for f in streams:f.close()
 if seen!=set(wanted):raise ValueError('Missing real own trajectories')
 atomic(SUITE/'inputs/train-groups.json',train);atomic(SUITE/'inputs/validation-groups.json',validation);plans=[]
 for epoch in range(1,4):
  p,proof=epoch_plan(train,epoch);atomic(SUITE/'inputs'/('epoch-%02d-plan.json'%epoch),p);plans.append(proof)
 fixed_train,ftproof=epoch_plan(train,0);fixed_train=sorted(fixed_train,key=lambda x:x['qid'])[:2048];atomic(SUITE/'inputs/fixed-train2048.json',fixed_train)
 result=dict(status='ONLINE_MIXED_MATCHED8192_G10_FIXED2048_TOKENS_READY',training_questions=8192,training_trajectories=81920,validation_questions=2048,validation_trajectories=10240,fixed_G5_sha256=fixed['summary']['selected_g5_sha256'],source_raw=t['raw_complete'],source_identity=t['identity_plan'],source_full=proofs,plans=plans,fixed_train_plan_sha256=digest(fixed_train),token_shards=[fp(SUITE/'inputs'/('tokens-rank-%d.jsonl.gz'%r))for r in range(4)],question_hash_train_validation_overlap=bool({g['question_hash']for g in train.values()}&{g['question_hash']for g in validation.values()}),actor_updates=0,hidden_features_saved=False,not_full169028_replacement=True,final4096_read=False,seconds=time.time()-start)
 assert not result['question_hash_train_validation_overlap'];atomic(SUITE/'fixture-complete.json',result);return result

def metrics(rows):
 import torch
 z=torch.tensor([r['logit']for r in rows],dtype=torch.float64);y=torch.tensor([r['true_mixed']for r in rows]);pred=z>=0;tp=int((pred&y).sum());tn=int((~pred&~y).sum());np=int(y.sum());nn=len(y)-np
 def error(k):
  own=[r for r in rows if sum(r['labels'])==k];return dict(questions=len(own),predicted_Mixed=sum(r['logit']>=0 for r in own),Mixed_false_positive_rate=sum(r['logit']>=0 for r in own)/len(own)if own else None)
 top=sorted(rows,key=lambda r:(-r['logit'],r['qid']))[:512]
 return dict(questions=len(rows),binary_Mixed_BCE=float(torch.nn.functional.binary_cross_entropy_with_logits(z,y.double())),balanced_accuracy=.5*(tp/np+tn/nn),Mixed_recall=tp/np,nonMixed_false_positive_rate=1-tn/nn,confusion=dict(tp=tp,tn=tn,fp=nn-tn,fn=np-tp),all_wrong=error(0),all_correct=error(5),selected512=dict(questions=len(top),Mixed=sum(r['true_mixed']for r in top),Mixed_fraction=sum(r['true_mixed']for r in top)/len(top)))

def run(args):
 import numpy as np
 import torch
 import torch.distributed as dist
 from torch import nn
 from torch.nn.parallel import DistributedDataParallel as DDP
 from transformers import AutoModel
 from .native_mixed_group_classifier import MixedTokenHead,NativeMixedGroupProbe
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);dist.init_process_group('nccl');assert world==4
 base=SUITE/('layer%d-prefix%d'%(args.layer,args.prefix));base.mkdir(parents=True,exist_ok=True);lock=(base/('writer-rank-%d.lock'%rank)).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 fixture=read(SUITE/'fixture-complete.json');assert fixture['fixed_G5_sha256']=='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7';checked(fixture['token_shards'][rank]);records={}
 with gzip.open(fixture['token_shards'][rank]['path'],'rt')as f:
  for line in f:r=json.loads(line);records[r['trajectory_id']]=cut(r,args.prefix)
 groups=read(SUITE/'inputs/train-groups.json');validation=read(SUITE/'inputs/validation-groups.json');fixed_train=read(SUITE/'inputs/fixed-train2048.json');torch.manual_seed(9137);np.random.seed(9137);random.seed(9137)
 if rank==0:atomic(base/'controller.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],stage='LOADING_FROZEN_OFFICIAL_ACTOR_FOR_ONLINE_PREFILL',layer=args.layer,capture_layer=args.layer-1,prefix=args.prefix,world_size=4,training_questions=8192,training_G5_per_epoch=8192,validation_questions=2048,raw_gradient_updates_EMA=True,EMA_decay=.99,fresh_official_block_and_head=True,not_E1_continuation=True,hidden_feature_cache=False,final4096_read=False,time=time.time()))
 model_dir=B/'assets/qwen25-7b-instruct-a09a35458c70';actor=AutoModel.from_pretrained(str(model_dir),local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to(device).eval().requires_grad_(False)
 block=copy.deepcopy(actor.layers[args.layer]).float().requires_grad_(True);rotary=copy.deepcopy(actor.rotary_emb).float();actor.layers=nn.ModuleList(list(actor.layers[:args.layer]));actor.norm=nn.Identity();torch.cuda.empty_cache()
 head=MixedTokenHead();
 if args.prefix!=20:
  slots=nn.Embedding(args.prefix,256);slots.weight.data[:20].copy_(head.policy_slots.weight.data);head.policy_slots=slots;head.prefix=args.prefix
 probe=NativeMixedGroupProbe(block,rotary,'cache_position'in inspect.signature(block.forward).parameters,head).to(device);ema=copy.deepcopy(probe).eval().requires_grad_(False);optim=torch.optim.Adam(probe.parameters(),lr=args.lr);ddp=DDP(probe,device_ids=[device.index],broadcast_buffers=False)
 cursor=0;start_epoch=1;resume=base/'latest.json'
 if resume.exists():
  cp=read(resume);blob=torch.load(checked(cp),map_location='cpu',weights_only=False);probe.load_state_dict(blob['raw']);ema.load_state_dict(blob['EMA']);optim.load_state_dict(blob['Adam']);cursor=blob['cursor'];start_epoch=blob['epoch']+1
  for state in optim.state.values():
   for k,v in state.items():
    if torch.is_tensor(v)and k!='step':state[k]=v.to(device)
  rng=blob['RNG_by_rank'][rank];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'],device)
 qrank={q:i%4 for role,gs in(('train',groups),('validation',validation))for i,q in enumerate(sorted(gs))}
 # Every rank retains only its own questions; all ranks see equal train/eval counts.
 def capture(views):
  started=time.monotonic();own=[records[t]for v in views for t in v['trajectory_ids']];width=QMAX+args.prefix;features=torch.zeros((len(own),width,3584),dtype=torch.bfloat16,device=device);mask=torch.zeros((len(own),width),dtype=torch.bool,device=device);positions=torch.zeros((len(own),width),dtype=torch.long,device=device);segments=[]
  for index,r in enumerate(own):
   qlen=len(r['prompt']);plen=r['policy_tokens'];mask[index,:qlen]=True;mask[index,QMAX:QMAX+plen]=True;positions[index,:qlen]=torch.arange(qlen,device=device);positions[index,QMAX:QMAX+plen]=torch.tensor(r['positions'],device=device)
   for j,s in enumerate(r['segments']):segments.append((index,j==0,qlen,s))
  for offset in range(0,len(segments),32):
   part=segments[offset:offset+32];length=max(len(s['context_tokens'])for _,_,_,s in part);ids=torch.full((len(part),length),151643,dtype=torch.long,device=device);am=torch.zeros_like(ids)
   for i,(_,_,_,s)in enumerate(part):t=s['context_tokens'];ids[i,:len(t)]=torch.tensor(t,device=device);am[i,:len(t)]=1
   with torch.no_grad():hidden=actor(input_ids=ids,attention_mask=am,use_cache=False,return_dict=True).last_hidden_state
   for i,(index,first,qlen,s)in enumerate(part):
    if first:features[index,:qlen].copy_(hidden[i,:qlen])
    n=s['policy_length'];begin=s['policy_start'];request=s['request_length'];features[index,QMAX+begin:QMAX+begin+n].copy_(hidden[i,request:request+n])
   del hidden
  return dict(features=features.detach(),attention_mask=mask,position_ids=positions),time.monotonic()-started
 def evaluate(epoch):
  probe.eval();before=time.time();out={}
  for role,allviews in [('fixed_train2048',fixed_train),('fixed_validation2048',[dict(g,true_mixed=0<sum(g['labels'])<5)for g in validation.values()])]:
   own=[v for v in allviews if qrank[v['qid']]==rank];rows=[]
   for begin in range(0,len(own),8):
    vs=own[begin:begin+8];data,cost=capture(vs)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):rawlog=probe(data).cpu().tolist();emalog=ema(data).cpu().tolist()
    rows.extend([dict(qid=v['qid'],trajectory_ids=v['trajectory_ids'],labels=v['labels'],true_mixed=v['true_mixed'],raw_logit=x,EMA_logit=y)for v,x,y in zip(vs,rawlog,emalog)])
   received=[None]*world;dist.all_gather_object(received,rows)
   if rank==0:
    allrows=[r for rr in received for r in rr];assert len(allrows)==len(allviews)and len({r['qid']for r in allrows})==len(allviews)
    path=base/('epoch-%02d-%s-predictions.jsonl.gz'%(epoch,role))
    with gzip.open(path,'wt')as f:
     for row in allrows:f.write(json.dumps(row)+'\n')
    out[role]={label:metrics([dict(r,logit=r[label+'_logit'])for r in allrows])for label in('raw','EMA')}
   del rows
  if rank==0:atomic(base/('evaluation-epoch-%02d.json'%epoch),dict(epoch=epoch,cursor=cursor,metrics=out,seconds=time.time()-before,layer=args.layer,capture_layer=args.layer-1,prefix=args.prefix,hidden_features_persisted=False,final4096_scored=False,not_full169028_replacement=True))
  dist.barrier();probe.train()
 if start_epoch==1:evaluate(0)
 for epoch in range(start_epoch,args.epochs+1):
  plan=read(SUITE/'inputs'/('epoch-%02d-plan.json'%epoch));mine=[v for v in plan if qrank[v['qid']]==rank];assert len(mine)==2048 and len({v['qid']for v in mine})==2048;counts={True:sum(v['true_mixed']for v in plan),False:sum(not v['true_mixed']for v in plan)};begin=time.time();cumulative=0.
  for offset in range(0,len(mine),32):
   batch=mine[offset:offset+32];optim.zero_grad(set_to_none=True);data,prefill=capture(batch);train_start=time.monotonic();lossnum=0.
   for lo in range(0,len(batch),8):
    sub=batch[lo:lo+8];sl={k:v[lo*5:(lo+len(sub))*5]for k,v in data.items()};target=torch.tensor([v['true_mixed']for v in sub],device=device,dtype=torch.float32);weights=torch.tensor([len(plan)/(2*counts[v['true_mixed']])for v in sub],device=device)
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=ddp(sl);loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*weights).sum()/len(batch)
    loss.backward();lossnum+=float(loss.detach())
   native_grad_norm=math.sqrt(sum(float(p.grad.detach().float().square().sum())for p in probe.block.parameters()if p.grad is not None));head_grad_norm=math.sqrt(sum(float(p.grad.detach().float().square().sum())for p in probe.group_head.parameters()if p.grad is not None));assert native_grad_norm>0 and head_grad_norm>0
   torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);optim.step()
   with torch.no_grad():
    for p,e in zip(probe.parameters(),ema.parameters()):e.mul_(.99).add_(p,alpha=.01)
   cursor+=1;cumulative+=lossnum;progress=dict(stage='ONLINE_PREFILL_REAL_MIXED_UPDATES',epoch=epoch,cursor=cursor,rank=rank,epoch_questions_seen=offset+len(batch),epoch_unique_questions=8192,prefill_seconds=prefill,train_seconds=time.monotonic()-train_start,loss=lossnum,native_gradient_L2=native_grad_norm,head_gradient_L2=head_grad_norm,global_batch_questions=128,trajectory_rows_per_rank=len(batch)*5,layer=args.layer,prefix=args.prefix,time=time.time());atomic(base/('progress-rank-%d.json'%rank),progress)
   if rank==0:
    print(json.dumps(progress),flush=True)
    with(base/'curve.jsonl').open('a')as f:f.write(json.dumps(progress)+'\n')
  rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device));rngs=[None]*world;dist.all_gather_object(rngs,rng)
  if rank==0:
   path=base/('checkpoint-epoch-%02d.pt'%epoch);tmp=path.with_suffix('.tmp');torch.save(dict(raw=probe.state_dict(),EMA=ema.state_dict(),Adam=optim.state_dict(),RNG_by_rank=rngs,cursor=cursor,epoch=epoch,plan_sha256=digest(plan),layer=args.layer,prefix=args.prefix,learning_rate=args.lr,raw_updates_EMA=True,actor_updates=0),tmp);tmp.replace(path);atomic(base/'latest.json',fp(path));atomic(base/('coverage-epoch-%02d.json'%epoch),dict(epoch=epoch,questions=8192,unique_questions=8192,G5_trajectory_visits=40960,Mixed=counts[True],nonMixed=counts[False],plan_sha256=digest(plan),seconds=time.time()-begin))
  dist.barrier();evaluate(epoch)
 if rank==0:atomic(base/'complete.json',dict(status='MATCHED_ONLINE_MIXED_LAYER_PREFIX_PILOT_COMPLETE',epochs=args.epochs,cursor=cursor,layer=args.layer,prefix=args.prefix,not_full169028_replacement=True,final4096_scored=False))
 dist.destroy_process_group()

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=['prepare','train'],required=True);p.add_argument('--layer',type=int,choices=[21,27],default=27);p.add_argument('--prefix',type=int,choices=[20,40,80],default=20);p.add_argument('--epochs',type=int,default=3);p.add_argument('--lr',type=float,default=5e-6);a=p.parse_args()
 if a.action=='prepare':print(json.dumps(prepare()));return
 base=SUITE/('layer%d-prefix%d'%(a.layer,a.prefix));base.mkdir(parents=True,exist_ok=True)
 try:run(a)
 except BaseException as e:atomic(base/('failure-rank-%s.json'%os.environ.get('RANK','unknown')),dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=time.time()));raise
if __name__=='__main__':main()
