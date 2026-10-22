"""Random M3..10, one question + M Prefix20, one packed actor row per group.

Official block21 and group head regress four times the selected own-M EM
population variance. No BCE, trajectory prediction, ranking loss or EM-based
selection. The 8192 pilot is not completion of full169028 scaling.
"""
import argparse,copy,fcntl,gzip,hashlib,inspect,json,math,os,random,time,traceback
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021'
DEFAULT=A/'online-packed-M3to10-P20-variance-8192-20261022-v1'
FIXTURE=A/'online-mixed-layer-prefix-20261022-v1';REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
SEED=2026102227;DIAGNOSTIC_SEED=2026102228;THRESHOLD_VARIANCE_G5=.08

def read(p):return json.loads(Path(p).read_bytes())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);q=p.with_name(p.name+'.%d.tmp'%os.getpid());q.write_bytes(canonical(x)+b'\n');q.replace(p)
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for z in iter(lambda:f.read(8*1024**2),b''):h.update(z)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 p=Path(ref['path']);assert proof(p)=={k:ref[k]for k in('path','bytes','sha256')};return p

def compact(r):
 policy=[]
 for s in r['segments']:
  take=min(20-len(policy),s['policy_length']);policy+=s['context_tokens'][s['request_length']:s['request_length']+take]
  if len(policy)==20:break
 assert 0<len(policy)<=20
 return dict(qid=r['qid'],trajectory_id=r['trajectory_id'],prompt=r['prompt'],policy20=policy)

def prepare(args):
 base=args.base;base.mkdir(exist_ok=True,parents=True);guard=(base/'prepare.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if(base/'fixture-complete.json').exists():return read(base/'fixture-complete.json')
 fixture=read(FIXTURE/'fixture-complete.json');train=read(FIXTURE/'inputs/train-groups.json')
 full=None
 if args.training_questions!=8192:
  if args.full_training_descriptor is None:raise ValueError('Scaling requires exact full G10 groups and accepted Prefix20 source refs descriptor')
  full=read(args.full_training_descriptor);raw=read(checked(full['groups']));train=raw['train']if'train'in raw else raw
  if isinstance(train,list):train={g['qid']:g for g in train}
  assert len(train)==args.training_questions
 assert len(train)==args.training_questions and all(len(g['trajectory_ids'])==len(g['labels'])==10 for g in train.values())
 vb=A/'validation-g10-20261021/materialized/bank';vg=read(vb/'groups.json')['validation'];validation={g['qid']:g for g in vg};assert len(validation)==2048 and all(len(g['trajectory_ids'])==10 for g in validation.values())
 assert not(set(train)&set(validation))and not({g['question_hash']for g in train.values()}&{g['question_hash']for g in validation.values()})
 wanted={t:(role,q)for role,gs in(('train',train),('validation',validation))for q,g in gs.items()for t in g['trajectory_ids']};assert len(wanted)==10*(len(train)+2048)
 inputs=base/'inputs';inputs.mkdir(exist_ok=True);streams=[gzip.open(inputs/('tokens-rank-%d.jsonl.gz'%i),'wt',compresslevel=1)for i in range(4)];qrank={q:i%4 for role,gs in(('train',train),('validation',validation))for i,q in enumerate(sorted(gs))};seen=set();sources=[];started=time.time()
 def accept(r):
  tid=r['trajectory_id']
  if tid not in wanted or tid in seen:return
  role,q=wanted[tid];assert r['qid']==q and r['prompt']and 0<len(r['policy20'])<=20
  r=dict(r,role=role);streams[qrank[q]].write(json.dumps(r,separators=(',',':'))+'\n');seen.add(tid)
 try:
  for ref in fixture['token_shards']:
   p=checked(ref);sources.append(ref)
   with gzip.open(p,'rt')as f:
    for line in f:accept(compact(json.loads(line)))
  if full:
   for ref in full['prefix_sources']:
    p=checked(ref);sources.append(ref)
    with gzip.open(p,'rt')as f:
     for line in f:
      r=json.loads(line)
      if r['trajectory_id']in wanted and r['trajectory_id']not in seen:accept(dict(qid=r['qid'],trajectory_id=r['trajectory_id'],prompt=r['initial_prompt_token_ids'],policy20=r['policy_token_ids'][:20]))
  audit=read(vb/'source-audit.json');refs=[batch['phase_payloads']['full']for batches in audit['events'].values()for batch in batches]
  for ref in refs:
   if set(wanted)<=seen:break
   p=checked(ref);sources.append(ref)
   with gzip.open(p,'rt')as f:
    for line in f:
     r=json.loads(line)
     if r['trajectory_id']not in wanted or r['trajectory_id']in seen:continue
     # Exactly the first20 retained policy tokens; no full answer/tool suffix enters the new input.
     policy=[]
     for turn in r['behavior_turns']:
      policy+=turn['raw_generated_token_ids'][:min(turn['kept_token_count'],20-len(policy))]
      if len(policy)==20:break
     assert r['actor_version']=='official7b-'+REVISION
     accept(dict(qid=r['qid'],trajectory_id=r['trajectory_id'],prompt=r['initial_prompt_token_ids'],policy20=policy))
   atomic(base/'prepare-progress.json',dict(stage='EXTRACTING_POLICY20_TOKENS_ONLY',trajectories=len(seen),expected=len(wanted),seconds=time.time()-started))
 finally:
  for f in streams:f.close()
 assert seen==set(wanted),'Missing genuine own Prefix20 rows; no GPU submit until accepted'
 atomic(inputs/'train-groups.json',train);atomic(inputs/'validation-groups.json',validation);atomic(inputs/'fixed-validation-G5.json',read(FIXTURE/'inputs/validation-groups.json'));atomic(inputs/'fixed-train-G5.json',read(FIXTURE/'inputs/fixed-train2048.json'))
 result=dict(status='PACKED_VARIANCE_OWN_G10_POLICY20_FIXTURE_READY',training_questions=len(train),validation_questions=2048,trajectories=len(seen),token_shards=[proof(inputs/('tokens-rank-%d.jsonl.gz'%i))for i in range(4)],source_proofs=sources,train_groups=proof(inputs/'train-groups.json'),validation_groups=proof(inputs/'validation-groups.json'),fixed_validation_G5=proof(inputs/'fixed-validation-G5.json'),fixed_train_G5=proof(inputs/'fixed-train-G5.json'),source_existing_pilot_fixture=proof(FIXTURE/'fixture-complete.json'),full169028_completed=False,hidden_features_saved=False,final4096_read=False,seconds=time.time()-started)
 atomic(base/'fixture-complete.json',result);return result

def ranks(a):
 import numpy as np
 order=np.argsort(a,kind='stable');out=np.empty(len(a),dtype=float);start=0
 while start<len(a):
  end=start+1
  while end<len(a)and a[order[end]]==a[order[start]]:end+=1
  out[order[start:end]]=(start+end-1)/2;start=end
 return out

def metrics(rows,fixed_g5=False):
 import numpy as np
 y=np.array([r['target_variance']for r in rows]);p=np.array([r['predicted_variance']for r in rows]);mixed=np.array([r['true_mixed']for r in rows],dtype=bool)
 def corr(a,b):return float(np.corrcoef(a,b)[0,1])if np.std(a)>0 and np.std(b)>0 else None
 top=sorted(rows,key=lambda r:(-r['predicted_variance'],r['qid']))[:512];classes={}
 for name,mask in(('all_wrong',np.array([sum(r['labels'])==0 for r in rows])),('Mixed',mixed),('all_correct',np.array([sum(r['labels'])==len(r['labels'])for r in rows]))):
  classes[name]=dict(questions=int(mask.sum()),variance_MSE=float(((p[mask]-y[mask])**2).mean())if mask.any()else None,mean_predicted_variance=float(p[mask].mean())if mask.any()else None)
 result=dict(questions=len(rows),variance_MSE=float(((p-y)**2).mean()),variance_MAE=float(np.abs(p-y).mean()),target4variance_MSE=float(((4*p-4*y)**2).mean()),variance_Pearson=corr(p,y),variance_Spearman=corr(ranks(p),ranks(y)),true_Mixed_fraction=float(mixed.mean()),Top512=dict(Mixed=sum(r['true_mixed']for r in top),questions=len(top),Mixed_fraction=sum(r['true_mixed']for r in top)/len(top)),classwise=classes,score='predicted_population_variance_not_Mixed_probability',selection_score='predicted_population_EM_variance',selection_order='descending',strict_EM_used_for_ranking=False)
 if fixed_g5:
  pred=p>=THRESHOLD_VARIANCE_G5;tp=int((pred&mixed).sum());tn=int((~pred&~mixed).sum());pos=int(mixed.sum());neg=len(rows)-pos;result['predeclared_G5_classification_diagnostic']=dict(variance_threshold=THRESHOLD_VARIANCE_G5,threshold_fitted=False,balanced_accuracy=.5*(tp/pos+tn/neg),Mixed_recall=tp/pos,nonMixed_false_positive_rate=1-tn/neg,confusion=dict(tp=tp,tn=tn,fp=neg-tn,fn=pos-tp))
  for name,mask in(('all_wrong',np.array([sum(r['labels'])==0 for r in rows])),('all_correct',np.array([sum(r['labels'])==5 for r in rows]))):result['classwise'][name]['diagnostic_false_positive_rate']=float(pred[mask].mean())if mask.any()else None
 return result

def run(args):
 import numpy as np
 import torch
 import torch.distributed as dist
 from torch import nn
 from torch.nn.parallel import DistributedDataParallel as DDP
 from transformers import AutoModel,AutoTokenizer
 from .packed_variance_probe import PackedVarianceProbe,PackedVarianceHead,pack_group,plan,view
 base=args.base;rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2;device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);dist.init_process_group('nccl');lock=(base/('writer-rank-%d.lock'%rank)).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 fixture=read(base/'fixture-complete.json');assert fixture['training_questions']==args.training_questions and fixture['status']=='PACKED_VARIANCE_OWN_G10_POLICY20_FIXTURE_READY';inputs=base/'inputs';groups=read(checked(fixture['train_groups']));validation=read(checked(fixture['validation_groups']));records={}
 for i,ref in enumerate(fixture['token_shards']):
  if i%world!=rank:continue
  with gzip.open(checked(ref),'rt')as f:
   for line in f:r=json.loads(line);records[r['trajectory_id']]=r
 torch.manual_seed(9137);np.random.seed(9137);random.seed(9137);model_dir=B/'assets/qwen25-7b-instruct-a09a35458c70';tok=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True);separator=tok.encode('\n### trajectory\n',add_special_tokens=False);assert separator and all(0<=t<tok.vocab_size+1000 for t in separator)
 actor=AutoModel.from_pretrained(str(model_dir),local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to(device).eval().requires_grad_(False);block=copy.deepcopy(actor.layers[21]).float().requires_grad_(True);rotary=copy.deepcopy(actor.rotary_emb).float();actor.layers=nn.ModuleList(list(actor.layers[:21]));actor.norm=nn.Identity();torch.cuda.empty_cache();probe=PackedVarianceProbe(block,rotary,'cache_position'in inspect.signature(block.forward).parameters,PackedVarianceHead()).to(device);ema=copy.deepcopy(probe).eval().requires_grad_(False);optim=torch.optim.Adam(probe.parameters(),lr=args.lr);ddp=DDP(probe,device_ids=[device.index],broadcast_buffers=False);cursor=0;start_epoch=1
 if(base/'latest.json').exists():
  old=torch.load(checked(read(base/'latest.json')),map_location='cpu',weights_only=False);assert old['architecture']=='one_question_M3to10_policy20_packed_variance_v1';probe.load_state_dict(old['raw']);ema.load_state_dict(old['EMA']);optim.load_state_dict(old['Adam']);cursor=old['cursor'];start_epoch=old['epoch']+1
  for state in optim.state.values():
   for k,v in state.items():
    if torch.is_tensor(v)and k!='step':state[k]=v.to(device)
  rng=old['RNG_by_rank'][rank];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'],device)
 qrank={q:i%world for role,gs in(('train',groups),('validation',validation))for i,q in enumerate(sorted(gs))}
 if rank==0:atomic(base/'controller.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],world_size=2,layer=21,capture_layer=20,prefix=20,M_range=[3,10],question_prompt_occurrences_per_group=1,actor_rows_per_group=1,separator_text='\n### trajectory\n',separator_token_ids=separator,gradient_accumulation=8,global_batch=128,target='4*selected_M_population_EM_variance',loss='MSE_only',selection_score='predicted_population_EM_variance',selection_order='descending',candidates_same_M=True,variance_G5_diagnostic_threshold=.08,EMA_decay=.99,raw_updates_EMA=True,training_questions=len(groups),pilot_not_full169028=args.training_questions==8192,final4096_read=False,time=time.time()))
 def capture(vs):
  started=time.monotonic();packed=[pack_group(v,records,separator)for v in vs];width=max(len(v['input_ids'])for v in packed);ids=torch.full((len(vs),width),151643,dtype=torch.long,device=device);mask=torch.zeros_like(ids,dtype=torch.bool);positions=torch.zeros_like(ids);types=torch.zeros_like(ids)
  for i,row in enumerate(packed):
   n=len(row['input_ids']);ids[i,:n]=torch.tensor(row['input_ids'],device=device);mask[i,:n]=torch.tensor(row['attention_mask'],device=device);positions[i,:n]=torch.tensor(row['position_ids'],device=device);types[i,:n]=torch.tensor(row['token_types'],device=device)
  with torch.no_grad():features=actor(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True).last_hidden_state.detach()
  return dict(features=features,attention_mask=mask,position_ids=positions,token_types=types),time.monotonic()-started
 fixed_validation=read(checked(fixture['fixed_validation_G5']));fixed_train=read(checked(fixture['fixed_train_G5']))
 def explicit(g):
  em=g['labels'];m=len(em);mean=sum(em)/m;return dict(g,M=m,target_variance=mean*(1-mean),target4variance=4*mean*(1-mean),true_mixed=0<sum(em)<m)
 sets={'fixed_train2048_G5':[explicit(v)for v in fixed_train],'fixed_validation2048_G5':[explicit(g)for g in fixed_validation.values()]}
 for m in(3,5,10):sets['fixed_validation2048_M%d'%m]=[view(validation[q],seed=DIAGNOSTIC_SEED,epoch=0,fixed_m=m)for q in sorted(validation)]
 def evaluate(epoch):
  probe.eval();before=time.time();out={}
  for role,allviews in sets.items():
   own=[v for v in allviews if qrank[v['qid']]==rank];rows=[]
   for begin in range(0,len(own),8):
    vs=own[begin:begin+8];data,_=capture(vs)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):raw=torch.sigmoid(probe(data)).cpu().tolist();av=torch.sigmoid(ema(data)).cpu().tolist()
    rows.extend([dict(qid=v['qid'],trajectory_ids=v['trajectory_ids'],labels=v['labels'],M=v['M'],true_mixed=v['true_mixed'],target_variance=v['target_variance'],raw_predicted_variance=x/4,EMA_predicted_variance=y/4)for v,x,y in zip(vs,raw,av)])
   received=[None]*world;dist.all_gather_object(received,rows)
   if rank==0:
    allrows=[r for rs in received for r in rs];assert len(allrows)==len(allviews)and len({r['qid']for r in allrows})==len(allviews)
    with gzip.open(base/('epoch-%02d-%s-predictions.jsonl.gz'%(epoch,role)),'wt')as f:
     for row in allrows:f.write(json.dumps(row)+'\n')
    out[role]={state:metrics([dict(r,predicted_variance=r[state+'_predicted_variance'])for r in allrows],role.endswith('_G5'))for state in('raw','EMA')}
   atomic(base/('evaluation-progress-rank-%d.json'%rank),dict(epoch=epoch,role=role,own_questions=len(own),seconds=time.time()-before))
  if rank==0:atomic(base/('evaluation-epoch-%02d.json'%epoch),dict(epoch=epoch,cursor=cursor,metrics=out,seconds=time.time()-before,final4096_scored=False,loss='MSE(4*population_variance)',threshold_predeclared_G5_variance=.08))
  dist.barrier();probe.train()
 if start_epoch==1:evaluate(0)
 elif not(base/('evaluation-epoch-%02d.json'%(start_epoch-1))).exists():evaluate(start_epoch-1)
 for epoch in range(start_epoch,args.epochs+1):
  fullplan=plan(groups,epoch,SEED);mine=[v for v in fullplan if qrank[v['qid']]==rank];assert len(mine)==len(groups)//world and len({v['qid']for v in mine})==len(mine);started=time.time();hist={m:sum(v['M']==m for v in fullplan)for m in range(3,11)}
  for offset in range(0,len(mine),64):
   batch=mine[offset:offset+64];optim.zero_grad(set_to_none=True);lossnum=0.;prefill=0.;actor_rows=0
   for lo in range(0,len(batch),8):
    vs=batch[lo:lo+8];data,cost=capture(vs);prefill+=cost;actor_rows+=len(vs);target=torch.tensor([v['target4variance']for v in vs],dtype=torch.float32,device=device)
    with torch.autocast('cuda',dtype=torch.bfloat16):prediction=torch.sigmoid(ddp(data));loss=(prediction-target).square().sum()/len(batch)
    loss.backward();lossnum+=float(loss.detach())
   ng=math.sqrt(sum(float(p.grad.detach().float().square().sum())for p in probe.block.parameters()if p.grad is not None));hg=math.sqrt(sum(float(p.grad.detach().float().square().sum())for p in probe.group_head.parameters()if p.grad is not None));assert ng>0 and hg>0;torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);optim.step()
   with torch.no_grad():
    for p,e in zip(probe.parameters(),ema.parameters()):e.mul_(.99).add_(p,alpha=.01)
   cursor+=1;row=dict(stage='ONE_QUESTION_VARIABLE_M_PACKED_VARIANCE_MSE_UPDATES',epoch=epoch,cursor=cursor,rank=rank,epoch_questions_seen=offset+len(batch),target4variance_MSE=lossnum,variance_MSE=lossnum/16,native_gradient_L2=ng,head_gradient_L2=hg,actor_rows=actor_rows,question_groups=len(batch),prefill_rows_equal_groups=actor_rows==len(batch),prefill_seconds=prefill,configured_global_batch=128,actual_global_batch=len(batch)*world,local_batch=len(batch),micro=8,gradient_accumulation=math.ceil(len(batch)/8),M_counts={m:sum(v['M']==m for v in batch)for m in range(3,11)},time=time.time());atomic(base/('progress-rank-%d.json'%rank),row)
   if rank==0:
    print(json.dumps(row),flush=True)
    with(base/'curve.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
  rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device));rngs=[None]*world;dist.all_gather_object(rngs,rng)
  if rank==0:
   path=base/('checkpoint-epoch-%02d.pt'%epoch);tmp=path.with_suffix('.tmp');torch.save(dict(raw=probe.state_dict(),EMA=ema.state_dict(),Adam=optim.state_dict(),RNG_by_rank=rngs,cursor=cursor,epoch=epoch,plan_sha256=digest(fullplan),architecture='one_question_M3to10_policy20_packed_variance_v1',training_questions=len(groups),layer=21,prefix=20,lr=args.lr,raw_updates_EMA=True,loss='MSE_only_target4variance'),tmp);tmp.replace(path);atomic(base/'latest.json',proof(path));atomic(base/('coverage-epoch-%02d.json'%epoch),dict(epoch=epoch,questions=len(fullplan),unique_questions=len({v['qid']for v in fullplan}),M_counts=hist,trajectory_visits=sum(v['M']for v in fullplan),plan_sha256=digest(fullplan),sampling_independent_of_EM=True,seconds=time.time()-started))
  dist.barrier();evaluate(epoch)
 if rank==0:atomic(base/'complete.json',dict(status='PACKED_VARIANCE_PILOT_COMPLETE'if len(groups)==8192 else'PACKED_VARIANCE_FULL_SWEEP_COMPLETE',epochs=args.epochs,cursor=cursor,training_questions=len(groups),pilot_not_full169028=len(groups)==8192,final4096_scored=False))
 dist.destroy_process_group()

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=['prepare','train'],required=True);p.add_argument('--base',type=Path,default=DEFAULT);p.add_argument('--training-questions',type=int,default=8192);p.add_argument('--full-training-descriptor',type=Path);p.add_argument('--epochs',type=int,default=3);p.add_argument('--lr',type=float,default=5e-6);a=p.parse_args();a.base.mkdir(parents=True,exist_ok=True)
 try:
  if a.action=='prepare':print(json.dumps(prepare(a)))
  else:run(a)
 except BaseException:
  atomic(a.base/('failure-rank-%s.json'%os.environ.get('RANK','cpu')),dict(time=time.time(),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
