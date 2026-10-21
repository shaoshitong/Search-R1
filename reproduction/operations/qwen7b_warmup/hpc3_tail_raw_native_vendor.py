"""Original 7B native G5 sampler, no hidden writer; a shared local retriever."""
import argparse,gzip,json,os,time,signal,traceback
from pathlib import Path
from reproduction.operations.qwen7b_warmup import sampling as s
from reproduction.operations.qwen7b_warmup.download_model import atomic,fingerprints,REVISION
from reproduction.prefix20_online.pool import digest,read

def validate(config_path,config,outer,phase,rank,path):
 row=read(path);assert row['run_id']==config['run_id']and row['config_sha256']==digest(config_path)
 for key in ('states','prefix')if phase=='prefix'else('full','own_em'):
  item=row[key];assert Path(item['path']).stat().st_size==item['bytes']and digest(item['path'])==item['sha256']
 return row,None
s.validate_rank_receipt=validate

def fullprefill(config,model,seed,free,total):
 if 'llm'not in s._CACHED:
  from vllm import LLM
  import gc,torch
  utilization=min(float(config.get('rollout_utilization_cap',.90)),(free-3*1024**3)/total);limits=dict(max_num_seqs=int(config.get('rollout_max_num_seqs',512)),max_num_batched_tokens=int(config.get('rollout_max_batched_tokens',32768)));attempts=[]
  while True:
   try:
    llm=LLM(model=model,tokenizer=model,trust_remote_code=False,dtype=s.COMPUTE_DTYPE,tensor_parallel_size=1,max_model_len=s.BUDGET['total_tokens'],seed=seed,gpu_memory_utilization=utilization,swap_space=0,enforce_eager=True,enable_chunked_prefill=False,enable_prefix_caching=False,disable_async_output_proc=True,disable_log_stats=True,**limits);break
   except Exception as e:
    if 'out of memory'not in str(e).lower()or len(attempts)>=2:raise
    attempts.append(dict(actual_oom=True,utilization=utilization,limits=dict(limits)));gc.collect();torch.cuda.empty_cache();utilization*=.85;limits['max_num_seqs']=max(32,limits['max_num_seqs']//2);limits['max_num_batched_tokens']=max(4096,limits['max_num_batched_tokens']//2)
  s._CACHED['llm']=llm,utilization,limits,attempts
 return s._CACHED['llm']
s._build_llm=fullprefill

def phase(path,mode):
 c=read(path);root=s.root_dir(c,1);out=root/('prefix'if mode=='prefix'else'full');out.mkdir(parents=True,exist_ok=True)
 if not(out/'rank-0-complete.json').exists():
  partial=[p for p in out.glob('rank-0-*')]
  if partial:
   q=out/('unsealed-preserved-%d'%time.time_ns());q.mkdir()
   for p in partial:p.rename(q/p.name)
 row=s.worker(path,1,mode,0,root/'probe-score/selection.json')
 if mode=='prefix':
  assert 'actor_hidden'not in row
  scan=dict(ranks=[row],questions=row['questions'],trajectories=row['trajectories'],head_prefix_tokens=20,model_revision=REVISION,hidden_cached=False)
  atomic(out/'scan.json',scan);complete=dict(scan,scan_sha256=digest(out/'scan.json'),interaction_protocol=row['interaction_protocol'],actual_compute_dtype=row['actual_compute_dtype']);atomic(out/'complete.json',complete)
  groups={}
  with gzip.open(row['prefix']['path'],'rt')as f:
   for line in f:
    x=json.loads(line);groups.setdefault(x['qid'],[]).append(x['trajectory_id'])
  expected=read(root/'questions.json')['rows'];assert len(groups)==len(expected)and all(len(x)==5 for x in groups.values())
  selected=[dict(qid=g['qid'],question_hash=g['question_hash'],candidate_ordinal=g['candidate_ordinal'],trajectory_ids=groups[g['qid']])for g in expected]
  (root/'probe-score').mkdir(exist_ok=True);atomic(root/'probe-score/selection.json',dict(run_id=c['run_id'],outer=1,actor_version=c['actor_version'],prefix_scan_sha256=complete['scan_sha256'],kind='ALL_G5_NO_SELECTION',selected=selected))
 else:
  labels={}
  with Path(row['own_em']['path']).open()as f:
   for line in f:
    x=json.loads(line);assert x['trajectory_id']not in labels and x['strict_em']in(0,1);labels[x['trajectory_id']]=x['strict_em']
  assert set(labels)=={tid for g in read(root/'probe-score/selection.json')['selected']for tid in g['trajectory_ids']}
  atomic(out/'complete.json',dict(row,labels=len(labels),actor_optimizer_updates=0,model_revision=REVISION))
 return row

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();base=Path(a.base);c=read(base/'config.json');root=base/'workers'/('worker-%02d'%a.shard);root.mkdir(parents=True,exist_ok=True);accepted=[];started=time.monotonic()
 assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==2
 inputs=sorted((base/'inputs').glob('shard-*/batch-*.json'));own=inputs[a.shard::a.workers]
 atomic(root/'assignment.json',dict(worker=a.shard,workers=a.workers,batches=[str(p)for p in own],run_id=c['run_id']))
 for inp in own:
  batch=read(inp);dest=base/'sampling'/inp.parent.name/'batches'/inp.stem;receipt=dest/'complete.json'
  if receipt.exists():
   x=read(receipt);assert x['input_sha256']==digest(inp)and x['run_id']==c['run_id'];accepted.append(x);continue
  outer=dest/'outers/outer-000001';outer.mkdir(parents=True,exist_ok=True);atomic(outer/'questions.json',dict(outer=1,rows=batch['rows']))
  cfg=dict(c,base=str(dest),world_size=1,actor_model_dir=c['model_dir'],actor_version='official7b-'+REVISION,actor_hidden_native=False,prefix_policy_tokens=20,rollout_memory_utilization='auto',rollout_utilization_cap=.90,rollout_free_reserve_gib=3,rollout_max_num_seqs=512,rollout_max_batched_tokens=32768,rollout_enforce_eager=True,rollout_retry_initialization_oom=True)
  atomic(dest/'config.json',cfg);prefix=phase(dest/'config.json','prefix');full=phase(dest/'config.json','continue')
  x=dict(run_id=c['run_id'],input_sha256=digest(inp),batch=inp.stem,questions=len(batch['rows']),trajectories=5*len(batch['rows']),prefix=prefix,full=full,hidden_cached=False,model_revision=REVISION,actor_optimizer_updates=0,epoch=time.time());atomic(receipt,x);accepted.append(x)
  progress=dict(worker=a.shard,completed_assigned_batches=len(accepted),total_assigned_batches=len(own),questions=sum(x['questions']for x in accepted),trajectories=sum(x['trajectories']for x in accepted),latest_input=str(inp),elapsed=time.monotonic()-started,epoch=time.time());atomic(root/'progress.json',progress);print(json.dumps(progress),flush=True)
 atomic(root/'complete.json',dict(status='ASSIGNED_NEW_G5_BATCHES_COMPLETE',run_id=c['run_id'],worker=a.shard,questions=sum(x['questions']for x in accepted),trajectories=sum(x['trajectories']for x in accepted),batches=len(accepted),epoch=time.time()))
if __name__=='__main__':
 def terminate(signum,frame):raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,terminate)
 try:main()
 except BaseException as e:
  traceback.print_exc()
  base=Path(__import__('sys').argv[__import__('sys').argv.index('--base')+1]);shard=__import__('sys').argv[__import__('sys').argv.index('--shard')+1]
  try:atomic(base/('worker-failure-'+shard+'.json'),dict(error_type=type(e).__name__,detail=str(e)[:1000],traceback=traceback.format_exc(),epoch=time.time()))
  except BaseException:pass
  raise

