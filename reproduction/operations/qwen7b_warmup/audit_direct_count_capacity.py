"""Independent bounded CPU acceptance of finished capacity diagnostic outputs.

No model/tensor load, no hidden cache payload read, no external/final access.
Counts NLL/accuracy remain producer-qualified (predictions contain q and K,
not six probabilities); all Mixed metrics and labels are recomputed here.
"""
import argparse,collections,fcntl,gzip,hashlib,importlib.util,json,math,os,resource,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
T=B/'algorithm-improvement-20261021/direct-count-head-capacity1536-v1'
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):return json.loads(Path(p).read_bytes())
def checked(ref):
 p=Path(ref['path']);r=proof(p)
 if any(r[k]!=ref[k]for k in ('bytes','sha256')):raise ValueError('Saved descriptor changed')
 return p
def auc(scores,y):
 n=sum(y);m=len(y)-n
 if not n or not m:return None
 a=sorted(zip(scores,y));rank=0.;i=0
 while i<len(a):
  j=i+1
  while j<len(a)and a[j][0]==a[i][0]:j+=1
  rank+=sum(v for _,v in a[i:j])*(i+1+j)/2;i=j
 return (rank-n*(n+1)/2)/(n*m)
def metric(rows,top):
 y=[0<r['K']<5 for r in rows];s=[r['q']for r in rows];pos=sum(y);neg=len(y)-pos
 tp=sum(a and b>=.5 for a,b in zip(y,s));tn=sum(not a and b<.5 for a,b in zip(y,s));a=tp/pos if pos else None;b=tn/neg if neg else None
 chosen=sorted(rows,key=lambda r:(-r['q'],r['qid']))[:256]if top else []
 return dict(K_histogram=[sum(r['K']==k for r in rows)for k in range(6)],Mixed_fraction=pos/len(rows),Mixed_BA=.5*(a+b)if pos and neg else None,
  Mixed_recall=a,nonMixed_recall=b,Mixed_AUC=auc(s,y),predicted_Mixed_fraction=sum(v>=.5 for v in s)/len(s),
  Top256_of1024=dict(questions=256,Mixed_count=sum(0<r['K']<5 for r in chosen),Mixed_fraction=sum(0<r['K']<5 for r in chosen)/256)if top else None)
def close(a,b):
 if isinstance(a,dict):return a.keys()==b.keys()and all(close(a[k],b[k])for k in a)
 if isinstance(a,list):return a==b
 if a is None:return b is None
 return abs(a-b)<1e-10
def audit(out):
 started=time.time();out=Path(out);out.mkdir(exist_ok=True)
 with (out/'audit.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if not (T/'complete.json').exists():raise ValueError('Wait for all three actual complete forks; no partial audit')
  complete=read(T/'complete.json');c=read(checked(complete['config']));cache=read(checked(complete['cache']))
  if complete['status']!='FINITE_FROZEN_E1_HEAD_ONLY_THREE_LR_CAPACITY_COMPLETE'or not complete['legacy_unchanged']or complete['legacy_optimizer_updates']!=0 or complete['actor_updates']!=0:raise ValueError('Finite/frozen branch not complete')
  if complete['original_frozen_legacy_before']!=complete['original_frozen_legacy_after']or complete['original_frozen_legacy_before']!=cache['frozen_legacy_before']or not all(cache['first_real_same_API_checks'].values()):raise ValueError('Frozen native API cache/hash evidence differs')
  split=read(checked(c['split']));orders=read(checked(c['orders']))['epochs'];fit=split['fit_qids'];hold=split['holdout_qids']
  if len(fit)!=512 or len(hold)!=1024 or set(fit)&set(hold)or len(orders)!=20:raise ValueError('Fixed independent train-only split/order differs')
  for order in orders:
   if len(order)!=16 or any(len(q)!=32 for q in order)or [q for batch in order for q in batch].count(fit[0])!=1 or set(q for batch in order for q in batch)!=set(fit):raise ValueError('Same full512 questions per epoch required')
  bank=read(checked(c['g10_complete']));lookup={g['qid']:g for g in read(checked(bank['outputs']['groups.json']))['train']}
  plan=read(checked(c['source_plan']));cohort=[q for row in plan['batches'][:256]for q in row]
  rank=sorted(cohort,key=lambda q:(digest(['frozen-E1-head-capacity-v1',9137,q]),q))
  if fit!=rank[:512]or hold!=rank[512:1536]:raise ValueError('Split no longer qid-only hash selected')
  epochs={q:plan['batch_epochs'][i]for i,bs in enumerate(plan['batches'][:256])for q in bs};ref=c['source_files']['g10_grouping.py'];path=checked(ref)
  spec=importlib.util.spec_from_file_location('auditor_original_g10_grouping',path);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
  expected={};views={}
  for q in fit+hold:
   vs=g.build_training_views([lookup[q]],role='train',epoch=epochs[q],**c['grouping'])['groups'];views[q]=vs;expected[q]=min(vs,key=lambda x:tuple(x['source_indices']))
  fixed=read(T/'primary-G5-plan.json')
  if fixed['labels_used']or len(fixed['groups'])!=1536:raise ValueError('One label-blind primary per actual question required')
  for row in fixed['groups']:
   if row['source_indices']!=expected[row['qid']]['source_indices']or row['trajectory_ids']!=expected[row['qid']]['trajectory_ids']:raise ValueError('Primary G5 identity changed')
  results={};checkpoints={};prediction_refs={}
  for lr,final in complete['results'].items():
   base=T/('lr-'+lr);done=read(base/'complete.json');first=read(base/'first-update-replay.json');cp=read(base/'latest.json')
   if cp['cursor']!=320 or cp['new_Adam_steps']!=320 or first['status']!='PASSED'or first['expected']!=first['actual']or first['Adam_before']!=0 or first['Adam_after']!=1:raise ValueError('Real320 updates and first replay required')
   if [r['cursor']for r in done['first_six']]!=list(range(1,7))or any(x['new_Adam_steps']!=x['cursor']for x in done['first_six']):raise ValueError('Actual continuous5plusnext missing')
   cps=sorted((base/'checkpoints').glob('cursor-*.pt'))
   if [int(p.stem.split('-')[-1])for p in cps]!=[0,1]+list(range(16,321,16)):raise ValueError('All22 intermediate/full epoch checkpoints required')
   checkpoints[lr]=[proof(p)for p in cps]
   results[lr]={};prediction_refs[lr]={}
   for ep in (0,1,2,5,10,20):
    folder=base/'evaluations'/('epoch-%02d'%ep);ev=read(folder/'complete.json');checked(ev['checkpoint'])
    if ev['cursor']!=ep*16 or not ev['training_components_unchanged']or not ev['checkpoint']['full_CPU_reload_passed']:raise ValueError('Checkpoint-first eval/full CPU reload evidence differs')
    if ep==0 and (ev['checkpoint']['components']['raw']!=complete['same_initial_head_raw']or ev['checkpoint']['components']['ema']!=complete['same_initial_head_EMA']):raise ValueError('Fork init raw/EMA differ')
    results[lr][str(ep)]={};prediction_refs[lr][str(ep)]={}
    for model in ('raw','EMA'):
     results[lr][str(ep)][model]={}
     for name,qs,all4 in [('fit_all4',fit,True),('fit_primaryG5',fit,False),('internal_holdout_primaryG5',hold,False)]:
      p=folder/(model+'-'+name+'.jsonl.gz')
      with gzip.open(p,'rt')as f:rows=[json.loads(line)for line in f]
      if len(rows)!=len(qs)*(4 if all4 else 1):raise ValueError('Output coverage differs')
      actual=collections.defaultdict(list)
      for row in rows:
       if row['qid']not in qs or type(row['K'])is not int or not 0<=row['K']<=5 or not math.isfinite(row['q'])or not 0<=row['q']<=1:raise ValueError('Invalid score/own label')
       actual[row['qid']].append(row['K'])
      for q in qs:
       want=[sum(v['labels'])for v in views[q]]if all4 else [sum(expected[q]['labels'])]
       if sorted(actual[q])!=sorted(want):raise ValueError('Own actual strict-EM G5 labels differ')
      m=metric(rows,name=='internal_holdout_primaryG5');published=ev['metrics'][model][name]
      if any(not close(value,published[key])for key,value in m.items()):raise ValueError('Independent Mixed metric mismatch')
      if not math.isfinite(published['count_NLL'])or published['count_NLL']<0 or abs(round(published['count_accuracy']*len(rows))-published['count_accuracy']*len(rows))>1e-6:raise ValueError('Invalid producer-qualified count metrics')
      m.update(count_NLL=published['count_NLL'],count_accuracy=published['count_accuracy'],count_NLL_accuracy_independently_recomputed=False)
      results[lr][str(ep)][model][name]=m;prediction_refs[lr][str(ep)][model+'-'+name]=proof(p)
  result=dict(status='HEAD_CAPACITY_FINITE_THREE_FORK_FULL_FILES_OWN_LABELS_AND_MIXED_METRICS_INDEPENDENTLY_ACCEPTED',source_complete=proof(T/'complete.json'),config=complete['config'],cache_receipt=complete['cache'],
   frozen_legacy_unchanged=True,cache_same_native_API_golden_passed=True,fit_questions=512,internal_holdout_questions=1024,three_forks_same_init=True,three_forks_Adam0_to320=True,
   checkpoints=checkpoints,metrics=results,predictions=prediction_refs,
   count_NLL_and_count_accuracy='Qualified producer FP32 classifier outputs; independent q-only predictions do not permit rederiving six-class metrics',
   CPU_tensor_load=False,hidden_cache_payload_read=False,full_head_checkpoint_SHA_payload_read=True,external_or_final_read=False,elapsed_seconds=time.time()-started,epoch=time.time())
  raw=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n';p=out/'complete.json';p.write_bytes(raw);return dict(receipt=proof(p),result=result)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.parent!=B/'algorithm-improvement-20261021':p.error('Separate remote audit only')
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120))
 print(json.dumps(audit(a.output)))
if __name__=='__main__':main()
