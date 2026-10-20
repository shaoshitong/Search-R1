"""CPU-only paired old5/new5 diagnosis on a sealed seen-training NQ8192 G10.
No fitting, tensor/model/feature reads, threshold search, external or final pools.
"""
import argparse, collections, fcntl, gzip, hashlib, json, math, os, pathlib, resource, time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
SC=B/'algorithm-improvement-20261021/peer8192-frozen-calibration-scores-v2'
OUT=B/'algorithm-improvement-20261021/peer8192-paired-old-new-20261021'
ROLE='SEEN_NQ8192_TRAINING_CALIBRATION_ONLY'
GROUP_SHA='2c94b6413261c7cdbcd53f20dbfd69f3838c71665749e01d83928ede81c2ceeb'
E4_SHA='969faa1c8fc3ff9a81afa72e9abdd8fccad3fec78ddeb99844a54e3275994f3a'
E4_CP='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'
def read(p):return json.loads(pathlib.Path(p).read_text())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def proof(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 p=pathlib.Path(ref['path'])
 if not p.is_absolute() or p.resolve()!=p or not str(p).startswith(str(B)+'/'):raise ValueError('Bound remote shared source only')
 actual=proof(p)
 if actual['sha256']!=ref['sha256'] or ('bytes' in ref and actual['bytes']!=ref['bytes']):raise ValueError('Source SHA/size differs: '+str(p))
 return p
def atomic(p,v):
 p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def immutable(p,v):
 if pathlib.Path(p).exists():
  if read(p)!=v:raise ValueError('Existing immutable result differs')
 else:atomic(p,v)
def plan_ids(qids,seed=9137,stratum_size=2048):
 """Inputs are IDs only. Labels/probabilities cannot affect these disjoint sets."""
 if len(set(qids))!=len(qids) or len(qids)!=4*stratum_size:raise ValueError('Four exact disjoint identity strata required')
 order=sorted(qids,key=lambda q:(digest(['old5-new5-id-strata-v1',seed,q]),str(q)))
 parts=[order[i*stratum_size:(i+1)*stratum_size] for i in range(4)]
 return dict(seed=seed,selection_rule='SHA256(canonical([old5-new5-id-strata-v1,9137,qid])) then qid; four consecutive equal blocks',labels_consulted=False,probabilities_consulted=False,groups=[dict(stratum=i,qids=part,qid_sha256=digest(part)) for i,part in enumerate(parts)],all_qids_sha256=digest(order),questions=len(order))
def product(xs):
 value=1.
 for x in xs:value*=x
 return value
def qscore(ps):return max(0.,1-product(ps)-product([1-p for p in ps]))
def ranks(values):
 order=sorted(range(len(values)),key=lambda i:values[i]);out=[0.]*len(values);i=0
 while i<len(order):
  j=i+1
  while j<len(order) and values[order[j]]==values[order[i]]:j+=1
  for k in range(i,j):out[order[k]]=(i+j-1)/2
  i=j
 return out
def pearson(a,b):
 if len(a)!=len(b) or not a:return None
 ma=sum(a)/len(a);mb=sum(b)/len(b);aa=sum((x-ma)**2 for x in a);bb=sum((x-mb)**2 for x in b)
 return sum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(aa*bb) if aa and bb else None
def auc(scores,ys):
 positive=sum(ys);negative=len(ys)-positive
 if not positive or not negative:return None
 rr=ranks(scores);return (sum(r+1 for r,y in zip(rr,ys) if y)-positive*(positive+1)/2)/(positive*negative)
def bce(logit,y):return max(logit,0)-logit*y+math.log1p(math.exp(-abs(logit)))
def avg(xs):return sum(xs)/len(xs) if xs else None
def classify(k):return 'allwrong' if k==0 else 'allcorrect' if k==5 else 'mixed'
def group_metrics(rows):
 ks=[r['k'] for r in rows];ys=[0<k<5 for k in ks];scores=[r['q'] for r in rows];ps=[q>=.5 for q in scores]
 tp=sum(p and y for p,y in zip(ps,ys));fp=sum(p and not y for p,y in zip(ps,ys));fn=sum(not p and y for p,y in zip(ps,ys));tn=sum(not p and not y for p,y in zip(ps,ys))
 tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
 classes={}
 for name in ('allwrong','mixed','allcorrect'):
  ix=[i for i,k in enumerate(ks) if classify(k)==name];n=sum(ps[i] for i in ix)
  classes[name]=dict(questions=len(ix),predicted_mixed=n,predicted_mixed_fraction=n/len(ix) if ix else None,false_positive_rate=n/len(ix) if ix and name!='mixed' else None,mean_q=avg([scores[i] for i in ix]))
 return dict(questions=len(rows),k_histogram=[ks.count(k) for k in range(6)],mixed_questions=sum(ys),mixed_fraction=sum(ys)/len(rows),confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),mixed_recall=tpr,nonmixed_recall=tnr,mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,mixed_roc_auc=auc(scores,ys),classwise=classes)
def trajectory_metrics(rows):
 zs=[z for r in rows for z in r['z']];ps=[p for r in rows for p in r['p']];ys=[y for r in rows for y in r['y']]
 result=dict(trajectories=len(ys),correct=sum(ys),wrong=len(ys)-sum(ys),trajectory_roc_auc=auc(ps,ys),trajectory_bce=avg([bce(z,y) for z,y in zip(zs,ys)]))
 for y,name in ((1,'correct'),(0,'wrong')):result[name+'_bce']=avg([bce(z,t) for z,t in zip(zs,ys) if t==y])
 mixed=[r for r in rows if 0<r['k']<5];gaps=[];local_aucs=[];my=[];mp=[]
 for r in mixed:
  good=[p for p,y in zip(r['p'],r['y']) if y];bad=[p for p,y in zip(r['p'],r['y']) if not y]
  gaps.append(avg(good)-avg(bad));local_aucs.append(sum((a>b)+.5*(a==b) for a in good for b in bad)/(len(good)*len(bad)));my+=r['y'];mp+=r['p']
 result['within_true_Mixed']=dict(questions=len(mixed),question_mean_correct_minus_wrong_probability=avg(gaps),positive_gap_questions=sum(g>0 for g in gaps),zero_gap_questions=sum(g==0 for g in gaps),negative_gap_questions=sum(g<0 for g in gaps),question_mean_pairwise_trajectory_AUC=avg(local_aucs),pooled_trajectory_AUC=auc(mp,my),pooled_correct_minus_wrong_probability=avg([p for p,y in zip(mp,my) if y])-avg([p for p,y in zip(mp,my) if not y]) if my else None,macro_denominator='eligible true Mixed questions; every question weight1')
 return result
def pairing(old,new):
 if [r['qid'] for r in old]!=[r['qid'] for r in new]:raise ValueError('Exact same question ordering required')
 q0=[r['q'] for r in old];q1=[r['q'] for r in new];transition=collections.Counter();binary=collections.Counter();errors=collections.Counter();per={}
 for a,b in zip(old,new):
  y0=0<a['k']<5;y1=0<b['k']<5;p0=a['q']>=.5;p1=b['q']>=.5
  transition[classify(a['k'])+'->'+classify(b['k'])]+=1;key=str(int(y0))+'->'+str(int(y1));binary[key]+=1
  errors[str(int(p0!=y0))+'->'+str(int(p1!=y1))]+=1
  per.setdefault(key,[]).append((a,b))
 for key,values in per.items():
  per[key]=dict(questions=len(values),mean_old_q=avg([a['q'] for a,b in values]),mean_new_q=avg([b['q'] for a,b in values]),mean_new_minus_old_q=avg([b['q']-a['q'] for a,b in values]),old_errors=sum((a['q']>=.5)!=(0<a['k']<5) for a,b in values),new_errors=sum((b['q']>=.5)!=(0<b['k']<5) for a,b in values))
 return dict(questions=len(old),actual_group_class_transition=dict(transition),actual_Mixed_transition=dict(binary),classification_error_transition=dict(errors),score_pearson=pearson(q0,q1),score_spearman=pearson(ranks(q0),ranks(q1)),mean_new_minus_old_q=avg([b-a for a,b in zip(q0,q1)]),mean_absolute_q_change=avg([abs(b-a) for a,b in zip(q0,q1)]),predicted_Mixed_flip_0_to_1=sum(a<.5<=b for a,b in zip(q0,q1)),predicted_Mixed_flip_1_to_0=sum(b<.5<=a for a,b in zip(q0,q1)),by_actual_Mixed_transition=per,trajectory_index_pairing_claim=False)
def selections(rows,plan):
 by={r['qid']:r for r in rows};out=[]
 for part in plan['groups']:
  own=[by[q] for q in part['qids']];chosen=sorted(own,key=lambda r:(-r['q'],str(r['qid'])))[:512]
  out.append(dict(stratum=part['stratum'],source_qid_sha256=part['qid_sha256'],pool=group_metrics(own),selected512=group_metrics(chosen),selected_qids_sha256=digest([r['qid'] for r in chosen])))
 chosen=sorted(rows,key=lambda r:(-r['q'],str(r['qid'])))[:2048]
 return dict(strata=out,global_top2048=group_metrics(chosen),global_selected_qids_sha256=digest([r['qid'] for r in chosen]),selection_uses_labels=False)
def validate_row(g,row):
 for key in ('qid','question_hash','trajectory_ids'):
  if row[key]!=g[key]:raise ValueError('Wrong own identity: '+key)
 if row['strict_em']!=g['labels'] or len(row['strict_em'])!=10 or any(type(y) is not int or y not in (0,1) for y in row['strict_em']):raise ValueError('Wrong own strict EM')
 zs=row['logits'];ps=row['p_j']
 if len(zs)!=10 or len(ps)!=10 or any(not math.isfinite(z) for z in zs) or any(not math.isfinite(p) or not 0<=p<=1 for p in ps):raise ValueError('Ten finite own logits/probabilities required')
 for z,p in zip(zs,ps):
  want=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
  if abs(want-p)>2e-7:raise ValueError('Sigmoid/logit mismatch')
 if abs(qscore(ps)-row['predicted_mixed_G10'])>1e-12:raise ValueError('G10 score differs from own probabilities')
 return [dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'][start:start+5],y=row['strict_em'][start:start+5],p=ps[start:start+5],z=zs[start:start+5],k=sum(row['strict_em'][start:start+5]),q=qscore(ps[start:start+5])) for start in (0,5)]
def cross_model(results,plan):
 if set(results)!={'E4','peer256'}:return None
 def load(item):
  with gzip.open(str(checked(item['paired_rows'])),'rt') as f:return [json.loads(line) for line in f]
 left=load(results['E4']);right=load(results['peer256']);assert len(left)==len(right)==8192
 out=dict(reference_model='E4',comparison_model='peer256',same_original_questions_and_strata=True)
 for scope in ('old5','new5'):
  a=[r[scope] for r in left];b=[r[scope] for r in right]
  if any(any(x[k]!=y[k] for k in ('qid','trajectory_ids','y','k')) for x,y in zip(a,b)):raise ValueError('Cross-model own inputs/EM differ')
  comparison=pairing(a,b);comparison['score_difference_direction']='peer256 minus E4; same actual labels'
  sub=[];aa={r['qid']:r for r in a};bb={r['qid']:r for r in b}
  for part in plan['groups']:
   sa={r['qid'] for r in sorted([aa[q] for q in part['qids']],key=lambda r:(-r['q'],str(r['qid'])))[:512]}
   sb={r['qid'] for r in sorted([bb[q] for q in part['qids']],key=lambda r:(-r['q'],str(r['qid'])))[:512]}
   sub.append(dict(stratum=part['stratum'],top512_overlap=len(sa&sb),top512_Mixed_E4=sum(0<aa[q]['k']<5 for q in sa),top512_Mixed_peer256=sum(0<bb[q]['k']<5 for q in sb)))
  comparison['same_strata_selection_pairing']=sub
  sa={r['qid'] for r in sorted(a,key=lambda r:(-r['q'],str(r['qid'])))[:2048]};sb={r['qid'] for r in sorted(b,key=lambda r:(-r['q'],str(r['qid'])))[:2048]}
  comparison['global_top2048_overlap']=len(sa&sb);out[scope]=comparison
 return out

def bounded_cpu():
 if not hasattr(os,'sched_getaffinity'):raise ValueError('Linux CPU metadata execution required')
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(300,300))
 return dict(cpus=sorted(os.sched_getaffinity(0)),address_space_limit_bytes=3*1024**3,cpu_seconds_limit=300,torch_imported=False,model_or_feature_opened=False)
def context():
 groupsref=dict(path=str(SC/'groups-8192.json'),bytes=6265928,sha256=GROUP_SHA);groups=read(checked(groupsref))
 # Freeze solely from qids BEFORE accessing any labels/probabilities.
 plan=plan_ids([g['qid'] for g in groups]);immutable(OUT/'identity-strata-plan.json',plan)
 if len(groups)!=8192 or len({g['qid'] for g in groups})!=8192 or len({g['question_hash'] for g in groups})!=8192:raise ValueError('Exact unique8192 identities required')
 seen=set()
 for g in groups:
  tids=g['trajectory_ids']
  if len(tids)!=10 or len(set(tids))!=10 or seen.intersection(tids) or len(g['labels'])!=10 or any(type(y) is not int or y not in (0,1) for y in g['labels']):raise ValueError('Authentic complete ownG10 required')
  if json.loads(g['qid'])[0]!='nq':raise ValueError('This is explicitly NQ-only enrichment')
  seen.update(tids)
 cfgref=proof(SC/'config.json');cfg=read(cfgref['path'])
 if cfg['role']!=ROLE or cfg['evaluation_or_final_scored'] is not False or cfg['seen_training_data'] is not True or cfg['optimizer_constructed'] is not False or cfg['groups']!=groupsref:raise ValueError('Only declared seen-training calibration source')
 checked(cfg['training_input']);checked(cfg['source_training_complete']);checked(cfg['source_plan'])
 return groups,plan,groupsref,cfg,cfgref

def run_model(name,groups,plan,groupsref,cfg,cfgref,cpu):
 target=OUT/name;target.mkdir(exist_ok=True)
 if (target/'complete.json').exists():
  prior=read(target/'complete.json')
  if prior['diagnostic_runtime']!=proof(__file__):raise ValueError('Existing diagnosis used a different immutable source')
  return prior
 source=SC/name/'complete.json'
 if not source.exists():return None
 receipt=read(source);pred=receipt['predictions']
 if (receipt['status']!='ONE_FROZEN_TRAINING_G10_CALIBRATION_MODEL_COMPLETE' or receipt['config']!=cfgref or receipt['role']!=ROLE or receipt['seen_training_data'] is not True or receipt['evaluation_or_final_scored'] is not False or receipt['parameters_unchanged'] is not True or receipt['actor_updates']!=0 or receipt['probe_updates']!=0 or receipt['optimizer_constructed'] is not False):raise ValueError('Actual sealed frozen score receipt required')
 if receipt['checkpoint']['name']!=name or (name=='E4' and receipt['checkpoint']['sha256']!=E4_CP):raise ValueError('Wrong frozen model source')
 if name=='E4' and (pred['sha256']!=E4_SHA or pred['bytes']!=1751028):raise ValueError('Actual E4 predictions differ')
 if receipt['checkpoint']!=next(x for x in cfg['checkpoints'] if x['name']==name):raise ValueError('Wrong source checkpoint binding')
 path=checked(pred);old=[];new=[];curve=[];started=time.time()
 with gzip.open(str(path),'rt') as stream:
  for index,g in enumerate(groups,1):
   line=stream.readline()
   if not line:raise ValueError('Incomplete sealed prediction coverage')
   a,b=validate_row(g,json.loads(line));old.append(a);new.append(b)
   if index%1024==0:
    row=dict(epoch=time.time(),stage='ACTUAL_OWN_ID_EM_SIGMOID_PAIRED_ROWS_VERIFIED',model=name,questions=index,trajectories=index*10,elapsed_seconds=time.time()-started);atomic(OUT/'progress.json',row);curve.append(row)
    with (OUT/'progress-curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
  if stream.readline():raise ValueError('Unexpected extra score rows')
 paired_path=target/'paired-rows.jsonl.gz';tmp=paired_path.with_suffix('.tmp')
 with gzip.open(str(tmp),'wt') as stream:
  for a,b in zip(old,new):stream.write(json.dumps(dict(qid=a['qid'],old5=a,new5=b))+'\n')
 tmp.replace(paired_path)
 result=dict(status='ONE_FROZEN_MODEL_SEEN_NQ8192_OLD5_NEW5_PAIRED_CPU_DIAGNOSIS_COMPLETE',model=name,diagnostic_runtime=proof(__file__),source_receipt=proof(source),source_predictions=pred,groups=groupsref,identity_strata=proof(OUT/'identity-strata-plan.json'),source_checkpoint=receipt['checkpoint'],source_config=cfgref,source_training_input=cfg['training_input'],source_training_complete=cfg['source_training_complete'],cpu=cpu,own_ID_EM_sigmoid_verified=True,old5_was_in_original_warmup=True,new5_independent_seed=True,unseen_question_claim=False,seen_training_diagnostic_only=True,fitting_performed=False,threshold=.5,Platt_candidate_grid_changed=False,old5=dict(group=group_metrics(old),trajectory=trajectory_metrics(old),selection=selections(old,plan)),new5=dict(group=group_metrics(new),trajectory=trajectory_metrics(new),selection=selections(new,plan)),paired=pairing(old,new),paired_rows=proof(paired_path),elapsed_seconds=time.time()-started,epoch=time.time(),progress_checkpoints=curve)
 immutable(target/'complete.json',result);return result

def run(names):
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'diagnostic.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cpu=bounded_cpu();groups,plan,groupsref,cfg,cfgref=context()
  atomic(OUT/'controller.json',dict(pid=os.getpid(),epoch=time.time(),source=cfgref,source_groups=groupsref,diagnostic_runtime=proof(__file__),cpu=cpu,action='PAIRED_TRAINING_COHORT_CPU_ONLY'))
  atomic(OUT/'progress.json',dict(epoch=time.time(),stage='IDENTITY_ONLY_STRATA_FROZEN_BEFORE_LABEL_ANALYSIS',questions=8192,plan=proof(OUT/'identity-strata-plan.json')))
  results={}
  for name in names:
   result=run_model(name,groups,plan,groupsref,cfg,cfgref,cpu)
   if result is not None:results[name]=result
  complete=len(results)==2
  out=dict(status='BOTH_FROZEN_MODELS_PAIRED_TRAINING_COHORT_CPU_DIAGNOSIS_COMPLETE' if complete else 'PARTIAL_PAIRED_TRAINING_DIAGNOSIS_WAITING_FOR_OTHER_SEALED_MODEL',results=results,cross_model=cross_model(results,plan),identity_strata=proof(OUT/'identity-strata-plan.json'),groups=groupsref,seed=9137,questions=8192,domain='NQ-only',seen_original_warmup_questions=True,original5_seen_during_warmup=True,model_or_feature_payload_opened=False,external_validation_or_final_opened=False,fitting_performed=False,Platt_candidate_grid_changed=False,epoch=time.time())
  atomic(OUT/'latest.json',out)
  if complete:immutable(OUT/'complete.json',out)
  return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=('E4','peer256','both'),default='both');a=p.parse_args()
 if os.name!='posix' or not str(OUT).startswith(str(B)+'/'):raise ValueError('Remote Linux only')
 try:
  result=run(('E4','peer256') if a.model=='both' else (a.model,));print(json.dumps(dict(status=result['status'],models=list(result['results']),questions=8192,output=str(OUT))),flush=True)
 except BaseException as e:
  OUT.mkdir(parents=True,exist_ok=True);atomic(OUT/'failure.json',dict(type=type(e).__name__,message=str(e),epoch=time.time()));raise
if __name__=='__main__':main()
