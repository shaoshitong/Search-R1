"""Remote CPU-only final file/own-label/score/stratum and real-progress audit."""
import collections,fcntl,gzip,hashlib,json,math,os,pathlib,resource,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-paired-old-new-20261021')
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
def read(p):return json.loads(pathlib.Path(p).read_text())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(ref):
 actual=proof(ref['path']);assert all(actual[k]==v for k,v in ref.items() if k in actual);return pathlib.Path(ref['path'])
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def product(xs):
 p=1.
 for x in xs:p*=x
 return p
def auc(rows):
 pairs=sorted((r['q'],0<r['k']<5) for r in rows);pos=sum(y for q,y in pairs);neg=len(pairs)-pos;i=0;rs=0.
 if not pos or not neg:return None
 while i<len(pairs):
  j=i+1
  while j<len(pairs) and pairs[j][0]==pairs[i][0]:j+=1
  rs+=((i+1+j)/2)*sum(y for q,y in pairs[i:j]);i=j
 return (rs-pos*(pos+1)/2)/(pos*neg)
def counts(rows):
 ys=[0<r['k']<5 for r in rows];pred=[r['q']>=.5 for r in rows]
 tp=sum(y and p for y,p in zip(ys,pred));tn=sum(not y and not p for y,p in zip(ys,pred));pos=sum(ys);neg=len(rows)-pos
 return dict(questions=len(rows),mixed_questions=pos,mixed_fraction=pos/len(rows),mixed_balanced_accuracy=(tp/pos+tn/neg)/2 if pos and neg else None,mixed_roc_auc=auc(rows),confusion=dict(tp=tp,fn=pos-tp,tn=tn,fp=neg-tn))
def same(actual,accepted):
 for k,v in actual.items():
  expected=accepted[k]
  assert abs(v-expected)<1e-12 if isinstance(v,float) else v==expected,(k,v,expected)
started=time.time();done=read(B/'complete.json');assert done['status']=='BOTH_FROZEN_MODELS_PAIRED_TRAINING_COHORT_CPU_DIAGNOSIS_COMPLETE'
groups=read(checked(done['groups']));plan=read(checked(done['identity_strata']));ids=[g['qid'] for g in groups];ordered=sorted(ids,key=lambda q:(digest(['old5-new5-id-strata-v1',9137,q]),str(q)))
assert len(set(ids))==8192 and plan['labels_consulted'] is False and plan['probabilities_consulted'] is False
assert [q for s in plan['groups'] for q in s['qids']]==ordered and all(len(s['qids'])==2048 for s in plan['groups'])
curves=[json.loads(s) for s in (B/'progress-curve.jsonl').read_text().splitlines()];progress={};models={}
for name,item in done['results'].items():
 checked(item['source_predictions']);receipt=read(checked(item['source_receipt']));assert receipt['predictions']==item['source_predictions']
 with gzip.open(str(checked(item['paired_rows'])),'rt') as f:pairs=[json.loads(s) for s in f]
 assert len(pairs)==8192;parts={scope:[] for scope in ('old5','new5')};maxq=maxsig=0.
 for g,row in zip(groups,pairs):
  assert row['qid']==g['qid']
  for scope,start in (('old5',0),('new5',5)):
   r=row[scope];assert r['qid']==g['qid'] and r['trajectory_ids']==g['trajectory_ids'][start:start+5] and r['y']==g['labels'][start:start+5] and r['k']==sum(r['y'])
   for p,z in zip(r['p'],r['z']):
    value=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z));maxsig=max(maxsig,abs(value-p))
   maxq=max(maxq,abs(r['q']-(1-product(r['p'])-product([1-p for p in r['p']]))));parts[scope].append(r)
 assert maxsig<2e-7 and maxq<1e-12
 for scope,rows in parts.items():
  same(counts(rows),item[scope]['group']);by={r['qid']:r for r in rows}
  for part,result in zip(plan['groups'],item[scope]['selection']['strata']):
   own=[by[q] for q in part['qids']];chosen=sorted(own,key=lambda r:(-r['q'],str(r['qid'])))[:512]
   same(counts(own),result['pool']);same(counts(chosen),result['selected512']);assert digest([r['qid'] for r in chosen])==result['selected_qids_sha256']
  selected=sorted(rows,key=lambda r:(-r['q'],str(r['qid'])))[:2048];same(counts(selected),item[scope]['selection']['global_top2048'])
 points=[r for r in curves if r['model']==name];assert [r['questions'] for r in points]==list(range(1024,8193,1024));assert all(a['epoch']<=b['epoch'] for a,b in zip(points,points[1:]))
 progress[name]=dict(first_actual=points[0],last_actual=points[-1],model_complete_epoch=item['epoch'],count=len(points))
 models[name]=dict(source_predictions=item['source_predictions'],source_receipt=item['source_receipt'],paired_rows=item['paired_rows'],max_sigmoid_error=maxsig,max_q_formula_error=maxq,own_ID_EM_order_exact=True,all_group_metrics_and_fixed_strata_top_rankings_match=True)
lock=(B/'diagnostic.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
controller=read(B/'controller.json');pid=controller['pid'];live=pathlib.Path('/proc/%d'%pid).exists();fail=(B/'failure.json').exists();assert not fail
small=dict(status='INDEPENDENT_PAIRED_SOURCE_OWN_EM_Q_METRICS_AND_TWO_REAL_PROGRESS_STAGES_PASS',epoch=time.time(),elapsed_seconds=time.time()-started,source_complete=proof(B/'complete.json'),models=models,actual_progress=progress,controller_pid=pid,controller_process_present=live,diagnostic_lock_free=True,failure_absent=True,identity_only_strata_seed=9137,all_question_ids_disjoint_across_four2048=True,full_probability_rows_returned_to_Mac=False,model_or_feature_opened=False,external_validation_or_final_opened=False,Platt_grid_changed=False)
p=B/'independent-final-audit.json';p.write_text(json.dumps(small,sort_keys=True)+'\n')
print(json.dumps(dict(summary=done,independent_final_audit=small)))
