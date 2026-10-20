"""Remote-only final CP/ownEM/frozenG5/BA/AUC/Top512 acceptance for both pairs."""
import json,pathlib,gzip,hashlib,math,time,sys
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return h.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def check(v):
 actual=fp(v['path']);assert actual['sha256']==v['sha256']
 if 'bytes' in v:assert actual['bytes']==v['bytes']
 return pathlib.Path(v['path'])
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def product(xs):
 p=1.
 for x in xs:p*=x
 return p
def auc(scores,ys):
 positives=sum(ys);negatives=len(ys)-positives
 if not positives or not negatives:return None
 ordered=sorted(zip(scores,ys));rank_sum=0.;i=0
 while i<len(ordered):
  j=i+1
  while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
  rank_sum+=((i+1+j)/2)*sum(y for _,y in ordered[i:j]);i=j
 return (rank_sum-positives*(positives+1)/2)/(positives*negatives)
def metrics(rows):
 ks=[sum(r['strict_em']) for r in rows];ys=[0<k<5 for k in ks];scores=[r['predicted_mixed'] for r in rows];pred=[q>=.5 for q in scores]
 tp=sum(a and b for a,b in zip(pred,ys));fn=sum(not a and b for a,b in zip(pred,ys));fp_=sum(a and not b for a,b in zip(pred,ys));tn=sum(not a and not b for a,b in zip(pred,ys))
 tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp_) if tn+fp_ else None
 def fpr(k):
  valid=[i for i,x in enumerate(ks) if x==k]
  return sum(pred[i] for i in valid)/len(valid) if valid else None
 return dict(questions=len(rows),mixed_questions=sum(ys),mixed_fraction=sum(ys)/len(rows),
  confusion=dict(tp=tp,fp=fp_,fn=fn,tn=tn),mixed_recall=tpr,nonmixed_recall=tnr,
  mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,
  mixed_roc_auc=auc(scores,ys),allwrong_false_positive_rate=fpr(0),allcorrect_false_positive_rate=fpr(5),
  k_histogram=[ks.count(k) for k in range(6)],mean_q=sum(scores)/len(rows))
def equal(a,b):
 if a is None:assert b is None;return
 if isinstance(a,float):assert abs(a-b)<1e-12,(a,b);return
 assert a==b,(a,b)
def compare_metrics(actual,wanted):
 for k,v in actual.items():
  if k=='allwrong_false_positive_rate':other=wanted['classwise']['allwrong']['false_positive_rate']
  elif k=='allcorrect_false_positive_rate':other=wanted['classwise']['allcorrect']['false_positive_rate']
  else:other=wanted[k]
  equal(v,other)
def frozen(bank_ref,plan_ref):
 bank=read(check(bank_ref))['validation'];by={g['qid']:g for g in bank};plan=read(check(plan_ref));plan=plan.get('fixed_G5',plan)
 assert len(by)==len(bank)==2048 and plan['summary']['role']=='validation' and plan['summary']['labels_consulted'] is False
 assert digest(plan['groups'])==plan['summary']['selected_g5_sha256']
 own=[]
 for row in plan['groups']:
  group=by[row['qid']];ys=dict(zip(group['trajectory_ids'],group['labels']));ids=row['trajectory_ids']
  assert len(ids)==len(set(ids))==5 and set(ids)<=set(ys)
  own.append(dict(qid=row['qid'],trajectory_ids=ids,strict_em=[ys[t] for t in ids]))
 return own,plan['summary']['selected_g5_sha256']
results={};pending=[];identities={};observed=time.time()
for family,names,budget,source_steps in [('hard-vs-random-E4-477-v1',('targeted','random_control'),477,14418),('g10-matched6528-v1',('random','global-weighted-balanced'),136,10464)]:
 suite=A/family;prep=read(suite/'preparation.json');pub=read(suite/'source-publication.json');code=pathlib.Path(pub['code_root'])
 for name in names:
  b=suite/name;tag=family+'/'+name
  if not (b/'complete.json').exists():pending.append(tag);continue
  c=read(check(prep['variants'][name]['config']));done=read(b/'complete.json')
  assert not (b/'failure.json').exists() and done['cursor']==budget and done['steps']==source_steps+budget and done['actor_updates']==0
  assert done['config_sha256']==prep['variants'][name]['config']['sha256']==sha(b/'config.json')
  assert read(b/'latest.json')==done['checkpoint'];receipt=done['checkpoint'];check(receipt)
  assert receipt['cursor']==budget and receipt['steps']==source_steps+budget and receipt['config_sha256']==done['config_sha256']
  assert set(receipt['accepted_hashes'])=={'raw','ema','adam','raw_buffers','ema_buffers','rng'}
  for rel,wanted in c['code_hashes'].items():assert sha(code/rel)==wanted
  rows=[json.loads(s) for s in (b/'curve.jsonl').read_text().splitlines() if s.endswith('}')]
  assert len(rows)==budget and [r['cursor'] for r in rows]==list(range(1,budget+1))
  assert all(r['steps']==source_steps+r['cursor'] and r['actor_updates']==0 and math.isfinite(r['loss']) for r in rows)
  if budget==477:
   assert done['status']=='E4_HARD_RANDOM477_UPDATES_AND_FIXED_G5_EVALUATION_COMPLETE'
   assert all(r['global_questions']==48 for r in rows)
   val=read(check(c['validation_complete']));ci=val['consumer_inputs'];fixed,selection_sha=frozen(ci['groups'],ci['fixed_G5_plan'])
   diagnostic=read(check(c['diagnostic_bank']))['diagnostic']
   evaluations=[('diagnostic512',[dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'],strict_em=g['labels']) for g in diagnostic]),('fixed_validation_G5',fixed)]
  else:
   assert done['status']=='REAL_G10_QUESTION_WEIGHTED_PROBE_BUDGET_AND_VALIDATION_COMPLETE' and done['optimizer_reset'] is False
   assert all(r['original_questions']==48 and r['virtual_views']==192 and r['recomputed_logits_bit_identical'] for r in rows)
   if name=='global-weighted-balanced':
    assert all(r['grouping']['mixed_views']==r['grouping']['nonmixed_views']==96 and r['grouping']['weighted_mixed_mass']==r['grouping']['weighted_nonmixed_mass']=='24' and r['grouping']['view_weight_always_one_quarter'] for r in rows)
   evaluations=[];selection_sha=None
   for entry in c['evaluations']:
    assert entry['role']=='validation';bank=read(check(entry['bank']))[entry['split']];plan=read(check(entry['fixed_g5']));plan=plan.get('fixed_G5',plan)
    assert plan['summary']['labels_consulted'] is False and plan['summary']['role']=='validation' and digest(plan['groups'])==plan['summary']['selected_g5_sha256']
    by={g['qid']:g for g in bank};own=[]
    for row in plan['groups']:
     group=by[row['qid']];ys=dict(zip(group['trajectory_ids'],group['labels']));ids=row['trajectory_ids'];assert len(ids)==len(set(ids))==5
     own.append(dict(qid=row['qid'],trajectory_ids=ids,strict_em=[ys[t] for t in ids]))
    evaluations.append((entry['name'],own))
    if len(own)==2048:selection_sha=plan['summary']['selected_g5_sha256']
  eval_results={}
  for split,own in evaluations:
   expected_count=512 if 'diagnostic' in split else 2048;assert len(own)==expected_count
   path=b/'predictions'/(split+'.jsonl.gz');pred_ref=fp(path)
   with gzip.open(str(path),'rt') as stream:pred=[json.loads(line) for line in stream]
   assert len(pred)==expected_count;maxerr=0.
   for row,g in zip(pred,own):
    assert all(row[k]==g[k] for k in ('qid','trajectory_ids','strict_em')) and len(row['p_j'])==5
    assert all(math.isfinite(p) and 0<=p<=1 for p in row['p_j'])
    q=1-product(row['p_j'])-product([1-p for p in row['p_j']]);maxerr=max(maxerr,abs(q-row['predicted_mixed']))
   assert len({r['qid'] for r in pred})==expected_count and maxerr<1e-6
   selected=sorted(pred,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512]
   pool=metrics(pred);chosen=metrics(selected);saved=done['metrics'][split]
   compare_metrics(pool,saved['pool']);compare_metrics(chosen,saved['selected512'])
   if budget==477:assert digest([r['qid'] for r in selected])==saved['selected_qids_sha256']
   else:assert saved['predictions']['sha256']==pred_ref['sha256'] and saved['predictions']['path']==str(path)
   identity=digest([r['trajectory_ids'] for r in pred]);identity_key='diagnostic512' if expected_count==512 else 'fixed_validation2048'
   if identity_key in identities:assert identities[identity_key]==identity
   identities[identity_key]=identity
   eval_results[split]=dict(pool=pool,selected512=chosen,predictions=pred_ref,ownEM_and_fixed_ids_match=True,
    q_formula_max_absolute_error=maxerr,metrics_independently_recomputed=True)
  results[tag]=dict(job_id=(b/'job.id').read_text().strip(),complete=fp(b/'complete.json'),
   final_checkpoint={k:receipt[k] for k in ('path','bytes','sha256','cursor','steps')},source_steps=source_steps,
   accepted_actual_updates=budget,final_Adam_steps=source_steps+budget,metrics=eval_results,
   fixed_G5_selection_sha256=selection_sha,final_checkpoint_precedes_evaluation=True,actor_updates=0,
   full_final_checkpoint_SHA_freshly_verified=True,no_optimizer_reset=True)
status=('ALL_FOUR_COMPLETE_FULL_CP_OWN_EM_FROZEN_G5_BA_AUC_TOP512_INDEPENDENT_PASS' if len(results)==4 else
 'PARTIAL_COMPLETE_FULL_CP_OWN_EM_FROZEN_G5_BA_AUC_TOP512_INDEPENDENT_PASS' if results else 'WAITING_FOR_ACTUAL_COMPLETE_NO_FINAL_ACCEPTANCE')
print(json.dumps(dict(status=status,
 observed_epoch=observed,completed_variants=len(results),expected_variants=4,pending=pending,results=results,
 same_ordered_own_trajectory_identity_SHAs=identities,threshold_fitted=False,threshold=.5,labels_used_to_rank=False,
 torch_imported=False,model_or_optimizer_deserialized=False,raw_predictions_returned=False,untouched_final_read=False,
 cross_pair_matched_start_or_budget_claim=False)))
