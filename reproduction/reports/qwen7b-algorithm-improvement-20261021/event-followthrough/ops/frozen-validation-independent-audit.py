"""Remote read-only independent ownEM/BA/AUC/Top512 check; no torch/imported eval."""
import json,pathlib,gzip,hashlib,math,time,subprocess
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/frozen-validation-ema3-20261021')
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(1024*1024),b''):h.update(raw)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def checked(ref):
 p=pathlib.Path(ref['path']);assert sha(p)==ref['sha256']
 if 'bytes' in ref:assert p.stat().st_size==ref['bytes']
 return p
def product(v):
 p=1.
 for x in v:p*=x
 return p
def auc(scores,ys):
 pos=sum(ys);neg=len(ys)-pos
 if not pos or not neg:return None
 ordered=sorted(zip(scores,ys));total=0.;i=0
 while i<len(ordered):
  j=i+1
  while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
  total+=((i+1+j)/2)*sum(y for _,y in ordered[i:j]);i=j
 return (total-pos*(pos+1)/2)/(pos*neg)
def metrics(rows):
 ks=[sum(r['strict_em']) for r in rows];y=[0<k<5 for k in ks];s=[r['predicted_mixed'] for r in rows];p=[q>=.5 for q in s]
 tp=sum(a and b for a,b in zip(p,y));fn=sum(not a and b for a,b in zip(p,y))
 fp=sum(a and not b for a,b in zip(p,y));tn=sum(not a and not b for a,b in zip(p,y))
 tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
 def cfpr(k):
  mask=[i for i,t in enumerate(ks) if t==k]
  return sum(p[i] for i in mask)/len(mask) if mask else None
 return dict(questions=len(rows),mixed_questions=sum(y),mixed_fraction=sum(y)/len(rows),
  confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),mixed_recall=tpr,nonmixed_recall=tnr,
  mixed_false_negative_rate=1-tpr if tpr is not None else None,
  nonmixed_false_positive_rate=1-tnr if tnr is not None else None,
  mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,
  mixed_roc_auc=auc(s,y),k_histogram=[ks.count(k) for k in range(6)],
  allwrong_false_positive_rate=cfpr(0),allcorrect_false_positive_rate=cfpr(5),mean_q=sum(s)/len(s))
def equal(actual,expected):
 if actual is None:assert expected is None;return
 if isinstance(actual,float):assert abs(actual-expected)<1e-12,(actual,expected);return
 assert actual==expected,(actual,expected)
c=read(B/'inputs/config.json');v=read(checked(c['validation_complete']));ci=v['consumer_inputs']
plan=read(checked(ci['fixed_G5_plan']));assert plan['summary']['labels_consulted'] is False
assert plan['summary']['role']=='validation' and digest(plan['groups'])==plan['summary']['selected_g5_sha256']
groups=read(checked(ci['groups']))['validation'];by={g['qid']:g for g in groups};assert len(by)==len(groups)==2048
fixed=[]
for own in plan['groups']:
 source=by[own['qid']];labels=dict(zip(source['trajectory_ids'],source['labels']))
 assert len(own['trajectory_ids'])==len(set(own['trajectory_ids']))==5
 fixed.append(dict(qid=own['qid'],trajectory_ids=own['trajectory_ids'],strict_em=[labels[t] for t in own['trajectory_ids']]))
results={};ordered_ids=None
aggregate_path=B/'evaluation-output/complete.json'
aggregate=read(aggregate_path) if aggregate_path.exists() else None
config_by_name={d['name']:d for d in c['checkpoints']}
for name in ('E2','E4','event128'):
 cp=B/'evaluation-output'/name/'complete.json'
 if not cp.exists():continue
 accepted=read(cp);path=checked(accepted['predictions'])
 assert accepted['checkpoint']==config_by_name[name] and accepted['config_sha256']==sha(B/'inputs/config.json')
 assert accepted['frozen_parameters_unchanged'] is True and accepted['optimizer_constructed'] is False
 assert accepted['actor_optimizer_updates']==accepted['probe_optimizer_updates']==0
 if aggregate is not None:assert aggregate['checkpoints'][name]==accepted
 with gzip.open(str(path),'rt') as f:rows=[json.loads(line) for line in f]
 assert len(rows)==2048
 max_error=0.
 for row,own in zip(rows,fixed):
  for key in ('qid','trajectory_ids','strict_em'):assert row[key]==own[key],key
  assert len(row['p_j'])==5 and all(math.isfinite(p) and 0<=p<=1 for p in row['p_j'])
  q=1-product(row['p_j'])-product([1-p for p in row['p_j']]);max_error=max(max_error,abs(q-row['predicted_mixed']))
 assert max_error<1e-6,max_error
 selected=sorted(rows,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512]
 pool,chosen=metrics(rows),metrics(selected)
 for split,m in [('pool',pool),('selected512',chosen)]:
  for key,value in m.items():equal(value,accepted['metrics'][split][key])
 identity=digest([r['trajectory_ids'] for r in rows])
 if ordered_ids is not None:assert identity==ordered_ids
 ordered_ids=identity
 assert digest([r['qid'] for r in selected])==accepted['metrics']['selected_qids_sha256']
 results[name]=dict(pool=pool,selected512=chosen,selected_qids_sha256=digest([r['qid'] for r in selected]),
  ownEM_from_accepted_G10_and_precommitted_G5_match=True,q_formula_max_absolute_error=max_error,
  predictions=accepted['predictions'],checkpoint_steps=accepted['checkpoint']['steps'],
  checkpoint_sha256=accepted['checkpoint']['sha256'],ema_parameter_sha256=accepted['ema_parameter_sha256'],
  actual_metrics_match_independent_recompute=True)
if aggregate is not None:
 assert aggregate['status']=='FIXED_VALIDATION_G5_E2_E4_EVENT128_EMA_EVALUATION_COMPLETE'
 assert aggregate['same_question_and_trajectory_ids'] is True and aggregate['fitting_performed'] is False
 assert aggregate['questions']==2048 and aggregate['trajectories']==10240
 assert aggregate['binding']['config']['sha256']==sha(B/'inputs/config.json')
report=dict(status='INDEPENDENT_FROZEN_G5_OWN_EM_BA_AUC_TOP512_PASS' if len(results)==3 else 'PARTIAL_INDEPENDENT_FROZEN_G5_OWN_EM_BA_AUC_TOP512_PASS',
 observed_epoch=time.time(),completed_checkpoints=len(results),expected_checkpoints=3,results=results,
 questions=2048,trajectories=10240,baseline_K5=[sum(sum(g['strict_em'])==k for g in fixed) for k in range(6)],
 fixed_G5_selection_sha256=plan['summary']['selected_g5_sha256'],same_ordered_own_trajectory_ids_sha256=ordered_ids,
 aggregate_complete_SHA=sha(aggregate_path) if aggregate is not None else None,torch_imported=False,model_or_optimizer_loaded=False,raw_predictions_returned=False,untouched_final_read=False)
print(json.dumps(report))
