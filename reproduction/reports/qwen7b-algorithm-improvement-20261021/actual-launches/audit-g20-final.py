"""Remote read-only CPU audit of completed G20 plus exact native E4 baseline.
No tensor loads, retraining, threshold fitting, label-aware selection or final pools.
"""
import collections, gzip, hashlib, json, math, pathlib, time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
BASE=A/'g20-matched6528-v1'
E4=A.parent/'frozen-validation-ema3-20261021/evaluation-output/E4/predictions.jsonl.gz'
E4SHA='2c76bdb5d0d34ddc2bf8966f8cde639b84d772dcbc68ef7c42f094a6b4498d5d'
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def checked(ref):
 p=pathlib.Path(ref['path']);assert sha(p)==ref['sha256'],str(p)
 if 'bytes' in ref:assert p.stat().st_size==ref['bytes']
 return p
def product(xs):
 r=1.
 for x in xs:r*=x
 return r
def auc(rows):
 scores=[(r['predicted_mixed'],int(0<sum(r['strict_em'])<5)) for r in rows]
 pos=sum(y for _,y in scores);neg=len(scores)-pos
 if not pos or not neg:return None
 scores.sort();i=0;rs=0.
 while i<len(scores):
  j=i+1
  while j<len(scores) and scores[j][0]==scores[i][0]:j+=1
  rs+=((i+1+j)/2)*sum(y for _,y in scores[i:j]);i=j
 return (rs-pos*(pos+1)/2)/(pos*neg)
def metrics(rows):
 ks=[sum(r['strict_em']) for r in rows];target=[0<k<5 for k in ks];pred=[r['predicted_mixed']>=.5 for r in rows]
 tp=sum(y and p for y,p in zip(target,pred));fn=sum(y and not p for y,p in zip(target,pred))
 fp=sum(not y and p for y,p in zip(target,pred));tn=sum(not y and not p for y,p in zip(target,pred))
 tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
 classes={}
 for name,mask in [('allwrong',[k==0 for k in ks]),('mixed',target),('allcorrect',[k==5 for k in ks])]:
  ix=[i for i,m in enumerate(mask) if m];n=sum(pred[i] for i in ix);f=n/len(ix) if ix else None
  classes[name]=dict(questions=len(ix),predicted_mixed=n,predicted_mixed_fraction=f,false_positive_rate=f if name!='mixed' else None,correct_group_recall=((1-f) if name!='mixed' else f) if ix else None)
 return dict(questions=len(rows),mixed_questions=sum(target),mixed_fraction=sum(target)/len(rows),confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),mixed_recall=tpr,nonmixed_recall=tnr,mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,mixed_roc_auc=auc(rows),k_histogram=[ks.count(k) for k in range(6)],classwise=classes)
def fixed_own(e):
 bank=read(checked(e['bank']))[e['split']];by={g['qid']:g for g in bank};assert len(by)==len(bank)
 plan=read(checked(e['fixed_g5']));plan=plan.get('fixed_G5',plan)
 assert plan['summary']['role']=='validation' and plan['summary']['labels_consulted'] is False
 assert digest(plan['groups'])==plan['summary']['selected_g5_sha256']
 own=[];seen=set()
 for v in plan['groups']:
  g=by[v['qid']];tids=v['trajectory_ids'];labels=dict(zip(g['trajectory_ids'],g['labels']))
  assert len(tids)==len(set(tids))==5 and not seen&set(tids);seen.update(tids)
  assert len(labels)==len(g['labels']) and all(type(y) is int and y in (0,1) for y in labels.values())
  own.append(dict(qid=v['qid'],trajectory_ids=tids,strict_em=[labels[t] for t in tids]))
 assert len(own)==len(by)
 return own,plan['summary']['selected_g5_sha256']
def prediction_audit(ref,own,declared=None):
 path=checked(ref)
 with gzip.open(str(path),'rt') as f:rows=[json.loads(s) for s in f]
 assert len(rows)==len(own) and len({r['qid'] for r in rows})==len(rows)
 maxerror=0.
 for row,source in zip(rows,own):
  for k in ('qid','trajectory_ids','strict_em'):assert row[k]==source[k],k
  ps=row['p_j'];q=row['predicted_mixed']
  assert len(ps)==5 and all(math.isfinite(p) and 0<=p<=1 for p in ps) and math.isfinite(q) and 0<=q<=1
  want=1-product(ps)-product([1-p for p in ps]);maxerror=max(maxerror,abs(want-q))
 assert maxerror<1e-6,maxerror
 selected=sorted(rows,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512]
 pool,top=metrics(rows),metrics(selected)
 if declared:
  for split,actual in [('pool',pool),('selected512',top)]:
   for key in ('questions','mixed_questions','mixed_fraction','confusion','mixed_recall','nonmixed_recall','mixed_balanced_accuracy','mixed_roc_auc','k_histogram'):
    value=actual[key];want=declared[split][key]
    assert abs(value-want)<1e-12 if isinstance(value,float) else value==want,(split,key,value,want)
   for c in ('allwrong','mixed','allcorrect'):
    for k in ('questions','predicted_mixed','predicted_mixed_fraction','false_positive_rate','correct_group_recall'):
     value=actual['classwise'][c][k];want=declared[split]['classwise'][c][k]
     assert abs(value-want)<1e-12 if isinstance(value,float) else value==want,(split,c,k,value,want)
 return dict(pool=pool,selected512=top,q_formula_max_absolute_error=maxerror,predictions=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)),own_EM_and_frozen_ID_order_exact=True,selected_qid_sha256=digest([r['qid'] for r in selected]),ordered_own_trajectory_ids_sha256=digest([r['trajectory_ids'] for r in rows]))
results={};evaluation_own=None;fixedsha=None
for variant in ('random','global-weighted-balanced'):
 d=BASE/variant;done=read(d/'complete.json');c=read(d/'config.json')
 assert done['status']=='REAL_G20_QUESTION_WEIGHTED_PROBE_BUDGET_AND_VALIDATION_COMPLETE' and done['cursor']==136 and done['steps']==10600
 assert done['optimizer_reset'] is False and done['actor_updates']==0 and done['config_sha256']==sha(d/'config.json')
 assert c['source_checkpoint']['sha256']==done['source_checkpoint']['sha256'] and read(d/'next-update-replay.json')['status']=='PASSED'
 cp=done['checkpoint'];assert cp==read(d/'latest.json') and cp['cursor']==136 and cp['steps']==10600
 assert pathlib.Path(cp['path']).stat().st_size==cp['bytes']
 # Full checkpoint bytes were freshly SHA/load/component checked by producer.
 # Do not reload four GB tensors merely to reproduce those same checks.
 result=dict(complete=dict(path=str(d/'complete.json'),sha256=sha(d/'complete.json')),checkpoint=cp,first_update_replay='PASSED',evaluations={},checkpoint_bulk_load_and_component_acceptance_inherited_from_producer=True)
 for e in c['evaluations']:
  own,freeze=fixed_own(e);result['evaluations'][e['name']]=prediction_audit(done['metrics'][e['name']]['predictions'],own,done['metrics'][e['name']])
  if e['name']=='selection2048FreshG5':
   assert len(own)==2048
   if evaluation_own is not None:assert evaluation_own==own and fixedsha==freeze
   evaluation_own=own;fixedsha=freeze
 memory=read(d/'gpu-memory.json');result['gpu_memory']=memory
 results[variant]=result
assert fixedsha=='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7'
e4=prediction_audit(dict(path=str(E4),sha256=E4SHA,bytes=155083),evaluation_own)
assert e4['pool']['mixed_balanced_accuracy']==.5552907416785098 and e4['selected512']['mixed_questions']==200
for r in results.values():
 actual=r['evaluations']['selection2048FreshG5'];assert actual['ordered_own_trajectory_ids_sha256']==e4['ordered_own_trajectory_ids_sha256']=='acf5e010707351cdc442e844a7dc6d1a03bdb8d60fcbafe2b7907f749ddfac3c'
 ba=actual['pool']['mixed_balanced_accuracy']-e4['pool']['mixed_balanced_accuracy'];top=actual['selected512']['mixed_questions']-e4['selected512']['mixed_questions']
 r['versus_same_frozen_G5_native_E4']=dict(BA_delta=ba,AUC_delta=actual['pool']['mixed_roc_auc']-e4['pool']['mixed_roc_auc'],Top512_Mixed_count_delta=top,both_BA_and_Top512_strictly_improved=(ba>0 and top>0),same_input_backend_and_exact_G5=True,same_training_budget_as_historical_E4_claim=False)
print(json.dumps(dict(status='G20_BOTH_FINAL_BUDGETS_OWN_EM_Q_BA_AUC_TOP512_NATIVE_E4_INDEPENDENT_AUDIT_PASS',epoch=time.time(),results=results,native_E4=e4,freeze_sha256=fixedsha,torch_imported=False,model_or_optimizer_loaded=False,packed_tensor_payload_opened=False,final_pools_read=False,random_balanced_same6528_B48_136=True,G10_equal_trajectory_compute_claim=False)))
