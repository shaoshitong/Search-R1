"""After complete only: independent stdlib own-EM/ID/metric/SHA audit, remote data."""
import importlib.util
import json
import pathlib
import shlex
import subprocess

ROOT=pathlib.Path(__file__).resolve().parents[4]
CONTROL=ROOT.parents[1]
OUT=pathlib.Path(__file__).parent
REMOTE=r'''
import gzip,hashlib,json,math,pathlib,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021';X=A/'g10-exact-uniform252-v1'
def load(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(r):
 p=pathlib.Path(r['path']);assert sha(p)==r['sha256']
 if 'bytes' in r:assert p.stat().st_size==r['bytes']
 return p
def rows(p):
 with gzip.open(str(p),'rt') as f:return [json.loads(s) for s in f if s.strip()]
def auc(scores,labels):
 pos=sum(labels);neg=len(labels)-pos
 if not pos or not neg:return None
 ordered=sorted(zip(scores,labels));rs=0.;i=0
 while i<len(ordered):
  j=i+1
  while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
  rs+=((i+1+j)/2)*sum(y for _,y in ordered[i:j]);i=j
 return (rs-pos*(pos+1)/2)/(pos*neg)
def metrics(rs):
 ks=[sum(r['strict_em']) for r in rs];ys=[int(0<k<5) for k in ks]
 qs=[r['predicted_mixed'] for r in rs];ps=[q>=.5 for q in qs]
 tp=sum(p and y for p,y in zip(ps,ys));fn=sum((not p) and y for p,y in zip(ps,ys))
 fp=sum(p and not y for p,y in zip(ps,ys));tn=sum((not p) and not y for p,y in zip(ps,ys))
 tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
 def category(k):
  ids=[i for i,v in enumerate(ks) if v==k]
  return dict(questions=len(ids),called_mixed=sum(ps[i] for i in ids),
   false_positive_rate=sum(ps[i] for i in ids)/len(ids) if ids else None)
 return dict(questions=len(rs),mixed_questions=sum(ys),mixed_fraction=sum(ys)/len(ys),
  confusion=dict(tp=tp,fn=fn,fp=fp,tn=tn),mixed_recall=tpr,nonmixed_recall=tnr,
  mixed_false_negative_rate=1-tpr if tpr is not None else None,
  nonmixed_false_positive_rate=1-tnr if tnr is not None else None,
  mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,
  mixed_roc_auc=auc(qs,ys),
  k_histogram=[ks.count(k) for k in range(6)],allwrong=category(0),allcorrect=category(5))
def summaries(rs):
 top=sorted(rs,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512]
 return dict(pool=metrics(rs),selected512=metrics(top))
def own_ids(rs,truth):
 assert len(rs)==len(truth) and len({r['qid'] for r in rs})==len(rs)
 seen=set()
 for r in rs:
  assert r['qid'] in truth and len(r['trajectory_ids'])==5 and len(set(r['trajectory_ids']))==5
  pairs=dict(zip(r['trajectory_ids'],r['strict_em']))
  assert len(r['strict_em'])==5 and pairs==truth[r['qid']]
  assert not seen.intersection(pairs);seen.update(pairs)
  assert all(type(y) is int and y in (0,1) for y in r['strict_em'])
  assert len(r['p_j'])==5 and all(math.isfinite(p) and 0<=p<=1 for p in r['p_j'])
  assert math.isfinite(r['predicted_mixed']) and 0<=r['predicted_mixed']<=1
def metric_match(got,want):
 for k in ('questions','mixed_questions','confusion','k_histogram'):assert got[k]==want[k],k
 for k in ('mixed_fraction','mixed_recall','nonmixed_recall','mixed_balanced_accuracy','mixed_roc_auc'):
  assert (got[k] is None and want[k] is None) or (got[k] is not None and want[k] is not None and abs(got[k]-want[k])<1e-12),(k,got[k],want[k])
complete_path=X/'complete.json';assert complete_path.exists(),'Actual final complete required'
complete=load(complete_path);c=load(X/'config.json');latest=load(X/'latest.json')
assert sha(X/'config.json')=='66866b17f374cee5384582806000038f0bbd1cbe2d3d4830f74478c05e1d812f'
assert complete['status']=='REAL_G10_QUESTION_WEIGHTED_PROBE_BUDGET_AND_VALIDATION_COMPLETE'
assert complete['cursor']==136 and complete['steps']==10600 and complete['checkpoint']==latest
assert not complete['optimizer_reset'] and complete['actor_updates']==0
assert latest['cursor']==136 and latest['steps']==10600 and latest['config_sha256']==sha(X/'config.json')
checkpoint=checked(latest)
assert c['global_questions']==48 and c['update_budget']==136 and c['fit_questions']==6528
assert c['plan']==c['inherited_reference_plan'] and c['source_checkpoint']['sha256']=='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'
assert c['recipe']['loss_kwargs']==dict(trajectory_weight=1.,three_event_weight=1.)
assert sha(checked(c['plan']))=='3bff8f5331bd60634456b1f24a106a752f84b196bc0a0cd8961c5d17405e86d9'
replay=load(X/'next-update-replay.json');assert replay['status']=='PASSED' and replay['expected']==replay['actual']
assert not (X/'failure.json').exists() and not (X/'interrupted.json').exists()
curve=[json.loads(s) for s in (X/'curve.jsonl').read_text().splitlines() if s]
assert [r['cursor'] for r in curve]==list(range(1,137))
assert all(r['steps']==10464+r['cursor'] and r['actor_updates']==0 and
 r['grouping']['combinations_per_question']==252 and r['grouping']['per_question_total_weight_exactly_one'] and
 r['recomputed_logits_bit_identical'] and r['inter_view_pairs_allocated']==0 for r in curve)
evaluations={};truth2048=None
for e in c['evaluations']:
 bank=load(checked(e['bank']))[e['split']]
 fixed=load(checked(e['fixed_g5']));fixed=fixed.get('fixed_G5',fixed)
 assert fixed['summary']['labels_consulted'] is False and fixed['summary']['role']=='validation'
 by={g['qid']:dict(zip(g['trajectory_ids'],g['labels'])) for g in bank}
 assert len(by)==len(bank)==len(fixed['groups'])
 truth={v['qid']:{tid:by[v['qid']][tid] for tid in v['trajectory_ids']} for v in fixed['groups']}
 assert len(truth)==len(by) and all(len(v)==5 for v in truth.values())
 p=checked(complete['metrics'][e['name']]['predictions']);rs=rows(p);own_ids(rs,truth)
 assert latest['epoch_time']<=p.stat().st_mtime<=complete_path.stat().st_mtime
 got=summaries(rs)
 for field in ('pool','selected512'):metric_match(got[field],complete['metrics'][e['name']][field])
 evaluations[e['name']]=dict(predictions=ref(p),bank=e['bank'],fixed_G5=e['fixed_g5'],
  own_EM_and_five_real_fixed_IDs_exact=True,checkpoint_accepted_before_prediction=True,metrics=got)
 if len(rs)==2048:truth2048=truth;main=got
assert truth2048 is not None
baseline_defs={
 'native_E4':dict(path=str(B/'frozen-validation-ema3-20261021/evaluation-output/E4/predictions.jsonl.gz'),bytes=155083,sha256='2c76bdb5d0d34ddc2bf8966f8cde639b84d772dcbc68ef7c42f094a6b4498d5d'),
 'same_G10_random4_B48x136':dict(path=str(A/'g10-matched6528-v1/random/predictions/selection2048FreshG5.jsonl.gz'),bytes=149705,sha256='7973c42ea94ddb877fc4840da7e8130d6630a8bab6b8f1d708e759f0e79c3983')}
baselines={}
for name,r in baseline_defs.items():
 rs=rows(checked(r));own_ids(rs,truth2048);m=summaries(rs)
 baselines[name]=dict(predictions=r,same_2048_qids_five_own_IDs_and_EM=True,metrics=m,
  delta_BA=main['pool']['mixed_balanced_accuracy']-m['pool']['mixed_balanced_accuracy'],
  delta_AUC=main['pool']['mixed_roc_auc']-m['pool']['mixed_roc_auc'],
  delta_Top512_Mixed=main['selected512']['mixed_questions']-m['selected512']['mixed_questions'])
memory=load(X/'gpu-memory.json');assert memory['exit_normally'] is True
result=dict(status='ACTUAL_ALL252_136_UPDATES_NATIVE_FIXED_G5_INDEPENDENT_FULL_ACCEPTANCE',
 time=time.time(),complete=ref(complete_path),config=ref(X/'config.json'),checkpoint=latest,
 updates=136,Adam_steps=10600,source_Adam_steps=10464,source_questions=6528,source_real_trajectories=65280,
 fixed_threshold=.5,threshold_fitted=False,actor_updates=0,optimizer_reset=False,
 native_backend_unchanged=True,all_252_uniform_views_total_question_weight_one=True,
 first_GPU_update_double_replay_exact=True,final_checkpoint_fresh_full_SHA=True,
 final_CPU_load_and_six_component_validation_precedes_checkpoint_receipt=True,
 source_and_config_SHA_bound=True,no_final_confirmation_or_stability_pool_read=True,
 evaluations=evaluations,baselines=baselines,
 actual_update_seconds=dict(min=min(r['seconds'] for r in curve),max=max(r['seconds'] for r in curve),
  mean=sum(r['seconds'] for r in curve)/len(curve)),GPU_memory=memory,
 production_promotion_claim=False)
dest=X/'independent-final-audit.json';assert not dest.exists(),'Preserve an existing independent acceptance'
dest.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');result['audit_ref']=ref(dest)
print(json.dumps(result,sort_keys=True))
'''
if __name__=='__main__':
 spec=importlib.util.spec_from_file_location('all252_final_acceptance_private',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
 ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 r=ctl.run('python3 -c '+shlex.quote(REMOTE),account='mou176',source_address=ip,timeout=90)
 if r.returncode:raise SystemExit(r.stderr[-3000:])
 result=json.loads(r.stdout.splitlines()[-1])
 (OUT/'actual-final-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,sort_keys=True))
