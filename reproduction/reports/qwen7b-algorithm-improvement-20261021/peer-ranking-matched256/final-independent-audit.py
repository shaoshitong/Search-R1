"""Remote stdlib-only final audit: no feature/model/optimizer tensor loads."""
import copy,gzip,hashlib,json,math,pathlib,subprocess,sys,time
RECIPE=sys.argv[1]
assert RECIPE in ('inter1','bothrank1')
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
b=A/('peer8192-'+RECIPE+'-matched256-v1');old=A/'peer-newseed8192-g10-train-v1'
def read(p):return json.loads(pathlib.Path(p).read_text())
def proof(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(r):
 p=pathlib.Path(r['path']);a=proof(p)
 assert a['sha256']==r['sha256'] and ('bytes' not in r or a['bytes']==r['bytes'])
 return p
def rows(p):
 with gzip.open(str(p),'rt') as f:return [json.loads(s) for s in f if s.strip()]
def auc(scores,labels):
 pos=sum(labels);neg=len(labels)-pos
 if not pos or not neg:return None
 ordered=sorted(zip(scores,labels));rank=0.;i=0
 while i<len(ordered):
  j=i+1
  while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
  rank+=(i+1+j)/2*sum(y for _,y in ordered[i:j]);i=j
 return (rank-pos*(pos+1)/2)/(pos*neg)
def metrics(rs):
 ks=[sum(r['strict_em']) for r in rs];truth=[0<k<5 for k in ks];q=[r['predicted_mixed'] for r in rs];called=[v>=.5 for v in q]
 tp=sum(y and p for y,p in zip(truth,called));fn=sum(y and not p for y,p in zip(truth,called));fp=sum(not y and p for y,p in zip(truth,called));tn=sum(not y and not p for y,p in zip(truth,called))
 tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
 def fpr(k):
  ix=[i for i,v in enumerate(ks) if v==k]
  return sum(called[i] for i in ix)/len(ix) if ix else None
 return dict(questions=len(rs),mixed_questions=sum(truth),mixed_fraction=sum(truth)/len(rs),confusion=dict(tp=tp,fn=fn,fp=fp,tn=tn),mixed_recall=tpr,nonmixed_recall=tnr,mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,mixed_roc_auc=auc(q,truth),allwrong_FPR=fpr(0),allcorrect_FPR=fpr(5),k_histogram=[ks.count(k) for k in range(6)])
def summarize(rs):return dict(pool=metrics(rs),selected512=metrics(sorted(rs,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512]))
def own(rs,truth):
 assert len(rs)==len(truth) and len({r['qid'] for r in rs})==len(rs)
 seen=set();error=0.
 for r in rs:
  assert r['qid'] in truth and len(r['trajectory_ids'])==len(r['strict_em'])==5 and len(set(r['trajectory_ids']))==5
  assert dict(zip(r['trajectory_ids'],r['strict_em']))==truth[r['qid']]
  assert not set(r['trajectory_ids'])&seen;seen.update(r['trajectory_ids'])
  assert all(type(v) is int and v in (0,1) for v in r['strict_em'])
  assert len(r['p_j'])==5 and all(math.isfinite(v) and 0<=v<=1 for v in r['p_j'])
  q=r['predicted_mixed'];assert math.isfinite(q) and 0<=q<=1
  positive=negative=1.
  for p in r['p_j']:positive*=p;negative*=1-p
  error=max(error,abs(q-(1-positive-negative)))
 assert error<1e-6
 return error
def metric_match(actual,stored):
 for k in ('questions','mixed_questions','confusion','k_histogram'):assert actual[k]==stored[k],k
 for k in ('mixed_fraction','mixed_recall','nonmixed_recall','mixed_balanced_accuracy','mixed_roc_auc'):
  assert actual[k]==stored[k] or (actual[k] is not None and stored[k] is not None and abs(actual[k]-stored[k])<1e-12),k
 assert actual['allwrong_FPR']==stored['classwise']['allwrong']['false_positive_rate']
 assert actual['allcorrect_FPR']==stored['classwise']['allcorrect']['false_positive_rate']
if not (b/'complete.json').exists():raise RuntimeError('Actual256 and evaluations complete required before final audit')
c=read(b/'config.json');control=read(old/'config.json');done=read(b/'complete.json');original=read(old/'complete.json');latest=read(b/'latest.json');pub=read(b/'source-publication.json');csha=proof(b/'config.json')['sha256']
assert proof(old/'config.json')['sha256']=='16b2fd34e3e8907307b3866298b33216478fd77c2dc1b9ca80638e0929800130'
assert c['source_commit']=='3c6faad3b821015b36b6c92ba51308898ac1b381' and pub['commit']==c['source_commit']
expected=dict(trajectory_weight=1.,three_event_weight=1.,inter_rank_weight=1.)
if RECIPE=='bothrank1':expected['within_rank_weight']=1.
assert c['recipe']['loss_kwargs']==expected
ca=copy.deepcopy(c);cb=copy.deepcopy(control)
for v in (ca,cb):
 for k in ('source_commit','code_hashes','within_pair_inputs'):v.pop(k,None)
ca['recipe']=cb['recipe'];assert ca==cb
assert c['plan']['sha256']=='0b2212b46e2046d097f7c1a90490406ba67babf81002f15b129a61efe4dba012'
assert c['source_checkpoint']['sha256']=='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724' and c['source_adam_steps']==10464 and not c['optimizer_reset'] and c['actor_updates']==0
assert done['status']=='REAL_G10_QUESTION_WEIGHTED_PROBE_BUDGET_AND_VALIDATION_COMPLETE' and done['cursor']==256 and done['steps']==10720 and done['config_sha256']==csha and done['checkpoint']==latest
assert latest['cursor']==256 and latest['steps']==10720 and original['cursor']==256 and original['steps']==10720
assert original['config_sha256']==proof(old/'config.json')['sha256']
assert not (b/'failure.json').exists() and not (b/'first-cp-audit-failure.json').exists()
first=read(b/'first-cp-independent-audit.json');replay=read(b/'next-update-replay.json')
assert first['recipe']==RECIPE and first['steps']==10465 and first['config_sha256']==csha and first['status']=='FIRST_CP1_INDEPENDENT_FULL_SHA_AND_CPU_ALL_COMPONENT_RELOAD_PASSED'
assert first['component_hashes']==replay['actual']==replay['expected'] and replay['status']=='PASSED' and first['source_all_four_RNG_preserved']
curve=[json.loads(s) for s in (b/'curve.jsonl').read_text().splitlines() if s];reference=[json.loads(s) for s in (old/'curve.jsonl').read_text().splitlines() if s]
assert len(curve)==len(reference)==256;mask=dict(actual_feature_pairs=0,kept=0,excluded_same_causal_feature_drift=0);points={}
for i,(r,oldr) in enumerate(zip(curve,reference),1):
 assert r['cursor']==i and r['steps']==10464+i and r['original_questions']==32 and r['unique_real_trajectories']==320 and r['virtual_views']==128 and r['actual_forward_trajectory_exposures']==640
 assert r['grouping']==oldr['grouping'] and r['recomputed_logits_bit_identical'] and r['denominators']['inter_ordered_original_question_pairs']==992
 names=['trajectory','three_event','inter_rank']+(['within_rank'] if RECIPE=='bothrank1' else [])
 assert abs(r['loss']-sum(r['components'][k] for k in names))<1e-5*(1+abs(r['loss']))
 if RECIPE=='bothrank1':
  assert r['within_causal_signature_rule'] and r['within_canonical_and_feature_pairs']<=r['within_actual_feature_pairs'] and r['within_same_causal_excluded_feature_pairs']==r['within_actual_feature_pairs']-r['within_canonical_and_feature_pairs']
  mask['actual_feature_pairs']+=r['within_actual_feature_pairs'];mask['kept']+=r['within_canonical_and_feature_pairs'];mask['excluded_same_causal_feature_drift']+=r['within_same_causal_excluded_feature_pairs']
 else:assert r['within_mask_constructed'] is False and r['canonical_signature_read'] is False
 if i in (1,32,64,128,192,256):points[str(i)]=dict(loss=r['loss'],components=r['components'],seconds=r['seconds'])
final_checkpoint=proof(latest['path']);assert all(final_checkpoint[k]==latest[k] for k in ('path','bytes','sha256'))
evaluations={};truth2048=None
for e in c['evaluations']:
 bank=read(checked(e['bank']))[e['split']];fixed=read(checked(e['fixed_g5']));fixed=fixed.get('fixed_G5',fixed)
 assert fixed['summary']['role']=='validation' and fixed['summary']['labels_consulted'] is False
 by={g['qid']:dict(zip(g['trajectory_ids'],g['labels'])) for g in bank};truth={g['qid']:{t:by[g['qid']][t] for t in g['trajectory_ids']} for g in fixed['groups']}
 actual_rows=rows(checked(done['metrics'][e['name']]['predictions']));err=own(actual_rows,truth);m=summarize(actual_rows)
 for k in ('pool','selected512'):metric_match(m[k],done['metrics'][e['name']][k])
 assert latest['epoch_time']<=pathlib.Path(done['metrics'][e['name']]['predictions']['path']).stat().st_mtime
 prior_rows=rows(checked(original['metrics'][e['name']]['predictions']));own(prior_rows,truth);prior=summarize(prior_rows)
 evaluations[e['name']]=dict(metrics=m,control_metrics=prior,predictions=proof(done['metrics'][e['name']]['predictions']['path']),own_EM_fixed_IDs_exact=True,q_formula_max_error=err,delta_BA=m['pool']['mixed_balanced_accuracy']-prior['pool']['mixed_balanced_accuracy'],delta_AUC=m['pool']['mixed_roc_auc']-prior['pool']['mixed_roc_auc'],delta_Top512=m['selected512']['mixed_questions']-prior['selected512']['mixed_questions'])
 if len(actual_rows)==2048:truth2048=truth
assert truth2048 is not None
e4ref=dict(path=str(old/'E4-same-HF-validation/E4/predictions.jsonl.gz'),bytes=155299,sha256='66cdf3e7c748bda7f508c4f381ad0456a413d5d73c73e5d9f1eb774b71bdb707')
e4rows=rows(checked(e4ref));own(e4rows,truth2048);e4=summarize(e4rows)
jid=(b/'job.id').read_text().strip();r=subprocess.run(['sacct','-j',jid,'-n','-P','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25);assert not r.returncode
state=next(s.split('|') for s in r.stdout.splitlines() if s.split('|')[0]==jid);assert state[1:3]==['COMPLETED','0:0']
result=dict(status='MATCHED_RANK256_CP_OWN_EM_FIXED_HF_METRICS_INDEPENDENTLY_ACCEPTED',recipe=RECIPE,job_id=jid,sacct=r.stdout.strip(),config=proof(b/'config.json'),complete=proof(b/'complete.json'),final_checkpoint=latest,first_CPU_audit=proof(b/'first-cp-independent-audit.json'),same_actual_grouping_all256=True,within_canonical_mask=mask,learning_curve=points,evaluations=evaluations,E4_same_HF_reference=dict(predictions=e4ref,metrics=e4),fixed_threshold=.5,threshold_fitted=False,source_Adam_steps=10464,final_Adam_steps=10720,optimizer_reset=False,actor_updates=0,no_model_optimizer_or_feature_tensor_loaded_on_login=True,final_confirmation_or_stability_pool_read=False,production_promotion_claim=False,time=time.time())
destination=b/'final-independent-audit.json';assert not destination.exists(),'Preserve an existing accepted audit';destination.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');result['audit_ref']=proof(destination);print(json.dumps(result))
