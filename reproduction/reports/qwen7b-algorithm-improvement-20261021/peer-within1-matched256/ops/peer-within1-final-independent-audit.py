"""Read-only matched end-to-end audit; no tensors/features loaded on login."""
from pathlib import Path
import json,gzip,hashlib,math,functools,operator,time,subprocess,copy
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');b=A/'peer8192-within1-matched256-v1';old=A/'peer-newseed8192-g10-train-v1';read=lambda p:json.loads(Path(p).read_text())
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(r):
 p=Path(r['path']);a=proof(p);assert a['sha256']==r['sha256'] and ('bytes' not in r or a['bytes']==r['bytes']);return p
if not (b/'complete.json').exists():print(json.dumps(dict(status='WAITING_ACTUAL_WITHIN1_FULL256_AND_EVALUATION_COMPLETE')));raise SystemExit(0)
c=read(b/'config.json');control_c=read(old/'config.json');done=read(b/'complete.json');original=read(old/'complete.json');pub=read(b/'source-publication.json');csha=proof(b/'config.json')['sha256'];latest=read(b/'latest.json')
a=copy.deepcopy(c);base=copy.deepcopy(control_c)
for key in ('source_commit','code_hashes','within_pair_inputs'):a.pop(key,None);base.pop(key,None)
assert a['recipe']['loss_kwargs'].pop('within_rank_weight')==1. and a==base and c['plan']['sha256']=='0b2212b46e2046d097f7c1a90490406ba67babf81002f15b129a61efe4dba012'
assert done['status']=='REAL_G10_QUESTION_WEIGHTED_PROBE_BUDGET_AND_VALIDATION_COMPLETE' and done['cursor']==256 and done['steps']==10720 and done['config_sha256']==csha and done['checkpoint']==latest and latest['cursor']==256 and latest['steps']==10720
assert c['source_checkpoint']['sha256']=='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724' and c['source_adam_steps']==10464 and c['optimizer_reset'] is False and c['actor_updates']==0
root=Path(pub['code_root'])
for name,item in pub['files'].items():assert proof(root/name)['sha256']==item['sha256'] and c['code_hashes'][name]==item['sha256']
first=read(b/'first-cp-independent-audit.json');replay=read(b/'next-update-replay.json');assert first['status']=='FIRST_CP1_INDEPENDENT_FULL_SHA_AND_CPU_ALL_COMPONENT_RELOAD_PASSED' and first['component_hashes']==replay['expected']==replay['actual'] and replay['status']=='PASSED' and first['steps']==10465
signatures=read(checked(c['within_pair_inputs']));assert signatures['trajectories']==81920 and len(read(checked(signatures['signatures'])))==81920
with (b/'curve.jsonl').open() as f:curve=[json.loads(line) for line in f if line.endswith('\n')]
with (old/'curve.jsonl').open() as f:oldcurve=[json.loads(line) for line in f if line.endswith('\n')]
assert len(curve)==len(oldcurve)==256;excluded=0;actual_pairs=0;kept_pairs=0;point={}
for i,(row,reference) in enumerate(zip(curve,oldcurve),1):
 assert row['cursor']==i and row['steps']==10464+i and row['original_questions']==32 and row['unique_real_trajectories']==320 and row['virtual_views']==128 and row['actual_forward_trajectory_exposures']==640 and row['recomputed_logits_bit_identical'] and row['grouping']==reference['grouping'] and row['within_causal_signature_rule']
 co=row['components'];assert abs(row['loss']-sum(co[k] for k in ('trajectory','three_event','within_rank')))<1e-5*(1+abs(row['loss']))
 assert row['within_canonical_and_feature_pairs']<=row['within_actual_feature_pairs'] and row['within_same_causal_excluded_feature_pairs']==row['within_actual_feature_pairs']-row['within_canonical_and_feature_pairs']
 excluded+=row['within_same_causal_excluded_feature_pairs'];actual_pairs+=row['within_actual_feature_pairs'];kept_pairs+=row['within_canonical_and_feature_pairs']
 if i in (1,32,64,128,192,256):point[str(i)]=dict(components=co,loss=row['loss'],seconds=row['seconds'])
# One mandatory final file SHA; CPU component reload already ran inside GPU writer.
actual_checkpoint=proof(latest['path']);assert all(actual_checkpoint[k]==latest[k] for k in ('path','bytes','sha256'))
evaluations={}
for e in c['evaluations']:
 bank=read(checked(e['bank']))[e['split']];plan=read(checked(e['fixed_g5']));plan=plan.get('fixed_G5',plan);assert plan['summary']['role']=='validation' and plan['summary']['labels_consulted'] is False;lookup={g['qid']:dict(zip(g['trajectory_ids'],g['labels'])) for g in bank};m=done['metrics'][e['name']]
 with gzip.open(checked(m['predictions']),'rt') as f:rows=[json.loads(line) for line in f]
 assert len(rows)==len(plan['groups']);error=0.
 for r,selection in zip(rows,plan['groups']):
  assert r['qid']==selection['qid'] and r['trajectory_ids']==selection['trajectory_ids'] and r['strict_em']==[lookup[r['qid']][t] for t in r['trajectory_ids']];p=r['p_j'];assert len(p)==5 and all(math.isfinite(v) and 0<=v<=1 for v in p) and math.isfinite(r['predicted_mixed']);q=1-functools.reduce(operator.mul,p,1.)-functools.reduce(operator.mul,[1-v for v in p],1.);error=max(error,abs(q-r['predicted_mixed']))
 assert error<1e-6;truth=[0<sum(r['strict_em'])<5 for r in rows];called=[r['predicted_mixed']>=.5 for r in rows];tp=sum(y and p for y,p in zip(truth,called));fn=sum(y and not p for y,p in zip(truth,called));fp=sum(not y and p for y,p in zip(truth,called));tn=sum(not y and not p for y,p in zip(truth,called));BA=.5*(tp/(tp+fn)+tn/(tn+fp));top=sorted(rows,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512];mixed=sum(0<sum(r['strict_em'])<5 for r in top)
 ordered=sorted((r['predicted_mixed'],y) for r,y in zip(rows,truth));rank=0.;j=0
 while j<len(ordered):
  k=j+1
  while k<len(ordered) and ordered[k][0]==ordered[j][0]:k+=1
  rank+=(j+1+k)/2*sum(y for _,y in ordered[j:k]);j=k
 pos=sum(truth);auc=(rank-pos*(pos+1)/2)/(pos*(len(rows)-pos));fprs={}
 assert m['pool']['confusion']==dict(tp=tp,fn=fn,fp=fp,tn=tn) and m['pool']['mixed_balanced_accuracy']==BA and m['pool']['mixed_roc_auc']==auc and m['selected512']['mixed_questions']==mixed
 for name,k in [('allwrong',0),('allcorrect',5)]:
  own=[r for r in rows if sum(r['strict_em'])==k];fprs[name]=sum(r['predicted_mixed']>=.5 for r in own)/len(own);assert fprs[name]==m['pool']['classwise'][name]['false_positive_rate']
 ref=original['metrics'][e['name']];evaluations[e['name']]=dict(questions=len(rows),trueMixed=pos,BA=BA,TPR=tp/(tp+fn),TNR=tn/(tn+fp),AUC=auc,FPRs=fprs,Top512=mixed,Top512_fraction=mixed/512,predictions=m['predictions'],q_formula_max_error=error,control_BA=ref['pool']['mixed_balanced_accuracy'],control_Top512=ref['selected512']['mixed_questions'],delta_BA=BA-ref['pool']['mixed_balanced_accuracy'],delta_Top512=mixed-ref['selected512']['mixed_questions'])
jid=(b/'job.id').read_text().strip();r=subprocess.run(['sacct','-j',jid,'-n','-P','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25);assert r.returncode==0;state=next(x.split('|') for x in r.stdout.splitlines() if x.split('|')[0]==jid)
result=dict(status='MATCHED_WITHIN1_ALL256_CP_AND_FIXED_OWN_EM_PRIMARY_METRICS_INDEPENDENTLY_VERIFIED',job_id=jid,sacct=r.stdout.strip(),terminal_normal_exit=state[1:3]==['COMPLETED','0:0'],config=proof(b/'config.json'),complete=proof(b/'complete.json'),final_checkpoint=latest,first_CP1_independent_audit=proof(b/'first-cp-independent-audit.json'),same_actual_four_G5_view_grouping_all256=True,canonical_mask=dict(actual_feature_pairs=actual_pairs,kept_canonical_and_feature_pairs=kept_pairs,excluded_same_causal_feature_drift_pairs=excluded),learningcurve=point,evaluations=evaluations,actor_updates=0,no_model_or_optimizer_tensor_loaded_on_login=True,final_pool_read=False,larger_goal_solved=False,time=time.time());(b/'final-independent-audit.json').write_text(json.dumps(result,sort_keys=True));print(json.dumps(result))
