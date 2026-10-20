from pathlib import Path
import json,gzip,hashlib,math,functools,operator,time,subprocess
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');b=B/'algorithm-improvement-20261021/E2-same-HF-baseline-20261021';read=lambda p:json.loads(Path(p).read_text())
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(r):
 p=Path(r['path']);actual=proof(p);assert actual['sha256']==r['sha256'] and ('bytes' not in r or actual['bytes']==r['bytes']);return p
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert (b/'complete.json').exists(),'E2 actual completion required';c=read(b/'config.json');done=read(b/'complete.json');assert done['status']=='SEALED_E2_SAME_QUALIFIED_NEW_HF_FIXED_G5_BASELINE_COMPLETE' and done['config']==proof(b/'config.json') and done['actor_updates']==done['probe_updates']==0 and done['parameters_unchanged'] and done['optimizer_constructed'] is False and done['calibration_grid_member'] is False and done['final_pool_read'] is False
assert done['checkpoint']['sha256']=='6a970815ff3aec26eedf4d6a65589bbebac3c4ccbdf8eefdf859a83e2a11bd2d' and done['checkpoint']['steps']==11782 and done['checkpoint']['epoch']==2
old=read(B/'frozen-validation-ema3-20261021/evaluation-output/E2/complete.json');assert old['checkpoint']['sha256']==done['checkpoint']['sha256'] and old['ema_parameter_sha256']==done['ema_sha256']
v=read(checked(c['validation_complete']));assert v['fixed_G5_sha256']==c['fixed_G5_sha256']==done['fixed_G5_sha256']=='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7';g10=read(checked(v['source_groups']))['validation'];plan=read(checked(v['source_fixed_G5']));plan=plan.get('fixed_G5',plan);assert plan['summary']['labels_consulted'] is False and digest(plan['groups'])==c['fixed_G5_sha256'];by={g['qid']:dict(zip(g['trajectory_ids'],g['labels'])) for g in g10};groups=[dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'],labels=[by[g['qid']][t] for t in g['trajectory_ids']]) for g in plan['groups']];assert groups==read(checked(c['groups'])) and len(groups)==2048
with gzip.open(checked(done['predictions']),'rt') as f:rows=[json.loads(x) for x in f]
assert len(rows)==2048 and len({g['qid'] for g in rows})==2048;err=0.
for g,r in zip(groups,rows):
 assert r['qid']==g['qid'] and r['trajectory_ids']==g['trajectory_ids'] and r['strict_em']==g['labels'];p=r['p_j'];assert len(p)==5 and all(math.isfinite(x) and 0<=x<=1 for x in p);q=1-functools.reduce(operator.mul,p,1)-functools.reduce(operator.mul,[1-x for x in p],1);err=max(err,abs(q-r['predicted_mixed']))
assert err<1e-6;truth=[0<sum(r['strict_em'])<5 for r in rows];called=[r['predicted_mixed']>=.5 for r in rows];tp=sum(y and p for y,p in zip(truth,called));fn=sum(y and not p for y,p in zip(truth,called));fp=sum(not y and p for y,p in zip(truth,called));tn=sum(not y and not p for y,p in zip(truth,called));BA=.5*(tp/(tp+fn)+tn/(tn+fp));top=sorted(rows,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512];mixed=sum(0<sum(r['strict_em'])<5 for r in top)
ordered=sorted((r['predicted_mixed'],y) for r,y in zip(rows,truth));rank=0.;i=0
while i<len(ordered):
 j=i+1
 while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
 rank+=(i+1+j)/2*sum(y for _,y in ordered[i:j]);i=j
pos=sum(truth);auc=(rank-pos*(pos+1)/2)/(pos*(len(rows)-pos));m=done['metrics'];assert m['pool']['confusion']==dict(tp=tp,fn=fn,fp=fp,tn=tn) and m['pool']['mixed_balanced_accuracy']==BA and m['pool']['mixed_roc_auc']==auc and m['selected512']['mixed_questions']==mixed
fprs={}
for tag,k in [('allwrong',0),('allcorrect',5)]:
 own=[r for r in rows if sum(r['strict_em'])==k];fprs[tag]=sum(r['predicted_mixed']>=.5 for r in own)/len(own);assert fprs[tag]==m['pool']['classwise'][tag]['false_positive_rate']
T=b.parent/'peer-newseed8192-g10-train-v1';baseline=read(T/'E4-same-HF-validation/complete.json');peer=read(T/'complete.json');comparisons={name:r['metrics'] for name,r in baseline['checkpoints'].items()};comparisons.update(peer256=peer['metrics']['selection2048FreshG5HF'],E2=m);table={}
for name,r in comparisons.items():
 if 'ordered_qids_sha256' in r:assert r['ordered_qids_sha256']==digest([g['qid'] for g in rows]) and r['ordered_own_trajectory_ids_sha256']==digest([g['trajectory_ids'] for g in rows])
 table[name]=dict(BA=r['pool']['mixed_balanced_accuracy'],AUC=r['pool']['mixed_roc_auc'],Mixed_recall=r['pool']['mixed_recall'],nonMixed_recall=r['pool']['nonmixed_recall'],allwrong_FPR=r['pool']['classwise']['allwrong']['false_positive_rate'],allcorrect_FPR=r['pool']['classwise']['allcorrect']['false_positive_rate'],Top512_mixed=r['selected512']['mixed_questions'],Top512_fraction=r['selected512']['mixed_fraction'])
jid=(b/'job.id').read_text().strip();p=subprocess.run(['sacct','-j',jid,'-n','-P','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25);assert p.returncode==0;state=next(x.split('|') for x in p.stdout.splitlines() if x.split('|')[0]==jid)
r=dict(status='SEALED_E2_SAME_HF_FIXED_G5_OWN_EM_BA_AUC_FPRS_AND_TOP512_INDEPENDENTLY_VERIFIED',job_id=jid,sacct=p.stdout.strip(),terminal_normal_exit=state[1:3]==['COMPLETED','0:0'],config=proof(b/'config.json'),complete=proof(b/'complete.json'),predictions=done['predictions'],EMA_same_as_original_native_E2=True,ema_sha256=done['ema_sha256'],questions=2048,trajectories=10240,Mixed_questions=pos,q_formula_max_error=err,confusion=dict(tp=tp,fn=fn,fp=fp,tn=tn),fixed_G5_sha256=c['fixed_G5_sha256'],comparison_same_HF_frozen2048=table,calibration_grid_member=False,actor_updates=0,probe_updates=0,final_pool_read=False,time=time.time());(b/'independent-audit.json').write_text(json.dumps(r,sort_keys=True));print(json.dumps(r))
