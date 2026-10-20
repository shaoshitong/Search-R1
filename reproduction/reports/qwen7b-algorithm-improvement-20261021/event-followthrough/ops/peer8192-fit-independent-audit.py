import pathlib,json,hashlib,gzip,math,functools,operator,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed8192-g10-train-v1');read=lambda p:json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
 return h.hexdigest()
def checked(ref):
 p=pathlib.Path(ref['path']);assert p.is_file() and sha(p)==ref['sha256'];assert 'bytes' not in ref or p.stat().st_size==ref['bytes'];return p
c=read(b/'config.json');csha=sha(b/'config.json');pre=read(b/'preparation.json');assert pre['config']['sha256']==csha;assert c['source_checkpoint']['sha256']=='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724';assert c['source_adam_steps']==10464 and c['global_questions']==32 and c['update_budget']==256 and c['fit_questions']==8192 and c['fit_real_trajectories']==81920 and c['optimizer_reset'] is False
r=read(b/'next-update-replay.json');assert r['status']=='PASSED' and r['expected']==r['actual'] and set(r['actual'])=={'raw','ema','adam','raw_buffers','ema_buffers','rng'};assert r['source_steps']==10464 and r['candidate_steps']==10465
latest=read(b/'latest.json');checked(latest);assert latest['config_sha256']==csha and latest['steps']==10464+latest['cursor'];assert set(latest['accepted_hashes'])=={'raw','ema','adam','raw_buffers','ema_buffers','rng'}
rows=[]
with (b/'curve.jsonl').open() as stream:
 for line in stream:
  if not line.endswith('\n'):break
  rows.append(json.loads(line))
for i,row in enumerate(rows,1):
 assert row['cursor']==i and row['steps']==10464+i and row['original_questions']==32 and row['unique_real_trajectories']==320 and row['virtual_views']==128 and row['actual_forward_trajectory_exposures']==640 and row['recomputed_logits_bit_identical'] is True and row['actor_updates']==0
assert len(rows)>=latest['cursor'];result={'status':'ACTUAL_PEER8192_FIRST_REPLAY_CONTINUOUS_UPDATES_AND_CHECKPOINT_SHA_VERIFIED','config_sha256':csha,'latest':latest,'actual_curve_updates':len(rows),'first_actual_next_update_replay':r['status'],'first_steps':rows[0]['steps'],'last_steps':rows[-1]['steps'],'last_update_seconds':rows[-1]['seconds'],'last_loss':rows[-1]['loss'],'last_components':rows[-1]['components'],'actor_updates':0,'optimizer_reset':False,'checkpoint_tensors_loaded_on_login':False,'feature_payload_loaded_on_login':False,'final_pool_read':False,'observed_epoch':time.time()}
if (b/'complete.json').exists():
 done=read(b/'complete.json');assert done['status']=='REAL_G10_QUESTION_WEIGHTED_PROBE_BUDGET_AND_VALIDATION_COMPLETE' and done['cursor']==256 and done['steps']==10720 and done['config_sha256']==csha and done['checkpoint']==latest and len(rows)==256
 evaluations={}
 for e in c['evaluations']:
  groups=read(checked(e['bank']))[e['split']];plan=read(checked(e['fixed_g5']));plan=plan.get('fixed_G5',plan);assert plan['summary']['role']=='validation' and plan['summary']['labels_consulted'] is False;lookup={g['qid']:dict(zip(g['trajectory_ids'],g['labels'])) for g in groups};metric=done['metrics'][e['name']];pred=[]
  with gzip.open(checked(metric['predictions']),'rt') as stream:
   for line in stream:pred.append(json.loads(line))
  assert len(pred)==len(plan['groups']) and [x['qid'] for x in pred]==[x['qid'] for x in plan['groups']]
  error=0.
  for row,own in zip(pred,plan['groups']):
   assert row['trajectory_ids']==own['trajectory_ids'] and row['strict_em']==[lookup[row['qid']][t] for t in row['trajectory_ids']];p=row['p_j'];assert len(p)==5 and all(math.isfinite(x) and 0<=x<=1 for x in p) and math.isfinite(row['predicted_mixed']) and 0<=row['predicted_mixed']<=1;q=1-functools.reduce(operator.mul,p,1.)-functools.reduce(operator.mul,(1-x for x in p),1.);error=max(error,abs(q-row['predicted_mixed']))
  assert error<1e-6;truth=[0<sum(x['strict_em'])<5 for x in pred];called=[x['predicted_mixed']>=.5 for x in pred];tp=sum(a and z for a,z in zip(truth,called));fn=sum(a and not z for a,z in zip(truth,called));fp=sum(not a and z for a,z in zip(truth,called));tn=sum(not a and not z for a,z in zip(truth,called));BA=.5*(tp/(tp+fn)+tn/(tn+fp));chosen=sorted(pred,key=lambda x:(-x['predicted_mixed'],str(x['qid'])))[:512];n=sum(0<sum(x['strict_em'])<5 for x in chosen);assert metric['pool']['mixed_balanced_accuracy']==BA and metric['pool']['confusion']==dict(tp=tp,fn=fn,fp=fp,tn=tn) and metric['selected512']['mixed_questions']==n
  ordered=sorted((x['predicted_mixed'],y) for x,y in zip(pred,truth));rank=0.;i=0
  while i<len(ordered):
   j=i+1
   while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
   rank+=((i+1+j)/2)*sum(y for _,y in ordered[i:j]);i=j
  pos=sum(truth);neg=len(truth)-pos;auc=(rank-pos*(pos+1)/2)/(pos*neg);assert auc==metric['pool']['mixed_roc_auc'];fprs={}
  for label,k in [('allwrong',0),('allcorrect',5)]:
   own=[x for x in pred if sum(x['strict_em'])==k];value=sum(x['predicted_mixed']>=.5 for x in own)/len(own);assert metric['pool']['classwise'][label]['false_positive_rate']==value;fprs[label]=value
  evaluations[e['name']]={'questions':len(pred),'mixed':pos,'BA':BA,'Mixed_recall':tp/(tp+fn),'nonMixed_recall':tn/(tn+fp),'AUC':auc,'FPRs':fprs,'top512_mixed':n,'top512_fraction':n/512,'q_formula_max_error':error,'predictions_sha256':metric['predictions']['sha256']}
 result.update(status='PEER8192_ALL256_UPDATES_FINAL_CP_OWN_EM_BA_AUC_FPRS_AND_TOP512_INDEPENDENTLY_VERIFIED',evaluations=evaluations)
raw=json.dumps(result,sort_keys=True);(b/'independent-audit-latest.json').write_text(raw);print(raw)
