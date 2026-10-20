"""Remote CPU, read-only acceptance: CP512 SHA and ownEM/group metrics, no torch."""
import pathlib,json,gzip,hashlib,math,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/multilayer-four-epoch-v1')
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
 return h.hexdigest()
def checked(ref):
 p=pathlib.Path(ref['path']);assert sha(p)==ref['sha256'],str(p)
 if 'bytes' in ref:assert p.stat().st_size==ref['bytes']
 return p
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def product(values):
 p=1.
 for value in values:p*=value
 return p
def auc(scores,y):
 pos=sum(y);neg=len(y)-pos
 if not pos or not neg:return None
 ordered=sorted(zip(scores,y));ranks=0.;i=0
 while i<len(ordered):
  j=i+1
  while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
  ranks+=((i+1+j)/2)*sum(label for score,label in ordered[i:j]);i=j
 return (ranks-pos*(pos+1)/2)/(pos*neg)
def metrics(rows):
 ks=[sum(r['strict_em']) for r in rows];y=[0<k<5 for k in ks];scores=[r['predicted_mixed'] for r in rows];pred=[q>=.5 for q in scores]
 tp=sum(p and t for p,t in zip(pred,y));fn=sum(not p and t for p,t in zip(pred,y));fp=sum(p and not t for p,t in zip(pred,y));tn=sum(not p and not t for p,t in zip(pred,y))
 tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
 classes={}
 for name,values in [('allwrong',[k==0 for k in ks]),('mixed',y),('allcorrect',[k==5 for k in ks])]:
  ids=[i for i,valid in enumerate(values) if valid];called=sum(pred[i] for i in ids)
  classes[name]=dict(questions=len(ids),predicted_mixed=called,predicted_mixed_fraction=called/len(ids) if ids else None,
   mean_q=sum(scores[i] for i in ids)/len(ids) if ids else None,
   false_positive_rate=called/len(ids) if ids and name!='mixed' else None,
   false_negative_rate=1-called/len(ids) if ids and name=='mixed' else None,
   correct_group_recall=(called if name=='mixed' else len(ids)-called)/len(ids) if ids else None)
 return dict(questions=len(rows),mixed_questions=sum(y),mixed_fraction=sum(y)/len(rows),confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),
  mixed_recall=tpr,nonmixed_recall=tnr,mixed_false_negative_rate=1-tpr if tpr is not None else None,
  nonmixed_false_positive_rate=1-tnr if tnr is not None else None,
  mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,
  mixed_roc_auc=auc(scores,y),classwise=classes,k_histogram=[ks.count(k) for k in range(6)],mean_q=sum(scores)/len(scores))
def compare(left,right):
 if isinstance(left,dict):
  assert set(left)==set(right)
  for k in left:compare(left[k],right[k])
 elif isinstance(left,float):assert abs(left-right)<1e-12,(left,right)
 else:assert left==right,(left,right)
results={};identities={};initial_heads=set();plan_hashes=set()
for layer,cap in [(21,20),(24,23),(27,26)]:
 base=A/('layer%d'%layer);cfg=read(base/'config.json');p=base/'evaluation-epoch-0004.json'
 if not p.exists():continue
 config_sha=sha(base/'config.json');accepted=read(p)
 final=read(base/'complete.json')
 assert final['status']=='FRESH_OFFICIAL_MULTILAYER_HF_REPLAY_FOUR_EPOCHS_COMPLETE' and final['cursor']==final['steps']==512 and final['epochs']==4 and final['actor_updates']==0
 assert final['config_sha256']==config_sha and len(final['epoch_evaluations'])==4
 for epoch in range(1,5):
  prior=read(base/('evaluation-epoch-%04d.json'%epoch));assert final['epoch_evaluations'][epoch-1]==prior and prior['cursor']==128*epoch and prior['config_sha256']==config_sha
  cp_path=pathlib.Path(prior['checkpoint']['path']);assert cp_path.is_file() and cp_path.stat().st_size==prior['checkpoint']['bytes']
  for m in prior['metrics'].values():checked(m['predictions'])
 assert accepted['status']=='EPOCH_FULLSTATE_SAVED_THEN_FIXED_POOLS_EVALUATED' and accepted['epoch']==4 and accepted['cursor']==512
 assert accepted['config_sha256']==config_sha and accepted['feature_backend']=='huggingface_frozen_causal_prefill'
 assert accepted['native_bitexact_claim'] is False and accepted['untouched_final_read'] is False and accepted['threshold_fitted'] is False
 cp=accepted['checkpoint'];assert cp['cursor']==cp['steps']==512 and cp['config_sha256']==config_sha
 assert cp['phase']=='EPOCH_TRAINED_EVALUATION_PENDING'
 checkpoint_path=checked(cp)
 assert checkpoint_path==base/'checkpoints/cursor-000512.pt'
 assert set(cp['accepted_hashes'])=={'raw','ema','adam','raw_buffers','ema_buffers','rng'}
 assert all(len(value)==64 for value in cp['accepted_hashes'].values())
 replay=read(base/'next-update-replay-cursor000000.json');assert replay['passed'] is True and replay['actual']==replay['expected'] and replay['source_cursor']==0 and replay['steps']==1
 restored=read(base/'restore-acceptance.json');assert restored['fresh_official_branch'] is True and restored['E1_resume_claim'] is False and restored['cursor']==restored['steps']==0
 initial_heads.add(restored['common_head_initial_sha256']);plan_hashes.add(cfg['plan']['sha256'])
 assert cfg['block_layer']==layer and cfg['update_budget']==512 and cfg['epochs']==4 and cfg['head_seed']==cfg['training_seed']==1729
 bank=read(checked(cfg['bank']));sources={'diagnostic512':bank['diagnostic'],'selection2048':read(checked(cfg['selection_bank']))['selection_validation']};del bank
 out={}
 for role,count in [('diagnostic512',512),('selection2048',2048)]:
  groups=sources[role];assert len(groups)==count
  by={g['qid']:g for g in groups};assert len(by)==count
  source_identity=digest([[g['qid'],g['trajectory_ids'],g['labels']] for g in groups])
  if role in identities:assert identities[role]==source_identity
  identities[role]=source_identity
  record=accepted['metrics'][role];prediction_path=checked(record['predictions'])
  with gzip.open(str(prediction_path),'rt') as f:rows=[json.loads(line) for line in f]
  assert len(rows)==count and len(set(r['qid'] for r in rows))==count
  max_q_error=0.
  for r,g in zip(rows,groups):
   assert r['qid']==g['qid'] and r['trajectory_ids']==g['trajectory_ids'] and r['strict_em']==g['labels']
   assert len(r['p_j'])==5 and all(math.isfinite(v) and 0<=v<=1 for v in r['p_j'])
   q=1-product(r['p_j'])-product([1-v for v in r['p_j']]);max_q_error=max(max_q_error,abs(q-r['predicted_mixed']))
  assert max_q_error<1e-6
  selected=sorted(rows,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512]
  pool,top=metrics(rows),metrics(selected);compare(pool,record['pool']);compare(top,record['selected512'])
  out[role]=dict(pool=pool,selected512=top,source_own_G5_ID_EM_sha256=source_identity,
   selected_qids_sha256=digest([r['qid'] for r in selected]),q_formula_max_absolute_error=max_q_error,predictions=record['predictions'],
   independent_own_EM_BA_AUC_Top512_verified=True)
 results['layer%d'%layer]=dict(block_layer=layer,capture_layer=cap,epoch=4,cursor=512,checkpoint=cp,
  full_checkpoint_file_SHA_freshly_verified=True,all_four_epoch_evaluation_receipts_and_predictions_SHA_verified=True,actual_final_complete=dict(path=str(base/'complete.json'),sha256=sha(base/'complete.json')),checkpoint_tensor_loaded_for_this_audit=False,
  saved_by_trainer_fullstate_CPU_reload_component_hashes_verified=True,
  first_actual_next_update_replay_all_hashes_equal=True,evaluation_receipt=dict(path=str(p),sha256=sha(p)),metrics=out)
assert len(initial_heads)<=1 and len(plan_hashes)<=1
report=dict(status='FOUR_EPOCH_THREE_LAYER_FINAL_FULLCP_OWN_EM_METRICS_INDEPENDENT_PASS' if len(results)==3 else 'PARTIAL_FOUR_EPOCH_LAYER_FINAL_ACCEPTANCE',
 observed_epoch=time.time(),completed_layers=len(results),expected_layers=3,results=results,
 identical_head_initialization=len(initial_heads)==1,identical_4epoch_plan=len(plan_hashes)==1,
 backend='uniform_huggingface_frozen_causal_prefill',native_bitexact_claim=False,untouched_final_read=False,
 torch_imported=False,raw_predictions_returned=False)
print(json.dumps(report))
