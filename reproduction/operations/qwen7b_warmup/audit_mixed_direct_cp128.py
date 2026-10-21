"""CPU stdlib-only paired E0/CP128 own-G5 metric audit; no model or new scoring."""
import gzip,hashlib,json,math,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');MAIN=A/'joint-mixed-direct-official-169028-v1';SNAP=A/'mixed-direct-official-cp128-eval-v1';CONFIG_SHA='2a7f039f4172f5284e3b8282e8ca6200e566bd92bb3013e9802306eca84e7393'
def proof(path):
 p=Path(path);assert p.stat().st_size<16*1024**2 and p.suffix!='.pt','Small prediction/metadata only';raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(path):proof(path);return json.loads(Path(path).read_bytes())
def checked(ref):
 assert proof(ref['path'])==ref;return Path(ref['path'])
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def auc(rows):
 ordered=sorted((r['predicted_mixed'],int(r['true_mixed']))for r in rows);pos=sum(y for _,y in ordered);neg=len(rows)-pos
 if not pos or not neg:return None
 total=0.;i=0
 while i<len(ordered):
  j=i+1
  while j<len(ordered)and ordered[j][0]==ordered[i][0]:j+=1
  total+=(i+1+j)/2*sum(y for _,y in ordered[i:j]);i=j
 return(total-pos*(pos+1)/2)/(pos*neg)
def summarize(rows):
 tp=fp=fn=tn=0;losses=[];classes={n:[]for n in('allwrong','mixed','allcorrect')};k_hist=[0]*6
 for r in rows:
  q=r['predicted_mixed'];y=r['true_mixed'];k=sum(r['strict_em']);k_hist[k]+=1;pred=q>=.5;tp+=int(pred and y);fp+=int(pred and not y);fn+=int(not pred and y);tn+=int(not pred and not y);clipped=max(1e-15,min(1-1e-15,q));loss=-math.log(clipped if y else 1-clipped);losses.append(loss);classes['allwrong'if k==0 else'allcorrect'if k==5 else'mixed'].append((q,pred,loss))
 tpr=tp/(tp+fn)if tp+fn else None;tnr=tn/(tn+fp)if tn+fp else None;cw={}
 for name,items in classes.items():
  n=len(items);called=sum(p for _,p,_ in items);cw[name]=dict(questions=n,predicted_mixed=called,error_count=n-called if name=='mixed'else called,error_rate=(n-called if name=='mixed'else called)/n if n else None,mean_q=sum(q for q,_,_ in items)/n if n else None,binaryMixedBCE_from_probability=sum(v for _,_,v in items)/n if n else None)
 return dict(questions=len(rows),mixed_questions=tp+fn,confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),mixed_recall=tpr,nonmixed_recall=tnr,mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,mixed_roc_auc=auc(rows),classwise=cw,k_histogram=k_hist,mean_q=sum(r['predicted_mixed']for r in rows)/len(rows),binaryMixedBCE_from_probability=sum(losses)/len(rows),saturated_probability_count=sum(r['predicted_mixed']in(0.,1.)for r in rows),fixed_threshold=.5)
def rows_for(ref,count):
 p=checked(ref);seen=set();tids=set();rows=[]
 with gzip.open(str(p),'rt')as f:
  for line in f:
   r=json.loads(line);assert set(r)=={'qid','trajectory_ids','strict_em','true_mixed','predicted_mixed'}and r['qid']not in seen and len(r['trajectory_ids'])==len(set(r['trajectory_ids']))==5 and not tids.intersection(r['trajectory_ids']);assert len(r['strict_em'])==5 and all(type(y)is int and y in(0,1)for y in r['strict_em']);assert type(r['true_mixed'])is bool and r['true_mixed']==(0<sum(r['strict_em'])<5)and math.isfinite(r['predicted_mixed'])and 0<=r['predicted_mixed']<=1;seen.add(r['qid']);tids.update(r['trajectory_ids']);rows.append(r)
 assert len(rows)==count;return rows

def audit(main=MAIN,snap=SNAP):
 main=Path(main);snap=Path(snap);current=snap/'complete.json'
 if not current.exists():return dict(status='WAITING_FOR_CP128_EVALUATION_COMPLETE',GPU_requested=False)
 c=read(current);initial_path=main/'evaluations/epoch-00/complete.json';initial=read(initial_path);source_cfg=read(snap/'source-config.json');assert proof(snap/'source-config.json')['sha256']==proof(main/'config.json')['sha256']==CONFIG_SHA;assert c['status']=='OFFICIAL_DIRECT_MIXED_CP128_RAW_EMA_FIXED512_2048_FROZEN_EVALUATION_COMPLETE'and c['cursor']==128 and c['all_saved_training_components_unchanged']is True and c['full_components_before']==c['full_components_after']==c['checkpoint']['accepted_hashes']and c['restoration']['full_CP_fresh_SHA_verified']is True and c['restoration']['actual_complete_restore_passed']is True;assert initial['cursor']==0 and initial['models_two_Adam_RNG_native_buffers_unchanged']is True and initial['live_components_before']==initial['live_components_after'];assert c['source_main_config']['sha256']==initial['config_sha256']==CONFIG_SHA and c['final4096_scored']is initial['final4096_scored']is False and c['validation_proof']==initial['validation_proof'];result={};references={};own_identity={}
 for phase,complete in(('E0',initial),('CP128',c)):
  result[phase]={};references[phase]={}
  for label in('raw','EMA'):
   result[phase][label]={};references[phase][label]={}
   for pool,count in(('diagnostic512',512),('fixed_selection2048',2048)):
    item=complete['metrics'][label][pool];rows=rows_for(item['predictions'],count);identity={r['qid']:(r['trajectory_ids'],r['strict_em'])for r in rows}
    if pool in own_identity:assert identity==own_identity[pool],'Initial/current raw/EMA ownIDs or ownEM changed'
    else:own_identity[pool]=identity
    summary=summarize(rows);chosen=sorted(rows,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512];top=summarize(chosen);assert item['ordered_qids_sha256']==digest([r['qid']for r in rows])and item['ordered_own_trajectory_ids_sha256']==digest([r['trajectory_ids']for r in rows])and item['selected_qids_sha256']==digest([r['qid']for r in chosen]);producer=item['pool'];assert summary['confusion']==producer['confusion']and summary['k_histogram']==producer['k_histogram']and top['mixed_questions']==item['selected512']['mixed_questions']and top['k_histogram']==item['selected512']['k_histogram']
    for name in('mixed_balanced_accuracy','mixed_recall','nonmixed_recall','mixed_roc_auc'):
     assert summary[name]==producer[name],name
    if summary['saturated_probability_count']==0:assert abs(summary['binaryMixedBCE_from_probability']-item['binaryMixedBCE'])<1e-5
    summary['binaryMixedBCE']=item['binaryMixedBCE'];summary['producer_classwise_stable_binaryMixedBCE']=item['classwise_binaryMixedBCE'];summary['selected512']=dict(mixed=top['mixed_questions'],fraction=top['mixed_questions']/len(chosen),k_histogram=top['k_histogram'],class_questions={n:v['questions']for n,v in top['classwise'].items()});result[phase][label][pool]=summary;references[phase][label][pool]=item['predictions']
 assert not set(own_identity['diagnostic512']).intersection(own_identity['fixed_selection2048'])
 out=dict(status='E0_TO_CP128_DIRECT_MIXED_FIXED512_2048_INDEPENDENT_CPU_PAIRED_METRIC_AUDIT_PASSED',initial_complete=proof(initial_path),snapshot_complete=proof(current),source_config=proof(snap/'source-config.json'),source_checkpoint=c['checkpoint'],prediction_refs=references,metrics=result,own_G5_IDs_EM_and_questions_identical_across_all_four_scores=True,actual_full_CP_restore_inherited_from_owned_GPU=True,no_checkpoint_or_tensor_loaded_by_auditor=True,no_new_model_scoring=True,no_threshold_fit=True,final4096_read=False,actor_updates=0,epoch=time.time());path=snap/'initial-vs-cp128-independent-audit.json';raw=(json.dumps(out,sort_keys=True,separators=(',',':'))+chr(10)).encode()
 if path.exists():
  prior=read(path);assert prior['snapshot_complete']==out['snapshot_complete']and prior['prediction_refs']==out['prediction_refs']and prior['metrics']==out['metrics'];return prior
 with path.open('xb')as f:f.write(raw)
 return out
