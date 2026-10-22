"""Independent small-output audit of four completed direct-Mixed epochs.

Pure metric helpers are reused unchanged from CP512 audit source1e95d531.
No model/checkpoint/feature payload or new forward is opened.
"""
import gzip,hashlib,json,math,time,subprocess
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
BASE=A/'joint-mixed-direct-official-169028-b128-v1';OUT=A/'mixed-direct-b128-four-epoch-metrics-v1'
CONFIG_SHA='e5f0027dfcff4b6db442b38e9b2452e86cc2c9adcabd97a194971dc4d8f8d837'
CP512_AUDIT=dict(path=str(A/'mixed-direct-official-cp512-eval-v1/E0-CP128-CP512-independent-audit.json'),bytes=24262,sha256='445529fe00e9a81edaf4321a1ee12c0cd89f449025360519ad18191e49ef3366')
EPOCH_REFS={'1': {'bytes': 27129, 'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1/evaluations/epoch-01/complete.json', 'sha256': '3d832d00b2d40eaf5ffad4568c1785c84270b0a81e033c57e6c7030a93cd0de8'}, '2': {'bytes': 26607, 'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1/evaluations/epoch-02/complete.json', 'sha256': 'e570c8be473423092d4c177476c7c670cb3343db3485859392ee8aca1bf7b954'}, '3': {'bytes': 27065, 'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1/evaluations/epoch-03/complete.json', 'sha256': 'aa9bff3e29086d52721b6bd692ad74908a8089fd33f091bc16e39bcaabff33ed'}, '4': {'bytes': 26594, 'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1/evaluations/epoch-04/complete.json', 'sha256': '8a555cda65bcb902683d164996e00e96151ea4233f4bea07df87511c5af4e341'}}
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

def main():
 terminal=subprocess.check_output(['sacct','-X','-n','-P','-j','746266','--format=JobIDRaw,State,ExitCode,Elapsed']).decode().strip().splitlines();terminal=[r for r in terminal if r.split('|')[0]=='746266'];assert len(terminal)==1 and terminal[0].split('|')[:3]==['746266','COMPLETED','0:0']
 config=proof(BASE/'config.json');assert config['sha256']==CONFIG_SHA;final_ref=proof(BASE/'complete.json');final=read(final_ref['path']);assert final['status']=='OFFICIAL_DIRECT_MIXED_B128_FOUR_EPOCH_QUESTION_BUDGET_COMPLETE'and final['cursor']==final['native_steps']==final['head_steps']==6540 and final['global_batch_questions']==128 and final['final4096_scored']is False
 accepted=read(checked(CP512_AUDIT));reference=rows_for(accepted['prediction_refs']['CP512']['raw']['fixed_selection2048'],2048);identity={r['qid']:(r['trajectory_ids'],r['strict_em'])for r in reference};pool_k=summarize(reference)['k_histogram'];result={};prediction_refs={};receipt_refs={};bce_errors=[];validation=None
 for epoch,cursor in((1,2577),(2,3898),(3,5219),(4,6540)):
  ref=EPOCH_REFS[str(epoch)];complete=read(checked(ref));cp=complete['checkpoint'];assert complete['status']=='DIRECT_MIXED_B128_FIXED512_2048_EVALUATION_COMPLETE'and complete['epoch']==epoch and complete['cursor']==cursor and complete['config_sha256']==CONFIG_SHA and complete['checkpoint_saved_before_evaluation']is True and complete['models_two_Adam_RNG_native_buffers_unchanged']is True and complete['final4096_scored']is False and complete['threshold_fitted']is False
  assert cp['cursor']==cp['native_steps']==cp['head_steps']==cursor and cp['allocation_job_id']=='746266'and cp['full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed']is True
  if validation is None:validation=complete['validation_proof']
  assert validation==complete['validation_proof'];result[str(epoch)]={};prediction_refs[str(epoch)]={};receipt_refs[str(epoch)]=ref
  for kind in('raw','EMA'):
   item=complete['metrics'][kind]['fixed_selection2048'];rows=rows_for(item['predictions'],2048);assert{r['qid']:(r['trajectory_ids'],r['strict_em'])for r in rows}==identity
   summary=summarize(rows);chosen=sorted(rows,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512];top=summarize(chosen);assert summary['k_histogram']==pool_k and summary['confusion']==item['pool']['confusion']and top['mixed_questions']==item['selected512']['mixed_questions']and top['k_histogram']==item['selected512']['k_histogram'];assert item['ordered_qids_sha256']==digest([r['qid']for r in rows])and item['ordered_own_trajectory_ids_sha256']==digest([r['trajectory_ids']for r in rows])and item['selected_qids_sha256']==digest([r['qid']for r in chosen])
   for key in('mixed_balanced_accuracy','mixed_recall','nonmixed_recall','mixed_roc_auc'):assert summary[key]==item['pool'][key],key
   if summary['saturated_probability_count']==0:
    error=abs(summary['binaryMixedBCE_from_probability']-item['binaryMixedBCE']);assert error<1e-5;bce_errors.append(error)
    for key,value in summary['classwise'].items():assert abs(value['binaryMixedBCE_from_probability']-item['classwise_binaryMixedBCE']['Mixed'if key=='mixed'else key]['binaryMixedBCE'])<1e-5
   summary['producer_stable_binaryMixedBCE']=item['binaryMixedBCE'];summary['producer_classwise_stable_binaryMixedBCE']=item['classwise_binaryMixedBCE'];summary['selected512']=dict(mixed=top['mixed_questions'],fraction=top['mixed_questions']/512,k_histogram=top['k_histogram'],class_questions={n:v['questions']for n,v in top['classwise'].items()});summary['cursor']=cursor;result[str(epoch)][kind]=summary;prediction_refs[str(epoch)][kind]=item['predictions']
 assert validation==read(checked(accepted['complete_refs']['CP512']))['validation_proof'];out=dict(status='FOUR_B128_EPOCHS_FIXED2048_OWN_G5_EM_MIXED_METRICS_INDEPENDENTLY_RECOMPUTED',job_id='746266',terminal_sacct=terminal[0],config=config,training_complete=final_ref,epoch_complete_refs=receipt_refs,prediction_refs=prediction_refs,reference_identity_audit=CP512_AUDIT,metrics=result,pool_K0_to_K5=pool_k,all_eight_predictions_same_own_G5_IDs_and_EM=True,BCE_probability_vs_stable_max_error=max(bce_errors)if bce_errors else None,threshold=.5,new_scoring=False,final4096_read=False,checkpoint_tensor_or_feature_payload_read=False,own_EM_provenance_inherited_not_recomputed_from_raw=True,producer_CPU_checkpoint_reload_inherited_not_repeated=True,epoch_time=time.time())
 OUT.mkdir(exist_ok=True);p=OUT/'independent-metrics.json'
 if p.exists():
  old=read(p);assert old['metrics']==out['metrics']and old['epoch_complete_refs']==out['epoch_complete_refs'];out=old
 else:
  with p.open('xb')as f:f.write((json.dumps(out,sort_keys=True,separators=(',',':'))+'\n').encode())
 out=dict(out,independent_audit_ref=proof(p));print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
