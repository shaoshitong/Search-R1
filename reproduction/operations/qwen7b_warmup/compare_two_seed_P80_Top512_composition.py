"""Supplement fixed selected-checkpoint Top512 classes; no new selection/epochs."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from compare_full169028_packed_variance import read,proof,rows,data
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');OUT=A/'viewed2048-two-seed-P80-paired-comparison-20261023-v1'
ANALYSIS=OUT/'analysis-1792697695.json';ANALYSIS_SHA='b0aa2ca135e501d03bbd49777dedab0e42a32f56887308d9488de54d75434bed'
def run():
 assert os.uname().sysname=='Linux'and os.environ.get('CUDA_VISIBLE_DEVICES')=='';assert proof(ANALYSIS)['sha256']==ANALYSIS_SHA;anchor=read(ANALYSIS);out={};identity=None
 for name in ['E1_EMA','CURRENT64_seed1','CURRENT64_seed2','MSE_M5_seed1','MSE_M5_seed2']:
  ref=anchor['source_proofs'][name]['predictions']if name=='E1_EMA'else anchor['source_proofs'][name]['prediction_proofs']['fixed_validation2048_G5'];assert proof(ref['path'])=={k:ref[k]for k in('path','bytes','sha256')};rr=rows(ref['path']);ids,ys=data(rr)
  if identity is None:identity=(ids,ys)
  assert ids==identity[0]and np.array_equal(ys,identity[1]);score=np.array([r['predicted_mixed']if name=='E1_EMA'else r['raw_score']for r in rr]);order=np.argsort(-score,kind='stable')[:512];k=ys[order].sum(1);hist={str(i):int((k==i).sum())for i in range(6)};classes=dict(all_wrong=int((k==0).sum()),Mixed=int(((k>0)&(k<5)).sum()),all_correct=int((k==5).sum()));assert sum(classes.values())==512 and classes['Mixed']==anchor['independent_summaries'][name]['Top512']['Mixed'];wrong=512-classes['Mixed'];out[name]=dict(Top512=classes,EM_count_histogram=hist,selection_errors=wrong,error_composition=dict(all_wrong_fraction_of_nonMixed_selected=classes['all_wrong']/wrong,all_correct_fraction_of_nonMixed_selected=classes['all_correct']/wrong),selected_question_ids_sha256=hashlib.sha256(json.dumps([ids[i][0]for i in order],separators=(',',':')).encode()).hexdigest(),source_prediction=ref)
 baseline=out['E1_EMA']['Top512'];families={}
 for name,pair in [('CURRENT64_E5_raw',['CURRENT64_seed1','CURRENT64_seed2']),('MSE_M5_E7_raw',['MSE_M5_seed1','MSE_M5_seed2'])]:
  mean={c:sum(out[n]['Top512'][c]for n in pair)/2 for c in baseline};families[name]=dict(seeds=pair,mean_Top512_class_counts=mean,mean_minus_E1_EMA_counts={c:mean[c]-baseline[c]for c in baseline},not_score_ensemble=True)
 result=dict(status='FIXED_SELECTED_P80_TWO_SEED_TOP512_CLASS_COMPOSITION_COMPLETE',time=time.time(),source_frozen_analysis=proof(ANALYSIS),exactly_same2048_ownG5_strictEM=True,selected_checkpoints=out,two_seed_mean=families,classification_threshold_not_used_for_ranking=True,bootstrap_not_changed_or_rerun=True,epochs_or_models_not_reselected=True,final4096_read=False,model_or_checkpoint_tensor_read=False,new_inference=False);p=OUT/'Top512-class-composition.json';p.write_text(json.dumps(result,sort_keys=True,allow_nan=False)+'\n');print(json.dumps(dict(receipt=proof(p),result=result)))
if __name__=='__main__':run()
