"""Paired-question bootstrap of fixed, already-viewed P80 two-seed predictions.

No models, checkpoints, hidden tensors, new inference or final4096 are opened.
The two-seed average averages separate models' Top512 yields, not their scores.
"""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from compare_full169028_packed_variance import read,proof,rows,data,metric,E1_SHA
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
OUT=A/'viewed2048-two-seed-P80-paired-comparison-20261023-v1'
REPEATS=2000;BOOTSTRAP_SEED=2026102304
ARMS=[('CURRENT64_seed1',A/'online-packed-global64-window2018-L15P80-8192-20261023-v1/current64',5,False,266),('CURRENT64_seed2',A/'online-packed-current64-seed19137-plan2026102302-L15P80-8192-20261023-v1/current64',5,False,267),('MSE_M5_seed1',A/'online-packed-matched-L15P80-8192-20261023-v1/mse-M5',7,True,278),('MSE_M5_seed2',A/'online-packed-matched-MSE-M5-seed19137-plan2026102302-L15P80-8192-20261023-v1/mse-M5',7,True,277)]
def identity_digest(ids,ys):return hashlib.sha256(json.dumps(dict(identities=ids,strict_EM=ys.tolist()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def run():
 assert os.uname().sysname=='Linux'and os.environ.get('CUDA_VISIBLE_DEVICES')=='';started=time.time();summaries={};sources={};scores={};thresholds={};fixedtrain={};trainidentity=None
 epath=A/'joint-mixed-direct-official-169028-b128-v1/evaluations/epoch-01/complete.json';assert proof(epath)['sha256']==E1_SHA;eref=read(epath)['metrics']['EMA']['fixed_selection2048']['predictions'];assert proof(eref['path'])=={k:eref[k]for k in('path','bytes','sha256')};rr=rows(eref['path']);ids,ys=data(rr);identity=identity_digest(ids,ys);baseline=np.array([r['predicted_mixed']for r in rr]);summaries['E1_EMA']=metric(ys,baseline,.5,False);assert summaries['E1_EMA']['Top512']['Mixed']==205;scores['E1_EMA']=baseline;thresholds['E1_EMA']=.5;sources['E1_EMA']=dict(receipt=proof(epath),predictions=eref,role='original_169028_E1_EMA_fixed_selection2048')
 for name,base,epoch,isvariance,expectedtop in ARMS:
  ev=read(base/('evaluation-epoch-%02d.json'%epoch));assert ev['epoch']==epoch and not ev['final4096_scored'];roles={};paths={}
  for role in('fixed_validation2048_G5','fixed_train2048_G5'):
   path=base/('epoch-%02d-%s-predictions.jsonl.gz'%(epoch,role));rr=rows(path);ownids,ownys=data(rr);score=np.array([r['raw_score']for r in rr]);assert np.isfinite(score).all()and (score>=0).all()and(score<=1).all()
   if isvariance:score=score/4
   threshold=.08 if isvariance else.5;value=metric(ownys,score,threshold,isvariance)
   if not isvariance:value['Mixed_BCE']=None;value['score_semantics']='sigmoid_ranking_score_not_calibrated_Mixed_probability'
   original=ev['metrics'][role]['raw'];assert value['Top512']['Mixed']==original['Top512']['Mixed']and abs(value['Mixed_balanced_accuracy']-original['balanced_accuracy'])<1e-12
   if role=='fixed_validation2048_G5':
    assert ownids==ids and np.array_equal(ownys,ys);assert value['Top512']['Mixed']==expectedtop;summaries[name]=value;scores[name]=score;thresholds[name]=threshold
   else:
    token=identity_digest(ownids,ownys)
    if trainidentity is None:trainidentity=token
    assert token==trainidentity;fixedtrain[name]=dict(pool_identity='fixed_train_B2048_ownG5_not_old_E1_trainA',identity_sha256=token,metrics=value)
   paths[role]=proof(path)
  sources[name]=dict(epoch=epoch,state='raw',prediction_proofs=paths,evaluation=proof(base/('evaluation-epoch-%02d.json'%epoch)),source_publication=proof(base.parent/'source-publication.json'),root_submission=read(base/'root-submission.json'),selected_after_viewing_dev=True)
 names=['E1_EMA']+[r[0]for r in ARMS];matrix=np.column_stack([scores[n]for n in names]);thr=np.array([thresholds[n]for n in names]);y=(ys.sum(1)>0)&(ys.sum(1)<5);rng=np.random.default_rng(BOOTSTRAP_SEED);tops=[];bas=[]
 for unused in range(REPEATS):
  ix=rng.integers(0,2048,2048);labels=y[ix];s=matrix[ix];pred=s>=thr;bas.append(.5*(pred[labels].mean(0)+(~pred[~labels]).mean(0)));order=np.argsort(-s,axis=0,kind='stable')[:512];tops.append(np.take_along_axis(np.broadcast_to(labels[:,None],s.shape),order,axis=0).mean(0))
 tops=np.array(tops);bas=np.array(bas);families={}
 for family,columns in [('CURRENT64_E5_raw',[1,2]),('MSE_M5_E7_raw',[3,4])]:
  mtop=tops[:,columns].mean(1);mba=bas[:,columns].mean(1);observed_top=np.mean([summaries[names[c]]['Top512']['fraction']for c in columns]);observed_ba=np.mean([summaries[names[c]]['Mixed_balanced_accuracy']for c in columns]);families[family]=dict(seed_models=[names[c]for c in columns],seed_Top512_Mixed=[summaries[names[c]]['Top512']['Mixed']for c in columns],two_seed_mean_Top512_Mixed=float(observed_top*512),two_seed_mean_Top512_fraction=float(observed_top),two_seed_mean_BA=float(observed_ba),minus_original_E1_EMA=dict(Top512_fraction_delta=float(observed_top-summaries['E1_EMA']['Top512']['fraction']),paired_question_bootstrap_percentile95=np.quantile(mtop-tops[:,0],[.025,.975]).tolist(),BA_delta=float(observed_ba-summaries['E1_EMA']['Mixed_balanced_accuracy']),BA_paired_question_bootstrap_percentile95=np.quantile(mba-bas[:,0],[.025,.975]).tolist()))
 delta=tops[:,[3,4]].mean(1)-tops[:,[1,2]].mean(1);cross=dict(candidate='MSE_M5_E7_raw_two_seed_average',baseline='CURRENT64_E5_raw_two_seed_average',Top512_fraction_delta=families['MSE_M5_E7_raw']['two_seed_mean_Top512_fraction']-families['CURRENT64_E5_raw']['two_seed_mean_Top512_fraction'],paired_question_bootstrap_percentile95=np.quantile(delta,[.025,.975]).tolist(),causal_objective_only_claimed=False,confound='Different epoch, optimizer batch/number of updates and objective; observational selected checkpoints.')
 result=dict(status='VIEWED2048_TWO_SEED_P80_PAIRED_ANALYSIS_COMPLETE',time=time.time(),independent_summaries=summaries,family_two_seed_average=families,MSE_minus_current64=cross,fixed_train_B_separate=fixedtrain,dev_identity=dict(questions=2048,own_G5=True,real_Mixed=int(y.sum()),qid_trajectory_order_and_strictEM_exactly_equal=True,sha256=identity),source_proofs=sources,bootstrap=dict(repeats=REPEATS,seed=BOOTSTRAP_SEED,unit='paired_question_resampled2048_then_rerank_each_model_Top512',two_seed_mean='Arithmetic mean of per-model Top512 fractions, not score ensemble.',not_training_seed_uncertainty=True,limitation='Both epochs and architectures already explored on this development pool; conditional paired-question intervals do not measure seed uncertainty or untouched-data generalization.'),candidate_frozen=False,threshold_tuned=False,old_E1_trainA_used=False,model_or_checkpoint_or_hidden_tensors_read=False,final4096_read=False,new_inference=False,Slurm_mutated=False,seconds=time.time()-started)
 OUT.mkdir(exist_ok=True);dest=OUT/('analysis-%d.json'%int(result['time']));dest.write_text(json.dumps(result,sort_keys=True,allow_nan=False)+'\n');(OUT/'latest.json').write_text(json.dumps(result,sort_keys=True,allow_nan=False)+'\n');print(json.dumps(dict(receipt=proof(dest),result=result),allow_nan=False))
if __name__=='__main__':run()
