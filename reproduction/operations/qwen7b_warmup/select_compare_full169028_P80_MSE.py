"""Apply the existing seed1-only rule after both full runs finish.

Reads already-viewed prediction files only. This is family checkpoint selection
and descriptive paired analysis, not the final cross-family candidate freeze.
"""
import hashlib,json,os,subprocess,time
from pathlib import Path

A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
BASES=[A/'full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3',A/'full169028-L15P80-MSE-ownG5-physical-seed19137-plan2026102302-20261023-v3']
OUT=A/'full169028-P80-MSE-actual-family-selection-20261023-v1'
RULE_SHA='17fc2ae23d8cfa5598eebf590ac01d2ed09b155a8453f68487052762a39483ee'

def choose_seed1(candidates):
 if len(candidates)!=20 or {(r['epoch'],r['state'])for r in candidates}!={(e,s)for e in range(1,11)for s in('raw','EMA')}:raise ValueError('All20 eligible seed1 epoch/state results required')
 if any(type(r['Top512_Mixed'])is not int or not 0<=r['Top512_Mixed']<=512 for r in candidates):raise ValueError('Actual Top512 integer counts required')
 # Predeclared: highest seed1 count, then earliest epoch, then EMA.
 return dict(sorted(candidates,key=lambda r:(-r['Top512_Mixed'],r['epoch'],r['state']!='EMA'))[0])

def compare_arrays(ys,scores,thresholds,repeats=2000,seed=2026102308):
 import numpy as np
 y=(ys.sum(1)>0)&(ys.sum(1)<5);matrix=np.column_stack(scores);threshold=np.array(thresholds);rng=np.random.default_rng(seed);tops=[];bas=[]
 for _ in range(repeats):
  ix=rng.integers(0,len(y),len(y));labels=y[ix];s=matrix[ix];pred=s>=threshold
  bas.append(.5*(pred[labels].mean(0)+(~pred[~labels]).mean(0)))
  # Original rows are sorted by qid: preserve the producer's qid tie-break,
  # including when resampling repeats an original question.
  order=np.column_stack([np.lexsort((ix,-s[:,j]))[:512]for j in range(s.shape[1])]);tops.append(np.take_along_axis(np.broadcast_to(labels[:,None],s.shape),order,axis=0).mean(0))
 tops=np.array(tops);bas=np.array(bas)
 return dict(repeats=repeats,seed=seed,unit='paired_question_resample2048_then_reselect_Top512_per_model',two_seed_average_not_score_ensemble=True,not_training_seed_uncertainty=True,development_selection_bias_not_removed=True,Top512_two_seed_mean_minus_E1_percentile95=np.quantile(tops[:,1:].mean(1)-tops[:,0],[.025,.975]).tolist(),BA_two_seed_mean_minus_E1_percentile95=np.quantile(bas[:,1:].mean(1)-bas[:,0],[.025,.975]).tolist())

def run():
 import numpy as np
 from compare_full169028_packed_variance import read,proof,rows,data,metric,E1_SHA
 assert os.uname().sysname=='Linux'and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 started=time.time();sources={};evals=[]
 for i,base in enumerate(BASES):
  job=str(read(base/'root-submission.json')['job_id']);assert job==('750189','750190')[i]
  account=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],universal_newlines=True,timeout=15)
  assert [job,'COMPLETED','0:0']in[line.split('|')[:3]for line in account.splitlines()],'Wait for actual successful terminal job; do not select a partial run'
  complete=read(base/'complete.json');assert complete['epochs']==10 and complete['actual_Adam_updates']==complete['EMA_updates']==complete['batch_attempts']==13210 and complete['training_questions']==169028 and not complete['final4096_scored']
  assert proof(base/'predeclared-selection-rule.json')['sha256']==RULE_SHA
  publication=read(base/'source-publication.json');assert publication['commit']=='7a8ff6ae8a33324846d85f77ffc28e9ab699c028'
  by={e:read(base/('evaluation-epoch-%02d.json'%e))for e in range(1,11)}
  for e,ev in by.items():
   assert ev['epoch']==e and ev['actual_Adam_updates']==1321*e and ev['binding']['init_seed']==(9137,19137)[i]and ev['binding']['training_questions']==169028 and ev['binding']['fixed_m']==5 and ev['binding']['prefix']==80
  evals.append(by);sources['seed%d'%(i+1)]=dict(job_id=job,actual_terminal_accounting=account.strip(),complete=proof(base/'complete.json'),source_publication=proof(base/'source-publication.json'),predeclared_rule=proof(base/'predeclared-selection-rule.json'))
 assert evals[0][1]['binding']['fixture']==evals[1][1]['binding']['fixture']and evals[0][1]['binding']['physical_source_contract']==evals[1][1]['binding']['physical_source_contract']
 choices=[dict(epoch=e,state=s,Top512_Mixed=evals[0][e]['metrics']['fixed_validation2048_G5'][s]['Top512']['Mixed'],evaluation=proof(BASES[0]/('evaluation-epoch-%02d.json'%e)))for e in range(1,11)for s in('raw','EMA')]
 selected=choose_seed1(choices);epoch,state=selected['epoch'],selected['state']
 epath=A/'joint-mixed-direct-official-169028-b128-v1/evaluations/epoch-01/complete.json';assert proof(epath)['sha256']==E1_SHA
 eref=read(epath)['metrics']['EMA']['fixed_selection2048']['predictions'];assert proof(eref['path'])=={k:eref[k]for k in('path','bytes','sha256')}
 rr=rows(eref['path']);ids,ys=data(rr);baseline=np.array([r['predicted_mixed']for r in rr]);scores=[baseline];summary={'E1_EMA':metric(ys,baseline,.5,False)};train={};trainidentity=None
 sources['E1_EMA']=dict(receipt=proof(epath),predictions=eref)
 for i,base in enumerate(BASES):
  name='MSE_seed%d'%(i+1);ev=evals[i][epoch]
  for role in('fixed_validation2048_G5','fixed_train2048_G5'):
   path=base/('epoch-%02d-%s-predictions.jsonl.gz'%(epoch,role));rr=rows(path);ownids,ownys=data(rr);score=np.array([r[state+'_score']for r in rr])/4
   assert np.isfinite(score).all()and(score>=0).all()and(score<=.25).all();value=metric(ownys,score,.08,True);expected=ev['metrics'][role][state]
   assert value['Top512']['Mixed']==expected['Top512']['Mixed']and abs(value['Mixed_balanced_accuracy']-expected['balanced_accuracy'])<1e-12
   assert abs(value['variance_MSE']-expected['variance_MSE'])<1e-10
   if role=='fixed_validation2048_G5':
    assert ownids==ids and np.array_equal(ownys,ys);scores.append(score);summary[name]=value
   else:
    identity=json.dumps([ownids,ownys.tolist()],sort_keys=True)
    if trainidentity is None:trainidentity=identity
    assert identity==trainidentity;train[name]=dict(pool='fixed_train_B2048_ownG5_not_historical_E1_trainA',metrics=value)
   sources[name+'_'+role]=proof(path)
  sources[name+'_selected_evaluation']=proof(base/('evaluation-epoch-%02d.json'%epoch))
 boot=compare_arrays(ys,scores,[.5,.08,.08]);tops=[summary[n]['Top512']['fraction']for n in('MSE_seed1','MSE_seed2')];bas=[summary[n]['Mixed_balanced_accuracy']for n in('MSE_seed1','MSE_seed2')]
 result=dict(status='ACTUAL_FULL169028_P80_FAMILY_SELECTED_AND_VIEWED_PAIRED_ANALYSIS_COMPLETE',time=time.time(),primary_metric='Top512_true_Mixed_fraction',selection_rule='maxseed1count_then_earliest_epoch_then_EMA',actual_selected=selected,all20_seed1_choices=choices,seed2_used_for_selection=False,same_epoch_state_on_seed2=True,selected_epoch=epoch,selected_state=state,summaries=summary,fixed_trainB=train,two_seed_mean=dict(Top512_fraction=float(np.mean(tops)),Top512_minus_E1=float(np.mean(tops)-summary['E1_EMA']['Top512']['fraction']),BA=float(np.mean(bas)),BA_minus_E1=float(np.mean(bas)-summary['E1_EMA']['Mixed_balanced_accuracy'])),bootstrap=boot,source_proofs=sources,selected_CP_CPU_acceptance_required_before_candidate_freeze=True,actual_cross_family_candidate_frozen=False,model_or_checkpoint_tensors_read=False,new_inference=False,Slurm_mutated=False,final4096_read=False,seconds=time.time()-started)
 OUT.mkdir(exist_ok=True);dest=OUT/'actual-family-selection-and-paired.json'
 if dest.exists():
  old=read(dest);assert(old['selected_epoch'],old['selected_state'],old['source_proofs'])==(epoch,state,sources),'Existing selection inputs differ'
  result=old
 else:dest.write_text(json.dumps(result,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps(dict(receipt=proof(dest),result=result),allow_nan=False))

if __name__=='__main__':run()
