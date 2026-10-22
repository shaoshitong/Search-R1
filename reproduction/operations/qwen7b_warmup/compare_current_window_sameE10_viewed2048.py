"""CURRENT64 vs WINDOW2018, same E10/raw, same two seeds and own-G5 pool."""
import ast,hashlib,json,os,time
from pathlib import Path
import numpy as np
from compare_full169028_packed_variance import read,proof,rows,data,metric
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');OUT=A/'viewed2048-current64-window2018-sameE10-comparison-20261023-v1';REPEATS=2000;BOOTSTRAP_SEED=2026102305
ARMS=[('CURRENT64_seed1',A/'online-packed-global64-window2018-L15P80-8192-20261023-v1/current64',0,'online_packed_window_topk.py'),('WINDOW2018_seed1',A/'online-packed-global64-window2018-L15P80-8192-20261023-v1/window2018',2018,'online_packed_window_topk.py'),('CURRENT64_seed2',A/'online-packed-current64-seed19137-plan2026102302-L15P80-8192-20261023-v1/current64',0,'online_packed_window_topk_seeded.py'),('WINDOW2018_seed2',A/'online-packed-window2018-seed19137-plan2026102302-L15P80-8192-20261023-v2/window2018',2018,'online_packed_window_topk_seeded_v2.py')]
def identity_digest(ids,ys):return hashlib.sha256(json.dumps(dict(identities=ids,strictEM=ys.tolist()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def run():
 assert os.uname().sysname=='Linux'and os.environ.get('CUDA_VISIBLE_DEVICES')=='';started=time.time();refids=refys=train_identity=None;summaries={};train={};scores=[];sources={};metadata={}
 for name,base,capacity,module in ARMS:
  evpath=base/'evaluation-epoch-10.json';ev=read(evpath);binding=ev['binding'];assert ev['epoch']==10 and ev['cursor']==1280 and not ev['final4096_scored'];assert binding['global_batch']==64 and binding['fixed_m']==5 and binding['layer']==15 and binding['prefix']==80 and binding['head_max_positions']==2048 and binding['EMA_decay']==.9995 and binding['lr']==5e-6 and binding['objective']=='topk_ramp'and binding['cache_window_capacity']==capacity
  pub=read(base.parent/'source-publication.json');path=Path(pub['code_root'])/'reproduction/operations/qwen7b_warmup'/module;src=proof(path);assert {k:src[k]for k in('bytes','sha256')}==pub['files']['reproduction/operations/qwen7b_warmup/'+module];fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='run');run_sha=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest();plans={str(e):read(base/('coverage-epoch-%02d.json'%e))['plan_sha256']for e in range(1,11)}
  metadata[name]=dict(epoch=10,state='raw',batch_attempts=1280,actual_Adam_updates=ev['actual_Adam_updates'],EMA_updates=ev['EMA_updates'],binding=binding,run_body_AST_SHA256=run_sha,training_plan_SHAs=plans)
  paths={}
  for role in('fixed_validation2048_G5','fixed_train2048_G5'):
   path=base/('epoch-10-%s-predictions.jsonl.gz'%role);rr=rows(path);ids,ys=data(rr);score=np.array([r['raw_score']for r in rr]);assert np.isfinite(score).all()and(score>=0).all()and(score<=1).all();value=metric(ys,score,.5,False);value['Mixed_BCE']=None;value['score_semantics']='sigmoid_ranking_score_not_calibrated_Mixed_probability';k=ys[np.argsort(-score,kind='stable')[:512]].sum(1);value['Top512']['true_class_composition']=dict(all_wrong=int((k==0).sum()),Mixed=int(((k>0)&(k<5)).sum()),all_correct=int((k==5).sum()));original=ev['metrics'][role]['raw'];assert value['Top512']['Mixed']==original['Top512']['Mixed']and abs(value['Mixed_balanced_accuracy']-original['balanced_accuracy'])<1e-12
   if role=='fixed_validation2048_G5':
    if refids is None:refids,refys=ids,ys
    assert ids==refids and np.array_equal(ys,refys);summaries[name]=value;scores.append(score)
   else:
    ident=identity_digest(ids,ys)
    if train_identity is None:train_identity=ident
    assert ident==train_identity;train[name]=dict(pool_identity='fixed_train_B2048_ownG5_not_old_trainA',identity_sha256=ident,metrics=value)
   paths[role]=proof(path)
  sources[name]=dict(predictions=paths,evaluation=proof(evpath),source_publication=proof(base.parent/'source-publication.json'),root_submission=read(base/'root-submission.json'),module_source=src)
 for cur,win in [('CURRENT64_seed1','WINDOW2018_seed1'),('CURRENT64_seed2','WINDOW2018_seed2')]:
  assert metadata[cur]['training_plan_SHAs']==metadata[win]['training_plan_SHAs']and metadata[cur]['run_body_AST_SHA256']==metadata[win]['run_body_AST_SHA256']
  assert metadata[cur]['binding']['fixture']==metadata[win]['binding']['fixture']and metadata[cur]['binding']['seed']==metadata[win]['binding']['seed']
  if cur.endswith('seed2'):assert all(metadata[cur]['binding'][k]==metadata[win]['binding'][k]for k in('init_seed','train_plan_seed','diagnostic_seed','initial_head_sha256'))
 matrix=np.column_stack(scores);y=(refys.sum(1)>0)&(refys.sum(1)<5);rng=np.random.default_rng(BOOTSTRAP_SEED);tops=[];bas=[]
 for unused in range(REPEATS):
  ix=rng.integers(0,2048,2048);yy=y[ix];s=matrix[ix];pred=s>=.5;bas.append(.5*(pred[yy].mean(0)+(~pred[~yy]).mean(0)));order=np.argsort(-s,axis=0,kind='stable')[:512];tops.append(np.take_along_axis(np.broadcast_to(yy[:,None],s.shape),order,axis=0).mean(0))
 tops=np.array(tops);bas=np.array(bas);names=[r[0]for r in ARMS];comparisons={}
 for label,current,window in [('seed1',[0],[1]),('seed2',[2],[3]),('two_seed_average',[0,2],[1,3])]:
  curtop=float(np.mean([summaries[names[i]]['Top512']['fraction']for i in current]));wintop=float(np.mean([summaries[names[i]]['Top512']['fraction']for i in window]));curba=float(np.mean([summaries[names[i]]['Mixed_balanced_accuracy']for i in current]));winba=float(np.mean([summaries[names[i]]['Mixed_balanced_accuracy']for i in window]));comparisons[label]=dict(candidate=[names[i]for i in window],baseline=[names[i]for i in current],CURRENT64_Top512_fraction=curtop,WINDOW2018_Top512_fraction=wintop,WINDOW_minus_CURRENT_Top512_delta=wintop-curtop,Top512_paired_question_percentile95=np.quantile(tops[:,window].mean(1)-tops[:,current].mean(1),[.025,.975]).tolist(),CURRENT64_BA=curba,WINDOW2018_BA=winba,BA_delta=winba-curba,BA_paired_question_percentile95=np.quantile(bas[:,window].mean(1)-bas[:,current].mean(1),[.025,.975]).tolist())
 result=dict(status='CURRENT64_WINDOW2018_SAME_E10_RAW_TWO_SEED_PAIRED_COMPARISON_COMPLETE',time=time.time(),independent_summaries=summaries,fixed_train_B_separate=train,paired_comparisons=comparisons,actual_training_metadata=metadata,source_proofs=sources,identity=dict(questions=2048,ownG5=True,strictEM_identical=True,true_Mixed=int(y.sum()),qid_trajectory_strictEM_SHA256=identity_digest(refids,refys)),fairness=dict(same_E10_raw_not_E5=True,same1280_global64_batch_attempts=True,per_seed_run_body_AST_and_all10_sampling_plans_equal=True,actual_Adam_skip_difference_is_part_of_cache_algorithm=True,classification_threshold_diagnostic_only_not_calibrated=True),bootstrap=dict(repeats=REPEATS,seed=BOOTSTRAP_SEED,unit='paired_question_2048_resample_then_rerank_each_Top512',two_seed_average='Mean of per-model fractions, not score ensemble',not_training_seed_CI=True,limitation='Conditional exploratory question-resampling uncertainty; not new-question generalization or training-seed confidence.'),final4096_read=False,model_or_checkpoint_or_hidden_read=False,new_inference=False,Slurm_mutated=False,seconds=time.time()-started)
 OUT.mkdir(exist_ok=True);p=OUT/('analysis-%d.json'%int(result['time']));p.write_text(json.dumps(result,sort_keys=True,allow_nan=False)+'\n');(OUT/'latest.json').write_text(json.dumps(result,sort_keys=True,allow_nan=False)+'\n');print(json.dumps(dict(receipt=proof(p),result=result),allow_nan=False))
if __name__=='__main__':run()
