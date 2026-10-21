"""Exact full938 train-view prior; two frozen decision rules on existing q only.
Numeric train-only prior stays in remote complete-private.json. The public
summary contains aggregates and binding hashes, never the calibration value.
"""
import argparse,fcntl,gzip,hashlib,importlib.util,json,math,os,resource,time
from pathlib import Path

def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def auc(rows):
 v=sorted((r['score'],0<sum(r['strict_em'])<5) for r in rows);pos=sum(y for _,y in v);neg=len(v)-pos;s=0.;i=0
 if not pos or not neg:return None
 while i<len(v):
  j=i+1
  while j<len(v) and v[j][0]==v[i][0]:j+=1
  s+=((i+1+j)/2)*sum(y for _,y in v[i:j]);i=j
 return (s-pos*(pos+1)/2)/(pos*neg)
def metrics(rows,threshold):
 ks=[sum(r['strict_em']) for r in rows];y=[0<k<5 for k in ks];pr=[r['score']>=threshold for r in rows]
 tp=sum(a and b for a,b in zip(y,pr));fp=sum(not a and b for a,b in zip(y,pr));fn=sum(a and not b for a,b in zip(y,pr));tn=sum(not a and not b for a,b in zip(y,pr));tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
 per={}
 for name,mask in [('allwrong',[k==0 for k in ks]),('mixed',y),('allcorrect',[k==5 for k in ks])]:
  ix=[i for i,a in enumerate(mask) if a];count=sum(pr[i] for i in ix);per[name]=dict(questions=len(ix),predicted_Mixed=count,false_positive_rate=count/len(ix) if ix and name!='mixed' else None)
 return dict(questions=len(rows),confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),Mixed_TPR=tpr,nonMixed_TNR=tnr,balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,AUC=auc(rows),classwise=per)
def rankings(rows):
 chosen=sorted(rows,key=lambda r:(-r['score'],str(r['qid'])))[:512]
 return dict(AUC=auc(rows),Top512_Mixed=sum(0<sum(r['strict_em'])<5 for r in chosen),Top512_K=[sum(sum(r['strict_em'])==k for r in chosen) for k in range(6)],Top512_ordered_qids_sha256=digest([r['qid'] for r in chosen]))

def run(config_path,output):
 helper=Path(__file__).with_name('direct_count_label_prior.py');p=load_module('exact_prior_shared_helpers',helper);base=load_module('frozen_ownG5_shared_helpers',Path(__file__).with_name('e2_e4_fixed_ensemble.py'))
 cfg,config_ref=p.load(config_path);output=Path(output);output.mkdir(exist_ok=True)
 if cfg['role']!='FULL938_TRAIN_PRIOR_TWO_FIXED_DECISION_DIAGNOSTIC' or cfg['rules']!=['PRIMARY_FIXED_0.5','DIAGNOSTIC_TRAIN_ONLY_MIXED_PRIOR'] or cfg['threshold_sweep']is not False or cfg['new_GPU_or_model_scoring']is not False:raise ValueError('Only two predeclared decisions, no new scoring')
 lock=(output/'audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (output/'summary.json').exists():return p.checked(p.proof(output/'summary.json'))
 started=time.time();c,cref=p.load(p.TARGET/'config.json');pub,pref=p.load(p.TARGET/'source-publication.json')
 if cref['sha256']!=p.CONFIG_SHA or pub['commit']!=p.RUNTIME or c['grouping']!=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise') or c['global_questions']!=32 or c['fit_questions']!=30016 or c['update_budget']!=938:raise ValueError('Exact original30016/938/current factory binding required')
 plan=p.checked(c['plan']);bank=p.checked(c['g10_complete']);groups=p.checked(bank['outputs']['groups.json'])['train'];lookup={g['qid']:g for g in groups}
 if len(lookup)!=30016 or len(plan['batches'])!=938 or len(plan['batch_epochs'])!=938:raise ValueError('Full real938 data/epoch plan required')
 src=Path(pub['code_root'])/p.GROUPING;factory_ref=p.proof(src)
 if factory_ref['sha256']!=pub['files'][p.GROUPING]['sha256'] or c['code_hashes'][p.GROUPING]!=factory_ref['sha256']:raise ValueError('Actual original immutable grouping source differs')
 factory=load_module('actual_frozen938_group_factory',src);hist=[0]*6;bindings=[];seen=set();progress=[]
 for i,(qids,epoch) in enumerate(zip(plan['batches'],plan['batch_epochs'])):
  if len(qids)!=32 or len(set(qids))!=32 or seen.intersection(qids):raise ValueError('Historical938 must cover distinct original30016 once')
  seen.update(qids);own=[lookup[q] for q in qids];views=factory.build_training_views(own,role='train',epoch=epoch,**c['grouping'])
  if len(views['groups'])!=128:raise ValueError('Exactly four G5 views/original question required')
  for v in views['groups']:
   if v['weight_numerator']!=1 or v['weight_denominator']!=4 or len(v['trajectory_ids'])!=5 or len(v['labels'])!=5 or any(type(y)is not int or y not in (0,1) for y in v['labels']):raise ValueError('Authentic own weighted G5 labels required')
   hist[sum(v['labels'])]+=1
  summary=views['summary'];bindings.append(dict(cursor=i+1,epoch=epoch,selected_views_sha256=summary['selected_views_sha256'],source_groups_sha256=summary['source_groups_sha256'],qid_sha256=summary['qid_sha256']))
  if (i+1)%128==0 or i==937:
   item=dict(stage='ACTUAL_FULL938_TRAIN_VIEW_LABELS_ADVANCING',cursor=i+1,questions=len(seen),views=sum(hist),epoch=time.time());p.atomic(output/'progress.json',item);progress.append(item);print(json.dumps(item),flush=True)
 if len(seen)!=30016 or sum(hist)!=120064 or seen!=set(lookup):raise ValueError('Full original30016 four-view denominator differs')
 prior=sum(hist[1:5])/sum(hist)
 if not 0<prior<1:raise ValueError('Degenerate train-only prior')
 # Store numeric calibration source before opening existing validation outputs.
 prior_blob=dict(role='FULL_ORIGINAL30016_TRAIN_ONLY_EXACT_FOUR_VIEW_MIXED_PRIOR',config=config_ref,training_config=cref,source_publication=pref,groups=bank['outputs']['groups.json'],plan=c['plan'],factory=factory_ref,view_bindings_sha256=digest(bindings),ordered_actual_own_G10_IDs_sha256=digest([lookup[q]['trajectory_ids'] for batch in plan['batches'] for q in batch]),questions=30016,views=120064,view_weight=.25,K0_to5=hist,Mixed_prior=prior,computed_before_validation_predictions_opened=True,not_a_claim_all938_optimizer_updates_completed=True)
 p.atomic(output/'train-only-prior-private.json',prior_blob);prior_ref=p.proof(output/'train-only-prior-private.json')
 val=base.read(base.checked(cfg['validation_complete']))
 if val['status']!='NEW_VALIDATION2048_FIXED_G5_QUALIFIED_HF_REPLAY_LAYER26_COMPLETE' or val['fixed_G5_sha256']!=base.FIXED or val['source_runtime']!='huggingface_frozen_causal_prefill' or val['questions']!=2048 or val['new_G5_selected']is not False or val['final_pool_read']is not False:raise ValueError('Exact existing sameHF viewed validation only')
 fixed=base.frozen_truth(val);results=[]
 for source in cfg['sources']:
  path=base.checked(source['predictions'])
  with gzip.open(path,'rt') as f:rows=[json.loads(x) for x in f]
  if len(rows)!=2048 or len({r['qid'] for r in rows})!=2048:raise ValueError('Exact existing2048 score coverage required')
  for r,g in zip(rows,fixed):
   if any(r[k]!=g[k] for k in ('qid','trajectory_ids','strict_em')):raise ValueError('Existing q own IDs/EM mismatch')
   q=r['predicted_mixed']
   if not math.isfinite(q) or not 0<=q<=1:raise ValueError('Finite existing q required')
   if source['kind']=='product_Bernoulli':
    ps=r['p_j']
    if len(ps)!=5 or any(not math.isfinite(x) or not 0<=x<=1 for x in ps) or abs(q-base.qscore(ps))>=1e-6:raise ValueError('Frozen original product-q formula differs')
   elif source['kind']=='hidden_direct_count6':
    cp=r['count_probabilities']
    if r['score_kind']!='hidden_direct_count6' or len(cp)!=6 or any(not math.isfinite(x) or not 0<=x<=1 for x in cp) or abs(sum(cp)-1)>1e-5 or abs(sum(cp[1:5])-q)>1e-6:raise ValueError('Frozen direct-count q differs')
   else:raise ValueError('No extra score types')
   r['score']=q
  primary=metrics(rows,.5);diagnostic=metrics(rows,prior);rank=rankings(rows)
  if primary['AUC']!=diagnostic['AUC'] or primary['AUC']!=rank['AUC']:raise ValueError('Decision rule changed original ranking/AUC')
  results.append(dict(name=source['name'],predictions=source['predictions'],questions=2048,PRIMARY_FIXED_0_5=primary,DIAGNOSTIC_TRAIN_ONLY_MIXED_PRIOR=diagnostic,original_q_rankings=rank,score_type=source['kind'],new_model_scoring_performed=False))
  print(json.dumps(dict(stage='ACTUAL_EXISTING_PREDICTIONS_TWO_DECISIONS_ACCEPTED',sources=len(results),name=source['name'],epoch=time.time())),flush=True)
 result=dict(status='FULL938_TRAIN_ONLY_PRIOR_TWO_DECISION_DIAGNOSTIC_COMPLETE',config=config_ref,prior_private_receipt=prior_ref,source_training_config=cref,source_plan=c['plan'],source_factory=factory_ref,original_questions=30016,original_views=120064,view_weight=.25,view_bindings_sha256=prior_blob['view_bindings_sha256'],ordered_actual_own_G10_IDs_sha256=prior_blob['ordered_actual_own_G10_IDs_sha256'],actual_training_view_progress=progress,validation_complete=cfg['validation_complete'],fixed_own_G5_sha256=base.FIXED,results=results,numeric_prior_value_exported=False,primary_decision_and_goal_gate_unchanged=True,diagnostic_is_not_a_new_model_improvement=True,thresholds_scanned_or_external_fitted=False,new_GPU_or_model_scoring=False,model_tensor_or_feature_payload_read=False,final4096_outcomes_read=False,elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,epoch=time.time())
 p.atomic(output/'summary.json',result);return result

def main():
 a=argparse.ArgumentParser();a.add_argument('--config',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120));result=run(args.config,args.output);print(json.dumps(dict(status=result['status'],summary=str(args.output/'summary.json'),models=len(result['results']))),flush=True)
if __name__=='__main__':main()
