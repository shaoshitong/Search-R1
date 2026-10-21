"""Two already-frozen decisions plus descriptive fixed bins, no model fitting.

The numeric original full938 training prior is consumed only remotely. Nothing
selects a threshold from these already-viewed recipe-validation statistics.
"""
import argparse,fcntl,gzip,importlib.util,json,math,os,resource,time
from pathlib import Path


def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def describe(values):
 v=sorted(values);n=len(v)
 if not n:return dict(n=0)
 def q(p):
  x=(n-1)*p;i=int(x);j=min(i+1,n-1);return v[i]+(v[j]-v[i])*(x-i)
 mean=sum(v)/n
 return dict(n=n,mean=mean,std=math.sqrt(sum((x-mean)**2 for x in v)/n),min=v[0],q05=q(.05),q25=q(.25),median=q(.5),q75=q(.75),q95=q(.95),max=v[-1])


def descriptive(rows):
 bins=[];byk=[];predicted=[0]*6;mean_distribution=[0.]*6;confusion=[[0]*6 for _ in range(6)]
 for r in rows:
  cp=r['count_probabilities'];k=sum(r['strict_em']);guess=max(range(6),key=lambda i:cp[i]);predicted[guess]+=1;confusion[k][guess]+=1
  for i in range(6):mean_distribution[i]+=cp[i]/len(rows)
 for i in range(10):
  subset=[r for r in rows if min(9,int(r['score']*10))==i]
  bins.append(dict(index=i,lower_inclusive=i/10,upper=(i+1)/10,upper_inclusive=i==9,questions=len(subset),true_K_histogram=[sum(sum(r['strict_em'])==k for r in subset) for k in range(6)],Mixed_fraction=sum(0<sum(r['strict_em'])<5 for r in subset)/len(subset) if subset else None,q=describe([r['score'] for r in subset])))
 for k in range(6):byk.append(dict(true_K=k,q=describe([r['score'] for r in rows if sum(r['strict_em'])==k]),predicted_argmax_K=confusion[k]))
 return dict(q_all=describe([r['score'] for r in rows]),fixed_ten_equal_width_bins=bins,by_true_K=byk,predicted_argmax_K=predicted,mean_count_probability_distribution=mean_distribution,true_K_vs_predicted_argmax_K=confusion,bin_or_quantile_used_to_choose_threshold=False)


def run(config_path,output):
 started=time.time();cfg=json.loads(Path(config_path).read_text());output=Path(output);output.mkdir(exist_ok=True)
 if cfg['role']!='ALREADY_VIEWED_HEADLR_TWO_FROZEN_DECISION_DIAGNOSTIC' or cfg['rules']!=['PRIMARY_FIXED_0.5','DIAGNOSTIC_TRAIN_ONLY_MIXED_PRIOR'] or cfg['q_bins']!=10 or cfg['threshold_sweep'] is not False:raise ValueError('Only predeclared two decisions and descriptive bins allowed')
 lock=(output/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (output/'summary.json').exists():raise RuntimeError('Immutable existing result; inspect before any repeat')
 source=Path(cfg['prior_code_root']);p=module('frozen_prior_helpers',source/'direct_count_label_prior.py')
 pub=p.checked(cfg['prior_publication'])
 if pub['commit']!='95f28e6f9630ae19d57cddf0be51873a95074664':raise ValueError('Exact frozen95f prior implementation required')
 for name in ('direct_count_prior_decision.py','direct_count_label_prior.py','e2_e4_fixed_ensemble.py'):
  ref=pub['files'][name]
  if p.proof(ref['path'])!=ref:raise ValueError('Frozen helper source SHA changed')
 metrics=module('frozen_two_decision_helpers',source/'direct_count_prior_decision.py');truth=module('frozen_fixedG5_helpers',source/'e2_e4_fixed_ensemble.py')
 prior=p.checked(cfg['prior_private_receipt']);old=p.checked(cfg['prior_summary'])
 if (prior['computed_before_validation_predictions_opened'] is not True or prior['questions']!=30016 or prior['views']!=120064 or prior['view_weight']!=.25 or old['prior_private_receipt']!=cfg['prior_private_receipt']):raise ValueError('Previously frozen train-only calibration source differs')
 threshold=prior['Mixed_prior']
 if not 0<threshold<1:raise ValueError('Invalid frozen private threshold')
 val=p.checked(cfg['validation_complete'])
 if val['source_runtime']!='huggingface_frozen_causal_prefill' or val['fixed_G5_sha256']!=truth.FIXED or val['questions']!=2048 or val['final_pool_read'] is not False:raise ValueError('Only accepted sameHF already-viewed2048')
 fixed=truth.frozen_truth(val);results=[]
 for src in cfg['sources']:
  accepted=p.checked(src['accepted_audit']);evaluation=p.checked(src['evaluation_receipt'])
  if (accepted['cursor']!=src['cursor'] or accepted['evaluation_receipt']!=src['evaluation_receipt'] or accepted['config']!=src['config'] or accepted['predictions']['selection2048FreshG5HF']!=src['predictions'] or accepted['all_nine_states_unchanged_during_evaluation'] is not True or accepted['final_pool_read'] is not False or evaluation['config_sha256']!=src['config']['sha256']):raise ValueError('Existing independent ownID/EM binding differs')
  config=p.checked(src['config']);ev=[e for e in config['evaluations'] if e['name']=='selection2048FreshG5HF']
  if len(ev)!=1 or ev[0]['bank']!=val['source_groups'] or ev[0]['fixed_g5']!=val['source_fixed_G5'] or ev[0]['features']!=val['features'] or config['plan']['sha256']!=prior['plan']['sha256']:raise ValueError('Same frozen backend/plan required')
  with gzip.open(truth.checked(src['predictions']),'rt') as f:rows=[json.loads(x) for x in f]
  if len(rows)!=2048 or len({r['qid'] for r in rows})!=2048:raise ValueError('Exact existing2048 coverage')
  for r,g in zip(rows,fixed):
   if any(r[k]!=g[k] for k in ('qid','trajectory_ids','strict_em')):raise ValueError('Saved own IDs/EM changed')
   cp=r['count_probabilities'];q=r['predicted_mixed']
   if r['score_kind']!='hidden_direct_count6' or len(cp)!=6 or any(not math.isfinite(x) or not 0<=x<=1 for x in cp+[q]) or abs(sum(cp)-1)>1e-5 or abs(sum(cp[1:5])-q)>1e-6:raise ValueError('Saved direct count score changed')
   r['score']=q
  primary=metrics.metrics(rows,.5);diagnostic=metrics.metrics(rows,threshold);ranking=metrics.rankings(rows);bound=accepted['metrics']['selection2048FreshG5HF']
  if (abs(primary['balanced_accuracy']-bound['pool']['mixed_balanced_accuracy'])>1e-12 or abs(primary['AUC']-bound['pool']['mixed_roc_auc'])>1e-12 or ranking['Top512_Mixed']!=bound['selected512']['mixed_questions'] or primary['AUC']!=diagnostic['AUC'] or primary['AUC']!=ranking['AUC']):raise ValueError('Primary scores/rankings differ from accepted receipt')
  results.append(dict(name=src['name'],accepted_audit=src['accepted_audit'],predictions=src['predictions'],PRIMARY_FIXED_0_5=primary,DIAGNOSTIC_TRAIN_ONLY_MIXED_PRIOR=diagnostic,original_q_rankings=ranking,descriptive_only=descriptive(rows)))
  print(json.dumps(dict(stage='EXISTING_OWN_EM_DIRECT_Q_TWO_FIXED_DECISIONS_AND_BINS_ADVANCED',sources=len(results),epoch=time.time())),flush=True)
 summary=dict(status='HEADLR_ACCEPTED256_512_TWO_FROZEN_DECISIONS_AND_DESCRIPTIVE_BINS_COMPLETE',config=p.proof(config_path),prior_private_receipt=cfg['prior_private_receipt'],prior_source_summary=cfg['prior_summary'],prior_publication=cfg['prior_publication'],results=results,numeric_threshold_exported=False,raw_rows_exported=False,primary_fixed_half_and_joint_gate_unchanged=True,threshold_sweep_or_fit=False,new_model_forward=False,model_or_tensor_or_feature_payload_read=False,final4096_read=False,elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 p.atomic(output/'summary.json',summary);return summary


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);v=p.parse_args()
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(60,60))
 result=run(v.config,v.output);print(json.dumps(dict(status=result['status'],sources=len(result['results']),elapsed_seconds=result['elapsed_seconds'])))
