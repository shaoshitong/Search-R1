"""Read-only first256 exact training-view label/prior diagnostic, no torch.

Only accepted training groups, the frozen plan, runtime grouping source and
accepted curve rows are opened. No model/checkpoint/features/evaluation data.
"""
import argparse,fcntl,hashlib,importlib.util,json,math,os,resource,time
from fractions import Fraction
from pathlib import Path

B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
TARGET=B/'algorithm-improvement-20261021/direct-hidden-count30016-train-v2'
CONFIG_SHA='3e910b12c0ff52eac70e61566c6135e2ec188bb47ec64ad79524abbd7a657526'
RUNTIME='501cf80478b77f68c3e13a96d4659e3da0971b2e'
GROUPING='reproduction/operations/qwen7b_warmup/g10_grouping.py'

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def proof(p,raw=None):
 p=Path(p);raw=p.read_bytes()if raw is None else raw
 return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def load(p):
 p=Path(p)
 if not p.is_relative_to(B)or p.resolve()!=p:raise ValueError('Real remote owned metadata required')
 raw=p.read_bytes();return json.loads(raw),proof(p,raw)
def checked(ref):
 value,actual=load(ref['path'])
 if any(actual[k]!=ref[k]for k in ('bytes','sha256')):raise ValueError('Input metadata binding changed')
 return value
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_bytes(canonical(v)+b'\n');t.replace(p)
def entropy(counts):
 n=sum(counts)
 if not n:raise ValueError('Nonempty class counts required')
 return -sum(x/n*math.log(x/n)for x in counts if x)
def prior_loss(counts,prior):return -sum(x/sum(counts)*math.log(prior[k])for k,x in enumerate(counts)if x)
def summarize(records,prior,p_trajectory):
 hist=[sum(r['K_histogram'][k]for r in records)for k in range(6)];n=sum(hist)
 mean=lambda key:sum(r[key]for r in records)/len(records)
 actual=mean('count_NLL');traj=mean('trajectory_BCE');p=sum(k*x for k,x in enumerate(hist))/(5*n)
 tp=sum(r['mixed_recall']*r['mixed_views']for r in records if r['mixed_views'])
 tn=sum(r['nonmixed_recall']*r['nonmixed_views']for r in records if r['nonmixed_views'])
 pos=sum(hist[1:5]);neg=hist[0]+hist[5]
 return dict(first_cursor=records[0]['cursor'],last_cursor=records[-1]['cursor'],updates=len(records),questions=32*len(records),views=n,K_histogram=hist,
  K_frequencies=[x/n for x in hist],true_Mixed_fraction=pos/n,trajectory_correct_fraction=p,
  best_window_constant_count_NLL=entropy(hist),global8192_prior_count_NLL_on_window=prior_loss(hist,prior),uniform_count_NLL=math.log(6),
  actual_mean_count_NLL=actual,count_excess_over_best_window_constant=actual-entropy(hist),count_excess_over_global8192_prior=actual-prior_loss(hist,prior),
  actual_mean_trajectory_BCE=traj,best_window_constant_trajectory_BCE=entropy([5*n-sum(k*x for k,x in enumerate(hist)),sum(k*x for k,x in enumerate(hist))]),
  global8192_prior_trajectory_BCE_on_window=-p*math.log(p_trajectory)-(1-p)*math.log(1-p_trajectory),
  actual_mean_total_loss=mean('loss'),actual_training_Mixed_recall=tp/pos if pos else None,
  actual_training_nonMixed_recall=tn/neg if neg else None,
  actual_training_Mixed_BA=.5*(tp/pos+tn/neg)if pos and neg else None,
  reconstructed_predicted_Mixed_fraction=(tp+neg-tn)/n)

def analyze(output):
 started=time.time();output=Path(output);output.mkdir(exist_ok=True)
 guard=(output/'audit.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (output/'complete.json').exists():return checked(proof(output/'complete.json'))
 c,cref=load(TARGET/'config.json');pub,pref=load(TARGET/'source-publication.json')
 if cref['sha256']!=CONFIG_SHA or pub['commit']!=RUNTIME or c['grouping']!=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise'):raise ValueError('Exact current direct-count branch required')
 if c['global_questions']!=32 or c['fit_questions']!=30016 or c['update_budget']!=938:raise ValueError('Exact B32 frozen938 plan required')
 plan=checked(c['plan']);bank=checked(c['g10_complete']);groupsref=bank['outputs']['groups.json'];groups=checked(groupsref)['train']
 lookup={g['qid']:g for g in groups}
 if len(lookup)!=30016 or len(plan['batches'])!=938 or len(plan['batch_epochs'])!=938:raise ValueError('Actual accepted full training metadata differs')
 source=Path(pub['code_root'])/GROUPING;actual=proof(source)
 if actual['sha256']!=pub['files'][GROUPING]['sha256']:raise ValueError('Original runtime grouping source changed')
 spec=importlib.util.spec_from_file_location('exact_original_g10_training_grouping',source);grouping=importlib.util.module_from_spec(spec);spec.loader.exec_module(grouping)
 # Snapshot curve bytes once: the active trainer may append later accepted rows.
 raw=(TARGET/'curve.jsonl').read_bytes();rows=[]
 for line in raw.splitlines():
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  if 1<=r['cursor']<=256:rows.append(r)
 if [r['cursor']for r in rows]!=list(range(1,257)):raise ValueError('Unique accepted cursor1..256 required')
 atomic(output/'progress.json',dict(stage='ACCEPTED_TRAINING_METADATA_AND_FIRST256_CURVE_LOADED',cursor=0,questions=0,epoch=time.time()))
 records=[];seen=set();view_bindings=[]
 for i,row in enumerate(rows):
  qids=plan['batches'][i];epoch=plan['batch_epochs'][i]
  if len(qids)!=32 or seen.intersection(qids):raise ValueError('First256 must be distinct8192 questions')
  seen.update(qids);own=[lookup[q]for q in qids]
  g=grouping.build_training_views(own,role='train',epoch=epoch,**c['grouping']);views=g['groups'];summary=g['summary']
  if row['steps']!=10464+i+1 or row['head_steps']!=i+1:raise ValueError('Old/new real Adam clocks differ')
  if row['original_questions']!=32 or row['virtual_G5_views']!=128:raise ValueError('Actual training view budget differs')
  for key in ('selected_views_sha256','source_groups_sha256','qid_sha256','epoch','seed','mode','weighted_mixed_mass','weighted_nonmixed_mass'):
   if row['grouping'][key]!=summary[key]:raise ValueError('Exact training grouping replay differs: '+key)
  hist=[0]*6
  for v in views:
   if Fraction(v['weight_numerator'],v['weight_denominator'])!=Fraction(1,4):raise ValueError('Every actual view must have weight1/4')
   hist[sum(v['labels'])]+=1
  if sum(hist)!=128 or any(not math.isfinite(row[k])for k in ('count_NLL','trajectory_BCE','loss'))or abs(row['loss']-row['count_NLL']-row['trajectory_BCE'])>2e-5:raise ValueError('Loss component denominator/finite check differs')
  records.append(dict(cursor=i+1,K_histogram=hist,mixed_views=sum(hist[1:5]),nonmixed_views=hist[0]+hist[5],**{k:row[k]for k in ('count_NLL','trajectory_BCE','loss','mixed_recall','nonmixed_recall')}))
  view_bindings.append(dict(cursor=i+1,epoch=epoch,selected_views_sha256=summary['selected_views_sha256']))
  if (i+1)%32==0:atomic(output/'progress.json',dict(stage='EXACT_ORIGINAL_RANDOM4_G5_LABELS_REPLAYED',cursor=i+1,questions=len(seen),epoch=time.time()))
 hist=[sum(r['K_histogram'][k]for r in records)for k in range(6)];n=sum(hist);prior=[x/n for x in hist];pt=sum(k*x for k,x in enumerate(hist))/(5*n)
 result=dict(status='DIRECT_COUNT256_EXACT_TRAINING_VIEW_LABEL_PRIOR_AND_CURVE_ACCEPTED',source_config=cref,source_publication=pref,runtime_commit=RUNTIME,exact_grouping_source=actual,groups=groupsref,plan=c['plan'],
  curve_first256_canonical_sha256=hashlib.sha256(canonical(rows)).hexdigest(),exact256_grouping_bindings_sha256=hashlib.sha256(canonical(view_bindings)).hexdigest(),
  questions=len(seen),original_question_weight=1,views=32768,view_weight=dict(numerator=1,denominator=4),loss_reduction='mean_question(sum_four_views(weight1/4 * [mean5_trajectoryBCE + sixclass_countNLL]))',
  K_histogram=hist,K_frequencies=prior,Mixed_prior=sum(prior[1:5]),best_constant_count_NLL_entropy=entropy(hist),trajectory_correct_prior=pt,best_constant_trajectory_BCE_entropy=entropy([5*n-sum(k*x for k,x in enumerate(hist)),sum(k*x for k,x in enumerate(hist))]),
  overall=summarize(records,prior,pt),windows32=[summarize(records[j:j+32],prior,pt)for j in range(0,256,32)],windows64=[summarize(records[j:j+64],prior,pt)for j in range(0,256,64)],
  empirical_constant_baselines_are_posthoc_training_diagnostics_not_candidates=True,losses_are_online_raw_preupdate_not_a_frozen_EMA_training_evaluation=True,
  conclusion_limit='An online mean near the empirical constant entropy is compatible with fitting class priors, not proof of constant per-question scores; curve does not expose full count vectors. Training TPR/TNR are raw preupdate, unlike external EMA validation.',
  model_tensors_loaded=False,features_read=False,evaluation_payload_read=False,final_pool_read=False,GPU_used=False,training_modified=False,
  elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 atomic(output/'complete.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test:
  assert abs(entropy([1]*6)-math.log(6))<1e-14 and entropy([8,0,0,0,0,0])==0
  assert abs(prior_loss([1,1,0,0,0,0],[.5,.5,0,0,0,0])-math.log(2))<1e-14
  print('Two constant-loss fixture checks passed');return
 if a.output is None or a.output.parent!=B/'algorithm-improvement-20261021':p.error('Separate remote diagnostic directory required')
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120))
 r=analyze(a.output);print(json.dumps(dict(status=r['status'],receipt=proof(a.output/'complete.json'),result=r)))
if __name__=='__main__':main()
