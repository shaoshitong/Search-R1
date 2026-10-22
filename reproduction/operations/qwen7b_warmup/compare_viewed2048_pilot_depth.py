"""Matched existing viewed2048 scores only; no inference, final data or tuning.

Compare accepted E1 raw/EMA with completed L21/P20 and L27/P20 E3 pilots.
The two pilots share8192 training questions, epochs, groups, seeds, optimizer,
head and learning rate, allowing a matched depth comparison in this pilot.
E1 used a different data/budget; E1 comparisons do not isolate depth or LR.
"""
import fcntl,gzip,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021'
ORIGINAL=A/'joint-mixed-direct-official-169028-b128-v1'
SUITE=A/'online-mixed-layer-prefix-20261022-2gpu-v1'
CPU=A/'online-mixed-pilot-CPU-acceptance-20261022-v1'
OUT=A/'viewed2048-pilot-depth-comparison-20261022-v1'
E1_EVALUATION_SHA='3d832d00b2d40eaf5ffad4568c1785c84270b0a81e033c57e6c7030a93cd0de8'
BOOTSTRAP_REPEATS=2000
BOOTSTRAP_SEED=2026102213


def read(p):return json.loads(Path(p).read_bytes())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for z in iter(lambda:f.read(8*1024**2),b''):h.update(z)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 actual=proof(ref['path'])
 if any(actual[k]!=ref[k]for k in('path','bytes','sha256')):raise ValueError('Viewed prediction/receipt SHA differs')
 return Path(ref['path'])
def atomic(p,v):
 p=Path(p);q=p.with_name(p.name+'.%d.tmp'%os.getpid())
 with q.open('wb')as f:f.write(canonical(v)+b'\n');f.flush();os.fsync(f.fileno())
 q.replace(p)
def gzread(p):
 with gzip.open(p,'rt')as f:return [json.loads(line)for line in f]


def identity(rows):
 rows=sorted(rows,key=lambda r:r['qid'])
 if len(rows)!=2048 or len({r['qid']for r in rows})!=2048:raise ValueError('Actual intact viewed2048 questions required')
 ids=[dict(qid=r['qid'],trajectory_ids=r['trajectory_ids'])for r in rows]
 tids=[t for r in rows for t in r['trajectory_ids']]
 if len(tids)!=10240 or len(set(tids))!=10240:raise ValueError('Own genuine fixedG5 identities required')
 return rows,ids


def auc_layout(y,logits):
 order=np.argsort(logits,kind='stable');z=logits[order]
 starts=np.r_[0,np.flatnonzero(z[1:]!=z[:-1])+1]
 return order,starts,y[order]


def weighted_auc(layout,counts):
 """Mann-Whitney AUC with half-credit exact ties; bootstrap question counts."""
 order,starts,positive=layout;w=counts[order]
 pos=np.add.reduceat(w*positive,starts);neg=np.add.reduceat(w*(~positive),starts)
 npos=pos.sum();nneg=neg.sum()
 if npos==0 or nneg==0:raise ValueError('Both Mixed classes required for AUC')
 return float((pos*(np.cumsum(neg)-.5*neg)).sum()/(npos*nneg))


def metrics(labels,logits,qids):
 y=(labels.sum(1)>0)&(labels.sum(1)<5);p=logits>=0
 tp=int((p&y).sum());fn=int((~p&y).sum());tn=int((~p&~y).sum());fp=int((p&~y).sum())
 top=sorted(range(len(qids)),key=lambda i:(-float(logits[i]),qids[i]))[:512]
 probability=1/(1+np.exp(-np.clip(logits,-700,700)))
 loss=np.logaddexp(0.,logits)-y*logits;types={}
 for name,mask in [('all_wrong',labels.sum(1)==0),('Mixed',y),('all_correct',labels.sum(1)==5)]:
  n=int(mask.sum());types[name]=dict(questions=n,predicted_Mixed=int(p[mask].sum()),predicted_nonMixed=int((~p[mask]).sum()),
   predicted_Mixed_fraction=float(p[mask].mean())if n else None,BCE=float(loss[mask].mean())if n else None,
   false_positive_rate=float(p[mask].mean())if name!='Mixed'and n else None,
   false_negative_rate=float((~p[mask]).mean())if name=='Mixed'and n else None,
   mean_predicted_Mixed_probability=float(probability[mask].mean())if n else None,
   probability_quantiles_10_25_50_75_90=np.quantile(probability[mask],[.1,.25,.5,.75,.9]).tolist()if n else None)
 bins=np.minimum((probability*10).astype(int),9);calibration=[];ece=0.
 for index in range(10):
  mask=bins==index;n=int(mask.sum());mean=float(probability[mask].mean())if n else None;truth=float(y[mask].mean())if n else None
  if n:ece+=n/len(y)*abs(mean-truth)
  calibration.append(dict(bin=index,lower=index/10,upper=(index+1)/10,questions=n,mean_predicted_Mixed=mean,actual_Mixed_fraction=truth))
 return dict(questions=len(y),Mixed_questions=int(y.sum()),nonMixed_questions=int((~y).sum()),BCE=float(loss.mean()),
  Mixed_balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)),Mixed_ROC_AUC=weighted_auc(auc_layout(y,logits),np.ones(len(y),dtype=np.int64)),
  Mixed_recall=tp/(tp+fn),nonMixed_false_positive_rate=fp/(fp+tn),
  confusion=dict(tp=tp,tn=tn,fp=fp,fn=fn),classwise=types,
  Top512=dict(questions=512,Mixed=int(y[top].sum()),Mixed_fraction=float(y[top].mean()),
   selected_qid_sha256=digest([qids[i]for i in top])),threshold=.5,labels_used_for_ranking=False,
  score_and_calibration=dict(observed_Mixed_prevalence=float(y.mean()),mean_predicted_Mixed_probability=float(probability.mean()),
   mean_logit=float(logits.mean()),probability_quantiles_10_25_50_75_90=np.quantile(probability,[.1,.25,.5,.75,.9]).tolist(),
   Brier_score=float(((probability-y)**2).mean()),ECE10_equal_width=ece,fixed_probability_bins=calibration,
   calibration_fitted=False,threshold_tuned=False))


def run():
 if os.uname().sysname!='Linux'or os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise RuntimeError('CPU-only existing viewed predictions analysis required')
 OUT.mkdir(exist_ok=True);guard=(OUT/'analysis.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=time.time();ep=ORIGINAL/'evaluations/epoch-01/complete.json'
 if proof(ep)['sha256']!=E1_EVALUATION_SHA:raise ValueError('Exact original E1 completed evaluation required')
 e1=read(ep)
 if e1['checkpoint']['cursor']!=2577 or e1['final4096_scored']is not False:raise ValueError('Wrong E1 checkpoint or viewed-only provenance')
 sources={'E1_complete':proof(ep)};methods={};reference_ids=None;reference_labels=None;stored={}
 for state in('raw','EMA'):
  value=e1['metrics'][state]['fixed_selection2048'];rows= gzread(checked(value['predictions']));rows,ids=identity(rows)
  ys=np.array([r['strict_em']for r in rows],dtype=np.int64)
  if ys.shape!=(2048,5)or not np.isin(ys,[0,1]).all():raise ValueError('Real own strict EM5 required')
  if any(r['true_mixed']!=(0<sum(r['strict_em'])<5)for r in rows):raise ValueError('E1 own labels disagree')
  if reference_ids is None:reference_ids,reference_labels=ids,ys
  if ids!=reference_ids or not np.array_equal(ys,reference_labels):raise ValueError('E1 raw/EMA fixed trajectory identities differ')
  probability=np.array([r['predicted_mixed']for r in rows],dtype=np.float64)
  if not np.isfinite(probability).all()or(probability<0).any()or(probability>1).any():raise ValueError('Finite original direct Mixed probability required')
  prob=np.clip(probability,1e-12,1-1e-12);logits=np.log(prob)-np.log1p(-prob)
  methods['E1_'+state]=logits;sources['E1_'+state]=value['predictions'];stored['E1_'+state]=value.get('binaryMixedBCE')
 for layer in(27,21):
  name='layer%d-prefix20'%layer;root=SUITE/name;accepted=read(CPU/name/'accepted.json');evaluation=read(root/'evaluation-epoch-03.json')
  if (accepted['status']!='COMPLETED_ONLINE_PILOT_CPU_SHA_MODEL_ADAM_TWO_RANK_RNG_ACCEPTED'or accepted['cursor']!=192
      or accepted['completed_epoch']!=3 or evaluation['epoch']!=3 or evaluation['cursor']!=192
      or evaluation['layer']!=layer or evaluation['prefix']!=20 or evaluation['final4096_scored']is not False):
   raise ValueError('Completed independently CPU-accepted E3 pilot required')
  pp=root/'epoch-03-fixed_validation2048-predictions.jsonl.gz';rows,ids=identity(gzread(pp));ys=np.array([r['labels']for r in rows],dtype=np.int64)
  if ids!=reference_ids or not np.array_equal(ys,reference_labels):raise ValueError('Pilot and E1 question/ownG5/strict EM are not identical')
  if any(r['true_mixed']!=(0<sum(r['labels'])<5)for r in rows):raise ValueError('Pilot own Mixed labels differ')
  sources[name]=dict(predictions=proof(pp),evaluation=proof(root/'evaluation-epoch-03.json'),CPU_acceptance=proof(CPU/name/'accepted.json'),checkpoint=accepted['checkpoint'])
  for state in('raw','EMA'):
   tag='L%dP20_E3_%s'%(layer,state);methods[tag]=np.array([r[state+'_logit']for r in rows],dtype=np.float64)
   if not np.isfinite(methods[tag]).all():raise ValueError('Finite actual pilot logits required')
   stored[tag]=evaluation['metrics']['fixed_validation2048'][state]['binary_Mixed_BCE']
 qids=[r['qid']for r in reference_ids];labels=reference_labels;y=(labels.sum(1)>0)&(labels.sum(1)<5)
 summary={name:metrics(labels,z,qids)for name,z in methods.items()}
 for name,value in summary.items():
  value['stored_original_GPU_BCE']=stored[name]
  if stored[name]is not None and abs(value['BCE']-stored[name])>2e-6:raise ValueError('Recomputed BCE disagrees with original actual GPU evaluation')
 names=list(methods);scores=np.column_stack([methods[name]for name in names]);index={name:i for i,name in enumerate(names)}
 pairs=[]
 for candidate in[n for n in names if n.startswith('L')]:
  for baseline in('E1_raw','E1_EMA'):pairs.append((candidate,baseline))
 for state in('raw','EMA'):pairs.append(('L21P20_E3_'+state,'L27P20_E3_'+state))
 layouts=[auc_layout(y,scores[:,j])for j in range(len(names))]
 rng=np.random.default_rng(BOOTSTRAP_SEED);ba_samples=[];top_samples=[];auc_samples=[]
 for sample in range(BOOTSTRAP_REPEATS):
  ids=rng.integers(0,2048,size=2048);yy=y[ids];zz=scores[ids];pred=zz>=0
  if not yy.any()or yy.all():raise ValueError('Unexpected one-class bootstrap resample')
  ba_samples.append(.5*(pred[yy].mean(0)+(~pred[~yy]).mean(0)))
  ranks=np.argsort(-zz,axis=0,kind='stable')[:512]
  top_samples.append(np.take_along_axis(np.broadcast_to(yy[:,None],zz.shape),ranks,axis=0).mean(0))
  counts=np.bincount(ids,minlength=2048);auc_samples.append([weighted_auc(layout,counts)for layout in layouts])
 ba=np.array(ba_samples);top=np.array(top_samples);aucs=np.array(auc_samples);comparisons={}
 for candidate,baseline in pairs:
  a,b=index[candidate],index[baseline];delta_ba=summary[candidate]['Mixed_balanced_accuracy']-summary[baseline]['Mixed_balanced_accuracy']
  delta_top=summary[candidate]['Top512']['Mixed_fraction']-summary[baseline]['Top512']['Mixed_fraction']
  ccorrect=(scores[:,a]>=0)==y;bcorrect=(scores[:,b]>=0)==y
  comparisons[candidate+'_minus_'+baseline]=dict(candidate=candidate,baseline=baseline,
   Mixed_balanced_accuracy_difference=delta_ba,Mixed_balanced_accuracy_percentile95=np.quantile(ba[:,a]-ba[:,b],[.025,.975]).tolist(),
   Top512_true_Mixed_fraction_difference=delta_top,Top512_true_Mixed_fraction_percentile95=np.quantile(top[:,a]-top[:,b],[.025,.975]).tolist(),
   BCE_difference=summary[candidate]['BCE']-summary[baseline]['BCE'],
   Mixed_ROC_AUC_difference=summary[candidate]['Mixed_ROC_AUC']-summary[baseline]['Mixed_ROC_AUC'],
   Mixed_ROC_AUC_percentile95=np.quantile(aucs[:,a]-aucs[:,b],[.025,.975]).tolist(),
   Brier_score_difference=summary[candidate]['score_and_calibration']['Brier_score']-summary[baseline]['score_and_calibration']['Brier_score'],
   ECE10_difference=summary[candidate]['score_and_calibration']['ECE10_equal_width']-summary[baseline]['score_and_calibration']['ECE10_equal_width'],
   paired_correctness=dict(candidate_only=int((ccorrect&~bcorrect).sum()),baseline_only=int((~ccorrect&bcorrect).sum()),both_correct=int((ccorrect&bcorrect).sum()),both_wrong=int((~ccorrect&~bcorrect).sum())),
   scope='matched_pilot_depth_with_same8192_budget'if baseline.startswith('L')else'different_E1_training_budget_and_data_recipe_not_LR_or_depth_causality')
 receipt=dict(status='EXISTING_VIEWED2048_MATCHED_E1_AND_PILOT_DEPTH_PAIRED_ANALYSIS_COMPLETE',questions=2048,
  own_G5_trajectory_IDs_and_order_exactly_identical=True,own_strict_EM_exactly_identical=True,
  question_and_fixed_trajectory_identity_sha256=digest(reference_ids),labels_sha256=digest(labels.tolist()),
  metrics=summary,comparisons=comparisons,source_proofs=sources,
  bootstrap=dict(repeats=BOOTSTRAP_REPEATS,seed=BOOTSTRAP_SEED,unit='question',paired_same_resampled_questions_all_methods=True,
   reselect_Top512_within_resample=True,interval='percentile95',limitation='Viewed pool; exploratory selection evidence, not untouched confirmation or training-seed uncertainty'),
  E1_probability_reconstructed_BCE_clipping=[1e-12,1-1e-12],stored_original_GPU_BCE_also_reported=True,
  no_new_inference=True,no_model_or_feature_tensor_read=True,no_final4096_read=True,candidate_frozen=False,
  threshold_fitted=False,actor_updates=0,probe_updates=0,seconds=time.time()-started,time=time.time())
 atomic(OUT/'complete.json',receipt)
 print(json.dumps(dict(receipt=proof(OUT/'complete.json'),metrics=summary,comparisons=comparisons,
  question_and_fixed_trajectory_identity_sha256=receipt['question_and_fixed_trajectory_identity_sha256'],
  own_strict_EM_exactly_identical=True,no_final4096_read=True,seconds=receipt['seconds']),allow_nan=False))
 return receipt


if __name__=='__main__':run()
