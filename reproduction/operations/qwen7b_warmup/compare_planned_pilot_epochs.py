"""All planned pilot epochs, raw and EMA, on existing viewed2048 only.

Run after all seven terminal pilot checkpoints were CPU-accepted. This reads
only completed prediction/metric JSONs; no model or final holdout is opened.
The complete preplanned table prevents selective reporting of only EMA/layers.
"""
import fcntl,gzip,hashlib,importlib.util,json,os,time
from pathlib import Path
import numpy as np
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021';SUITE=A/'online-mixed-layer-prefix-20261022-2gpu-v1'
ACCEPTED=A/'online-mixed-pilot-CPU-acceptance-20261022-v1'
OUT=A/'planned-pilot-all-epochs-viewed2048-20261022-v1'
CELLS=((27,20),(21,20),(27,80),(21,80),(15,20),(18,20),(24,20))
REPEATS=1000;SEED=2026102219


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def atomic(p,v):
 p=Path(p);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,sort_keys=True,allow_nan=False)+'\n');q.replace(p)
def gzread(p):
 with gzip.open(p,'rt')as f:return[json.loads(x)for x in f]


def run(helper):
 if os.uname().sysname!='Linux'or os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise RuntimeError('CPU existing-predictions-only process required')
 spec=importlib.util.spec_from_file_location('viewed_metrics_exact_helper',helper);u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
 OUT.mkdir(exist_ok=True);guard=(OUT/'analysis.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time()
 original=A/'joint-mixed-direct-official-169028-b128-v1/evaluations/epoch-01/complete.json'
 if sha(original)['sha256']!=u.E1_EVALUATION_SHA:raise ValueError('Exact E1 evaluation required')
 e1=read(original);sources=dict(E1=sha(original),helper=sha(helper),pilot_cells=[]);reference=None;labels=None;methods={};metadata={};stored={}
 for state in('raw','EMA'):
  ref=e1['metrics'][state]['fixed_selection2048'];rows,ids=u.identity(gzread(u.checked(ref['predictions'])))
  ys=np.array([r['strict_em']for r in rows],dtype=np.int64)
  if reference is None:reference,labels=ids,ys
  if ids!=reference or not np.array_equal(ys,labels):raise ValueError('E1 raw/EMA ownG5 differs')
  q=np.clip(np.array([r['predicted_mixed']for r in rows]),1e-12,1-1e-12);tag='E1_'+state
  methods[tag]=np.log(q)-np.log1p(-q);metadata[tag]=dict(kind='accepted_original_E1',state=state,layer=27,prefix=20,epoch=1,cursor=2577)
  stored[tag]=ref['binaryMixedBCE']
 for layer,prefix in CELLS:
  variant='layer%d-prefix%d'%(layer,prefix);base=SUITE/variant;accepted=read(ACCEPTED/variant/'accepted.json');complete=read(base/'complete.json')
  if (accepted['status']!='COMPLETED_ONLINE_PILOT_CPU_SHA_MODEL_ADAM_TWO_RANK_RNG_ACCEPTED'or accepted['cursor']!=192
      or complete['cursor']!=192 or complete['epochs']!=3 or complete['layer']!=layer or complete['prefix']!=prefix):raise ValueError('All seven actual completed and independently accepted cells required')
  cell=dict(layer=layer,prefix=prefix,CPU_acceptance=sha(ACCEPTED/variant/'accepted.json'),source_complete=sha(base/'complete.json'),checkpoint=accepted['checkpoint'],evaluations=[])
  for epoch in range(4):
   ep=base/('evaluation-epoch-%02d.json'%epoch);evaluation=read(ep)
   if (evaluation['epoch']!=epoch or evaluation['cursor']!=64*epoch or evaluation['layer']!=layer or evaluation['prefix']!=prefix
       or evaluation['final4096_scored']is not False):raise ValueError('Wrong actual predeclared pilot epoch evaluation')
   pred=base/('epoch-%02d-fixed_validation2048-predictions.jsonl.gz'%epoch);rows,ids=u.identity(gzread(pred));ys=np.array([r['labels']for r in rows],dtype=np.int64)
   if ids!=reference or not np.array_equal(ys,labels)or any(r['true_mixed']!=(0<sum(r['labels'])<5)for r in rows):raise ValueError('All methods must have exactly matching ownG5/strictEM')
   cell['evaluations'].append(dict(epoch=epoch,metrics=sha(ep),predictions=sha(pred)))
   for state in('raw','EMA'):
    tag='L%dP%d_E%d_%s'%(layer,prefix,epoch,state);methods[tag]=np.array([r[state+'_logit']for r in rows],dtype=np.float64)
    if not np.isfinite(methods[tag]).all():raise ValueError('Finite existing true logits required')
    metadata[tag]=dict(kind='planned_matched8192_pilot',state=state,layer=layer,prefix=prefix,epoch=epoch,cursor=64*epoch)
    stored[tag]=evaluation['metrics']['fixed_validation2048'][state]['binary_Mixed_BCE']
  sources['pilot_cells'].append(cell)
 qids=[r['qid']for r in reference];y=(labels.sum(1)>0)&(labels.sum(1)<5);summary={}
 for name,z in methods.items():
  v=u.metrics(labels,z,qids)
  if abs(v['BCE']-stored[name])>2e-6:raise ValueError('Recomputed actual BCE differs from saved evaluation')
  v['stored_actual_GPU_BCE']=stored[name];v['model']=metadata[name]
  # Keep this compact ledger; full old calibration bins already have their own receipt.
  v['score_and_calibration'].pop('fixed_probability_bins');summary[name]=v
 names=list(methods);scores=np.column_stack([methods[k]for k in names]);index={k:i for i,k in enumerate(names)};layouts=[u.auc_layout(y,scores[:,j])for j in range(len(names))]
 rng=np.random.default_rng(SEED);ba=[];top=[];auc=[]
 for i in range(REPEATS):
  ids=rng.integers(0,2048,size=2048);yy=y[ids];zz=scores[ids];pred=zz>=0
  ba.append(.5*(pred[yy].mean(0)+(~pred[~yy]).mean(0)))
  # Original rows are lexicographic qid order. Preserve the same exact tie rule
  # after bootstrap duplication rather than letting random draw order break ties.
  ranks=np.lexsort((np.broadcast_to(ids[:,None],zz.shape),-zz),axis=0)[:512]
  top.append(np.take_along_axis(np.broadcast_to(yy[:,None],zz.shape),ranks,axis=0).mean(0));counts=np.bincount(ids,minlength=2048)
  auc.append([u.weighted_auc(layout,counts)for layout in layouts])
  if i%100==0:atomic(OUT/'progress.json',dict(stage='PAIRED_BOOTSTRAP_EXISTING_VIEWED_PREDICTIONS',replicate=i,replicates=REPEATS,methods=len(names),time=time.time()))
 ba=np.asarray(ba);top=np.asarray(top);auc=np.asarray(auc);comparisons={}
 pairs=[(name,baseline)for name in names if name.startswith('L')for baseline in('E1_raw','E1_EMA')]
 for layer,prefix in CELLS:
  if layer!=27:pairs.extend(('L%dP%d_E3_%s'%(layer,prefix,state),'L27P%d_E3_%s'%(prefix,state))for state in('raw','EMA'))
 pairs.extend([('L18P20_E3_raw','L21P20_E3_raw'),('L18P20_E3_raw','L15P20_E3_raw'),
  ('L21P80_E3_raw','L18P20_E3_raw'),('L21P80_E3_raw','L21P20_E3_raw'),
  ('L21P80_E3_raw','L21P80_E2_raw'),('L21P80_E3_raw','L21P80_E3_EMA')])
 for candidate,baseline in pairs:
  a,b=index[candidate],index[baseline];c0=summary[candidate];b0=summary[baseline]
  comparisons[candidate+'_minus_'+baseline]=dict(candidate=candidate,baseline=baseline,
   BA_difference=c0['Mixed_balanced_accuracy']-b0['Mixed_balanced_accuracy'],BA_percentile95=np.quantile(ba[:,a]-ba[:,b],[.025,.975]).tolist(),
   AUC_difference=c0['Mixed_ROC_AUC']-b0['Mixed_ROC_AUC'],AUC_percentile95=np.quantile(auc[:,a]-auc[:,b],[.025,.975]).tolist(),
   Top512_fraction_difference=c0['Top512']['Mixed_fraction']-b0['Top512']['Mixed_fraction'],Top512_fraction_percentile95=np.quantile(top[:,a]-top[:,b],[.025,.975]).tolist(),
   BCE_difference=c0['BCE']-b0['BCE'])
 rankBA=sorted(names,key=lambda k:(-summary[k]['Mixed_balanced_accuracy'],k));rankAUC=sorted(names,key=lambda k:(-summary[k]['Mixed_ROC_AUC'],k));rankTop=sorted(names,key=lambda k:(-summary[k]['Top512']['Mixed'],k))
 result=dict(status='ALL_SEVEN_PLANNED_PILOTS_EPOCH0_1_2_3_RAW_EMA_VIEWED2048_PAIRED_ANALYSIS_COMPLETE',methods=len(names),
  metrics=summary,comparisons=comparisons,ranking=dict(by_MixedBA=rankBA,by_MixedAUC=rankAUC,by_Top512=rankTop),
  identity=dict(questions=2048,ownG5_trajectories=10240,same_qid_trajectory_order_and_strict_EM=True,identity_SHA256=u.digest(reference),labels_SHA256=u.digest(labels.tolist())),
  source_proofs=sources,bootstrap=dict(repeats=REPEATS,seed=SEED,paired_question_resampling=True,reselected_Top512=True,interval='percentile95_unadjusted_exploratory',training_seed_uncertainty_included=False),
  threshold=.5,threshold_tuned=False,calibration_fitted=False,no_final4096_read=True,candidate_frozen=False,
  no_model_tensor_read=True,no_new_inference=True,no_Slurm_mutation=True,all_preplanned_epochs_reported=True,
  seconds=time.time()-start,time=time.time())
 atomic(OUT/'complete.json',result)
 # Return only compact summaries, never predictions or final data.
 print(json.dumps(dict(receipt=sha(OUT/'complete.json'),identity=result['identity'],metrics=summary,comparisons=comparisons,ranking=result['ranking'],seconds=result['seconds']),allow_nan=False))
 return result


if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--helper',type=Path,required=True);a=p.parse_args();run(a.helper)
