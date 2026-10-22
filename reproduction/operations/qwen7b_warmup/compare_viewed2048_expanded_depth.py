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
OUT=A/'viewed2048-expanded-depth-comparison-20261022-v1'
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
 if os.uname().sysname!='Linux'or os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise RuntimeError('CPU-only existing viewed scores required')
 OUT.mkdir(exist_ok=True);guard=(OUT/'analysis.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.time()
 epath=ORIGINAL/'evaluations/epoch-01/complete.json';assert proof(epath)['sha256']==E1_EVALUATION_SHA;e1=read(epath);assert e1['checkpoint']['cursor']==2577 and not e1['final4096_scored']
 refids=None;refys=None;methods={};sources={'E1_complete':proof(epath)};baselines={}
 for state in('raw','EMA'):
  value=e1['metrics'][state]['fixed_selection2048'];rows,ids=identity(gzread(checked(value['predictions'])));ys=np.array([r['strict_em']for r in rows],dtype=np.int64)
  if refids is None:refids,refys=ids,ys
  assert ids==refids and np.array_equal(ys,refys)
  prob=np.clip(np.array([r['predicted_mixed']for r in rows]),1e-12,1-1e-12);z=np.log(prob)-np.log1p(-prob);methods['E1_'+state]=z;baselines['E1_'+state]=metrics(ys,z,[r['qid']for r in ids]);sources['E1_'+state]=value['predictions']
 layers={}
 for layer in(15,18,21,24,27):
  root=SUITE/('layer%d-prefix20'%layer);complete=read(root/'complete.json')if(root/'complete.json').exists()else None;result={'layer':layer,'prefix':20,'complete':complete,'epochs':{}}
  if complete:assert complete['status']=='MATCHED_ONLINE_MIXED_LAYER_PREFIX_PILOT_COMPLETE'and complete['cursor']==192
  for epoch in range(4):
   ev=root/('evaluation-epoch-%02d.json'%epoch)
   if not ev.exists():continue
   evaluation=read(ev);assert evaluation['layer']==layer and evaluation['prefix']==20 and not evaluation['final4096_scored'];roles={}
   for role in('fixed_train2048','fixed_validation2048'):
    path=root/('epoch-%02d-%s-predictions.jsonl.gz'%(epoch,role));rows,ids=identity(gzread(path));ys=np.array([r['labels']for r in rows],dtype=np.int64);assert ys.shape==(2048,5)and all(r['true_mixed']==(0<sum(r['labels'])<5)for r in rows)
    if role=='fixed_validation2048':assert ids==refids and np.array_equal(ys,refys)
    qids=[r['qid']for r in ids];values={}
    for state in('raw','EMA'):
     z=np.array([r[state+'_logit']for r in rows],dtype=np.float64);assert np.isfinite(z).all();values[state]=metrics(ys,z,qids);stored=evaluation['metrics'][role][state]['binary_Mixed_BCE'];assert abs(values[state]['BCE']-stored)<2e-6;values[state]['stored_original_GPU_BCE']=stored
     if role=='fixed_validation2048'and epoch==3 and complete:methods['L%dP20_E3_%s'%(layer,state)]=z
    roles[role]=values;sources['L%d_E%d_%s'%(layer,epoch,role)]=proof(path)
   result['epochs'][str(epoch)]={'cursor':evaluation['cursor'],'evaluation_source':proof(ev),'roles':roles}
  result['checkpoint_latest_as_recorded']=read(root/'latest.json')if(root/'latest.json').exists()else None
  result['new_case_model_Adam_RNG_CPU_load_acceptance_claimed']=False if layer in(15,18,24)else 'see separately accepted source receipt'
  layers[str(layer)]=result
 names=list(methods);scores=np.column_stack([methods[n]for n in names]);ys=refys;y=(ys.sum(1)>0)&(ys.sum(1)<5);qids=[r['qid']for r in refids];summaries={n:metrics(ys,z,qids)for n,z in methods.items()};layouts=[auc_layout(y,scores[:,j])for j in range(len(names))];rng=np.random.default_rng(BOOTSTRAP_SEED);bs=[];ts=[];asamples=[]
 for _ in range(BOOTSTRAP_REPEATS):
  idx=rng.integers(0,2048,size=2048);yy=y[idx];zz=scores[idx];pred=zz>=0;assert yy.any()and not yy.all();bs.append(.5*(pred[yy].mean(0)+(~pred[~yy]).mean(0)));ranks=np.argsort(-zz,axis=0,kind='stable')[:512];ts.append(np.take_along_axis(np.broadcast_to(yy[:,None],zz.shape),ranks,axis=0).mean(0));counts=np.bincount(idx,minlength=2048);asamples.append([weighted_auc(layout,counts)for layout in layouts])
 bs=np.array(bs);ts=np.array(ts);asamples=np.array(asamples);comparisons={}
 for candidate in[n for n in names if n.startswith('L')]:
  for baseline in('E1_EMA','L21P20_E3_EMA','L27P20_E3_EMA'):
   if baseline not in names or candidate==baseline:continue
   a,b=names.index(candidate),names.index(baseline);cm,bm=summaries[candidate],summaries[baseline];comparisons[candidate+'_minus_'+baseline]=dict(candidate=candidate,baseline=baseline,Mixed_balanced_accuracy_difference=cm['Mixed_balanced_accuracy']-bm['Mixed_balanced_accuracy'],Mixed_balanced_accuracy_percentile95=np.quantile(bs[:,a]-bs[:,b],[.025,.975]).tolist(),Top512_true_Mixed_fraction_difference=cm['Top512']['Mixed_fraction']-bm['Top512']['Mixed_fraction'],Top512_true_Mixed_fraction_percentile95=np.quantile(ts[:,a]-ts[:,b],[.025,.975]).tolist(),Mixed_ROC_AUC_difference=cm['Mixed_ROC_AUC']-bm['Mixed_ROC_AUC'],Mixed_ROC_AUC_percentile95=np.quantile(asamples[:,a]-asamples[:,b],[.025,.975]).tolist(),BCE_difference=cm['BCE']-bm['BCE'],scope='same8192_depth_recipe'if baseline.startswith('L')else'different_E1_budget_and_recipe')
 receipt=dict(status='EXPANDED_MATCHED_DEPTH_VIEWED2048_ANALYSIS_COMPLETE'if all(v['complete']for v in layers.values())else'EXPANDED_MATCHED_DEPTH_VIEWED2048_PARTIAL_ACTUAL_RESULTS',layers=layers,E1_baselines=baselines,completed_E3_metrics=summaries,comparisons=comparisons,source_proofs=sources,question_and_fixed_trajectory_identity_sha256=digest(refids),own_strict_EM_exactly_identical=True,bootstrap=dict(repeats=BOOTSTRAP_REPEATS,seed=BOOTSTRAP_SEED,unit='question',interval='percentile95',limitation='Viewed exploratory pool, not untouched confirmation and not training-seed uncertainty'),training_recipe=dict(actual_Mixed_questions=3489,actual_nonMixed_questions=4703,positive_BCE_weight=8192/(2*3489),negative_BCE_weight=8192/(2*4703),all_G10Mixed_assigned_G5Mixed=True,natural_random_G5_control=False),no_final4096_read=True,no_new_inference=True,no_model_or_feature_tensor_read=True,threshold_tuned=False,calibration_fitted=False,candidate_frozen=False,probe_updates=0,seconds=time.time()-started,time=time.time());atomic(OUT/'complete.json',receipt);print(json.dumps(dict(receipt=proof(OUT/'complete.json'),**receipt),allow_nan=False));return receipt

if __name__=='__main__':run()
