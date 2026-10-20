"""Read sealed predictions remotely; emit aggregate diagnosis only."""
import json,gzip,math,hashlib,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
PH=[(B/'mixed-only-ft-cp2-prefix20-20261020',1),(B/'balanced-rank1-from-epoch1-prefix20-20261020',2)]
def read(p):return json.loads(p.read_text())
def rows(p):
 with gzip.open(str(p),'rt') as f:return [json.loads(x) for x in f]
def quant(v):
 v=sorted(v);return {str(q):v[round((len(v)-1)*q)] for q in [0,.1,.25,.5,.75,.9,1]}
def confusion(rs,t=.5):
 tp=fp=tn=fn=0
 for r in rs:
  y=r['y'];p=r['predicted_mixed']>=t
  tp+=y and p;fp+=(not y) and p;tn+=(not y) and not p;fn+=y and not p
 return dict(TP=tp,FP=fp,TN=tn,FN=fn,TPR=tp/(tp+fn),TNR=tn/(tn+fp),balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)))
def analyze(rs):
 order=sorted(rs,key=lambda r:(-r['predicted_mixed'],r['qid']));P=sum(r['y'] for r in rs);N=len(rs)-P
 # AUC: pairwise exact ties; n=2048 only.
 pos=[r['predicted_mixed'] for r in rs if r['y']];neg=[r['predicted_mixed'] for r in rs if not r['y']]
 auc=sum((p>n)+.5*(p==n) for p in pos for n in neg)/(P*N)
 # AP grouped by ties.
 tp=seen=0;ap=0.;idx=0
 while idx<len(order):
  j=idx+1
  while j<len(order) and order[j]['predicted_mixed']==order[idx]['predicted_mixed']:j+=1
  hit=sum(r['y'] for r in order[idx:j]);tp+=hit;seen+=j-idx;ap+=hit/P*tp/seen;idx=j
 classes={}
 for k,name in [(0,'allwrong'),(None,'mixed'),(5,'allcorrect')]:
  sub=[r for r in rs if (r['y'] if k is None else r['k']==k)]
  pm=[sum(r['p_j'])/5 for r in sub]
  classes[name]=dict(n=len(sub),q_quantiles=quant([r['predicted_mixed'] for r in sub]),pmean_quantiles=quant(pm),wrong_extreme_basin_count=(sum(x>.5271343224 for x in pm) if k==0 else sum(x<.4728656776 for x in pm) if k==5 else None),predicted_mixed=sum(r['predicted_mixed']>=.5 for r in sub))
 best=max([dict(threshold=t,**confusion(rs,t)) for t in sorted(set(r['predicted_mixed'] for r in rs))],key=lambda d:d['balanced_accuracy'])
 top=order[:512];hist=[sum(r['k']==k for r in top) for k in range(6)]
 return dict(n=len(rs),mixed=P,confusion_at_half=confusion(rs),auc=auc,average_precision=ap,best_threshold_in_this_split_analysis_only=best,classes=classes,top512_K_histogram=hist,top512_mixed=sum(hist[1:5]),top512_mean_q=sum(r['predicted_mixed'] for r in top)/len(top))
out=[];allrs=[]
for base,ep in PH:
 receipt=read(base/'probe'/('epoch-%04d.json'%ep));assert receipt['completed_epoch'] and receipt['cursor']==0
 c=read(base/'config.json');truth={r['qid']:r for r in read(base/'bank/groups.json')['diagnostic']};diag=[]
 for rank in range(4):diag+=rows(base/'probe/predictions'/('epoch-%04d-diagnostic-rank%d.jsonl.gz'%(ep,rank)))
 for r in diag:
  g=truth[r['qid']];assert r['strict_em']==g['labels'] and r['trajectory_ids']==g['trajectory_ids'];r['k']=sum(g['labels']);r['y']=0<r['k']<5
 p=Path(receipt['metrics']['selection2048']['predictions']['path']);rawsha=hashlib.sha256(p.read_bytes()).hexdigest();assert rawsha==receipt['metrics']['selection2048']['predictions']['sha256']
 pool=rows(p);tr={r['qid']:r for r in read(Path(c['selection_base'])/'bank/groups.json')['selection_validation']}
 assert len(pool)==len(tr)==2048
 for r in pool:
  g=tr[r['qid']];assert r['trajectory_ids']==g['trajectory_ids'];r['k']=sum(g['labels']);r['y']=0<r['k']<5
  a=b=1.
  for x in r['p_j']:a*=x;b*=1-x
  assert abs(1-a-b-r['predicted_mixed'])<1e-12
  r['strict_em']=g['labels']
 d=analyze(diag);s=analyze(pool);s['confusion_at_diagnostic_optimal_threshold']=confusion(pool,d['best_threshold_in_this_split_analysis_only']['threshold'])
 out.append(dict(epoch=ep,base=str(base),prediction_sha=rawsha,diagnostic=d,independent2048=s));allrs.append(pool)
a,b=[{r['qid']:r for r in rr} for rr in allrs];assert set(a)==set(b)
for q in a:assert a[q]['strict_em']==b[q]['strict_em'] and a[q]['trajectory_ids']==b[q]['trajectory_ids']
x,y=[set(r['qid'] for r in sorted(rr,key=lambda r:(-r['predicted_mixed'],r['qid']))[:512]) for rr in allrs]
comparison=dict(top512_overlap=len(x&y),top512_replaced=len(x-y),mixed_overlap=sum(a[q]['y'] for q in x&y),mixed_left=sum(a[q]['y'] for q in x-y),mixed_entered=sum(b[q]['y'] for q in y-x),identical_allfive_p_rows=sum(a[q]['p_j']==b[q]['p_j'] for q in a),max_abs_q_difference=max(abs(a[q]['predicted_mixed']-b[q]['predicted_mixed']) for q in a))
print(json.dumps(dict(status='SEALED_MIXED_ERROR_AND_RANKING_DIAGNOSIS',epochs=out,comparison=comparison,time=time.time(),training_updated=False)))
