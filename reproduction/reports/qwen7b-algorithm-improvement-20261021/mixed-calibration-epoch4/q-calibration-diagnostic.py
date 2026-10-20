import json,gzip,hashlib,time,math
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1024**2),b''):h.update(z)
 return h.hexdigest()
held=B/'selection-2048-20261020/bank';proof=read(held/'complete.json');assert sha(held/'groups.json')==proof['groups_sha256']
truth={g['qid']:g for g in read(held/'groups.json')['selection_validation']}
sources=[]
for label,base,epoch in [('E1',B/'mixed-only-ft-cp2-prefix20-20261020',1),('E2',B/'balanced-rank1-from-epoch1-prefix20-20261020',2),('E3',B/'balanced-rank1-from-epoch1-prefix20-20261020',3),('E4',B/'balanced-rank1-from-epoch1-prefix20-20261020',4)]:
 seal=read(base/'probe'/('epoch-%04d.json'%epoch));sources.append((label,seal['metrics']['selection2048']['predictions']))
sources.append(('event128',read(B/'algorithm-improvement-20261021/matched-loss128-v2/event/complete.json')['metrics']['selection2048']['predictions']))
result={}
for label,ref in sources:
 p=Path(ref['path']);assert sha(p)==ref['sha256']
 with gzip.open(str(p),'rt') as f:rows=[json.loads(x) for x in f]
 assert len(rows)==2048 and {r['qid'] for r in rows}==set(truth)
 vals=[]
 for r in rows:
  g=truth[r['qid']];assert r['trajectory_ids']==g['trajectory_ids']
  ps=r['p_j'];ar=aw=1.
  for x in ps:ar*=x;aw*=1-x
  q=r['predicted_mixed'];assert abs(q-(1-ar-aw))<2e-6
  k=sum(g['labels']);vals.append((q,0<k<5,k,sum(ps)/5,sum((x-sum(ps)/5)**2 for x in ps)/5))
 P=sum(y for q,y,k,mean,var in vals);N=len(vals)-P
 pairs=sorted(vals,reverse=True);tp=fp=0;best=.5;i=0
 while i<len(pairs):
  score=pairs[i][0];j=i
  while j<len(pairs) and pairs[j][0]==score:
   tp+=pairs[j][1];fp+=not pairs[j][1];j+=1
  best=max(best,.5*(tp/P+(N-fp)/N));i=j
 classes={}
 for key,check in [('allwrong',lambda k:k==0),('Mixed',lambda k:0<k<5),('allcorrect',lambda k:k==5)]:
  own=[r for r in vals if check(r[2])];classes[key]=dict(n=len(own),mean_predicted_q=sum(r[0] for r in own)/len(own),fraction_q_ge_half=sum(r[0]>=.5 for r in own)/len(own),mean_p=sum(r[3] for r in own)/len(own),mean_within_p_variance=sum(r[4] for r in own)/len(own))
 bins=[]
 for b in range(10):
  own=[r for r in vals if min(9,int(r[0]*10))==b]
  if own:bins.append(dict(bin=b,n=len(own),predicted=sum(r[0] for r in own)/len(own),actual=sum(r[1] for r in own)/len(own)))
 top=sorted(vals,reverse=True)[:512]
 tp=sum(q>=.5 and y for q,y,k,mean,var in vals);fp=sum(q>=.5 and not y for q,y,k,mean,var in vals)
 result[label]=dict(confusion=dict(tp=tp,fn=P-tp,fp=fp,tn=N-fp),mixed_BA=.5*(tp/P+(N-fp)/N),mixed_recall=tp/P,nonmixed_recall=(N-fp)/N,prediction_sha256=ref['sha256'],actual_mixed_fraction=P/len(vals),mean_q=sum(r[0] for r in vals)/len(vals),Brier=sum((q-y)**2 for q,y,k,mean,var in vals)/len(vals),ECE10=sum(b['n']*abs(b['predicted']-b['actual']) for b in bins)/len(vals),classes=classes,bins=bins,diagnostic_only_best_threshold_BA=best,threshold_selected_for_future=False,top512=dict(Mixed=sum(r[1] for r in top),nonMixed=sum(not r[1] for r in top),mean_q=sum(r[0] for r in top)/512),not_a_deployment_or_final_result=True)
print(json.dumps(dict(time=time.time(),status='OWN_EM_Q_CALIBRATION_AND_POSTHOC_THRESHOLD_UPPER_BOUND',results=result,final_pool_read=False,model_or_threshold_updated=False)))
