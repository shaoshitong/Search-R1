import pathlib,json,gzip,hashlib,math,subprocess,time,os,functools,operator
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed8192-g10-train-v1');out=b/'E4-same-HF-validation';cp=out/'complete.json'
if not cp.exists():
 print(json.dumps({'status':'BASELINE_NOT_COMPLETE_NO_METRICS_CLAIM','observed_epoch':time.time()}));raise SystemExit(0)
read=lambda p:json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def check(r):
 p=pathlib.Path(r['path']);assert sha(p)==r['sha256'];assert 'bytes' not in r or p.stat().st_size==r['bytes'];return p
done=read(cp);config=read(b/'inputs/E4-only-config.json');assert sha(b/'inputs/E4-only-config.json')==done['config_sha256'];e=done['evaluation'];bank=read(check(e['bank']))['validation'];fixed=read(check(e['fixed_g5']));plan=fixed.get('fixed_G5',fixed);assert plan['summary']['labels_consulted'] is False;own={g['qid']:dict(zip(g['trajectory_ids'],g['labels'])) for g in bank};identities=plan['groups'];result={}
for name,item in done['checkpoints'].items():
 rows=[]
 with gzip.open(check(item['predictions']),'rt') as f:
  for line in f:rows.append(json.loads(line))
 assert len(rows)==2048 and [x['qid'] for x in rows]==[x['qid'] for x in identities]
 errors=[]
 for row,want in zip(rows,identities):
  assert row['trajectory_ids']==want['trajectory_ids'];assert row['strict_em']==[own[row['qid']][t] for t in row['trajectory_ids']]
  p=row['p_j'];q=1-functools.reduce(operator.mul,p,1.)-functools.reduce(operator.mul,(1-x for x in p),1.);errors.append(abs(q-row['predicted_mixed']))
 assert max(errors)<1e-6
 def metrics(values):
  ys=[0<sum(x['strict_em'])<5 for x in values];ps=[x['predicted_mixed']>=.5 for x in values];tp=sum(a and b for a,b in zip(ys,ps));fn=sum(a and not b for a,b in zip(ys,ps));fp=sum(not a and b for a,b in zip(ys,ps));tn=sum(not a and not b for a,b in zip(ys,ps));return {'mixed':sum(ys),'questions':len(values),'mixed_recall':tp/(tp+fn),'nonmixed_recall':tn/(tn+fp),'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp)),'confusion':{'tp':tp,'fn':fn,'fp':fp,'tn':tn}}
 pool=metrics(rows)
 scores=sorted((x['predicted_mixed'],0<sum(x['strict_em'])<5) for x in rows);ranks=0.;i=0
 while i<len(scores):
  j=i+1
  while j<len(scores) and scores[j][0]==scores[i][0]:j+=1
  ranks+=((i+1+j)/2)*sum(y for _,y in scores[i:j]);i=j
 pos=pool['mixed'];neg=len(rows)-pos;pool['auc']=(ranks-pos*(pos+1)/2)/(pos*neg)
 for label,k in [('allwrong',0),('allcorrect',5)]:
  part=[x for x in rows if sum(x['strict_em'])==k];pool[label+'_fpr']=sum(x['predicted_mixed']>=.5 for x in part)/len(part)
 pool['K_histogram']=[sum(sum(x['strict_em'])==k for x in rows) for k in range(6)]
 selected=sorted(rows,key=lambda x:(-x['predicted_mixed'],str(x['qid'])))[:512];n=sum(0<sum(x['strict_em'])<5 for x in selected);prior=item['metrics'];assert pool['balanced_accuracy']==prior['pool']['mixed_balanced_accuracy'];assert n==prior['selected512']['mixed_questions'];assert pool['auc']==prior['pool']['mixed_roc_auc'];assert pool['allwrong_fpr']==prior['pool']['classwise']['allwrong']['false_positive_rate'];assert pool['allcorrect_fpr']==prior['pool']['classwise']['allcorrect']['false_positive_rate']
 result[name]={'pool':pool,'selected512_mixed':n,'selected512_fraction':n/512,'q_formula_max_error':max(errors),'predictions_sha256':item['predictions']['sha256']}
v={'status':'E1_E4_IDENTICAL_NEW_HF_G5_OWN_EM_Q_AND_BA_TOP512_INDEPENDENTLY_VERIFIED','checkpoints':result,'questions':2048,'trajectories':10240,'fixed_G5_sha256':plan['summary']['selected_g5_sha256'],'actor_updates':0,'probe_updates':0,'final_pool_read':False,'observed_epoch':time.time()};p=b/'E4-same-HF-validation/independent-audit.json';p.write_text(json.dumps(v,sort_keys=True));print(json.dumps(v))
