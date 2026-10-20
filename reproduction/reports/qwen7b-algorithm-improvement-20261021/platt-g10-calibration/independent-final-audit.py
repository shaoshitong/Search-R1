"""Read-only remote CPU audit; no model scalars or probability rows leave HPC3."""
import gzip,hashlib,json,math,pathlib,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
B=A/'platt-g10-calibration-v1';O=B/'output'
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);x=p.read_bytes();return dict(path=str(p),bytes=len(x),sha256=hashlib.sha256(x).hexdigest())
def checked(r):
 p=pathlib.Path(r['path']);v=fp(p);assert v['sha256']==r['sha256']
 if 'bytes' in r:assert v['bytes']==r['bytes']
 return p
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def metric(rows):
 labels=[int(0<sum(r['strict_em'])<5) for r in rows];scores=[r['q'] for r in rows]
 tp=sum(y and s>=.5 for y,s in zip(labels,scores));fn=sum(labels)-tp
 fp=sum(not y and s>=.5 for y,s in zip(labels,scores));tn=len(rows)-sum(labels)-fp
 pairs=sorted(zip(scores,labels));i=0;rank=0.
 while i<len(pairs):
  j=i+1
  while j<len(pairs) and pairs[j][0]==pairs[i][0]:j+=1
  rank+=(i+j+1)/2*sum(y for _,y in pairs[i:j]);i=j
 n=sum(labels);neg=len(rows)-n;auc=(rank-n*(n+1)/2)/(n*neg)
 top=sorted(rows,key=lambda r:(-r['q'],str(r['qid'])))[:512]
 def fpr(k):
  s=[r['q'] for r in rows if sum(r['strict_em'])==k];return sum(v>=.5 for v in s)/len(s)
 return dict(balanced_accuracy=(tp/n+tn/neg)/2,Mixed_recall=tp/n,nonMixed_recall=tn/neg,allwrong_FPR=fpr(0),allcorrect_FPR=fpr(5),AUC=auc,confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),questions=len(rows),Top512=dict(Mixed=sum(0<sum(r['strict_em'])<5 for r in top),Mixed_fraction=sum(0<sum(r['strict_em'])<5 for r in top)/512,K_histogram=[sum(sum(r['strict_em'])==k for r in top) for k in range(6)],questions=512))
def compare(a,b):
 for k in a:
  if isinstance(a[k],dict):compare(a[k],b[k])
  elif isinstance(a[k],float):assert abs(a[k]-b[k])<1e-12,(k,a[k],b[k])
  else:assert a[k]==b[k],(k,a[k],b[k])
def product(x):
 v=1.
 for t in x:v*=t
 return v
def q(ps):return 1-product(ps)-product([1-p for p in ps])
def sigmoid(z):return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
c=read(B/'config.json');done=read(O/'complete.json');exit_status=read(B/'exit.json');freeze=read(checked(done['frozen_candidate']))
assert exit_status['returncode']==0 and not (O/'failure.json').exists()
assert done['status']=='FROZEN_TWO_PARAMETER_CALIBRATION_EXTERNAL_COMPLETE'
assert done['external_scored_models']==1 and done['external_threshold_fitted'] is False and done['final_pool_read'] is False
assert freeze['external_opened'] is False and freeze['threshold']==.5 and freeze['winner']==done['winner']
assert freeze['frozen_epoch']<exit_status['epoch']
assert len(done['internal_candidates'])==12 and len({x['id'] for x in done['internal_candidates']})==12
cp_refs=[]
for x in done['internal_candidates']:
 assert x['fit']['full_checkpoint']['raw_LBFGS_RNG_loaded'] is True
 r=x['fit']['full_checkpoint'];checked(r);cp_refs.append(dict(id=x['id'],**fp(r['path'])))
 accepted=[x for x in done['internal_candidates'] if x['internal_joint_improvement']]
 accepted.sort(key=lambda x:(-(x['internal']['balanced_accuracy']+x['internal']['Top512']['Mixed_fraction']),x['id']))
assert accepted[0]['id']==done['winner']['id']
model=read(checked(freeze['model']));assert model['id']==done['winner']['id']
u,v=model['raw'];assert abs(model['a']-math.exp(3*math.tanh(u)))<1e-12 and abs(model['b']-10*math.tanh(v))<1e-12
source=next(s for s in c['sources'] if s['name']==done['winner']['source'])
e=next(e for e in read(A/'peer-newseed8192-g10-train-v1/config.json')['evaluations'] if e['name']=='selection2048FreshG5HF')
bank=read(checked(e['bank']))[e['split']];by={str(g['qid']):g for g in bank};plan=read(checked(e['fixed_g5']));plan=plan.get('fixed_G5',plan)
assert plan['summary']['labels_consulted'] is False and plan['summary']['role']=='validation'
assert digest(plan['groups'])==plan['summary']['selected_g5_sha256']=='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7'
with gzip.open(checked(source['external_predictions']),'rt') as f:rows=[json.loads(s) for s in f]
assert len(rows)==len(by)==len(plan['groups'])==2048 and len({str(r['qid']) for r in rows})==2048
raw=[];cal=[];error=0.;tids=set()
for r,p in zip(rows,plan['groups']):
 assert r['qid']==p['qid'] and r['trajectory_ids']==p['trajectory_ids'];g=by[str(r['qid'])];labels=dict(zip(g['trajectory_ids'],g['labels']))
 assert r['strict_em']==[labels[t] for t in r['trajectory_ids']]
 assert len(r['p_j'])==5 and all(math.isfinite(t) and 0<=t<=1 for t in r['p_j'])
 assert len(set(r['trajectory_ids']))==5 and not tids&set(r['trajectory_ids']);tids.update(r['trajectory_ids'])
 error=max(error,abs(q(r['p_j'])-r['predicted_mixed']))
 raw.append(dict(qid=r['qid'],strict_em=r['strict_em'],q=q(r['p_j'])))
 probs=[min(1-1e-6,max(1e-6,p)) for p in r['p_j']]
 transformed=[sigmoid(model['a']*math.log(p/(1-p))+model['b']) for p in probs]
 cal.append(dict(qid=r['qid'],strict_em=r['strict_em'],q=q(transformed)))
assert error<1e-6
actual=metric(cal);baseline=metric(raw);compare(actual,done['external']);compare(baseline,done['uncalibrated_same_model_external'])
refs={s['name']:s['external_predictions'] for s in c['sources']};e4ref=refs['E4']
with gzip.open(checked(e4ref),'rt') as f:e4rows=[json.loads(s) for s in f]
for r,s in zip(rows,e4rows):
 for k in ('qid','trajectory_ids','strict_em'):assert r[k]==s[k]
e4=metric([dict(qid=r['qid'],strict_em=r['strict_em'],q=q(r['p_j'])) for r in e4rows])
result=dict(status='PLATT_FINAL_INDEPENDENT_FROZEN_SOURCE_OWN_EM_G5_METRICS_SHA_AUDIT_PASS',epoch=time.time(),complete=fp(O/'complete.json'),freeze=fp(O/'candidate-freeze.json'),winner=done['winner']['id'],external=actual,uncalibrated_same_model=baseline,same_HF_E4=e4,checkpoint_refs=cp_refs,full_checkpoint_load_acceptance_inherited_from_producer=True,checkpoint_SHA_all12_independent_fresh=True,own_EM_fixed_ID_order_exact=True,q_formula_original_max_error=error,source_predictions=fp(source['external_predictions']['path']),joint_gain_versus_same_model=(actual['balanced_accuracy']>baseline['balanced_accuracy'] and actual['Top512']['Mixed']>baseline['Top512']['Mixed']),joint_gain_versus_same_HF_E4=(actual['balanced_accuracy']>e4['balanced_accuracy'] and actual['Top512']['Mixed']>e4['Top512']['Mixed']),elapsed_seconds=done['elapsed'],peak_RSS_bytes=done['peak_RSS_bytes'],learned_scalars_exported=False,raw_rows_exported=False,final_pool_read=False)
print(json.dumps(result))
