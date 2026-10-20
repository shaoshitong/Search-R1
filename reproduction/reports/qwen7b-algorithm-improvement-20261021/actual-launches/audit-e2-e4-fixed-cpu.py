"""One bounded independent score/own-label audit and remote-only result archive."""
import collections,fcntl,gzip,hashlib,json,math,os,pathlib,resource,tarfile,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/E2-E4-fixedG5-ensemble-validation-20261021')
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def check(r):
 p=pathlib.Path(r['path']);v=fp(p);assert all(v[k]==x for k,x in r.items() if k in v);return p
def gz(p):
 with gzip.open(str(p),'rt') as f:return [json.loads(x) for x in f]
def product(xs):
 z=1.
 for x in xs:z*=x
 return z
def auc(rows):
 ordered=sorted((r['score'],0<sum(r['strict_em'])<5) for r in rows);pos=sum(y for _,y in ordered);neg=len(rows)-pos;i=0;rs=0.
 while i<len(ordered):
  j=i+1
  while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
  rs+=((i+j+1)/2)*sum(y for _,y in ordered[i:j]);i=j
 return (rs-pos*(pos+1)/2)/(pos*neg) if pos and neg else None
def metrics(rows):
 ks=[sum(r['strict_em']) for r in rows];ys=[0<k<5 for k in ks];ps=[r['score']>=.5 for r in rows]
 tp=sum(y and p for y,p in zip(ys,ps));fpv=sum(not y and p for y,p in zip(ys,ps));fn=sum(y and not p for y,p in zip(ys,ps));tn=sum(not y and not p for y,p in zip(ys,ps));tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fpv) if tn+fpv else None
 return dict(questions=len(rows),mixed_questions=sum(ys),mixed_fraction=sum(ys)/len(rows),confusion=dict(tp=tp,fp=fpv,fn=fn,tn=tn),mixed_recall=tpr,nonmixed_recall=tnr,mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,mixed_roc_auc=auc(rows),k_histogram=[ks.count(k) for k in range(6)])
def same(actual,stored):
 for k,v in actual.items():assert abs(v-stored[k])<1e-12 if isinstance(v,float) else v==stored[k],(k,v,stored[k])
start=time.time();done=read(B/'complete.json');assert done['status']=='FIXED_SIX_CPU_ENSEMBLE_RECIPE_VALIDATION_COMPLETE';reg=read(check(done['prereg']));pub=read(B/'source-publication.json');assert pub['prereg_before_combinations'] is True
for name,item in pub['files'].items():check(dict(path=pub['code_root']+'/'+name,**{k:item[k] for k in ('bytes','sha256')}))
val=read(check(done['validation_source']));bank=read(check(val['source_groups']))['validation'];plan=read(check(val['source_fixed_G5']));plan=plan.get('fixed_G5',plan);by={g['qid']:g for g in bank};truth=[]
assert len(by)==len(bank)==2048 and plan['summary']['labels_consulted'] is False
for g in plan['groups']:
 own=by[g['qid']];ys=dict(zip(own['trajectory_ids'],own['labels']));truth.append(dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'],strict_em=[ys[t] for t in g['trajectory_ids']]))
base={};baseline={}
for name in ('E2','E4'):
 rows=gz(check(reg['sources'][name]));assert len(rows)==2048
 for r,t in zip(rows,truth):
  assert all(r[k]==t[k] for k in t);assert abs(r['predicted_mixed']-(1-product(r['p_j'])-product([1-p for p in r['p_j']])))<1e-6;r['score']=r['predicted_mixed']
 same(metrics(rows),done['baselines'][name]['pool']);chosen=sorted(rows,key=lambda r:(-r['score'],str(r['qid'])))[:512];same(metrics(chosen),done['baselines'][name]['selected512']);base[name]=rows;baseline[name]=dict(BA=metrics(rows)['mixed_balanced_accuracy'],Top512Mixed=metrics(chosen)['mixed_questions'])
assert len(done['results'])==len(reg['grid'])==6;accept=[];checks=[]
for result,recipe in zip(done['results'],reg['grid']):
 assert result['recipe']==recipe;rows=gz(check(result['predictions']));assert len(rows)==2048;maxerror=0.;w=recipe['E2_weight']
 for r,t,a,b in zip(rows,truth,base['E2'],base['E4']):
  assert all(r[k]==t[k] for k in t)
  if recipe['kind']=='trajectory_probability_mix':
   pp=[w*x+(1-w)*y for x,y in zip(a['p_j'],b['p_j'])];assert all(abs(x-y)<1e-15 for x,y in zip(r['p_j_mixture'],pp));want=1-product(pp)-product([1-p for p in pp])
  else:assert recipe['kind']=='direct_Mixed_score_mix' and 'p_j_mixture' not in r;want=w*a['score']+(1-w)*b['score']
  maxerror=max(maxerror,abs(r['score']-want))
 assert maxerror<1e-12;pool=metrics(rows);selected=sorted(rows,key=lambda r:(-r['score'],str(r['qid'])))[:512];top=metrics(selected);same(pool,result['pool']);same(top,result['selected512'])
 for label,k in (('allwrong',0),('allcorrect',5)):
  for part,stored in ((rows,result['pool']),(selected,result['selected512'])):
   valid=[r for r in part if sum(r['strict_em'])==k];fpr=sum(r['score']>=.5 for r in valid)/len(valid) if valid else None;assert fpr==stored['classwise'][label]['false_positive_rate']
 joint=pool['mixed_balanced_accuracy']>baseline['E2']['BA'] and top['mixed_questions']>baseline['E4']['Top512Mixed'];assert joint==result['joint_strict_pass']
 if joint:accept.append(recipe['id'])
 checks.append(dict(id=recipe['id'],q_formula_max_error=maxerror,own_ID_EM_exact=True,metrics_and_ranking_exact=True,joint=joint))
assert not accept and done['joint_pass_items']==0 and done['winner_validation_candidate'] is None and done['promote_none_if_all_fail'] is True
assert read(B/'cpu-exit.json')['returncode']==0;lock=(B/'cpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (B/'failure.json').exists()
files=[B/'complete.json',B/'source-publication.json',B/'cpu-intent.json',B/'cpu-exit.json',pathlib.Path(pub['code_root'])/'prereg.json',pathlib.Path(pub['code_root'])/'e2_e4_fixed_ensemble.py']+[pathlib.Path(r['predictions']['path']) for r in done['results']]
archive=B/'six-fixed-validation-results.tar';tmp=archive.with_suffix('.tmp');before=[fp(p) for p in files]
with tarfile.open(str(tmp),'w') as tar:
 for p in files:
  assert p.is_file() and not p.is_symlink();rel=p.relative_to(B);assert not rel.is_absolute() and '..' not in rel.parts;tar.add(str(p),arcname=str(rel),recursive=False)
tmp.replace(archive);assert before==[fp(p) for p in files]
report=dict(status='ALL_SIX_PREREG_OWN_EM_FORMULA_METRICS_AND_JOINT_GATE_INDEPENDENTLY_VERIFIED_FAIL_ALL',epoch=time.time(),elapsed_seconds=time.time()-start,source_complete=fp(B/'complete.json'),baseline_exact=baseline,recipes=checks,archive_format='tar',archive=fp(archive),archive_members=[dict(path=str(pathlib.Path(x['path']).relative_to(B)),bytes=x['bytes'],sha256=x['sha256']) for x in before],CPU_exit0=True,own_lock_free=True,raw_probability_rows_returned_to_Mac=False,model_or_feature_read=False,new_GPU=False,fitting_or_threshold_tuning=False,Platt_grid_changed=False,final4096_read=False,winner=None,promote_none=True)
(B/'independent-audit-and-archive.json').write_text(json.dumps(report,sort_keys=True)+'\n');print(json.dumps(report))
