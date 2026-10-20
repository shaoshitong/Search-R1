"""Exactly six preregistered CPU-only combinations on viewed recipe validation.
Never fits parameters/thresholds, extends the grid, or reads an untouched final.
"""
import argparse,fcntl,gzip,hashlib,json,math,os,pathlib,resource,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021';OUT=A/'E2-E4-fixedG5-ensemble-validation-20261021'
VALIDATION=A/'validation-fixedG5-hf-replay-20261021/capture/complete.json'
FIXED='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7'
E2CP='6a970815ff3aec26eedf4d6a65589bbebac3c4ccbdf8eefdf859a83e2a11bd2d';E4CP='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'
def read(p):return json.loads(pathlib.Path(p).read_text())
def proof(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 p=pathlib.Path(ref['path'])
 if not p.is_absolute() or p.resolve()!=p or not str(p).startswith(str(B)+'/'):raise ValueError('Only bound remote experiment source')
 f=proof(p)
 if any(f[k]!=v for k,v in ref.items() if k in f):raise ValueError('SHA/size mismatch: '+str(p))
 return p
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic(p,v):
 p=pathlib.Path(p);t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def product(xs):
 v=1.
 for x in xs:v*=x
 return v
def qscore(ps):return max(0.,1-product(ps)-product([1-p for p in ps]))
def combine(a,b,weight,kind):
 if len(a)!=5 or len(b)!=5 or weight not in (.25,.5,.75):raise ValueError('Only frozen G5/weight grid')
 if kind=='trajectory_probability_mix':
  ps=[weight*x+(1-weight)*y for x,y in zip(a,b)];return qscore(ps),ps
 if kind=='direct_Mixed_score_mix':return weight*qscore(a)+(1-weight)*qscore(b),None
 raise ValueError('No extra score formulas')
def auc(rows):
 ordered=sorted((r['score'],0<sum(r['strict_em'])<5) for r in rows);pos=sum(y for _,y in ordered);neg=len(rows)-pos;i=0;rs=0.
 if not pos or not neg:return None
 while i<len(ordered):
  j=i+1
  while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
  rs+=((i+1+j)/2)*sum(y for _,y in ordered[i:j]);i=j
 return (rs-pos*(pos+1)/2)/(pos*neg)
def metrics(rows):
 ks=[sum(r['strict_em']) for r in rows];ys=[0<k<5 for k in ks];pred=[r['score']>=.5 for r in rows]
 tp=sum(y and p for y,p in zip(ys,pred));fp=sum(not y and p for y,p in zip(ys,pred));fn=sum(y and not p for y,p in zip(ys,pred));tn=sum(not y and not p for y,p in zip(ys,pred));tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
 per={}
 for name,mask in [('allwrong',[k==0 for k in ks]),('mixed',ys),('allcorrect',[k==5 for k in ks])]:
  ix=[i for i,v in enumerate(mask) if v];n=sum(pred[i] for i in ix);per[name]=dict(questions=len(ix),predicted_mixed=n,false_positive_rate=n/len(ix) if ix and name!='mixed' else None,predicted_mixed_fraction=n/len(ix) if ix else None)
 return dict(questions=len(rows),mixed_questions=sum(ys),mixed_fraction=sum(ys)/len(rows),k_histogram=[ks.count(k) for k in range(6)],confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),mixed_recall=tpr,nonmixed_recall=tnr,mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,mixed_roc_auc=auc(rows),classwise=per)
def summary(rows):
 selected=sorted(rows,key=lambda r:(-r['score'],str(r['qid'])))[:512]
 return dict(pool=metrics(rows),selected512=metrics(selected),selected_qids_sha256=digest([r['qid'] for r in selected]))
def frozen_truth(val):
 bank=read(checked(val['source_groups']))['validation'];plan=read(checked(val['source_fixed_G5']));plan=plan.get('fixed_G5',plan)
 if plan['summary']['labels_consulted'] is not False or plan['summary']['role']!='validation' or digest(plan['groups'])!=FIXED:raise ValueError('Fixed label-blind G5 changed')
 by={g['qid']:g for g in bank};fixed=[];seen=set()
 if len(bank)!=len(by) or len(by)!=2048:raise ValueError('Validation identities differ')
 for p in plan['groups']:
  g=by[p['qid']];labels=dict(zip(g['trajectory_ids'],g['labels']));ids=p['trajectory_ids']
  if len(ids)!=5 or len(set(ids))!=5 or set(ids)&seen:raise ValueError('Own distinct G5 required')
  if len(labels)!=len(g['labels']) or any(type(y) is not int or y not in (0,1) for y in labels.values()):raise ValueError('Real binary ownEM required')
  seen.update(ids);fixed.append(dict(qid=g['qid'],trajectory_ids=ids,strict_em=[labels[t] for t in ids]))
 if len(fixed)!=2048 or len(seen)!=10240:raise ValueError('Exact fixed2048G5 required')
 return fixed

def load_scores(ref,fixed):
 path=checked(ref)
 with gzip.open(str(path),'rt') as f:rows=[json.loads(s) for s in f]
 if len(rows)!=2048 or len({r['qid'] for r in rows})!=2048:raise ValueError('Wrong score row identities')
 maxerror=0.
 for r,g in zip(rows,fixed):
  if any(r[k]!=g[k] for k in ('qid','trajectory_ids','strict_em')):raise ValueError('Score ownIDs/EM do not match rejoined bank')
  ps=r['p_j'];q=r['predicted_mixed']
  if len(ps)!=5 or any(not math.isfinite(p) or not 0<=p<=1 for p in ps) or not math.isfinite(q) or not 0<=q<=1:raise ValueError('Invalid G5 probabilities/score')
  maxerror=max(maxerror,abs(qscore(ps)-q));r['score']=q
 if maxerror>=1e-6:raise ValueError('Single-model own Bernoulli formula mismatch')
 return rows,maxerror

def run(prereg,prereg_sha):
 if proof(prereg)['sha256']!=prereg_sha:raise ValueError('Published prereg SHA differs')
 reg=read(prereg);expected=[(kind,w) for kind in ('trajectory_probability_mix','direct_Mixed_score_mix') for w in (.25,.5,.75)]
 if [(x['kind'],x['E2_weight']) for x in reg['grid']]!=expected or reg['classification_threshold']!=.5:raise ValueError('Frozen six-item grid/threshold changed')
 val=read(VALIDATION)
 if val['status']!='NEW_VALIDATION2048_FIXED_G5_QUALIFIED_HF_REPLAY_LAYER26_COMPLETE' or val['questions']!=2048 or val['trajectories']!=10240 or val['source_runtime']!='huggingface_frozen_causal_prefill' or val['capture_layers']!=[26] or val['fixed_G5_sha256']!=FIXED or val['new_G5_selected'] is not False or val['final_pool_read'] is not False:raise ValueError('Only exact same qualified HF viewed fixed validation')
 fixed=frozen_truth(val);e2c=read(A/'E2-same-HF-baseline-20261021/config.json');e2r=read(A/'E2-same-HF-baseline-20261021/complete.json');e4r=read(A/'peer-newseed8192-g10-train-v1/E4-same-HF-validation/E4/complete.json');ev=e4r['evaluation']
 if e2r['config']!=proof(A/'E2-same-HF-baseline-20261021/config.json') or e2c['validation_complete']!=proof(VALIDATION) or e4r['config_sha256']!=proof(A/'peer-newseed8192-g10-train-v1/inputs/E4-only-config.json')['sha256']:raise ValueError('Actual scored source config/validation binding changed')
 if e2r['checkpoint']['sha256']!=E2CP or e4r['checkpoint']['sha256']!=E4CP or not e2r['parameters_unchanged'] or e2r['optimizer_constructed'] is not False or e4r['optimizer_constructed'] is not False:raise ValueError('Wrong sealed read-only E2/E4 identity')
 if e2c['features']!=val['features'] or ev['features']!=val['features'] or ev['bank']!=val['source_groups'] or ev['fixed_g5']!=val['source_fixed_G5'] or ev['backend']!='huggingface_frozen_causal_prefill':raise ValueError('E2/E4 feature backend/frozen selection mismatch')
 if e2r['predictions']!=reg['sources']['E2'] or e4r['predictions']!=reg['sources']['E4']:raise ValueError('Predictions differ from prereg')
 e2,max2=load_scores(reg['sources']['E2'],fixed);e4,max4=load_scores(reg['sources']['E4'],fixed);baselines=dict(E2=summary(e2),E4=summary(e4))
 if baselines['E4']['selected512']['mixed_questions']!=196 or baselines['E2']['pool']['mixed_balanced_accuracy']<baselines['E4']['pool']['mixed_balanced_accuracy']:raise ValueError('Predeclared baseline ordering differs')
 atomic(OUT/'progress.json',dict(stage='EXACT_SOURCES_OWN_EM_AND_BASELINES_VALIDATED_BEFORE_COMBINATIONS',epoch=time.time(),recipes=0))
 error_counts={};mixed_disagree={}
 for a,b in zip(e2,e4):
  y=0<sum(a['strict_em'])<5;aa=a['score']>=.5;bb=b['score']>=.5;key=str(int(aa!=y))+'->'+str(int(bb!=y));error_counts[key]=error_counts.get(key,0)+1
  if aa!=bb:mixed_disagree['mixed' if y else 'nonmixed']=mixed_disagree.get('mixed' if y else 'nonmixed',0)+1
 results=[]
 for recipe in reg['grid']:
  rows=[]
  for a,b in zip(e2,e4):
   # Direct Mixed-score interpolation uses ORIGINAL saved q, not rederived p.
   if recipe['kind']=='direct_Mixed_score_mix':score=recipe['E2_weight']*a['score']+recipe['E4_weight']*b['score'];ps=None
   else:score,ps=combine(a['p_j'],b['p_j'],recipe['E2_weight'],recipe['kind'])
   r=dict(qid=a['qid'],trajectory_ids=a['trajectory_ids'],strict_em=a['strict_em'],score=score)
   if ps is not None:r['p_j_mixture']=ps
   rows.append(r)
  result=summary(rows);ba=result['pool']['mixed_balanced_accuracy'];top=result['selected512']['mixed_questions'];result.update(recipe=recipe,BA_delta_vs_E2=ba-baselines['E2']['pool']['mixed_balanced_accuracy'],Top512_Mixed_delta_vs_E4=top-baselines['E4']['selected512']['mixed_questions'],joint_strict_pass=ba>baselines['E2']['pool']['mixed_balanced_accuracy'] and top>baselines['E4']['selected512']['mixed_questions'])
  path=OUT/(recipe['id']+'.jsonl.gz');temp=path.with_suffix('.tmp')
  with gzip.open(str(temp),'wt') as f:
   for r in rows:f.write(json.dumps(r)+'\n')
  temp.replace(path);result['predictions']=proof(path);results.append(result);atomic(OUT/'progress.json',dict(stage='FIXED_CPU_RECIPE_COMPLETED',epoch=time.time(),recipes=len(results),last_id=recipe['id']))
 passes=[x for x in results if x['joint_strict_pass']];passes.sort(key=lambda x:(-x['pool']['mixed_balanced_accuracy'],-x['selected512']['mixed_questions'],-x['pool']['mixed_roc_auc'],x['recipe']['id']))
 return dict(status='FIXED_SIX_CPU_ENSEMBLE_RECIPE_VALIDATION_COMPLETE',prereg=proof(prereg),diagnostic_runtime=proof(__file__),validation_source=proof(VALIDATION),source_model_receipts=dict(E2=proof(A/'E2-same-HF-baseline-20261021/complete.json'),E4=proof(A/'peer-newseed8192-g10-train-v1/E4-same-HF-validation/E4/complete.json')),source_pred_formula_max_errors=dict(E2=max2,E4=max4),frozen_G5_sha256=FIXED,ordered_own_trajectory_ids_sha256=digest([g['trajectory_ids'] for g in fixed]),baselines=baselines,model_error_pair_E2_to_E4=error_counts,models_predicted_Mixed_disagreement=mixed_disagree,results=results,joint_pass_items=len(passes),winner_validation_candidate=passes[0]['recipe']['id'] if passes else None,promote_none_if_all_fail=not passes,fitting_performed=False,threshold_fitted=False,Platt_grid_changed=False,no_new_GPU=True,final4096_read_or_generated=False,unseen_or_final_generalization_claim=False,selection_scope='previously viewed tuning_allowed recipe validation; winner is validation candidate only',epoch=time.time())
def main():
 p=argparse.ArgumentParser();p.add_argument('--prereg',type=pathlib.Path,required=True);p.add_argument('--prereg-sha256',required=True);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'cpu.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(60,60))
  if (OUT/'complete.json').exists():raise ValueError('Existing immutable six-grid result preserved')
  try:
   result=run(a.prereg,a.prereg_sha256);atomic(OUT/'complete.json',result);print(json.dumps(dict(status=result['status'],joint_pass_items=result['joint_pass_items'],winner=result['winner_validation_candidate'])),flush=True)
  except BaseException as e:atomic(OUT/'failure.json',dict(type=type(e).__name__,message=str(e),epoch=time.time()));raise
if __name__=='__main__':main()
