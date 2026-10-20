"""Remote stdlib-only append milestone/final acceptance; no tensor loads."""
import gzip,hashlib,importlib.util,json,math,pathlib,subprocess,sys,time
CURSOR=int(sys.argv[1]);assert CURSOR in (512,768,938)
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
b=A/'peer-expanded-append-only-train-v2';old=A/'peer-newseed8192-g10-train-v1'
def read(p):return json.loads(pathlib.Path(p).read_text())
def proof(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(r):
 p=pathlib.Path(r['path']);a=proof(p)
 assert a['sha256']==r['sha256'] and ('bytes' not in r or a['bytes']==r['bytes'])
 return p
def rows(p):
 with gzip.open(str(p),'rt') as f:return [json.loads(s) for s in f if s.strip()]
def auc(scores,labels):
 pos=sum(labels);neg=len(labels)-pos
 if not pos or not neg:return None
 ordered=sorted(zip(scores,labels));rank=0.;i=0
 while i<len(ordered):
  j=i+1
  while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
  rank+=(i+1+j)/2*sum(y for _,y in ordered[i:j]);i=j
 return (rank-pos*(pos+1)/2)/(pos*neg)
def metrics(rs):
 ks=[sum(r['strict_em']) for r in rs];truth=[0<k<5 for k in ks];q=[r['predicted_mixed'] for r in rs];called=[v>=.5 for v in q]
 tp=sum(y and p for y,p in zip(truth,called));fn=sum(y and not p for y,p in zip(truth,called));fp=sum(not y and p for y,p in zip(truth,called));tn=sum(not y and not p for y,p in zip(truth,called))
 tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
 def fpr(k):
  ix=[i for i,v in enumerate(ks) if v==k]
  return sum(called[i] for i in ix)/len(ix) if ix else None
 return dict(questions=len(rs),mixed_questions=sum(truth),mixed_fraction=sum(truth)/len(rs),confusion=dict(tp=tp,fn=fn,fp=fp,tn=tn),mixed_recall=tpr,nonmixed_recall=tnr,mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,mixed_roc_auc=auc(q,truth),allwrong_FPR=fpr(0),allcorrect_FPR=fpr(5),k_histogram=[ks.count(k) for k in range(6)])
def summarize(rs):return dict(pool=metrics(rs),selected512=metrics(sorted(rs,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512]))
def own(rs,truth):
 assert len(rs)==len(truth) and len({r['qid'] for r in rs})==len(rs)
 seen=set();error=0.
 for r in rs:
  assert r['qid'] in truth and len(r['trajectory_ids'])==len(r['strict_em'])==5 and len(set(r['trajectory_ids']))==5
  assert dict(zip(r['trajectory_ids'],r['strict_em']))==truth[r['qid']]
  assert not set(r['trajectory_ids'])&seen;seen.update(r['trajectory_ids'])
  assert all(type(v) is int and v in (0,1) for v in r['strict_em'])
  assert len(r['p_j'])==5 and all(math.isfinite(v) and 0<=v<=1 for v in r['p_j'])
  q=r['predicted_mixed'];assert math.isfinite(q) and 0<=q<=1
  positive=negative=1.
  for p in r['p_j']:positive*=p;negative*=1-p
  error=max(error,abs(q-(1-positive-negative)))
 assert error<1e-6
 return error
def metric_match(actual,stored):
 for k in ('questions','mixed_questions','confusion','k_histogram'):assert actual[k]==stored[k],k
 for k in ('mixed_fraction','mixed_recall','nonmixed_recall','mixed_balanced_accuracy','mixed_roc_auc'):
  assert actual[k]==stored[k] or (actual[k] is not None and stored[k] is not None and abs(actual[k]-stored[k])<1e-12),k
 assert actual['allwrong_FPR']==stored['classwise']['allwrong']['false_positive_rate']
 assert actual['allcorrect_FPR']==stored['classwise']['allcorrect']['false_positive_rate']

c=read(b/'config.json');pub=read(b/'source-publication.json');csha=proof(b/'config.json')['sha256']
assert pub['commit']==c['source_commit']=='b8c3f8a76fa195efc7c2e95cfa1a7d36669576ec'
assert csha=='d56668bdd2e9839abe042c0d4f720225cb0cafd62f22924867dbfda883b17a80'
assert c['source_checkpoint']['sha256']=='36d9bcc3be71ba94eaca3a1e1b292c0ed9a65297fb96a746ea5295432f09c000'
assert c['source_cursor']==256 and c['source_adam_steps']==10720 and c['update_budget']==938 and c['new_update_budget']==682
assert c['recipe']['loss_kwargs']==dict(trajectory_weight=1.,three_event_weight=1.)
assert c['source_world_size']==c['current_world_size']==1 and c['hardware_topology_changed'] is False and c['original_E1_archive_world_size']==4
assert not c['optimizer_reset'] and c['historical_prefix_retrained'] is False and c['actor_updates']==0
assert c['evaluation_cumulative_cursors']==[512,768,938]
assert not (b/'failure.json').exists() and not (b/'first-cp-audit-failure.json').exists()
evalbase=b/'evaluations'/('cursor-%06d'%CURSOR);receipt=read(evalbase/'complete.json');cp=receipt['checkpoint']
assert receipt['cursor']==CURSOR and receipt['steps']==10464+CURSOR and cp['cursor']==CURSOR
assert receipt['config_sha256']==csha and receipt['before_six_component_hashes']==receipt['after_six_component_hashes'] and receipt['training_six_components_unchanged'] is True
assert cp['config_sha256']==csha and cp['steps']==10464+CURSOR
fresh=proof(cp['path']);assert all(fresh[k]==cp[k] for k in ('bytes','sha256','path'))
first=read(b/'first-cp-independent-audit.json');replay=read(b/'next-update-replay.json')
assert first['cursor']==257 and first['steps']==10721 and first['source_world_size']==first['current_world_size']==1
assert first['status']=='APPEND_CP257_INDEPENDENT_FULL_SHA_AND_SIX_COMPONENT_CPU_RELOAD_PASSED'
assert first['component_hashes']==replay['actual']==replay['expected'] and replay['status']=='PASSED'
assert replay['source_steps']==10720 and replay['source_cursor']==256 and replay['candidate_cursor']==257
plan=read(checked(c['plan']));history=read(checked(c['original_plan']));assert plan['batches'][:256]==history['batches'] and plan['batch_epochs'][:256]==history['batch_epochs']
bank=read(checked(c['g10_complete']));groups=read(checked(bank['outputs']['groups.json']))['train'];lookup={g['qid']:g for g in groups}
assert len(groups)==len(lookup)==30016 and set(plan['cohort_qids'])==set(lookup)
name='reproduction/operations/qwen7b_warmup/g10_grouping.py';p=pathlib.Path(pub['code_root'])/name;assert proof(p)['sha256']==c['code_hashes'][name]
spec=importlib.util.spec_from_file_location('actual_append_grouping',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
curve=[json.loads(s) for s in (b/'curve.jsonl').read_text().splitlines() if s];curve=[r for r in curve if r['cursor']<=CURSOR]
assert len(curve)==CURSOR-256;points={}
for expected,r in enumerate(curve,257):
 assert r['cursor']==expected and r['additional_completed']==expected-256 and r['steps']==10464+expected
 assert r['original_questions']==32 and r['unique_real_trajectories']==320 and r['virtual_views']==128 and r['actual_forward_trajectory_exposures']==640 and r['recomputed_logits_bit_identical']
 grouping=dict(c['grouping']);grouping['epoch']=plan['batch_epochs'][expected-1]
 actual=module.build_training_views([lookup[q] for q in plan['batches'][expected-1]],role='train',**grouping)['summary']
 assert actual==r['grouping']
 assert abs(r['loss']-r['components']['trajectory']-r['components']['three_event'])<1e-5*(1+abs(r['loss']))
 if expected in (257,272,384,512,640,768,896,938):points[str(expected)]={k:r[k] for k in ('loss','components','seconds','steps')}
assert set(q for batch in plan['batches'][256:] for q in batch).isdisjoint(q for batch in history['batches'] for q in batch)
evaluations={};truth2048=None;original=read(old/'complete.json')
for e in c['evaluations']:
 source=read(checked(e['bank']))[e['split']];fixed=read(checked(e['fixed_g5']));fixed=fixed.get('fixed_G5',fixed)
 assert fixed['summary']['role']=='validation' and fixed['summary']['labels_consulted'] is False
 by={g['qid']:dict(zip(g['trajectory_ids'],g['labels'])) for g in source};truth={g['qid']:{t:by[g['qid']][t] for t in g['trajectory_ids']} for g in fixed['groups']}
 pr=receipt['predictions'][e['name']];assert all(pr[k]==receipt['metrics'][e['name']]['predictions'][k] for k in ('path','sha256'))
 rs=rows(checked(pr));err=own(rs,truth);m=summarize(rs)
 for key in ('pool','selected512'):metric_match(m[key],receipt['metrics'][e['name']][key])
 assert cp['epoch_time']<=pathlib.Path(pr['path']).stat().st_mtime
 prior=rows(checked(original['metrics'][e['name']]['predictions']));own(prior,truth);baseline=summarize(prior)
 evaluations[e['name']]=dict(metrics=m,control_metrics=baseline,predictions=pr,own_EM_fixed_IDs_exact=True,q_formula_max_error=err,delta_BA=m['pool']['mixed_balanced_accuracy']-baseline['pool']['mixed_balanced_accuracy'],delta_Top512=m['selected512']['mixed_questions']-baseline['selected512']['mixed_questions'])
 if len(rs)==2048:truth2048=truth
assert truth2048 is not None
e4ref=dict(path=str(old/'E4-same-HF-validation/E4/predictions.jsonl.gz'),bytes=155299,sha256='66cdf3e7c748bda7f508c4f381ad0456a413d5d73c73e5d9f1eb774b71bdb707')
e4=rows(checked(e4ref));own(e4,truth2048);e4metrics=summarize(e4)
jid=(b/'job.id').read_text().strip();r=subprocess.run(['sacct','-j',jid,'-n','-P','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25);assert not r.returncode
state=next(s.split('|') for s in r.stdout.splitlines() if s.split('|')[0]==jid)
complete=None
if CURSOR==938:
 assert state[1:3]==['COMPLETED','0:0'];done=read(b/'complete.json');assert done['cursor']==938 and done['steps']==11402 and done['additional_updates']==682 and done['checkpoint']==cp
 complete=proof(b/'complete.json')
result=dict(status='APPEND_MILESTONE_OWN_EM_PLAN_METRICS_CP_SHA_INDEPENDENTLY_ACCEPTED',cursor=CURSOR,steps=10464+CURSOR,additional_updates=CURSOR-256,job_id=jid,sacct=r.stdout.strip(),config=proof(b/'config.json'),evaluation_receipt=proof(evalbase/'complete.json'),checkpoint=cp,complete=complete,first_CPU_audit=proof(b/'first-cp-independent-audit.json'),actual_new_batch_grouping_all_checked=True,old256_retrained=False,learning_curve=points,evaluations=evaluations,E4_same_HF_reference=dict(predictions=e4ref,metrics=e4metrics),fixed_threshold=.5,threshold_fitted=False,source_Adam=10720,optimizer_reset=False,source_world_size=1,current_world_size=1,original_E1_archive_world_size=4,no_model_optimizer_or_feature_tensor_loaded_on_login=True,final_pool_read=False,production_promotion_claim=False,time=time.time())
destination=evalbase/'independent-audit.json';assert not destination.exists(),'Preserve an accepted audit';destination.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');result['audit_ref']=proof(destination);print(json.dumps(result))
