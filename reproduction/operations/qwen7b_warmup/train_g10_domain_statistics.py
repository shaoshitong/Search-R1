"""Training-only G10 domain/K statistics and identity-only overlap audit.

No evaluation labels/scores, feature/model tensors, GPU, fitting, or changes to
current training/selection. G10 heterogeneity is never reported as G5 Mixed.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib,json,math,os
from pathlib import Path
import resource,time

B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021'
OUTPUT=A/'training-G10-delta-domain-statistics-v1'

def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
 p=Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(r):
 p=Path(r['path'])
 if p.resolve()!=p or not p.is_file() or B not in p.parents:raise ValueError('Bound actual remote artifact required')
 a=proof(p)
 if a['sha256']!=r['sha256'] or ('bytes'in r and a['bytes']!=r['bytes']):raise ValueError('Actual accepted data SHA differs')
 return p

def pinned(path,sha):return checked(dict(path=str(path),sha256=sha))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def choose(n,r):return 0 if n<r else math.factorial(n)//math.factorial(r)//math.factorial(n-r)
def expected_mixed5(k):
 if type(k)is not int or not 0<=k<=10:raise ValueError('True integer K0..10 required')
 return Fraction(252-choose(k,5)-choose(10-k,5),252)

def groups(path,expected):
 value=read(path);rs=value['train'] if isinstance(value,dict) else value;qids=set();tids=set()
 if len(rs)!=expected:raise ValueError('Accepted training group count differs')
 for g in rs:
  own=g['trajectory_ids'];ys=g['labels']
  if (g['qid'] in qids or len(own)!=10 or len(set(own))!=10 or tids.intersection(own)
      or len(ys)!=10 or any(type(y)is not int or y not in (0,1) for y in ys)):raise ValueError('Real unique own-G10 labels/IDs required')
  qids.add(g['qid']);tids.update(own)
  if 'real_G5_events'in g:
   if len(g['real_G5_events'])!=2:raise ValueError('Two authentic independent G5 events required')
   for event,start in zip(g['real_G5_events'],(0,5)):
    if event['trajectory_ids']!=own[start:start+5] or event['labels']!=ys[start:start+5]:raise ValueError('Own event IDs/EM concatenation differs')
 return rs

def summarize(rs):
 hist=[0]*11;old=[0]*6;new=[0]*6
 for g in rs:hist[sum(g['labels'])]+=1;old[sum(g['labels'][:5])]+=1;new[sum(g['labels'][5:])]+=1
 n=len(rs);mass=sum((count*expected_mixed5(k) for k,count in enumerate(hist)),Fraction())
 return dict(questions=n,authentic_trajectories=10*n,K0_to10=hist,
  G10_allwrong_questions=hist[0],G10_allcorrect_questions=hist[10],G10_heterogeneous_questions=n-hist[0]-hist[10],
  G10_allwrong_fraction=hist[0]/n if n else None,G10_allcorrect_fraction=hist[10]/n if n else None,
  G10_heterogeneous_fraction=(n-hist[0]-hist[10])/n if n else None,
  uniform_distinct_G5_expected_mixed_question_mass_exact=str(mass),
  uniform_distinct_G5_expected_mixed_fraction=float(mass/n) if n else None,
  original5_K0_to5=old,newseed5_K0_to5=new,
  original5_actual_Mixed_fraction=sum(old[1:5])/n if n else None,newseed5_actual_Mixed_fraction=sum(new[1:5])/n if n else None,
  original_question_total_weight=1.,expected_G5_formula='1 - [C(K,5)+C(10-K,5)]/C(10,5), C(n,5)=0 if n<5')

def by_domain(rs):
 own={}
 for g in rs:
  domain=json.loads(g['qid'])[0]
  if not isinstance(domain,str):raise ValueError('Real source domain identity missing')
  own.setdefault(domain,[]).append(g)
 return dict(all=summarize(rs),domains={d:summarize(g) for d,g in sorted(own.items())})

def identity_rows(rs):
 out=[];seen=set()
 for g in rs:
  if g['qid']in seen or not isinstance(g['question_hash'],str)or len(g['question_hash'])!=64:raise ValueError('Canonical unique identity required')
  out.append(dict(qid=g['qid'],question_hash=g['question_hash']));seen.add(g['qid'])
 return out

def overlap(left,right):
 lq={g['qid']:g['question_hash'] for g in left};rq={g['qid']:g['question_hash']for g in right};rh=set(rq.values());lh=set(lq.values())
 for q in set(lq)&set(rq):
  if lq[q]!=rq[q]:raise ValueError('Same canonical qid has conflicting hash')
 return dict(left_questions=len(lq),right_questions=len(rq),qid_overlap=len(set(lq)&set(rq)),
  shared_distinct_question_hashes=len(lh&rh),left_questions_with_any_hash_overlap=sum(h in rh for h in lq.values()),
  right_questions_with_any_hash_overlap=sum(h in lh for h in rq.values()))

def canonical_core(qids,config):
 """Identity-only parquet projection, no reward/target/EM/model reads."""
 import pyarrow as pa
 import pyarrow.parquet as pq
 pa.set_cpu_count(2);c=read(checked(config['original_config']));path=Path(c['canonical_train']);st=path.stat()
 if c['input_sha256']['train']!='5d40180ba54d35513bb780408caa068d7d60123214a49b0739f835a2ae47c8a1':raise ValueError('Accepted warmup canonical data binding differs')
 f=pq.ParquetFile(str(path));assert f.metadata.num_rows==169028;want=set(qids);out=[];allids=[]
 for batch in f.iter_batches(batch_size=2048,columns=['prompt','data_source','extra_info.index']):
  for row in batch.to_pylist():
   q=json.dumps([str(row['data_source']),str(row['extra_info']['index'])]);allids.append(q)
   if q in want:out.append(dict(qid=q,question_hash=hashlib.sha256('\n'.join(m['content']for m in row['prompt']).encode()).hexdigest()))
 manifest=read(checked(config['original_inputs']))
 if hashlib.sha256(json.dumps(allids).encode()).hexdigest()!=manifest['ordered_train_qid_sha256']or len(out)!=6528:raise ValueError('Exact original warmup/core membership differs')
 if (st.st_size,st.st_mtime_ns)!=(path.stat().st_size,path.stat().st_mtime_ns):raise ValueError('Canonical source changed')
 return identity_rows(out),dict(path=str(path),projected_columns=['prompt','data_source','extra_info.index'],
  canonical_questions=169028,ordered_qid_sha256=manifest['ordered_train_qid_sha256'],whole_file_SHA_inherited=c['input_sha256']['train'],
  gold_labels_or_rewards_read=False,original_warmup_question_unseen_claim=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120));started=time.time()
 a.output.mkdir(parents=True,exist_ok=True);destination=a.output/'complete.json'
 if destination.exists():print(json.dumps(dict(receipt=proof(destination),result=read(destination))));return
 newpath=pinned(A/'peer-newseed-ingest-audited122-delta-20261021-673b8cb6/complete.json','13f7e6a24699dd9858387210552a4d922ab521bd9a9c7c99648dab147f36048e');newdone=read(newpath)
 oldpath=pinned(A/'peer-newseed-ingest-newseed-20261021-185747-445fc691/complete.json','1fd5c29862267bc202419973ba6238d9c7b66de7ef2199dba725dda7fecf0804');olddone=read(oldpath)
 for done in (newdone,olddone):
  if done['status']!='RECEIVED_NEWSEED5_PLUS_CANONICAL_OLD5_RAW_G10_STRICT_EM_CAUSAL_SEEDS_ACCEPTED'or done['strict_EM_mismatches']!=0 or done['actual_same_question_old_new_seed_collisions']!=0:raise ValueError('Full real old5/new5 strictEM and seed acceptance required')
 cfg=read(checked(newdone['config']));new=groups(checked(newdone['outputs']['groups']),15616);old=groups(checked(olddone['outputs']['groups']),31232)
 for done,rs in ((newdone,new),(olddone,old)):
  if summarize(rs)['K0_to10']!=done['histogram_K0_to10']:raise ValueError('Accepted ownEM histogram differs')
 currentpath=pinned(A/'peer-expanded-L26-replay-v1/bank/complete.json','12f5e89581628e1e6d08f2d1b76829445e9f69237cecf77db057ff8b377e340d');currentdone=read(currentpath);current=groups(checked(currentdone['outputs']['groups.json']),30016)
 planpath=pinned(A/'peer-expanded-append-only-plan-v1/plan.json','8d6f5b046c8ee459c529f0750c80bac63455b2663f19cdcdce48d0fddd648cbf');plan=read(planpath)
 if len(plan['batches'])!=938 or set(plan['cohort_qids'])!={g['qid']for g in current}:raise ValueError('Current frozen30016 plan membership differs')
 oldby={g['qid']:g for g in old}
 for g in current:
  if any(g[k]!=oldby[g['qid']][k]for k in ('question_hash','trajectory_ids','labels')):raise ValueError('Current30016 changed authentic old31232 ownG10')
 scopes={'initial31232':identity_rows(old),'current_frozen30016':identity_rows(current)}
 corepath=pinned(A/'matched-loss128-v2/plan.json','3423f65455a60c76d38d09802502e6bc343205fb3ffa2316541e35b80f7d195a');coreplan=read(corepath);coreq=coreplan['cohort_qids']
 if len(coreq)!=6528 or len(set(coreq))!=6528:raise ValueError('Exact controlled core6528 required')
 scopes['controlled6528'],canonical=canonical_core(coreq,cfg)
 for name,n in [('diagnostic512',512),('selection2048',2048),('frozen_final4096',4096)]:
  value=cfg['exclusions'][name]
  if value['identity_only']is not True or len(value['groups'])!=n or any(set(g)!={'qid','question_hash'}for g in value['groups']):raise ValueError('Pure identity-only frozen scope required')
  scopes[name]=identity_rows(value['groups']) # Never dereference source_ref.
 identities=identity_rows(new);intersections={k:overlap(identities,v)for k,v in scopes.items()}
 for key in ('initial31232','current_frozen30016','diagnostic512','selection2048','frozen_final4096'):
  if intersections[key]['qid_overlap']or intersections[key]['shared_distinct_question_hashes']:raise ValueError('Unexpected new increment/current or evaluation identity overlap')
 result=dict(status='ACCEPTED_TRAIN_G10_DOMAIN_K_AND_UNIFORM_G5_EXPECTATION_COMPLETE',
  source_refs=dict(new_complete=proof(newpath),old_complete=proof(oldpath),new_groups=newdone['outputs']['groups'],old_groups=olddone['outputs']['groups'],
   current_complete=proof(currentpath),current_groups=currentdone['outputs']['groups.json'],current_frozen_plan=proof(planpath),controlled6528_plan=proof(corepath),identity_only_scopes_from_config=newdone['config']),
  cohorts=dict(new_audited122_15616=by_domain(new),initial_imported31232=by_domain(old),current_frozen30016=by_domain(current)),
  new_increment_identity_intersections=intersections,
  identity_scope_counts={k:dict(questions=len(v),unique_hashes=len({g['question_hash']for g in v}),identity_digest=digest(v))for k,v in scopes.items()},
  accepted_original_warmup_identity_projection=canonical,
  expected_Mixed_is_uniform_distinct_G5_not_G10_heterogeneity=True,
  original_warmup_saw_all_questions=True,unseen_original_warmup_claim=False,
  label_based_question_selection=False,threshold_or_score_fitting=False,evaluation_label_prediction_or_rollout_opened=False,
  final_identity_source_refs_followed=False,current_job_or_plan_modified=False,GPU_used=False,model_tensor_or_feature_payload_read=False,
  elapsed_seconds=time.time()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,epoch=time.time())
 temp=destination.with_suffix('.tmp');temp.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');temp.replace(destination);print(json.dumps(dict(receipt=proof(destination),result=result)),flush=True)

if __name__=='__main__':main()
