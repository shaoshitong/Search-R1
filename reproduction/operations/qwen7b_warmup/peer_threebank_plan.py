"""CPU-only metadata merge and preregistered internal5120 / append728 plan.

No models, tensors, calibration, scoring, rollout, or training submission.
The original938 ordered batches remain immutable. Holdout domain quotas are
not correctness-class balancing. controlled6528 excludes holdout eligibility
only; fixed evaluation identities exclude the complete merged cohort.
"""
import argparse,copy,fcntl,hashlib,json,os,resource,time
from pathlib import Path

STATUS='THREE_ACCEPTED_BANKS_58432_INTERNAL5120_FIT53312_APPEND728_METADATA_ACCEPTED'
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021';TARGET=A/'peer-threebank58432-internal5120-plan-v1'

def read(p):return json.loads(Path(p).read_bytes())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def proof(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(ref):
 p=Path(ref['path'])
 if p.resolve()!=p or not str(p).startswith(str(B)+'/') or proof(p)!=ref:raise ValueError('Actual immutable remote SHA/bytes differs: '+str(p))
 return p

def immutable(p,v):
 p=Path(p);data=canonical(v)+b'\n'
 if p.exists():
  if p.read_bytes()!=data:raise ValueError('Immutable metadata differs: '+str(p))
 else:
  q=p.with_name(p.name+'.%d.tmp'%os.getpid());q.write_bytes(data);q.replace(p)

def identities(groups):
 out=[];qs=set();hs=set();ts=set()
 for g in groups:
  q,h,ids=g['qid'],g['question_hash'],list(g['trajectory_ids'])
  if not isinstance(q,str) or not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) or json.loads(q)[0] not in ('nq','hotpotqa') or q in qs or h in hs or len(ids)!=10 or len(set(ids))!=10 or ts.intersection(ids):raise ValueError('Distinct genuine G10 qid/hash/trajectory identities required')
  if any(not isinstance(t,str) or not t for t in ids):raise ValueError('Authentic own trajectory IDs required')
  qs.add(q);hs.add(h);ts.update(ids);out.append(dict(qid=q,question_hash=h,trajectory_ids=ids))
 return out

def identity_set(rows):
 qs={};hs=set()
 for g in rows:
  if set(g)!={'qid','question_hash'} or g['qid'] in qs or len(g['question_hash'])!=64:raise ValueError('Pure unique exclusion identities required')
  qs[g['qid']]=g['question_hash'];hs.add(g['question_hash'])
 return qs,hs

def build_partitions(core,added,oldplan,controlled,excluded,*,expected_core=30016,expected_added=28416,hold_per_domain=2560,batch=32):
 core=identities(core);added=identities(added);allrows=identities(core+added)
 if len(core)!=expected_core or len(added)!=expected_added or len(core)%batch or len(added)%batch:raise ValueError('Exact explicit accepted bank budgets required')
 if oldplan['cohort_qids']!=[g['qid'] for g in core] or oldplan['questions']!=len(core) or oldplan['updates']!=len(core)//batch or oldplan['global_questions']!=batch or oldplan['views_per_question']!=4 or oldplan['view_weight']!=.25 or oldplan['effective_total_weight_per_original_question']!=1 or oldplan['labels_used_for_question_order'] is not False or oldplan['all_questions_once'] is not True:raise ValueError('Actual current938 plan or four-view semantics differ')
 flat=[q for x in oldplan['batches'] for q in x]
 if len(oldplan['batches'])!=len(core)//batch or any(len(x)!=batch for x in oldplan['batches']) or len(flat)!=len(set(flat)) or set(flat)!={g['qid'] for g in core} or oldplan['batch_epochs']!=[0]*(len(core)//batch):raise ValueError('Historical prefix is not exactly once ordered complete batches')
 eq,eh=identity_set(excluded);cq,ch=identity_set(controlled)
 for g in allrows:
  if g['qid'] in eq or g['question_hash'] in eh:raise ValueError('Fixed evaluation qid/hash intersects merged complete cohort')
  if g['qid'] in cq and cq[g['qid']]!=g['question_hash']:raise ValueError('Controlled canonical question hash differs')
 eligible={'nq':[],'hotpotqa':[]};control_overlaps=[]
 for g in added:
  if g['qid'] in cq or g['question_hash'] in ch:control_overlaps.append(g['qid']);continue
  eligible[json.loads(g['qid'])[0]].append(g)
 held=[]
 for domain in ('nq','hotpotqa'):
  own=sorted(eligible[domain],key=lambda g:(digest(['peer-new28416-internal5120',9137,domain,g['qid'],g['question_hash']]),g['qid']))
  if len(own)<hold_per_domain:raise ValueError('Not enough domain-qualified holdout identities')
  held+=own[:hold_per_domain]
 heldq={g['qid'] for g in held};tail=[g for g in added if g['qid'] not in heldq]
 tail.sort(key=lambda g:(digest(['peer-next-append-only-qid',1729,g['qid']]),g['qid']))
 if len(tail)%batch:raise ValueError('No silent drop or pad to full append batch')
 fixed=[dict(g,trajectory_ids=g['trajectory_ids'][5:10],source_indices=[5,6,7,8,9],domain=json.loads(g['qid'])[0]) for g in held]
 batches=copy.deepcopy(oldplan['batches'])+[[g['qid'] for g in tail[i:i+batch]] for i in range(0,len(tail),batch)]
 if batches[:len(oldplan['batches'])]!=oldplan['batches']:raise ValueError('Historical ordered batches changed')
 plan=copy.deepcopy(oldplan);plan.update(cohort_qids=[g['qid'] for g in core+tail],batches=batches,batch_epochs=copy.deepcopy(oldplan['batch_epochs'])+[0]*(len(tail)//batch),questions=len(core)+len(tail),updates=len(batches),historical_updates=len(oldplan['batches']),additional_updates=len(tail)//batch,historical_batches_sha256=digest(oldplan['batches']),historical_full_plan_canonical_sha256=digest(oldplan),source_plan_canonical_sha256=digest(oldplan),added_question_order_sha256=digest([g['qid'] for g in tail]),domain_scope='NQ_and_Hotpot_real_G10_training',historical_prefix_retrained=False,selection_uses_labels=False,selection_uses_scores=False,budget_scope='Metadata-only proposed continuation from future verified CP938; no model or LR chosen')
 selection=dict(role='internal_diagnostic_holdout',groups=fixed,summary=dict(questions=len(held),domains={d:hold_per_domain for d in eligible},labels_consulted=False,scores_consulted=False,holdout_identity_seed=9137,source_indices=[5,6,7,8,9],original_warmup_unseen_claim=False,not_untouched_final=True,controlled6528_excluded_from_holdout_only=True))
 return dict(all_identity=allrows,fit_identity=core+tail,holdout_identity=held,selection=selection,plan=plan,holdout_eligible_domains={k:len(v) for k,v in eligible.items()},controlled_overlap_in_new=len(control_overlaps),controlled_overlap_retained_in_fit=len(control_overlaps))

def labeled(identities_,lookup):
 out=[]
 for ident in identities_:
  g=lookup[ident['qid']]
  if any(g[k]!=ident[k] for k in ('qid','question_hash','trajectory_ids')) or len(g['labels'])!=10 or any(type(y) is not int or y not in (0,1) for y in g['labels']):raise ValueError('Own ten-trajectory strict EM join differs')
  out.append(dict(ident,labels=list(g['labels'])))
 return out

def run(config_path):
 from .train_g10_domain_statistics import canonical_core
 config_ref=proof(config_path);c=read(config_path)
 if (c['role']!='three_bank_internal_holdout_append_metadata_only' or c['source_counts']!=[30016,15616,12800] or c['holdout_per_domain']!=2560 or c['holdout_seed']!=9137 or c['append_order_seed']!=1729 or c['source_cursor']!=938 or c['target_cursor']!=1666 or c['global_questions']!=32 or c['GPU_submission'] is not False):raise ValueError('Exact predeclared CPU-only scope required')
 base=TARGET;base.mkdir(parents=True,exist_ok=True)
 with (base/'prepare.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (base/'complete.json').exists():
   done=read(base/'complete.json')
   if done['config']!=config_ref:raise ValueError('Existing completion config differs')
   for item in done['outputs'].values():checked(item)
   return done
  started=time.time();banks=[];sets=[];features=[];seen_features=set();sources=[]
  for ref,n,status in zip(c['source_banks'],c['source_counts'],c['source_statuses']):
   bank=read(checked(ref))
   if bank['status']!=status or bank['questions']!=n or bank['trajectories']!=10*n or bank['capture_layer']!=26 or bank['prefix_tokens']!=20 or bank['question_max_tokens']!=273 or bank['actor_updates']!=0 or bank['final_pool_read'] is not False:raise ValueError('Actual accepted qualified L26/P20 feature bank differs')
   gs=read(checked(bank['outputs']['groups.json']))['train'];identities(gs)
   if len(gs)!=n:raise ValueError('Accepted source group count differs')
   ownfeatures=read(checked(bank['outputs']['features-layer26.json']));count=0
   for feature in ownfeatures:
    if feature['path'] in seen_features:raise ValueError('Feature manifest appears in multiple banks')
    m=read(checked(feature))
    if m['status']!='LOSSLESS_HF_REPLAY_FEATURES_COMPLETE' or m['capture_layer']!=26 or m['prefix_tokens']!=20 or m['question_max_tokens']!=273 or m['dimension']!=3584 or m['model_revision']!=c['model_revision']:raise ValueError('Feature geometry or frozen actor revision differs')
    count+=m['count'];seen_features.add(feature['path']);features.append(feature)
   if count!=10*n:raise ValueError('Source accepted metadata trajectory count differs')
   banks.append(bank);sets.append(gs);sources.append(dict(complete=ref,groups=bank['outputs']['groups.json'],features=bank['outputs']['features-layer26.json'],questions=n,trajectories=count))
   immutable(base/('source-%d-accepted.json'%len(sets)),sources[-1])
   print(json.dumps(dict(status='ACTUAL_SOURCE_BANK_METADATA_ADVANCING',accepted_banks=len(sets),questions=sum(len(x) for x in sets),epoch=time.time())),flush=True)
  actual_config=read(checked(c['current_training_config']));checked(c['historical_plan'])
  if actual_config['plan']!=c['historical_plan'] or actual_config['fit_questions']!=30016 or actual_config['update_budget']!=938:raise ValueError('Current actual config does not bind exact original938 plan')
  oldplan=read(c['historical_plan']['path']);ingest=read(checked(c['identity_exclusion_config']));controlled_plan=read(checked(c['controlled6528_plan']))
  if len(controlled_plan['cohort_qids'])!=6528 or len(set(controlled_plan['cohort_qids']))!=6528:raise ValueError('Exact controlled6528 cohort required')
  controlled,canonical_proof=canonical_core(controlled_plan['cohort_qids'],ingest);excluded=[];scope_counts={}
  for name,n in [('diagnostic512',512),('selection2048',2048),('frozen_final4096',4096)]:
   scope=ingest['exclusions'][name]
   if scope['identity_only']is not True or scope['role']!=name or len(scope['groups'])!=n:raise ValueError('Explicit copied identity-only fixed exclusions required')
   identity_set(scope['groups']);excluded+=scope['groups'];scope_counts[name]=n
  parts=build_partitions(sets[0],sets[1]+sets[2],oldplan,controlled,excluded)
  # Freeze fixed G5 identity BEFORE creating labeled own-EM evaluation rows.
  immutable(base/'internal-holdout-selection.json',parts['selection']);selection_ref=proof(base/'internal-holdout-selection.json')
  held_ids=parts['holdout_identity'];hqs={g['qid'] for g in held_ids};hhs={g['question_hash'] for g in held_ids};coreqs={g['qid'] for g in sets[0]};corehs={g['question_hash'] for g in sets[0]};cq,ch=identity_set(controlled)
  if hqs&coreqs or hhs&corehs or hqs&set(cq) or hhs&ch:raise ValueError('Internal holdout identity exclusion failed')
  immutable(base/'identity-exclusion-acceptance.json',dict(role='internal_holdout_identity_exclusion_acceptance',merged_questions=58432,holdout_questions=5120,holdout_selection=selection_ref,merged_fixed_evaluation_qid_and_hash_overlap={k:0 for k in scope_counts},holdout_original30016_qid_overlap=0,holdout_original30016_hash_overlap=0,holdout_controlled6528_qid_overlap=0,holdout_controlled6528_hash_overlap=0,holdout_fixed_evaluation_qid_and_hash_overlap={k:0 for k in scope_counts},controlled6528_excluded_from_holdout_only=True,controlled_training_overlap_not_a_holdout_violation=True,labels_consulted_for_selection=False,scores_consulted_for_selection=False,identity_exclusion_config=c['identity_exclusion_config'],controlled_plan=c['controlled6528_plan'],canonical_projection=canonical_proof,final_source_refs_followed=False,final_outcomes_read=False))
  immutable(base/'append-plan.json',parts['plan'])
  original=Path(c['historical_plan']['path']).read_bytes();copy_path=base/'historical-plan-verbatim.json'
  if copy_path.exists() and copy_path.read_bytes()!=original:raise ValueError('Original938 byte copy changed')
  if not copy_path.exists():copy_path.write_bytes(original)
  lookup={g['qid']:g for group in sets for g in group};allgroups=labeled(parts['all_identity'],lookup);fit=labeled(parts['fit_identity'],lookup);held=labeled(parts['holdout_identity'],lookup)
  fixed=[]
  for g in held:fixed.append(dict(qid=g['qid'],question_hash=g['question_hash'],trajectory_ids=g['trajectory_ids'][5:10],labels=g['labels'][5:10],source_indices=[5,6,7,8,9],domain=json.loads(g['qid'])[0]))
  if len(allgroups)!=58432 or len(fit)!=53312 or len(fixed)!=5120 or parts['plan']['updates']!=1666 or parts['plan']['additional_updates']!=728:raise ValueError('Exact final source/split/append budgets differ')
  (base/'merged-bank').mkdir(exist_ok=True);immutable(base/'merged-bank/groups.json',{'train':allgroups});immutable(base/'features-layer26.json',features);immutable(base/'fit-g10-groups.json',{'train':fit});immutable(base/'internal-holdout-g5.json',{'internal_holdout':fixed})
  immutable(base/'evaluation-protocol.json',c['evaluation_protocol'])
  names=['merged-bank/groups.json','features-layer26.json','fit-g10-groups.json','internal-holdout-selection.json','internal-holdout-g5.json','append-plan.json','historical-plan-verbatim.json','evaluation-protocol.json','identity-exclusion-acceptance.json']
  if proof(copy_path)['sha256']!=c['historical_plan']['sha256'] or digest(fit[:30016])!=digest([{k:g[k] for k in ('qid','question_hash','trajectory_ids','labels')} for g in sets[0]]):raise ValueError('Original plan bytes or original30016 records changed')
  result=dict(status=STATUS,config=config_ref,source_banks=sources,outputs={n:proof(base/n) for n in names},questions=58432,trajectories=584320,fit_questions=53312,fit_trajectories=533120,internal_holdout_questions=5120,internal_holdout_actual_G5_trajectories=25600,holdout_domains={'nq':2560,'hotpotqa':2560},historical_questions=30016,historical_updates=938,append_questions=23296,append_updates=728,cumulative_future_updates=1666,historical_prefix_preserved_exactly=True,merged_ordered_qid_hash_sha256=digest([[g['qid'],g['question_hash']] for g in allgroups]),fit_ordered_qid_hash_sha256=digest([[g['qid'],g['question_hash']] for g in fit]),holdout_ordered_qid_hash_sha256=digest([[g['qid'],g['question_hash']] for g in held]),holdout_ordered_own_G5_IDs_sha256=digest([g['trajectory_ids'] for g in fixed]),historical_core_groups_canonical_sha256=digest(fit[:30016]),historical_source_groups_canonical_sha256=digest([{k:g[k] for k in ('qid','question_hash','trajectory_ids','labels')} for g in sets[0]]),historical_batches_sha256=digest(oldplan['batches']),historical_verbatim_sha256=proof(copy_path)['sha256'],holdout_eligible_domains=parts['holdout_eligible_domains'],controlled_overlap_in_new=parts['controlled_overlap_in_new'],controlled_overlap_retained_in_fit=parts['controlled_overlap_retained_in_fit'],fixed_evaluation_identity_overlap=0,exclusion_scope_counts=scope_counts,controlled_canonical_identity_projection=canonical_proof,own_G5_source_indices=[5,6,7,8,9],selection_frozen_before_own_EM_join=True,selection_ref=selection_ref,source_checkpoint_bound=False,future_fullstate_resume_and_checkpoint_binding_required=True,LR_or_recipe_chosen=False,actor_updates=0,probe_updates=0,GPU_submitted=False,model_or_tensor_payload_read_or_copied=False,final_labels_rollouts_predictions_read=False,original_warmup_question_unseen_claim=False,elapsed_seconds=time.time()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,epoch=time.time())
  immutable(base/'complete.json',result);return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
 if os.uname().sysname!='Linux':raise ValueError('Actual metadata and ownEM stay remote')
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(180,180))
 print(json.dumps(dict(complete=proof(TARGET/'complete.json'),result=run(a.config)))if (TARGET/'complete.json').exists() else json.dumps(run(a.config)),flush=True)
if __name__=='__main__':main()
