"""Remote CPU independent all32768 ownEM/q and hard/random plan acceptance."""
import pathlib,json,gzip,hashlib,math,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
H=B/'algorithm-improvement-20261021/hard-enrichment-20261021'
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(ref):
 p=pathlib.Path(ref['path']);assert p.is_absolute() and p.resolve()==p and sha(p)==ref['sha256']
 if 'bytes' in ref:assert p.stat().st_size==ref['bytes']
 return p
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def rank(seed,scope,qid):return hashlib.sha256(canonical(['training-hard-enrichment-v1',seed,scope,qid])).hexdigest()
def product(values):
 result=1.
 for value in values:result*=value
 return result
def own(g):return {k:g[k] for k in ('qid','question_hash','trajectory_ids','labels','split')}
def distribution(groups):
 ks=[sum(g['labels']) for g in groups];return [ks.count(i) for i in range(6)]
done=read(H/'selection-complete.json');assert done['status']=='TRAIN_ONLY_HARD_AND_RANDOM22896Q_CLASS_MATCHED_PLANS_COMPLETE'
assert done['GPU_used'] is False and done['actor_updates']==done['probe_updates']==0
config=read(checked(done['source_config']));candidate=read(checked(done['source_candidates']))
score=read(checked(done['source_scoring_receipt']));assert score['status']=='E4_PREFIX20_TRAIN_ONLY_CANDIDATE_SCORING_COMPLETE' and score['questions']==32768 and score['trajectories']==163840
assert score['binding']['candidate_plan']==done['source_candidates'] and score['binding']['checkpoint']['sha256']==done['checkpoint']['sha256']
assert score['validation_or_final_scored'] is False and score['frozen_parameters_unchanged'] is True and score['optimizer_constructed'] is False
assert score['actor_optimizer_updates']==score['probe_optimizer_updates']==0
assert done['source_predictions']==score['predictions']
checked(done['checkpoint']);bank=read(checked(done['source_bank']));source={g['qid']:g for g in bank['train']};assert len(source)==169028
matched=read(checked(done['source_matched_plan']));core=[dict(source[q],split='train') for q in matched['cohort_qids']];assert len(core)==6528
core=[own(g) for g in core];cq={g['qid'] for g in core};ch={g['question_hash'] for g in core}
excludedq=set();excludedhash=set()
for ref in done['exclusion_refs']:
 audit=read(checked(ref));assert audit['role']=='training_mining_exclusions'
 for g in audit['groups']:
  assert set(g)=={'qid','question_hash'};excludedq.add(g['qid']);excludedhash.add(g['question_hash'])
assert candidate['role']=='training_hard_mining_candidates' and candidate['source_checkpoint']==candidate['checkpoint']
assert candidate['summary']['questions']==32768 and candidate['summary']['trajectories']==163840
assert candidate['summary']['question_selection_uses_labels'] is False and candidate['summary']['question_selection_uses_scores'] is False
assert candidate['summary']['validation_or_final_questions_used'] is False and candidate['summary']['unseen_in_initial169028_warmup_claim'] is False
assert candidate['summary']['unseen_in_previous_finetuning_claim'] is False and candidate['summary']['excluded_current_matched6528_fit'] is True
pool=candidate['groups'];assert len(pool)==32768 and len({g['qid'] for g in pool})==32768 and len({g['question_hash'] for g in pool})==32768
assert digest([g['qid'] for g in pool])==candidate['summary']['ordered_qid_sha256']
assert digest([g['trajectory_ids'] for g in pool])==candidate['summary']['ordered_own_trajectory_ids_sha256']
for g in pool:
 assert g['split']=='train' and g['qid'] not in excludedq and g['question_hash'] not in excludedhash
 for key in ('qid','question_hash','trajectory_ids','labels'):assert g[key]==source[g['qid']][key]
assert not cq&{g['qid'] for g in pool} and not ch&{g['question_hash'] for g in pool}
with gzip.open(str(checked(done['source_predictions'])),'rt') as f:rows=[json.loads(line) for line in f]
assert len(rows)==32768;scored={};maximum_q_error=0.
for row,g in zip(rows,pool):
 assert row['qid']==g['qid'] and row['trajectory_ids']==g['trajectory_ids'] and row['strict_em']==g['labels']
 assert len(row['p_j'])==5 and all(math.isfinite(p) and 0<=p<=1 for p in row['p_j'])
 q=1-product(row['p_j'])-product([1-p for p in row['p_j']]);maximum_q_error=max(maximum_q_error,abs(q-row['predicted_mixed']))
 assert row['qid'] not in scored
 scored[row['qid']]=dict(group=g,q=row['predicted_mixed'],K=sum(g['labels']))
assert maximum_q_error<1e-6
wrong=[r for r in scored.values() if r['K']==0];correct=[r for r in scored.values() if r['K']==5];mixed=[r for r in scored.values() if 0<r['K']<5]
seed=config['seed'];quota=4092;assert min(len(wrong),len(correct),len(mixed)//2)>=quota
def hashed(rows,scope):return sorted(rows,key=lambda r:(rank(seed,scope,r['group']['qid']),r['group']['qid']))
low=sorted(mixed,key=lambda r:(r['q'],rank(seed,'hard-low-Mixed-tie',r['group']['qid'])))[:quota];low_ids={r['group']['qid'] for r in low}
random_mixed=hashed(mixed,'random-control-Mixed')[:2*quota]
strata=dict(targeted=dict(
 high_q_allwrong=sorted(wrong,key=lambda r:(-r['q'],rank(seed,'hard-allwrong-tie',r['group']['qid'])))[:quota],
 high_q_allcorrect=sorted(correct,key=lambda r:(-r['q'],rank(seed,'hard-allcorrect-tie',r['group']['qid'])))[:quota],
 low_q_Mixed=low,Mixed_hash_anchor=hashed([r for r in mixed if r['group']['qid'] not in low_ids],'hard-remaining-Mixed-anchor')[:quota]),
 random_control=dict(random_allwrong=hashed(wrong,'random-control-allwrong')[:quota],random_allcorrect=hashed(correct,'random-control-allcorrect')[:quota],random_Mixed_A=random_mixed[:quota],random_Mixed_B=random_mixed[quota:]))
branches={}
for name,parts in strata.items():
 actual=done['outputs'][name];groups=read(checked(actual['groups']))['train'];plan=read(checked(actual['plan']))
 supplement=[r['group'] for part in parts.values() for r in part]
 expected=sorted(core+supplement,key=lambda g:(rank(seed,'paired-training-epoch-order',g['qid']),g['qid']))
 assert groups==expected and len(groups)==22896 and len({g['qid'] for g in groups})==22896 and len({g['question_hash'] for g in groups})==22896
 qids=[g['qid'] for g in groups];assert plan['cohort_qids']==qids and plan['ordered_qid_sha256']==digest(qids)
 assert plan['batches']==[qids[i:i+48] for i in range(0,22896,48)] and len(plan['batches'])==477 and plan['global_groups']==48 and plan['update_budget']==477
 assert plan['source_checkpoint']==candidate['checkpoint'] and plan['each_question_once'] is True and plan['epochs']==1
 summary=actual['summary'];assert summary['full_K_histogram']==distribution(groups) and summary['supplement_K_histogram']==distribution(supplement)
 assert summary['supplement_class_counts']==dict(allwrong=4092,allcorrect=4092,Mixed=8184)
 assert summary['unseen_in_initial169028_warmup_claim'] is False and summary['unseen_in_previous_finetuning_claim'] is False and summary['validation_or_final_questions_used'] is False
 for key,items in parts.items():
  s=summary['strata'][key];assert s['questions']==4092 and s['qid_sha256']==digest([r['group']['qid'] for r in items])
  assert abs(s['q_min']-min(r['q'] for r in items))<1e-12 and abs(s['q_max']-max(r['q'] for r in items))<1e-12
 branches[name]=dict(groups=actual['groups'],plan=actual['plan'],summary=summary,independently_reconstructed_strata_and_exact_plan_match=True)
assert done['controls']['same_core'] and done['controls']['same_unique_count'] and done['controls']['same_macro_class_counts'] and done['controls']['same_seed'] and done['controls']['same_update_budget']
report=dict(status='ALL32768_OWN_EM_FROZEN_E4_SHA_Q_AND_HARD_RANDOM_EXACT_SELECTION_INDEPENDENT_PASS',observed_epoch=time.time(),
 selection_complete=fp(H/'selection-complete.json'),source_candidates=done['source_candidates'],source_scoring_complete=done['source_scoring_receipt'],source_predictions=done['source_predictions'],
 source_E4_checkpoint_full_SHA_freshly_verified=True,candidates=32768,own_trajectories=163840,candidate_K_histogram=distribution(pool),q_formula_max_absolute_error=maximum_q_error,
 branches=branches,controls=done['controls'],untouched_final_read=False,final_identity_source_files_opened=False,
 torch_imported=False,model_or_optimizer_loaded=False,raw_predictions_returned=False)
print(json.dumps(report))
