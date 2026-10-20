import pathlib,json,hashlib,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');B=A/'peer-expanded-identity-v1'
def read(p):return json.loads(pathlib.Path(p).read_text())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def fp(p):
 p=pathlib.Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):
 v=fp(r['path']);assert v==r;return r['path']
s=read(B/'complete.json');plan=read(checked(s['identity_plan']));old=read(checked(s['prior_identity']));raw=read(checked(s['raw_complete']));groups=read(checked(raw['outputs']['groups']))['train'];by={g['qid']:g for g in groups}
chosen=plan['groups'];new=plan['remaining_groups'];assert len(chosen)==30016 and len(new)==21824 and len(old['groups'])==8192
assert chosen[:8192]==old['groups'] and chosen[8192:]==new and set(g['qid'] for g in old['groups']).isdisjoint(g['qid'] for g in new)
assert len({g['qid'] for g in chosen})==len({g['question_hash'] for g in chosen})==30016
assert len({t for g in chosen for t in g['trajectory_ids']})==300160
for g in chosen:
 assert set(g)=={'qid','question_hash','trajectory_ids'};source=by[g['qid']]
 assert g['question_hash']==source['question_hash'] and g['trajectory_ids']==source['trajectory_ids'] and len(g['trajectory_ids'])==10
assert digest([g['qid'] for g in chosen])==s['ordered_qid_sha256'] and digest([g['qid'] for g in new])==s['ordered_new_qid_sha256']
assert len(plan['omitted_tail_identities'])==13 and s['onepass_training_updates']==938 and not s['labels_consulted'] and not s['scores_consulted']
assert not s['feature_capture_started'] and not s['probe_training_started'] and not s['untouched_final_labels_or_rollouts_read']
result=dict(status='EXPANDED_30016_REAL_G10_IDENTITIES_INDEPENDENT_SOURCE_IDS_HASHES_DISJOINTNESS_AND_SHA_PASS',epoch=time.time(),questions=30016,existing_preserved=8192,new_questions=21824,new_trajectories_needing_capture=218240,onepass_B32_updates=938,identity_plan=s['identity_plan'],complete=fp(B/'complete.json'),old_new_question_disjoint=True,all_question_text_hashes_unique=True,all_real_trajectory_IDs_unique=True,full_raw_EM_recomputation_inherited_from_accepted_ingest=True,correctness_labels_not_consulted_by_this_audit=True,GPU_jobs_started=0,model_tensors_read=False,final_labels_or_rollouts_read=False)
p=B/'independent-identity-audit.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,sort_keys=True)+'\n');tmp.replace(p);result['audit']=fp(p);print(json.dumps(result))
