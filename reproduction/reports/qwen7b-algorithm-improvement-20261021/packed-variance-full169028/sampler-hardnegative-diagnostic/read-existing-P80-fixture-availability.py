"""Availability and identities only; do not read token payloads, models or final data."""
import hashlib
import json
from pathlib import Path
import time
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
P80=A/'online-mixed-layer-prefix-20261022-v1'
P20=A/'online-packed-full169028-M3to10-P20-20261022-v1/fixture'
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def metadata(path):
 path=Path(path);raw=path.read_bytes();return json.loads(raw),dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
fixture,fref=metadata(P80/'fixture-complete.json');assert fixture['status']=='ONLINE_MIXED_MATCHED8192_G10_FIXED2048_TOKENS_READY'
train,tref=metadata(P80/'inputs/train-groups.json');dev,dref=metadata(P80/'inputs/validation-groups.json');fixed_train,ftref=metadata(P80/'inputs/fixed-train2048.json')
assert len(train)==8192 and len(dev)==2048 and len(fixed_train)==2048 and not(set(train)&set(dev))
assert all(len(g['trajectory_ids'])==len(g['labels'])==10 for g in train.values())
assert all(len(g['trajectory_ids'])==len(g['labels'])==5 for g in dev.values())
full_train,full_tref=metadata(P20/'inputs/train-groups.json');full_dev,full_dref=metadata(P20/'inputs/validation-groups.json');assert len(full_train)==169028 and len(full_dev)==2048
assert set(train)<=set(full_train)and set(dev)==set(full_dev)
assert all(train[q]['trajectory_ids']==full_train[q]['trajectory_ids']and train[q]['labels']==full_train[q]['labels']for q in train)
for q,g in dev.items():
 whole=full_dev[q];by=dict(zip(whole['trajectory_ids'],whole['labels']))
 assert len(by)==10 and all(by[t]==y for t,y in zip(g['trajectory_ids'],g['labels']))
assert len({r['qid']for r in fixed_train})==2048 and all(r['qid']in train for r in fixed_train)
shards=[]
for ref in fixture['token_shards']:
 p=Path(ref['path']);st=p.stat();assert st.st_size==ref['bytes']
 shards.append(dict(declared_complete_proof=ref,exists=True,actual_bytes=st.st_size,fresh_payload_SHA_not_read=True))
full_fixture,full_fref=metadata(P20/'fixture-complete.json')
assert full_fixture['training_questions']==169028 and full_fixture['validation_questions']==2048
bank,bref=metadata(A/'validation-g10-20261021/materialized/bank/groups.json');audit,aref=metadata(A/'validation-g10-20261021/materialized/bank/source-audit.json')
assert {g['qid']for g in bank['validation']}==set(dev)
full_refs=[]
for batches in audit['events'].values():
 for batch in batches:
  ref=batch['phase_payloads']['full'];p=Path(ref['path']);st=p.stat();assert st.st_size==ref['bytes'];full_refs.append(dict(declared_complete_proof=ref,exists=True,actual_bytes=st.st_size,fresh_payload_SHA_not_read=True))
print(json.dumps(dict(status='EXISTING_P80_TOKEN_FIXTURE_IDENTITIES_AND_SOURCE_AVAILABILITY_ONLY',time=time.time(),
 P80_fixture=fref,train_groups=tref,validation_fixed_G5_groups=dref,fixed_train2048=ftref,
 train_questions=8192,train_own_G10_trajectories=81920,validation_questions=2048,validation_current_P80_trajectories=10240,
 train_qid_sha256=digest(sorted(train)),validation_qid_sha256=digest(sorted(dev)),fixed_train2048_qid_sha256=digest([r['qid']for r in fixed_train]),
 train_question_and_own10_identities_exactly_match_full169028=True,validation_question_identities_exactly_match_full2048=True,
 P80_existing_fixed_G5_is_same_own_subset_and_EM_as_full_validation=True,
 metadata_train_validation_qid_overlap=0,token_shards=shards,
 source_full_P20_fixture=full_fref,full_P20_train_groups=full_tref,full_P20_validation_groups=full_dref,
 P20_compact_policy20_cannot_supply_P80=True,
 can_reuse_without_new_rollouts='Existing8192 ownG10 P80 token shards + exact2048 fixedG5 dev; online prefill still required, no hidden cache reused.',
 missing_for_validation_M10='Existing P80 dev token fixture has only5 selected trajectories/question; the other5 require CPU reconstruction from existing full journals before randomM10 evaluation.',
 existing_validation_G10_bank=bref,existing_validation_full_source_audit=aref,existing_validation_full_journal_references=full_refs,
 no_new_reconstruction_or_regeneration_performed=True,no_gpu_or_deployment_or_Slurm_changes=True,
 token_payloads_or_hidden_features_or_CP_not_read=True,final4096_read=False),sort_keys=True))
