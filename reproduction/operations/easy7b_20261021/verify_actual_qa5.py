"""CPU-only post-hoc QA5 identity, saved-boundary and update verification."""
import hashlib
import json
from pathlib import Path
import sys
import time

BASE=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021')
sys.path.insert(0,str(BASE/'repo'))
from reproduction.easy2560_joint4.final_cpu_acceptance import canonical,scores
from search_g1.easy2560_rolling_checkpoint import verify_checkpoint_manifest
run=json.loads((BASE/'logs/run-manifest.json').read_text())
identity=canonical(BASE,'validation')
records=[json.loads(line)for line in(BASE/'logs/validation-identities.jsonl').read_text().splitlines()if line]
fixed=[row for row in records if row.get('step')==5 and row.get('split')=='validation']
assert len(fixed)==1
result=scores(fixed[0],identity,run,5,split='validation',final=False)
after=json.loads((BASE/'logs/formal-inline-development-after.json').read_text());rows=after['rows']
assert after['run_id']==run['run_id']and after['step']==5 and after['role']=='formal_inline_development_evaluation'
assert after['validation_sha256']==identity['sha256']and after['question_ids']==fixed[0]['question_ids']and after['data_sources']==fixed[0]['data_sources']
assert len(rows)==512 and[row['question_id']for row in rows]==fixed[0]['question_ids']and[row['data_source']for row in rows]==fixed[0]['data_sources']
assert[row['em']for row in rows]==fixed[0]['em']and[row['f1']for row in rows]==fixed[0]['f1']
assert after['weighted_em']==result['weighted_em']and after['weighted_f1']==result['weighted_f1']
manifest=BASE/'preparation/rolling-checkpoints/global_step_5/full-manifest.json'
verified=verify_checkpoint_manifest(BASE,manifest,expected_run_id=run['run_id'],full_sha=False)
complete=json.loads((BASE/'checkpoints/search-g1-common8/actor/global_step_5/complete.json').read_text())
assert complete['step']==5 and complete['run_id']==run['run_id']and complete['finite_all_saved_tensors']is True
ranks=[]
for rank in range(4):
 ledger=[json.loads(line)for line in(BASE/('logs/scheduler-rank-%d.jsonl'%rank)).read_text().splitlines()if line]
 row=next(row for row in reversed(ledger)if row.get('logical_update')==5 and row.get('kind')=='COMPLETE')
 assert row['writes_before']==80 and row['writes_after']==100 and row['actual_optimizer_writes']==20 and row['scheduler_advances']==1 and row['scheduler_last_epoch_after']==5 and row['full_budget_complete']is True
 ranks.append(dict(rank=rank,writes_before=80,writes_after=100,actual_optimizer_writes=20,scheduler_advances=1,scheduler_epoch=5))
print(json.dumps(dict(epoch=time.time(),job=(BASE/'job.id').read_text().strip(),run_id=run['run_id'],source_commit=run['source_commit'],
 status='ACTUAL_FIXED_QA5_CANONICAL_OCCURRENCES_SCORES_AND_SEALED_NATIVE4_BOUNDARY_PASS',fixed512=result,
 formal_artifact_rows_equal=True,rolling_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),all4_updates=ranks,
 model_tensors=complete.get('validated_tensor_count'),checkpoint_metadata_and_rolling_boundary_verified=True,
 model_state_reloaded=False,gpu_allocations=0,training_modified=False,actual_cuda_restore_accepted=False)))
