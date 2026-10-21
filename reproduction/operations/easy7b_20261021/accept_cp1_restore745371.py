"""Post-hoc authentication of actual all4 readbacks; never restores or trains."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

BASE=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021')
COMMIT='e082a23b5a65b302e2016aa0e5ab43b10b543d4f'
JOB='745371'
NAMES=['logs/run-manifest.json','logs/native-allocation-admission.json',
       'logs/full-state-restored-workers.json','logs/full-state-restored-driver.json',
       'logs/easy2560-restored-stream-before-next-outer.json']
NAMES += [f'logs/full-state-restored-rank-{rank}.json' for rank in range(4)]
missing=[name for name in NAMES if not (BASE/name).is_file()]
if missing:
    print(json.dumps(dict(status='WAITING_ACTUAL_ALL4_CP1_STATE_RESTORE_EVIDENCE',job=JOB,missing=missing,checked_unix=time.time(),actual_cuda_restore_accepted=False)))
    raise SystemExit(0)
assert os.getuid()==14090
sys.path.insert(0,str(BASE/'repo'))
from search_g1.easy2560_restore_runtime import verified_runtime_prefix
from search_g1.full_state_restore import digest,publish
run=json.loads((BASE/NAMES[0]).read_text());allocation=json.loads((BASE/NAMES[1]).read_text())
workers=json.loads((BASE/NAMES[2]).read_text());driver=json.loads((BASE/NAMES[3]).read_text());stream=json.loads((BASE/NAMES[4]).read_text())
assert (BASE/'job.id').read_text().strip()==JOB and run['source_commit']==COMMIT and allocation['slurm_job_id']==JOB
binding=run['full_state_restore'];assert binding['source_receipt_sha256']=='0a671ffd2941bfb417cd6968ee7b60d62e02ee67f0c400d929f1127aab7e5984'
source=json.loads(Path(binding['source_receipt']).read_text())
assert digest(binding['source_receipt'])==binding['source_receipt_sha256'] and source['source_job']=='745022'
assert source['step']==1 and source['actual_optimizer_writes_per_rank']==20 and source['hf_saved_tensor_count']==339
summary=verified_runtime_prefix(BASE,run)
assert summary['completed_outer']==1 and summary['writes']==20 and summary['post_callback_rng_captured'] is True
assert workers['slurm_job_id']==JOB and workers['source_commit']==COMMIT and workers['actual_world_size']==4
assert workers['source_receipt_sha256']==binding['source_receipt_sha256']
files={record['relative_path']:record for record in source['files']}
for row in workers['ranks']:
    rank=row['rank'];individual=json.loads((BASE/f'logs/full-state-restored-rank-{rank}.json').read_text())
    assert individual==row and row['run_id']==run['run_id'] and row['world_size']==4
    assert row['actual_model_collective_readback_equal'] is True
    assert row['source_state_full_sha256']==files[f'training_state/rank-{rank}.pt']['full_read_sha256']
    assert row['source_receipt_sha256']==binding['source_receipt_sha256']
assert allocation['accepted'] is True and run['training_world_size']==4
assert workers['source_step']==1 and workers['actual_optimizer_writes_per_rank']==20
assert driver['slurm_job_id']==JOB and driver['run_id']==run['run_id']
assert driver['source_receipt_sha256']==binding['source_receipt_sha256'] and driver['source_step']==1
assert driver['source_checkpoint']==source['checkpoint'] and driver['source_job']=='745022'
assert driver['actual_driver_dataequal'] is True and driver['source_driver_full_sha256']==source['driver_file']['full_read_sha256']
assert driver['actual_candidate_source_cursor']==stream['candidate_source_cursor']==2560
assert stream['completed_outer']==1 and stream['current_world_size']==4
assert stream['actual_candidate_stream_readback_equal'] is True and stream['actual_driver_rng_readback_equal'] is True
records=[dict(path=name,bytes=(BASE/name).stat().st_size,sha256=digest(BASE/name))for name in NAMES]
result=dict(status='ACTUAL_CP1_NATIVE4_STATE_RESTORE_ACCEPTED_TRAINING_FOLLOWUP_PENDING',job=JOB,
    source_commit=COMMIT,run_id=run['run_id'],source_job='745022',source_completed_outer=1,
    model_tensors=339,actual_optimizer_writes_per_rank=20,candidate_source_cursor=2560,
    actual_all4_model_optimizer_scheduler_rng_dataequal=True,actual_cuda_restore_accepted=True,
    restore_performed_by_this_cpu_reader=False,gpu_allocations_by_this_reader=0,new_training_updates_by_this_reader=0,
    post_callback_rng_captured=True,actual_worker_processes=[row['process_identity']for row in workers['ranks']],
    verified_actual_producer_evidence=records,checked_unix=time.time(),
    required_followup_completed_outers=[2,3,4,5,6],required_followup_next_outer_started=7,goal_complete=False)
output=BASE/'preparation/actual-cp1-state-restore-acceptance.json'
if not output.exists():publish(output,result)
else:
    prior=json.loads(output.read_text());assert prior['verified_actual_producer_evidence']==records;result=prior
print(json.dumps(result))
