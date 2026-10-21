"""Submit one replacement only after the own legacy-layer restore job failed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021')
FAILED = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-20261021')
commit = json.load(sys.stdin)['commit']
assert os.getuid() == 14090 and len(commit) == 40
if any((BASE/name).exists() for name in ['job.id','preparation/training-submit-claim.json','preparation/training-submission.json','logs/run-manifest.json']):
    print(json.dumps(dict(status='EXISTING_REPAIR_CLAIM_INSPECT_ACTUAL_DO_NOT_RESUBMIT',base=str(BASE))))
    raise SystemExit(0)
sys.path.insert(0, str(BASE/'source'/commit))
from reproduction.easy2560_joint4.control import publish, sha, submit
queue = subprocess.run(['squeue','-j','745320,745022','-h','-o','%i|%T'],capture_output=True,text=True,timeout=12)
accounting = subprocess.run(['sacct','-j','745320','-n','-P','-o','JobIDRaw,State,ExitCode'],capture_output=True,text=True,timeout=12)
assert queue.returncode == accounting.returncode == 0 and not queue.stdout.strip()
assert '745320|FAILED|1:0' in accounting.stdout
preserved = FAILED/'preparation/failed-job-745320/failure-evidence.json'
failure = json.loads(preserved.read_text())
assert failure['status']=='FAILED_OWN_JOB_PRESERVED_NO_NEW_UPDATES' and failure['actual_new_optimizer_writes']==0
receipt = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-sparse-cp1-20261021/preparation/cp1-full-state-source-receipt.json')
assert sha(receipt)=='0a671ffd2941bfb417cd6968ee7b60d62e02ee67f0c400d929f1127aab7e5984'
source = json.loads(receipt.read_text())
assert source['step']==1 and source['world_size']==4 and source['actual_optimizer_writes_per_rank']==20 and source['hf_saved_tensor_count']==339
external = BASE/'preparation/external-retriever-receipt.json'
publish(external,json.loads((FAILED/'preparation/external-retriever-receipt.json').read_text()))
publish(BASE/'preparation/cp1-structure-repair-decision.json',dict(checked_unix=time.time(),source_commit=commit,
    failed_job='745320',preserved_failure=str(preserved),preserved_failure_sha256=sha(preserved),
    failure_cause='3B36layer assumption rejected real7B28layer restore before Adam/RNG',
    fix='Use actual loaded config.num_hidden_layers, preserve FA2/FP32master/BF16 checks; no forward inserted',
    source_completed_outer=1,source_optimizer_writes_per_rank=20,source_receipt_sha256=sha(receipt),
    failed_new_optimizer_writes=0,scientific_recipe_unchanged=True,manual_deletions=0,existing_service_job='745198'))
value = submit(BASE,commit,actor_micro=8,restore_receipt=str(receipt),
               infra_profile='exact-cache-v1',external_retriever_receipt=str(external))
print(json.dumps({key:value.get(key) for key in ['job','source_commit','returncode','stdout','stderr','actor_micro_batch_per_rank','training_world_size','infra_profile']}))
