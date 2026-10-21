"""Once-only recovery of failed sparse diagnostic; keep dense math and CP1."""
import json, os, subprocess, sys, time
from pathlib import Path

b = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-20261021')
failed = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-sparse-cp1-20261021')
commit = json.loads(sys.stdin.read())['commit']
sys.path.insert(0, str(b/'source'/commit))
from reproduction.easy2560_joint4.control import submit, publish, sha
assert os.getuid() == 14090 and len(commit) == 40
if any((b/p).exists() for p in ('job.id','preparation/training-submit-claim.json','preparation/training-submission.json','logs/run-manifest.json')):
    print(json.dumps(dict(status='EXISTING_FALLBACK_CLAIM_INSPECT_ACTUAL_DO_NOT_RESUBMIT', base=str(b))))
    raise SystemExit(0)
q = subprocess.run(['squeue','-j','745258','-h','-o','%i|%T'],capture_output=True,text=True,timeout=10)
a = subprocess.run(['sacct','-j','745258','-n','-P','-o','JobIDRaw,State,ExitCode'],capture_output=True,text=True,timeout=10)
assert not q.stdout.strip() and '745258|FAILED|1:0' in a.stdout
report_path = failed/'diagnostics/cp1-sparse-numerical-qualification.json'
report = json.loads(report_path.read_text())
assert report['accepted'] is False and report['optimizer_calls'] == report['sampling_calls'] == 0
assert report['actual_optimizer_state_restored'] is False and all(r['update_calls'] == dict(optimizer_step=0,scheduler_step=0) for r in report['ranks'])
receipt = failed/'preparation/cp1-full-state-source-receipt.json'
assert sha(receipt) == '0a671ffd2941bfb417cd6968ee7b60d62e02ee67f0c400d929f1127aab7e5984'
source = json.loads(receipt.read_text())
assert source['step'] == 1 and source['actual_optimizer_writes_per_rank'] == 20 and source['hf_saved_tensor_count'] == 339
(b/'preparation').mkdir(parents=True,exist_ok=True)
external = b/'preparation/external-retriever-receipt.json'
publish(external, json.loads((failed/'preparation/external-retriever-receipt.json').read_text()))
comparison = report['ranks'][0]['numerical_comparison']['comparisons']
publish(b/'preparation/dense-exact-fallback-decision.json',dict(epoch=time.time(),source_commit=commit,failed_own_diagnostic_job='745258',failed_job_accounting=a.stdout,diagnostic_report_path=str(report_path),diagnostic_report_sha256=sha(report_path),production_updates_in_failed_diagnostic=0,reason='Strict gradient qualification not met; original dense repeats also exceed the same threshold. Sparse remains unqualified; do not relax thresholds.',comparison=comparison,profile='exact-cache-v1',selected_policy_logits=False,exact_retrieval_cache=True,source_completed_outer=1,source_receipt_sha256=sha(receipt),external_service_job='745198',scientific_recipe_unchanged=True,manual_deletions=0))
value = submit(b,commit,actor_micro=8,restore_receipt=str(receipt),infra_profile='exact-cache-v1',external_retriever_receipt=str(external))
print(json.dumps({k:value.get(k) for k in ('job','source_commit','returncode','stdout','stderr','actor_micro_batch_per_rank','training_world_size','infra_profile')}))
