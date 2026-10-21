import json,sys,subprocess,time,os
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-sparse-cp1-20261021');commit='921610632eece86be319098351c21b450fd4a236';old=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021')
sys.path.insert(0,str(b/'source'/commit))
from reproduction.easy2560_joint4.control import bootstrap,publish,restore_binding,launch_command
from reproduction.easy2560_joint4.retrieval_proxy import external_binding
assert os.getuid()==14090
claim=b/'preparation/training-submit-claim.json';submission=b/'preparation/training-submission.json'
if any(p.exists()for p in [claim,submission,b/'job.id',b/'logs/run-manifest.json']):print(json.dumps(dict(status='EXISTING_TRANSITION_CLAIM_INSPECT_ACTUAL_NOT_RESUBMIT',claim_exists=claim.exists())));raise SystemExit(0)
exit_path=b/'preparation/cp1-source-audit/exit.json'
if not exit_path.exists():print(json.dumps(dict(status='WAITING_CP1_SOURCE_AUDIT')));raise SystemExit(0)
result=json.loads(exit_path.read_text());assert result['returncode']==0,result
receipt=b/'preparation/cp1-full-state-source-receipt.json';r=json.loads(receipt.read_text());assert r['step']==1 and r['world_size']==4 and r['hf_saved_tensor_count']==339 and r['actual_optimizer_writes_per_rank']==20
assert r['post_callback_rng_captured'] is True and r['actual_cuda_restore_accepted'] is False
queue=subprocess.run(['squeue','-j','745022','-h','-o','%i|%T'],capture_output=True,text=True,timeout=10);assert not queue.stdout.strip(),queue.stdout
assert json.loads((old/'preparation/cp1-transition-scancel-result.json').read_text())['returncode']==0
bootstrap(b,commit)
restoration=restore_binding(receipt);external_receipt=b/'preparation/external-retriever-receipt.json';external=external_binding(external_receipt)
cmd=launch_command(b,commit,8,str(receipt),'sparse-exact-v1',str(external_receipt))
entry=str(b/'source'/commit/'reproduction/easy2560_joint4/train.sh');index=cmd.index(entry);cmd[index]=str(b/'source'/commit/'reproduction/easy2560_joint4/transition_cp1.sh')
raw=old/'artifacts/mixed-candidate-pools/857092f9bb234821ab62664bf273a1f0-pool-0/0.pt';assert raw.is_file();cmd.append(str(raw))
publish(claim,dict(source_commit=commit,command=cmd,actor_micro_batch_per_rank=8,logical_micro_batch_per_rank=4,infra_profile='sparse-exact-v1',external_retriever=external,**restoration,personal_storage_budget_bytes=2*10**12,created_unix=time.time(),explicit_cp1_numerical_transition=True,numerical_failure_policy='No production on failure; retain evidence; no threshold relaxation',authorization_timestamps=['2026-10-21T10:28:22.987Z','2026-10-21T10:29:25.701Z','2026-10-21T10:31:07.043Z']))
r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);value=dict(source_commit=commit,command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,checked_unix=time.time(),actor_micro_batch_per_rank=8,training_world_size=4,infra_profile='sparse-exact-v1',external_retriever=external,**restoration)
if r.returncode==0:
 job=r.stdout.strip().split(';')[0];assert job.isdecimal();value['job']=job
 with (b/'job.id').open('x')as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
publish(submission,value)
print(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,job=value.get('job'),base=str(b),source_commit=commit,restore_outer=1,external_service_job='745198',numerical_and_actual_restore_pending=True)))
raise SystemExit(bool(r.returncode))
