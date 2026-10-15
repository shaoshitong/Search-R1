import sys
from pathlib import Path
REPO=Path('/Users/shaoshitong/Documents/Search-G1/qprobe-restore-only-20261016');sys.path.insert(0,str(REPO))
from reproduction.h1008_qprobe.management_transport import remote
script='''import json,hashlib,sys,datetime
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-g1-restore-only-38a431d-20261016')
proof=json.loads((root/'publication-proof.json').read_text());receipt=json.loads((root/'deploy-receipt.json').read_text())
assert proof['published_control_commit']==receipt['source_commit']=='38a431d0e2b0166e5941b45115d5d837968c7a89'
for name,sha in proof['files'].items():
 p=root/'control'/name;assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==sha,name
sys.path.insert(0,str(root/'control'))
from reproduction.restore_only.contract import bind_step
source=Path('/mnt/data/shaoshitong/search-g1-qprobe3-binary10-c5fc096-20261015')
p=source/'artifacts/question-probe/completed-pairs/global_step_5/source-receipt.json'
raw=p.read_bytes();paired=json.loads(raw);parentpath=Path(paired['source_run_manifest']['path']);parentraw=parentpath.read_bytes();parent=json.loads(parentraw)
parent_sha=hashlib.sha256(parentraw).hexdigest();assert parent_sha==paired['source_run_manifest']['full_read_sha256']
prospective=root/'acceptance-f77-after200-prospective'
bound=bind_step(parent,paired,p,prospective)
assert bound['budget']==parent['budget'] and bound['input_sha256']==parent['input_sha256'] and bound['initial_model_sha256']==parent['initial_model_sha256'] and bound['reference_model']==parent['reference_model']
assert paired['step']==5 and bound['full_state_restore']['actual_optimizer_writes_per_rank']==paired['attempt_progress']['cumulative_writes']
assert not prospective.exists()
print(json.dumps(dict(control_commit=proof['published_control_commit'],verified_code_files=len(proof['files']),independent_second_target_full_sha_match=True,actual_cp5_pure_binding_compatible=True,actual_cp5_receipt_path=str(p),actual_cp5_receipt_sha256=hashlib.sha256(raw).hexdigest(),actual_cp5_source_manifest_path=str(parentpath),actual_cp5_source_manifest_sha256=parent_sha,actual_parent_local_micro=parent['actor_micro_batch_per_rank'],actual_parent_profile=parent['profile'],actual_parent_execution_site=parent['execution_site'],actual_parent_precision_profile=parent['precision_profile'],actual_parent_fixed_ref=parent['reference_model'],saved_actual_optimizer_writes=paired['attempt_progress']['cumulative_writes'],scientific_budget_identity_preserved=True,prospective_output_created=False,all_five_binding_prepared=False,native_submitted=False,gpu_started=False,actual_cuda_restore_verified=False,training_modified=False,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
'''
print(remote(script,Path(__file__).parent/'restore-only-38a431d0-second-target-and-pure-binding.json','/var/folders/6k/lf0fpt5j2ps0ztkt_tyvwntw0000gn/T/qprobe3-mgmt-88myknuw/control.sock'))
