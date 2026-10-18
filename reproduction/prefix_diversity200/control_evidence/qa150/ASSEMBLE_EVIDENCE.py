"""Assemble only small local QA150 evidence from the completed published wrapper."""
from pathlib import Path
import base64,hashlib,json,time
folder=Path(__file__).resolve().parent;out=folder.parent
read=lambda path:json.loads(path.read_text())
v=read(folder/'REMOTE_RESULT.json');p=read(folder/'PAYLOAD.json');publication=read(folder/'PUBLICATION.json')
assert v['status']=='PASS' and v['execution']['returncode']==0
raw=base64.b64decode(v['proof_base64']);assert hashlib.sha256(raw).hexdigest()==v['proof_sha256']
with(folder/'PROOF.json').open('xb')as stream:stream.write(raw)
a=v['acceptance'];assert a['qa_count']==512 and a['actor_version']==p['run_id']+'-actor%04d'%p['outer']
assert {r['rank']for r in a['rank_rows']}==set(range(8)) and all(r['count']==64 and r['compute_dtype']=='torch.bfloat16'for r in a['rank_rows'])
assert a['correct']/a['qa_count']==a['em'] and a['config_sha256']==p['config_sha256']
for stage in('preflight','postflight'):
    assert not v[stage]['cgroup_exists'] and v[stage]['cgroup_pids']==[]
    assert 'MainPID=0'in v[stage]['systemd'] and 'ActiveState=inactive'in v[stage]['systemd'] and 'SubState=dead'in v[stage]['systemd']
assert v['cost']['nice']==19 and len(v['cost']['cpu_affinity'])==1 and v['cost']['CUDA_VISIBLE_DEVICES']==''
identity=dict(run_id=p['run_id'],outer=p['outer'],node='h100-11',qa_source_commit=p['source_commit'],
    qa_source_overlay_sha256=v['source_overlay_sha256'],wrapper_publication_commit=publication['wrapper_publication_commit'],
    wrapper_sha256=publication['script_sha256'],config_sha256=a['config_sha256'],joint_sha256=p['joint_sha256'],
    reused_checkpoint_cpu_proof_sha256=v['reused_checkpoint_cpu_proof_sha256'],
    canonical_question_source_sha256=a['question_source_sha256'],unique_canonical_questions=a['qa_count'],
    per_source=v['per_source'],shared_bf16_interaction_receipt_identity_verified=True,
    tokenizer_and_model_metadata_sha_verified=v['model_metadata'],hf_weights_stat_only=v['model_weight_sizes_stat_only'],
    complete_hf_weights_read=False,model_weights_loaded=False,new_generation=False)
acceptance=dict(acceptance=a,proof_path=v['proof_path'],proof_sha256=v['proof_sha256'],proof_bytes=len(raw),
    local_exact_proof=str(folder/'PROOF.json'),qa_receipt_sha256=v['qa_receipt_sha256'],
    systemd=v['postflight']['systemd'],cgroup_exists=False,cgroup_pids=[],terminal=v['terminal'],identity=identity,
    cpu_cost=v['cost'],cpu_command_exit=v['execution'],preflight=v['preflight'],postflight=v['postflight'],
    qa_receipt_unchanged=v['qa_receipt_unchanged'],terminal_sha256=v['terminal_sha256'],observed_unix=v['postflight']['observed_unix'],
    scope=dict(single_saved_raw_cpu_recompute=True,raw_shards_stayed_remote=True,new_generation=False,
        complete_hf_weights_read=False,gpu_forward_performed=False,training_modified=False,grounding_modified=False,
        original_qa_results_modified=False,instances_started_stopped_or_restarted=False,other_pids_touched=False,
        cpu_verifier_exit_zero_and_reaped_by_subprocess_run=True,cp_checkpoint_acceptance_repeated=False))
cost=dict(status='QA150_INDEPENDENT_CPU_COST_RECORDED',run_id=p['run_id'],outer=p['outer'],cpu_acceptance=v['cost'],
    cpu_total_seconds=v['cost']['cpu_user_seconds']+v['cost']['cpu_system_seconds'],wrapper_subprocess=v['execution'],
    actual_saved_response_tokens_including_observations=a['actual_response_tokens_including_observations'],
    logical_retrieval_calls_from_saved_outputs=a['logical_retrieval_calls'],
    prior_qa_generation_terminal_cost={k:v['terminal'][k]for k in('started_unix','ended_unix','cpu_seconds_self_and_reaped_children','wall_gpu8_hours','slurm_reserved_gpus')},
    additional_gpu_hours_for_cpu_acceptance=0,scope='New CPU verification cost separate from already completed QA generation; no new GPU/generation work.')
summary=dict(status='QA150_SAVED_RAW_INDEPENDENT_CPU_ACCEPTANCE_PASS',run_id=p['run_id'],outer=p['outer'],node='h100-11',
    wrapper_publication_commit=publication['wrapper_publication_commit'],qa_source_commit=p['source_commit'],
    config_sha256=a['config_sha256'],joint_sha256=p['joint_sha256'],qa_receipt_sha256=v['qa_receipt_sha256'],
    proof_path=v['proof_path'],local_exact_proof=str(folder/'PROOF.json'),proof_sha256=v['proof_sha256'],proof_bytes=len(raw),
    count=a['qa_count'],correct=a['correct'],em=a['em'],f1=a['f1'],per_source=v['per_source'],
    rank_counts=[r['count']for r in a['rank_rows']],all_ranks_bf16=True,all_eight_generation_workers_exit_zero=True,
    cpu_recompute_wall_seconds=a['cpu_recompute_wall_seconds'],cpu_cost=v['cost'],cpu_verifier_exit_code=0,
    preflight_observed_unix=v['preflight']['observed_unix'],postflight_observed_unix=v['postflight']['observed_unix'],
    unit_inactive_dead_mainpid0=True,cgroup_exists=False,cgroup_pids=[],cleanup_errors=[],unreaped_child_pids=[],
    complete_hf_weights_read=False,new_generation=False,gpu_forward_performed=False,training_modified=False,
    grounding_modified=False,instances_started_stopped_or_restarted=False,original_qa_results_unchanged=True,
    source_raw_shards_stayed_remote=True,cp_checkpoint_acceptance_repeated=False,final200_full51713_completion_assessed=False)
for path,document in((out/'QA150_INDEPENDENT_CPU_ACCEPTANCE.json',acceptance),(folder/'COST.json',cost),(folder/'SUMMARY.json',summary)):
    with path.open('x')as stream:stream.write(json.dumps(document,indent=2)+'\n')
manifest={}
paths=[path for path in folder.iterdir()if path.is_file()and path.name!='MANIFEST.json']+[out/'QA150_INDEPENDENT_CPU_ACCEPTANCE.json']
for path in sorted(paths):manifest[str(path.relative_to(out))]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
with(folder/'MANIFEST.json').open('x')as stream:stream.write(json.dumps(dict(scope='Small QA150 CPU-only evidence; no raw shard/model tensor transferred',created_unix=time.time(),files=manifest),indent=2)+'\n')
for name,entry in manifest.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==entry['sha256']
print(json.dumps(dict(status=summary['status'],proof_sha256=v['proof_sha256'],proof_bytes=len(raw),
    summary_sha256=hashlib.sha256((folder/'SUMMARY.json').read_bytes()).hexdigest(),
    acceptance_sha256=hashlib.sha256((out/'QA150_INDEPENDENT_CPU_ACCEPTANCE.json').read_bytes()).hexdigest(),
    manifest_sha256=hashlib.sha256((folder/'MANIFEST.json').read_bytes()).hexdigest(),count=a['qa_count'],correct=a['correct'],em=a['em'],f1=a['f1']),indent=2))
