"""Assemble small local evidence already returned by the completed CP150 reader."""
from pathlib import Path
import hashlib
import json
import time

OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
terminal=OUT/'terminal'
proof=read(terminal/'PROOF.json');details=proof['details']
exit_=read(terminal/'reader-exit.json');deploy=read(terminal/'deploy-receipt.json')
result=read(OUT/'TERMINAL_COLLECTION_RESULT.json')
assert proof['status']=='PREFIX_SELECTION200_JOINT_MILESTONE_FULL_SHA_CPU_PASS'
assert proof['outer']==150 and proof['actor_actual_adam_steps_per_rank']==proof['probe_actual_adam_steps_per_rank']==[3000]*8
assert len(details['actor_rank_states'])==8 and {x['rank']for x in details['actor_rank_states']}==set(range(8))
for row in details['actor_rank_states']:
    assert row['actual_adam_steps']==3000 and row['scheduler_completed_outer']==150
    assert row['logical_pair_objective']['actual_writes']==20 and row['rng']['python_numpy_cpu_torch_state_loaded']
assert details['probe']['summary']['rank_rng_count']==8 and details['probe_clock']['writes_per_rank']==20
assert details['probe_clock']['group_denominator']==1024 and details['probe_clock']['trajectory_denominator']==5120
assert details['actual_grounding_binding']['round']==100 and details['original_native_records']['original_native_records_equal']
assert details['actor_hf']['tokenizer_semantics_equal'] and details['actor_hf']['official_initial_parameter_change']['nonzero_difference_elements']>0
assert hashlib.sha256((terminal/'PROOF.json').read_bytes()).hexdigest()==exit_['proof_sha256']
check_path=next(OUT.glob('checkpoint150-cpu-check-*-result.json'));check=read(check_path)
preflight=dict(status=check['status'],observed_unix=check['observed_unix'],
    joint_sha256=check['candidate']['joint_sha256'],
    writers={k:{f:w[f]for f in('state','exit_code')}for k,w in check['candidate']['writers'].items()},
    audit_markers=check['audit_markers'],control_root_exists=check['control_root_exists'],existing_unit=check['existing_unit'],
    actual_intent_created=check['audit_intent_created'],
    source_receipt=dict(path=str(check_path),sha256=hashlib.sha256(check_path.read_bytes()).hexdigest()))
change=details['actor_hf']['official_initial_parameter_change']
summary=dict(status='CP150_FIRST_V1_CPU_ACCEPTANCE_PASS',run_id=proof['run_id'],outer=150,
    base=deploy['config']['base'],target='h100-10',execution_helper_commit=deploy['helper_commit'],
    runtime_source_commit=proof['source_commit'],runtime_config_sha256=proof['config_sha256'],
    control_config_sha256='8de9e69ccc7723dbb7e5f322c8317e7f4857b009c9d0c666cacf3628bf246e62',
    joint_sha256=proof['joint_complete_sha256'],audit_version='v1',predecessor=None,
    proof=dict(path='terminal/PROOF.json',remote_path=exit_['proof_path'],sha256=exit_['proof_sha256'],
        bytes=(terminal/'PROOF.json').stat().st_size,status=proof['status']),
    exit={k:exit_[k]for k in('status','subprocess_exit','exit_code','elapsed_seconds','cpu_user_seconds',
        'cpu_system_seconds','max_rss_kib','finished_unix')},
    budget=dict(cpu_quota='100%',memory_max='12G',nice=19,io_weight=10,runtime_max_seconds=3600,
        CUDA_VISIBLE_DEVICES='',gpu_forward_performed=False),
    verified=dict(actor_actual_adam_steps_per_rank=proof['actor_actual_adam_steps_per_rank'],
        probe_actual_adam_steps_per_rank=proof['probe_actual_adam_steps_per_rank'],
        actor_ranks=[dict(rank=r['rank'],scheduler_completed_outer=r['scheduler_completed_outer'],
            actual_adam_states=r['actual_adam_states'],successful_writes=r['logical_pair_objective']['actual_writes'],
            private_rng=r['rng'])for r in details['actor_rank_states']],
        probe=details['probe']['summary'],probe_clock=details['probe_clock'],
        saved_actor_driver_cpu_deserialized=details['actor_driver_file']['actual_cpu_deserialization'],
        saved_candidate_pool=details['pool'],grounding_round=details['actual_grounding_binding']['round'],
        grounding_version=details['actual_grounding_binding']['version'],
        hf_tensors_cpu_read=details['actor_hf']['actual_cpu_tensor_reads'],
        hf_fp32_elements=details['actor_hf']['actual_fp32_elements'],tokenizer_semantics_equal=True,
        parameter_change={k:change[k]for k in('compared_elements','nonzero_difference_elements',
            'max_absolute_difference','all_model_parameters_compared')},
        original_full_trajectories=details['original_native_records']['original_full_trajectories'],
        actor_selected_trajectories=details['original_native_records']['actor_selected_trajectories'],native_records_equal=True,
        strict_em_top512_recomputed=details['strict_em_diversity']['deterministic_population_variance_top512_recomputed'],
        fixed_test_manifest_sha256=details['fixed_test_manifest_file']['sha256'],
        frozen_runtime_file_count=len(details['i1i6_contract']['frozen_source_files'])),
    progress_samples='PROGRESS_SAMPLES.json',preflight='PREFLIGHT_COMPACT.json',cleanup=result['cleanup'],
    scope=dict(independent_full_file_sha=True,actual_cpu_deserialization=True,single_deployment=True,
        large_payload_replay=False,cp5_cp50_cp100_tensor_reads=False,large_payload_transferred_to_Mac=False,
        training_modified=False,qa_or_grounding_modified=False,gpu_forward_performed=False,cuda_rng_applied=False,
        cuda_executor_exact_restore_verified=False,independent_backup_performed=False,
        official_initial_small_tensor_reads_performed=proof['official_initial_small_tensor_reads_performed'],
        legacy_old_weight_files_read_flag_scope='Only official initial selected small tensors; historical training checkpoint weight files not read.',
        qa_or_grounding_success_assessed=False,final200_or_full51713_completion_assessed=False),created_unix=time.time())
summary['exit']['reader_proof_elapsed_seconds']=proof['elapsed_seconds']
manifest=[]
for remote,entry in result['files'].items():
    name=Path(remote).name
    if 'independent-full-sha-cpu-once-v1' in name:name='INTENT.json'if name.endswith('-intent.json')else'PROOF.json'
    path=terminal/name
    assert path.stat().st_size==entry['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    manifest.append(dict(remote_path=remote,local_path=str(path.relative_to(OUT)),bytes=entry['bytes'],sha256=entry['sha256']))
for name,value in [('SUMMARY.json',summary),
    ('REMOTE_SMALL_FILES_MANIFEST.json',dict(files=manifest,scope='Exact small metadata/proof/log bytes only; no tensor or rollout payload transferred'))]:
    with(OUT/name).open('x')as stream:stream.write(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(summary=str(OUT/'SUMMARY.json'),proof_sha256=exit_['proof_sha256'],
    actor_ranks=8,hf_tensors=details['actor_hf']['actual_cpu_tensor_reads'],
    changed=f"{change['nonzero_difference_elements']}/{change['compared_elements']}",cleanup=result['cleanup']['status'])))
