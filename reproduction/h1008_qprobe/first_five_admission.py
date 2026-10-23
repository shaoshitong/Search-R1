"""Prospective native fresh admission; no invented model-smoke acceptance."""
import os
import time
from pathlib import Path


def selected(run):
    from reproduction.common8.precision_contract import easy2560_native4
    original=(run.get('execution_site')=='h1008' and run.get('startup_admission_mode')=='actual_first_five'
        and run.get('question_probe',{}).get('probe_architecture')in ('qwen2_decoder3_count6','qwen2_decoder3_binary_mix_v1')
        and run.get('training_world_size')==8)
    prefix20=(run.get('profile')=='common8' and run.get('execution_site')=='hpc3'
        and run.get('execution_backend','slurm')=='slurm' and run.get('training_world_size')==4
        and run.get('startup_admission_mode')=='actual_first_five'
        and run.get('data_selection_mode')=='prefix20_probe'
        and run.get('question_probe',{}).get('probe_architecture')=='qwen2_prefix20_trajectory_em_v1'
        and not run.get('full_state_restore'))
    easy = (easy2560_native4(run) and run.get('startup_admission_mode') == 'actual_first_five'
        and not run.get('full_state_restore'))
    from reproduction.dynrpo7b.infra_profile import native4
    dynrpo = (native4(run) and run.get('startup_admission_mode') == 'actual_first_five'
        and not run.get('full_state_restore'))
    return original or prefix20 or easy or dynrpo


def _native_four(run):
    from reproduction.common8.precision_contract import native4_7b
    return (run.get('question_probe', {}).get('probe_architecture') == 'qwen2_prefix20_trajectory_em_v1'
        or native4_7b(run))


def verify_probability(base,run,probability,receipts):
    from reproduction.common8.formal_inline_safety import read,owned_path
    from reproduction.common8.policy_calibration_plan import document_sha256
    assert selected(run) and not run.get('full_state_restore')
    allocation=read(owned_path(base,'logs/native-allocation-admission.json'))
    assert probability['role']=='native_fresh_actual_first_five' and probability['run_id']==run['run_id']
    assert probability['source_sha256']==run['source_sha256'] and probability['initial_model_sha256']==run['initial_model_sha256']
    assert probability['slurm_job_id']==allocation['slurm_job_id']
    assert probability['actual_cuda_runtime_uuids']==allocation['actual_cuda_runtime_uuids']
    assert probability['actual_optimizer_writes']==0 and probability['actual_behavior_probability_accepted'] is False
    assert probability['actual_model_initialization_required_before_first_write'] is True
    assert probability['standalone_synthetic_gpu_smoke_performed'] is False
    assert probability['behavior_probability_policy']==run['behavior_probability_policy']
    if _native_four(run):
        assert len(allocation['actual_cuda_runtime_uuids'])==len(set(allocation['actual_cuda_runtime_uuids']))==4
        assert probability['training_world_size']==4
        assert probability['actual_model_initialization_accepted'] is False
        assert probability['actual_first_outer_accepted'] is False
        assert probability['post_start_evidence_mode']=='actual_completed_outer_log_audit'
    resolved=Path(base)/'logs/resolved-training-config.json'
    if resolved.exists():assert probability['resolved_config_sha256']==read(resolved)['resolved_config_sha256']
    return True


def prepare(base,run,document,method,initial,receipts):
    from reproduction.common8 import formal_inline_safety as safety
    from reproduction.common8.freeze_calibration_plan import _unstarted
    from reproduction.common8.formal_contract import FORMAL_COUNTING,FORMAL_SAFEGUARDS
    from reproduction.common8.policy_calibration_plan import document_sha256
    from reproduction.common8.verify_data import digest
    assert selected(run) and not run.get('full_state_restore')
    allocation=safety.read(safety.owned_path(base,'logs/native-allocation-admission.json'))
    assert allocation['accepted'] and allocation['slurm_job_id']==os.environ['SLURM_JOB_ID']
    probability=dict(role='native_fresh_actual_first_five',run_id=run['run_id'],slurm_job_id=allocation['slurm_job_id'],
        source_sha256=run['source_sha256'],initial_model_sha256=run['initial_model_sha256'],
        resolved_config_sha256=document_sha256(document),actual_cuda_runtime_uuids=allocation['actual_cuda_runtime_uuids'],
        actual_optimizer_writes=0,actual_behavior_probability_accepted=False,
        actual_model_initialization_required_before_first_write=True,standalone_synthetic_gpu_smoke_performed=False,
        behavior_probability_policy=run['behavior_probability_policy'],trajectory_count=2560,
        semantics='actual resource binding; real initialized workers and first five full outers required; not model or quality acceptance')
    if _native_four(run):
        probability.update(training_world_size=4,actual_model_initialization_accepted=False,
            actual_first_outer_accepted=False,post_start_evidence_mode='actual_completed_outer_log_audit',
            semantics='prospective native4 resource/source binding; real worker initialization, full-rollout probability gate and completed-outer log audit remain pending; no standalone GPU smoke acceptance')
    for relative in ('logs/native-allocation-admission.json','preparation/formal-inline-method.json',
            'logs/retrieval-admission.json','logs/retrieval-http-admission.json','logs/evaluator-release-admission.json','logs/probe-coexistence-admission.json'):
        receipts[relative]=digest(safety.owned_path(base,relative))
    artifact=dict(schema_version=4,role=safety.ROLE,run_id=run['run_id'],source_commit=run['source_commit'],
        source_sha256=run['source_sha256'],initial_model_sha256=run['initial_model_sha256'],input_sha256=run['input_sha256'],budget=run['budget'],
        accepted_for_guarded_training=True,short_train_accepted=False,healthy_checkpoint_accepted=False,
        initial_probe_version=run['initial_probe_version'],initial_probe_archive=initial['archive'],
        method_sha256=digest(Path(base)/'preparation/formal-inline-method.json'),probability=probability,receipts_sha256=receipts,
        slurm_job_id=allocation['slurm_job_id'],native_birth=safety.native_birth(allocation),actual_optimizer_writes_at_freeze=0,
        created_unix=time.time(),optimizer_counting=FORMAL_COUNTING,**FORMAL_SAFEGUARDS,
        monitoring_scope='actual full-rollout sampling journal plus original finite/KL/tail safeguards',
        semantics='prospective candidates; real first1..5 and next start required, same process to200')
    safety.verify_plan(base,artifact,run);_unstarted(base,run)
    safety.write_once(Path(base)/safety.PLAN,artifact)
    return artifact
