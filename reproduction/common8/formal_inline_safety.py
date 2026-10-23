"""Prospective formal safeguards; the first five are part of this same run.

Candidates are frozen before any Adam write. This receipt does not assert
healthy training. Failed diagnostics supply neither state nor admission.
"""
import argparse
import hashlib
import json
import math
import os
import time
from reproduction.common8.topology import actor_local_micro, actor_logical_micro
from pathlib import Path

from reproduction.common8.formal_contract import (
    FORMAL_COUNTING, FORMAL_SAFEGUARDS, SAFETY_MODE, assert_formal_run, consumed_lr,
)
from reproduction.common8.policy_calibration_plan import canonical_bytes, document_sha256
from reproduction.common8.verify_data import digest
from reproduction.common8.topology import (world_size, ranks, execution_job_id,
                                          monitor_trajectories_per_rank, monitor_scope)
from reproduction.common8.precision_contract import sampling_journal_only

ROLE = 'formal_inline_first_five_candidates'
METHOD_ROLE = 'formal_inline_training_method'
PLAN = 'logs/policy-safety-calibration.json'
INITIAL = 'logs/formal-inline-initial-probe.json'


def read(path):
    from reproduction.common8.policy_calibration_plan import _strict_json
    return _strict_json(Path(path).read_bytes())


def write_once(path, value):
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(canonical_bytes(value)); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def owned_path(base, relative):
    name = Path(relative)
    assert not name.is_absolute() and '..' not in name.parts and name.as_posix() == relative
    path = base / name
    assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base)
    return path


def native_birth(allocation):
    if allocation.get('execution_backend', 'slurm') == 'ssh_local':
        assert allocation['slurm_job_id'] is None and allocation['training_world_size'] == 4
        producer = allocation['pipeline_identity']
        assert all(type(producer[key]) is int for key in ('pid', 'start_ticks', 'sid', 'uid'))
        assert producer['pid'] > 0 and producer['start_ticks'] > 0 and producer['sid'] > 0 and producer['uid'] >= 0
        training = allocation['training_physical_indices']
        assert len(training) == 4 and all(type(index) is int for index in training) and len(set(training)) == 4
        assert not set(training).intersection({5, 6})
        assert allocation['actual_allocated_indices'] == training + [5, 6]
        assert allocation['auxiliary_physical_indices'] == allocation['retriever_physical_indices'] == [5, 6]
        assert allocation['probe_physical_index'] == 6 and allocation['assigned_compute_applications'] == []
        assert set(allocation['assigned_gpu_uuids']) == {str(index) for index in training + [5, 6]}
        inventory = {int(row[0]): row[1].strip() for row in allocation['physical_gpu_inventory']}
        assert len(inventory) == len(allocation['physical_gpu_inventory']) and set(training + [5, 6]) <= set(inventory)
        assert allocation['excluded_physical_indices'] == sorted(set(inventory) - set(training + [5, 6]))
        assert all(allocation['assigned_gpu_uuids'][str(index)] == inventory[index] for index in training + [5, 6])
        assert len(set(allocation['assigned_gpu_uuids'].values())) == 6
        assert len(set(allocation['actual_cuda_runtime_uuids'])) == 4
        assert allocation['actual_cuda_runtime_uuids'] == [allocation['assigned_gpu_uuids'][str(i)] for i in training]
        return dict(role='actual_ssh_local_resource_birth', execution_backend='ssh_local',
            execution_id=allocation['execution_id'], hostname=allocation['node'],
            pipeline_identity={key: producer[key] for key in ('pid', 'start_ticks', 'sid', 'uid')},
            training_physical_indices=allocation['training_physical_indices'],
            assigned_gpu_uuids=allocation['assigned_gpu_uuids'],
            training_cuda_runtime_uuids=allocation['actual_cuda_runtime_uuids'])
    from reproduction.common8.native_allocation import fields
    job = fields(allocation['actual_allocation_detail'])
    names = ('JobId', 'SubmitTime', 'StartTime', 'UserId', 'Command', 'WorkDir', 'NodeList')
    assert all(name in job and job[name] not in ('Unknown', '(null)', 'N/A') for name in names)
    assert job['JobState'] == 'RUNNING' and job['JobId'] == allocation['slurm_job_id']
    return {name: job[name] for name in names}


def verify_prospective_probability(base, run, probability, receipts):
    """Bind local launch inputs; only the real full-rollout gate admits Adam."""
    assert run['execution_backend'] == 'ssh_local' and execution_job_id(run) is None
    allocation = read(owned_path(base, 'logs/native-allocation-admission.json'))
    assert allocation['accepted'] is True and allocation['run_id'] == run['run_id']
    assert allocation['execution_id'] == run['execution_id'] and allocation['execution_backend'] == 'ssh_local'
    assert probability['schema_version'] == 1 and probability['role'] == 'prospective_probability_admission'
    assert probability['run_id'] == run['run_id'] and probability['execution_id'] == run['execution_id']
    assert probability['execution_backend'] == 'ssh_local' and probability['slurm_job_id'] is None
    assert probability['resource_birth'] == native_birth(allocation)
    assert probability['initial_model_sha256'] == run['initial_model_sha256']
    assert probability['source_sha256'] == run['source_sha256']
    assert probability['input_sha256'] == run['input_sha256']
    assert probability['budget_sha256'] == document_sha256(run['budget'])
    assert probability['initial_probe_version'] == run['initial_probe_version']
    assert probability['training_world_size'] == world_size(run) == 4
    assert allocation['training_physical_indices'] == run['training_physical_indices']
    assert allocation['auxiliary_physical_indices'] == run['auxiliary_physical_indices']
    assert allocation['retriever_physical_indices'] == run['retriever_physical_indices']
    assert allocation['probe_physical_index'] == run['probe_physical_index']
    assert type(probability['actual_optimizer_writes']) is int and probability['actual_optimizer_writes'] == 0
    assert probability['actual_behavior_probability_accepted'] is False
    assert probability['actual_full_rollout_gate_required'] is True
    assert probability['trajectory_count'] == 2560
    if sampling_journal_only(run):
        assert probability['tolerance'] is None
        assert probability['hf_recomputation_required'] is False
        assert probability['behavior_probability_policy'] == run['behavior_probability_policy']
    else:
        assert probability['tolerance'] == 0.05
    assert probability['sampling'] == dict(temperature=1.0, top_p=1.0)
    resource = read(owned_path(base, 'logs/local-resource-admission.json'))
    assert resource['accepted'] is True and resource['run_id'] == run['run_id']
    assert resource['execution_id'] == run['execution_id'] and resource['slurm_job_id'] is None
    assert resource['host'] == allocation['node'] and resource['training_world_size'] == 4
    assert resource['allocation_sha256'] == receipts['logs/native-allocation-admission.json']
    assert resource['historical_gpu_smoke_executed'] is False
    assert resource['occupancy_policy'] == 'monitor_only' and resource['sampled_peak_fraction_stop_limit'] is None
    assert resource['retrieval_admission_sha256'] == receipts['logs/retrieval-admission.json']
    assert resource['retrieval_http_admission_sha256'] == receipts['logs/retrieval-http-admission.json']
    assert resource['evaluator_release_admission_sha256'] == receipts['logs/evaluator-release-admission.json']
    initial = read(owned_path(base, INITIAL))
    assert resource['fitted_probe_manifest_sha256'] == initial['archive_sha256']
    accepted_probe = read(owned_path(base, 'logs/initial-probes-accepted.json'))
    assert accepted_probe['accepted'] is True and accepted_probe['run_id'] == run['run_id']
    assert accepted_probe['version'] == run['initial_probe_version']
    assert accepted_probe['live_manifest_sha256'] == initial['archive_sha256']
    from reproduction.common8.formal_contract import verify_formal_config
    proof = verify_formal_config(probability['resolved_config'], run, base)
    assert probability['resolved_config_sha256'] == proof['resolved_config_sha256']
    relative = 'logs/prospective-probability-admission.json'
    assert relative in receipts and digest(owned_path(base, relative)) == receipts[relative]
    assert canonical_bytes(read(base / relative)) == canonical_bytes(probability)
    return proof


def _prepare_local(base, run, document, method, initial, receipts):
    """Freeze honest local inputs without claiming duplicate smoke executions."""
    from reproduction.common8.freeze_calibration_plan import _unstarted
    allocation = read(base / 'logs/native-allocation-admission.json')
    assert execution_job_id(run) is None and allocation['slurm_job_id'] is None
    resource_birth = native_birth(allocation)
    probability = dict(schema_version=1, role='prospective_probability_admission',
        run_id=run['run_id'], execution_backend='ssh_local', execution_id=run['execution_id'],
        slurm_job_id=None, resource_birth=resource_birth, training_world_size=world_size(run),
        initial_model_sha256=run['initial_model_sha256'], source_sha256=run['source_sha256'],
        input_sha256=run['input_sha256'], budget_sha256=document_sha256(run['budget']),
        initial_probe_version=run['initial_probe_version'], resolved_config=document,
        resolved_config_sha256=document_sha256(document), actual_optimizer_writes=0,
        actual_behavior_probability_accepted=False, actual_full_rollout_gate_required=True,
        trajectory_count=2560, tolerance=None if sampling_journal_only(run) else 0.05,
        sampling=dict(temperature=1.0, top_p=1.0),
        semantics='prospective bound local inputs only; actual selected vLLM/HF log-probability gate for all2560 trajectories is mandatory before every Adam attempt')
    if sampling_journal_only(run):
        probability.update(behavior_probability_policy=run['behavior_probability_policy'],
            hf_recomputation_required=False,
            semantics='prospective inputs only; actual sampling journal/context/finite audit covers all2560 trajectories before Adam; no additional full HF forward or maximum-discrepancy admission; runtime policy KL/tail safeguards remain')
    relative = 'logs/prospective-probability-admission.json'
    write_once(base / relative, probability)
    receipts[relative] = digest(base / relative)
    for relative in ('logs/native-allocation-admission.json', 'preparation/formal-inline-method.json',
                     'logs/local-resource-admission.json', 'logs/retrieval-admission.json',
                     'logs/retrieval-http-admission.json', 'logs/evaluator-release-admission.json',
                     'logs/initial-probes-accepted.json'):
        receipts[relative] = digest(base / relative)
    artifact = dict(schema_version=4, role=ROLE, run_id=run['run_id'], source_commit=run['source_commit'],
        source_sha256=run['source_sha256'], initial_model_sha256=run['initial_model_sha256'],
        input_sha256=run['input_sha256'], budget=run['budget'],
        execution_backend='ssh_local', execution_id=run['execution_id'], training_world_size=world_size(run),
        accepted_for_guarded_training=True, short_train_accepted=False, healthy_checkpoint_accepted=False,
        initial_probe_version=run['initial_probe_version'], initial_probe_archive=initial['archive'],
        method_sha256=digest(base / 'preparation/formal-inline-method.json'),
        probability=probability, receipts_sha256=receipts, slurm_job_id=None,
        native_birth=resource_birth, actual_optimizer_writes_at_freeze=0,
        created_unix=time.time(), **FORMAL_SAFEGUARDS, optimizer_counting=FORMAL_COUNTING,
        monitoring_scope=monitor_scope(run),
        semantics='prospective candidates and actual local inputs; full2560 behavior gate precedes Adam; same process1..5 acceptance precedes6..200; no inherited state')
    verify_plan(base, artifact, run); _unstarted(base, run)
    write_once(base / PLAN, artifact)
    return artifact


def verify_method(method, run):
    assert method['schema_version'] == 1 and method['role'] == METHOD_ROLE
    assert method['source_commit'] == run['source_commit']
    assert method['source_map_sha256'] == document_sha256(run['source_sha256'])
    assert method['policy_safety_mode'] == SAFETY_MODE
    assert method['target_updates'] == 200 and method['warmup_steps'] == 57
    assert method['same_process_after_five'] is True
    assert method['fresh_states'] is (not bool(run.get('full_state_restore')))
    assert method['optimizer_counting'] == FORMAL_COUNTING
    assert canonical_bytes(method['safeguards']) == canonical_bytes(FORMAL_SAFEGUARDS)
    assert type(method['created_unix']) in (int, float) and math.isfinite(method['created_unix'])
    assert 0 < method['created_unix'] <= run['started_unix']
    assert method['negative_tail_semantics'] == 'prospective operational candidate; not calibrated health'


def archive_initial_probe(base, run):
    base=Path(base).resolve()
    live = read(base / 'artifacts/live/manifest.json')
    binding=run.get('full_state_restore') or {}
    if binding.get('kind') in ('question_probe_pair','easy2560_native4','dynrpo7b_native4'):
        from search_g1.full_state_restore import source as read_source
        _,source,_,_,_=read_source(binding['source_receipt'])
        pinned=read(source['reward_probe_manifest']['path'])
        assert live['run_id']==run['run_id'] and live['round']==binding['reward_probe_round']
        assert live['version']==run['initial_probe_version']==binding['reward_probe_version']
        assert live['encoder']==pinned['encoder'] and live['encoder_sha256']==pinned['encoder_sha256']
        archive=(Path(live['commit']).parent.parent if binding['kind'] in ('easy2560_native4','dynrpo7b_native4') else
                 Path(live['commit']).parent)/'version-manifest.json'
        assert archive.resolve().is_relative_to(base.resolve()) and read(archive)==live
        for key in ('calibration_report','label_proof','head_sha256'):
            assert live[key]==pinned[key]
        files={str(archive.relative_to(base)):digest(archive)}
        for name,key in (('commit_probe.pkl','commit'),('boundary_probe.pkl','boundary')):
            path=Path(live[key]);assert path.resolve().is_relative_to(base.resolve())
            assert digest(path)==pinned['head_sha256'][name]
            files[str(path.relative_to(base))]=digest(path)
        receipt=dict(schema_version=1,role='fresh200_actual_initial_probe',run_id=run['run_id'],
            source_map_sha256=document_sha256(run['source_sha256']),archive=str(archive.relative_to(base)),
            archive_sha256=digest(archive),version=live['version'],model_revision=run['model_revision'],
            initial_model_sha256=run['initial_model_sha256'],input_sha256=run['input_sha256'],
            restored_probe_round=live['round'],source_pair_receipt_sha256=digest(Path(binding['source_receipt'])),
            semantics='pinned completed-pair reward probe, not fresh0 calibration')
        if binding['kind']=='easy2560_native4':
            receipt.update(semantics='actual saved Easy grounding probe; original calibration remains source-pinned, no refit',
                           source_restore_kind=binding['kind'])
        write_once(base/INITIAL,receipt);files[INITIAL]=digest(base/INITIAL)
        return receipt,files
    assert live['run_id'] == run['run_id'] and type(live['round']) is int and live['round'] == 0
    assert live['version'] == run['initial_probe_version'] and live['version'].startswith(run['run_id'] + ':0:')
    assert live['encoder'] == run['initial_model']
    round_dir = Path(live['commit']).parent.parent
    archive = round_dir / 'version-manifest.json'
    if 'adoption_origin' in live:
        from reproduction.common8.verify_full import verify_a800_adopted_initial_probe
        verify_a800_adopted_initial_probe(base, run, live, archive)
    elif run.get('dynrpo7b_selection') and 'reused_origin' in live:
        from reproduction.dynrpo7b.reward_adapter import verify_adopted
        verify_adopted(base, run, live, archive)
    else:
        assert round_dir.parent == base / 'artifacts' and round_dir.name.startswith('round_0-attempt-')
    assert read(archive) == live
    receipt = dict(schema_version=1, role='fresh200_actual_initial_probe', run_id=run['run_id'],
        source_map_sha256=document_sha256(run['source_sha256']),
        archive=archive.relative_to(base).as_posix(), archive_sha256=digest(archive),
        version=live['version'], model_revision=run['model_revision'],
        initial_model_sha256=run['initial_model_sha256'], input_sha256=run['input_sha256'])
    files = {archive.relative_to(base).as_posix(): digest(archive)}
    for directory, hashes in (('heads', dict(live['head_sha256'], **{
            'calibration-report.json': live['calibration_report_sha256']})), ('dumps', live['dump_sha256'])):
        assert hashes
        for name, sha in hashes.items():
            assert Path(name).name == name
            path = round_dir / directory / name
            assert digest(path) == sha
            files[path.relative_to(base).as_posix()] = sha
    assert all(digest(Path(live['encoder']) / name) == sha for name, sha in live['encoder_sha256'].items())
    assert all(run['initial_model_sha256'][name] == sha for name, sha in live['encoder_sha256'].items())
    report = live['calibration_report']
    assert report['configured_cv_folds'] == 2 and report['require_all_cv_folds'] is True
    assert all(type(report[k + '_cv_auc']) in (int, float) and math.isfinite(report[k + '_cv_auc'])
               and 0 <= report[k + '_cv_auc'] <= 1 for k in ('commit', 'boundary'))
    assert set(live['label_proof']['question_ids']) == set(run['calibration_question_ids'])
    write_once(base / INITIAL, receipt)
    files[INITIAL] = digest(base / INITIAL)
    return receipt, files


def verify_plan(base, artifact, run):
    """Historical reader: no liveness claims and no writes after training starts."""
    from reproduction.h1008_qprobe.first_five_admission import selected as selected_native_startup
    base = Path(base).resolve(); assert_formal_run(run)
    assert run['policy_safety_mode'] == SAFETY_MODE
    assert artifact['schema_version'] == 4 and artifact['role'] == ROLE
    assert artifact['accepted_for_guarded_training'] is True
    assert artifact['short_train_accepted'] is False and artifact['healthy_checkpoint_accepted'] is False
    for key in ('run_id', 'source_commit', 'source_sha256', 'initial_model_sha256', 'input_sha256', 'budget'):
        assert canonical_bytes(artifact[key]) == canonical_bytes(run[key]), 'Inline binding differs: ' + key
    assert artifact['initial_probe_version'] == run['initial_probe_version']
    for key, value in FORMAL_SAFEGUARDS.items():
        assert type(artifact[key]) in (int, float) and artifact[key] == value
    binding=run.get('full_state_restore') or {}
    frozen_writes=(binding['actual_optimizer_writes_per_rank'] if binding.get('kind') in ('question_probe_pair','easy2560_native4','dynrpo7b_native4') else
                   100 if binding else 0)
    assert artifact['actual_optimizer_writes_at_freeze'] == frozen_writes
    assert artifact['optimizer_counting'] == FORMAL_COUNTING
    receipts = artifact['receipts_sha256']
    for relative, sha in receipts.items():
        assert digest(owned_path(base, relative)) == sha, 'Frozen inline receipt changed: ' + relative
    method = read(owned_path(base, 'preparation/formal-inline-method.json'))
    verify_method(method, run)
    assert digest(base / 'preparation/formal-inline-method.json') == artifact['method_sha256']
    initial = read(owned_path(base, INITIAL))
    assert initial['run_id'] == run['run_id'] and initial['role'] == 'fresh200_actual_initial_probe'
    assert initial['source_map_sha256'] == document_sha256(run['source_sha256'])
    assert initial['initial_model_sha256'] == run['initial_model_sha256']
    assert initial['input_sha256'] == run['input_sha256'] and initial['version'] == run['initial_probe_version']
    archive = owned_path(base, initial['archive'])
    assert digest(archive) == initial['archive_sha256']
    live = read(archive)
    assert live['run_id'] == run['run_id'] and live['version'] == run['initial_probe_version']
    if binding.get('kind') in ('question_probe_pair','easy2560_native4','dynrpo7b_native4'):
        from search_g1.full_state_restore import source as read_source
        _,source,_,_,_=read_source(binding['source_receipt'])
        pinned=read(source['reward_probe_manifest']['path'])
        assert live['round']==binding['reward_probe_round'] and live['encoder']==pinned['encoder']
        assert initial['source_pair_receipt_sha256']==digest(Path(binding['source_receipt']))
    else:
        assert live['round']==0 and live['encoder']==run['initial_model']
    assert artifact['initial_probe_archive'] == initial['archive']
    allocation = read(owned_path(base, 'logs/native-allocation-admission.json'))
    assert allocation['accepted'] is True and allocation['run_id'] == run['run_id']
    assert artifact['native_birth'] == native_birth(allocation)
    assert artifact['slurm_job_id'] == allocation['slurm_job_id']
    if binding.get('kind')=='question_probe_pair':
        from search_g1.full_state_restore import verify_guarded_restore
        verify_guarded_restore(base,run,artifact['probability'])
    elif run.get('execution_backend', 'slurm') == 'ssh_local':
        assert artifact['execution_backend'] == 'ssh_local' and artifact['execution_id'] == run['execution_id']
        assert artifact['training_world_size'] == world_size(run)
        assert artifact['monitoring_scope'] == monitor_scope(run)
        verify_prospective_probability(base, run, artifact['probability'], receipts)
    elif run.get('full_state_restore'):
        from search_g1.full_state_restore import verify_guarded_restore
        verify_guarded_restore(base,run,artifact['probability'])
    elif selected_native_startup(run):
        from reproduction.h1008_qprobe.first_five_admission import verify_probability
        verify_probability(base,run,artifact['probability'],receipts)
    else:
        from reproduction.common8.production_safety_admission import _verify_parent_probability
        _verify_parent_probability(base, run, artifact['probability'], receipts)
    assert artifact['probability']['slurm_job_id'] == artifact['slurm_job_id']
    return artifact['negative_tail_limit'], document_sha256(artifact)


def prepare(base):
    from omegaconf import OmegaConf
    from reproduction.common8.configuration import resolved
    from reproduction.common8.freeze_calibration_plan import _unstarted
    from reproduction.common8.formal_contract import verify_formal_config
    base = Path(base).resolve(); run, config, _ = resolved(base)
    _unstarted(base, run)
    assert not list((base / 'logs').glob('scheduler-rank-*.jsonl'))
    assert not list((base / 'logs').glob('formal-inline-optimizer-*.json'))
    method = read(base / 'preparation/formal-inline-method.json'); verify_method(method, run)
    document = OmegaConf.to_container(config, resolve=True); verify_formal_config(document, run, base)
    initial, receipts = archive_initial_probe(base, run)
    if run.get('full_state_restore'):
        from search_g1.full_state_restore import prepare_guarded_restore
        return prepare_guarded_restore(base,run,document,method,initial,receipts)
    from reproduction.h1008_qprobe.first_five_admission import selected,prepare as prepare_first_five
    if selected(run):return prepare_first_five(base,run,document,method,initial,receipts)
    if run.get('execution_backend', 'slurm') == 'ssh_local':
        return _prepare_local(base, run, document, method, initial, receipts)
    root = base / 'logs/policy-safety-admission-receipts/parent'
    root.mkdir(parents=True, exist_ok=False)
    phases = {}
    for phase in ('initial', 'coexistence'):
        original = base / ('logs/model-smoke-' + phase)
        for name, target in [('complete.json', 'model-smoke-' + phase + '-complete.json')] + [
                (f'rank-{rank}.json', f'model-smoke-{phase}-rank-{rank}.json') for rank in ranks(run)]:
            source = original / name; path = root / target
            with path.open('xb') as stream:
                stream.write(source.read_bytes()); stream.flush(); os.fsync(stream.fileno())
            relative = path.relative_to(base).as_posix(); receipts[relative] = digest(path)
            if name == 'complete.json': phases[phase] = (relative, digest(path))
    for relative in ('logs/native-allocation-admission.json', 'preparation/formal-inline-method.json'):
        receipts[relative] = digest(base / relative)
    initial_smoke = read(base / 'logs/model-smoke-initial/complete.json')
    allocation = read(base / 'logs/native-allocation-admission.json')
    assert allocation['slurm_job_id'] == os.environ['SLURM_JOB_ID']
    probability = dict(run_id=run['run_id'], slurm_job_id=allocation['slurm_job_id'], actual_optimizer_writes=0,
        source_sha256=run['source_sha256'], resolved_config_sha256=document_sha256(document),
        question_ids=[qid for row in initial_smoke['ranks'] for qid in row['question_ids']],
        actual_cuda_runtime_uuids=allocation['actual_cuda_runtime_uuids'],
        initial_proof=phases['initial'][0], initial_proof_sha256=phases['initial'][1],
        coexistence_proof=phases['coexistence'][0], coexistence_proof_sha256=phases['coexistence'][1])
    artifact = dict(schema_version=4, role=ROLE, run_id=run['run_id'], source_commit=run['source_commit'],
        source_sha256=run['source_sha256'], initial_model_sha256=run['initial_model_sha256'],
        input_sha256=run['input_sha256'], budget=run['budget'],
        accepted_for_guarded_training=True, short_train_accepted=False, healthy_checkpoint_accepted=False,
        initial_probe_version=run['initial_probe_version'], initial_probe_archive=initial['archive'],
        method_sha256=digest(base / 'preparation/formal-inline-method.json'),
        probability=probability, receipts_sha256=receipts, slurm_job_id=allocation['slurm_job_id'],
        native_birth=native_birth(allocation), actual_optimizer_writes_at_freeze=0,
        created_unix=time.time(), **FORMAL_SAFEGUARDS,
        optimizer_counting=FORMAL_COUNTING,
        monitoring_scope=monitor_scope(run),
        semantics='prospective candidates; real first1..5 acceptance precedes6 in same process; no diagnostic state inherited')
    verify_plan(base, artifact, run); _unstarted(base, run)
    write_once(base / PLAN, artifact)
    return artifact


def validate_optimizer_state(optimizer, logical_update, writes, *, complete):
    assert type(logical_update) is int and 1 <= logical_update <= 200
    assert type(writes) is int and writes >= 0
    expected = writes
    if complete:
        assert writes >= logical_update
    assert optimizer.param_groups and all(g['lr'] == consumed_lr(logical_update) for g in optimizer.param_groups)
    if expected == 0:
        assert len(optimizer.state) == 0, 'Fresh formal Adam state is not empty'
    else:
        assert optimizer.state
        for state in optimizer.state.values():
            step = state['step']; step = step.item() if hasattr(step, 'item') else step
            assert type(step) in (int, float) and math.isfinite(step) and step == expected


def publish_optimizer_state(module, optimizer, logical_update, writes, group=None, *, complete=True,
                            attempt_id, attempt_index):
    """Capture actual tensors/counters at0 and completed20-opportunity outers before scheduler advance."""
    import torch
    import torch.distributed as dist
    from search_g1.numerics import require_finite
    from search_g1.policy_safety import calibration_state_snapshot
    from reproduction.common8.native_allocation import cuda_uuids
    fields = {}; proof = {}; valid = True
    device = 'cuda:' + str(torch.cuda.current_device()) if dist.is_initialized() and dist.get_backend(group) == 'nccl' else 'cpu'
    try:
        validate_optimizer_state(optimizer, logical_update, writes, complete=complete)
        if not complete:
            assert logical_update == 1 and writes == 0 and attempt_index == 1
        assert isinstance(attempt_id, str) and len(attempt_id) == 32 and all(c in '0123456789abcdef' for c in attempt_id)
        assert type(attempt_index) is int and attempt_index >= logical_update
        fields, snapshot = calibration_state_snapshot(module, optimizer)
        base = Path(os.environ['SEARCH_G1_BASE']).resolve(); run = read(base / 'logs/run-manifest.json')
        plan = read(base / PLAN); _, sha = verify_plan(base, plan, run)
        pid = os.getpid(); ticks = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
        uuids = cuda_uuids(count=torch.cuda.device_count())
        proof = dict(schema_version=1, role='actual_fresh200_inline_optimizer_state', run_id=run['run_id'],
            plan_sha256=sha, logical_update=logical_update if complete else 0, optimizer_writes=writes,
            attempt_id=attempt_id, attempt_index=attempt_index,
            consumed_lr=consumed_lr(logical_update), rank=dist.get_rank(),
            slurm_job_id=execution_job_id(run), pid=pid, start_ticks=ticks,
            cuda_uuid=uuids[torch.cuda.current_device()], observed_unix=time.time(), **snapshot)
        if run.get('execution_backend', 'slurm') == 'ssh_local':
            import socket
            proof.update(execution_backend='ssh_local', execution_id=run['execution_id'], hostname=socket.gethostname())
        valid = snapshot['parameter_numel'] > 0
    except Exception:
        valid = False
    fields['valid_inline_optimizer_state'] = torch.tensor(int(valid), device=device)
    require_finite(fields, stage='formal_inline_actual_optimizer_state', process_group=group,
                   positive=('valid_inline_optimizer_state',))
    proof['all_parameters_and_adam_finite'] = True
    written = True
    try:
        path = base / 'logs' / f'formal-inline-optimizer-step-{proof["logical_update"]}-rank-{proof["rank"]}.json'
        write_once(path, proof)
    except Exception:
        written = False
    require_finite({'durable_inline_optimizer_proof': torch.tensor(int(written), device=device)},
        stage='formal_inline_optimizer_proof_fsync', process_group=group, positive=('durable_inline_optimizer_proof',))


def verify_first_five(base, run, behavior_gates, scheduler_receipts):
    from reproduction.common8.safety_calibration_evidence import verify_formal_segment
    from reproduction.common8.formal_attempt_audit import verify_attempt_history
    base = Path(base).resolve(); plan = read(base / PLAN); _, sha = verify_plan(base, plan, run)
    actual = [json.loads(line) for line in (base / 'logs/actual-updates.jsonl').read_text().splitlines()]
    complete_attempts = [r for r in actual if r['step'] <= 5]
    assert [r['step'] for r in complete_attempts] == list(range(1, 6))
    assert all(r['run_id'] == run['run_id'] for r in complete_attempts)
    attempt_ids = {r['attempt_id'] for r in complete_attempts}
    assert len(attempt_ids) == 5
    all_attempts = verify_attempt_history(base, run, complete_attempts, scheduler_receipts,
                                         last=5, plan_sha256=sha)
    rows_by_rank = {}; files = {}; selections = {}; births = set()
    allocation = read(base / 'logs/native-allocation-admission.json')
    for rank in ranks(run):
        path = base / 'logs' / f'optimizer-rank-{rank}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows_by_rank[rank] = [r for r in rows if (r.get('context') or r).get('attempt_id') in attempt_ids]
        selections[str(rank)] = document_sha256(rows_by_rank[rank])
        process = scheduler_receipts[str(rank)]['selected_rows'][0]
        identity = (process['process_pid'], process['process_start_ticks'])
        assert identity not in births; births.add(identity)
        for step in (0, 5):
            path = base / 'logs' / f'formal-inline-optimizer-step-{step}-rank-{rank}.json'
            proof = read(path); files[path.relative_to(base).as_posix()] = digest(path)
            assert proof['schema_version'] == 1 and proof['role'] == 'actual_fresh200_inline_optimizer_state'
            assert proof['run_id'] == run['run_id'] and proof['rank'] == rank
            assert (proof['pid'], proof['start_ticks']) == identity
            assert proof['slurm_job_id'] == plan['slurm_job_id'] and proof['plan_sha256'] == sha
            if run.get('execution_backend', 'slurm') == 'ssh_local':
                assert proof['execution_backend'] == 'ssh_local' and proof['execution_id'] == run['execution_id']
                assert proof['hostname'] == allocation['node']
            assert proof['cuda_uuid'] == allocation['actual_cuda_runtime_uuids'][rank]
            assert proof['logical_update'] == step
            expected_writes = complete_attempts[-1]['optimizer_writes_after'] if step else 0
            assert proof['optimizer_writes'] == expected_writes
            assert proof['all_parameters_and_adam_finite'] is True and proof['parameter_numel'] > 0
            assert proof['consumed_lr'] == consumed_lr(step if step else 1)
            if step == 0:
                assert proof['adam_state_entries'] == proof['adam_state_tensor_count'] == 0 and proof['adam_steps'] == []
            else:
                assert proof['adam_state_entries'] > 0 and len(proof['adam_steps']) == proof['adam_state_entries']
                assert all(x == expected_writes for x in proof['adam_steps'])
                assert proof['attempt_id'] == complete_attempts[-1]['attempt_id']
                assert proof['attempt_index'] == complete_attempts[-1]['attempt_index']
    binding = dict(schema_version=1, role='production', policy_safety_mode=SAFETY_MODE,
        run_id=run['run_id'], calibration_sha256=sha, outer_count=5, initial_optimizer_writes=0,
        candidate_thresholds=FORMAL_SAFEGUARDS, complete_attempts=complete_attempts,
        training_world_size=world_size(run),actor_micro_batch_per_rank=actor_local_micro(run))
    if run.get('easy2560_selection') is True:
        binding.update(actor_logical_micro_batch_per_rank=actor_logical_micro(run))
        for name in ('easy2560_selection', 'execution_site', 'execution_backend', 'data_selection_mode', 'question_probe'):
            if name in run:
                binding[name] = run[name]
    transactional=any(row.get('recovery_protocol_version',1)==2 for row in complete_attempts)
    if transactional:
        # All actual monitor/write/rollback rows were checked by the attempt
        # auditor above; no invented 40-pass requirement after an early guard.
        selected=rows_by_rank[0]
        segment=dict(global_monitors=[row for row in selected if row['kind']=='POLICY_MONITOR'],
            opportunity_dispositions=[row for row in selected if row['kind']=='OPPORTUNITY'],
            guard_triggered_monitor_count=sum(not row['allowed'] for row in selected if row['kind']=='POLICY_MONITOR'),
            completed_outer_actual_writes_per_rank=sum(row['actual_optimizer_writes'] for row in complete_attempts))
    else:
        segment = verify_formal_segment(rows_by_rank, binding)
        assert len(segment['global_monitors']) == 200
    for row in segment['global_monitors']:
        step = str(row['context']['logical_update'])
        assert row['context']['behavior_probability_gate_sha256'] == behavior_gates[step]['proof_sha256']
    from search_g1.monitor_forward_reuse import summarize_monitor_execution
    monitor_execution=summarize_monitor_execution(segment['global_monitors'])
    return dict(schema_version=1, role='actual_formal_first_five_safety_audit', run_id=run['run_id'],
        plan_sha256=sha, structural_accepted=True, source_map_sha256=document_sha256(run['source_sha256']),
        actual_complete_outers=5,
        actual_full_budget_outers=sum(row.get('full_budget_complete',row['actual_optimizer_writes']==20) for row in complete_attempts),
        completed_outer_optimizer_opportunities_per_rank=sum(row['opportunities_processed'] for row in complete_attempts),
        actual_optimizer_writes_per_rank=complete_attempts[-1]['optimizer_writes_after'],
        unique_global_monitor_count=len(segment['global_monitors']),
        actual_monitor_forward_count=monitor_execution['actual_monitor_forward_count'],
        reused_monitor_forward_count=monitor_execution['reused_monitor_forward_count'],
        candidate_thresholds=FORMAL_SAFEGUARDS, worker_process_births=sorted(births),
        global_monitors=segment['global_monitors'], receipt_sha256=files,
        opportunity_dispositions=segment['opportunity_dispositions'],
        all_attempts_audit=all_attempts,
        guard_triggered_monitor_count=segment['guard_triggered_monitor_count'],
        completed_outer_actual_writes_per_rank=segment['completed_outer_actual_writes_per_rank'],
        selected_evidence_sha256=document_sha256(dict(plan_sha256=sha, complete_attempts=complete_attempts,
            selected_rank_rows_sha256=selections, state_proofs_sha256=files, worker_births=sorted(births),
            all_attempts_selected_evidence_sha256=all_attempts['selected_evidence_sha256'])),
        sampled_kl_max=max((r['sampled_old_policy_kl'] for r in segment['global_monitors']
                          if r['kl_estimate_complete']), default=None),
        negative_tail_weight_fraction_max=max((r['negative_tail_weight_fraction'] for r in segment['global_monitors']),default=None),
        extra_policy_monitor_forwards=run.get('extra_policy_monitor_forwards',True),
        sampled_policy_diagnostics_available=bool(segment['global_monitors']),
        semantics=('fixed32-trajectory monitors, not all2560; structural acceptance does not assert accuracy success'
            if run.get('extra_policy_monitor_forwards',True) else
            'Optional extra policy-monitor forwards not executed; sampled KL/tail unavailable, not zero or PASS; actual updates and finite-state evidence audited'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true', required=True)
    artifact = prepare(parser.parse_args().base)
    print(json.dumps(dict(run_id=artifact['run_id'], short_train_accepted=False, plan_sha256=document_sha256(artifact))))
