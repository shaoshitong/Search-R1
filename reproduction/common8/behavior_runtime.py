"""Native driver/worker binding for the actual common8 behavior denominator.

The context reader runs before any old/reference forward. The probability
reader runs on every retained model token before Adam. Published receipts contain
only scalar/hash evidence; raw token journals remain private on the native node.
"""
import hashlib
import json
import os
import stat
import uuid
from pathlib import Path

import numpy as np
import torch

from reproduction.common8.behavior_trace import (
    audit_outer, make_actor_row, make_trace, write_private_evidence,
    _exclusive_private_file, _evidence_value,
)
from reproduction.common8.policy_calibration_plan import canonical_bytes, document_sha256
from reproduction.common8.precision_contract import sampling_journal_only
from reproduction.common8.topology import (
    execution_job_id, local_trajectories, logprob_local_micro, ranks, validate_base, world_size,
)


TOKEN_FIELDS = frozenset(('input_ids', 'prompts', 'responses', 'responses_with_info_mask',
                          'attention_mask', 'position_ids', 'info_mask', 'loss_mask'))
JOURNAL_ONLY_POLICY = 'sampling_journal_no_extra_hf_forward'
JOURNAL_ONLY_DENOMINATOR = 'actual_selected_sampling_log_probs_after_all_token_context_audit'
HF_PARITY_DENOMINATOR = 'actual_selected_sampling_log_probs_after_all_token_context_and_0.05_gate'


def cast_token_fields(batch):
    """Do not integer-cast actual selected probabilities or numerical metrics."""
    for name in TOKEN_FIELDS.intersection(batch.batch.keys()):
        batch.batch[name] = batch.batch[name].long()


def rollout_logprob_metadata(rollout):
    """Read actual per-rank config after the worker's global32 / real-world split."""
    values = dict(micro_batch_size=rollout.log_prob_micro_batch_size,
                  temperature=rollout.temperature,
                  use_dynamic_bsz=rollout.log_prob_use_dynamic_bsz,
                  max_token_len=rollout.log_prob_max_token_len_per_gpu)
    assert type(values['micro_batch_size']) is int and values['micro_batch_size'] == logprob_local_micro()
    assert type(values['temperature']) in (int, float) and values['temperature'] == 1.0
    assert values['use_dynamic_bsz'] is False and type(values['max_token_len']) is int
    return values


def _native_run():
    base = Path(os.environ['SEARCH_G1_BASE'])
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    validate_base(base, run)
    assert run['profile'] == 'common8' and run['completed_updates'] == 0
    world_size(run)
    if run.get('easy2560_selection') is True:
        from reproduction.common8.precision_contract import easy2560_native4
        assert easy2560_native4(run), 'Easy behavior journal requires native4 mixed_first and logical4 without a question head'
    job = execution_job_id(run)
    # This is a current disk-source identity check, not proof of loaded modules.
    for name, expected in run['source_sha256'].items():
        path = base / 'repo' / name
        assert path.resolve() == path and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    return base, run, job


def prepare_identity(parent_batch, logical_update, *, repeats, validation=False,attempt_id=None,attempt_index=None,candidate_offset=0,teacher_warmup=False,candidate_prefix=False):
    base, run, job = _native_run()
    count = len(parent_batch)
    assert type(logical_update) is int and logical_update >= 0
    assert type(candidate_prefix) is bool
    if candidate_prefix:
        assert not validation and not teacher_warmup and repeats == 5
        assert world_size(run) == 4
        if run.get('data_selection_mode') == 'dynrpo7b':
            from reproduction.dynrpo7b.infra_profile import native4
            from reproduction.dynrpo7b.selection import candidate_budget
            assert native4(run)
            assert run['question_probe'].get('probe_architecture') == 'qwen7b_L15_packed_group_variance_v1'
            # Identity is allocated once for the original full candidate pool.
            # Continuation and Backstop inherit its exact G5 ID subsets.
            assert candidate_offset == 0 and count == candidate_budget(logical_update)
        else:
            assert count == 5120 and run.get('data_selection_mode') == 'prefix20_probe'
            assert run.get('question_probe',{}).get('probe_architecture') == 'qwen2_prefix20_trajectory_em_v1'
    else:
        assert repeats == (1 if validation else 5) and (validation or count == 512)
    assert type(teacher_warmup) is bool
    if teacher_warmup:
        assert not validation and logical_update == 0 and run.get('question_probe', {}).get('teacher_warmup_cache')
        assert run.get('completed_updates') == 0 and not run.get('full_state_restore')
    else:
        assert validation or logical_update >= 1
    parent_ids = []
    for index, group in enumerate(parent_batch.non_tensor_batch['index']):
        if hasattr(group, 'item'):
            group = group.item()
        parent_ids.append(str(index + candidate_offset) + '-' + document_sha256(group))
    inline=run.get('policy_safety_mode')=='inline_first_five' and not validation and not teacher_warmup
    if inline:
        assert isinstance(attempt_id,str) and len(attempt_id)==32 and all(char in '0123456789abcdef' for char in attempt_id)
        assert type(attempt_index) is int and attempt_index>=1
    else:
        assert attempt_id is None and attempt_index is None
    occurrence=('-attempt-'+attempt_id) if inline else ''
    ids = [run['run_id'] + '-' + str(logical_update) + occurrence + '-' + parent + '-sample-' + str(sample)
           for parent in parent_ids for sample in range(repeats)]
    # A validation nonce distinguishes repeated before/after calls at one step.
    snapshot = run['run_id'] + ':' + str(logical_update) + ':' + ('validation-' if validation else 'policy-') + (attempt_id if inline else uuid.uuid4().hex)
    identity=dict(run_id=run['run_id'], source_sha256=run['source_sha256'], slurm_job_id=job,
                logical_update=logical_update, policy_snapshot_id=snapshot,
                parent_group_ids=parent_ids, trajectory_ids=ids, validation=validation)
    if teacher_warmup:
        identity.update(teacher_warmup=True, completed_training_outers=0)
    if run.get('execution_backend', 'slurm') == 'ssh_local':
        identity.update(execution_backend='ssh_local', execution_id=run['execution_id'])
    if inline:identity.update(attempt_id=attempt_id,attempt_index=attempt_index)
    return identity


def attach_generation_identity(gen_batch, identity, *, do_sample):
    assert len(gen_batch) == len(identity['trajectory_ids'])
    gen_batch.non_tensor_batch['behavior_row_id'] = np.array(identity['trajectory_ids'], dtype=object)
    gen_batch.meta_info.update(behavior_identity=identity, policy_snapshot_id=identity['policy_snapshot_id'],
                               recompute_log_prob=False, do_sample=do_sample)
    if 'attempt_id' in identity:
        gen_batch.meta_info.update(attempt_id=identity['attempt_id'],attempt_index=identity['attempt_index'])


def actor_rows(batch):
    width = batch.batch['prompts'].shape[1]
    ids = batch.non_tensor_batch['behavior_trajectory_id'].tolist()
    values = {name: batch.batch[name].detach().cpu().tolist() for name in
              ('input_ids', 'attention_mask', 'info_mask', 'position_ids')}
    return [make_actor_row(trajectory_id=identity, prompt_width=width,
                           **{name: rows[index] for name, rows in values.items()})
            for index, identity in enumerate(ids)]


def _bindings(identity, sampling):
    return {'expected_' + name: identity[name] for name in
            ('run_id', 'source_sha256', 'slurm_job_id', 'logical_update', 'parent_group_ids',
             'trajectory_ids', 'policy_snapshot_id')} | {'expected_sampling': sampling}


def context_before_forward(batch):
    identity = batch.meta_info['behavior_identity']
    assert identity['validation'] is False and len(batch) == 2560
    base, run, job = _native_run()
    assert (run['run_id'], run['source_sha256'], job) == (
        identity['run_id'], identity['source_sha256'], identity['slurm_job_id'])
    if run.get('execution_backend', 'slurm') == 'ssh_local':
        assert identity['execution_backend'] == 'ssh_local' and identity['execution_id'] == run['execution_id']
    rows = actor_rows(batch)
    ids = identity['trajectory_ids']
    actual_ids = batch.non_tensor_batch['behavior_trajectory_id'].tolist()
    assert actual_ids == ids
    turns = batch.non_tensor_batch['behavior_turns'].tolist()
    trajectories = [dict(trajectory_id=value, parent_occurrence_index=index // 5,
                         sample_index=index % 5, turns=turns[index]) for index, value in enumerate(ids)]
    sampling = batch.meta_info['behavior_sampling']
    trace = make_trace(run_id=identity['run_id'], source_sha256=identity['source_sha256'],
        slurm_job_id=job, logical_update=identity['logical_update'],
        policy_snapshot_id=identity['policy_snapshot_id'], sampling=sampling,
        parent_group_ids=identity['parent_group_ids'], trajectories=trajectories, actor_rows=rows)
    bindings = _bindings(identity, sampling)
    private = write_private_evidence(base, trace,attempt_id=identity.get('attempt_id'))
    assert private['serialization_transformed'] is False
    audit = audit_outer(trace, actor_rows=rows, **bindings)
    assert audit['trace_sha256'] == private['private_trace_sha256']
    return dict(trace=trace, actor_rows=rows, audit=audit, bindings=bindings, private=private,
                sampling_journal_only=sampling_journal_only(run))


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(canonical_bytes(dict(dtype=str(value.dtype), shape=list(value.shape))) +
                          value.view(torch.uint8).numpy().tobytes()).hexdigest()


def row_hash(batch, index):
    fields = ('input_ids', 'responses', 'attention_mask', 'position_ids', 'info_mask',
              'old_log_probs', 'sampling_log_probs', 'sampling_mask')
    return document_sha256({name: _tensor_hash(batch.batch[name][index]) for name in fields})


def freeze_after_probability_gate(batch, context, hf_output):
    """Freeze actual sampling LP after the run-bound journal/parity audit."""
    from reproduction.common8.behavior_probability import verify
    base, run, job = _native_run()
    journal_only = sampling_journal_only(run)
    assert context.get('sampling_journal_only', False) == journal_only
    assert batch.meta_info['behavior_sampling']['temperature'] == 1.0
    if journal_only:
        assert run['behavior_probability_policy'] == JOURNAL_ONLY_POLICY and hf_output is None
        hf_log_probs = None
    else:
        assert hf_output is not None
        assert hf_output.meta_info['temperature'] == 1.0
        assert hf_output.meta_info['micro_batch_size'] == logprob_local_micro() and hf_output.meta_info['use_dynamic_bsz'] is False
        hf_log_probs = hf_output.batch['old_log_probs'].detach().cpu()
    width = batch.batch['responses'].shape[1]
    actual = batch.batch['sampling_log_probs'].detach().cpu()
    authored = batch.batch['info_mask'][:, -width:].detach().cpu()
    attention = batch.batch['attention_mask'][:, -width:].detach().cpu()
    assert torch.equal(batch.batch['sampling_mask'].detach().cpu(), authored.bool())
    proof = verify(hf_log_probs, actual, authored, attention,
        trace=context['trace'], actor_rows=context['actor_rows'], audit=context['audit'],
        require_hf_parity=not journal_only, **context['bindings'])
    batch.batch['old_log_probs'] = batch.batch['sampling_log_probs'].clone()
    batch.meta_info['temperature'] = batch.meta_info['behavior_sampling']['temperature']
    proof.update(row_tensor_sha256={value: row_hash(batch, index) for index, value in
                 enumerate(batch.non_tensor_batch['behavior_trajectory_id'].tolist())},
                 upstream_group_sha256={value: batch.meta_info['behavior_identity']['parent_group_ids'][index // 5].split('-', 1)[1]
                     for index, value in enumerate(batch.non_tensor_batch['behavior_trajectory_id'].tolist())},
                 private_trace_sha256=context['private']['private_trace_sha256'],
                 denominator=JOURNAL_ONLY_DENOMINATOR if journal_only else HF_PARITY_DENOMINATOR)
    if journal_only:
        proof.update(behavior_probability_policy=run['behavior_probability_policy'],
                     hf_parity_policy_note='No extra HF forward or maximum-discrepancy gate; '
                         'actual sampling journal/context/token/mask provenance and finite LP remain required.')
    identity=batch.meta_info['behavior_identity']
    attempt_id=identity.get('attempt_id');attempt_index=identity.get('attempt_index')
    if run.get('policy_safety_mode')=='inline_first_five':
        proof.update(attempt_id=attempt_id,attempt_index=attempt_index)
    if run.get('execution_backend', 'slurm') == 'ssh_local':
        proof.update(execution_backend='ssh_local', execution_id=run['execution_id'])
    path = gate_path(base,run,proof['logical_update'],attempt_id=attempt_id)
    content = canonical_bytes(proof)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    batch.meta_info['behavior_probability_gate'] = dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(),
        run_id=run['run_id'], slurm_job_id=job, logical_update=proof['logical_update'])
    if journal_only:
        batch.meta_info['behavior_probability_gate']['behavior_probability_policy'] = run['behavior_probability_policy']
    if run.get('execution_backend', 'slurm') == 'ssh_local':
        batch.meta_info['behavior_probability_gate'].update(execution_backend='ssh_local', execution_id=run['execution_id'])
    if attempt_id is not None:
        batch.meta_info['behavior_probability_gate'].update(attempt_id=attempt_id,attempt_index=attempt_index)
        batch.meta_info.update(attempt_id=attempt_id,attempt_index=attempt_index)
    return proof


def _failed_validation_evidence(identity, records, reason):
    """Private failed evaluation records; never a training outer or loaded proof."""
    base, run, job = _native_run()
    step = identity['logical_update']
    assert identity['validation'] is True and type(step) is int and step >= 0
    assert identity['run_id'] == run['run_id'] and identity['source_sha256'] == run['source_sha256']
    assert identity['slurm_job_id'] == job and isinstance(reason, str) and reason
    assert type(records) is list
    snapshot = identity['policy_snapshot_id']
    prefix = run['run_id'] + ':' + str(step) + ':validation-'
    assert isinstance(snapshot, str) and snapshot.startswith(prefix)
    nonce = snapshot[len(prefix):]
    assert len(nonce) == 32 and all(char in '0123456789abcdef' for char in nonce)
    artifacts = base / 'artifacts'
    assert artifacts.is_dir() and artifacts.resolve() == artifacts and artifacts.stat().st_uid == os.getuid()
    parent = artifacts / 'behavior-validation-failures'
    assert not parent.is_symlink() and parent.resolve() == parent
    parent.mkdir(mode=0o700, exist_ok=True)
    descriptor = os.open(str(artifacts), os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    assert parent.stat().st_uid == os.getuid() and stat.S_IMODE(parent.stat().st_mode) == 0o700
    trace = dict(role='common8_failed_validation_generation', run_id=run['run_id'],
        source_sha256=run['source_sha256'], source_snapshot_kind='current_disk_bytes_not_loaded_module_proof',
        loaded_source_authenticated=False, slurm_job_id=job, logical_update=step,
        policy_snapshot_id=snapshot, snapshot_nonce=nonce, validation=True,
        actual_sampling_records=records, failure_reason=reason, complete_outer=False,
        actual_training_outer=False, accepted=False, context_verified=False,
        production_admission=False, healthy_checkpoint_accepted=False,
        physical_execution_authenticated=False)
    value, transformed = _evidence_value(trace)
    content = canonical_bytes(value)
    path = parent / ('validation-step-' + str(step) + '-' + nonce + '.json')
    _exclusive_private_file(path, content)
    return dict(private_validation_trace_path=str(path), private_validation_trace_sha256=hashlib.sha256(content).hexdigest(),
        serialization_transformed=transformed, logical_update=step,
        role=trace['role'], raw_trace_must_not_be_published=True, actual_training_outer=False,
        loaded_source_authenticated=False, accepted=False, context_verified=False,
        production_admission=False, healthy_checkpoint_accepted=False,
        physical_execution_authenticated=False)


def failed_generation_evidence(identity, records, reason):
    if identity.get('validation'):
        return _failed_validation_evidence(identity, records, reason)
    base, run, job = _native_run()
    trace = {name: identity[name] for name in ('run_id', 'source_sha256', 'slurm_job_id',
                                               'logical_update', 'policy_snapshot_id')}
    trace.update(role='common8_failed_partial_generation', actual_sampling_records=records,
                 failure_reason=reason, complete_outer=False)
    write_private_evidence(base, trace,attempt_id=identity.get('attempt_id'))


def gate_path(base,run,step,*,attempt_id=None):
    if run.get('policy_safety_mode')=='inline_first_five':
        assert isinstance(attempt_id,str) and len(attempt_id)==32 and all(char in '0123456789abcdef' for char in attempt_id)
        return Path(base)/'logs'/('behavior-probability-attempt-'+attempt_id+'.json')
    assert attempt_id is None
    return Path(base)/'logs'/('behavior-probability-outer-'+str(step)+'.json')


def _verify_probability_policy(proof, run):
    """A journal-only proof is valid only for its explicit BF16 run policy."""
    assert proof['actual_frozen_old_all_finite'] is True
    if sampling_journal_only(run):
        assert proof['behavior_probability_policy'] == run['behavior_probability_policy'] == JOURNAL_ONLY_POLICY
        assert proof['hf_recomputation_performed'] is False and proof['hf_parity_accepted'] is None
        assert proof['max_abs_discrepancy'] is None and proof['tolerance'] is None
        assert proof['hf_dtype'] is None and 'hf_old_log_probs' not in proof['tensor_sha256']
        assert proof['denominator'] == JOURNAL_ONLY_DENOMINATOR
    else:
        assert proof.get('hf_recomputation_performed', True) is True
        assert proof.get('hf_parity_accepted', True) is True
        assert 0 <= proof['max_abs_discrepancy'] <= proof['tolerance'] == 0.05
        assert proof['denominator'] == HF_PARITY_DENOMINATOR


def read_gate_receipt(base, run, step, *, update_record=None,attempt_id=None,attempt_index=None):
    """Light scalar/hash binding; never claim this rereads raw trajectories."""
    base = Path(base)
    if run.get('policy_safety_mode')=='inline_first_five':
        if update_record is not None:
            assert attempt_id is None or attempt_id==update_record['attempt_id']
            assert attempt_index is None or attempt_index==update_record['attempt_index']
            attempt_id,attempt_index=update_record['attempt_id'],update_record['attempt_index']
        assert type(attempt_index) is int and attempt_index>=1
    path = gate_path(base,run,step,attempt_id=attempt_id)
    assert path.resolve() == path
    raw = path.read_bytes()
    proof = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()
    allocation = json.loads((base / 'logs/native-allocation-admission.json').read_text())
    assert allocation['accepted'] is True and allocation['run_id'] == run['run_id']
    assert proof['accepted'] is True and proof['run_id'] == run['run_id']
    assert proof['source_sha256'] == run['source_sha256']
    assert proof['slurm_job_id'] == allocation['slurm_job_id']
    if run.get('execution_backend', 'slurm') == 'ssh_local':
        assert proof['slurm_job_id'] is None
        assert proof['execution_backend'] == allocation['execution_backend'] == 'ssh_local'
        assert proof['execution_id'] == allocation['execution_id'] == run['execution_id']
    assert proof['logical_update'] == step and proof['trajectory_count'] == 2560
    if run.get('policy_safety_mode')=='inline_first_five':
        assert proof['attempt_id']==attempt_id and proof['attempt_index']==attempt_index
        assert proof['policy_snapshot_id']==run['run_id']+':'+str(step)+':policy-'+attempt_id
    _verify_probability_policy(proof, run)
    assert proof['sampling']['temperature'] == proof['sampling']['top_p'] == 1.0
    assert len(proof['row_tensor_sha256']) == len(proof['upstream_group_sha256']) == 2560
    if update_record is not None:
        assert update_record['behavior_probability_gate_sha256'] == sha
        assert update_record['behavior_trace_sha256'] == proof['trace_sha256'] == proof['private_trace_sha256']
    result=dict(proof_sha256=sha, private_trace_sha256=proof['private_trace_sha256'],
                included_actor_token_count=proof['included_actor_token_count'],
                max_abs_discrepancy=proof['max_abs_discrepancy'],
                hf_recomputation_performed=proof.get('hf_recomputation_performed', True),
                hf_parity_accepted=proof.get('hf_parity_accepted', True),
                behavior_probability_policy=proof.get('behavior_probability_policy'))
    if run.get('policy_safety_mode')=='inline_first_five':
        result.update(attempt_id=attempt_id,attempt_index=attempt_index)
    return result


def verify_write_receipt_links(base, gates, first, last):
    """One scalar-ledger scan binds every pre/post monitor and Adam write."""
    base = Path(base)
    run=json.loads((base/'logs/run-manifest.json').read_text())
    attempt_gates={}
    for rank in ranks(run):
        rows = [json.loads(line) for line in (base / 'logs' / ('optimizer-rank-' + str(rank) + '.jsonl')).read_text().splitlines()]
        for row in rows:
            if row['kind'] not in ('OPTIMIZER_WRITE', 'POLICY_MONITOR'):
                continue
            step = row['context']['logical_update']
            if first <= step <= last:
                if run.get('policy_safety_mode')=='inline_first_five':
                    attempt=row['context']['attempt_id'];index=row['context']['attempt_index']
                    if attempt not in attempt_gates:
                        attempt_gates[attempt]=read_gate_receipt(base,run,step,attempt_id=attempt,attempt_index=index)
                    expected=attempt_gates[attempt]['proof_sha256']
                else:
                    expected=gates[str(step)]['proof_sha256']
                assert row['context']['behavior_probability_gate_sha256'] == expected


def verify_worker_input(data, *, logical_update, group):
    """Every rank binds actual local bytes to the driver gate before any write.

    Failure is collective: a local file/row problem cannot leave other ranks in
    a later model collective. This check does not turn a JSON file into a GPU
    execution or healthy-policy proof.
    """
    failure = None
    try:
        base, run, job = _native_run()
        receipt = data.meta_info['behavior_probability_gate']
        path = gate_path(base,run,logical_update,attempt_id=data.meta_info.get('attempt_id'))
        assert receipt['path'] == str(path) and path.resolve() == path and path.stat().st_uid == os.getuid()
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == receipt['sha256']
        proof = json.loads(raw)
        assert proof['accepted'] is True and proof['production_admission'] is False
        assert proof['run_id'] == receipt['run_id'] == run['run_id']
        assert proof['source_sha256'] == run['source_sha256']
        assert proof['slurm_job_id'] == receipt['slurm_job_id'] == job
        if run.get('execution_backend', 'slurm') == 'ssh_local':
            assert proof['execution_backend'] == receipt['execution_backend'] == 'ssh_local'
            assert proof['execution_id'] == receipt['execution_id'] == run['execution_id']
        assert type(logical_update) is int and proof['logical_update'] == receipt['logical_update'] == logical_update
        if run.get('policy_safety_mode')=='inline_first_five':
            assert proof['attempt_id']==receipt['attempt_id']==data.meta_info['attempt_id']
            assert proof['attempt_index']==receipt['attempt_index']==data.meta_info['attempt_index']
        assert proof['trajectory_count'] == 2560 and len(proof['row_tensor_sha256']) == 2560
        _verify_probability_policy(proof, run)
        if sampling_journal_only(run):
            assert receipt['behavior_probability_policy'] == run['behavior_probability_policy']
        assert data.meta_info['temperature'] == 1.0
        assert len(data) == local_trajectories(run) and torch.equal(data.batch['old_log_probs'], data.batch['sampling_log_probs'])
        width = data.batch['responses'].shape[1]
        authored = data.batch['info_mask'][:, -width:]
        assert torch.equal(data.batch['loss_mask'], authored)
        assert torch.equal(data.batch['sampling_mask'], authored.bool())
        for index, identity in enumerate(data.non_tensor_batch['behavior_trajectory_id']):
            assert proof['row_tensor_sha256'][identity] == row_hash(data, index)
            upstream = data.non_tensor_batch['uid'][index]
            if hasattr(upstream, 'item'):
                upstream = upstream.item()
            assert proof['upstream_group_sha256'][identity] == document_sha256(upstream)
    except Exception as error:
        failure = type(error).__name__
    device = torch.device('cuda', torch.cuda.current_device()) if torch.distributed.get_backend(group) == 'nccl' else torch.device('cpu')
    flag = torch.tensor(int(failure is None), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group)
    if not bool(flag.item()):
        failures = [None] * torch.distributed.get_world_size(group)
        torch.distributed.all_gather_object(failures, failure, group=group)
        raise RuntimeError('Common8 actual behavior gate rejected before Adam on ranks: ' + repr(failures))
    all_ids = [None] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(all_ids, data.non_tensor_batch['behavior_trajectory_id'].tolist(), group=group)
    flattened = [value for rank_ids in all_ids for value in rank_ids]
    assert len(flattened) == len(set(flattened)) == 2560 and set(flattened) == set(proof['trajectory_ids']), \
        'All-rank actual behavior trajectory coverage changed before Adam'
