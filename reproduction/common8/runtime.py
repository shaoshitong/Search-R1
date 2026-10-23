"""Common-data runtime gates, separate from the immutable old A800 experiment."""
import json
import math
import os
import re
import time
import hashlib
from numbers import Real
from pathlib import Path
from reproduction.a800.runtime import checkpoint_complete as hf_checkpoint_complete
from reproduction.common8.verify_data import digest, verify
from reproduction.common8.formal_contract import (
    CONTRACT_KIND, PROBE_ROUNDS, TARGET_UPDATES, WARMUP_STEPS,
    FORMAL_COUNTING, assert_formal_run, consumed_lr,
)
from reproduction.common8.policy_calibration_plan import canonical_bytes
from reproduction.common8.topology import world_size, ranks, actor_local_micro, actor_local_mini, local_trajectories, execution_job_id


def checkpoint_complete(step, checkpoint):
    result = hf_checkpoint_complete(step, checkpoint)
    base = Path(os.environ['SEARCH_G1_BASE'])
    run = json.loads((base/'logs/run-manifest.json').read_text())
    if not run.get('mixed_first_selection', False) and run.get('question_probe') is None:
        return result
    import torch
    checkpoint = Path(checkpoint)
    state_dir = checkpoint/'training_state'
    state_manifest = json.loads((state_dir/'manifest.json').read_text())
    assert state_manifest['completed_outer'] == step
    proofs = state_manifest['worker_files']
    assert sorted(p['rank'] for p in proofs) == list(ranks(run))
    writes = []
    for proof in proofs + [state_manifest['driver_file']]:
        path = Path(proof['path'])
        assert path.resolve().is_relative_to(state_dir.resolve()) and path.stat().st_size == proof['bytes']
        assert digest(path) == proof['sha256'], 'Actual state readback differs'
        state = torch.load(path, map_location='cpu', weights_only=False)
        if 'rank' in proof:
            assert state['rank'] == proof['rank'] and state['run_id'] == run['run_id']
            assert state['world_size'] == world_size(run) and state['scheduler']['last_epoch'] == step
            writes.append(state['actual_optimizer_writes'])
            assert state['optimizer']['state'], 'Saved Adam is empty'
            for values in state['optimizer']['state'].values():
                for name in ('step', 'exp_avg', 'exp_avg_sq'):
                    assert name in values and torch.isfinite(values[name]).all()
                assert int(values['step']) == state['actual_optimizer_writes']
        else:
            assert state['completed_outer'] == step and state['candidate_stream']['consumed_source_rows'] >= 512*step
        result['files_sha256'][str(path.relative_to(checkpoint))] = proof['sha256']
        del state
    assert len(set(writes)) == 1 and writes[0] > 0
    result['files_sha256']['training_state/manifest.json'] = digest(state_dir/'manifest.json')
    qprobe = state_manifest.get('question_probe')
    if qprobe is not None:
        assert qprobe['completed_outer'] == step
        assert qprobe['actor_snapshot_id'] == run['run_id']+':'+str(step)
        proofs=([qprobe] if qprobe.get('kind')=='dynrpo7b_group_variance' else [qprobe]+qprobe['rank_files'] if qprobe.get('kind')=='prefix20_trajectory_probe'
                else [qprobe,qprobe['head']])
        for proof in proofs:
            path = Path(proof['path'])
            assert path.resolve().is_relative_to(state_dir.resolve())
            assert path.stat().st_size == proof['bytes'] and digest(path) == proof['sha256']
            result['files_sha256'][str(path.relative_to(checkpoint))] = proof['sha256']
        if qprobe.get('kind')=='prefix20_trajectory_probe':
            from search_g1.prefix20_runtime import validate_saved_state
            state=torch.load(qprobe['path'],map_location='cpu',weights_only=False)
            validate_saved_state(state,qprobe['path'],qprobe['actor_snapshot_id'])
            assert state['completed_outer']==step and world_size(run)==4
        if qprobe.get('kind')=='dynrpo7b_group_variance':
            reference=qprobe['reference']
            assert Path(reference['path']).is_file()
            result['question_probe']=dict(completed_outer=step,head_version=qprobe['head_version'],
                actor_snapshot_id=qprobe['actor_snapshot_id'],reference=reference)
        else:
            result['question_probe'] = dict(completed_outer=step, head_version=qprobe['head_version'],
            actor_snapshot_id=qprobe['actor_snapshot_id'], feature_spec_sha=qprobe['feature_spec_sha'])
    result.update(actual_optimizer_writes_per_rank=writes[0],
        training_state_readback='all rank Adam/scheduler states and actual driver cursor verified; actual restart/reload pending')
    from reproduction.a800.owned_processes import publish
    publish(checkpoint/'complete.json', result)
    return result


def budget_rows(text, first, last,complete_records=None, run=None):
    size = world_size(run)
    rows = []
    writes = []
    opportunities = []
    for line in text.splitlines():
        if 'SEARCH_G1_OPPORTUNITY=' in line:
            payload=line.split('SEARCH_G1_OPPORTUNITY=',1)[1]
            opportunities.append(json.loads(re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', payload)))
        if 'SEARCH_G1_OPTIMIZER_WRITE=' in line:
            payload = line.split('SEARCH_G1_OPTIMIZER_WRITE=', 1)[1]
            writes.append(json.loads(re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', payload)))
        if 'SEARCH_G1_OUTER_BUDGET=' in line:
            payload = line.split('SEARCH_G1_OUTER_BUDGET=', 1)[1]
            payload = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', payload)
            rows.append(json.loads(payload))
    selected = [r for r in rows if first <= r['logical_update'] <= last]
    assert len(selected) == (last-first+1)*size
    completed={} if complete_records is None else {r['step']:r for r in complete_records if first<=r['step']<=last}
    if complete_records is not None:
        assert set(completed)==set(range(first,last+1))
    for step in range(first, last+1):
        group = [r for r in selected if r['logical_update'] == step]
        assert sorted(r['rank'] for r in group) == list(range(size))
        for r in group:
            assert (r['local_trajectories'], r['local_mini'], r['local_micro']) == (local_trajectories(run), actor_local_mini(run), actor_local_micro(run))
            record=completed.get(step)
            if record is None:
                # Retain old pure receipt tests; actual inline callers supply
                # completed attempt records and never infer physical counts.
                before,after=(step-1)*20,step*20
                assert r['optimizer_writes']==20
            else:
                assert r['attempt_id']==record['attempt_id'] and r['attempt_index']==record['attempt_index']
                before,after=record['optimizer_writes_before'],record['optimizer_writes_after']
                assert type(before) is int and type(after) is int and before>=0 and 1<=after-before<=20
                assert r['opportunities_processed']==record['opportunities_processed']
                assert 1<=r['opportunities_processed']<=20
                assert r['optimizer_writes']==after-before
                actual_opportunities=[o for o in opportunities if o['rank']==r['rank'] and
                    o['context']['logical_update']==step and o['context']['attempt_id']==record['attempt_id'] and
                    o['context']['attempt_index']==record['attempt_index']]
                verify_opportunity_rows(actual_opportunities,before=before,after=after,processed=r['opportunities_processed'])
            assert r['cumulative_writes'] == after
            actual = [w for w in writes if w['rank'] == r['rank']
                      and w['context']['logical_update'] == step and
                      (record is None or (w['context']['attempt_id']==record['attempt_id'] and
                       w['context']['attempt_index']==record['attempt_index']))]
            assert len(actual) == after-before, 'Missing or duplicate actual optimizer write records'
            indices=[w['context']['minibatch_index'] for w in actual]
            assert len(set(indices))==len(indices) and all(type(i) is int and 0<=i<20 for i in indices)
            if record is None:assert sorted(indices)==list(range(20))
            else:assert sorted(indices)==sorted(o['opportunity_index'] for o in actual_opportunities if o['optimizer_write_succeeded'])
            assert sorted(w['write_count'] for w in actual) == list(range(before+1,after+1))
            expected_lr = consumed_lr(step)
            assert all(w['learning_rates'] and all(type(lr) in (int,float) and lr==expected_lr
                       for lr in w['learning_rates']) for w in actual), 'Consumed warmup LR differs'
    return selected


def verify_opportunity_rows(rows, *, before, after, processed):
    """Audit processed opportunities separately from actual retained Adam writes."""
    assert type(processed) is int and 0<=processed<=20
    assert [row['opportunity_index'] for row in rows]==list(range(processed)), 'Missing or duplicated opportunity receipt'
    cumulative=before
    for index,row in enumerate(rows):
        assert row['processed'] is True and row['context']['minibatch_index']==index
        assert type(row['optimizer_write_succeeded']) is bool
        success=row['optimizer_write_succeeded'];status=row['status']
        assert status in ('written','written_with_post_guard','skipped_guard','rolled_back_guard')
        assert success==(status in ('written','written_with_post_guard'))
        assert row['optimizer_writes_before']==cumulative
        cumulative+=int(success)
        assert row['optimizer_writes_after']==cumulative
    assert cumulative==after, 'Actual opportunity success counters differ from retained writes'
    return rows


def verify_no_fatal_scheduler_console(base, run):
    """Reject a console fallback even if readable ledger bytes look complete.

    A failed fsync may leave a COMPLETE line visible. If the fatal ledger
    append also fails, the real worker emits this independent console signal.
    The observed hash is historical; it cannot prove future log contents.
    """
    base=Path(base).resolve();path=Path(run['train_log'])
    assert path.is_absolute() and path.is_file() and not path.is_symlink()
    assert path.resolve().is_relative_to(base), 'Training console is outside this owned run'
    raw=path.read_bytes()
    assert b'SEARCH_G1_FATAL_SCHEDULER_EVIDENCE=' not in raw, \
        'Fatal scheduler console fallback invalidates readable completion receipts'
    return dict(console_path=str(path),observed_sha256=hashlib.sha256(raw).hexdigest(),
                scope='no fatal scheduler fallback in the current owned console snapshot')


def verify_scheduler_receipts(base, run, first, last, behavior_gates):
    """Bind each complete outer to actual durable rank counters and consumed LR."""
    assert_formal_run(run)
    assert 1 <= first <= last <= TARGET_UPDATES
    base = Path(base)
    console_check=verify_no_fatal_scheduler_console(base,run)
    claim = json.loads((base/'logs/training-launch-claim.json').read_text())
    assert claim['run_id'] == run['run_id']
    allocation = json.loads((base/'logs/native-allocation-admission.json').read_text())
    assert allocation['accepted'] is True and allocation['run_id'] == run['run_id']
    if run.get('execution_backend', 'slurm') == 'ssh_local':
        assert claim['slurm_job_id'] is None and allocation['slurm_job_id'] is None
        assert claim['execution_backend'] == allocation['execution_backend'] == 'ssh_local'
        assert claim['execution_id'] == allocation['execution_id'] == run['execution_id']
        assert claim['hostname'] == allocation['node']
    owned = json.loads((base/'logs/process-ledger.json').read_text())
    assert owned['run_id']==run['run_id']
    trainer=owned['jobs']['trainer']
    leader=trainer['leader']
    assert (leader['pid'],leader['start_ticks'])==(claim['shell_pid'],claim['shell_start_ticks'])
    admitted_births={(r['pid'],r['start_ticks']) for r in trainer['ledger'].values()
                     if r['sid']==leader['sid'] and r['uid']==leader['uid']}
    source_map_sha = hashlib.sha256(canonical_bytes(run['source_sha256'])).hexdigest()
    contract_sha = hashlib.sha256(canonical_bytes(run['budget'])).hexdigest()
    result = {}
    rank_processes=set()
    for rank in ranks(run):
        path = base/'logs'/f'scheduler-rank-{rank}.jsonl'
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        worker_rows=[json.loads(line) for line in (base/'logs'/f'optimizer-rank-{rank}.jsonl').read_text().splitlines()]
        selected = [r for r in rows if first <= r['logical_update'] <= last]
        history=[r for r in rows if r['logical_update']<=last]
        actual=[json.loads(line) for line in (base/'logs/actual-updates.jsonl').read_text().splitlines()]
        completed={row['step']:row for row in actual if row['step']<=last}
        restored=run.get('full_state_restore') or {}
        offset=restored.get('completed_outer',0)
        assert first>offset
        if offset:
            from search_g1.full_state_restore import verified_runtime_prefix
            prefix=verified_runtime_prefix(base,run)
        else:prefix=dict(completed_outer=0,writes=0,attempt_index=0)
        assert set(completed)==set(range(offset+1,last+1)), 'Missing actual completed logical updates'
        from reproduction.common8.behavior_runtime import read_gate_receipt
        attempt_rows=[]
        for row in history:
            assert isinstance(row['attempt_id'],str) and re.fullmatch('[0-9a-f]{32}',row['attempt_id'])
            assert type(row['attempt_index']) is int and row['attempt_index']>0
            if not attempt_rows or row['attempt_id']!=attempt_rows[-1][0]['attempt_id']:attempt_rows.append([])
            attempt_rows[-1].append(row)
        complete_count=prefix['completed_outer'];physical_writes=prefix['writes'];seen=set();attempt_signatures=[]
        expected_by_row=[]
        for ordinal,attempt in enumerate(attempt_rows,prefix['attempt_index']+1):
            beginning=attempt[0];ending=attempt[-1]
            attempt_id=beginning['attempt_id'];step=complete_count+1
            assert attempt_id not in seen and beginning['attempt_index']==ordinal
            seen.add(attempt_id)
            assert all(row['attempt_id']==attempt_id and row['attempt_index']==ordinal and row['logical_update']==step for row in attempt)
            kinds=[row['kind'] for row in attempt]
            assert kinds in (['BEGIN','ABORTED_ATTEMPT'],['BEGIN','COMMIT_PREPARED','COMPLETE']), \
                'Missing, partial, rejected or reordered scheduler attempt receipts'
            complete=kinds[-1]=='COMPLETE'
            delta=ending['actual_optimizer_writes']
            processed=ending['opportunities_processed']
            assert ending['optimizer_opportunities']==20
            assert type(processed) is int and 0<=processed<=20
            assert type(delta) is int and 0<=delta<=processed and (not complete or delta>=1)
            if not complete:assert processed<20 or delta==0, 'A fully processed nonzero-write attempt must complete'
            if complete:
                record=completed[step]
                assert (record['attempt_id'],record['attempt_index'],record['optimizer_writes_before'],record['optimizer_writes_after'])==(
                    attempt_id,ordinal,physical_writes,physical_writes+delta)
                assert record['opportunities_processed']==processed
                gate=read_gate_receipt(base,run,step,update_record=record)
                if first<=step<=last:assert gate==behavior_gates[str(step)]
            else:
                gate=read_gate_receipt(base,run,step,attempt_id=attempt_id,attempt_index=ordinal)
                assert ending['recoverable_guard'] is True and ending['scheduler_unchanged_on_rejection'] is True
            scoped=[row for row in worker_rows if row.get('context',{}).get('attempt_id')==attempt_id]
            assert all(row['run_id']==run['run_id'] and row['rank']==rank and
                row['context']['logical_update']==step and row['context']['attempt_index']==ordinal for row in scoped)
            opportunities=[row for row in scoped if row['kind']=='OPPORTUNITY']
            verify_opportunity_rows(opportunities,before=physical_writes,after=physical_writes+delta,processed=processed)
            writes=[row for row in scoped if row['kind']=='OPTIMIZER_WRITE']
            assert len(writes)==delta and [row['write_count'] for row in writes]==list(range(physical_writes+1,physical_writes+delta+1))
            assert [row['context']['minibatch_index'] for row in writes]==[
                row['opportunity_index'] for row in opportunities if row['optimizer_write_succeeded']]
            assert all(row['learning_rates'] and all(lr==consumed_lr(step) for lr in row['learning_rates']) for row in writes)
            for row in attempt:
                writes=0 if row['kind']=='BEGIN' else delta
                advanced=row['kind']=='COMPLETE'
                expected_by_row.append((row,step,physical_writes,writes,advanced,gate['proof_sha256'],
                                        0 if row['kind']=='BEGIN' else processed))
            attempt_signatures.append((attempt_id,ordinal,step,complete,physical_writes,physical_writes+delta,processed))
            physical_writes+=delta;complete_count+=int(complete)
        assert complete_count==last, 'Only retained safe writes advance logical rollout boundaries'
        if rank==0:canonical_attempts=attempt_signatures
        else:assert attempt_signatures==canonical_attempts, 'Rank attempt/partial-write history differs'
        process = None
        previous_unix = 0.
        for row,step,before,writes,complete,gate_sha,processed in expected_by_row:
            for name in ('logical_update','rank','world_size','warmup_steps','completed_outer_before',
                         'completed_outer_after','scheduler_last_epoch_before','scheduler_last_epoch_after',
                         'scheduler_step_count_before','scheduler_step_count_after','writes_before',
                         'writes_after','actual_optimizer_writes','opportunities_processed','optimizer_opportunities'):
                assert type(row[name]) is int, 'Noninteger actual scheduler counter: '+name
            assert row['schema_version']==1 and row['role']=='fresh200_outer_scheduler'
            assert row['contract_kind']==CONTRACT_KIND and row['run_id']==run['run_id']
            assert row['source_commit']==run['source_commit'] and row['source_map_sha256']==source_map_sha
            assert row['scheduler_contract_sha256']==contract_sha
            assert row['slurm_job_id']==claim['slurm_job_id']
            if run.get('execution_backend', 'slurm') == 'ssh_local':
                assert row['execution_backend'] == 'ssh_local' and row['execution_id'] == run['execution_id']
                assert row['hostname'] == allocation['node']
            assert row['rank']==rank and row['world_size']==world_size(run)
            assert row['warmup_steps']==WARMUP_STEPS and row['base_learning_rate']==1e-6
            assert type(row['observed_unix']) in (int,float) and math.isfinite(row['observed_unix'])
            assert row['observed_unix'] >= previous_unix and row['observed_unix'] > 0
            previous_unix = row['observed_unix']
            identity = (row['process_pid'], row['process_start_ticks'])
            assert all(type(v) is int and v>0 for v in identity)
            if process is None: process=identity
            assert identity==process, 'Scheduler rank process changed inside the admitted segment'
            assert identity in admitted_births, 'Scheduler process birth absent from independent owned trainer ledger'
            assert row['batch_gate_sha256']==gate_sha
            expected_after = step if complete else step-1
            assert row['completed_outer_before']==step-1 and row['completed_outer_after']==expected_after
            assert row['scheduler_last_epoch_before']==step-1 and row['scheduler_last_epoch_after']==expected_after
            # Torch LambdaLR performs its constructor initial step. Record the
            # actual internal counter rather than pretending it starts at zero.
            assert row['scheduler_step_count_before']==step and row['scheduler_step_count_after']==expected_after+1
            assert row['writes_before']==before and row['writes_after']==before+writes
            assert row['actual_optimizer_writes']==writes
            assert row['optimizer_opportunities']==20 and row['opportunities_processed']==processed
            assert type(row['scheduler_advances']) is int and row['scheduler_advances']==int(complete)
            consumed = consumed_lr(step)
            next_lr = 1e-6*min(expected_after/WARMUP_STEPS,1.0)
            rates, following = row['consumed_learning_rates'], row['next_learning_rates']
            assert rates and len(rates)==len(following)
            assert all(type(lr) in (int,float) and lr==consumed for lr in rates)
            assert all(type(lr) in (int,float) and lr==next_lr for lr in following)
        prefix = b'\n'.join(canonical_bytes(r) for r in selected)+b'\n'
        assert process not in rank_processes, 'Different scheduler ranks claim the same actual process birth'
        rank_processes.add(process)
        result[str(rank)] = dict(receipt_path=str(path), selected_rows=selected,
            fatal_console_check=console_check,
            selected_rows_sha256=hashlib.sha256(prefix).hexdigest(),
            observed_file_sha256=hashlib.sha256(raw).hexdigest(),
            owned_process_ledger_sha256=digest(base/'logs/process-ledger.json'),
            cumulative_actual_optimizer_writes=physical_writes,
            complete_outer_optimizer_opportunities=sum(row['opportunities_processed'] for row in selected if row['kind']=='COMPLETE'),
            full_budget_outer_count=sum(row['actual_optimizer_writes']==20 for row in selected if row['kind']=='COMPLETE'),
            optimizer_completion_rule='safe_progress_boundary_and_explicit_full20_budget',
            semantics='selected safe-progress outer receipts bound to independently recorded trainer births; '
                      'observed files may subsequently append; historical birth binding is not a current liveness claim')
    return result


def _validate_reduced_count(value, *, minimum, maximum):
    """Accept finite integer-valued real scalars from the metric reducer."""
    assert isinstance(value, Real) and not isinstance(value, bool)
    assert minimum <= value <= maximum
    assert math.isfinite(value) and value == int(value)
    return int(value)


def accept_five_updates(checkpoint, records):
    import torch
    from safetensors import safe_open
    base = Path(os.environ['SEARCH_G1_BASE'])
    run = json.loads((base/'logs/run-manifest.json').read_text())
    if run.get('dynrpo7b_selection'):
        from reproduction.dynrpo7b.acceptance import accept_first_five
        return accept_first_five(base,checkpoint,records)
    assert_formal_run(run)
    assert run['profile'] == 'common8' and run.get('completed_updates', 0) == 0
    assert run['acceptance_update'] == 5
    data_proof = verify(base/'data')
    proof = json.loads((Path(checkpoint)/'complete.json').read_text())
    assert proof['run_id'] == run['run_id'] and proof['step'] == 5
    assert proof['finite_all_saved_tensors'] is True
    for name,value in proof['files_sha256'].items():
        assert digest(Path(checkpoint)/name) == value
    assert [r['step'] for r in records] == list(range(1,6))
    actual = [json.loads(line) for line in (base/'logs/actual-updates.jsonl').read_text().splitlines()]
    assert [r['step'] for r in actual] == list(range(1,6))
    from reproduction.common8.behavior_runtime import read_gate_receipt, verify_write_receipt_links
    behavior_gates = {str(r['step']): read_gate_receipt(base, run, r['step'], update_record=r) for r in actual}
    verify_write_receipt_links(base, behavior_gates, 1, 5)
    for r in records:
        assert r['probe_version'] == run['initial_probe_version']
        for key in ('critic/rewards/mean','actor/pg_loss','actor/entropy_loss',
                    'actor/kl_loss','actor/ppo_kl','actor/grad_norm'):
            assert math.isfinite(r['metrics'][key])
        successes=_validate_reduced_count(r['metrics']['actor/optimizer_steps_this_update'],minimum=1,maximum=20)
        opportunities=_validate_reduced_count(r['metrics']['actor/update_opportunities_this_attempt'],minimum=1,maximum=20)
        skipped=_validate_reduced_count(r['metrics']['actor/skipped_update_opportunities'],minimum=0,maximum=19)
        assert successes<=opportunities<=20 and skipped==20-successes
        saved_record = actual[r['step']-1]
        assert saved_record['run_id'] == run['run_id']
        assert saved_record['probe_version'] == r['probe_version']
        assert all(saved_record['metrics'][key] == float(r['metrics'][key]) for key in saved_record['metrics'])
    from reproduction.common8.worker_evidence import read_console_form
    budgets = budget_rows(read_console_form(base, run['run_id']), 1, 5,complete_records=actual,run=run)
    scheduler_receipts = verify_scheduler_receipts(base, run, 1, 5, behavior_gates)
    from reproduction.common8.formal_inline_safety import verify_first_five,verify_plan
    artifact=json.loads((base/'logs/policy-safety-calibration.json').read_text())
    _,inline_plan_sha=verify_plan(base,artifact,run)
    inline_safety=verify_first_five(base,run,behavior_gates,scheduler_receipts)
    from reproduction.common8.formal_inline_evaluation import verify_pair
    paired_development=verify_pair(base,run)
    from reproduction.common8.gpu_training_admission import verify_training_gpu_samples
    gpu_samples = verify_training_gpu_samples(base, run, scheduler_receipts)
    validation = json.loads((base/'logs/latest-stage-identity.json').read_text())
    assert validation['run_id'] == run['run_id'] and validation['step'] == 5
    assert validation['counts'] == dict(nq=32, hotpotqa=32)
    assert validation['validation_sha256'] == run['input_sha256']['stage.parquet']
    initial, saved = Path(run['initial_model']), Path(checkpoint)
    from verl.utils.checkpoint_integrity import validate_saved_checkpoint
    fresh_tensor_count = validate_saved_checkpoint(saved, initial, run['started_unix'])
    old = json.loads((initial/'model.safetensors.index.json').read_text())['weight_map']
    new = json.loads((saved/'model.safetensors.index.json').read_text())['weight_map']
    changed = []
    for name in old:
        with safe_open(str(initial/old[name]), framework='pt', device='cpu') as f:
            a = f.get_tensor(name)
        with safe_open(str(saved/new[name]), framework='pt', device='cpu') as f:
            z = f.get_tensor(name)
        if not torch.equal(a,z):
            changed.append(name)
        del a,z
    assert changed
    result = dict(run_id=run['run_id'], accepted=True, actual_updates=records,
        checkpoint=proof, changed_parameters=changed, budget_rows=budgets,
        independently_verified_saved_tensor_count=fresh_tensor_count,
        shared_data=data_proof, stage_validation=validation, checked_unix=time.time(),
        behavior_probability_gates=behavior_gates,
        scheduler_receipts=scheduler_receipts,
        policy_safety_mode=run['policy_safety_mode'],formal_inline_plan_sha256=inline_plan_sha,
        formal_inline_first_five=inline_safety,paired_development=paired_development,
        actual_first_five_gpu_occupancy=gpu_samples,
        optimizer_completion_rule='safe_progress_boundary_and_explicit_full20_budget',
        first_five_optimizer_opportunities=sum(row['opportunities_processed'] for row in actual),
        first_five_full_budget_outers=sum(row['actual_optimizer_writes']==20 for row in actual),
        first_outer_semantics='at most20 opportunities at consumed LR0; partial and retained safe writes reported separately; fresh5 actual parameter delta verified',
        semantics='structural training gate, not accuracy reproduction success')
    with (base/'logs/five-updates-acceptance.json').open('x') as f:
        json.dump(result,f,default=lambda x:x.item(),allow_nan=False);f.flush();os.fsync(f.fileno())
    directory=os.open(str(base/'logs'),os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    print('SEARCH_G1_FIVE_UPDATES_ACCEPTED='+run['run_id'],flush=True)
    return result


def refresh_after_checkpoint(step, checkpoint):
    # Common8 has an explicitly admitted probe device; never borrow A800 GPU6.
    if step not in PROBE_ROUNDS or step == 0:
        return
    import subprocess
    import requests
    base = Path(os.environ['SEARCH_G1_BASE'])
    run = json.loads((base/'logs/run-manifest.json').read_text())
    assert_formal_run(run)
    session = requests.Session()
    session.trust_env = False
    previous = json.loads((base/'artifacts/live/manifest.json').read_text())
    from search_g1.service_ports import probe_url
    response = session.post(probe_url('suspend', run),
        json={'version':previous['version']},timeout=120)
    response.raise_for_status()
    released = response.json()
    assert released['suspended'] and released['version'] == previous['version']
    assert released['allocated_bytes_after_release'] < 256*1024*1024
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(run['probe_visible_device']))
    fitted = None
    if run.get('dynrpo7b_selection'):
        from reproduction.dynrpo7b.reward_adapter import reuse_fitted
        fitted = reuse_fitted(base, run, step, checkpoint)
    if fitted is None:
        subprocess.run([str(base/'env/bin/python'), '-m', 'reproduction.common8.refit',
            '--step',str(step),'--model',str(checkpoint)],env=env,check=True)
    expected = json.loads((base/'artifacts/live/manifest.json').read_text())
    if fitted is not None:
        response = session.post(probe_url('resume', run), json={'version':expected['version']}, timeout=300)
        response.raise_for_status()
        if not response.json().get('resumed'):
            raise RuntimeError('Fitted reward manifest was not adopted')
    for _ in range(120):
        status = session.get(probe_url('health', run),timeout=10).json()
        if status.get('version') == expected['version'] and status.get('round') == step:
            print('SEARCH_G1_REFIT_ADOPTED='+expected['version'],flush=True)
            return expected['version']
        time.sleep(2)
    raise RuntimeError('New frozen evaluator not adopted; refuse next update')
