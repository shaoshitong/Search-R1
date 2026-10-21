"""CPU-only post-hoc CP1 recovery followup; never submits, restores or trains.

Authenticate actual CUDA restore producers, completed2..6 ledgers and a sealed
checkpoint, then require a real saved outer7 generation chunk. Large tensor
bodies are neither loaded nor rehashed. This is not a training startup gate.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True

BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021')
RUN = '232f20e112ac4a978c5aba52a0367195'
COMMIT = 'e082a23b5a65b302e2016aa0e5ab43b10b543d4f'
JOB = '745371'
SOURCE_SHA = '0a671ffd2941bfb417cd6968ee7b60d62e02ee67f0c400d929f1127aab7e5984'
SMALL = 16 * 1024**2


class Waiting(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise ValueError(message)


def small(path):
    path = Path(path)
    if not path.is_file():
        raise Waiting('Missing evidence: ' + str(path))
    need(not path.is_symlink(), 'Symlink evidence: ' + str(path))
    before = path.stat()
    need(before.st_size <= SMALL, 'Not a small evidence file: ' + str(path))
    raw = path.read_bytes()
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'Evidence changed during read: ' + str(path))
    return raw, dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def document(path):
    raw, record = small(path)
    return json.loads(raw), record


def ledger(path):
    """Hash only the complete prefix observed, allowing a concurrent append."""
    path = Path(path)
    if not path.is_file():
        raise Waiting('Missing ledger: ' + str(path))
    need(not path.is_symlink(), 'Symlink ledger: ' + str(path))
    with path.open('rb') as stream:
        raw = stream.read(path.stat().st_size)
    end = raw.rfind(b'\n') + 1
    raw = raw[:end]
    return [json.loads(line) for line in raw.splitlines() if line], dict(
        path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), prefix_only=True)


def one(rows, condition, description):
    found = [row for row in rows if condition(row)]
    if not found:
        raise Waiting('Not yet observed: ' + description)
    need(len(found) == 1, 'Duplicate evidence: ' + description)
    return found[0]


def verify_cp1(run):
    accepted, acceptance_record = document(BASE/'preparation/actual-cp1-state-restore-acceptance.json')
    need(accepted['status'] == 'ACTUAL_CP1_NATIVE4_STATE_RESTORE_ACCEPTED_TRAINING_FOLLOWUP_PENDING' and
         accepted['job'] == JOB and accepted['run_id'] == RUN and accepted['source_commit'] == COMMIT and
         accepted['source_job'] == '745022' and accepted['source_completed_outer'] == 1 and
         accepted['model_tensors'] == 339 and accepted['actual_optimizer_writes_per_rank'] == 20 and
         accepted['candidate_source_cursor'] == 2560 and
         accepted['actual_all4_model_optimizer_scheduler_rng_dataequal'] is True and
         accepted['actual_cuda_restore_accepted'] is True, 'CP1 acceptance identity differs')
    expected_names = {'logs/run-manifest.json', 'logs/native-allocation-admission.json',
        'logs/full-state-restored-workers.json', 'logs/full-state-restored-driver.json',
        'logs/easy2560-restored-stream-before-next-outer.json'} | {
        'logs/full-state-restored-rank-%d.json' % rank for rank in range(4)}
    proofs = accepted['verified_actual_producer_evidence']
    need(len(proofs) == len(expected_names) and {row['path'] for row in proofs} == expected_names,
         'CP1 actual producer proof set differs')
    measured = {}
    for proof in proofs:
        value, record = document(BASE/proof['path'])
        need(record['bytes'] == proof['bytes'] and record['sha256'] == proof['sha256'],
             'CP1 actual producer SHA changed: ' + proof['path'])
        measured[proof['path']] = value
    binding = run['full_state_restore']
    source, source_record = document(binding['source_receipt'])
    need(source_record['sha256'] == binding['source_receipt_sha256'] == SOURCE_SHA and
         source['source_job'] == '745022' and source['step'] == 1 and source['world_size'] == 4 and
         source['hf_saved_tensor_count'] == 339 and source['actual_optimizer_writes_per_rank'] == 20,
         'CP1 source receipt differs')
    workers = measured['logs/full-state-restored-workers.json']
    allocation = measured['logs/native-allocation-admission.json']
    ranks = workers['ranks']
    need(workers['run_id'] == RUN and workers['source_commit'] == COMMIT and workers['slurm_job_id'] == JOB and
         workers['source_receipt_sha256'] == SOURCE_SHA and workers['source_step'] == 1 and
         workers['actual_world_size'] == 4 and workers['actual_optimizer_writes_per_rank'] == 20 and
         allocation['accepted'] is True and allocation['run_id'] == RUN and allocation['slurm_job_id'] == JOB and
         len(ranks) == 4 and sorted(row['rank'] for row in ranks) == list(range(4)) and
         len({row['actual_cuda_uuid'] for row in ranks}) == 4 and
         {row['actual_cuda_uuid'] for row in ranks} == set(allocation['actual_cuda_runtime_uuids']),
         'CP1 four-rank allocation/aggregate differs')
    files = {row['relative_path']: row for row in source['files']}
    births = {}
    for row in ranks:
        rank = row['rank']
        need(measured['logs/full-state-restored-rank-%d.json' % rank] == row and
             row['run_id'] == row['source_run_id'] == RUN and row['slurm_job_id'] == JOB and
             row['source_job'] == '745022' and row['source_checkpoint'] == source['checkpoint'] and
             row['source_receipt_sha256'] == SOURCE_SHA and row['source_step'] == row['scheduler_last_epoch'] == 1 and
             row['world_size'] == 4 and row['actual_optimizer_writes'] == 20 and row['last_attempt_index'] == 1 and
             row['policy_rejections'] == 0 and row['actual_model_collective_readback_equal'] is True and
             row['actual_adam_scheduler_readback_equal'] is True and row['actual_rng_readback_equal'] is True and
             row['source_state_full_sha256'] == files['training_state/rank-%d.pt' % rank]['full_read_sha256'],
             'CP1 actual rank readback differs: ' + str(rank))
        births[rank] = row['process_identity']
    need(len({(row['pid'], row['start_ticks']) for row in births.values()}) == 4 and
         next(row for row in ranks if row['rank'] == 0)['model_hf_exact_tensors_checked_on_rank0'] == 339,
         'CP1 model tensor count or worker births differ')
    driver = measured['logs/full-state-restored-driver.json']
    stream = measured['logs/easy2560-restored-stream-before-next-outer.json']
    need(driver['run_id'] == RUN and driver['slurm_job_id'] == JOB and driver['source_job'] == '745022' and
         driver['source_receipt_sha256'] == SOURCE_SHA and driver['source_step'] == driver['completed_outer'] == 1 and
         driver['source_driver_full_sha256'] == source['driver_file']['full_read_sha256'] and
         driver['actual_driver_dataequal'] is True and driver['actual_candidate_stream_readback_equal'] is True and
         driver['actual_driver_rng_readback_equal'] is True and driver['post_callback_rng_captured'] is True and
         driver['actual_candidate_source_cursor'] == stream['candidate_source_cursor'] == 2560 and
         stream['completed_outer'] == 1 and stream['current_world_size'] == 4 and
         stream['actual_candidate_stream_readback_equal'] is True and stream['actual_driver_rng_readback_equal'] is True,
         'CP1 actual driver/second stream readback differs')
    return births, dict(acceptance=acceptance_record, source_receipt=source_record,
                       actual_producer_proofs_rehashed=len(proofs), model_tensors=339,
                       all4_model_adam_scheduler_rng_dataequal=True, driver_cursor=2560)


def verify_outers(births):
    updates, update_record = ledger(BASE/'logs/actual-updates.jsonl')
    attempts, attempt_record = ledger(BASE/'logs/actual-attempts.jsonl')
    schedule, optimizer, references = {}, {}, [update_record, attempt_record]
    for rank in range(4):
        schedule[rank], record = ledger(BASE/('logs/scheduler-rank-%d.jsonl' % rank)); references.append(record)
        optimizer[rank], record = ledger(BASE/('logs/optimizer-rank-%d.jsonl' % rank)); references.append(record)
    result = []
    for step in range(2, 7):
        update = one(updates, lambda row: row.get('step') == step, 'completed update%d' % step)
        attempt = one(attempts, lambda row: row.get('attempt_index') == step, 'completed attempt%d' % step)
        aid, before, after = update['attempt_id'], (step-1)*20, step*20
        need(update['run_id'] == RUN and update['attempt_index'] == step and update['update_status'] == 'complete' and
             update['optimizer_writes_before'] == before and update['optimizer_writes_after'] == after and
             update['actual_optimizer_writes'] == update['opportunities_processed'] == 20 and
             update['full_budget_complete'] is True and update['opportunities_unattempted'] == 0 and
             update['metrics']['actor/skipped_update_opportunities'] == 0 and
             attempt['run_id'] == RUN and attempt['slurm_job_id'] == JOB and attempt['attempt_id'] == aid and
             attempt['logical_update'] == attempt['completed_outer'] == step and attempt['complete_outer'] is True and
             attempt['status'] == 'complete' and attempt['all_ranks_agreed'] is True and attempt['real_rank_count'] == 4 and
             attempt['cumulative_aborted_attempts'] == attempt['optimizer_rollbacks_this_attempt'] == 0 and
             attempt['optimizer_calls_this_attempt'] == attempt['optimizer_writes_this_attempt'] == 20 and
             attempt['opportunities_processed'] == 20 and attempt['opportunities_unattempted'] == 0 and
             attempt['optimizer_writes_before'] == before and attempt['optimizer_writes_after'] == after and
             attempt['scheduler_advances'] == 1 and attempt['full_budget_complete'] is True and
             len(attempt['opportunity_statuses']) == 20 and
             all(status in ('written', 'written_with_post_guard') for status in attempt['opportunity_statuses']),
             'Incomplete/skipped/repeated update or attempt%d' % step)
        selected, selected_record = document(BASE/'artifacts/mixed-selection'/(aid+'.json'))
        need(selected['step'] == step and selected['source_commit'] == COMMIT and selected['fixed_candidate_budget'] is True and
             selected['attempted_candidates'] == 2560 and selected['responses_per_group'] == 5 and selected['selected_groups'] == 512 and
             selected['pool_retry'] == 0 and selected['discarded_insufficient_pools'] == [] and
             selected['raw_generated_trajectories'] == 12800 and selected['source_consumed'] == step*2560,
             'Rerolled or changed candidate pool%d' % step)
        references.append(selected_record)
        for rank in range(4):
            rows = [row for row in schedule[rank] if row.get('logical_update') == step]
            need([row['kind'] for row in rows] == ['BEGIN', 'COMMIT_PREPARED', 'COMPLETE'],
                 'Scheduler did not advance exactly once: rank%d outer%d' % (rank, step))
            row = rows[-1]
            need(all(value['run_id'] == RUN and value['source_commit'] == COMMIT and value['slurm_job_id'] == JOB and
                     value['rank'] == rank and value['world_size'] == 4 and value['attempt_id'] == aid and value['attempt_index'] == step and
                     value['process_pid'] == births[rank]['pid'] and value['process_start_ticks'] == births[rank]['start_ticks']
                     for value in rows) and row['writes_before'] == before and row['writes_after'] == after and
                 row['actual_optimizer_writes'] == row['opportunities_processed'] == 20 and row['full_budget_complete'] is True and
                 row['scheduler_advances'] == 1 and row['scheduler_last_epoch_before'] == step-1 and row['scheduler_last_epoch_after'] == step and
                 row['scheduler_step_count_before'] == step and row['scheduler_step_count_after'] == step+1,
                 'Scheduler/counter/birth differs: rank%d outer%d' % (rank, step))
            scoped = [value for value in optimizer[rank] if value.get('context', {}).get('logical_update') == step and
                      value['kind'] in ('OPTIMIZER_WRITE', 'OPPORTUNITY')]
            writes = [value for value in scoped if value['kind'] == 'OPTIMIZER_WRITE']
            opportunities = [value for value in scoped if value['kind'] == 'OPPORTUNITY']
            need(len(writes) == len(opportunities) == 20, 'Missing/extra actual optimizer events: rank%d outer%d' % (rank, step))
            for index, (write, opportunity) in enumerate(zip(writes, opportunities)):
                context = dict(logical_update=step, minibatch_index=index, prior_optimizer_writes=before+index,
                               attempt_id=aid, attempt_index=step, behavior_probability_gate_sha256=update['behavior_probability_gate_sha256'])
                need(write['run_id'] == opportunity['run_id'] == RUN and write['rank'] == opportunity['rank'] == rank and
                     write['context'] == opportunity['context'] == context and write['write_count'] == before+index+1 and
                     write['optimizer_call_count'] == before+index+1 and opportunity['opportunity_index'] == index and
                     opportunity['status'] == attempt['opportunity_statuses'][index] and opportunity['optimizer_write_succeeded'] is True and
                     opportunity['optimizer_call_attempted'] is True and opportunity['optimizer_write_rolled_back'] is False and
                     opportunity['optimizer_writes_before'] == before+index and opportunity['optimizer_writes_after'] == before+index+1,
                     'Actual optimizer write/skip differs: rank%d outer%d mini%d' % (rank, step, index))
        result.append(dict(step=step, attempt_id=aid, all4_optimizer_writes_before=before,
                           all4_optimizer_writes_after=after, all4_writes_this_outer=20,
                           all4_scheduler_advances=1, scheduler_epoch=step,
                           skipped_opportunities=0, optimizer_rollbacks=0, pool_retries=0))
    return result, references


def verify_current_checkpoint():
    latest, latest_record = document(BASE/'preparation/rolling-checkpoints/latest.json')
    if latest.get('step', 0) < 6:
        raise Waiting('Latest completed rolling checkpoint is below6')
    need(latest['run_id'] == RUN and latest['source_commit'] == COMMIT and latest['status'] == 'EASY2560_ROLLING_CHECKPOINT_READY',
         'Latest checkpoint identity differs')
    _, manifest_record = document(latest['manifest_path'])
    need(manifest_record['sha256'] == latest['manifest_sha256'], 'Latest rolling manifest SHA differs')
    sys.path.insert(0, str(BASE/'repo'))
    from search_g1.easy2560_rolling_checkpoint import verify_checkpoint_manifest
    sealed = verify_checkpoint_manifest(BASE, Path(latest['manifest_path']), expected_run_id=RUN, full_sha=False)
    need(sealed['step'] == latest['step'] and sealed['checkpoint'] == latest['checkpoint'] and sealed['boundary_manifest'] is not None,
         'Latest checkpoint has no complete post-callback boundary')
    verified, skipped = [], []
    for name, record in list(sealed['files'].items()) + [('boundary/'+name, record) for name, record in sealed['boundary_files'].items()]:
        if record['bytes'] > SMALL:
            skipped.append(record)
            continue
        _, actual = small(record['path'])
        need(actual['sha256'] == record['sha256'] and actual['bytes'] == record['bytes'], 'Small checkpoint evidence SHA differs: ' + name)
        verified.append(actual)
    complete, _ = document(Path(sealed['checkpoint'])/'complete.json')
    need(complete['step'] == sealed['step'] and complete['run_id'] == RUN and complete['validated_tensor_count'] == 339 and
         complete['finite_all_saved_tensors'] is True and complete['actual_optimizer_writes_per_rank'] >= 120,
         'Current checkpoint completion/count differs')
    return dict(step=sealed['step'], checkpoint=sealed['checkpoint'], manifest=manifest_record,
                latest_pointer=latest_record, small_files_fresh_sha=verified,
                large_files_identity_only=len(skipped), large_bytes_not_rehashed=sum(row['bytes'] for row in skipped),
                all_sealed_file_identities_verified=True, post_callback_boundary_verified=True)


def verify_outer7_pool():
    parent = BASE/'artifacts/mixed-candidate-pools'
    candidates = []
    for pool in sorted(parent.iterdir()) if parent.is_dir() else []:
        progress_path = pool/'progress.json'
        if not progress_path.is_file():
            continue
        try:
            progress, progress_record = document(progress_path)
        except (ValueError, Waiting):
            continue  # A new progress.json can be mid-write; never infer completion.
        if progress.get('step') != 7:
            continue
        match = re.fullmatch(re.escape(RUN)+r':7:policy-([0-9a-f]{32})', progress.get('policy_snapshot_id', ''))
        need(match is not None and pool.name == match.group(1)+'-pool-0', 'Outer7 pool identity/retry differs')
        completed = progress['complete_ordinals']
        need(all(type(value) is int for value in completed) and completed == list(range(len(completed))) and
             len(completed) <= len(progress['reserved']) <= 2560 and
             progress['source_consumed'] == 7*2560, 'Outer7 completed ordinal/source cursor differs')
        chunks = []
        for start in range(0, 2560, 512):
            path = pool/(str(start)+'.pt')
            if len(completed) < start+512 or not path.is_file():
                continue
            stat = path.stat()
            need(not path.is_symlink() and stat.st_size > 0, 'Outer7 completed raw chunk is empty or rebound')
            chunks.append(dict(path=str(path), bytes=stat.st_size, inode=stat.st_ino,
                               mtime_ns=stat.st_mtime_ns, completed_questions=512, raw_tensor_body_read=False))
        if chunks:
            candidates.append(dict(pool=str(pool), progress=progress_record, policy_snapshot_id=progress['policy_snapshot_id'],
                                   completed_questions=len(completed), source_consumed=progress['source_consumed'], saved_chunks=chunks))
    if not candidates:
        raise Waiting('Outer7 has no completed producer progress plus actual saved raw chunk yet')
    need(len(candidates) == 1, 'Multiple outer7 pool producers found')
    return candidates[0]


def verify():
    run, _ = document(BASE/'logs/run-manifest.json')
    need((BASE/'job.id').read_text().strip() == JOB and run['run_id'] == RUN and run['source_commit'] == COMMIT and
         run['training_world_size'] == 4 and run['easy2560_selection'] is True and run.get('question_probe') is None and
         run['all_correct_virtual_zero'] is False and run['selection_candidate_questions'] == 2560 and
         run['selection_selected_questions'] == 512 and run['selection_fixed_candidate_budget'] is True,
         'Target training identity/scientific recipe differs')
    births, restore = verify_cp1(run)
    outers, references = verify_outers(births)
    checkpoint = verify_current_checkpoint()
    outer7 = verify_outer7_pool()
    need('torch' not in sys.modules, 'CPU evidence reader unexpectedly imported Torch')
    return dict(status='ACTUAL_CP1_ALL4_RESTORE_OUTER2_TO6_AND_SAVED_OUTER7_FOLLOWUP_PASS',
        checked_unix=time.time(), job=JOB, run_id=RUN, source_commit=COMMIT,
        actual_cp1_restore=restore, completed_followup_outers=outers,
        all4_cumulative_optimizer_writes_before=20, all4_cumulative_optimizer_writes_after=120,
        current_sealed_checkpoint=checkpoint, actual_outer7_saved_generation=outer7,
        observed_complete_evidence_prefixes=references, read_only=True,
        torch_imported=False, model_loads=0, gpu_allocations=0, large_tensor_body_reads=0,
        training_modified=False, jobs_submitted=0, restart_performed=False,
        cuda_restoration_performed_by_this_reader=False, recovery_followup_accepted=True, goal_complete=False)


if __name__ == '__main__':
    try:
        print(json.dumps(verify(), allow_nan=False))
    except Waiting as error:
        print(json.dumps(dict(status='WAITING_ACTUAL_RECOVERY_FOLLOWUP_EVIDENCE', job=JOB,
            checked_unix=time.time(), reason=str(error), recovery_followup_accepted=False,
            read_only=True, training_modified=False, goal_complete=False)))
    except Exception as error:
        print(json.dumps(dict(status='ACTUAL_RECOVERY_FOLLOWUP_EVIDENCE_REJECTED', job=JOB,
            checked_unix=time.time(), error_type=type(error).__name__, reason=str(error),
            recovery_followup_accepted=False, read_only=True, training_modified=False, goal_complete=False)))
        raise SystemExit(1)
