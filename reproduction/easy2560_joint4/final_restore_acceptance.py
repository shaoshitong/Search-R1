"""Read-only CPU authentication of a separate actual Easy CP200 CUDA reload.

No Torch import, model/state reload, GPU query, process query, or mutation occurs
here. Actual GPU equality comes from two linked native4 readbacks, a successful
Ray task result, and recorded process/context release observations. The source
checkpoint may live only in its accepted remote archive after these operations.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROLE = 'easy2560_final200_restore_only'
SOURCE_STATUS = 'EASY2560_FINAL200_CHAINED_FULL_SHA_CPU_STATES_PASS'
GPU_STATUS = 'EASY2560_FINAL200_ACTUAL_NATIVE4_RESTORE_ONLY_PASS'
STATUS = 'EASY2560_FINAL200_ACTUAL_RESTORE_PROOFS_CPU_AUTHENTICATED'
ARCHIVE_STATUS = 'EASY2560_FULL_CHECKPOINT_SOURCE_ARCHIVE_SHA_PASS'
MAX_SMALL = 128 * 1024**2
COUNTERS = ('actual_optimizer_writes', 'completed_outer', 'last_attempt_index',
            'used_attempt_ids', 'policy_rejections', 'full_budget_outers',
            'optimizer_calls', 'optimizer_rollbacks')
ZERO = ('optimizer_write_increment', 'scheduler_advance_increment',
        'candidate_consumption_increment', 'generation_calls', 'training_checkpoint_save_calls')
LEDGERS = ('actual-updates.jsonl', 'actual-attempts.jsonl') + tuple(
    role+'-rank-'+str(rank)+'.jsonl' for role in ('optimizer', 'scheduler') for rank in range(4))


def need(value, message):
    if not value:
        raise ValueError('EASY2560_FINAL_RESTORE_' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('EASY2560_FINAL_RESTORE_NONFINITE_JSON_' + value)


def _path(value):
    path = Path(value)
    need(path.is_absolute() and path.resolve() == path, 'ABSOLUTE_CANONICAL_PATH_REQUIRED')
    return path


def _sha(record):
    value = record.get('sha256', record.get('full_read_sha256'))
    need(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'EXACT_SHA_REQUIRED')
    return value


def _read(path, *, record=None, under=None):
    path = _path(path)
    if under is not None:
        need(path.is_relative_to(under), 'PROOF_ESCAPES_DEDICATED_TARGET')
    need(path.is_file() and not path.is_symlink(), 'REGULAR_SMALL_PROOF_REQUIRED')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        need(0 <= before.st_size <= MAX_SMALL, 'SMALL_PROOF_LIMIT')
        raw = stream.read(MAX_SMALL + 1)
        from reproduction.easy2560_joint4.final_cpu_acceptance import _verify_stable_fd_snapshot
        _verify_stable_fd_snapshot(stream, path, before, digest(raw), len(raw), check=need,
                                   changed='SMALL_PROOF_CHANGED_DURING_READ')
    actual = dict(path=str(path), bytes=len(raw), sha256=digest(raw))
    if record is not None:
        need(record.get('path') == str(path) and type(record.get('bytes')) is int and
             record['bytes'] == len(raw) and _sha(record) == actual['sha256'], 'SMALL_PROOF_SHA_OR_SIZE_DIFFERS')
    return json.loads(raw, parse_constant=_constant, object_pairs_hook=_pairs), actual


def _ref(record, *, under=None):
    need(isinstance(record, dict), 'LINKED_PROOF_RECORD_REQUIRED')
    return _read(record['path'], record=record, under=under)


def _birth(row):
    need(isinstance(row, dict) and all(type(row.get(key)) is int and row[key] > 0
         for key in ('pid', 'start_ticks', 'sid')) and type(row.get('uid')) is int and row['uid'] >= 0,
         'ACTUAL_PROCESS_BIRTH_REQUIRED')
    return tuple(row[key] for key in ('pid', 'start_ticks', 'sid', 'uid'))


def _host(row, node):
    need(isinstance(row, dict) and row.get('hostname') == node and
         isinstance(row.get('boot_id'), str) and
         re.fullmatch('[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', row['boot_id']) and
         isinstance(row.get('pid_namespace'), str) and re.fullmatch(r'pid:\[[0-9]+\]', row['pid_namespace']),
         'ACTUAL_HOST_KERNEL_IDENTITY_REQUIRED')
    return row


def _time(row):
    value = row.get('checked_unix')
    need(type(value) in (int, float) and math.isfinite(value) and value > 0, 'ACTUAL_OBSERVATION_TIME_REQUIRED')
    return value


def _source(source_record, *, base, run, checkpoint_summary, progress):
    receipt, source_identity = _ref(source_record)
    checkpoint = base/'checkpoints/search-g1-common8/actor/global_step_200'
    need(receipt.get('status') == SOURCE_STATUS and receipt.get('kind') == ROLE and
         receipt.get('restore_only') is True and receipt.get('step') == 200 and receipt.get('world_size') == 4 and
         receipt.get('source_base') == str(base) and receipt.get('checkpoint') == str(checkpoint) and
         receipt.get('run_id') == run['run_id'] and receipt.get('source_commit') == run['source_commit'] and
         receipt.get('current_source_commit') == run['source_commit'] and
         receipt.get('actual_cuda_restore_accepted') is False and receipt.get('question_probe') is None and
         receipt.get('single_original_job_prefix') is False and receipt.get('inherited_history_assembled') is True and
         receipt.get('post_callback_rng_captured') is False and receipt.get('reward_refresh_pending') is False and
         receipt.get('sampling_rng_stage') == 'saved_checkpoint_before_final51713' and
         receipt.get('no_update201') is True, 'EXACT_CHAINED_SOURCE200_REQUIRED')
    need(checkpoint_summary.get('step') == 200 and checkpoint_summary.get('source_checkpoint') == str(checkpoint) and
         checkpoint_summary.get('existing_full_file_sha_acceptance_authenticated') is True,
         'FINAL_CP200_ARCHIVE_SUMMARY_REQUIRED')
    documents = {}
    for name in ('source_snapshot', 'accepted_archive', 'checkpoint_complete', 'training_state_manifest'):
        document, record = _ref(checkpoint_summary[name]); documents[name] = document
        need(_sha(receipt['checkpoint_archive'][name]) == record['sha256'], 'SOURCE_RECEIPT_ARCHIVE_METADATA_DIFFERS')
    snapshot, accepted = documents['source_snapshot'], documents['accepted_archive']
    complete, state = documents['checkpoint_complete'], documents['training_state_manifest']
    for item in (snapshot, accepted):
        need(all(item.get(key) == value for key, value in
             dict(base=str(base), run_id=run['run_id'], source_commit=run['source_commit'], step=200).items()),
             'SOURCE_ARCHIVE_IDENTITY_DIFFERS')
    need(accepted.get('status') == ARCHIVE_STATUS and all(accepted.get(key) is True for key in
         ('fresh_destination_sha_verified', 'destination_fsynced', 'fresh_source_after_archive_sha_verified')) and
         accepted.get('snapshot_sha256') == _sha(checkpoint_summary['source_snapshot']), 'SOURCE_ARCHIVE_SHA_NOT_ACCEPTED')
    tensor_count = receipt.get('hf_saved_tensor_count')
    need(complete.get('run_id') == run['run_id'] and complete.get('step') == 200 and
         complete.get('checkpoint') == str(checkpoint) and complete.get('finite_all_saved_tensors') is True and
         type(tensor_count) is int and tensor_count > 0 and complete.get('validated_tensor_count') == tensor_count and
         state.get('completed_outer') == 200 and state.get('question_probe') is None, 'SAVED_NATIVE4_CP200_DIFFERS')
    files = receipt.get('files', [])
    need(isinstance(files, list) and len(files) == len({row['relative_path'] for row in files}) and
         len(files) == receipt.get('file_count'), 'SOURCE_FILE_SET_DIFFERS')
    by_name = {row['relative_path']: row for row in files}
    archived_checkpoint = {name[len('checkpoint/'):]: row for name, row in snapshot['files'].items()
                           if name.startswith('checkpoint/')}
    need(set(by_name) == set(archived_checkpoint) and {'complete.json'} | set(complete['files_sha256']) <= set(by_name),
         'SOURCE_ARCHIVE_MEMBERS_DIFFER')
    for name, row in by_name.items():
        relative = Path(name)
        need(not relative.is_absolute() and '..' not in relative.parts and row['path'] == str(checkpoint/relative),
             'SOURCE_MEMBER_PATH_DIFFERS')
        archived = archived_checkpoint[name]
        expected = (_sha(checkpoint_summary['checkpoint_complete']) if name == 'complete.json'
                    else complete['files_sha256'].get(name, archived.get('sha256')))
        need(_sha(row) == expected == archived.get('sha256') and type(row.get('bytes')) is int and
             row['bytes'] == archived.get('bytes'), 'SOURCE_MEMBER_ARCHIVE_SHA_DIFFERS')
    need(complete['files_sha256']['training_state/manifest.json'] == _sha(checkpoint_summary['training_state_manifest']),
         'SOURCE_MANIFEST_SHA_DIFFERS')
    saved_progress = receipt['attempt_progress']; writes = saved_progress['cumulative_writes']
    need(saved_progress.get('completed_outer') == 200 and type(writes) is int and 200 <= writes <= 4000 and
         type(saved_progress.get('attempt_index')) is int and saved_progress['attempt_index'] >= 200 and
         saved_progress.get('aborted_attempts') == saved_progress['attempt_index']-200 and
         receipt.get('actual_optimizer_writes_per_rank') == complete.get('actual_optimizer_writes_per_rank') == writes,
         'SAVED_PROGRESS_DIFFERS')
    need(progress.get('actual_completed_updates') == 200 and progress.get('rank_count') == 4 and
         progress.get('actual_attempt_count') == saved_progress['attempt_index'] and
         progress.get('cumulative_actual_optimizer_writes_per_rank') == writes and
         progress.get('no_update201') is True and progress.get('no_post200_driver_scheduler_worker_attempts') is True,
         'FINAL_CPU_PROGRESS_DIFFERS')
    history = progress.get('authenticated_history')
    need(isinstance(history, dict) and history.get('current_job') == receipt.get('source_job') and
         history.get('current_source_commit') == run['source_commit'] and
         receipt.get('authenticated_history') == history, 'SOURCE_EXECUTION_HISTORY_DIFFERS')
    ranks = receipt['ranks']; native = state['worker_files']
    need(len(ranks) == len(native) == 4 and sorted(row['rank'] for row in ranks) ==
         sorted(row['rank'] for row in native) == list(range(4)), 'FOUR_SAVED_RANKS_REQUIRED')
    for rank in range(4):
        row = next(value for value in ranks if value['rank'] == rank)
        member = next(value for value in native if value['rank'] == rank)
        need(row.get('actual_optimizer_writes') == writes and row.get('completed_outer') ==
             row.get('scheduler_last_epoch') == 200 and row.get('scheduler_step_count') == 201 and
             row.get('last_attempt_index') == saved_progress['attempt_index'] and row.get('optimizer_tensors_finite') is True and
             member.get('run_id') == run['run_id'] and member.get('actual_optimizer_writes') == writes and
             member.get('sha256') == _sha(by_name[f'training_state/rank-{rank}.pt']), 'SAVED_RANK_COUNTER_OR_SHA_DIFFERS')
        ids = row.get('used_attempt_ids')
        need(isinstance(ids, list) and len(ids) == len(set(ids)) == saved_progress['attempt_index'] and
             ids == ranks[0]['used_attempt_ids'] and all(isinstance(aid, str) and
             re.fullmatch('[0-9a-f]{32}', aid) for aid in ids) and
             type(row.get('policy_rejections')) is int and row['policy_rejections'] >= 0 and
             row['policy_rejections'] == ranks[0]['policy_rejections'] and
             all(type(row.get(key)) is int and row[key] >= 0 for key in
                 ('optimizer_calls', 'optimizer_rollbacks', 'full_budget_outers')) and
             row['optimizer_calls'] >= writes and row['full_budget_outers'] == saved_progress['full_budget_outer'],
             'SAVED_ATTEMPTS_OR_ACTUAL_CALL_COUNTERS_DIFFER')
    driver = receipt['driver_state']
    need(driver.get('completed_outer') == 200 and driver.get('probe_round') == receipt.get('reward_probe_round') == 150 and
         driver.get('probe_version') == receipt.get('reward_probe_version') and driver.get('reward_refresh_pending') is False and
         type(driver.get('source_consumed')) is int and driver['source_consumed'] >= 512000 and
         driver['source_consumed'] % 2560 == 0 and _sha(receipt['driver_file']) ==
         _sha(by_name['training_state/driver.pt']), 'SAVED_DRIVER_BINDING_DIFFERS')
    original, _ = _ref(receipt['source_run_manifest'])
    need(original == run, 'SOURCE_RUN_MANIFEST_DIFFERS')
    return receipt, source_identity, by_name


def _allocation(value, run, target):
    uuids = value.get('actual_cuda_runtime_uuids', [])
    need(value.get('accepted') is True and value.get('run_id') == run['run_id'] and
         value.get('training_world_size') == 4 and value.get('actual_job_state') == 'RUNNING' and
         isinstance(value.get('slurm_job_id'), str) and value['slurm_job_id'].isdecimal() and
         value.get('requested_gres') in ('gres:gpu:h100:4', 'gres:gpu:4', 'gres/gpu:4') and
         len(uuids) == len(set(uuids)) == 4 and all(isinstance(x, str) and x.startswith('GPU-') for x in uuids) and
         isinstance(value.get('node'), str) and bool(value['node']) and
         len(value.get('actual_allocated_indices', [])) == len(set(value['actual_allocated_indices'])) == 4,
         'ACTUAL_NATIVE4_ALLOCATION_REQUIRED')
    # Reuse the producer's pure string parser; no Slurm, CUDA or process call.
    from reproduction.common8.native_allocation import allocated_indices
    indices = allocated_indices(value.get('actual_allocation_detail', ''), value['node'], count=4)
    inventory = value.get('physical_gpu_inventory', [])
    need(indices == value['actual_allocated_indices'] and inventory and
         all(isinstance(row, list) and len(row) >= 2 for row in inventory), 'ACTUAL_DEVICE_INVENTORY_REQUIRED')
    nvml = {str(row[0]).strip(): str(row[1]).strip() for row in inventory}
    minors = {int(row[3] if len(row) > 3 else row[0]): str(row[1]).strip() for row in inventory}
    visible = value.get('visible_devices', [])
    need(len(nvml) == len(minors) == len(inventory) and len(set(nvml.values())) == len(inventory) and
         set(indices) <= set(minors) and {minors[index] for index in indices} == set(uuids) and
         len(visible) == len(set(visible)) == 4 and
         [nvml.get(str(device), str(device)) for device in visible] == uuids,
         'ACTUAL_SLURM_NVML_CUDA_MAPPING_DIFFERS')
    return uuids


def _no_training(target):
    logs = target/'logs'
    need(not any((logs/name).exists() for name in ('training-launch-claim.json', 'forbidden-training.log',
         'actual-updates.jsonl', 'actual-attempts.jsonl')) and not list(logs.glob('optimizer-rank-*.jsonl')) and
         not list(logs.glob('scheduler-rank-*.jsonl')) and
         not (target/'checkpoints/search-g1-common8/actor/global_step_201').exists(),
         'RESTORE_TARGET_HAS_TRAINING_OR_UPDATE201')


def _phase(document, phase, *, run, receipt, source_sha, files, allocation, host):
    need(document.get('schema_version') == 1 and document.get('role') == ROLE and document.get('phase') == phase and
         document.get('run_id') == run['run_id'] and document.get('source_receipt_sha256') == source_sha and
         _host(document.get('host_identity'), allocation['node']) == host and document.get('forbidden_call_counts') == {} and
         all(type(document.get(key)) is int and document[key] == 0 for key in ZERO) and
         document.get('independent_second_readback_equal') is (phase == 'B'), 'REAL_READBACK_PHASE_REQUIRED')
    phase_time = _time(document); driver_birth = _birth(document.get('driver_process_identity'))
    rows = document.get('actual_actor_readbacks', [])
    need(len(rows) == 4 and sorted(row.get('rank') for row in rows) == list(range(4)), 'FOUR_ACTUAL_READBACK_RANKS_REQUIRED')
    births = []; invariant = []
    for rank in range(4):
        row = next(value for value in rows if value['rank'] == rank)
        birth = _birth(row.get('process_identity')); births.append(birth)
        saved = next(value for value in receipt['ranks'] if value['rank'] == rank)
        need(row.get('phase') == phase and row.get('run_id') == run['run_id'] and
             row.get('source_step') == 200 and row.get('source_checkpoint') == receipt['checkpoint'] and
             row.get('slurm_job_id') == allocation['slurm_job_id'] and
             row.get('source_receipt_sha256') == source_sha and row.get('source_state_full_sha256') ==
             _sha(files[f'training_state/rank-{rank}.pt']) and
             row.get('actual_cuda_uuid') in allocation['actual_cuda_runtime_uuids'] and
             row.get('actual_optimizer_writes') == receipt['attempt_progress']['cumulative_writes'] and
             row.get('scheduler_last_epoch') == 200 and row.get('scheduler_step_count') == 201 and
             row.get('model_hf_exact_tensors_checked_on_rank0') == (receipt['hf_saved_tensor_count'] if rank == 0 else 0) and
             all(row.get(key) is True for key in ('model_equal', 'optimizer_equal', 'scheduler_equal', 'saved_rng_equal')) and
             row.get('forbidden_call_counts') == {} and _time(row) <= phase_time, 'ACTUAL_RANK_READBACK_DIFFERS')
        counters = row.get('actual_counters', {})
        need(set(COUNTERS) <= set(counters), 'ACTUAL_COUNTER_READBACK_REQUIRED')
        for key in COUNTERS:
            if key in saved:
                need(counters[key] == saved[key], 'SOURCE_COUNTER_READBACK_DIFFERS_'+key)
        need(counters['completed_outer'] == 200 and counters['actual_optimizer_writes'] == row['actual_optimizer_writes'] and
             counters['last_attempt_index'] == receipt['attempt_progress']['attempt_index'] and
             counters['full_budget_outers'] == receipt['attempt_progress']['full_budget_outer'], 'ACTUAL_COUNTER_PROGRESS_DIFFERS')
        invariant.append({key: value for key, value in row.items() if key not in ('phase', 'checked_unix', 'process_identity')})
    need(len(set(births+[driver_birth])) == 5 and len({x[3] for x in births+[driver_birth]}) == 1 and
         len({row['actual_cuda_uuid'] for row in rows}) == 4 and
         {row['actual_cuda_uuid'] for row in rows} == set(allocation['actual_cuda_runtime_uuids']),
         'FIVE_DISTINCT_OWNED_PROCESS_BIRTHS_REQUIRED')
    driver = document.get('driver', {})
    need(driver.get('completed_outer') == 200 and driver.get('actual_driver_rng_readback_equal') is True and
         driver.get('actual_candidate_stream_readback_equal') is True and driver.get('eligible_mapping_equal') is True and
         driver.get('actual_candidate_source_cursor') == receipt['driver_state']['source_consumed'] and
         driver.get('attempt_progress') == receipt['attempt_progress'] and driver.get('pending_offset') ==
         receipt['driver_state']['pending_offset'] and driver.get('actual_probe_round') == 150 and
         driver.get('actual_probe_version') == receipt['reward_probe_version'] and
         driver.get('reward_refresh_pending') is False and driver.get('question_probe') is None and
         driver.get('source_driver_full_sha256') == _sha(files['training_state/driver.pt']), 'ACTUAL_DRIVER_READBACK_DIFFERS')
    return dict(rows=invariant, driver=driver, births=births, driver_birth=driver_birth, time=phase_time)


def _source_no201(before, after, base):
    for row in (before, after):
        need(row.get('last_completed_outer') == 200 and row.get('update201_present') is False and
             row.get('checkpoint201_present') is False, 'SOURCE_UPDATE201_OR_INCOMPLETE')
        ledgers = row.get('source_ledgers')
        need(isinstance(ledgers, dict) and set(ledgers) == set(LEDGERS), 'ALL10_SOURCE_LEDGERS_REQUIRED')
        for name, record in ledgers.items():
            need(record.get('path') == str(base/'logs'/name) and type(record.get('bytes')) is int and
                 record['bytes'] > 0, 'SOURCE_LEDGER_RECORD_DIFFERS')
            _sha(record)
    need(before['source_ledgers'] == after['source_ledgers'], 'SOURCE_LEDGERS_CHANGED_DURING_RESTORE')


def verify_actual_final_restore(proof_path, *, base, run, checkpoint_summary, progress):
    """Authenticate completed linked native4 CUDA evidence; do not execute it."""
    base = _path(base); proof, proof_record = _read(proof_path)
    target = _path(proof['target_base'])
    need(target != base and Path(proof_record['path']) == target/'logs/final-restore-only-complete.json' and
         proof.get('schema_version') == 1 and proof.get('status') == GPU_STATUS and
         proof.get('kind') == proof.get('role') == proof.get('restore_only_role') == ROLE and
         proof.get('source_base') == str(base) and proof.get('source_commit') == run['source_commit'] and
         proof.get('run_id') == run['run_id'] and proof.get('source_step') == 200 and
         all(proof.get(key) is True for key in ('actual_all4_model_optimizer_scheduler_rng_dataequal',
             'actual_cuda_restore_accepted', 'actual_two_independent_readbacks', 'no_update201')),
         'COMPLETED_SEPARATE_RESTORE_ONLY_PROOF_REQUIRED')
    receipt, source_record, files = _source(proof['source_receipt'], base=base, run=run,
                                           checkpoint_summary=checkpoint_summary, progress=progress)
    need(proof.get('source_checkpoint') == receipt['checkpoint'] and proof.get('source_job') == receipt['source_job'] and
         proof.get('attempt_progress') == receipt['attempt_progress'] and proof.get('actual_optimizer_writes_per_rank') ==
         receipt['attempt_progress']['cumulative_writes'] and proof.get('candidate_source_cursor') ==
         receipt['driver_state']['source_consumed'], 'PROOF_SOURCE_COUNTERS_DIFFER')
    target_run, target_record = _read(target/'logs/run-manifest.json')
    binding = target_run.get('full_state_restore', {})
    need(target_run.get('run_id') == run['run_id'] and target_run.get('source_commit') == proof.get('target_source_commit') and
         isinstance(target_run['source_commit'], str) and re.fullmatch('[0-9a-f]{40}', target_run['source_commit']) and
         target_run.get('restore_only_role') == binding.get('kind') == ROLE and target_run.get('training_world_size') == 4 and
         target_run.get('question_probe') is None and target_run.get('all_correct_virtual_zero') is False and
         target_run.get('actor_micro_batch_per_rank') == run['actor_micro_batch_per_rank'] and
         target_run.get('actor_logical_micro_batch_per_rank') == run['actor_logical_micro_batch_per_rank'] == 4 and
         binding.get('source_receipt') == source_record['path'] and binding.get('source_receipt_sha256') == source_record['sha256'] and
         binding.get('source_base') == str(base) and binding.get('source_checkpoint') == receipt['checkpoint'] and
         binding.get('completed_outer') == 200 and
         isinstance(target_run.get('native_ray_execution_id'), str) and
         re.fullmatch('[0-9a-f]{32}', target_run['native_ray_execution_id']) and
         target_run['native_ray_execution_id'] != run.get('native_ray_execution_id'), 'EXACT_TARGET_RESTORE_BINDING_REQUIRED')
    from reproduction.easy2560_joint4.final_cpu_acceptance import published_source_binding
    operator_binding = published_source_binding(target, target_run, target_run['source_commit'])
    allocation, allocation_record = _ref(proof['allocation'], under=target)
    need(allocation_record['path'] == str(target/'logs/native-allocation-admission.json'), 'TARGET_ALLOCATION_PATH_DIFFERS')
    uuids = _allocation(allocation, run, target)
    documents = {}; records = {}
    for phase in ('A', 'B'):
        documents[phase], records[phase] = _ref(proof['readbacks'][phase], under=target)
        need(records[phase]['path'] == str(target/'logs'/f'final-restore-only-readback-{phase}.json'), 'READBACK_PATH_DIFFERS')
    need(records['A']['sha256'] != records['B']['sha256'], 'TWO_INDEPENDENT_READBACK_DOCUMENTS_REQUIRED')
    host = _host(documents['A'].get('host_identity'), allocation['node'])
    a, b = [_phase(documents[phase], phase, run=run, receipt=receipt, source_sha=source_record['sha256'],
                   files=files, allocation=allocation, host=host) for phase in ('A', 'B')]
    need(a['rows'] == b['rows'] and a['driver'] == b['driver'] and a['births'] == b['births'] and
         a['driver_birth'] == b['driver_birth'] and a['time'] < b['time'], 'A_B_ACTUAL_STATES_OR_BIRTHS_CHANGED')
    completion, completion_record = _ref(proof['execution_completion'], under=target)
    result = completion.get('task_result', {})
    need(completion.get('role') == ROLE and completion.get('run_id') == run['run_id'] and
         completion.get('source_receipt_sha256') == source_record['sha256'] and completion.get('ray_get_returned') is True and
         completion.get('task_result_sha256') == digest(canonical(result)) and
         _birth(completion.get('driver_process_identity')) == b['driver_birth'] and
         _host(completion.get('host_identity'), allocation['node']) == host and
         all(result.get(key) is True for key in ('actual_all4_model_optimizer_scheduler_rng_dataequal',
             'actual_cuda_restore_accepted', 'actual_two_independent_readbacks', 'no_update201')) and
         b['time'] <= _time(completion), 'ACTUAL_SUCCESSFUL_RAY_TASK_COMPLETION_REQUIRED')
    release, release_record = _ref(proof['owned_release'], under=target)
    expected_births = set(b['births']+[b['driver_birth']])
    need(release.get('role') == ROLE and release.get('all_recorded_processes_terminal') is True and
         release.get('assigned_gpu_contexts_released') is True and release.get('assigned_gpu_applications_after') == [] and
         _host(release.get('host_identity'), allocation['node']) == host and
         {_birth(row) for row in release.get('recorded_process_births', [])} == expected_births and
         release.get('execution_completion') == proof['execution_completion'] and
         _time(completion) <= _time(release) <= _time(proof), 'ACTUAL_OWNED_RELEASE_REQUIRED')
    observations = release.get('terminal_observations', [])
    need(len(observations) == 5 and {_birth(row.get('process_identity')) for row in observations} == expected_births,
         'FIVE_ACTUAL_TERMINAL_OBSERVATIONS_REQUIRED')
    for row in observations:
        need(row.get('terminal') is True, 'RECORDED_PROCESS_STILL_LIVE')
        actual = row.get('observed_process_identity'); reason = row.get('observation')
        need((reason == 'absent' and actual is None) or
             (reason == 'zombie' and isinstance(actual, dict) and actual.get('state') == 'Z' and
              _birth(actual) == _birth(row['process_identity'])) or
             (reason == 'birth_replaced' and isinstance(actual, dict) and
              actual.get('pid') == row['process_identity']['pid'] and _birth(actual) != _birth(row['process_identity'])),
             'CONCRETE_TERMINAL_OBSERVATION_REQUIRED')
    before, before_record = _ref(proof['source_no201_before'], under=target)
    after, after_record = _ref(proof['source_no201_after'], under=target)
    _source_no201(before, after, base)
    _no_training(target)
    return dict(schema_version=1, status=STATUS, actual_cuda_restore_accepted=True,
        actual_all4_model_optimizer_scheduler_rng_dataequal=True, actual_two_independent_readbacks=True,
        base=str(base), target_base=str(target), run_id=run['run_id'], source_commit=run['source_commit'],
        source_job=receipt['source_job'], restore_job=allocation['slurm_job_id'], source_checkpoint=receipt['checkpoint'],
        source_step=200, actual_optimizer_writes_per_rank=receipt['attempt_progress']['cumulative_writes'],
        candidate_source_cursor=receipt['driver_state']['source_consumed'], proof=proof_record,
        source_receipt=source_record, target_run_manifest=target_record, allocation=allocation_record,
        restore_operator_publication=operator_binding,
        readbacks=records, execution_completion=completion_record, owned_release=release_record,
        source_no201_before=before_record, source_no201_after=after_record, actual_cuda_uuids=uuids,
        new_training_updates=0, gpu_allocations=0, model_reloads=0,
        actual_restore_performed_by_this_cpu_reader=False, post_callback_rng_captured=False,
        scope='CPU authentication of existing source/archive SHA, independent native4 A/B actual readbacks and successful completed restore task; no CUDA or model operation performed by this consumer')
