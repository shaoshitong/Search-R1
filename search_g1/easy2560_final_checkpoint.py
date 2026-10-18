"""Final native4 Easy checkpoint provenance for a separate restore-only role.

Creation authenticates every original execution segment through outer200,
all five accepted archives and exact native51713 results. It then reads every
actual CP200 byte and all four CPU Adam/scheduler states. No training, CUDA,
post-final-evaluation RNG capture or actual recovery is claimed here.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from search_g1 import easy2560_restore as saved

STATUS = 'EASY2560_FINAL200_CHAINED_FULL_SHA_CPU_STATES_PASS'
KIND = 'easy2560_final200_restore_only'
STEP = 200
WORLD_SIZE = 4
RNG_KEYS = {'torch_cpu', 'torch_cuda', 'numpy', 'python', 'cuda_device',
            'actor_cuda_saved', 'rollout_cuda_saved', 'request_generators', 'vllm_version'}


def need(condition, message):
    if not condition:
        raise ValueError('EASY2560_FINAL200_' + message)


def _scope(run):
    # This is the final checkpoint of the resumed allocation, whose authorized
    # physical fallback may be 8 or 4. The original CP5 source keeps the default
    # physical-16-only admission in easy2560_restore.source/create_receipt.
    saved._scope(run, physical_micros=(16, 8, 4))
    binding = run.get('full_state_restore') or {}
    need(binding.get('kind') == saved.KIND and type(binding.get('completed_outer')) is int and
         1 <= binding['completed_outer'] < 200,
         'AUTHENTICATED_CONTINUATION_REQUIRED')
    need(Path(run['checkpoint_dir']).is_absolute(), 'ABSOLUTE_CHECKPOINT_ROOT_REQUIRED')


def _sha_identity(record, *, content):
    """Cross-node metadata admission; same-observer checks retain st_dev."""
    path = saved._identity(record)
    if content:
        before = path.stat(); length = record['bytes']; count = 0; h = hashlib.sha256()
        with path.open('rb') as stream:
            while count < length:
                block = stream.read(min(8 * 1024**2, length-count))
                if not block: break
                h.update(block); count += len(block)
        after = path.stat()
        need(saved._stat(before) == saved._stat(after) and count == length and
             h.hexdigest() == record['full_read_sha256'], 'PINNED_CONTENT_SHA_CHANGED_' + str(path))
    return path


def _final_acceptance(base, run):
    from reproduction.easy2560_joint4 import final_cpu_acceptance as final
    # The explicit callable preserves the canonical shared parquet/SHA reader;
    # fixtures replace only canonical input loading, never history validation.
    result = final.verify_final(base, training_commit=run['source_commit'], load_canonical=final.canonical)
    need(result['cpu_final_accepted'] is True and result['progress']['actual_completed_updates'] == STEP and
         result['progress']['no_update201'] is True and result['progress']['rank_count'] == WORLD_SIZE and
         result['evaluations']['final51713']['question_count'] == 51713 and
         result['inherited_history_assembled'] is True and result['single_original_job_prefix'] is False,
         'COMPLETE_AUTHENTICATED_FINAL_ACCEPTANCE_REQUIRED')
    return result


def _pin_inputs(base, run, acceptance):
    """Pin exactly the real inputs consumed by the CPU chain consumer."""
    records = {}
    def pin(label, path, expected=None, original=None):
        path = Path(path).resolve()
        if original is not None and original.get('prefix_only'):
            with path.open('rb') as stream: raw = stream.read(original['bytes'])
            need(len(raw) == original['bytes'] and hashlib.sha256(raw).hexdigest() == original['full_read_sha256'],
                 'HISTORICAL_PREFIX_SHA_DIFFERS')
            value = saved._record(path, relative_path=label, prefix=raw)
        else:
            value = saved._record(path, relative_path=label, expected=expected)
        records[label] = value
    binding = run['full_state_restore']
    receipt_path = Path(binding['source_receipt'])
    pin('original/restore-receipt.json', receipt_path, binding['source_receipt_sha256'])
    original = saved._json(receipt_path)
    parent = original['source_run_manifest']
    pin('original/run-manifest.json', parent['path'], parent['full_read_sha256'])
    history = original.get('historical_artifacts', {}).get('prefix_files')
    if history is None: history = list(original['historical_files'].values())
    for index, row in enumerate(history):
        if Path(row['path']).is_relative_to(Path(original['checkpoint'])):
            continue
        # Flattened receipts retain repeated per-execution relative names.
        # Keep each exact prefix instead of overwriting an ancestor's record.
        label = ('original/prefix/' + str(index) + '/' + row['relative_path']
                 if original.get('schema_version') == 2 else 'original/' + row['relative_path'])
        pin(label, row['path'], row['full_read_sha256'], row)
    reward = original['reward_probe_manifest']
    pin('original/reward-probe-manifest.json', reward['path'], reward['full_read_sha256'])
    names = ['logs/run-manifest.json', 'logs/native-allocation-admission.json',
             'logs/full-state-restored-workers.json', 'logs/full-state-restored-driver.json',
             'logs/actual-updates.jsonl', 'logs/actual-attempts.jsonl',
             'logs/validation-identities.jsonl', 'logs/final-identity.json',
             'preparation/run-source.json', 'source/' + run['source_commit'] + '/source-publication.json']
    names += ['logs/' + role + '-rank-' + str(rank) + '.jsonl'
              for role in ('optimizer', 'scheduler') for rank in range(WORLD_SIZE)]
    for name in names: pin('current/' + name, base/name)
    for step, info in acceptance['checkpoints'].items():
        for key in ('source_snapshot', 'accepted_archive', 'checkpoint_complete', 'training_state_manifest'):
            record = info[key]
            pin('archives/' + step + '/' + key, record['path'], record['sha256'])
    return records


def _checkpoint_metadata(receipt, run, checkpoint):
    complete = saved._json(checkpoint/'complete.json')
    state = saved._json(checkpoint/'training_state/manifest.json')
    files = {row['relative_path']: row for row in receipt['files']}
    progress = receipt['attempt_progress']; writes = progress['cumulative_writes']
    need(complete['run_id'] == run['run_id'] == receipt['run_id'] and
         complete['step'] == state['completed_outer'] == progress['completed_outer'] == STEP and
         Path(complete['checkpoint']) == checkpoint and complete['finite_all_saved_tensors'] is True and
         complete['actual_optimizer_writes_per_rank'] == receipt['actual_optimizer_writes_per_rank'] == writes,
         'COMPLETE_COUNTERS_DIFFER')
    need(state.get('question_probe') is None and sorted(row['rank'] for row in state['worker_files']) == list(range(4))
         and len(receipt['ranks']) == 4 and sorted(row['rank'] for row in receipt['ranks']) == list(range(4)),
         'FOUR_SAVED_RANKS_REQUIRED')
    need(all(name in files and files[name]['full_read_sha256'] == sha
             for name, sha in complete['files_sha256'].items()), 'COMPLETE_SHA_MAP_DIFFERS')
    index = saved._json(checkpoint/'model.safetensors.index.json')
    need(len(index['weight_map']) == complete['validated_tensor_count'] == receipt['hf_saved_tensor_count'] == 435 and
         set(index['weight_map'].values()) <= set(files), 'FULL_HF435_REQUIRED')
    for rank, row in enumerate(receipt['ranks']):
        native = next(value for value in state['worker_files'] if value['rank'] == rank)
        key = 'training_state/rank-' + str(rank) + '.pt'
        need(row['rank'] == rank and row['completed_outer'] == row['scheduler_last_epoch'] == STEP and
             row['scheduler_step_count'] == 201 and row['actual_optimizer_writes'] == writes and
             row['last_attempt_index'] == progress['attempt_index'] and row['optimizer_tensors_finite'] is True and
             len(set(row['used_attempt_ids'])) == len(row['used_attempt_ids']) == progress['attempt_index'] and
             row['used_attempt_ids'] == receipt['ranks'][0]['used_attempt_ids'] and
             row['policy_rejections'] == receipt['ranks'][0]['policy_rejections'] and
             native['path'] == files[key]['path'] and native['run_id'] == run['run_id'] and
             native['actual_optimizer_writes'] == writes and native['sha256'] == files[key]['full_read_sha256'],
             'SAVED_RANK_COUNTER_OR_SHA_DIFFERS')
        need(all(type(row.get(key)) is int and row[key] >= 0 for key in
                 ('optimizer_calls', 'optimizer_rollbacks', 'full_budget_outers')) and
             row['optimizer_calls'] >= writes and row['full_budget_outers'] <= STEP,
             'SAVED_ACTUAL_CALL_OR_ROLLBACK_COUNTERS_MISSING')
    need(200 <= writes <= 4000 and progress['aborted_attempts'] == progress['attempt_index']-STEP and
         state['driver_file']['sha256'] == receipt['driver_file']['full_read_sha256'] ==
         files['training_state/driver.pt']['full_read_sha256'] and
         state['driver_file']['path'] == receipt['driver_file']['path'] == files['training_state/driver.pt']['path'],
         'DRIVER_SHA_OR_PROGRESS_DIFFERS')
    summary = receipt['driver_state']
    need(summary['completed_outer'] == STEP and summary['probe_round'] == receipt['reward_probe_round'] == 150 and
         summary['probe_version'] == receipt['reward_probe_version'] and summary['reward_refresh_pending'] is False and
         summary['actual_driver_rng_saved'] is True and summary['source_consumed'] >= STEP*2560 and
         summary['source_consumed'] % 2560 == 0, 'DRIVER_SAVED_BOUNDARY_DIFFERS')
    return complete, state



def _archive_members(receipt):
    snapshot = saved._json(receipt['checkpoint_archive']['source_snapshot']['path'])
    need(snapshot['run_id'] == receipt['run_id'] and snapshot['source_commit'] == receipt['source_commit'] and
         snapshot['step'] == 200 and snapshot['training_world_size'] == 4 and snapshot['question_probe_saved'] is False,
         'FINAL_ARCHIVE_IDENTITY_DIFFERS')
    for row in receipt['files']:
        archived = snapshot['files'].get('checkpoint/' + row['relative_path'], {})
        need(archived.get('sha256') == row['full_read_sha256'] and archived.get('bytes') == row['bytes'],
             'FINAL_ARCHIVE_CHECKPOINT_MEMBER_DIFFERS')
    for row in receipt['reward_files'] + receipt['reward_encoder_files'] + [receipt['reward_probe_manifest']]:
        need(any(archived.get('path') == row['path'] and archived.get('sha256') == row['full_read_sha256'] and
                 archived.get('bytes') == row['bytes'] for archived in snapshot['files'].values()),
             'FINAL_ARCHIVE_REWARD_DEPENDENCY_DIFFERS')
    probe = saved._json(receipt['reward_probe_manifest']['path'])
    need(probe['run_id'] == receipt['run_id'] and probe['round'] == 150 and
         probe['version'] == receipt['reward_probe_version'], 'SAVED_REWARD_MANIFEST_DIFFERS')
    heads = {Path(row['path']).name: row for row in receipt['reward_files']}
    for key, name in (('commit', 'commit_probe.pkl'), ('boundary', 'boundary_probe.pkl')):
        need(str(Path(probe[key]).resolve()) == heads[name]['path'] and
             probe['head_sha256'][name] == heads[name]['full_read_sha256'], 'SAVED_REWARD_HEAD_DIFFERS')
    need(probe['calibration_report_sha256'] == heads['calibration-report.json']['full_read_sha256'] and
         {row['relative_path']: row['full_read_sha256'] for row in receipt['reward_encoder_files']} ==
         probe['encoder_sha256'], 'SAVED_REWARD_ENCODER_DIFFERS')


def create_receipt(base, checkpoint, output):
    base = Path(base).resolve(); checkpoint = Path(checkpoint).resolve(); output = Path(output).resolve()
    need(not output.exists() and not output.is_relative_to(checkpoint) and
         checkpoint == base/'checkpoints/search-g1-common8/actor/global_step_200', 'EXACT_CP200_AND_NEW_OUTPUT_REQUIRED')
    run = saved._json(base/'logs/run-manifest.json'); _scope(run)
    acceptance = _final_acceptance(base, run)
    history_records = _pin_inputs(base, run, acceptance)
    files = [saved._record(path, relative_path=str(path.relative_to(checkpoint)))
             for path in sorted(checkpoint.rglob('*')) if path.is_file()]
    by_name = {row['relative_path']: row for row in files}
    complete = saved._json(checkpoint/'complete.json'); state = saved._json(checkpoint/'training_state/manifest.json')
    need(complete['step'] == STEP and complete['finite_all_saved_tensors'] is True and
         all(name in by_name and by_name[name]['full_read_sha256'] == sha
             for name, sha in complete['files_sha256'].items()), 'COMPLETE_FILE_SHA_DIFFERS')
    need(sorted(row['rank'] for row in state['worker_files']) == list(range(4)), 'FOUR_SAVED_RANKS_REQUIRED')
    writes = complete['actual_optimizer_writes_per_rank']; ranks = []
    for item in sorted(state['worker_files'], key=lambda row: row['rank']):
        rank = item['rank']; path = checkpoint/'training_state'/('rank-' + str(rank) + '.pt')
        need(item['path'] == str(path) and item['sha256'] == by_name[str(path.relative_to(checkpoint))]['full_read_sha256'],
             'SAVED_RANK_FILE_DIFFERS')
        value = saved._load_cpu(path)
        need(value['run_id'] == run['run_id'] and value['source_commit'] == run['source_commit'] and
             isinstance(value.get('rng'), dict) and RNG_KEYS <= set(value['rng']), 'SAVED_RANK_RNG_OR_IDENTITY_DIFFERS')
        checked = saved._validate_rank(value, rank, STEP, writes)
        for key in ('optimizer_calls', 'optimizer_rollbacks', 'full_budget_outers'):
            need(type(value.get(key)) is int and value[key] >= 0, 'SAVED_ACTUAL_COUNTER_MISSING_' + key)
            checked[key] = value[key]
        ranks.append(checked); del value
    driver_file = by_name['training_state/driver.pt']; driver = saved._load_cpu(driver_file['path'])
    summary = saved._driver(driver, run, STEP, ranks); progress = copy.deepcopy(driver['attempt_progress'])
    need(progress['cumulative_writes'] == acceptance['progress']['cumulative_actual_optimizer_writes_per_rank'] and
         progress['attempt_index'] == acceptance['progress']['actual_attempt_count'], 'FINAL_LEDGER_AND_SAVED_COUNTS_DIFFER')
    pool = saved._record(Path(run['selection_eligible_pool_path']).resolve(), relative_path='external/eligible-question-pool.json',
                         expected=summary['prefix20_pool_binding']['eligible_pool_sha256'])
    matches = []
    for path in (base/'artifacts').glob('round_*/version-manifest.json'):
        probe = saved._json(path)
        if probe.get('version') == driver['probe_version'] and probe.get('run_id') == run['run_id']:
            matches.append((path, probe))
    need(len(matches) == 1, 'EXACT_REWARD_VERSION_REQUIRED')
    probe_path, probe = matches[0]
    need(probe['round'] == driver['probe_round'] == 150, 'FINAL_SAVED_ROUND150_REQUIRED')
    reward_files = [saved._record(Path(probe[key]).resolve(), relative_path='heads/' + name,
                                 expected=probe['head_sha256'][name])
                    for key, name in (('commit', 'commit_probe.pkl'), ('boundary', 'boundary_probe.pkl'))]
    reward_files.append(saved._record((Path(probe['commit']).parent/'calibration-report.json').resolve(),
        relative_path='heads/calibration-report.json', expected=probe['calibration_report_sha256']))
    encoders = [saved._record((Path(probe['encoder'])/name).resolve(), relative_path=name, expected=sha)
                for name, sha in sorted(probe['encoder_sha256'].items())]
    receipt = dict(schema_version=1, status=STATUS, kind=KIND, restore_only=True, step=STEP, world_size=WORLD_SIZE,
        source_base=str(base), checkpoint=str(checkpoint), source_job=acceptance['authenticated_history']['current_job'],
        run_id=run['run_id'], source_commit=run['source_commit'], current_source_commit=run['source_commit'],
        files=files, file_count=len(files), ranks=ranks, hf_saved_tensor_count=complete['validated_tensor_count'],
        actual_optimizer_writes_per_rank=writes, attempt_progress=progress, driver_file=driver_file,
        boundary_driver=driver_file, driver_state=summary,
        source_run_manifest=history_records['current/logs/run-manifest.json'], eligible_pool=pool,
        reward_probe_manifest=saved._record(probe_path.resolve(), relative_path='grounding/version-manifest.json'),
        reward_probe_version=probe['version'], reward_probe_round=150, reward_files=reward_files,
        reward_encoder_files=encoders, question_probe=None, history_records=history_records,
        authenticated_history=acceptance['authenticated_history'], final_cpu_acceptance=acceptance,
        checkpoint_archive=acceptance['checkpoints']['200'],
        historical_original_prefix=dict(completed_outers=[1,run['full_state_restore']['completed_outer']], **copy.deepcopy(run['full_state_restore'])),
        current_suffix_completed_outers=[run['full_state_restore']['completed_outer']+1,200], single_original_job_prefix=False,
        inherited_history_assembled=True, no_update201=True,
        sampling_rng_stage='saved_checkpoint_before_final51713', checkpoint_saved_before_final_evaluation=True,
        post_callback_rng_captured=False, reward_refresh_pending=False, final_evaluation_pending_at_saved_boundary=True,
        final_native51713_completed_at_publication=True, same_job_continuation=False,
        actual_cuda_restore_accepted=False, actual_model_optimizer_rng_dataequal_accepted=False,
        model_tensor_reload_performed=False, finite_model_tensors_authenticated_by_producer_sha=True,
        cpu_adam_scheduler_states_validated=True, source_code_publication_binding=acceptance['source_publication_binding'],
        checked_unix=time.time(), scope='actual chained Easy CP200 bytes and CPU state; separate no-training native4 recovery remains pending')
    _checkpoint_metadata(receipt, run, checkpoint)
    _archive_members(receipt)
    for row in files + reward_files + encoders + [pool, receipt['reward_probe_manifest']]: saved._identity(row)
    for row in history_records.values(): _sha_identity(row, content=True)
    saved._publish(output, receipt)
    return dict(path=str(output), sha256=saved.digest(output), status=STATUS, step=STEP, world_size=4,
                run_id=run['run_id'], file_count=len(files), actual_cuda_restore_accepted=False)


def source(receipt_path, *, rehash_checkpoint=False):
    """Read without Torch. The restore-only launcher may full-SHA CP once.

    Normal repeated worker dispatch keeps large CP identities stable and hashes
    all pinned small history; rank loaders additionally full-SHA their own state.
    Filesystem mount device numbers never cross the publisher/consumer boundary.
    """
    path = saved._regular(Path(receipt_path).resolve()); receipt = saved._json(path)
    need(receipt.get('schema_version') == 1 and receipt.get('status') == STATUS and receipt.get('kind') == KIND and
         receipt.get('step') == STEP and receipt.get('world_size') == 4 and receipt.get('restore_only') is True and
         receipt.get('actual_cuda_restore_accepted') is False and receipt.get('actual_model_optimizer_rng_dataequal_accepted') is False,
         'EXACT_RESTORE_ONLY_SCHEMA_REQUIRED')
    base = Path(receipt['source_base']); checkpoint = Path(receipt['checkpoint'])
    need(base.is_absolute() and base.resolve() == base and checkpoint == base/'checkpoints/search-g1-common8/actor/global_step_200'
         and checkpoint.resolve() == checkpoint, 'EXACT_SOURCE_CP200_REQUIRED')
    run_path = _sha_identity(receipt['source_run_manifest'], content=True); run = saved._json(run_path); _scope(run)
    need(run_path == base/'logs/run-manifest.json' and run['run_id'] == receipt['run_id'] and
         run['source_commit'] == receipt['source_commit'] == receipt['current_source_commit'], 'SOURCE_RUN_BINDING_DIFFERS')
    need(receipt['single_original_job_prefix'] is False and receipt['inherited_history_assembled'] is True and
         receipt['no_update201'] is True and receipt['post_callback_rng_captured'] is False and
         receipt['sampling_rng_stage'] == 'saved_checkpoint_before_final51713' and
         receipt['reward_refresh_pending'] is False and receipt['final_native51713_completed_at_publication'] is True,
         'FINAL_PHASE_OR_CHAIN_DIFFERS')
    names = {str(p.relative_to(checkpoint)) for p in checkpoint.rglob('*') if p.is_file()}
    need(len(receipt['files']) == receipt['file_count'] == len(names) and
         {row['relative_path'] for row in receipt['files']} == names, 'CHECKPOINT_MEMBER_SET_DIFFERS')
    for row in receipt['files']:
        need(Path(row['path']) == checkpoint/row['relative_path'], 'CHECKPOINT_MEMBER_PATH_DIFFERS')
        _sha_identity(row, content=rehash_checkpoint or row['relative_path'] in
                      ('complete.json', 'model.safetensors.index.json', 'training_state/manifest.json'))
    records = receipt['history_records']
    required = {'original/restore-receipt.json', 'original/run-manifest.json'} | {
        'current/' + name for name in ('logs/run-manifest.json', 'logs/native-allocation-admission.json',
        'logs/full-state-restored-workers.json', 'logs/full-state-restored-driver.json',
        'logs/actual-updates.jsonl', 'logs/actual-attempts.jsonl', 'logs/validation-identities.jsonl',
        'logs/final-identity.json', 'preparation/run-source.json',
        'source/' + run['source_commit'] + '/source-publication.json')}
    required |= {'current/logs/' + role + '-rank-' + str(rank) + '.jsonl'
                 for role in ('optimizer', 'scheduler') for rank in range(4)}
    required |= {'archives/' + str(step) + '/' + key for step in (5, 50, 100, 150, 200)
                 for key in ('source_snapshot', 'accepted_archive', 'checkpoint_complete', 'training_state_manifest')}
    need(required <= set(records) and receipt['source_run_manifest'] == records['current/logs/run-manifest.json'] and
         records['original/restore-receipt.json']['full_read_sha256'] == run['full_state_restore']['source_receipt_sha256'],
         'AUTHENTICATED_CHAIN_INPUT_RECORDS_MISSING')
    for label, row in records.items():
        need(row['relative_path'] == label and (not label.startswith('current/') or
             Path(row['path']) == base/label[len('current/'):]), 'CHAIN_INPUT_PATH_DIFFERS')
        _sha_identity(row, content=True)
    allocation = saved._json(records['current/logs/native-allocation-admission.json']['path'])
    need(allocation['run_id'] == run['run_id'] and allocation['accepted'] is True and
         allocation['slurm_job_id'] == receipt['source_job'], 'SOURCE_ALLOCATION_DIFFERS')
    for row in receipt['reward_files'] + [receipt['reward_probe_manifest'], receipt['eligible_pool']]:
        _sha_identity(row, content=True)
    for row in receipt['reward_encoder_files']: _sha_identity(row, content=rehash_checkpoint)
    acceptance = receipt['final_cpu_acceptance']; history = receipt['authenticated_history']
    need(acceptance['run_id'] == run['run_id'] and acceptance['source_commit'] == run['source_commit'] and
         acceptance['cpu_final_accepted'] is True and acceptance['progress']['no_update201'] is True and
         acceptance['progress']['actual_completed_updates'] == 200 and acceptance['progress']['rank_count'] == 4 and
         acceptance['evaluations']['final51713']['question_count'] == 51713 and
         acceptance['authenticated_history'] == history and history['current_job'] == receipt['source_job'] and
         history['current_source_commit'] == run['source_commit'] and
         history['completed_outer'] == run['full_state_restore']['completed_outer'] and
         history['single_original_job_prefix'] is False and history['inherited_history_assembled'] is True and
         history['source_receipt_sha256'] == run['full_state_restore']['source_receipt_sha256'] and
         receipt['checkpoint_archive'] == acceptance['checkpoints']['200'], 'AUTHENTICATED_CHAIN_SUMMARY_DIFFERS')
    _checkpoint_metadata(receipt, run, checkpoint)
    _archive_members(receipt)
    return path, receipt, checkpoint, base, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    create = sub.add_parser('create'); create.add_argument('--base', required=True)
    create.add_argument('--checkpoint', required=True); create.add_argument('--output', required=True)
    read = sub.add_parser('source'); read.add_argument('--receipt', required=True)
    read.add_argument('--rehash-checkpoint', action='store_true')
    args = parser.parse_args()
    if args.mode == 'create': result = create_receipt(args.base, args.checkpoint, args.output)
    else:
        path, value, _, _, _ = source(args.receipt, rehash_checkpoint=args.rehash_checkpoint)
        result = dict(path=str(path), sha256=saved.digest(path), status=STATUS, step=200, world_size=4,
                      run_id=value['run_id'], full_checkpoint_sha_rechecked=args.rehash_checkpoint,
                      actual_cuda_restore_accepted=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
