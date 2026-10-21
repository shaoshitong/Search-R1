"""Immutable CPU evidence sidecar for the reviewed fresh Easy2560 run.

This independent operator does not load Torch, allocate GPUs, train, restore,
modify a producer checkpoint, or amend an accepted archive. Large model/data
and selected token traces remain references to the existing remote objects.
Exact original-job prefixes and a receipt-authenticated native-four recovery
chain retain separate source/job identities; this operator performs no recovery.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time
import uuid

REVIEWED_COMMIT = '93eda66a978dae7a1086fdb2e6f07ef4a7fc3d48'
STEPS = (5, 50, 100, 150, 200)
STATUS = 'EASY2560_CHECKPOINT_EVIDENCE_SEALED'
ARCHIVE_STATUS = 'EASY2560_FULL_CHECKPOINT_SOURCE_ARCHIVE_SHA_PASS'
MAX_SMALL = 128 * 1024**2
MAX_LINE = 32 * 1024**2


def need(ok, message):
    if not ok:
        raise ValueError('EASY2560_EVIDENCE_' + message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + '\n').encode()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_uid)


def owned_base(value):
    p = Path(value)
    need(p.is_absolute() and p.resolve() == p and re.fullmatch(
        r'/data/user/jxu957/experiments/easy2560-corollout-joint4-[A-Za-z0-9._-]+', str(p)),
        'DEDICATED_CANONICAL_BASE_REQUIRED')
    # The human-authorized shared experiment root was created by mou176;
    # jxu957 owns the actual run/evidence files and executes this operator.
    # Directory ownership is not equivalent to execution/file ownership.
    need(os.getuid() == os.geteuid() == 14090, 'JXU_EXECUTION_UID_REQUIRED')
    st = p.stat()
    need(stat.S_ISDIR(st.st_mode) and st.st_uid in (14090, 21930) and st.st_gid == 21919,
         'AUTHORIZED_SHARED_BASE_REQUIRED')
    need(os.access(p, os.W_OK | os.X_OK), 'SHARED_BASE_NOT_WRITABLE')
    return p


def regular(path, *, external=False):
    path = Path(path)
    need(path.is_absolute(), 'ABSOLUTE_SOURCE_REQUIRED')
    if external:
        path = path.resolve(strict=True)
    need(path.resolve() == path and not path.is_symlink(), 'SYMLINK_SOURCE')
    st = path.stat()
    need(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid(), 'REGULAR_OWNED_SOURCE_REQUIRED')
    return path


def read_small(path, *, expected=None, external=False):
    path = regular(path, external=external)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        need(before.st_size <= MAX_SMALL, 'SMALL_SOURCE_LIMIT')
        raw = stream.read(MAX_SMALL + 1)
        after = os.fstat(stream.fileno())
    need(identity(before) == identity(after) == identity(path.stat()), 'SOURCE_CHANGED_DURING_READ')
    actual = digest(raw)
    need(expected is None or actual == expected, 'SOURCE_SHA_DIFFERS')
    return raw, dict(path=str(path), bytes=len(raw), sha256=actual,
                     device=after.st_dev, inode=after.st_ino)


def json_small(path, **kwargs):
    raw, record = read_small(path, **kwargs)
    return json.loads(raw), raw, record


def prefix(path, terminal):
    """Stop at the requested complete row; never parse a later partial suffix."""
    path = regular(path)
    rows = []; chunks = []
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        while True:
            raw = stream.readline(MAX_LINE + 1)
            need(raw and raw.endswith(b'\n') and len(raw) <= MAX_LINE, 'MISSING_OR_PARTIAL_CP_PREFIX')
            row = json.loads(raw); rows.append(row); chunks.append(raw)
            if terminal(row):
                break
        length = stream.tell()
        after = os.fstat(stream.fileno())
        need((before.st_dev, before.st_ino, before.st_uid) ==
             (after.st_dev, after.st_ino, after.st_uid) and after.st_size >= length,
             'LEDGER_REPLACED_DURING_READ')
        stream.seek(0)
        need(stream.read(length) == b''.join(chunks), 'LEDGER_PREFIX_CHANGED_DURING_READ')
    need((path.stat().st_dev, path.stat().st_ino) == (before.st_dev, before.st_ino),
         'LEDGER_REPLACED')
    raw = b''.join(chunks)
    return rows, raw, dict(path=str(path), bytes=len(raw), sha256=digest(raw),
                           prefix_only=True, later_suffix_excluded=True,
                           device=before.st_dev, inode=before.st_ino)


def attempt_identity(row):
    value = row if 'attempt_index' in row else row.get('context', {})
    return value['attempt_index'], value['attempt_id']


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Bundle:
    def __init__(self, root):
        self.root = root; self.files = {}; self.source_reads = []

    def put(self, name, raw, source=None):
        part = Path(name)
        need(not part.is_absolute() and '..' not in part.parts and name not in self.files,
             'UNSAFE_OR_DUPLICATE_OUTPUT')
        target = self.root / part; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        actual = target.read_bytes()
        need(actual == raw, 'DESTINATION_READBACK_DIFFERS')
        self.files[name] = dict(bytes=len(raw), sha256=digest(actual))
        if source is not None:
            self.source_reads.append(dict(source, sealed_as=name))

    def copy(self, name, source, *, parse_json=True, **kwargs):
        raw, record = read_small(source, **kwargs); self.put(name, raw, record)
        return json.loads(raw) if parse_json else None


RESTORE_STATUS = 'EASY2560_NATIVE4_FULL_SHA_CPU_STATES_PASS'


def _receipt_record_raw(record, path):
    """Authenticate a receipt-bound immutable file or an exact append-only prefix."""
    need(isinstance(record, dict) and record.get('path') == str(path) and
         type(record.get('bytes')) is int and 0 < record['bytes'] <= MAX_SMALL,
         'HISTORICAL_FILE_RECORD_REQUIRED')
    sha = record.get('full_read_sha256', record.get('sha256'))
    need(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha), 'HISTORICAL_FILE_SHA_REQUIRED')
    path = regular(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno()); raw = stream.read(record['bytes'])
        after = os.fstat(stream.fileno())
    need((before.st_dev, before.st_ino, before.st_uid) ==
         (after.st_dev, after.st_ino, after.st_uid) and len(raw) == record['bytes'] and
         after.st_size >= len(raw) and digest(raw) == sha and
         (path.stat().st_dev, path.stat().st_ino) == (before.st_dev, before.st_ino),
         'HISTORICAL_FILE_PREFIX_SHA_DIFFERS')
    if not record.get('prefix_only', False):
        need(identity(before) == identity(after) == identity(path.stat()) and
             after.st_size == len(raw), 'HISTORICAL_IMMUTABLE_FILE_CHANGED')
    for key, value in (('device', after.st_dev), ('inode', after.st_ino)):
        need(record.get(key, value) == value, 'HISTORICAL_FILE_IDENTITY_DIFFERS')
    return raw, dict(path=str(path), bytes=len(raw), sha256=sha,
                     prefix_only=bool(record.get('prefix_only', False)))


def _legacy_restore_chain(base, run, *, bundle=None, sealed_directory=None, namespace=""):
    """Read a truthful two-execution chain; never relabel old rows as the new job.

    This validates existing CPU/full-SHA and actual worker/driver readback
    receipts. It performs no Torch/model reload, training, or CUDA allocation.
    Sealed verification reads only byte-identical small copies in the sidecar.
    """
    binding = run.get('full_state_restore')
    if not binding:
        return None
    need(isinstance(binding, dict) and binding.get('kind') == 'easy2560_native4',
         'CROSS_JOB_AUTHENTICATED_PREFIX_NOT_IMPLEMENTED')
    base = Path(base); sealed = Path(sealed_directory) if sealed_directory else None
    def get(name, path, *, expected=None, record=None):
        if sealed is not None:
            raw, actual = read_small(sealed/'chain'/namespace/name, expected=expected)
            if record is not None:
                need(len(raw) == record['bytes'] and digest(raw) ==
                     record.get('full_read_sha256', record.get('sha256')), 'SEALED_HISTORICAL_FILE_DIFFERS')
        elif record is not None:
            raw, actual = _receipt_record_raw(record, Path(path))
        else:
            raw, actual = read_small(Path(path), expected=expected)
        if bundle is not None:
            bundle.put('chain/'+namespace+name, raw, actual)
        return raw
    receipt_path = Path(binding['source_receipt'])
    receipt_sha = binding.get('source_receipt_sha256')
    need(isinstance(receipt_sha, str) and re.fullmatch('[0-9a-f]{64}', receipt_sha), 'RESTORE_RECEIPT_SHA_REQUIRED')
    receipt_raw = get('restore-receipt.json', receipt_path, expected=receipt_sha)
    receipt = json.loads(receipt_raw)
    source_base = Path(binding['source_base'])
    need(source_base.is_absolute() and source_base != base and
         (sealed is not None or source_base.resolve() == source_base), 'RESTORE_SOURCE_BASE_DIFFERS')
    checkpoint = source_base/'checkpoints/search-g1-common8/actor/global_step_5'
    need(receipt.get('status') == RESTORE_STATUS and receipt.get('kind') == 'easy2560_native4' and
         receipt.get('source_base') == str(source_base) and receipt.get('checkpoint') == str(checkpoint) and
         receipt.get('run_id') == run['run_id'] and
         receipt.get('step') == binding.get('completed_outer') == 5 and
         receipt.get('world_size') == 4 and receipt.get('actual_optimizer_writes_per_rank') ==
         binding.get('actual_optimizer_writes_per_rank') == 100 and
         binding.get('source_checkpoint') == str(checkpoint), 'RESTORE_CP5_RECEIPT_BINDING_DIFFERS')
    progress = receipt['attempt_progress']
    need(progress.get('completed_outer') == 5 and progress.get('cumulative_writes') == 100 and
         type(progress.get('attempt_index')) is int and progress['attempt_index'] >= 5,
         'RESTORE_CP5_PROGRESS_DIFFERS')
    old_run_record = receipt.get('source_run_manifest')
    need(isinstance(old_run_record, dict) and old_run_record.get('path') ==
         str(source_base/'logs/run-manifest.json'), 'SOURCE_RUN_MANIFEST_RECORD_REQUIRED')
    old_raw = get('source-run-manifest.json', source_base/'logs/run-manifest.json', record=old_run_record)
    old_run = json.loads(old_raw)
    need(old_run.get('run_id') == run['run_id'] and not old_run.get('full_state_restore') and
         old_run.get('source_commit') == receipt.get('source_commit') == binding.get('source_commit') and
         receipt.get('source_job') == binding.get('source_job') and
         old_run.get('easy2560_selection') is True and old_run.get('training_world_size') == 4 and
         old_run.get('all_correct_virtual_zero') is False and old_run.get('question_probe') is None,
         'RESTORE_ORIGINAL_RUN_BINDING_DIFFERS')
    artifacts = receipt.get('historical_artifacts', {})
    records = artifacts.get('prefix_files')
    if records is not None:
        need(isinstance(records, list), 'HISTORICAL_PREFIX_RECORDS_REQUIRED')
        history = {}
        for record in records:
            rel = record.get('relative_path')
            need(isinstance(rel, str) and rel not in history, 'DUPLICATE_HISTORICAL_PREFIX')
            history[rel] = record
    else:
        history = receipt.get('historical_files', {})
    need(isinstance(history, dict), 'HISTORICAL_PREFIX_RECORDS_REQUIRED')
    cached = {}
    for rel, record in history.items():
        part = Path(rel)
        need(isinstance(rel, str) and rel not in ('', '.') and not part.is_absolute() and
             '..' not in part.parts, 'HISTORICAL_PATH_ESCAPE')
        expected_path = checkpoint/Path(*part.parts[1:]) if part.parts[0] == 'checkpoint' else source_base/part
        need(record.get('path') == str(expected_path) and type(record.get('bytes')) is int and
             0 < record['bytes'] <= MAX_SMALL and isinstance(record.get('full_read_sha256'), str) and
             re.fullmatch('[0-9a-f]{64}', record['full_read_sha256']), 'HISTORICAL_PATH_OR_SHA_DIFFERS')
        cached[rel] = get('historical/'+rel, expected_path, record=record)
    required = {'logs/actual-updates.jsonl', 'logs/actual-attempts.jsonl',
                'logs/validation-identities.jsonl', 'logs/native-allocation-admission.json',
                'logs/formal-inline-development-after.json'} | {
                'logs/%s-rank-%d.jsonl' % (role, rank) for role in ('scheduler', 'optimizer') for rank in range(4)}
    need(required.issubset(cached), 'MISSING_AUTHENTICATED_HISTORICAL_PREFIX')
    # CP5 archive acceptance may have completed after the CPU restore receipt.
    # Read its real immutable metadata; never invent it or label it a new-job CP.
    archive_rel = 'preparation/checkpoint-archives/global_step_5/'
    for name in ('source-snapshot.json', 'complete.json'):
        rel = archive_rel + name
        if rel not in cached:
            cached[rel] = get('historical/'+rel, source_base/rel)
    source_snapshot = json.loads(cached[archive_rel+'source-snapshot.json'])
    source_archive = json.loads(cached[archive_rel+'complete.json'])
    source_complete = json.loads(cached['checkpoint/complete.json'])
    source_state = json.loads(cached['checkpoint/training_state/manifest.json'])
    need(all(source_snapshot.get(k) == v and source_archive.get(k) == v for k,v in
             dict(base=str(source_base), run_id=run['run_id'], source_commit=old_run['source_commit'], step=5).items()) and
         source_snapshot.get('training_world_size') == 4 and source_snapshot.get('question_probe_saved') is False and
         source_archive.get('status') == ARCHIVE_STATUS and source_archive.get('actual_cuda_restore_accepted') is False and
         source_archive.get('fresh_destination_sha_verified') is True and source_archive.get('destination_fsynced') is True and
         source_archive.get('fresh_source_after_archive_sha_verified') is True and
         source_archive.get('snapshot_sha256') == digest(cached[archive_rel+'source-snapshot.json']) and
         source_archive.get('file_count') == len(source_snapshot['files']) and
         source_archive.get('total_bytes') == source_snapshot.get('total_bytes') ==
         sum(v['bytes'] for v in source_snapshot['files'].values()), 'ORIGINAL_CP5_ARCHIVE_ACCEPTANCE_DIFFERS')
    source_files = source_snapshot['files']
    need(source_files['checkpoint/complete.json']['sha256'] == digest(cached['checkpoint/complete.json']) and
         source_files['metadata/run-manifest.json']['sha256'] == digest(old_raw) and
         source_complete.get('run_id') == run['run_id'] and source_complete.get('step') == 5 and
         source_complete.get('finite_all_saved_tensors') is True and
         source_complete.get('actual_optimizer_writes_per_rank') == 100 and
         source_state.get('completed_outer') == 5 and source_state.get('question_probe') is None and
         sorted(v['rank'] for v in source_state['worker_files']) == list(range(4)),
         'ORIGINAL_CP5_METADATA_SHA_DIFFERS')
    for row in source_state['worker_files']:
        need(row.get('run_id') == run['run_id'] and row.get('actual_optimizer_writes') == 100 and
             row.get('sha256') == source_complete['files_sha256'].get(f'training_state/rank-{row["rank"]}.pt'),
             'ORIGINAL_RANK_STATE_MANIFEST_DIFFERS')
    for name, sha in source_complete['files_sha256'].items():
        need(source_files.get('checkpoint/'+name, {}).get('sha256') == sha, 'ORIGINAL_CP5_ARCHIVE_MEMBER_DIFFERS')
    for name, sha in old_run['source_sha256'].items():
        need(source_files.get('source/'+name, {}).get('sha256') == sha, 'ORIGINAL_CP5_SOURCE_MAP_DIFFERS')
    old_publication = old_runtime = None
    if old_run['source_commit'] != REVIEWED_COMMIT:
        publication_rel = 'source/'+old_run['source_commit']+'/source-publication.json'
        for rel in (publication_rel, 'preparation/run-source.json'):
            if rel not in cached: cached[rel] = get('historical/'+rel, source_base/rel)
        old_publication = json.loads(cached[publication_rel])
        old_runtime = json.loads(cached['preparation/run-source.json'])
    old_source_binding = published_source_binding(source_base, old_run, old_run['source_commit'],
                                                old_publication, old_runtime, source_snapshot)
    driver_record = receipt.get('driver_file', {})
    need(driver_record.get('path') == str(checkpoint/'training_state/driver.pt') and
         driver_record.get('full_read_sha256') == source_complete['files_sha256'].get('training_state/driver.pt'),
         'ORIGINAL_DRIVER_RECEIPT_SHA_DIFFERS')
    reward_record = receipt.get('reward_probe_manifest')
    need(isinstance(reward_record, dict), 'ORIGINAL_REWARD_PROBE_RECORD_REQUIRED')
    reward_raw = get('reward-probe-manifest.json', Path(reward_record['path']), record=reward_record)
    reward_probe = json.loads(reward_raw)
    need(reward_probe.get('run_id') == run['run_id'] and reward_probe.get('round') ==
         receipt.get('reward_probe_round') == 0 and reward_probe.get('version') == receipt.get('reward_probe_version') and
         any(name.startswith('grounding/') and row.get('path') == reward_record['path'] and
             row.get('sha256') == digest(reward_raw) for name,row in source_files.items()),
         'ORIGINAL_REWARD_PROBE_ARCHIVE_BINDING_DIFFERS')
    historical_updates = _historical_rows(dict(historical=cached), 'logs/actual-updates.jsonl')
    need(source_snapshot.get('completed_update') == historical_updates[-1] and
         [r['step'] for r in historical_updates] == list(range(1,6)), 'ORIGINAL_CP5_UPDATE_PREFIX_DIFFERS')
    qa5_raw = cached['logs/formal-inline-development-after.json']; qa5 = json.loads(qa5_raw)
    need(qa5.get('run_id') == run['run_id'] and qa5.get('step') == 5 and
         qa5.get('role') == 'formal_inline_development_evaluation' and len(qa5.get('rows', [])) == 512,
         'ORIGINAL_QA5_RESULT_REQUIRED')
    old_allocation = json.loads(cached['logs/native-allocation-admission.json'])
    allocation = json.loads(get('current-allocation.json', base/'logs/native-allocation-admission.json'))
    need(old_allocation.get('accepted') is True and old_allocation.get('run_id') == run['run_id'] and
         old_allocation.get('slurm_job_id') == receipt['source_job'] and
         allocation.get('accepted') is True and allocation.get('run_id') == run['run_id'] and
         allocation.get('slurm_job_id') != receipt['source_job'], 'RESTORE_ALLOCATION_CHAIN_DIFFERS')
    job = allocation['slurm_job_id']
    workers_raw = get('full-state-restored-workers.json', base/'logs/full-state-restored-workers.json')
    driver_raw = get('full-state-restored-driver.json', base/'logs/full-state-restored-driver.json')
    workers = json.loads(workers_raw); driver = json.loads(driver_raw)
    need(workers.get('run_id') == run['run_id'] and workers.get('source_commit') == run['source_commit'] and
         workers.get('slurm_job_id') == job and workers.get('source_receipt_sha256') == receipt_sha and
         workers.get('actual_world_size') == 4, 'RESTORED_WORKERS_AGGREGATE_DIFFERS')
    ranks = workers.get('ranks', [])
    need(len(ranks) == 4 and sorted(r.get('rank') for r in ranks) == list(range(4)) and
         len(set(allocation.get('actual_cuda_runtime_uuids', []))) == 4 and
         {r.get('actual_cuda_uuid') for r in ranks} == set(allocation['actual_cuda_runtime_uuids']),
         'RESTORED_NATIVE4_UUIDS_DIFFERS')
    for row in ranks:
        need(row.get('run_id') == row.get('source_run_id') == run['run_id'] and
             row.get('source_receipt_sha256') == receipt_sha and row.get('source_job') == receipt['source_job'] and
             row.get('source_commit') == old_run['source_commit'] and row.get('source_step') == 5 and
             row.get('source_checkpoint') == str(checkpoint) and row.get('slurm_job_id') == job and
             row.get('world_size') == 4 and row.get('actual_optimizer_writes') == 100 and
             row.get('scheduler_last_epoch') == 5 and row.get('last_attempt_index') == progress['attempt_index'] and
             row.get('actual_model_collective_readback_equal') is True and
             row.get('actual_adam_scheduler_readback_equal') is True and row.get('actual_rng_readback_equal') is True,
             'RESTORED_RANK_READBACK_DIFFERS')
        source_state = next((r for r in receipt.get('files', []) if r.get('path') ==
                            str(checkpoint/f'training_state/rank-{row["rank"]}.pt')), None)
        need(source_state is not None and row.get('source_state_full_sha256') ==
             source_state.get('full_read_sha256') == source_complete['files_sha256'].get(
                 f'training_state/rank-{row["rank"]}.pt'), 'RESTORED_RANK_SOURCE_STATE_SHA_DIFFERS')
        process = row.get('process_identity', {})
        need(type(process.get('pid')) is int and process['pid'] > 0 and
             type(process.get('start_ticks')) is int and process['start_ticks'] > 0,
             'RESTORED_RANK_BIRTH_REQUIRED')
    need(len({(r['process_identity']['pid'], r['process_identity']['start_ticks']) for r in ranks}) == 4,
         'RESTORED_NATIVE4_WORKER_BIRTHS_NOT_DISTINCT')
    need(next(r for r in ranks if r['rank'] == 0).get('model_hf_exact_tensors_checked_on_rank0') == 435,
         'RESTORED_RANK0_FULL_MODEL_READBACK_REQUIRED')
    need(driver.get('run_id') == run['run_id'] and driver.get('source_receipt_sha256') == receipt_sha and
         driver.get('source_job') == receipt['source_job'] and driver.get('source_commit') == old_run['source_commit'] and
         driver.get('source_step') == 5 and driver.get('slurm_job_id') == job and
         driver.get('source_driver_full_sha256') == receipt.get('driver_file', {}).get('full_read_sha256') and
         isinstance(driver.get('source_driver_full_sha256'), str) and
         receipt.get('driver_state', {}).get('source_consumed') == 12800 and
         driver.get('actual_probe_version') == receipt.get('reward_probe_version') and
         driver.get('actual_probe_round') == receipt.get('reward_probe_round') == 0 and
         driver.get('actual_candidate_source_cursor') == 12800 and
         driver.get('attempt_progress') == progress and driver.get('actual_driver_rng_readback_equal') is True and
         driver.get('actual_candidate_stream_readback_equal') is True and
         driver.get('post_callback_rng_captured') is False and driver.get('same_job_continuation') is False,
         'RESTORED_DRIVER_READBACK_DIFFERS')
    public = dict(source_base=str(source_base), source_checkpoint=str(checkpoint), completed_outer=5,
        source_commit=old_run['source_commit'], source_job=receipt['source_job'], current_job=job,
        current_source_commit=run['source_commit'], source_receipt_sha256=receipt_sha,
        historical_attempt_index=progress['attempt_index'], historical_optimizer_writes=100,
        original_CP5_archive_snapshot_sha256=digest(cached[archive_rel+'source-snapshot.json']),
        original_CP5_archive_acceptance_sha256=digest(cached[archive_rel+'complete.json']),
        original_QA5_sha256=digest(qa5_raw), original_source_publication_binding=old_source_binding,
        original_reward_probe_manifest_sha256=digest(reward_raw),
        single_original_job_prefix=False, inherited_history_assembled=True,
        native4_worker_driver_restore_proofs_authenticated=True,
        worker_readback_receipt_sha256=digest(workers_raw), driver_readback_receipt_sha256=digest(driver_raw),
        post_callback_rng_captured=False, actual_restore_performed_by_this_cpu_reader=False)
    return dict(public=public, source_base=source_base, source_run=old_run,
                historical=cached, records=history, progress=progress, receipt=receipt,
                boundary=5, cutoff_attempt=progress['attempt_index'], source_job=receipt['source_job'], current_job=job,
                segments=[dict(first_outer=1, last_outer=5, first_attempt=1, last_attempt=progress['attempt_index'],
                    base=source_base, run=old_run, job=receipt['source_job'])])



def authenticated_restore_chain(base, run, *, bundle=None, sealed_directory=None, namespace="", _seen=None):
    """Authenticate each actual restore edge and preserve every execution segment.

    Schema2 receipts carry exact flattened update/attempt rows. Only immutable
    small receipts and ledger prefixes are consumed here; an obsolete rolling
    checkpoint's tensor/driver payloads need not remain after a newer save.
    """
    binding = run.get('full_state_restore')
    if not binding:
        return None
    need(isinstance(binding, dict) and binding.get('kind') == 'easy2560_native4',
         'CROSS_JOB_AUTHENTICATED_PREFIX_NOT_IMPLEMENTED')
    base = Path(base); sealed = Path(sealed_directory) if sealed_directory else None
    def get(name, path, *, expected=None, record=None):
        if sealed is not None:
            raw, actual = read_small(sealed/'chain'/namespace/name, expected=expected)
            if record is not None:
                need(len(raw) == record['bytes'] and digest(raw) ==
                     record.get('full_read_sha256', record.get('sha256')), 'SEALED_HISTORICAL_FILE_DIFFERS')
        elif record is not None:
            raw, actual = _receipt_record_raw(record, Path(path))
        else:
            raw, actual = read_small(Path(path), expected=expected)
        if bundle is not None:
            bundle.put('chain/'+namespace+name, raw, actual)
        return raw
    # Peek before copying so legacy sidecars keep their exact public format.
    receipt_path = Path(binding['source_receipt'])
    if sealed is not None:
        raw, _ = read_small(sealed/'chain'/namespace/'restore-receipt.json',
                            expected=binding['source_receipt_sha256'])
    else:
        raw, _ = read_small(receipt_path, expected=binding['source_receipt_sha256'])
    receipt = json.loads(raw)
    if receipt.get('schema_version', 1) == 1 and receipt.get('step') == 5:
        return _legacy_restore_chain(base, run, bundle=bundle, sealed_directory=sealed,
                                     namespace=namespace)
    receipt_raw = get('restore-receipt.json', receipt_path, expected=binding['source_receipt_sha256'])
    seen = set() if _seen is None else set(_seen)
    edge = (str(base), binding['source_receipt_sha256'])
    need(edge not in seen and len(seen) < 200, 'CYCLIC_OR_EXCESSIVE_RESTORE_LINEAGE')
    seen.add(edge)
    outer = binding.get('completed_outer'); source_base = Path(binding['source_base'])
    checkpoint = Path(binding['source_checkpoint'])
    need(type(outer) is int and 1 <= outer < 200 and source_base.is_absolute() and source_base != base and
         checkpoint == source_base/'checkpoints/search-g1-common8/actor'/('global_step_%d' % outer) and
         receipt.get('schema_version') == 2 and receipt.get('status') == RESTORE_STATUS and
         receipt.get('kind') == 'easy2560_native4' and receipt.get('step') == outer and receipt.get('world_size') == 4 and
         receipt.get('source_base') == str(source_base) and receipt.get('checkpoint') == str(checkpoint) and
         receipt.get('run_id') == run['run_id'] and receipt.get('source_commit') == binding.get('source_commit') and
         receipt.get('source_job') == binding.get('source_job'), 'RESTORE_RECEIPT_BINDING_DIFFERS')
    progress = receipt['attempt_progress']; writes = receipt['actual_optimizer_writes_per_rank']
    need(progress.get('completed_outer') == outer and progress.get('cumulative_writes') == writes and
         type(writes) is int and outer <= writes <= outer*20 and
         type(progress.get('attempt_index')) is int and progress['attempt_index'] >= outer and
         binding.get('actual_optimizer_writes_per_rank') == writes,
         'RESTORE_PROGRESS_DIFFERS')
    old_raw = get('source-run-manifest.json', source_base/'logs/run-manifest.json',
                  record=receipt['source_run_manifest'])
    old_run = json.loads(old_raw)
    need(old_run.get('run_id') == run['run_id'] and old_run.get('source_commit') == receipt['source_commit'] and
         old_run.get('easy2560_selection') is True and old_run.get('training_world_size') == 4 and
         old_run.get('all_correct_virtual_zero') is False and old_run.get('question_probe') is None,
         'RESTORE_SOURCE_RUN_BINDING_DIFFERS')
    parent = authenticated_restore_chain(source_base, old_run, bundle=bundle, sealed_directory=sealed,
        namespace=namespace+'parent/', _seen=seen)
    start = parent['boundary'] if parent else 0
    prior_attempt = parent['cutoff_attempt'] if parent else 0
    need(start < outer, 'RESTORE_LINEAGE_MUST_ADVANCE')
    history = receipt.get('historical_files', {})
    if not history:
        history = {r['relative_path']:r for r in receipt['historical_artifacts']['prefix_files']}
    cached = dict(parent['historical']) if parent else {}
    relatives = ['logs/actual-updates.jsonl', 'logs/actual-attempts.jsonl', 'logs/validation-identities.jsonl']
    relatives += ['logs/%s-rank-%d.jsonl' % (role, rank) for role in ('scheduler','optimizer') for rank in range(4)]
    for rel in relatives:
        record = history.get(rel)
        if record is None and rel == 'logs/validation-identities.jsonl':
            continue  # No new fixed QA is scheduled between milestones.
        need(record is not None and record['path'] == str(source_base/rel), 'SOURCE_SEGMENT_PREFIX_REQUIRED')
        value = get('historical/'+rel, source_base/rel, record=record)
        cached[rel] = (cached.get(rel, b'') + value)
    updates = _historical_rows(dict(historical=cached), 'logs/actual-updates.jsonl')
    attempts = _historical_rows(dict(historical=cached), 'logs/actual-attempts.jsonl')
    need(updates == receipt.get('authenticated_updates') and attempts == receipt.get('authenticated_attempts') and
         [r['step'] for r in updates] == list(range(1, outer+1)) and
         [attempt_identity(r)[0] for r in attempts] == list(range(1, progress['attempt_index']+1)) and
         len({r['attempt_id'] for r in attempts}) == len(attempts) and
         all(r['run_id'] == run['run_id'] for r in updates+attempts), 'AUTHENTICATED_FLATTENED_HISTORY_DIFFERS')
    current_attempts = [r for r in attempts if r['attempt_index'] > prior_attempt]
    need(current_attempts and all(r['slurm_job_id'] == receipt['source_job'] for r in current_attempts) and
         attempts[-1]['completed_outer'] == outer and attempts[-1]['optimizer_writes_after'] == writes,
         'SOURCE_SEGMENT_JOB_OR_COUNTERS_DIFFER')
    if 'logs/formal-inline-development-after.json' in history:
        record = history['logs/formal-inline-development-after.json']
        qa = get('historical/logs/formal-inline-development-after.json', record['path'], record=record)
        cached['logs/formal-inline-development-after.json'] = qa
    need(outer < 5 or 'logs/formal-inline-development-after.json' in cached, 'REAL_HISTORICAL_QA5_REQUIRED')
    # Source publication belongs to this execution, not to its ancestor or HEAD.
    publication = runtime = None
    if old_run['source_commit'] != REVIEWED_COMMIT:
        rel = 'source/'+old_run['source_commit']+'/source-publication.json'
        publication = json.loads(get('historical/'+rel, source_base/rel, record=history.get(rel)))
        rel = 'preparation/run-source.json'
        runtime = json.loads(get('historical/'+rel, source_base/rel, record=history.get(rel)))
    publication_binding = published_source_binding(source_base, old_run, old_run['source_commit'], publication, runtime)
    allocation = json.loads(get('current-allocation.json', base/'logs/native-allocation-admission.json'))
    old_allocation = json.loads(get('historical/logs/native-allocation-admission.json',
        source_base/'logs/native-allocation-admission.json', record=history['logs/native-allocation-admission.json']))
    job = allocation.get('slurm_job_id')
    need(allocation.get('accepted') is True and allocation.get('run_id') == run['run_id'] and
         old_allocation.get('accepted') is True and old_allocation.get('run_id') == run['run_id'] and
         old_allocation.get('slurm_job_id') == receipt['source_job'] and job != receipt['source_job'],
         'RESTORE_ALLOCATION_CHAIN_DIFFERS')
    workers_raw = get('full-state-restored-workers.json', base/'logs/full-state-restored-workers.json')
    driver_raw = get('full-state-restored-driver.json', base/'logs/full-state-restored-driver.json')
    workers, driver = json.loads(workers_raw), json.loads(driver_raw)
    need(workers.get('run_id') == run['run_id'] and workers.get('source_commit') == run['source_commit'] and
         workers.get('slurm_job_id') == job and workers.get('source_receipt_sha256') == binding['source_receipt_sha256'] and
         workers.get('actual_world_size') == 4, 'RESTORED_WORKERS_AGGREGATE_DIFFERS')
    ranks = workers.get('ranks', []); uuids = allocation.get('actual_cuda_runtime_uuids', [])
    need(len(ranks) == 4 and sorted(r['rank'] for r in ranks) == list(range(4)) and
         len(set(uuids)) == 4 and {r.get('actual_cuda_uuid') for r in ranks} == set(uuids), 'RESTORED_NATIVE4_UUIDS_DIFFERS')
    files = {r['path']:r for r in receipt['files']}
    for row in ranks:
        state = files.get(str(checkpoint/('training_state/rank-%d.pt' % row['rank'])))
        need(state is not None and row.get('source_state_full_sha256') == state['full_read_sha256'] and
             row.get('run_id') == row.get('source_run_id') == run['run_id'] and
             row.get('source_receipt_sha256') == binding['source_receipt_sha256'] and
             row.get('source_job') == receipt['source_job'] and row.get('source_commit') == old_run['source_commit'] and
             row.get('source_step') == outer and row.get('source_checkpoint') == str(checkpoint) and
             row.get('slurm_job_id') == job and row.get('world_size') == 4 and
             row.get('actual_optimizer_writes') == writes and row.get('scheduler_last_epoch') == outer and
             row.get('last_attempt_index') == progress['attempt_index'] and
             row.get('actual_model_collective_readback_equal') is True and
             row.get('actual_adam_scheduler_readback_equal') is True and row.get('actual_rng_readback_equal') is True,
             'RESTORED_RANK_READBACK_DIFFERS')
        birth = row.get('process_identity', {})
        need(type(birth.get('pid')) is int and birth['pid'] > 0 and
             type(birth.get('start_ticks')) is int and birth['start_ticks'] > 0, 'RESTORED_RANK_BIRTH_REQUIRED')
    expected_tensors = receipt.get('hf_saved_tensor_count')
    need(type(expected_tensors) is int and expected_tensors > 0 and
         len({(r['process_identity']['pid'], r['process_identity']['start_ticks']) for r in ranks}) == 4 and
         next(r for r in ranks if r['rank'] == 0).get('model_hf_exact_tensors_checked_on_rank0') == expected_tensors,
         'RESTORED_RANK0_FULL_MODEL_OR_DISTINCT_BIRTHS_REQUIRED')
    post = receipt.get('post_callback_rng_captured') is True
    need(driver.get('run_id') == run['run_id'] and driver.get('source_receipt_sha256') == binding['source_receipt_sha256'] and
         driver.get('source_job') == receipt['source_job'] and driver.get('source_commit') == old_run['source_commit'] and
         driver.get('source_step') == outer and driver.get('slurm_job_id') == job and
         driver.get('source_driver_full_sha256') == receipt['driver_file']['full_read_sha256'] and
         driver.get('actual_candidate_source_cursor') == receipt['driver_state']['source_consumed'] and
         driver.get('actual_probe_version') == receipt['reward_probe_version'] and
         driver.get('actual_probe_round') == receipt['reward_probe_round'] and
         driver.get('attempt_progress') == progress and driver.get('actual_driver_rng_readback_equal') is True and
         driver.get('actual_candidate_stream_readback_equal') is True and
         driver.get('post_callback_rng_captured') is post and driver.get('same_job_continuation') is False,
         'RESTORED_DRIVER_READBACK_DIFFERS')
    segments = list(parent['segments']) if parent else []
    segments.append(dict(first_outer=start+1, last_outer=outer, first_attempt=prior_attempt+1,
        last_attempt=progress['attempt_index'], base=source_base, run=old_run, job=receipt['source_job']))
    public_segments = [dict(first_outer=s['first_outer'], last_outer=s['last_outer'],
        first_attempt=s['first_attempt'], last_attempt=s['last_attempt'], base=str(s['base']),
        run_id=s['run']['run_id'], source_commit=s['run']['source_commit'], source_job=s['job']) for s in segments]
    public = dict(source_base=str(source_base), source_checkpoint=str(checkpoint), completed_outer=outer,
        source_commit=old_run['source_commit'], source_job=receipt['source_job'], current_job=job,
        current_source_commit=run['source_commit'], source_receipt_sha256=binding['source_receipt_sha256'],
        historical_attempt_index=progress['attempt_index'], historical_optimizer_writes=writes,
        source_segments=public_segments, parent_authenticated_history=parent['public'] if parent else None,
        source_publication_binding=publication_binding, single_original_job_prefix=False, inherited_history_assembled=True,
        native4_worker_driver_restore_proofs_authenticated=True,
        worker_readback_receipt_sha256=digest(workers_raw), driver_readback_receipt_sha256=digest(driver_raw),
        post_callback_rng_captured=post, actual_restore_performed_by_this_cpu_reader=False)
    return dict(public=public, source_base=source_base, source_run=old_run, historical=cached,
        records=history, progress=progress, receipt=receipt, boundary=outer,
        cutoff_attempt=progress['attempt_index'], source_job=receipt['source_job'], current_job=job, segments=segments,
        parent_chain=parent)


def historical_origin(chain, *, attempt_index=None, outer=None):
    """Resolve the original execution; never substitute a failed suffix."""
    need(chain is not None and (attempt_index is None) != (outer is None), 'ORIGIN_KEY_REQUIRED')
    for item in chain.get('segments', []):
        lower, upper = ((item['first_attempt'], item['last_attempt']) if attempt_index is not None else
                        (item['first_outer'], item['last_outer']))
        value = attempt_index if attempt_index is not None else outer
        if lower <= value <= upper:
            return item['base'], item['run'], item['job']
    # Legacy external fixtures may provide only the old two-execution shape.
    if not chain.get('segments') and ((attempt_index is not None and attempt_index <= chain['cutoff_attempt']) or
                                    (outer is not None and outer <= chain['boundary'])):
        return chain['source_base'], chain['source_run'], chain['source_job']
    raise ValueError('EASY2560_EVIDENCE_HISTORICAL_ORIGIN_NOT_FOUND')

def _historical_rows(chain, relative):
    raw = chain['historical'][relative]
    need(raw.endswith(b'\n'), 'PARTIAL_AUTHENTICATED_HISTORICAL_PREFIX')
    chunks = raw.splitlines(keepends=True)
    need(all(len(line) <= MAX_LINE and line.strip() for line in chunks), 'BAD_HISTORICAL_LEDGER_LINE')
    return [json.loads(line) for line in chunks]


def _attempt_origin(base, run, chain, attempt_index):
    if chain is not None and attempt_index <= chain['cutoff_attempt']:
        return historical_origin(chain, attempt_index=attempt_index)
    return base, run, chain['current_job'] if chain else None


def _validate_ledgers(base, run, step, driver, bundle, job, chain=None):
    progress = driver['attempt_progress']; cutoff = progress['attempt_index']
    def ledger(relative, terminal, output):
        rows, raw, record = prefix(base/relative, terminal)
        if chain is not None:
            need(all(attempt_identity(row)[0] > chain['cutoff_attempt'] for row in rows),
                 'CURRENT_LOG_REPLAYS_HISTORICAL_ATTEMPT')
            bundle.put('segments/current/'+relative, raw, record)
            return _historical_rows(chain, relative) + rows
        bundle.put(output, raw, record)
        return rows
    updates = ledger('logs/actual-updates.jsonl', lambda r: r.get('step') == step,
                     'prefix/actual-updates.jsonl')
    need([r['step'] for r in updates] == list(range(1, step + 1)), 'UPDATE_PREFIX_NOT_FRESH_CONTIGUOUS')
    need(all(r['run_id'] == run['run_id'] for r in updates), 'FOREIGN_UPDATE_RUN')
    need(updates[-1]['probe_version'] == driver['probe_version'], 'SAVED_PROBE_AND_UPDATE_DIFFER')
    attempts = ledger('logs/actual-attempts.jsonl',
                      lambda r: r.get('attempt_index') == cutoff, 'prefix/actual-attempts.jsonl')
    need([r['attempt_index'] for r in attempts] == list(range(1, cutoff + 1)), 'ATTEMPT_PREFIX_NOT_FRESH_CONTIGUOUS')
    need(len({r['attempt_id'] for r in attempts}) == cutoff, 'ATTEMPT_UUID_REUSED')
    writes = completed = aborted = full = 0
    successful = []
    for r in attempts:
        expected_job = historical_origin(chain, attempt_index=r['attempt_index'])[2] if chain and r['attempt_index'] <= chain['cutoff_attempt'] else job
        need(r['run_id'] == run['run_id'] and r['slurm_job_id'] == expected_job and
             r['all_ranks_agreed'] is True and r['real_rank_count'] == 4,
             'CROSS_JOB_OR_FOREIGN_ATTEMPT')
        need(r['logical_update'] == completed + 1 and r['optimizer_writes_before'] == writes,
             'ATTEMPT_COUNTER_CHAIN_DIFFERS')
        count = r['optimizer_writes_this_attempt']; processed = r['opportunities_processed']
        need(type(count) is int and 0 <= count <= processed <= 20 and
             r['optimizer_writes_after'] == writes + count and r['complete_outer'] is (count > 0),
             'ACTUAL_WRITES_OR_COMPLETION_DIFFERS')
        writes += count; completed += int(count > 0); aborted += int(count == 0)
        full += int(count == 20 and processed == 20)
        need(r['completed_outer'] == completed and r['cumulative_aborted_attempts'] == aborted,
             'ATTEMPT_PROGRESS_DIFFERS')
        if count: successful.append(r)
    need(completed == step and writes == progress['cumulative_writes'] and
         aborted == progress['aborted_attempts'] and full == progress['full_budget_outer'],
         'DRIVER_AND_LEDGER_PROGRESS_DIFFER')
    need(len(successful) == len(updates), 'UPDATE_ATTEMPT_LENGTH_DIFFERS')
    for u, a in zip(updates, successful):
        need((u['step'], u['attempt_id'], u['attempt_index'], u['optimizer_writes_after']) ==
             (a['completed_outer'], a['attempt_id'], a['attempt_index'], a['optimizer_writes_after']),
             'UPDATE_ATTEMPT_BINDING_DIFFERS')
    if chain is not None:
        history = [r for r in attempts if r['attempt_index'] <= chain['cutoff_attempt']]
        need(history[-1]['completed_outer'] == chain['boundary'] and
             history[-1]['optimizer_writes_after'] == chain['progress']['cumulative_writes'] and
             len([u for u in updates if u['step'] <= chain['boundary']]) == chain['boundary'], 'AUTHENTICATED_PREFIX_BOUNDARY_DIFFERS')
    by_id = {r['attempt_id']: r for r in attempts}
    for rank in range(4):
        for role, terminal in (('scheduler', ('COMPLETE', 'ABORTED_ATTEMPT')),
                               ('optimizer', ('OUTER_BUDGET', 'ABORTED_ATTEMPT'))):
            relative = f'logs/{role}-rank-{rank}.jsonl'
            rows = ledger(relative, lambda r: attempt_identity(r)[0] == cutoff and r['kind'] in terminal,
                          f'prefix/{role}-rank-{rank}.jsonl')
            seen = []; groups = {}
            for row in rows:
                index, aid = attempt_identity(row)
                need(aid in by_id and index == by_id[aid]['attempt_index'] and
                     row['run_id'] == run['run_id'] and row['rank'] == rank,
                     'RANK_PREFIX_IDENTITY_DIFFERS')
                if aid not in groups: groups[aid] = []; seen.append(aid)
                groups[aid].append(row)
                if role == 'scheduler':
                    expected_job = historical_origin(chain, attempt_index=index)[2] if chain and index <= chain['cutoff_attempt'] else job
                    need(row['slurm_job_id'] == expected_job and row['world_size'] == 4,
                         'RANK_PREFIX_CROSS_JOB')
                if role == 'optimizer' and row['kind'] == 'POLICY_MONITOR':
                    need(row['objective_weight_definition'] ==
                         'equal four-row logical micro masked_mean; weighted global numerator/denominator',
                         'LOGICAL_MICRO_LABEL_DIFFERS')
                    need(len(row['monitor_identity']['selection_indices']) == 8,
                         'MONITOR_COUNT_DIFFERS')
            need(seen == [r['attempt_id'] for r in attempts], 'MISSING_OR_REORDERED_RANK_ATTEMPT')
            for aid, events in groups.items():
                attempt = by_id[aid]; terminal_row = events[-1]
                expected = ('COMPLETE' if attempt['complete_outer'] else 'ABORTED_ATTEMPT') if role == 'scheduler' else (
                            'OUTER_BUDGET' if attempt['complete_outer'] else 'ABORTED_ATTEMPT')
                need(terminal_row['kind'] == expected, 'INCOMPLETE_RANK_ATTEMPT')
                if role == 'scheduler':
                    need([r['kind'] for r in events] == (['BEGIN', 'COMMIT_PREPARED', 'COMPLETE']
                         if attempt['complete_outer'] else ['BEGIN', 'ABORTED_ATTEMPT']), 'SCHEDULER_PREFIX_SHAPE')
                    need(terminal_row['writes_after'] == attempt['optimizer_writes_after'], 'SCHEDULER_WRITES_DIFFER')
                else:
                    need(terminal_row['cumulative_writes'] == attempt['optimizer_writes_after'], 'WORKER_WRITES_DIFFER')
                    need(sum(r['kind'] == 'OPTIMIZER_WRITE' for r in events) ==
                         attempt['optimizer_writes_this_attempt'], 'WORKER_WRITE_RECEIPTS_DIFFER')
    return updates, attempts


def _trace_references(base, run, updates, attempts, bundle, chain=None):
    references = []
    for attempt in attempts:
        aid = attempt['attempt_id']; name = f'logs/behavior-probability-attempt-{aid}.json'
        origin, origin_run, _ = _attempt_origin(base, run, chain, attempt['attempt_index'])
        gate = bundle.copy('gates/' + Path(name).name, origin/name)
        need(gate['run_id'] == run['run_id'] and gate['attempt_id'] == aid and
             gate['attempt_index'] == attempt['attempt_index'] and gate['logical_update'] == attempt['logical_update']
             and gate['accepted'] is True and gate['slurm_job_id'] == attempt['slurm_job_id']
             and gate['source_sha256'] == origin_run['source_sha256'], 'GATE_ATTEMPT_DIFFERS')
        trace = regular(origin/'artifacts/behavior-traces'/f'attempt-{aid}.json')
        h = hashlib.sha256()
        with trace.open('rb') as stream:
            before = os.fstat(stream.fileno())
            for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
            after = os.fstat(stream.fileno())
        need(identity(before) == identity(after) == identity(trace.stat()) and
             h.hexdigest() == gate['private_trace_sha256'] == gate['trace_sha256'], 'SELECTED_TRACE_SHA_DIFFERS')
        references.append(dict(path=str(trace), bytes=after.st_size, sha256=h.hexdigest(),
                               copied=False, requires_remote_retention=True, attempt_id=aid,
                               source_base=str(origin), source_commit=origin_run['source_commit'],
                               source_job=attempt['slurm_job_id']))
    for update in updates:
        gate_path = f'gates/behavior-probability-attempt-{update["attempt_id"]}.json'
        need(bundle.files[gate_path]['sha256'] == update['behavior_probability_gate_sha256'], 'UPDATE_GATE_SHA_DIFFERS')
        origin, origin_run, _ = _attempt_origin(base, run, chain, update['attempt_index'])
        selection = bundle.copy('selection/' + update['attempt_id'] + '.json',
                    origin/'artifacts/mixed-selection'/(update['attempt_id'] + '.json'))
        need(selection['step'] == update['step'] and selection['source_commit'] == origin_run['source_commit'] and
             selection['policy_snapshot_id'] == run['run_id']+':'+str(update['step'])+':policy-'+update['attempt_id']
             and selection['fixed_candidate_budget'] is True and selection['attempted_candidates'] == 2560
             and selection['selected_groups'] == 512, 'SELECTION_CP_BINDING_DIFFERS')
    bundle.put('selected-large-references.json', encoded(references))
    return references


def _validation_observation(base, step, bundle, chain=None):
    path = regular(base/'logs/validation-identities.jsonl')
    # Bounded current observation: complete records only; later rows are not a
    # continuation RNG boundary and do not change the immutable CP phase.
    with path.open('rb') as stream:
        bound = os.fstat(stream.fileno()).st_size
        raw = stream.read(bound)
    rows = []; chunks = []
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'): break
        value = json.loads(line)
        if value['step'] > step: break
        rows.append(value); chunks.append(line)
    sealed = b''.join(chunks)
    with path.open('rb') as stream:
        need(stream.read(len(sealed)) == sealed, 'VALIDATION_PREFIX_CHANGED')
    bundle.put('segments/current/logs/validation-identities.jsonl' if chain else
               'prefix/validation-identities.jsonl', sealed,
               dict(path=str(path), bytes=len(sealed), sha256=digest(sealed), prefix_only=True))
    if chain is not None:
        need(all(row['step'] > chain['boundary'] for row in rows), 'CURRENT_VALIDATION_REPLAYS_HISTORICAL_BOUNDARY')
        # CP1 may precede every formal validation when the original initial
        # evaluation was explicitly skipped. Only that pre5 absence is empty;
        # later boundaries must retain their authenticated validation prefix.
        relative = 'logs/validation-identities.jsonl'
        if not (chain['boundary'] < 5 and relative not in chain['historical']):
            rows = _historical_rows(chain, relative) + rows
    fixed = [r for r in rows if r['step'] == step and r['split'] == 'validation']
    need(step == 5 or len(fixed) == 1, 'PRE_SAVE_FIXED512_EVIDENCE_MISSING_OR_DUPLICATED')
    return dict(fixed512_seen_at_observation=bool(fixed),
                final51713_seen_at_observation=any(r['step'] == step and r['split'] == 'final' for r in rows))


def published_source_binding(base, run, requested, publication=None, runtime=None, snapshot=None):
    """Bind a caller-selected commit; never infer an allowed source from HEAD."""
    need(isinstance(requested, str) and re.fullmatch('[0-9a-f]{40}', requested) and
         run.get('source_commit') == requested, 'EXPLICIT_TRAINING_COMMIT_DIFFERS')
    if requested == REVIEWED_COMMIT:
        return dict(policy='fixed_reviewed_commit', requested_training_commit=requested)
    need(isinstance(publication, dict) and publication.get('source_commit') == requested and
         publication.get('fresh_source_all_files_sha_verified') is True and
         isinstance(publication.get('files'), dict) and publication['files'],
         'EXPLICIT_PUBLISHED_SOURCE_REQUIRED')
    need(isinstance(runtime, dict) and runtime.get('source_commit') == requested and
         runtime.get('files') == publication['files'] and runtime.get('runtime_repo') == str(Path(base)/'repo'),
         'PUBLISHED_RUNTIME_SOURCE_DIFFERS')
    source_map = run.get('source_sha256')
    need(isinstance(source_map, dict) and source_map, 'BOUND_RUNTIME_SOURCE_MAP_REQUIRED')
    for name, sha in source_map.items():
        entry = publication['files'].get(name)
        need(isinstance(name, str) and name not in ('', '.') and not Path(name).is_absolute() and
             '..' not in Path(name).parts and isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha) and
             isinstance(entry, dict) and entry.get('sha256') == sha and
             type(entry.get('bytes')) is int and entry['bytes'] >= 0,
             'PUBLISHED_RUNTIME_SOURCE_SHA_DIFFERS')
        if snapshot is not None:
            archived = snapshot['files'].get('source/' + name, {})
            need(archived.get('sha256') == sha and
                 archived.get('bytes') == publication['files'][name].get('bytes'),
                 'PUBLISHED_ARCHIVED_SOURCE_DIFFERS')
    return dict(policy='explicit_caller_commit_publication_runtime_run_map',
                requested_training_commit=requested, source_map_sha256=digest(canonical(source_map)),
                automatic_head_or_ancestor_allow=False, git_ancestry_verified_here=False)


def verify(directory):
    directory = Path(directory).resolve()
    marker, _, marker_record = json_small(directory/'complete.json')
    manifest, _, _ = json_small(directory/'supplement.json', expected=marker['supplement_sha256'])
    need(marker['status'] == STATUS and marker['actual_cuda_restore_accepted'] is False and
         marker['post_callback_rng_captured'] is False and
         manifest['actual_cuda_restore_accepted'] is False and
         manifest['role'] == 'easy2560_checkpoint_evidence_supplement' and
         manifest['post_callback_rng_captured'] is False and
         ((manifest['single_original_job_prefix'] is True and not manifest['inherited_history_assembled']) or
          (manifest['single_original_job_prefix'] is False and manifest['inherited_history_assembled'] is True)),
         'UNSUPPORTED_ACCEPTANCE_CLAIM')
    need(all(marker[k] == manifest[k] for k in ('source_commit', 'run_id', 'source_job', 'step')),
         'SEALED_IDENTITY_DIFFERS')
    for name, record in manifest['files'].items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts and
             (directory/name).resolve().is_relative_to(directory), 'SEALED_FILE_ESCAPE')
        _, actual = read_small(directory/name, expected=record['sha256'])
        need(actual['bytes'] == record['bytes'], 'SEALED_FILE_SIZE_DIFFERS')
    run_path = directory/'identity/run-manifest.json'
    if run_path.is_file():
        sealed_run, _, _ = json_small(run_path)
        need(bool(sealed_run.get('full_state_restore')) == manifest['inherited_history_assembled'] and
             bool(sealed_run.get('full_state_restore')) is (not manifest['single_original_job_prefix']),
             'SEALED_RUN_HISTORY_LABEL_DIFFERS')
    core = manifest['core_archive']
    archive, _, _ = json_small(directory/core['source_acceptance']['relative_path'],
                              expected=core['source_acceptance']['sha256'])
    snapshot, _, _ = json_small(directory/core['source_snapshot']['relative_path'],
                               expected=core['source_snapshot']['sha256'])
    need(archive['status'] == ARCHIVE_STATUS == core['status'] and
         archive['snapshot_sha256'] == core['source_snapshot']['sha256'] and
         all(archive[k] == snapshot[k] for k in ('base', 'run_id', 'source_commit', 'step', 'total_bytes')) and
         archive['base'] == manifest['source_base'] and archive['run_id'] == manifest['run_id'] and
         archive['source_commit'] == manifest['source_commit'] and archive['step'] == manifest['step'] and
         archive['actual_cuda_restore_accepted'] is False, 'SEALED_CORE_ARCHIVE_IDENTITY_DIFFERS')
    if manifest['source_commit'] != REVIEWED_COMMIT:
        run, _, _ = json_small(directory/'identity/run-manifest.json')
        publication, _, _ = json_small(directory/'identity/source-publication.json')
        runtime, _, _ = json_small(directory/'identity/run-source.json')
        need(manifest.get('source_publication_binding') == published_source_binding(
            manifest['source_base'], run, manifest['source_commit'], publication, runtime, snapshot),
            'SEALED_SOURCE_PUBLICATION_BINDING_DIFFERS')
    if manifest['inherited_history_assembled']:
        run, _, _ = json_small(directory/'identity/run-manifest.json')
        chain = authenticated_restore_chain(manifest['source_base'], run, sealed_directory=directory)
        need(chain is not None and chain['public'] == manifest.get('authenticated_history'),
             'SEALED_AUTHENTICATED_HISTORY_DIFFERS')
    else:
        need(manifest.get('authenticated_history') is None, 'UNDECLARED_AUTHENTICATED_HISTORY')
    files = dict(manifest['files'])
    for name in ('supplement.json', 'complete.json'):
        raw, record = read_small(directory/name); files[name] = dict(bytes=len(raw), sha256=record['sha256'])
    return dict(marker, source_base=manifest['source_base'], source_checkpoint=manifest['source_checkpoint'],
                core_archive=manifest['core_archive'], single_original_job_prefix=manifest['single_original_job_prefix'],
                inherited_history_assembled=manifest['inherited_history_assembled'],
                authenticated_history=manifest.get('authenticated_history'), selected_trace_bytes_copied=0,
                checkpoint_large_files_rehashed_by_this_operator=False,
                source_publication_binding=manifest.get('source_publication_binding', dict(
                    policy='fixed_reviewed_commit', requested_training_commit=REVIEWED_COMMIT)),
                scope=manifest['scope'], directory=str(directory), files_to_archive=[dict(
        relative_path=name, path=str(directory/name), **record) for name, record in sorted(files.items())])


def recheck_sources(records):
    """Copied prefixes remain stable across harmless later append operations."""
    for source in records:
        path = regular(source['path'])
        with path.open('rb') as f:
            if not source.get('prefix_only', False):
                need(os.fstat(f.fileno()).st_size == source['bytes'], 'IMMUTABLE_SOURCE_SIZE_CHANGED')
            raw = f.read(source['bytes'])
        need(len(raw) == source['bytes'] and digest(raw) == source['sha256'], 'SOURCE_AFTER_COPY_SHA_DIFFERS')


def seal(base, training_commit, run_id, source_job, step):
    base = owned_base(base)
    need(isinstance(training_commit, str) and re.fullmatch('[0-9a-f]{40}', training_commit) and
         step in STEPS, 'FULL_COMMIT_AND_FIXED_CP_REQUIRED')
    root = base/'preparation/checkpoint-evidence'; root.mkdir(parents=True, exist_ok=True)
    need(root.resolve() == root, 'EVIDENCE_ROOT_SYMLINK')
    fsync_directory(root.parent)
    with (root/'seal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = root/f'global_step_{step}'
        if target.exists():
            result = verify(target)
            need((result['source_commit'], result['run_id'], result['source_job'], result['step']) ==
                 (training_commit, run_id, source_job, step), 'EXISTING_SIDECAR_BINDING_DIFFERS')
            manifest, _, _ = json_small(target/'supplement.json')
            recheck_sources(manifest['source_reads'])
            return result
        stage = root/('.partial-' + uuid.uuid4().hex); stage.mkdir()
        try:
            bundle = Bundle(stage)
            run = bundle.copy('identity/run-manifest.json', base/'logs/run-manifest.json')
            need(run['run_id'] == run_id and run['source_commit'] == training_commit and
                 run.get('easy2560_selection') is True and run.get('profile') == 'common8' and
                 run.get('execution_site') == 'hpc3' and run.get('execution_backend', 'slurm') == 'slurm' and
                 run['training_world_size'] == 4 and
                 run['data_selection_mode'] == 'mixed_first' and run.get('question_probe') is None and
                 run['actor_logical_micro_batch_per_rank'] == 4 and
                 type(run['actor_micro_batch_per_rank']) is int and run['actor_micro_batch_per_rank'] in (4, 8, 16, 32) and
                 run.get('selection_candidate_questions') == 2560 and run.get('selection_selected_questions') == 512 and
                 run.get('selection_fixed_candidate_budget') is True and run['all_correct_virtual_zero'] is False,
                 'EXACT_EASY_SCOPE_DIFFERS')
            chain = authenticated_restore_chain(base, run, bundle=bundle)
            need(chain is None or step > chain['boundary'], 'RESTORED_CP_MUST_BE_NEW_SUFFIX_BOUNDARY')
            allocation = bundle.copy('identity/native-allocation-admission.json', base/'logs/native-allocation-admission.json')
            need(allocation['accepted'] is True and allocation['run_id'] == run_id and
                 allocation['slurm_job_id'] == source_job, 'SOURCE_JOB_DIFFERS')
            cp = base/'checkpoints/search-g1-common8/actor'/f'global_step_{step}'
            complete = bundle.copy('identity/checkpoint-complete.json', cp/'complete.json')
            need(complete['run_id'] == run_id and complete['step'] == step and
                 complete['finite_all_saved_tensors'] is True, 'CHECKPOINT_NOT_COMPLETE')
            archive_root = base/'preparation/checkpoint-archives'/f'global_step_{step}'
            snap = bundle.copy('identity/source-snapshot.json', archive_root/'source-snapshot.json')
            archive = bundle.copy('identity/accepted-archive.json', archive_root/'complete.json')
            need(all(snap[k] == v for k, v in dict(base=str(base), run_id=run_id, source_commit=training_commit,
                 step=step, training_world_size=4).items()) and
                 snap['role'] == 'easy2560_full_checkpoint_source_snapshot', 'ARCHIVE_CP_BINDING_DIFFERS')
            need(archive['status'] == ARCHIVE_STATUS and archive['actual_cuda_restore_accepted'] is False and
                 archive['fresh_source_after_archive_sha_verified'] is True and
                 archive['fresh_destination_sha_verified'] is True and archive['destination_fsynced'] is True and
                 archive['file_count'] == len(snap['files']) and
                 archive['snapshot_sha256'] == bundle.files['identity/source-snapshot.json']['sha256'] and
                 all(archive[k] == snap[k] for k in ('run_id', 'source_commit', 'step', 'base', 'total_bytes')),
                 'ARCHIVE_ACCEPTANCE_BINDING_DIFFERS')
            need(snap['files']['checkpoint/complete.json']['sha256'] ==
                 bundle.files['identity/checkpoint-complete.json']['sha256'] and
                 snap['files']['metadata/run-manifest.json']['sha256'] ==
                 bundle.files['identity/run-manifest.json']['sha256'], 'ARCHIVED_METADATA_DIFFERS')
            publication = runtime = None
            if training_commit != REVIEWED_COMMIT:
                publication = bundle.copy('identity/source-publication.json',
                    base/'source'/training_commit/'source-publication.json')
                runtime = bundle.copy('identity/run-source.json', base/'preparation/run-source.json')
            source_binding = published_source_binding(base, run, training_commit, publication, runtime, snap)
            state = bundle.copy('identity/training-state-manifest.json', cp/'training_state/manifest.json',
                                expected=complete['files_sha256']['training_state/manifest.json'])
            need(state['completed_outer'] == step and state['question_probe'] is None and
                 sorted(r['rank'] for r in state['worker_files']) == list(range(4)), 'NATIVE4_STATE_REQUIRED')
            from reproduction.prefix20_joint4.cpu_audit.safe_pt import Archive
            driver_raw, driver_record = read_small(cp/'training_state/driver.pt',
                          expected=complete['files_sha256']['training_state/driver.pt'])
            need(driver_record['sha256'] == snap['files']['checkpoint/training_state/driver.pt']['sha256'], 'ARCHIVED_DRIVER_DIFFERS')
            with Archive(driver_raw) as inert:
                driver = inert.value
                need(driver['schema_version'] == 1 and driver['completed_outer'] == step and driver['run_id'] == run_id and
                     driver['source_commit'] == training_commit and driver['question_probe'] is None and
                     driver['selection_mode'] == 'mixed_first' and driver['easy2560_selection'] is True,
                     'DRIVER_CP_BINDING_DIFFERS')
                need(driver['reward_refresh_pending'] is (step in (50, 100, 150)), 'SAVED_CALLBACK_FLAG_DIFFERS')
                stream = driver['candidate_stream']
                summary = {k: driver[k] for k in ('completed_outer', 'probe_version', 'probe_round',
                            'reward_refresh_pending', 'attempt_progress')}
                summary.update(driver_file=driver_record, stream={k: stream[k] for k in
                    ('candidate_cursor', 'epoch', 'consumed_source_rows', 'pending_offset', 'prefix20_pool_binding')},
                    candidate_order_count=len(stream['candidate_order']),
                    candidate_order_sha256=digest(canonical(stream['candidate_order'])),
                    candidate_stream_storage_signature=digest(canonical(inert.signature(stream))),
                    rng_storage_signature=digest(canonical(inert.signature(driver['rng']))),
                    tensor_or_model_load_performed=False, actual_restore_performed=False)
            progress = driver['attempt_progress']
            need(progress['completed_outer'] == step and complete['actual_optimizer_writes_per_rank'] ==
                 progress['cumulative_writes'] and all(r['run_id'] == run_id and
                 r['actual_optimizer_writes'] == progress['cumulative_writes'] for r in state['worker_files']),
                 'SAVED_COUNTERS_DIFFER')
            for rank in range(4):
                key = f'training_state/rank-{rank}.pt'
                need(complete['files_sha256'][key] == snap['files']['checkpoint/' + key]['sha256'], 'ARCHIVED_RANK_SHA_DIFFERS')
            bundle.put('saved-driver-summary.json', encoded(summary))
            resolved = bundle.copy('identity/resolved-training-config.json', base/'logs/resolved-training-config.json')
            need(resolved['run_id'] == run_id and digest(json.dumps(resolved['resolved_config'], sort_keys=True, separators=(',', ':')).encode()) ==
                 resolved['resolved_config_sha256'], 'RESOLVED_CONFIG_IDENTITY_DIFFERS')
            cfg = resolved['resolved_config']
            need(cfg['data']['selection_mode'] == 'mixed_first' and cfg['algorithm']['all_correct_virtual_zero'] is False
                 and cfg['actor_rollout_ref']['actor']['ppo_logical_micro_batch_size'] == 4, 'RESOLVED_EASY_MATH_DIFFERS')
            bundle.put('identity/source-data-identities.json', encoded(dict(
                source_sha256=run['source_sha256'], input_sha256=run['input_sha256'],
                source_and_large_data_bytes_in_existing_archive=True, large_file_SHA_recomputed_by_this_sidecar=False)))
            for name, expected in run['input_sha256'].items():
                need(snap['files']['data/' + name]['sha256'] == expected, 'ARCHIVED_DATA_IDENTITY_DIFFERS')
            for name, expected in run['source_sha256'].items():
                need(snap['files']['source/' + name]['sha256'] == expected, 'ARCHIVED_SOURCE_IDENTITY_DIFFERS')
            bundle.copy('identity/eligible-question-pool.json', Path(run['selection_eligible_pool_path']),
                        expected=stream['prefix20_pool_binding']['eligible_pool_sha256'], external=True)
            manifests = [r for name, r in snap['files'].items() if name.startswith('grounding/') and name.endswith('/version-manifest.json')]
            pinned = []
            for record in manifests:
                value, _, _ = json_small(record['path'], expected=record['sha256'])
                if value['version'] == driver['probe_version']: pinned.append((value, record))
            need(len(pinned) == 1, 'EXACT_SAVED_GROUNDING_VERSION_MISSING_OR_AMBIGUOUS')
            probe, record = pinned[0]
            need(probe['run_id'] == run_id and probe['round'] == driver['probe_round'], 'SAVED_GROUNDING_IDENTITY_DIFFERS')
            bundle.copy('grounding/version-manifest.json', record['path'], expected=record['sha256'])
            for name, key in (('commit_probe.pkl', 'commit'), ('boundary_probe.pkl', 'boundary')):
                bundle.copy('grounding/' + name, probe[key], expected=probe['head_sha256'][name], parse_json=False)
            bundle.copy('grounding/calibration-report.json', Path(probe['commit']).parent/'calibration-report.json',
                        expected=probe['calibration_report_sha256'])
            updates, attempts = _validate_ledgers(base, run, step, driver, bundle, source_job, chain)
            need(snap['completed_update'] == updates[-1], 'ARCHIVE_UPDATE_AND_PREFIX_DIFFER')
            refs = _trace_references(base, run, updates, attempts, bundle, chain)
            observed = _validation_observation(base, step, bundle, chain)
            phase = dict(checkpoint_saved_after=('stage64' if step == 5 else 'fixed512'),
                fixed512_callback_pending_at_checkpoint=step == 5,
                grounding_refresh_pending_at_checkpoint=step in (50, 100, 150),
                final51713_pending_at_checkpoint=step == 200,
                post_callback_rng_captured=False, current_observation_is_not_checkpoint_RNG_state=True,
                **observed)
            bundle.put('checkpoint-callback-boundary.json', encoded(phase))
            skipped = base/'logs/formal-inline-development-initial-skipped.json'
            if skipped.exists(): bundle.copy('identity/initial-evaluation-skipped.json', skipped)
            manifest = dict(schema_version=1, role='easy2560_checkpoint_evidence_supplement',
                source_base=str(base), source_commit=training_commit, run_id=run_id, source_job=source_job, step=step,
                single_original_job_prefix=chain is None, inherited_history_assembled=chain is not None,
                authenticated_history=chain['public'] if chain else None,
                source_checkpoint=str(cp), actual_cuda_restore_accepted=False, actual_restore_performed=False,
                post_callback_rng_captured=False, new_training_updates=0, gpu_allocations=0,
                files=bundle.files, source_reads=bundle.source_reads,
                selected_large_trace_references=len(refs), copied_selected_trace_bytes=0,
                checkpoint_large_files_rehashed_by_this_operator=False,
                source_publication_binding=source_binding,
                core_archive=dict(status=ARCHIVE_STATUS, training_world_size=4,
                    source_snapshot=dict(relative_path='identity/source-snapshot.json',
                        sha256=bundle.files['identity/source-snapshot.json']['sha256']),
                    source_acceptance=dict(relative_path='identity/accepted-archive.json',
                        sha256=bundle.files['identity/accepted-archive.json']['sha256']),
                    checkpoint_complete_sha256=bundle.files['identity/checkpoint-complete.json']['sha256'],
                    file_count=archive['file_count'], total_bytes=archive['total_bytes'],
                    actual_cuda_restore_accepted=False,
                    remote_location=dict(host='hkust3', directory='/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/checkpoints/'+run_id+'/global_step_'+str(step),
                        location_source='reviewed archive_controller.HK_ROOT layout', current_remote_read_performed=False)),
                operator_source_sha256=digest(Path(__file__).read_bytes()), checked_unix=time.time(),
                scope=('sealed authenticated original-job prefix plus restored-job suffix and saved callback phase' if chain else
                       'sealed exact original-job prefixes and saved callback phase') + '; byte preservation, not recovery performed by this CPU operator')
            bundle.put('supplement.json', encoded(manifest))
            marker = dict(status=STATUS, source_commit=training_commit, run_id=run_id, source_job=source_job,
                          step=step, supplement_sha256=bundle.files['supplement.json']['sha256'],
                          actual_cuda_restore_accepted=False, post_callback_rng_captured=False)
            bundle.put('complete.json', encoded(marker))
            verify(stage)
            # Re-read every copied source prefix/file after sealing; appends are
            # allowed, replacement or changed prefix bytes are not.
            recheck_sources(bundle.source_reads)
            for directory, _, _ in os.walk(stage, topdown=False): fsync_directory(directory)
            stage.rename(target); fsync_directory(root)
            return verify(target)
        except Exception:
            # Remove only this newly created operator-owned unpublished staging.
            shutil.rmtree(stage)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('seal'); p.add_argument('--base', required=True)
    p.add_argument('--training-commit', required=True); p.add_argument('--run-id', required=True)
    p.add_argument('--source-job', required=True); p.add_argument('--step', type=int, required=True, choices=STEPS)
    p = sub.add_parser('verify'); p.add_argument('--directory', required=True)
    args = parser.parse_args()
    result = seal(args.base, args.training_commit, args.run_id, args.source_job, args.step) if args.mode == 'seal' else verify(args.directory)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
