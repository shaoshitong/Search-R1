"""Preserve the terminal owned outer35 failure; never alter original evidence.

Only small logs, source/submit metadata and selected original runtime sources are
copied. Raw generations and CP34/model/Adam/RNG payloads stay in place. An
existing complete receipt is SHA-verified; a partial attempt is never replayed.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp12-20261022')
JOB = '746125'
COMMIT = 'ebb6844db6a1ae739a06d5cc31d3510e7b4779b1'
RUN = '232f20e112ac4a978c5aba52a0367195'
NODE = 'ACD1-32'
UID = 14090
MAX_FILE = 256 * 1024**2
SOURCE_NAMES = (
    'probe_server.py', 'ts_probe_dump.py', 'ts_probe_train.py',
    'verl/trainer/main_ppo_eco.py', 'verl/trainer/ppo/ray_trainer.py',
    'verl/workers/fsdp_workers.py', 'verl/workers/actor/dp_actor.py',
    'verl/workers/rollout/vllm_rollout/vllm_rollout.py',
    'search_g1/flash3_attention.py', 'search_g1/varlen_execution.py',
    'search_g1/logprob_memory.py', 'search_g1/monitor_forward_reuse.py',
    'search_g1/full_state_restore.py', 'search_g1/easy2560_restore.py',
    'search_g1/training_state.py', 'search_g1/mixed_runtime.py',
    'reproduction/common8/scheduler.py', 'reproduction/easy2560_joint4/train.sh',
    'reproduction/easy2560_joint4/native_ray_resources.py')


def need(value, message):
    if not value: raise ValueError('PRESERVE746125_' + message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): value.update(block)
    return value.hexdigest()


def read(path): return json.loads(Path(path).read_bytes())


def fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def publish(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    fsync_directory(path.parent)


def identity(path):
    seen = path.stat()
    return seen.st_dev, seen.st_ino, seen.st_size, seen.st_mtime_ns, seen.st_uid


def complete_lines(path):
    rows = []
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'): break
            if line.strip(): rows.append(json.loads(line))
    return rows


def verify_existing(target, receipt):
    value = read(receipt)
    need(value.get('source_job') == JOB and value.get('source_commit') == COMMIT and
         value.get('run_id') == RUN and value.get('completed_outer') == 34 and
         value.get('failed_outer') == 35, 'EXISTING_IDENTITY_DIFFERS')
    for row in value['files']:
        relative = Path(row['relative_path'])
        need(not relative.is_absolute() and '..' not in relative.parts, 'UNSAFE_RECEIPT_MEMBER')
        path = target/relative
        need(path.is_file() and not path.is_symlink() and path.resolve() == path and
             path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'EXISTING_COPY_SHA_DIFFERS')
    return value


def failed_suffix_observations():
    result = {}
    for rank in range(4):
        path = BASE/('logs/optimizer-rank-%d.jsonl' % rank)
        if not path.is_file(): continue
        maximum = None; terminal = []
        for row in complete_lines(path):
            context = row.get('context') or row
            step = context.get('logical_update', row.get('logical_update'))
            counts = [row.get(key) for key in ('write_count', 'optimizer_writes', 'cumulative_writes', 'writes_after',
                                              'optimizer_writes_after')]
            counts += [context.get('prior_optimizer_writes')]
            counts = [number for number in counts if type(number) is int and number >= 680]
            if step == 35 or any(number > 680 for number in counts):
                if counts: maximum = max(counts + ([maximum] if maximum is not None else []))
                if row.get('kind') in ('REJECTED', 'ABORTED_ATTEMPT', 'OUTER_BUDGET', 'POST_COMMIT_FAILURE'):
                    terminal.append({key:row[key] for key in ('kind', 'optimizer_writes', 'writes_after',
                        'optimizer_writes_after', 'cumulative_writes', 'reason', 'guard_reasons') if key in row})
        result[str(rank)] = dict(maximum_logged_cumulative_writes=maximum,
                                failed_suffix_terminal_records=terminal[-5:])
    return result


def preserve():
    need(os.getuid() == UID and BASE.resolve() == BASE and not BASE.is_symlink(), 'OWNED_SOURCE_REQUIRED')
    target = BASE/('failed-job-' + JOB); receipt = target/'failure-evidence.json'
    if target.exists():
        need(not target.is_symlink() and target.resolve() == target and receipt.is_file() and
             not receipt.is_symlink(), 'PARTIAL_PRESERVATION_EXISTS_INSPECT_WITHOUT_OVERWRITING')
        return verify_existing(target, receipt)
    accounting = subprocess.run(['sacct', '-j', JOB, '--noheader', '--parsable2',
        '--format=JobIDRaw,User,State,ExitCode,NodeList,Start,End'],
        capture_output=True, text=True, timeout=30)
    rows = [line.strip().split('|') for line in accounting.stdout.splitlines() if line.strip()]
    own = [row for row in rows if len(row) >= 7 and row[0] == JOB]
    need(accounting.returncode == 0 and len(own) == 1 and
         own[0][1:5] == ['jxu957', 'FAILED', '1:0', NODE], 'ACTUAL_OWN_TERMINAL_FAILED1_0_REQUIRED')
    run = read(BASE/'logs/run-manifest.json')
    submitted = read(BASE/'preparation/training-submission.json')
    claim = read(BASE/'preparation/training-submit-claim.json')
    source = read(BASE/'preparation/run-source.json')
    need((BASE/'job.id').read_text().strip() == JOB and run['run_id'] == RUN and
         run['source_commit'] == COMMIT and submitted.get('job') == JOB and submitted.get('returncode') == 0 and
         all(value.get('source_commit') == COMMIT for value in (submitted, claim, source)), 'SOURCE_SUBMIT_BINDING_DIFFERS')
    latest = read(BASE/'preparation/rolling-checkpoints/latest.json')
    checkpoint = BASE/'checkpoints/search-g1-common8/actor/global_step_34'
    complete = read(checkpoint/'complete.json')
    updates = complete_lines(BASE/'logs/actual-updates.jsonl')
    need(updates and updates[-1].get('step') == 34 and updates[-1].get('optimizer_writes_after') == 680 and
         latest.get('step') == 34 and latest.get('run_id') == RUN and latest.get('source_commit') == COMMIT and
         latest.get('status') == 'EASY2560_ROLLING_CHECKPOINT_READY' and
         complete.get('run_id') == RUN and complete.get('step') == 34 and
         complete.get('validated_tensor_count') == 339 and complete.get('actual_optimizer_writes_per_rank') == 680,
         'LAST_COMPLETE34_SOURCE680_DIFFERS')
    names = ['job.id', 'preparation/training-submit-claim.json', 'preparation/training-submission.json',
             'preparation/run-source.json', 'source/'+COMMIT+'/source-publication.json',
             'preparation/rolling-checkpoints/latest.json',
             'preparation/rolling-checkpoints/global_step_34/full-manifest.json']
    skipped = []
    for path in sorted((BASE/'logs').iterdir()):
        if path.is_file() and not path.is_symlink() and path.suffix in ('.json', '.jsonl', '.log'):
            if path.stat().st_size <= MAX_FILE: names.append(str(path.relative_to(BASE)))
            else: skipped.append(dict(path=str(path), bytes=path.stat().st_size,
                                      reason='Large log retained at its original path; not copied'))
    evidence = [(name, BASE/name, None) for name in names]
    evidence += [('runtime/'+name, BASE/'repo'/name, run['source_sha256'][name]) for name in SOURCE_NAMES]
    evidence.append(('operator/preserve_failed746125.py', Path(__file__).resolve(), None))
    target.mkdir(mode=0o770)
    publish(target/'claim.json', dict(source_job=JOB, source_commit=COMMIT, run_id=RUN,
        pid=os.getpid(), created_unix=time.time(), automatic_retry=False))
    publish(target/'terminal-observation.json', dict(checked_unix=time.time(),
        returncode=accounting.returncode, stdout=accounting.stdout, stderr=accounting.stderr))
    records = []
    for name, original, expected in evidence:
        need(original.is_file() and not original.is_symlink() and original.stat().st_size <= MAX_FILE,
             'ONLY_SMALL_REGULAR_EVIDENCE_MAY_BE_COPIED_' + name)
        before = identity(original); digest = sha(original)
        need(expected is None or digest == expected, 'ORIGINAL_RUNTIME_SOURCE_SHA_DIFFERS_' + name)
        copied = target/name; copied.parent.mkdir(parents=True, exist_ok=True)
        with original.open('rb') as src, copied.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1024**2); dst.flush(); os.fsync(dst.fileno())
        need(identity(original) == before and sha(original) == digest and sha(copied) == digest,
             'EVIDENCE_COPY_SHA_OR_SOURCE_IDENTITY_DIFFERS_' + name)
        records.append(dict(relative_path=name, bytes=copied.stat().st_size, sha256=digest))
    for name in ('claim.json', 'terminal-observation.json'):
        path = target/name
        records.append(dict(relative_path=name, bytes=path.stat().st_size, sha256=sha(path)))
    value = dict(status='FAILED746125_OUTER35_CUDA_OOM_EVIDENCE_PRESERVED', source_job=JOB,
        source_commit=COMMIT, run_id=RUN, source_base=str(BASE), source_node=NODE,
        completed_outer=34, failed_outer=35, latest_durable_checkpoint_outer=34,
        restorable_checkpoint_optimizer_writes_per_rank=680, resumed_training_must_recompute_from_outer=35,
        failed_suffix_partial_adam_writes_observed=True, failed_suffix_writes_not_claimed_zero=True,
        prior_actual_partial_suffix_observation=dict(rank=3, status='REJECTED', cumulative_writes=682,
            scope='Previously observed actual suffix; this is not a count for all four ranks'),
        observed_failed_suffix_counters=failed_suffix_observations(), files=records,
        skipped_large_logs_retained_at_original_paths=skipped,
        raw_checkpoints_and_tensors_retained_at_original_paths=True, tensors_copied=False,
        original_evidence_modified=False, training_modified=False, jobs_submitted=0, jobs_stopped=0,
        external_retrieval745198_modified=False, actual_cuda_restore_accepted=False,
        operator_sha256=sha(__file__), checked_unix=time.time())
    for folder, _, _ in os.walk(str(target), topdown=False): fsync_directory(Path(folder))
    publish(receipt, value)
    return value


if __name__ == '__main__':
    result = preserve()
    print(json.dumps(dict(status=result['status'], source_job=JOB, files=len(result['files']),
        bytes=sum(row['bytes'] for row in result['files']), completed_outer=34, failed_outer=35,
        receipt=str(BASE/('failed-job-'+JOB)/'failure-evidence.json'),
        receipt_sha256=sha(BASE/('failed-job-'+JOB)/'failure-evidence.json'), jobs_submitted=0, jobs_stopped=0),
        sort_keys=True), flush=True)
