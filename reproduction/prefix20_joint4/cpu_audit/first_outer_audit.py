#!/usr/bin/env python3
"""Read-only evidence audit of the first actual Prefix20 completed outer.

This operator never calls or changes the final consumer's COMPLETE200 gate.
Only the explicitly selected small JSON result may be written.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
import types

MAX_LEDGER_BYTES = 256 * 1024 ** 2
HELPER_NAMES = ('audit_prefix20.py', 'safe_pt.py')


class Rejected(ValueError):
    pass


class Waiting(Rejected):
    pass


def need(value, reason):
    if not value:
        raise Rejected(reason)


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(Rejected('NONFINITE_JSON')))


def live_rows(base, relative):
    """Read one bounded append-only prefix, retaining its exact SHA.

    Later appends are allowed; replacement/truncation is rejected. A partial
    trailing row is explicitly excluded and counted, never treated as complete.
    """
    path = base / relative
    need(path.resolve() == path and not path.is_symlink(), 'UNSAFE_LIVE_LEDGER')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(str(path), flags)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_LEDGER_BYTES,
             'LIVE_LEDGER_TYPE_OR_SIZE_BOUND')
        pieces = []
        left = before.st_size
        while left:
            piece = os.read(fd, min(left, 1024 ** 2))
            need(piece, 'LIVE_LEDGER_TRUNCATED_DURING_PREFIX_READ')
            pieces.append(piece)
            left -= len(piece)
        after = os.fstat(fd)
        current = path.lstat()
        need((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino) ==
             (current.st_dev, current.st_ino) and after.st_size >= before.st_size and
             current.st_size >= before.st_size, 'LIVE_LEDGER_REPLACED_OR_TRUNCATED')
    finally:
        os.close(fd)
    raw = b''.join(pieces)
    end = raw.rfind(b'\n') + 1
    complete = raw[:end]
    rows = [strict(row) for row in complete.splitlines()]
    record = dict(path=str(path), role='append_only_complete_row_prefix_snapshot',
                  initial_file_bytes=before.st_size, read_complete_prefix_bytes=end,
                  ignored_incomplete_tail_bytes=len(raw) - end,
                  read_complete_prefix_sha256=hashlib.sha256(complete).hexdigest(),
                  observed_device=before.st_dev, observed_inode=before.st_ino,
                  rows=len(rows), checked_unix=time.time())
    return rows, record


def validate_updates(rows, run_id):
    if not rows:
        raise Waiting('WAITING_FIRST_ACTUAL_COMPLETE: no complete update row')
    need(len(rows) <= 200 and [r.get('step') for r in rows] == list(range(1, len(rows) + 1)),
         'UPDATE_LEDGER_NONCONTIGUOUS_OR_UPDATE201')
    need(all(type(r.get('step')) is int and r.get('run_id') == run_id for r in rows),
         'UPDATE_LEDGER_RUN_OR_STEP_TYPE_DIFFERS')
    return rows[0]


def first_attempt_window(rows, first_update, run, job):
    """Validate real attempts through the first success, allowing skipped attempts.

    A successful outer needs >=1 retained actor write. It does not implicitly
    become a full20-write outer; probe still performs its own ten updates.
    """
    completed = 0
    writes = 0
    ids = set()
    selected = []
    for ordinal, row in enumerate(rows, 1):
        aid = row.get('attempt_id')
        need(isinstance(aid, str) and re.fullmatch('[0-9a-f]{32}', aid) and aid not in ids,
             'ATTEMPT_ID_INVALID_OR_REPEATED')
        need(row.get('run_id') == run['run_id'] and row.get('slurm_job_id') == job and
             type(row.get('attempt_index')) is int and row['attempt_index'] == ordinal and
             type(row.get('logical_update')) is int and row['logical_update'] == 1,
             'FIRST_OUTER_ATTEMPT_RUN_JOB_OR_SEQUENCE_DIFFERS')
        need(row.get('all_ranks_agreed') is True and row.get('real_rank_count') == 4 and
             row.get('optimizer_writes_before') == writes, 'FIRST_OUTER_REAL_FOUR_RANK_COUNTERS_DIFFER')
        count = row.get('optimizer_writes_this_attempt')
        processed = row.get('opportunities_processed')
        statuses = row.get('opportunity_statuses', [])
        need(type(count) is int and type(processed) is int and 0 <= count <= processed <= 20 and
             processed >= 1 and row.get('optimizer_writes_after') == writes + count and
             len(statuses) == processed and all(s in ('written', 'written_with_post_guard',
                 'skipped_guard', 'rolled_back_guard') for s in statuses) and
             sum(s in ('written', 'written_with_post_guard') for s in statuses) == count,
             'FIRST_OUTER_ACTUAL_WRITES_OR_OPPORTUNITIES_DIFFER')
        complete = count > 0
        need(row.get('complete_outer') is complete and row.get('scheduler_advances') == int(complete) and
             row.get('completed_outer') == int(complete), 'FIRST_OUTER_COMPLETION_OR_SCHEDULER_DIFFERS')
        if row.get('recovery_protocol_version', 1) == 2:
            full = count == processed == 20
            need(row.get('full_budget_complete') is full and row.get('opportunities_unattempted') == 20 - processed and
                 row.get('status') == ('complete' if full else 'partial' if complete else 'recoverable_guard'),
                 'FIRST_OUTER_SAFE_PROGRESS_VS_FULL_BUDGET_DIFFERS')
        else:
            need(processed == 20, 'LEGACY_ATTEMPT_BUDGET_DIFFERS')
        ids.add(aid)
        selected.append(row)
        writes += count
        if complete:
            need(first_update.get('attempt_id') == aid and first_update.get('attempt_index') == ordinal and
                 first_update.get('actual_optimizer_writes') == count and
                 first_update.get('optimizer_writes_before') == row['optimizer_writes_before'] and
                 first_update.get('optimizer_writes_after') == row['optimizer_writes_after'],
                 'FIRST_UPDATE_NOT_MATCHED_TO_ACTUAL_COMPLETED_ATTEMPT')
            completed = 1
            break
    if completed != 1:
        raise Waiting('WAITING_CAUSAL_FIRST_COMPLETED_ATTEMPT_VISIBILITY')
    return selected, dict(actual_attempts_through_first_complete=len(selected),
        aborted_attempts_before_first_complete=len(selected) - 1,
        completed_outers_audited=1, actual_actor_adam_writes_per_rank=writes,
        first_actor_full20_write_budget=(selected[-1]['optimizer_writes_this_attempt'] == 20),
        target_outers_unchanged=200)


def validate_rank_window(snapshots, attempts, run, job):
    by_id = {a['attempt_id']: a for a in attempts}
    last_index = attempts[-1]['attempt_index']
    writes = attempts[-1]['optimizer_writes_after']
    ranks = []
    for rank in range(4):
        all_scheduler = snapshots['logs/scheduler-rank-%d.jsonl' % rank]
        all_worker = snapshots['logs/optimizer-rank-%d.jsonl' % rank]
        for row in all_scheduler:
            need(row.get('run_id') == run['run_id'] and row.get('source_commit') == run['source_commit'] and
                 row.get('rank') == rank and row.get('world_size') == 4 and row.get('slurm_job_id') == job and
                 type(row.get('logical_update')) is int and 1 <= row['logical_update'] <= 200,
                 'OBSERVED_SCHEDULER_BINDING_DIFFERS_OR_UPDATE201')
            need(type(row.get('attempt_index')) is int and row['attempt_index'] >= 1 and
                 (row['attempt_index'] > last_index or row.get('attempt_id') in by_id),
                 'FIRST_OUTER_UNBOUND_SCHEDULER_ATTEMPT')
        scheduler = [r for r in all_scheduler if r.get('attempt_id') in by_id]
        expected = [(a['attempt_id'], kind) for a in attempts for kind in
                    (('BEGIN', 'COMMIT_PREPARED', 'COMPLETE') if a['complete_outer'] else
                     ('BEGIN', 'ABORTED_ATTEMPT'))]
        observed_sequence = [(r['attempt_id'], r['kind']) for r in scheduler]
        if len(observed_sequence) < len(expected) and observed_sequence == expected[:len(observed_sequence)]:
            raise Waiting('WAITING_CAUSAL_FIRST_OUTER_SCHEDULER_ROWS:rank-%d' % rank)
        need(observed_sequence == expected,
             'FIRST_OUTER_SCHEDULER_SEQUENCE_MISSING_OR_DUPLICATE')
        for row in scheduler:
            a = by_id[row['attempt_id']]
            terminal = row['kind'] != 'BEGIN'
            after = int(row['kind'] == 'COMPLETE')
            need(row.get('logical_update') == a['logical_update'] == 1 and
                 row.get('attempt_index') == a['attempt_index'] and row.get('scheduler_last_epoch_before') == 0 and
                 row.get('scheduler_last_epoch_after') == after and row.get('scheduler_step_count_before') == 1 and
                 row.get('scheduler_step_count_after') == after + 1 and
                 row.get('actual_optimizer_writes') == (a['optimizer_writes_this_attempt'] if terminal else 0) and
                 row.get('writes_before') == a['optimizer_writes_before'] and
                 row.get('writes_after') == (a['optimizer_writes_after'] if terminal else a['optimizer_writes_before']) and
                 row.get('consumed_learning_rates') and all(lr == 0. for lr in row['consumed_learning_rates']),
                 'FIRST_OUTER_ACTUAL_SCHEDULER_COUNTER_WRITES_OR_CONSUMED_LR_DIFFERS')
        worker = []
        for row in all_worker:
            context = row.get('context') or row
            need(row.get('run_id') == run['run_id'] and row.get('rank') == rank and
                 type(context.get('logical_update')) is int and 1 <= context['logical_update'] <= 200 and
                 type(context.get('attempt_index')) is int and context['attempt_index'] >= 1,
                 'OBSERVED_WORKER_BINDING_DIFFERS_OR_UPDATE201')
            if context['attempt_index'] <= last_index:
                aid = context.get('attempt_id')
                need(aid in by_id and context['logical_update'] == 1 and
                     context['attempt_index'] == by_id[aid]['attempt_index'],
                     'FIRST_OUTER_WORKER_ATTEMPT_CONTEXT_DIFFERS')
                worker.append(row)
        observed_writes = [r['write_count'] for r in worker if r['kind'] == 'OPTIMIZER_WRITE']
        expected_writes = list(range(1, writes + 1))
        if len(observed_writes) < len(expected_writes) and observed_writes == expected_writes[:len(observed_writes)]:
            raise Waiting('WAITING_CAUSAL_FIRST_OUTER_ADAM_WRITE_ROWS:rank-%d' % rank)
        need(observed_writes == expected_writes,
             'FIRST_OUTER_PHYSICAL_ADAM_WRITE_SEQUENCE_DIFFERS')
        for attempt in attempts:
            rows = [r for r in worker if (r.get('context') or r).get('attempt_id') == attempt['attempt_id']]
            opportunities = [r for r in rows if r['kind'] == 'OPPORTUNITY']
            retained = [r for r in rows if r['kind'] == 'OPTIMIZER_WRITE']
            if len(opportunities) < attempt['opportunities_processed']:
                raise Waiting('WAITING_CAUSAL_FIRST_OUTER_OPPORTUNITY_ROWS:rank-%d' % rank)
            need([r.get('opportunity_index') for r in opportunities] == list(range(attempt['opportunities_processed'])) and
                 len(retained) == attempt['optimizer_writes_this_attempt'] and
                 sum(r.get('optimizer_write_succeeded') is True for r in opportunities) == len(retained),
                 'FIRST_OUTER_ACTUAL_RANK_OPPORTUNITY_OR_WRITE_RECEIPTS_DIFFER')
            for row in opportunities:
                index = row['opportunity_index']
                need(row.get('status') == attempt['opportunity_statuses'][index],
                     'FIRST_OUTER_RANK_DISPOSITION_DIFFERS_FROM_DRIVER')
            counter = attempt['optimizer_writes_before']
            for row in opportunities:
                wrote = row['status'] in ('written', 'written_with_post_guard')
                need(row.get('optimizer_write_succeeded') is wrote and
                     row.get('optimizer_writes_before') == counter and
                     row.get('optimizer_writes_after') == counter + int(wrote),
                     'FIRST_OUTER_PHYSICAL_OPPORTUNITY_COUNTER_DIFFERS')
                if wrote:
                    matches = [r for r in retained if (r.get('context') or r).get('minibatch_index') == row['opportunity_index']]
                    need(len(matches) == 1 and matches[0].get('write_count') == counter + 1 and
                         matches[0].get('learning_rates') and
                         all(lr == 0. for lr in matches[0]['learning_rates']),
                         'FIRST_OUTER_PHYSICAL_WRITE_INDEX_OR_CONSUMED_LR_DIFFERS')
                counter += int(wrote)
            need(counter == attempt['optimizer_writes_after'], 'FIRST_OUTER_RANK_FINAL_COUNTER_DIFFERS')
        ranks.append(dict(rank=rank, actual_retained_adam_writes=writes,
                          scheduler_completed_outer=1, scheduler_step_count=2,
                          prefix_scheduler_rows=len(scheduler), prefix_worker_rows=len(worker),
                          later_scheduler_rows_not_audited=len(all_scheduler) - len(scheduler),
                          later_worker_rows_not_audited=len(all_worker) - len(worker)))
    return ranks


def load_helpers(base, commit, run):
    need('torch' not in sys.modules, 'TORCH_ALREADY_IMPORTED')
    directory = base / 'source' / commit / 'reproduction/prefix20_joint4/cpu_audit'
    hashes = {}
    for name in HELPER_NAMES:
        path = directory / name
        need(path.resolve() == path and path.is_file() and not path.is_symlink(), 'UNSAFE_BOUND_CPU_HELPER')
        raw = path.read_bytes()
        relative = 'reproduction/prefix20_joint4/cpu_audit/' + name
        digest = hashlib.sha256(raw).hexdigest()
        need(run['source_sha256'].get(relative) == digest, 'BOUND_CPU_HELPER_SHA_DIFFERS')
        hashes[name] = digest
    package_name = '_first_outer_bound_cpu'
    need(package_name not in sys.modules, 'CPU_HELPERS_ALREADY_IMPORTED')
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(package_name + '.audit_prefix20', directory / 'audit_prefix20.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    need('torch' not in sys.modules, 'BOUND_CPU_HELPER_IMPORTED_TORCH')
    return module, hashes


def require_causal_artifacts(base, attempts):
    missing = []
    for attempt in attempts:
        aid = attempt['attempt_id']
        prefix = 'artifacts/question-probe/attempts/' + aid + '/'
        relative = [prefix + name for name in ('selection.json', 'rollout-audit.json',
                    'candidate-prefixes.pt', 'real-rollout.pt')]
        relative += [prefix + phase + '-worker-rng/rank-%d.pt' % rank
                     for phase in ('before-prefix', 'after-prefix', 'after-prediction', 'after-continuation')
                     for rank in range(4)]
        relative += ['logs/behavior-probability-attempt-' + aid + '.json',
                     'artifacts/behavior-traces/attempt-' + aid + '.json']
        relative += [('artifacts/question-probe/completed/outer-0001.json' if attempt['complete_outer']
                      else 'artifacts/question-probe/aborted/' + aid + '.json')]
        for name in relative:
            if not (base / name).exists():
                missing.append(name)
    if missing:
        raise Waiting('WAITING_CAUSAL_FIRST_OUTER_ARTIFACTS:' + json.dumps(missing[:8]))


def audit(base_value, commit, run_id):
    base = Path(base_value)
    need(base.is_absolute() and base.resolve() == base and base.is_dir(), 'EXPLICIT_REAL_BASE_REQUIRED')
    need(re.fullmatch('[0-9a-f]{40}', commit) and re.fullmatch('[0-9a-f]{32}', run_id),
         'EXPLICIT_FULL_COMMIT_RUN_ID_REQUIRED')
    updates_path = base / 'logs/actual-updates.jsonl'
    if not updates_path.exists():
        raise Waiting('WAITING_FIRST_ACTUAL_COMPLETE: actual-updates.jsonl absent')
    update_rows, update_snapshot = live_rows(base, 'logs/actual-updates.jsonl')
    first = validate_updates(update_rows, run_id)
    manifest = base / 'logs/run-manifest.json'
    need(manifest.resolve() == manifest and not manifest.is_symlink() and manifest.stat().st_size < 2 * 1024 ** 2,
         'UNSAFE_RUN_MANIFEST')
    run = strict(manifest.read_bytes())
    need(run.get('run_id') == run_id and run.get('source_commit') == commit and
         run.get('training_world_size') == 4 and run.get('actor_micro_batch_per_rank') == 4 and
         run.get('data_selection_mode') == 'prefix20_probe' and run.get('mixed_first_selection') is False and
         run.get('target_updates') == 200 and run.get('completed_updates') == 0 and not run.get('full_state_restore'),
         'FIRST_OUTER_FRESH_NATIVE4_RUN_BINDING_DIFFERS')
    helper, helper_hashes = load_helpers(base, commit, run)
    need(run.get('model_revision') == helper.MODEL_REV and
         all(helper.same(run.get('question_probe', {}).get(k), v) for k, v in helper.SETTINGS.items()),
         'FIRST_OUTER_SCIENTIFIC_SETTINGS_DIFFER')
    reads = helper.Reads(base)
    source = helper.source_and_config(reads, run)
    try:
        all_attempts, attempt_snapshot = live_rows(base, 'logs/actual-attempts.jsonl')
    except FileNotFoundError:
        raise Waiting('WAITING_CAUSAL_ACTUAL_ATTEMPTS_LEDGER')
    for row in all_attempts:
        need(type(row.get('logical_update')) is int and 1 <= row['logical_update'] <= 200 and
             row.get('run_id') == run_id and row.get('slurm_job_id') == reads._job,
             'OBSERVED_ATTEMPT_BINDING_DIFFERS_OR_UPDATE201')
    attempts, history = first_attempt_window(all_attempts, first, run, reads._job)
    snapshots = {}
    ledgers = [update_snapshot, attempt_snapshot]
    for rank in range(4):
        for prefix in ('scheduler', 'optimizer'):
            relative = 'logs/%s-rank-%d.jsonl' % (prefix, rank)
            try:
                rows, record = live_rows(base, relative)
            except FileNotFoundError:
                raise Waiting('WAITING_CAUSAL_FIRST_OUTER_RANK_LEDGER:' + relative)
            snapshots[relative] = rows
            ledgers.append(record)
    ranks = validate_rank_window(snapshots, attempts, run, reads._job)
    require_causal_artifacts(base, attempts)
    reads._updates = [first]
    reads._update_by_id = {first['attempt_id']: first}
    reads._candidate_cycles = {}
    eligible, frames, common_manifest, tokenizer = helper.canonical_inputs(reads, run)
    outers = [helper.audit_outer(reads, run, attempt, attempt['complete_outer'],
              eligible, frames['train'], tokenizer) for attempt in attempts]
    cycles = reads._candidate_cycles
    need(sorted(cycles) == list(range(len(cycles))) and
         all(len(cycles[c]) == len(eligible) for c in sorted(cycles)[:-1]) and
         sum(len(v) for v in cycles.values()) == 5120 * len(attempts),
         'FIRST_OUTER_ACTUAL_ELIGIBLE_CYCLE_OR_CANDIDATE_CONSUMPTION_DIFFERS')
    checkpoint_root = base / 'checkpoints/search-g1-common8/actor'
    for path in checkpoint_root.glob('global_step_*'):
        suffix = path.name[len('global_step_'):]
        need(suffix.isdigit() and int(suffix) <= 200, 'OBSERVED_CHECKPOINT201_OR_UNKNOWN_STEP')
    need('torch' not in sys.modules, 'FIRST_OUTER_CONSUMER_IMPORTED_TORCH')
    return dict(schema_version=1, kind='prefix20_first_completed_outer_readonly_cpu_audit',
        status='FIRST_OUTER_RECORDED_EVIDENCE_PASS', accepted=True,
        source_base=str(base), source_commit=commit, run_id=run_id, slurm_job_id=reads._job,
        checked_unix=time.time(), target_outers=200, completed_outers_audited=1,
        final_acceptance=False, goal_complete=False, source_and_config=source,
        actual_history=history, actual_actor_rank_receipts=ranks, attempts=outers,
        helper_source_sha256=helper_hashes, reader_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        live_ledger_prefix_snapshots=ledgers, immutable_evidence_files=list(reads.files.values()),
        no_update201_in_observed_snapshots=True, torch_imported=False,
        model_reloads=0, gpu_allocations=0, training_updates=0,
        limitations=['Only actual attempts through completed outer1 are fully audited; later complete-row snapshots are checked for identity and no201 only.',
            'Live ledgers are bounded append-only complete-row prefix snapshots; trailing partial rows are explicitly recorded and excluded.',
            'Actual probe10/rank receipts are consumed; serialized final probe Adam2000 and checkpoints/fixed/final/recovery remain final200 gates.',
            'This CPU reader does not independently recompute gradients or authenticate GPU probabilities from JSON.'])


def write_once(path_value, base_value, result):
    path = Path(path_value)
    base = Path(base_value)
    need(path.is_absolute() and path.parent.resolve() == path.parent and
         path.parent in (base / 'logs', base / 'preparation'), 'SMALL_RECEIPT_IN_EXISTING_OWNED_LOGS_OR_PREPARATION_REQUIRED')
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    need(len(raw) <= 256 * 1024, 'SMALL_RECEIPT_SIZE_BOUND_EXCEEDED')
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(str(path.parent), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('base', 'commit', 'run-id', 'output'):
        parser.add_argument('--' + field, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.base, args.commit, args.run_id)
        code = 0
    except Exception as exc:
        code = 2 if isinstance(exc, Waiting) else 1
        result = dict(schema_version=1, kind='prefix20_first_completed_outer_readonly_cpu_audit',
            status='WAITING_FIRST_ACTUAL_COMPLETE' if code == 2 else 'REJECTED',
            accepted=False, final_acceptance=False, goal_complete=False,
            source_base=args.base, source_commit=args.commit, run_id=args.run_id,
            checked_unix=time.time(), reason=str(exc), error_type=type(exc).__name__,
            torch_imported='torch' in sys.modules, model_reloads=0, gpu_allocations=0, training_updates=0)
    write_once(args.output, args.base, result)
    print(json.dumps({key:result[key] for key in ('status', 'accepted', 'final_acceptance', 'goal_complete',
                     'source_base', 'source_commit', 'run_id')}, sort_keys=True), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
