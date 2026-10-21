"""Read-only Easy2560 final CPU consumer with explicit published-source binding.

This consumes existing scores and durable scalar/byte-preservation evidence.
It neither imports Torch nor generates answers, trains, allocates a GPU or
reloads a model. A successful report is not an actual CUDA recovery receipt.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

REVIEWED_COMMIT = '93eda66a978dae7a1086fdb2e6f07ef4a7fc3d48'
STEPS = (5, 50, 100, 150, 200)
STATUS = 'EASY2560_FINAL_CPU_EVALUATION_AND_ARCHIVE_EVIDENCE_ACCEPTED'
ARCHIVE_STATUS = 'EASY2560_FULL_CHECKPOINT_SOURCE_ARCHIVE_SHA_PASS'
MAX_SMALL = 128 * 1024**2
MAX_LINE = 32 * 1024**2


def need(condition, message):
    if not condition:
        raise ValueError('EASY2560_FINAL_CPU_' + message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _constant(value):
    raise ValueError('EASY2560_FINAL_CPU_NONFINITE_JSON_' + value)


def _pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, parse_constant=_constant, object_pairs_hook=_pairs)


def _verify_stable_fd_snapshot(stream, path, before, raw_sha256, byte_count, *,
                               check=need, changed='EVIDENCE_CHANGED_DURING_READ'):
    """DPC may settle a just-written mtime/ctime while an unchanged fd is read.

    Never tolerate content or identity differences: timestamps alone trigger
    one complete same-fd byte comparison. No time tolerance, sleep, new open,
    or retry is used. Each path snapshot is a single stat call.
    """
    after = os.fstat(stream.fileno()); path_after = path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_uid)
    check(identity(before) == identity(after) == identity(path_after) and byte_count == before.st_size, changed)
    timestamps = lambda st: (st.st_mtime_ns, st.st_ctime_ns)
    if timestamps(before) == timestamps(after) == timestamps(path_after):
        return
    stream.seek(0); reread = hashlib.sha256(); length = 0
    while True:
        block = stream.read(min(8 * 1024**2, byte_count-length+1))
        if not block:
            break
        reread.update(block); length += len(block)
        check(length <= byte_count, changed)
    final = os.fstat(stream.fileno()); path_final = path.stat()
    check(identity(before) == identity(final) == identity(path_final) and
          length == byte_count and reread.hexdigest() == raw_sha256, changed)


def small(path):
    path = Path(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        need(before.st_size <= MAX_SMALL, 'SMALL_EVIDENCE_LIMIT')
        raw = stream.read(MAX_SMALL + 1)
        sha256 = digest(raw)
        _verify_stable_fd_snapshot(stream, path, before, sha256, len(raw))
    return decode(raw), dict(path=str(path), bytes=len(raw), sha256=sha256)


def lines(path):
    """Unlike an in-flight prefix seal, final acceptance rejects partial tails."""
    path = Path(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno()); raw_sha256 = hashlib.sha256(); byte_count = 0
        for raw in iter(lambda: stream.readline(MAX_LINE + 1), b''):
            need(raw.endswith(b'\n') and len(raw) <= MAX_LINE, 'PARTIAL_OR_OVERSIZED_FINAL_LEDGER')
            need(raw.strip(), 'BLANK_FINAL_LEDGER_ROW')
            raw_sha256.update(raw); byte_count += len(raw)
            yield decode(raw)
        _verify_stable_fd_snapshot(stream, path, before, raw_sha256.hexdigest(), byte_count,
                                   changed='FINAL_LEDGER_CHANGED_DURING_READ')


def published_source_binding(base, run, training_commit):
    """Default93 stays fixed; a different source must be explicitly requested."""
    need(isinstance(training_commit, str) and re.fullmatch('[0-9a-f]{40}', training_commit) and
         run.get('source_commit') == training_commit, 'EXPLICIT_TRAINING_COMMIT_DIFFERS')
    if training_commit == REVIEWED_COMMIT:
        return dict(policy='fixed_reviewed_commit', requested_training_commit=training_commit)
    original = base/'source'/training_commit/'source-publication.json'
    runtime_path = base/'preparation/run-source.json'
    sidecar = base/'preparation/checkpoint-evidence/global_step_200/identity'
    publication, publication_record = small(original if original.is_file() else sidecar/'source-publication.json')
    runtime, runtime_record = small(runtime_path if runtime_path.is_file() else sidecar/'run-source.json')
    need(publication.get('source_commit') == training_commit and
         publication.get('fresh_source_all_files_sha_verified') is True and
         isinstance(publication.get('files'), dict) and publication['files'], 'PUBLISHED_SOURCE_REQUIRED')
    need(runtime.get('source_commit') == training_commit and runtime.get('files') == publication['files'] and
         runtime.get('runtime_repo') == str(base/'repo'), 'PUBLISHED_RUNTIME_SOURCE_DIFFERS')
    source_map = run.get('source_sha256')
    need(isinstance(source_map, dict) and source_map, 'BOUND_RUNTIME_SOURCE_MAP_REQUIRED')
    for name, sha in source_map.items():
        entry = publication['files'].get(name)
        need(isinstance(name, str) and name not in ('', '.') and not Path(name).is_absolute() and
             '..' not in Path(name).parts and isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha) and
             isinstance(entry, dict) and entry.get('sha256') == sha and
             type(entry.get('bytes')) is int and entry['bytes'] >= 0,
             'PUBLISHED_RUNTIME_SOURCE_SHA_DIFFERS')
    return dict(policy='explicit_caller_commit_publication_runtime_run_map',
                requested_training_commit=training_commit, publication=publication_record,
                runtime_source=runtime_record, automatic_head_or_ancestor_allow=False,
                git_ancestry_verified_here=False)


def canonical(base, split):
    # Lazy Arrow-only dataset access. The helper checks the pinned shared
    # manifest/parquet SHAs and keeps every ordered canonical occurrence.
    from reproduction.common8.validation_identity import expected
    record, frame, ids = expected(Path(base), split)
    return dict(record, question_ids=ids, data_sources=frame['data_source'].tolist())


def scores(value, identity, run, step, *, split, final):
    count = 51713 if final else 512
    need(value.get('run_id') == run['run_id'] and type(value.get('step')) is int and
         value['step'] == step and value.get('split') == split and value.get('final') is final,
         'EVALUATION_IDENTITY_DIFFERS')
    ids, sources = identity['question_ids'], identity['data_sources']
    need(len(ids) == len(sources) == identity['rows'] == count, 'CANONICAL_COUNT_DIFFERS')
    need(value.get('question_ids') == ids and value.get('data_sources') == sources and
         value.get('validation_sha256') == identity['sha256'], 'ORDERED_CANONICAL_OCCURRENCES_DIFFER')
    em, f1 = value.get('em'), value.get('f1')
    need(isinstance(em, list) and isinstance(f1, list) and len(em) == len(f1) == count,
         'EVALUATION_SCORE_COUNT_DIFFERS')
    need(all(type(x) in (int, float) and math.isfinite(x) and x in (0, 1) for x in em), 'INVALID_EM')
    need(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in f1), 'INVALID_F1')
    need(all(x <= y for x, y in zip(em, f1)), 'EM_F1_DISAGREE')
    counts = dict(Counter(sources))
    need(counts == identity['source_counts'] == value.get('counts'), 'SOURCE_COUNTS_DIFFER')
    weighted_em = math.fsum(em) / count; weighted_f1 = math.fsum(f1) / count
    need(value.get('correct') == sum(em) and value.get('weighted_em') == weighted_em and
         value.get('weighted_f1') == weighted_f1, 'PUBLISHED_SCORE_AGGREGATE_DIFFERS')
    if final:
        need(len(counts) == 7 and counts.get('nq') == 3610, 'NATIVE_FINAL_SEVEN_SOURCE_COUNTS_DIFFER')
    else:
        need(counts == {'nq': 256, 'hotpotqa': 256}, 'FIXED512_SOURCE_COUNTS_DIFFER')
    per_source = {}
    for source, n in sorted(counts.items()):
        indices = [i for i, current in enumerate(sources) if current == source]
        correct = math.fsum(em[i] for i in indices)
        per_source[source] = dict(count=n, correct=int(correct), em=correct/n,
                                  f1=math.fsum(f1[i] for i in indices)/n)
    return dict(step=step, question_count=count, counts=counts, correct=int(sum(em)),
                per_source=per_source, weighted_em=weighted_em, weighted_f1=weighted_f1,
                macro_em=math.fsum(row['em'] for row in per_source.values())/len(per_source),
                macro_f1=math.fsum(row['f1'] for row in per_source.values())/len(per_source),
                canonical_occurrences_retained=True, deduplicated=False)


def restore_chain(base, run):
    from reproduction.easy2560_joint4.checkpoint_evidence import authenticated_restore_chain
    return authenticated_restore_chain(base, run)


def chained_lines(base, relative, chain):
    current = list(lines(base/relative))
    if chain is None:
        return current
    # A genuine pre-QA5 restore (for example CP1 with initial evaluation
    # explicitly skipped) has no validation prefix yet. Only this one ledger
    # may be absent before5; evaluations still requires every fixed milestone
    # and the final native51713 result exactly once from the current execution.
    if (relative == 'logs/validation-identities.jsonl' and chain['boundary'] < 5 and
            relative not in chain['historical']):
        return current
    from reproduction.easy2560_joint4.checkpoint_evidence import _historical_rows
    return _historical_rows(chain, relative) + current


def evaluations(base, run, load_canonical=canonical, chain=None):
    fixed_identity = load_canonical(base, 'validation'); final_identity = load_canonical(base, 'final')
    rows = chained_lines(base, 'logs/validation-identities.jsonl', chain)
    need(all(row.get('run_id') == run['run_id'] and type(row.get('step')) is int and
             0 <= row['step'] <= 200 for row in rows), 'FOREIGN_OR_POST200_EVALUATION')
    fixed = {}
    for step in STEPS:
        selected = [row for row in rows if row.get('split') == 'validation' and row['step'] == step]
        need(len(selected) == 1, 'MISSING_OR_DUPLICATE_FIXED512_' + str(step))
        fixed[str(step)] = scores(selected[0], fixed_identity, run, step, split='validation', final=False)
    final, final_record = small(base/'logs/final-identity.json')
    selected = [row for row in rows if row.get('final') is True or row.get('split') == 'final']
    need(len(selected) == 1 and selected[0] == final, 'FINAL_FILE_AND_LEDGER_DIFFER')
    result = scores(final, final_identity, run, 200, split='final', final=True)
    # At5 the producer also writes a distinct fixed512 scalar-row artifact;
    # stage64 is intentionally excluded from the fixed validation curve.
    if chain and chain['boundary'] >= 5:
        from reproduction.easy2560_joint4.checkpoint_evidence import digest as history_digest
        raw = chain['historical']['logs/formal-inline-development-after.json']
        after = decode(raw)
        from reproduction.easy2560_joint4.checkpoint_evidence import historical_origin
        qa_base, _, _ = historical_origin(chain, outer=5)
        after_record = dict(path=str(qa_base/'logs/formal-inline-development-after.json'),
                            bytes=len(raw), sha256=history_digest(raw), authenticated_historical=True)
    else:
        after, after_record = small(base/'logs/formal-inline-development-after.json')
    fixed5 = next(row for row in rows if row.get('split') == 'validation' and row['step'] == 5)
    need(after.get('run_id') == run['run_id'] and after.get('step') == 5 and
         after.get('role') == 'formal_inline_development_evaluation' and
         after.get('validation_sha256') == fixed_identity['sha256'] and
         after.get('question_ids') == fixed5['question_ids'] and after.get('data_sources') == fixed5['data_sources'],
         'FIXED5_SCALAR_ARTIFACT_IDENTITY_DIFFERS')
    measured = after.get('rows', [])
    need(len(measured) == 512 and [r.get('question_id') for r in measured] == fixed5['question_ids'] and
         [r.get('data_source') for r in measured] == fixed5['data_sources'] and
         [r.get('em') for r in measured] == fixed5['em'] and [r.get('f1') for r in measured] == fixed5['f1'] and
         after.get('weighted_em') == fixed['5']['weighted_em'] and after.get('weighted_f1') == fixed['5']['weighted_f1'],
         'FIXED5_SCALAR_ARTIFACT_SCORES_DIFFER')
    return dict(fixed512=fixed, final51713=result, final_identity=final_record, fixed5_artifact=after_record)


def attempt_id(row):
    context = row if 'attempt_index' in row else row.get('context', {})
    index, aid = context.get('attempt_index'), context.get('attempt_id')
    need(type(index) is int and index >= 1 and isinstance(aid, str) and
         re.fullmatch('[0-9a-f]{32}', aid) is not None, 'MALFORMED_ATTEMPT_IDENTITY')
    return index, aid


def no201(base, run, chain=None):
    from reproduction.easy2560_joint4.checkpoint_evidence import historical_origin
    current_updates = list(lines(base/'logs/actual-updates.jsonl'))
    if run.get('full_state_restore'):
        binding = run['full_state_restore']
        need(isinstance(binding, dict) and binding.get('kind') == 'easy2560_native4' and
             type(binding.get('completed_outer')) is int and 1 <= binding['completed_outer'] < 200,
             'CROSS_JOB_AUTHENTICATED_CHAIN_NOT_IMPLEMENTED')
        # Still fail on missing200 before archive/evaluation/source-proofs work.
        need([r.get('step') for r in current_updates] == list(range(binding['completed_outer']+1, 201)),
             'MISSING200_OR_EXTRA_COMPLETED_UPDATE')
        if chain is None:
            chain = restore_chain(base, run)
    updates = chained_lines(base, 'logs/actual-updates.jsonl', chain) if chain else current_updates
    need(all(type(r.get('step')) is int for r in updates) and
         [r.get('step') for r in updates] == list(range(1, 201)), 'MISSING200_OR_EXTRA_COMPLETED_UPDATE')
    need(all(r.get('run_id') == run['run_id'] for r in updates), 'FOREIGN_COMPLETED_UPDATE')
    attempts = chained_lines(base, 'logs/actual-attempts.jsonl', chain)
    need(attempts and [attempt_id(r)[0] for r in attempts] == list(range(1, len(attempts)+1)) and
         len({attempt_id(r)[1] for r in attempts}) == len(attempts), 'ATTEMPT_PREFIX_NOT_CONTIGUOUS')
    completed = writes = aborted = 0; successes = []
    allocation, allocation_record = small(base/'logs/native-allocation-admission.json')
    need(allocation.get('accepted') is True and allocation.get('run_id') == run['run_id'], 'NATIVE_ALLOCATION_IDENTITY_DIFFERS')
    job = allocation['slurm_job_id']
    for row in attempts:
        expected_job = historical_origin(chain, attempt_index=attempt_id(row)[0])[2] if chain and attempt_id(row)[0] <= chain['cutoff_attempt'] else job
        need(row.get('run_id') == run['run_id'] and row.get('slurm_job_id') == expected_job and
             row.get('all_ranks_agreed') is True and type(row.get('real_rank_count')) is int and
             row['real_rank_count'] == 4 and row.get('logical_update') == completed+1 and
             completed < 200, 'FOREIGN_OR_POST200_DRIVER_ATTEMPT')
        count, processed = row.get('optimizer_writes_this_attempt'), row.get('opportunities_processed')
        need(type(count) is int and type(processed) is int and 0 <= count <= processed <= 20 and
             processed >= 1 and row.get('optimizer_writes_before') == writes and
             row.get('optimizer_writes_after') == writes+count and row.get('complete_outer') is (count > 0) and
             row.get('scheduler_advances') == int(count > 0), 'DRIVER_WRITE_CHAIN_DIFFERS')
        completed += int(count > 0); writes += count; aborted += int(count == 0)
        need(row.get('completed_outer') == completed and row.get('cumulative_aborted_attempts') == aborted,
             'DRIVER_OUTER_CHAIN_DIFFERS')
        if count: successes.append(row)
    need(completed == 200 and attempts[-1]['complete_outer'] is True and len(successes) == 200,
         'FINAL_COMPLETE200_REQUIRED')
    for update, attempt in zip(updates, successes):
        need(attempt_id(update) == attempt_id(attempt) and update['step'] == attempt['completed_outer'] and
             update.get('optimizer_writes_before') == attempt['optimizer_writes_before'] and
             update.get('optimizer_writes_after') == attempt['optimizer_writes_after'] and
             update.get('actual_optimizer_writes') == attempt['optimizer_writes_this_attempt'],
             'UPDATE_ATTEMPT_BINDING_DIFFERS')
    rank_summaries = {}
    for rank in range(4):
        roles = {}
        for role in ('scheduler', 'optimizer'):
            current = 0; events = []; total_rows = 0
            def check_group(index, group):
                attempt = attempts[index-1]; complete = attempt['complete_outer']; terminal = group[-1]
                if role == 'scheduler':
                    need([r['kind'] for r in group] == (['BEGIN', 'COMMIT_PREPARED', 'COMPLETE'] if complete else
                         ['BEGIN', 'ABORTED_ATTEMPT']), 'INCOMPLETE_SCHEDULER_ATTEMPT')
                    need(terminal.get('writes_after') == attempt['optimizer_writes_after'] and
                         terminal.get('scheduler_last_epoch_after') == attempt['completed_outer'] and
                         terminal.get('scheduler_advances') == int(complete), 'SCHEDULER_TERMINAL_COUNTER_DIFFERS')
                else:
                    need(terminal['kind'] == ('OUTER_BUDGET' if complete else 'ABORTED_ATTEMPT') and
                         terminal.get('cumulative_writes') == attempt['optimizer_writes_after'] and
                         [r.get('write_count') for r in group if r['kind'] == 'OPTIMIZER_WRITE'] ==
                         list(range(attempt['optimizer_writes_before']+1, attempt['optimizer_writes_after']+1)),
                         'INCOMPLETE_WORKER_ATTEMPT')
                    need(all(r['kind'] not in ('OUTER_BUDGET', 'ABORTED_ATTEMPT') for r in group[:-1]),
                         'WORKER_CONTINUES_AFTER_TERMINAL')
            for row in chained_lines(base, f'logs/{role}-rank-{rank}.jsonl', chain):
                index, aid = attempt_id(row); context = row.get('context') or row
                need(1 <= index <= len(attempts) and aid == attempts[index-1]['attempt_id'] and
                     row.get('run_id') == run['run_id'] and row.get('rank') == rank and
                     context.get('logical_update') == attempts[index-1]['logical_update'] <= 200,
                     'FOREIGN_EXTRA_OR_POST200_RANK_ATTEMPT')
                if role == 'scheduler':
                    expected_job = historical_origin(chain, attempt_index=index)[2] if chain and index <= chain['cutoff_attempt'] else job
                    need(row.get('world_size') == 4 and row.get('slurm_job_id') == expected_job, 'SCHEDULER_NATIVE4_IDENTITY_DIFFERS')
                else:
                    need(row.get('kind') in ('POLICY_MONITOR', 'OPTIMIZER_WRITE', 'OPPORTUNITY',
                         'OUTER_BUDGET', 'ABORTED_ATTEMPT'), 'UNKNOWN_WORKER_EVENT')
                if index != current:
                    need(index == current+1, 'REORDERED_OR_MISSING_RANK_ATTEMPT')
                    if events: check_group(current, events)
                    current = index; events = []
                events.append(row); total_rows += 1
            need(current == len(attempts) and events, 'MISSING_FINAL_RANK_ATTEMPT')
            check_group(current, events)
            roles[role] = dict(rows=total_rows, terminal_attempt_index=current,
                               final_completed_outer=completed, cumulative_actual_writes=writes)
        rank_summaries[str(rank)] = roles
    return dict(actual_completed_updates=200, actual_attempt_count=len(attempts), no_update201=True,
                no_post200_driver_scheduler_worker_attempts=True, rank_count=4,
                cumulative_actual_optimizer_writes_per_rank=writes, ranks=rank_summaries,
                allocation=allocation_record,
                authenticated_history=chain['public'] if chain else None,
                scope='durable actual counter/attempt chain; this CPU reader does not independently authenticate physical GPU execution')


def selection_evidence(base, run, chain=None):
    """Verify every completed outer's saved selection/reuse evidence on CPU.

    Call only after ``no201`` and the restore chain have been authenticated.
    Historical selected traces remain bound to their original execution; a
    restore does not turn the accepted first five outers into new-job output.
    This is a final consumer, not a partial-run or training-startup check.
    """
    updates = chained_lines(base, 'logs/actual-updates.jsonl', chain)
    need([row.get('step') for row in updates] == list(range(1, 201)) and
         all(type(row.get('step')) is int and row.get('run_id') == run['run_id'] for row in updates),
         'SELECTION_FULL200_COMPLETED_UPDATES_REQUIRED')
    identities = [attempt_id(row) for row in updates]
    need(len(set(identities)) == len({index for index, _ in identities}) ==
         len({aid for _, aid in identities}) == 200,
         'SELECTION_DUPLICATE_COMPLETED_ATTEMPT')
    allocation, _ = small(base/'logs/native-allocation-admission.json')
    job = allocation.get('slurm_job_id')
    need(allocation.get('accepted') is True and allocation.get('run_id') == run['run_id'] and
         isinstance(job, str) and job, 'SELECTION_CURRENT_ALLOCATION_REQUIRED')
    if chain is not None:
        need(1 <= chain['boundary'] < 200 and chain['current_job'] == job and
             chain['source_run']['run_id'] == run['run_id'], 'SELECTION_RESTORE_CHAIN_DIFFERS')
    # Import after the full200 requirement so an incomplete run cannot cause
    # expensive selected-trace reads, let alone be labelled final acceptance.
    from reproduction.easy2560_joint4.selection_evidence import verify_attempt
    summaries = []
    for update in updates:
        historical = chain is not None and update['step'] <= chain['boundary']
        if historical:
            from reproduction.easy2560_joint4.checkpoint_evidence import historical_origin
            origin, origin_run, origin_job = historical_origin(chain, outer=update['step'])
        else:
            origin, origin_run, origin_job = base, run, job
        summaries.append(verify_attempt(Path(origin), origin_run, update, expected_job=origin_job))
    segments = []
    if chain is not None:
        origins = chain.get('segments') or [dict(first_outer=1, last_outer=chain['boundary'],
            base=chain['source_base'], run=chain['source_run'], job=chain['source_job'])]
        for origin in origins:
            segments.append(dict(first_outer=origin['first_outer'], last_outer=origin['last_outer'],
                completed_outers=origin['last_outer']-origin['first_outer']+1, base=str(origin['base']),
                run_id=origin['run']['run_id'], source_commit=origin['run']['source_commit'], slurm_job_id=origin['job']))
    first = chain['boundary']+1 if chain is not None else 1
    segments.append(dict(first_outer=first, last_outer=200, completed_outers=201-first,
        base=str(base), run_id=run['run_id'], source_commit=run['source_commit'], slurm_job_id=job))
    return dict(schema_version=1, status='EASY2560_FULL200_SELECTION_REUSE_EVIDENCE_ACCEPTED',
                verified_completed_outers=200, full_training_history_verified=True,
                source_segments=segments, per_outer=summaries,
                new_training_updates=0, gpu_allocations=0, model_reloads=0,
                scope='saved complete2560 candidate identity/order, strict-EM mixed-first selection and original selected trajectory/token/native-sampling-logprob reuse; CPU posthoc verification only')


def checkpoint_evidence(base, run, step, *, archive_root=None, chain=None, historical_update=None):
    """Authenticate accepted SHA receipts; tensor retention is not required.

    An optional HK archive root supplies small checkpoint metadata directly.
    Otherwise a sealed immutable sidecar can supply retired producer metadata.
    Neither route pretends to rehash absent remote tensor bodies.
    """
    if chain is not None and step <= chain['boundary']:
        from reproduction.easy2560_joint4.checkpoint_evidence import _historical_rows
        old_update = next(r for r in _historical_rows(chain, 'logs/actual-updates.jsonl') if r['step'] == step)
        from reproduction.easy2560_joint4.checkpoint_evidence import historical_origin
        origin, origin_run, origin_job = historical_origin(chain, outer=step)
        inherited = checkpoint_evidence(origin, origin_run, step,
                                        archive_root=archive_root, historical_update=old_update)
        inherited.update(inherited_original_checkpoint=True, source_base=str(origin),
                         source_commit=origin_run['source_commit'], source_job=origin_job)
        return inherited
    cp = Path(run['checkpoint_dir'])/'actor'/f'global_step_{step}'
    archive = Path(archive_root)/f'global_step_{step}' if archive_root is not None else None
    sidecar = base/'preparation/checkpoint-evidence'/f'global_step_{step}'
    supplement = None
    if sidecar.exists():
        from reproduction.easy2560_joint4.checkpoint_evidence import verify
        supplement = verify(sidecar)
        need(all(supplement.get(k) == v for k, v in dict(source_base=str(base), source_commit=run['source_commit'],
             run_id=run['run_id'], step=step).items()) and supplement.get('actual_cuda_restore_accepted') is False,
             'SIDECAR_RUN_OR_SCOPE_DIFFERS')
        allocation, _ = small(base/'logs/native-allocation-admission.json')
        need(supplement.get('source_job') == allocation.get('slurm_job_id'), 'SIDECAR_SOURCE_JOB_DIFFERS')
    source_root = base/'preparation/checkpoint-archives'/f'global_step_{step}'
    def choose(*candidates):
        for path in candidates:
            if path is not None and path.is_file(): return small(path)
        raise ValueError('EASY2560_FINAL_CPU_MISSING_CHECKPOINT_SMALL_EVIDENCE_' + str(step))
    snapshot, snapshot_record = choose(source_root/'source-snapshot.json',
        sidecar/'identity/source-snapshot.json' if supplement else None,
        archive/'source-snapshot.json' if archive else None)
    accepted, accepted_record = choose(source_root/'complete.json',
        sidecar/'identity/accepted-archive.json' if supplement else None,
        archive/'source-acceptance.json' if archive else None)
    complete, complete_record = choose(cp/'complete.json',
        sidecar/'identity/checkpoint-complete.json' if supplement else None,
        archive/'checkpoint/complete.json' if archive else None)
    state, state_record = choose(cp/'training_state/manifest.json',
        sidecar/'identity/training-state-manifest.json' if supplement else None,
        archive/'checkpoint/training_state/manifest.json' if archive else None)
    binding = dict(base=str(base), run_id=run['run_id'], source_commit=run['source_commit'], step=step)
    need(all(snapshot.get(k) == v and accepted.get(k) == v for k, v in binding.items()) and
         snapshot.get('training_world_size') == 4 and snapshot.get('question_probe_saved') is False,
         'CHECKPOINT_ARCHIVE_IDENTITY_DIFFERS')
    files = snapshot['files']; total = sum(row['bytes'] for row in files.values())
    if run['source_commit'] != REVIEWED_COMMIT:
        for name, sha in run['source_sha256'].items():
            need(files.get('source/' + name, {}).get('sha256') == sha,
                 'ARCHIVED_PUBLISHED_SOURCE_MAP_DIFFERS')
    need(accepted.get('status') == ARCHIVE_STATUS and accepted.get('fresh_destination_sha_verified') is True and
         accepted.get('destination_fsynced') is True and accepted.get('fresh_source_after_archive_sha_verified') is True and
         accepted.get('actual_cuda_restore_accepted') is False and accepted.get('snapshot_sha256') == snapshot_record['sha256'] and
         accepted.get('file_count') == len(files) and accepted.get('total_bytes') == snapshot.get('total_bytes') == total,
         'CHECKPOINT_ARCHIVE_SHA_RECEIPT_DIFFERS')
    need(complete.get('run_id') == run['run_id'] and complete.get('step') == step and
         complete.get('finite_all_saved_tensors') is True and Path(complete['checkpoint']) == cp and
         state.get('completed_outer') == step and state.get('question_probe') is None and
         sorted(row['rank'] for row in state['worker_files']) == list(range(4)), 'CHECKPOINT_NATIVE4_STATE_DIFFERS')
    required = {'training_state/rank-%d.pt' % r for r in range(4)} | {'training_state/driver.pt', 'training_state/manifest.json'}
    need(required.issubset(complete['files_sha256']) and
         any(name.endswith('.safetensors') for name in complete['files_sha256']), 'FULL_NATIVE4_CHECKPOINT_MEMBERS_MISSING')
    for name, sha in complete['files_sha256'].items():
        need(files.get('checkpoint/'+name, {}).get('sha256') == sha, 'ARCHIVE_PRODUCER_FILE_SHA_DIFFERS')
    need(files['checkpoint/complete.json']['sha256'] == complete_record['sha256'] and
         complete['files_sha256']['training_state/manifest.json'] == state_record['sha256'] and
         files['metadata/run-manifest.json']['sha256'] == small(base/'logs/run-manifest.json')[1]['sha256'],
         'ARCHIVE_SMALL_METADATA_SHA_DIFFERS')
    for row in state['worker_files']:
        need(row.get('run_id') == run['run_id'] and row['sha256'] == complete['files_sha256'][f'training_state/rank-{row["rank"]}.pt'],
             'ARCHIVE_RANK_STATE_SHA_DIFFERS')
    actual = ([historical_update] if historical_update is not None else
              [r for r in lines(base/'logs/actual-updates.jsonl') if r.get('step') == step])
    need(len(actual) == 1 and snapshot.get('completed_update') == actual[0], 'ARCHIVE_COMPLETED_OUTER_DIFFERS')
    return dict(step=step, source_snapshot=snapshot_record, accepted_archive=accepted_record,
                checkpoint_complete=complete_record, training_state_manifest=state_record,
                source_checkpoint=str(cp), archive_files=len(files), archive_bytes=total,
                existing_full_file_sha_acceptance_authenticated=True,
                checkpoint_large_files_rehashed_by_this_consumer=False,
                local_tensor_presence_required=False, actual_cuda_restore_accepted=False,
                sealed_supplement=supplement)


def verify_final(base, *, archive_root=None, load_canonical=canonical, training_commit=REVIEWED_COMMIT,
                 final_restore_proof=None):
    base = Path(base).resolve(); run, run_record = small(base/'logs/run-manifest.json')
    need(run.get('source_commit') == training_commit and run.get('easy2560_selection') is True and
         run.get('training_world_size') == 4 and run.get('data_selection_mode') == 'mixed_first' and
         run.get('question_probe') is None and run.get('actor_logical_micro_batch_per_rank') == 4 and
         run.get('actor_micro_batch_per_rank') in (32, 16, 8, 4) and run.get('all_correct_virtual_zero') is False,
         'REVIEWED_EASY_NATIVE4_RUN_REQUIRED')
    # Check actual completion first. Missing200 cannot be hidden by optional
    # metadata or a final score copied from a different run.
    chain = None
    if run.get('full_state_restore'):
        binding = run['full_state_restore']
        need(isinstance(binding, dict) and binding.get('kind') == 'easy2560_native4' and
             type(binding.get('completed_outer')) is int and 1 <= binding['completed_outer'] < 200,
             'CROSS_JOB_AUTHENTICATED_CHAIN_NOT_IMPLEMENTED')
        need([row.get('step') for row in lines(base/'logs/actual-updates.jsonl')] == list(range(binding['completed_outer']+1, 201)),
             'MISSING200_OR_EXTRA_COMPLETED_UPDATE')
        chain = restore_chain(base, run)
    progress = no201(base, run, chain=chain)
    selection = selection_evidence(base, run, chain=chain)
    source_binding = published_source_binding(base, run, training_commit)
    evaluation = evaluations(base, run, load_canonical, chain=chain)
    checkpoints = {str(step): checkpoint_evidence(base, run, step, archive_root=archive_root, chain=chain) for step in STEPS}
    actual_restore = None
    if final_restore_proof is not None:
        from reproduction.easy2560_joint4.final_restore_acceptance import verify_actual_final_restore
        actual_restore = verify_actual_final_restore(final_restore_proof, base=base, run=run,
            checkpoint_summary=checkpoints['200'], progress=progress)
        need(actual_restore.get('actual_cuda_restore_accepted') is True, 'ACTUAL_FINAL_RESTORE_NOT_ACCEPTED')
    return dict(schema_version=1, status=STATUS, cpu_final_accepted=True, base=str(base),
                run_id=run['run_id'], source_commit=run['source_commit'], run_manifest=run_record,
                source_publication_binding=source_binding,
                progress=progress, selection_evidence=selection, evaluations=evaluation, checkpoints=checkpoints,
                authenticated_history=chain['public'] if chain else None,
                single_original_job_prefix=chain is None, inherited_history_assembled=chain is not None,
                new_training_updates=0, gpu_allocations=0, model_reloads=0, actual_restore_performed=False,
                actual_cuda_restore_accepted=actual_restore is not None, actual_final_restore=actual_restore,
                overall_goal_complete=False, checked_unix=time.time(),
                scope='full200 saved selection/reuse evidence, fixed five512 evaluations, ordered exact native51713 EM/F1, durable no201 counters and accepted checkpoint archive SHA evidence; actual recovery and physical execution acceptance remain separate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--archive-root', help='optional remote directory containing global_step_N archives')
    parser.add_argument('--training-commit', default=REVIEWED_COMMIT,
                        help='explicit full40 training source; a non93 commit requires publication/run-source/run-map binding')
    parser.add_argument('--output', help='new small JSON report; existing reports are never overwritten')
    parser.add_argument('--final-restore-proof', help='optional completed actual all4 CP200 restore-only proof; never starts it')
    args = parser.parse_args()
    result = verify_final(args.base, archive_root=args.archive_root, training_commit=args.training_commit,
                          final_restore_proof=args.final_restore_proof)
    raw = encoded(result)
    if args.output:
        with Path(args.output).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    print(raw.decode(), end='', flush=True)


if __name__ == '__main__':
    main()
