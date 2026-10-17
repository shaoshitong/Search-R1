"""Read-only Easy2560 final CPU consumer for the immutable 93eda66 producer.

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


def small(path):
    path = Path(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        need(before.st_size <= MAX_SMALL, 'SMALL_EVIDENCE_LIMIT')
        raw = stream.read(MAX_SMALL + 1)
        after = os.fstat(stream.fileno())
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    need(identity(before) == identity(after) == identity(path.stat()), 'EVIDENCE_CHANGED_DURING_READ')
    return decode(raw), dict(path=str(path), bytes=len(raw), sha256=digest(raw))


def lines(path):
    """Unlike an in-flight prefix seal, final acceptance rejects partial tails."""
    path = Path(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for raw in iter(lambda: stream.readline(MAX_LINE + 1), b''):
            need(raw.endswith(b'\n') and len(raw) <= MAX_LINE, 'PARTIAL_OR_OVERSIZED_FINAL_LEDGER')
            need(raw.strip(), 'BLANK_FINAL_LEDGER_ROW')
            yield decode(raw)
        after = os.fstat(stream.fileno())
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) ==
         (path.stat().st_dev, path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns),
         'FINAL_LEDGER_CHANGED_DURING_READ')


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


def evaluations(base, run, load_canonical=canonical):
    fixed_identity = load_canonical(base, 'validation'); final_identity = load_canonical(base, 'final')
    rows = list(lines(base/'logs/validation-identities.jsonl'))
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


def no201(base, run):
    updates = list(lines(base/'logs/actual-updates.jsonl'))
    need(all(type(r.get('step')) is int for r in updates) and
         [r.get('step') for r in updates] == list(range(1, 201)), 'MISSING200_OR_EXTRA_COMPLETED_UPDATE')
    need(all(r.get('run_id') == run['run_id'] for r in updates), 'FOREIGN_COMPLETED_UPDATE')
    attempts = list(lines(base/'logs/actual-attempts.jsonl'))
    need(attempts and [attempt_id(r)[0] for r in attempts] == list(range(1, len(attempts)+1)) and
         len({attempt_id(r)[1] for r in attempts}) == len(attempts), 'ATTEMPT_PREFIX_NOT_CONTIGUOUS')
    completed = writes = aborted = 0; successes = []
    allocation, allocation_record = small(base/'logs/native-allocation-admission.json')
    need(allocation.get('accepted') is True and allocation.get('run_id') == run['run_id'], 'NATIVE_ALLOCATION_IDENTITY_DIFFERS')
    job = allocation['slurm_job_id']
    for row in attempts:
        need(row.get('run_id') == run['run_id'] and row.get('slurm_job_id') == job and
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
            for row in lines(base/'logs'/f'{role}-rank-{rank}.jsonl'):
                index, aid = attempt_id(row); context = row.get('context') or row
                need(1 <= index <= len(attempts) and aid == attempts[index-1]['attempt_id'] and
                     row.get('run_id') == run['run_id'] and row.get('rank') == rank and
                     context.get('logical_update') == attempts[index-1]['logical_update'] <= 200,
                     'FOREIGN_EXTRA_OR_POST200_RANK_ATTEMPT')
                if role == 'scheduler':
                    need(row.get('world_size') == 4 and row.get('slurm_job_id') == job, 'SCHEDULER_NATIVE4_IDENTITY_DIFFERS')
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
                scope='durable actual counter/attempt chain; this CPU reader does not independently authenticate physical GPU execution')


def checkpoint_evidence(base, run, step, *, archive_root=None):
    """Authenticate accepted SHA receipts; tensor retention is not required.

    An optional HK archive root supplies small checkpoint metadata directly.
    Otherwise a sealed immutable sidecar can supply retired producer metadata.
    Neither route pretends to rehash absent remote tensor bodies.
    """
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
    actual = [r for r in lines(base/'logs/actual-updates.jsonl') if r.get('step') == step]
    need(len(actual) == 1 and snapshot.get('completed_update') == actual[0], 'ARCHIVE_COMPLETED_OUTER_DIFFERS')
    return dict(step=step, source_snapshot=snapshot_record, accepted_archive=accepted_record,
                checkpoint_complete=complete_record, training_state_manifest=state_record,
                source_checkpoint=str(cp), archive_files=len(files), archive_bytes=total,
                existing_full_file_sha_acceptance_authenticated=True,
                checkpoint_large_files_rehashed_by_this_consumer=False,
                local_tensor_presence_required=False, actual_cuda_restore_accepted=False,
                sealed_supplement=supplement)


def verify_final(base, *, archive_root=None, load_canonical=canonical):
    base = Path(base).resolve(); run, run_record = small(base/'logs/run-manifest.json')
    need(run.get('source_commit') == REVIEWED_COMMIT and run.get('easy2560_selection') is True and
         run.get('training_world_size') == 4 and run.get('data_selection_mode') == 'mixed_first' and
         run.get('question_probe') is None and run.get('actor_logical_micro_batch_per_rank') == 4 and
         run.get('actor_micro_batch_per_rank') in (32, 16, 8, 4) and run.get('all_correct_virtual_zero') is False,
         'REVIEWED_EASY_NATIVE4_RUN_REQUIRED')
    need(not run.get('full_state_restore'), 'CROSS_JOB_AUTHENTICATED_CHAIN_NOT_IMPLEMENTED')
    # Check actual completion first. Missing200 cannot be hidden by optional
    # metadata or a final score copied from a different run.
    progress = no201(base, run)
    evaluation = evaluations(base, run, load_canonical)
    checkpoints = {str(step): checkpoint_evidence(base, run, step, archive_root=archive_root) for step in STEPS}
    return dict(schema_version=1, status=STATUS, cpu_final_accepted=True, base=str(base),
                run_id=run['run_id'], source_commit=run['source_commit'], run_manifest=run_record,
                progress=progress, evaluations=evaluation, checkpoints=checkpoints,
                new_training_updates=0, gpu_allocations=0, model_reloads=0, actual_restore_performed=False,
                actual_cuda_restore_accepted=False, overall_goal_complete=False, checked_unix=time.time(),
                scope='fixed five512 evaluations, ordered exact native51713 EM/F1, durable no201 counters and accepted checkpoint archive SHA evidence; actual recovery and physical execution acceptance remain separate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--archive-root', help='optional remote directory containing global_step_N archives')
    parser.add_argument('--output', help='new small JSON report; existing reports are never overwritten')
    args = parser.parse_args()
    result = verify_final(args.base, archive_root=args.archive_root)
    raw = encoded(result)
    if args.output:
        with Path(args.output).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    print(raw.decode(), end='', flush=True)


if __name__ == '__main__':
    main()
