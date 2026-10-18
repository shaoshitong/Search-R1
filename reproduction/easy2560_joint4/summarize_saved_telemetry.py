"""Read existing native4 Easy scalar metadata; never load tensors or change a run.

The report preserves source/job segments. GPU memory is device occupancy sampled
inside the scheduler's actor window, not instantaneous or actor-only allocation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re


MAX_JSON = 32 * 1024**2
MAX_LINE = 2 * 1024**2


def need(condition, message):
    if not condition: raise ValueError('EASY2560_TELEMETRY_' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def number(value, *, integer=False):
    need(type(value) in ((int,) if integer else (int, float)) and math.isfinite(value) and value >= 0,
         'INVALID_NONNEGATIVE_NUMBER')
    return value


def small(path):
    path = Path(path)
    with path.open('rb') as stream: raw = stream.read(MAX_JSON + 1)
    need(len(raw) <= MAX_JSON, 'SMALL_JSON_LIMIT')
    return json.loads(raw), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def ledger(path, *, stop=None):
    """Read one initial-size complete prefix; ignore an incomplete final row.

    The digest covers only the returned complete prefix, not the whole active
    file or any later append. No EOF read chases a concurrently growing writer.
    """
    path = Path(path); rows = []; count = 0; sha = hashlib.sha256()
    with path.open('rb') as stream:
        observed = os.fstat(stream.fileno()); remaining = observed.st_size
        while remaining:
            raw = stream.readline(min(MAX_LINE + 1, remaining)); remaining -= len(raw)
            if not raw or not raw.endswith(b'\n'): break
            need(len(raw) <= MAX_LINE and raw.strip(), 'INVALID_COMPLETE_LEDGER_ROW')
            row = json.loads(raw); rows.append(row); sha.update(raw); count += len(raw)
            if stop is not None and stop(row): break
    return rows, dict(path=str(path), bytes=count, sha256=sha.hexdigest(),
                      complete_prefix_only=True, whole_active_file_stability_claimed=False,
                      observed_source_bytes=observed.st_size,
                      ignored_or_unread_suffix_bytes=observed.st_size-count)


def identity(row):
    value = row if 'attempt_index' in row else row.get('context', {})
    return value.get('attempt_index'), value.get('attempt_id')


def segment(base, through):
    base = Path(base).resolve(); run, manifest = small(base/'logs/run-manifest.json')
    job_raw = (base/'job.id').read_bytes(); job = job_raw.decode().strip()
    need(re.fullmatch('[1-9][0-9]*', job) and
         isinstance(run.get('run_id'), str) and re.fullmatch('[0-9a-f]{32}', run['run_id']) and
         isinstance(run.get('source_commit'), str) and re.fullmatch('[0-9a-f]{40}', run['source_commit']),
         'RUN_OR_JOB_IDENTITY_INVALID')
    need(run.get('easy2560_selection') is True and run.get('training_world_size') == 4 and
         run.get('actor_logical_micro_batch_per_rank') == 4 and run.get('question_probe') is None and
         run.get('data_selection_mode') == 'mixed_first' and run.get('all_correct_virtual_zero') is False and
         run.get('selection_candidate_questions') == 2560 and run.get('selection_selected_questions') == 512,
         'NATIVE4_EASY_SCOPE_REQUIRED')
    need('slurm_job_id' not in run or run['slurm_job_id'] == job, 'MANIFEST_JOB_DIFFERS')
    allocation, allocation_record = small(base/'logs/native-allocation-admission.json')
    uuids = allocation.get('actual_cuda_runtime_uuids', [])
    need(allocation.get('accepted') is True and allocation.get('run_id') == run['run_id'] and
         allocation.get('slurm_job_id') == job and len(uuids) == len(set(uuids)) == 4,
         'NATIVE4_ALLOCATION_IDENTITY_DIFFERS')
    source_map = run.get('source_sha256')
    need(isinstance(source_map, dict) and source_map, 'SOURCE_MAP_REQUIRED')
    source_sha = hashlib.sha256(canonical(source_map)).hexdigest()
    updates, updates_record = ledger(base/'logs/actual-updates.jsonl', stop=lambda r: r.get('step') == through)
    offset = (run.get('full_state_restore') or {}).get('completed_outer', 0)
    need(type(offset) is int and 0 <= offset < 200, 'INVALID_RESTORED_OUTER')
    need([row.get('step') for row in updates] == list(range(offset+1, offset+len(updates)+1)),
         'SEGMENT_COMPLETED_OUTERS_NOT_CONTIGUOUS')
    need(all(row.get('run_id') == run['run_id'] for row in updates), 'FOREIGN_UPDATE_RUN')
    attempts = []; attempts_record = None
    attempt_path = base/'logs/actual-attempts.jsonl'
    if attempt_path.is_file():
        relevant = [r for r in updates if r['step'] <= through]
        cutoff = relevant[-1]['attempt_index'] if relevant else None
        attempts, attempts_record = ledger(attempt_path,
            stop=lambda r: cutoff is not None and r.get('attempt_index') == cutoff)
    scheduler = {}; records = {}
    for rank in range(4):
        relative = 'logs/scheduler-rank-%d.jsonl' % rank
        scheduler[rank], records[str(rank)] = ledger(base/relative,
            stop=lambda r: r.get('logical_update') == through and r.get('kind') == 'COMPLETE')
    samples, sample_record = ledger(base/('logs/gpu-samples-' + run['run_id'] + '.jsonl'))
    for sample in samples:
        need(sample.get('role') == 'actual_common8_gpu_occupancy' and sample.get('run_id') == run['run_id'] and
             sample.get('slurm_job_id') == job and sample.get('source_map_sha256') == source_sha,
             'FOREIGN_GPU_SAMPLE_IDENTITY')
        need(len(sample.get('gpus', [])) == 4 and {g.get('uuid') for g in sample['gpus']} == set(uuids),
             'GPU_SAMPLE_UUIDS_DIFFER')
        number(sample['query_started_unix']); number(sample['unix'])
        need(sample['query_started_unix'] <= sample['unix'], 'GPU_SAMPLE_TIME_REVERSED')
    return dict(base=base, run=run, job=job, uuids=uuids, updates=updates, attempts=attempts,
        scheduler=scheduler, samples=samples,
        evidence=dict(run_manifest=manifest, allocation=allocation_record,
            job_id=dict(path=str(base/'job.id'), bytes=len(job_raw), sha256=hashlib.sha256(job_raw).hexdigest()),
            actual_updates_complete_prefix=updates_record, scheduler_complete_prefixes=records,
            actual_attempts_complete_prefix=attempts_record, gpu_samples_complete_prefix=sample_record))


def write_activity(part, update):
    """Expose saved counts verbatim; an absent field never means zero."""
    values = {}; sources = {}; metrics = update.get('metrics', {})
    for field, metric in (('opportunities_processed', 'actor/update_opportunities_this_attempt'),
                          ('skipped_opportunities', 'actor/skipped_update_opportunities'),
                          ('opportunities_unattempted', 'actor/unattempted_update_opportunities')):
        if field in update:
            values[field] = number(update[field], integer=True); sources[field] = 'actual-updates.'+field
        elif metric in metrics:
            value = number(metrics[metric])
            need(float(value).is_integer(), 'NONINTEGER_OPPORTUNITY_COUNT')
            values[field] = int(value); sources[field] = 'actual-updates.metrics.'+metric
        else:
            values[field] = None; sources[field] = 'unavailable'
    matches = [r for r in part['attempts'] if identity(r) == (update['attempt_index'], update['attempt_id'])]
    need(len(matches) <= 1, 'DUPLICATE_MATCHING_ATTEMPT')
    if matches:
        attempt = matches[0]
        need(attempt.get('run_id') == part['run']['run_id'] and attempt.get('slurm_job_id') == part['job'] and
             attempt.get('logical_update') == update['step'] and
             attempt.get('optimizer_writes_this_attempt') == update['actual_optimizer_writes'] and
             attempt.get('optimizer_writes_before') == update['optimizer_writes_before'] and
             attempt.get('optimizer_writes_after') == update['optimizer_writes_after'], 'ATTEMPT_WRITE_IDENTITY_DIFFERS')
    for field in ('optimizer_calls_this_attempt', 'optimizer_rollbacks_this_attempt'):
        values[field] = number(matches[0][field], integer=True) if matches and field in matches[0] else None
        sources[field] = 'actual-attempts.'+field if values[field] is not None else 'unavailable'
    values.update(write_activity_sources=sources,
        optimizer_call_counts_scope='matching chosen attempt only; excludes earlier failed attempts',
        skipped_opportunities_definition='producer field/metric; actor skipped_update_opportunities is20 minus retained writes, including unattempted opportunities',
        matching_actual_attempt_available=bool(matches))
    return values


def selection(base, run, update):
    attempt = update['attempt_id']; need(re.fullmatch('[0-9a-f]{32}', attempt), 'INVALID_ATTEMPT_ID')
    audit, record = small(base/'artifacts/mixed-selection'/(attempt+'.json'))
    need(audit.get('step') == update['step'] and audit.get('source_commit') == run['source_commit'] and
         audit.get('policy_snapshot_id') == run['run_id']+':'+str(update['step'])+':policy-'+attempt and
         audit.get('fixed_candidate_budget') is True and audit.get('responses_per_group') == 5,
         'SELECTION_AUDIT_RUN_OR_ATTEMPT_DIFFERS')
    groups = audit['groups']; need(len(groups) == audit['attempted_candidates'] == 2560, 'CANDIDATE_COUNT_DIFFERS')
    need([g['ordinal'] for g in groups] == list(range(2560)), 'CANDIDATE_ORDINALS_DIFFER')
    correct = mixed = chosen_correct = chosen_mixed = fallback = chosen = 0
    for group in groups:
        labels = group['strict_correctness']
        need(len(labels) == 5 and all(type(v) is int and v in (0,1) for v in labels) and
             type(group['selected']) is bool, 'INVALID_STRICT_LABELS')
        count = sum(labels); category = 'all_correct' if count == 5 else 'all_wrong' if count == 0 else 'mixed'
        need(group['category'] == category, 'CANDIDATE_CATEGORY_DIFFERS')
        correct += count; mixed += int(category == 'mixed')
        if group['selected']:
            need(category != 'all_wrong', 'SELECTED_ALL_WRONG_GROUP')
            chosen += 1; chosen_correct += count; chosen_mixed += int(category == 'mixed')
            fallback += int(category == 'all_correct')
    need(chosen == audit['selected_groups'] == 512 and chosen_mixed == audit['selected_mixed'] and
         fallback == audit['selected_all_correct_fallback'], 'SELECTED_COUNTS_DIFFER')
    retries = number(audit.get('pool_retry', 0), integer=True)
    discarded = audit.get('discarded_insufficient_pools', [])
    need(len(discarded) == retries, 'RETRY_POOL_COUNT_DIFFERS')
    raw = {key:number(audit[key], integer=True) for key in
           ('raw_generated_policy_tokens', 'raw_actual_retrieval_calls', 'raw_generated_trajectories')}
    expected_trajectories = len(groups)*5 + sum(number(p['raw_generated_trajectories'], integer=True) for p in discarded)
    need(raw['raw_generated_trajectories'] == expected_trajectories, 'RAW_TRAJECTORY_RETRY_SCOPE_DIFFERS')
    return dict(selected_groups=chosen, selected_mixed_groups=chosen_mixed, selected_all_correct_fallback_groups=fallback,
        selected_mixed_fraction=chosen_mixed/chosen, selected_all_correct_fallback_fraction=fallback/chosen,
        selected_training_em=dict(correct_trajectories=chosen_correct, trajectories=chosen*5,
                                  fraction=chosen_correct/(chosen*5), held_out_evaluation=False),
        successful_candidate_pool=dict(groups=len(groups), trajectories=len(groups)*5, mixed_groups=mixed,
            mixed_fraction=mixed/len(groups), strict_correct_trajectories=correct, strict_em=correct/(len(groups)*5)),
        generated_work=dict(kept_model_tokens=raw['raw_generated_policy_tokens'],
            logical_retrieval_calls=raw['raw_actual_retrieval_calls'], trajectories=raw['raw_generated_trajectories'],
            discarded_insufficient_pool_count=retries, scope='successful candidate pool plus discarded insufficient pools',
            tokens_exclude_retrieved_and_correction_text=True, original_engine_generated_token_total=None,
            http_transport_attempt_count=None),
        candidate_fill_seconds=number(audit['candidate_fill_seconds']),
        candidate_generation_timings=audit.get('candidate_generation_timings', {}), audit_evidence=record)


def actor_memory(part, update):
    starts = []; ends = []; ranks = []; terminals = []
    for rank, rows in part['scheduler'].items():
        events = [r for r in rows if identity(r) == (update['attempt_index'], update['attempt_id'])]
        for row in events:
            need(row.get('run_id') == part['run']['run_id'] and row.get('slurm_job_id') == part['job'] and
                 row.get('rank') == rank and row.get('world_size') == 4 and
                 row.get('logical_update') == update['step'], 'SCHEDULER_ATTEMPT_IDENTITY_DIFFERS')
        terminal = ('ABORTED_ATTEMPT' if update['actual_optimizer_writes'] == 0 and
                    any(r.get('kind') == 'ABORTED_ATTEMPT' for r in events) else 'COMPLETE')
        begin = [r for r in events if r.get('kind') == 'BEGIN']; complete = [r for r in events if r.get('kind') == terminal]
        need(len(begin) == len(complete) == 1, 'MISSING_OR_DUPLICATE_SCHEDULER_WINDOW')
        start = number(begin[0]['observed_unix']); end = number(complete[0]['observed_unix'])
        need(start <= end and complete[0]['writes_after'] == update['optimizer_writes_after'], 'SCHEDULER_WINDOW_DIFFERS')
        starts.append(start); ends.append(end); terminals.append(terminal)
        ranks.append(dict(rank=rank, begin_unix=start, terminal_unix=end,
                          complete_unix=end if terminal == 'COMPLETE' else None, terminal_kind=terminal))
    need(len(set(terminals)) == 1, 'SCHEDULER_TERMINAL_KINDS_DIFFER')
    start = min(starts); end = max(ends)
    within = [r for r in part['samples'] if start <= r['query_started_unix'] and r['unix'] <= end]
    peaks = []
    for uuid in part['uuids']:
        values = [number(g['used_mib'], integer=True) for s in within for g in s['gpus'] if g['uuid'] == uuid]
        peaks.append(dict(uuid=uuid, sample_count=len(values), sampled_peak_used_mib=max(values) if values else None))
    return dict(actor_window=dict(begin_unix=start, complete_unix=end if terminals[0] == 'COMPLETE' else None,
                                 terminal_unix=end, terminal_kind=terminals[0], elapsed_seconds=end-start, ranks=ranks,
                                 definition='min all4 scheduler BEGIN to max all4 scheduler '+terminals[0]),
        actor_update_timer_seconds=None, actor_update_timer_available_from_this_metadata=False,
        gpu_memory=dict(sample_count=len(within), per_gpu=peaks,
            window='actor scheduler window only; complete query intervals contained within window',
            meaning='sampled device occupancy including colocated processes; not actor-only allocation',
            instantaneous_peak_claimed=False, whole_outer_peak_claimed=False))


def summarize(bases, through_outer):
    need(type(through_outer) is int and 1 <= through_outer <= 200, 'THROUGH_OUTER_OUT_OF_SCOPE')
    parts = [segment(base, through_outer) for base in bases]
    need(parts and len({p['base'] for p in parts}) == len(parts), 'EXPLICIT_UNIQUE_BASES_REQUIRED')
    run_ids = {p['run']['run_id'] for p in parts}; need(len(run_ids) == 1, 'BASE_RUN_IDS_DIFFER')
    by_base = {str(p['base']):p for p in parts}
    for part in parts:
        restore = part['run'].get('full_state_restore')
        if restore:
            parent = by_base.get(restore.get('source_base'))
            need(restore.get('kind') == 'easy2560_native4' and
                 type(restore.get('completed_outer')) is int and 1 <= restore['completed_outer'] < 200 and parent is not None and
                 parent['job'] == restore.get('source_job') and
                 parent['run']['source_commit'] == restore.get('source_commit'), 'EXPLICIT_PARENT_SEGMENT_REQUIRED')
    ends = {}
    for part in parts:
        restore = part['run'].get('full_state_restore')
        if restore:
            key = restore['source_base']; boundary = restore['completed_outer']
            need(key not in ends or ends[key] == boundary, 'AMBIGUOUS_PARENT_SUFFIX')
            ends[key] = boundary
    outers = {}; segments = []
    for part in parts:
        run = part['run']; steps = []
        for update in part['updates']:
            step = update['step']
            if step > min(through_outer, ends.get(str(part['base']), through_outer)): continue
            need(step not in outers, 'DUPLICATE_OUTER_ACROSS_BASES')
            writes = number(update['actual_optimizer_writes'], integer=True)
            before = number(update['optimizer_writes_before'], integer=True); after = number(update['optimizer_writes_after'], integer=True)
            need(writes == after-before and type(update.get('full_budget_complete')) is bool,
                 'ACTUAL_WRITES_DIFFERS')
            reward = update.get('metrics', {}).get('critic/rewards/mean')
            need(type(reward) in (int,float) and math.isfinite(reward), 'REWARD_SCALAR_MISSING_OR_NONFINITE')
            row = dict(step=step, run_id=run['run_id'], source_base=str(part['base']), source_commit=run['source_commit'],
                source_job=part['job'], attempt_id=update['attempt_id'], attempt_index=update['attempt_index'],
                retained_adam_writes_per_rank=writes, cumulative_retained_adam_writes_per_rank=after,
                full_budget_complete=update['full_budget_complete'], update_status=update.get('update_status'),
                selected_sequence_reward_mean=reward,
                reward_definition='sum token_level_rewards per selected trajectory, averaged over selected512x5; not EM',
                **write_activity(part, update), **selection(part['base'], run, update), **actor_memory(part, update))
            row['exact_full_outer_walltime_seconds'] = None
            outers[step] = row; steps.append(step)
        segments.append(dict(base=str(part['base']), run_id=run['run_id'], source_commit=run['source_commit'],
            source_job=part['job'], completed_outers_in_report=steps,
            restored_completed_outer=(run.get('full_state_restore') or {}).get('completed_outer', 0), evidence=part['evidence']))
    missing = sorted(set(range(1,through_outer+1))-set(outers))
    ordered = [outers[step] for step in sorted(outers)]
    for previous, current in zip(ordered, ordered[1:]):
        if current['step'] == previous['step']+1:
            need(current['cumulative_retained_adam_writes_per_rank']-current['retained_adam_writes_per_rank'] ==
                 previous['cumulative_retained_adam_writes_per_rank'], 'CROSS_OUTER_WRITE_CHAIN_DIFFERS')
    return dict(schema_version=1, role='easy2560_saved_metadata_telemetry_summary', run_id=next(iter(run_ids)),
        through_outer=through_outer, missing_outers=missing, segments=segments, outers=ordered,
        read_only=True, training_or_restore_performed=False, actual200_acceptance_claimed=False,
        caveats=['Selected EM is training-sample EM under the sampling policy, not held-out accuracy.',
                 'Candidate ratios use the successful2560 pool; generated-work totals include discarded insufficient pools.',
                 'Kept model tokens exclude retrieval/corrections and exclude generated tokens cut after terminal markup/EOS.',
                 'Retrieval counts are logical searches, not HTTP retry/transport attempts.',
                 'GPU memory is sampled device occupancy within the actor scheduler window only.',
                 'Candidate fill is recorded exactly; actor scheduler-window elapsed is not the actor timer or exact full-outer walltime.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', action='append', required=True)
    parser.add_argument('--through-outer', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.base, args.through_outer), sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__': main()
