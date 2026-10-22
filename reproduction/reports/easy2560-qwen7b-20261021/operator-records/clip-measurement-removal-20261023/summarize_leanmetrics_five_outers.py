"""Summarize the five actual followup observations; local JSON only."""
import json
from pathlib import Path
import re
import statistics
import sys

out = Path(sys.argv[1])
evidence = Path(sys.argv[2])
names = {
    61: 'leanmetrics-transition-live-1792700011358230000.json',
    62: 'leanmetrics-transition-live-1792701811110677000.json',
    63: 'leanmetrics-transition-live-1792703570078400000.json',
    64: 'leanmetrics-transition-live-1792705334756256000.json',
    65: 'leanmetrics-transition-live-1792707200337292000.json',
}

def read(name):
    return json.loads(json.loads((out / name).read_text())['stdout'])

latest = read(names[65])
starts = {v['step']: v['first_before_generation'] for v in latest['candidate_progress']}
starts[66] = max(v['mtime'] for v in latest['candidate_start_boundaries'])
selection = {v['step']: v for v in latest['selection_summary']}
rows = []
actor_keys = ('logical_update', 'actual_optimizer_writes', 'actual_flash3_training_executed',
              'compact_gqa_enabled', 'forward_success_returns', 'kernel_backward_success_returns',
              'qkv_gradient_returns')
for step, name in names.items():
    observed = read(name)
    metrics = {k: float(v) for k, v in re.findall(
        r'([\w/]+):(-?\d+\.\d+)', observed['log_tails']['training']['tail'])}
    rows.append(dict(
        step=step, full_outer_seconds=starts[step + 1] - starts[step],
        candidate_start_mtime=starts[step], next_candidate_start_mtime=starts[step + 1],
        candidate_fill_seconds=selection[step]['candidate_fill_seconds'],
        actor_update_seconds=metrics['timing_s/update_actor'], adv_seconds=metrics['timing_s/adv'],
        save_checkpoint_seconds=metrics['timing_s/save_checkpoint'],
        post_candidate_step_seconds=metrics['timing_s/step'],
        training_em_rounded=metrics['train/perf/em'],
        training_reward_mean_rounded=metrics['critic/rewards/mean'],
        selected_mixed=selection[step]['selected_mixed'],
        selected_mixed_fraction=selection[step]['selected_mixed'] / 512,
        raw_generated_policy_tokens=selection[step]['raw_generated_policy_tokens'],
        raw_actual_retrieval_calls=selection[step]['raw_actual_retrieval_calls'],
        actual_optimizer_budgets={k: next(z for z in v['completed_outer_budgets']
            if z['logical_update'] == step) for k, v in observed['optimizer_ranks'].items()},
        last_write_after_measured={k: v['latest_actual_write']['gradient_norm_after_measured']
            for k, v in observed['optimizer_ranks'].items()},
        scheduler_complete={k: next(z for z in v if z['kind'] == 'COMPLETE'
            and z['logical_update'] == step) for k, v in observed['scheduler_ranks'].items()},
        actor_flash3={k: {key: v['latest'].get(key) for key in actor_keys}
            for k, v in observed['actor_flash3_ranks'].items()},
        candidate_phases=next(v for v in observed['candidate_phase_observations'] if v['step'] == step),
        checkpoint_seal=observed['documents']['preparation/rolling-checkpoints/latest.json']['value'],
        evidence=name))

whole = statistics.mean(v['full_outer_seconds'] for v in rows)
actor = statistics.mean(v['actor_update_seconds'] for v in rows)
candidate = statistics.mean(v['candidate_fill_seconds'] for v in rows)
summary = dict(
    schema='easy7b_leanmetrics_actual_five_complete_outers_v1', job='750029',
    runtime_commit='2d5183455551145c52d1783a38663e2fcf170ceb', source_completed_outer=60,
    source_actual_native4_restore_evidence='leanmetrics-actual-cp60-native4-restore-summary.json',
    observed_unix=latest['epoch'], complete_outers=list(names), next_outer=66,
    next_outer_actual_before_generation_mtime=starts[66], next_outer_actual_generated_questions=0,
    post_restore_five_outers_accepted=False,
    acceptance_note='Five complete sealed outers and actual next66 sampler boundary observed; '
        'waiting next66 first generated chunk before final followup handoff.',
    per_outer=rows,
    aggregate=dict(full_outer_seconds_mean=whole, actor_update_seconds_mean=actor,
        candidate_fill_seconds_mean=candidate, actor_fraction_of_whole=actor / whole,
        candidate_fraction_of_whole=candidate / whole,
        selected_mixed_fraction_weighted=sum(v['selected_mixed'] for v in rows) / (5 * 512),
        completed_new_adam_writes_each_rank=100, cumulative_adam_writes_each_rank=1300,
        scheduler_each_rank=65, extra_monitor_forwards_each_rank=0),
    latest_checkpoint=latest['documents']['preparation/rolling-checkpoints/latest.json']['value'],
    latest_sealed_total_bytes=91405597196, latest_checkpoint_file_count=23, latest_boundary_file_count=7,
    comparison_scope='Different policy/data outers; observational timing, not same-input causality. '
        'Latest clip-only patch independently saves about3 seconds/outer, no whole-outer multiplier claim.',
    errors=latest['actual_error_lines'], goal_complete=False,
    remaining_goal='Continue native4 to200; fixed milestones/native51713/sevenmetrics/fullarchive/'
        'actualfinalnative4restore; no201.', evidence=list(names.values()))
if len(sys.argv) > 3:
    next_name = sys.argv[3]
    following = read(next_name)
    actual_next = next(v for v in following['candidate_progress'] if v['step'] == 66)
    generated = actual_next['complete_questions']
    summary.update(observed_unix=following['epoch'], next_outer_actual_generated_questions=generated,
        next_outer_actual_stream_cursor=actual_next['source_consumed'],
        post_restore_five_outers_accepted=generated >= 512,
        acceptance_note='Five complete sealed outers and first actual generated next66 chunk observed.',
        next_outer_evidence=next_name, actual_queue=following['queue'],
        errors=following['actual_error_lines'])
    summary['evidence'].append(next_name)
name = 'leanmetrics-five-complete-outers-summary.json'
for target in (out / name, evidence / name):
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(rows=[{k: v[k] for k in ('step', 'full_outer_seconds', 'candidate_fill_seconds',
    'actor_update_seconds', 'adv_seconds', 'save_checkpoint_seconds', 'training_em_rounded',
    'training_reward_mean_rounded', 'selected_mixed')} for v in rows], aggregate=summary['aggregate']),
    ensure_ascii=False))
