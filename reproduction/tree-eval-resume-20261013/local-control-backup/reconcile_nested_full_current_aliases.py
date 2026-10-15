"""Reconcile local current aliases after an already finalized FULL outer6..199.

This entry never collects evidence, starts a process, evaluates, or repairs a
pending publication. Its only writes are small local report data and snapshots.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import gzip
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
CORE_SHA = '676dca88489fb4d3c84891603990fc96256a14fb7da8ac2bc5c2def053e016c1'
FINALIZER_SHA = '61385504e4bb389cdba508c1b74f84fe6464e1d4f8ec11989e8cfe0c230dc541'
HERE = Path(__file__).resolve().parent


def dependencies():
    import hashlib
    paths = [HERE/'sync_nested_native_progress.py', HERE/'finalize_nested_progress.py']
    for path, expected in zip(paths, (CORE_SHA, FINALIZER_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned local dependency changed: ' + path.name)
    sys.path.insert(0, str(HERE))
    import sync_nested_native_progress as core
    spec = importlib.util.spec_from_file_location('pinned_alias_finalizer', paths[1])
    finalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finalizer)
    return core, finalizer, paths


class Inputs:
    def __init__(self, workspace, core):
        self.workspace, self.core, self.pins = workspace, core, []

    def path(self, value):
        path = Path(value)
        path = (path if path.is_absolute() else self.workspace/path).resolve()
        self.core.need(path.is_relative_to(self.workspace), 'Input is not local workspace evidence',
                       'REJECTED_INPUT_NO_WRITE')
        return path

    def read(self, value):
        data, pin = self.core.held(self.path(value))
        self.pins.append(pin)
        return data, pin

    def bound(self, descriptor, default_root=None):
        c = self.core
        c.need(isinstance(descriptor, dict) and c.sha64(descriptor.get('sha256')),
               'Missing immutable local descriptor', 'REJECTED_INPUT_NO_WRITE')
        name = descriptor.get('path') or descriptor.get('file')
        c.need(isinstance(name, str) and name, 'Missing descriptor path', 'REJECTED_INPUT_NO_WRITE')
        path = Path(name)
        if not path.is_absolute() and default_root is not None:
            path = Path(default_root)/path
        value, pin = self.read(path)
        c.need(pin['sha256'] == descriptor['sha256'] and pin['bytes'] == descriptor.get('bytes'),
               'Bound evidence bytes changed: ' + str(path), 'REJECTED_INPUT_NO_WRITE')
        return value, pin

    def unchanged(self):
        for pin in self.pins:
            self.core.need(self.core.digest(Path(pin['path']).read_bytes()) == pin['sha256'],
                           'Input changed while preparing aliases', 'CONCURRENT_UPDATE_NO_WRITE')


def identity(core, poll, outer):
    core.need(all(poll.get(k) == v for k, v in dict(source_commit=core.SOURCE,
              unit=core.UNIT, job_id=core.JOB, run=core.RUN, node='h100-10').items()),
              'Wrong fixed current training identity', 'REJECTED_INPUT_NO_WRITE')
    progress = poll.get('progress') or {}
    core.need(progress.get('actual_completed_outer_from_attempt_results') == outer
              and not progress.get('errors') and poll.get('run_result') is None,
              'Selected poll is pending, failed, or already past this FULL', 'REJECTED_INPUT_NO_WRITE')
    return progress


def service_cpu(core, resource_state):
    value = resource_state.get('CPUUsageNSec')
    core.need(isinstance(value, str) and re.fullmatch(r'\d+', value) is not None,
              'Actual resource CPUUsageNSec is required', 'REJECTED_INPUT_NO_WRITE')
    return int(value), int(value)/1_000_000_000


def monotonic_all(core, previous, outer, epoch):
    for _, value, _ in previous:
        completed = max(value.get(k, 0) or 0 for k in (
            'current_outer', 'completed_outer', 'actual_completed_outer',
            'actual_optimizer_completed_outer', 'last_confirmed_outer',
            'last_confirmed_optimizer_outer', 'latest_native_checkpoint_outer',
            'last_confirmed_native_saved_outer'))
        observed = max(value.get(k, 0) or 0 for k in (
            'observed_epoch', 'last_observed_epoch', 'last_confirmed_epoch',
            'last_actual_poll_epoch', 'last_current_progress_poll_epoch'))
        core.need(value.get('goal_complete') is not True and outer >= completed and epoch >= observed,
                  'Newer canonical or mirror state must not be overwritten', 'STALE_SNAPSHOT_NO_WRITE')


def validate(args, core, finalizer, inputs, base):
    n = args.outer
    core.need(type(n) is int and 6 <= n <= 199,
              'Only actual FULL outer6..199; pending and final200 are excluded', 'REJECTED_INPUT_NO_WRITE')
    proof, proof_pin = inputs.read(args.proof)
    core.need(Path(proof_pin['path']) == base/f'OUTER{n}_FULL_NEXT{n+1}_ACTUAL_PROGRESS.json'
              and all(proof.get(k) == v for k, v in dict(source_commit=core.SOURCE,
                  config_sha256=core.CONFIG, unit=core.UNIT, main_pid=int(core.PID),
                  invocation_id=core.INVOCATION, completed_outer=n, active_outer=n+1).items())
              and proof.get('same_original_instance') is True and proof.get('loaded_native_restore_outer') == 5
              and proof.get('current_validator_dependency_sha256') == CORE_SHA
              and proof.get('finalizer_sha256') == FINALIZER_SHA
              and not proof.get('errors') and proof.get('run_result') is None
              and proof.get('new_training_submitted') is False
              and proof.get('running_source_hot_modified') is False,
              'Completed two-observation proof identity differs', 'REJECTED_INPUT_NO_WRITE')
    full, full_pin = inputs.bound(proof.get('full_outer_receipt'))
    core.need(Path(full_pin['path']) == base/f'FULL_OUTER{n}_PROGRESS_RECEIPT.json'
              and full.get('local_control_script_sha256') == CORE_SHA
              and full.get('source_commit') == core.SOURCE and full.get('config_sha256') == core.CONFIG
              and full.get('unit') == core.UNIT and full.get('run') == core.RUN
              and full.get('restored_outer') == 5 and full.get('completed_outer') == n
              and full.get('actual_next_outer_entered') == n+1
              and full.get('actual_training_pid') == int(core.PID)
              and full.get('actual_training_invocation_id') == core.INVOCATION
              and full.get('full_iteration_metric_pending') is False,
              'Immutable FULL identity/complete metric differs', 'REJECTED_INPUT_NO_WRITE')
    bindings = full.get('input_bindings') or {}
    first, first_pin = inputs.bound(bindings.get('poll'))
    resource1, _ = inputs.bound(bindings.get('resource'))
    native, native_pin = inputs.bound(proof.get('native_readback'))
    core.need(native_pin['sha256'] == (bindings.get('native_readback') or {}).get('sha256')
              and native['manifest_sha256'] == proof.get('native_manifest_sha256')
              == full.get('native_manifest_sha256'), 'Native SHA binding differs', 'REJECTED_INPUT_NO_WRITE')
    core.validate_unit(first, resource1, n)
    core.validate_restore(first)
    metric = core.validate_metric(first, n)
    ranks = core.validate_ranks(first, metric, n)
    native_record, guard, completed, totals = core.validate_native(native, identity(core, first, n), n)
    core.need(full.get('iteration_metrics') == metric
              and full.get('per_rank_successful_adam') == completed
              and full.get('per_rank_actual_total_adam') == totals,
              'FULL does not match original rank/attempt evidence', 'REJECTED_INPUT_NO_WRITE')
    snapshots = proof.get('snapshots') or []
    core.need(len(snapshots) == 4, 'Two poll/resource bindings required', 'REJECTED_INPUT_NO_WRITE')
    original, original_pin = inputs.bound(snapshots[0])
    second, second_pin = inputs.bound(snapshots[1])
    original_resource, _ = inputs.bound(snapshots[2])
    second_resource, _ = inputs.bound(snapshots[3])
    core.need(original_pin == first_pin and original == first and original_resource == resource1,
              'Finalized proof first observation differs from FULL', 'REJECTED_INPUT_NO_WRITE')
    core.validate_unit(second, second_resource, n)
    core.validate_restore(second)
    core.need(second['epoch'] > first['epoch']
              and core.counts(identity(core, second, n).get('completed_actual_optimizer_calls_per_rank'),
                              'Second calls') == completed,
              'Second actual completed boundary differs', 'REJECTED_INPUT_NO_WRITE')
    sizes1, sizes2 = first.get('own_output_files') or {}, second.get('own_output_files') or {}
    growth = {name:sizes2[name]-sizes1[name] if core.integer(sizes1.get(name))
              and core.integer(sizes2.get(name)) else None for name in ('stage.log', 'retriever.log')}
    activity = finalizer.validate_next_activity(first, second, native, n, growth, first_pin, second_pin)
    core.need(activity == proof.get('continued_progress_evidence') and growth == proof.get('log_byte_growth'),
              'Actual second progress proof differs', 'REJECTED_INPUT_NO_WRITE')
    poll, poll_pin = inputs.read(args.poll)
    resource, resource_pin = inputs.read(args.resource)
    ps, rs, active = core.validate_unit(poll, resource, n)
    core.validate_restore(poll)
    latest_metric = core.validate_metric(poll, n)
    core.validate_ranks(poll, latest_metric, n)
    core.validate_native(native, identity(core, poll, n), n)
    core.need(active and poll['epoch'] >= second['epoch'] and latest_metric == metric
              and re.search(r'epoch\s+\d+,\s*step\s+' + str(n+1) + r'\b',
                            (poll.get('logs') or {}).get('stage.log') or ''),
              'Latest actual same-N/next-stage observation differs', 'REJECTED_INPUT_NO_WRITE')
    watcher = resource.get('state') or {}
    core.need(not watcher or (watcher.get('training_source_commit') == core.SOURCE
              and watcher.get('training_unit') == core.UNIT and watcher.get('run_directory') == core.RUN
              and watcher.get('expected_training_invocation_id') == core.INVOCATION
              and not watcher.get('last_errors') and not watcher.get('audit_error_count')),
              'Current native watcher binding/health differs', 'REJECTED_INPUT_NO_WRITE')
    cpu_nsec, cpu_seconds = service_cpu(core, rs)
    costs, cost_pin = inputs.bound(proof.get('cumulative_cost'))
    core.need(Path(cost_pin['path']) == base/f'CUMULATIVE_COST_THROUGH{n}.json',
              'Immutable cost filename differs', 'REJECTED_INPUT_NO_WRITE')
    costs, cost_parts = finalizer.expand_cost_fields(costs, cost_pin, inputs.bound)
    cost_full = (costs.get('immutable_metadata_input_hashes') or {}).get(Path(full_pin['path']).name) or {}
    cost_rows = costs.get('rows_counted_once') or []
    core.need(costs.get('through_effective_outer') == n and costs.get('target_outer') == 200
              and costs.get('origin') == 'fresh' and cost_full.get('sha256') == full_pin['sha256']
              and cost_full.get('bytes') == full_pin['bytes'] and len(cost_rows) == n
              and sorted(row.get('outer') for row in cost_rows) == list(range(1, n+1))
              and (costs.get('successful_optimizer_accounting') or {}).get('direct_observed_calls_per_rank') == completed,
              'Cost/FULL/actual write boundary differs', 'REJECTED_INPUT_NO_WRITE')
    core.need(all(row.get('source_commit') == (
              '0befbaf99fc44942105831b3fbec94e4bf1b8ccb' if row['outer'] == 1 else
              'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4' if row['outer'] <= 5 else core.SOURCE)
              for row in cost_rows), 'Immutable cost source segments differ', 'REJECTED_INPUT_NO_WRITE')
    index, index_pin = inputs.bound(proof.get('candidate_index'))
    candidate, candidate_pin = inputs.bound(proof.get('candidate_current_supplement'))
    rates, rates_pin = inputs.bound(proof.get('current_branch_fallback_rates'))
    registry, registry_pin = inputs.bound(proof.get('candidate_registry'))
    core.need(index.get('source_commit') == candidate.get('source_commit') == rates.get('source_commit') == core.SOURCE
              and index.get('config_sha256') == candidate.get('config_sha256') == rates.get('config_sha256') == core.CONFIG
              and candidate.get('requested_outer') == rates.get('outer') == n
              and candidate.get('status') == 'ACTUAL_COMPLETED_CANDIDATE_SCALARS_AGGREGATED'
              and (rates.get('cumulative_snapshot') or {}).get('sha256') == cost_pin['sha256'],
              'Actual candidateN supplement/rates/cost binding differs', 'REJECTED_INPUT_NO_WRITE')
    curve, curve_pin = inputs.bound((proof.get('attachments') or {}).get('learning_curve_complete'))
    curve_end = (curve.get('training_outer_range') or [None, None])[-1]
    core.need(core.integer(curve_end) and 5 <= curve_end <= n and curve.get('target_outer') == 200
              and curve.get('training_point_count') == len(curve.get('training_points') or []),
              'Current immutable curve coverage differs', 'REJECTED_INPUT_NO_WRITE')
    actual_dev, latest_dev = full.get('actual_dev512'), full.get('latest_actual_dev512')
    core.need(core.valid_dev(latest_dev) and latest_dev['outer'] <= curve_end
              and ((n % 5 == 0 and core.valid_dev(actual_dev) and actual_dev['outer'] == n)
                   or (n % 5 != 0 and actual_dev is None)),
              'FULL current/latest dev is pending or assigned to another outer', 'REJECTED_INPUT_NO_WRITE')
    return dict(proof=proof, proof_pin=proof_pin, full=full, full_pin=full_pin, poll=poll,
                poll_pin=poll_pin, resource=resource, resource_pin=resource_pin, native=native,
                native_pin=native_pin, native_record=native_record, guard=guard, completed=completed,
                totals=totals, ranks=ranks, metric=metric, costs=costs, cost_pin=cost_pin,
                cost_parts=cost_parts, candidate_pin=candidate_pin, index_pin=index_pin,
                registry_pin=registry_pin, rates=rates, rates_pin=rates_pin, curve_pin=curve_pin,
                curve_end=int(curve_end), actual_dev=actual_dev, latest_dev=latest_dev,
                cpu_nsec=cpu_nsec, cpu_seconds=cpu_seconds, rs=rs, activity=activity)


def aliases(core, n, data, receipt_pin, now):
    p, f, proof, metric = [data[k] for k in ('poll', 'full', 'proof', 'metric')]
    completed, totals = data['completed'], data['totals']
    latest, dev = data['latest_dev'], data['actual_dev']
    child = {rank:value-100 for rank, value in totals.items()}
    cost_scope = proof['candidate_cost_join_scope']
    dev_seconds = f['normal_dev512_wall_seconds'] if dev is not None else None
    variances = copy.deepcopy(f['current_training_variances'])
    phase = f'FULL{n}_NATIVE{n}_COMPLETE_NEXT{n+1}_SAMPLING'
    value = dict(status=phase, phase=phase, current_phase=phase, training_phase=phase,
        source_commit=core.SOURCE, config_sha256=core.CONFIG, unit=core.UNIT, run=core.RUN,
        main_pid=int(core.PID), pid=int(core.PID), invocation_id=core.INVOCATION, invocation=core.INVOCATION,
        goal_status='active', goal_active=True, goal_complete=False, formal_effect_complete=False,
        goal_turn_classification='progress', consecutive_blocked_goal_turns=0,
        current_consecutive_blocked_goal_turns=0, blocked_audit_consecutive_goal_turns=0,
        completed_outer=n, current_outer=n, actual_completed_outer=n, actual_optimizer_completed_outer=n,
        last_confirmed_outer=n, last_confirmed_chain_outer=n, last_confirmed_optimizer_outer=n,
        last_durable_completed_outer=n, new_completed_outer=n-5,
        actual_total_adam_per_rank=totals, per_rank_actual_total_adam=totals,
        per_rank_successful_adam=completed, successful_adam_per_rank=completed,
        actual_successful_adam_per_rank=completed, child_successful_adam_per_rank=child,
        actual_physical_rank_optimizer_calls=sum(totals.values()), inherited_completed_outer=5,
        actual_current_restore_outer=5, restored_outer=5, loaded_native_restore_outer=5,
        actual_current_restore_adam_per_rank={str(r):100 for r in range(8)},
        actual_current_restore_scope=f'Actual cold-loaded5/100 each remains unchanged; writer-complete native{n} is a distinct saved boundary.',
        actual_child_counters_known=True, effective_counters_known=True,
        actual_training_progress=p['progress'], all_skipped_attempts=p['progress']['all_skipped_attempts'],
        direct_rank_record_count=p['direct_rank_records'], direct_child_rank_record_count=p['direct_rank_records'],
        inherited_rank_record_count=40, effective_chain_rank_record_count=p['direct_rank_records']+40,
        actual_outer_successful_writes_per_rank=f['actual_outer_successful_writes_per_rank'],
        actual_outer_processed_opportunities_per_rank=f['actual_outer_processed_opportunities_per_rank'],
        actual_outer_skipped_opportunities_per_rank=f['actual_outer_skipped_opportunities_per_rank'],
        full_metric_processed_opportunities=metric['actor/optimizer_opportunities_processed'],
        full_metric_skipped_opportunities=metric['actor/optimizer_opportunities_skipped'],
        no_all20_writes_or_zero_skip_assumption=True, active_outer=n+1, active_outer_in_progress=n+1,
        actual_next_outer_entered=n+1, actual_next_outer=n+1, current_next_outer=n+1,
        last_confirmed_active_outer=n+1, latest_active_sampling_outer=n+1,
        active_outer_sampling_complete=None, active_outer_sampling_complete_confirmed=None,
        active_outer_actual_partial_Adam_calls=None, current_partial_optimizer_calls_per_rank=None,
        active_outer_advantage_variance=None, active_outer_full_update_unconfirmed=True,
        active_outer_optimizer_completion_confirmed=False, next_outer_sampling_complete=None,
        current_active_rollout_progress=None, current_live_sampling_progress=None,
        active_outer_rollout_cost=None,
        current_active_sampling_cost_scope=f'Next outer{n+1} sampling cost is unknown; complete outer{n} cost remains bound to existing FULL/current/cumulative fields.',
        current_active_outer_full_metric_pending=True,
        current_next_outer_progress=data['activity'], current_next_outer_progress_evidence=data['activity'],
        latest_native_checkpoint_outer=n, last_confirmed_native_saved_outer=n, latest_saved_native_outer=n,
        native_saved_driver_next_step=n+1, native_saved_cursor_is_not_live_next_outer_entry=True,
        native_manifest_sha256=data['native']['manifest_sha256'],
        native_manifest_history_receipt_sha256=data['native']['receipt_sha256'],
        latest_native_metadata_scope=f'Actual writer/archive native{n} JSON metadata; no independent PT cold reload.',
        current_iteration_metrics=metric, current_outer_full_metric=metric,
        current_iteration_metric_pending=False, current_full_outer_metric_pending=False,
        current_outer_full_metric_pending=False, full_iteration_metric_pending=False, full_iteration_log_pending=False,
        current_training_variances=variances, current_variances=variances, latest_training_variance_outer=n,
        current_candidate_cost_scalars=f['current_candidate_cost_scalars'],
        current_generation_wall_seconds=metric['timing_s/gen'],
        current_rollout_cost=None, latest_rollout_cost=None,
        latest_training_variance_scope=f'Actual complete outer{n}; next{n+1} variance is unknown.',
        current_outer_wall_seconds=metric['timing_s/step'], latest_full_outer_wall_seconds=metric['timing_s/step'],
        latest_full_outer_wall_minutes=metric['timing_s/step']/60,
        latest_full_outer_wall_scope=f'Actual complete FULL{n}, not next{n+1} elapsed time.',
        last_complete_outer_wall_seconds=metric['timing_s/step'],
        current_ordinary_training_seconds_excluding_dev=f['training_archive_wall_seconds_excluding_current_dev'],
        latest_ordinary_training_seconds_excluding_dev=f['training_archive_wall_seconds_excluding_current_dev'],
        current_training_archive_wall_seconds_excluding_dev=f['training_archive_wall_seconds_excluding_current_dev'],
        latest_training_archive_wall_seconds_excluding_dev=f['training_archive_wall_seconds_excluding_current_dev'],
        training_archive_wall_seconds_excluding_current_dev=f['training_archive_wall_seconds_excluding_current_dev'],
        current_reward_mean=metric.get('critic/rewards/mean'), current_score_mean=metric.get('critic/score/mean'),
        current_dev512=dev, actual_dev512=dev, current_dev512_outer=n if dev else None,
        current_dev512_wall_seconds=dev_seconds, normal_dev512_wall_seconds=dev_seconds,
        current_dev512_f1=dev['metrics']['val/f1/all'] if dev else None,
        current_dev512_em=dev['metrics']['val/em/all'] if dev else None,
        latest_dev512=latest, latest_actual_dev512=latest, latest_dev512_outer=latest['outer'],
        latest_confirmed_dev_outer=latest['outer'], latest_confirmed_dev_f1=latest['metrics']['val/f1/all'],
        latest_confirmed_dev_em=latest['metrics']['val/em/all'], latest_dev512_f1_percent=100*latest['metrics']['val/f1/all'],
        latest_dev512_em_percent=100*latest['metrics']['val/em/all'],
        score_updated=bool(f['dev_score_updated_this_turn']) if dev else False,
        dev_score_updated=bool(f['dev_score_updated_this_turn']) if dev else False,
        new_score_updated=bool(f['dev_score_updated_this_turn']) if dev else False,
        dev_score_updated_this_turn=bool(f['dev_score_updated_this_turn']) if dev else False,
        score_updated_this_turn=bool(f['dev_score_updated_this_turn']) if dev else False,
        score_update_scope=f'Existing immutable FULL{n} publication event; this script never evaluates.',
        normal_dev512_pending=False, last_full_outer=n, last_complete_outer=n,
        last_complete_full_metric_outer=n, last_confirmed_full_metric_outer=n,
        full_metric_last_seen=n, current_metric_outer=n,
        current_full_outer_progress_receipt=Path(data['full_pin']['path']).name,
        last_full_outer_progress_receipt=Path(data['full_pin']['path']).name,
        current_completed_progress_evidence=data['proof_pin']['path'],
        current_completed_progress_sha256=data['proof_pin']['sha256'],
        evidence=data['poll_pin']['path'], last_progress_receipt=Path(data['full_pin']['path']).name,
        current_actual_progress_proof=data['proof_pin'], actual_progress_receipt=data['proof_pin'],
        current_actual_readback=receipt_pin, actual_live_progress_receipt=receipt_pin,
        current_observation_status=phase, current_remote_state='KNOWN_AT_ACTUAL_BOUND_POLL',
        current_remote_health_known=True, current_observation_unknown=False,
        current_observation_error=None, current_failure_evidence=None, current_observation_failure_evidence=None,
        observed_epoch=p['epoch'], last_observed_epoch=p['epoch'], last_confirmed_epoch=p['epoch'],
        last_actual_poll_epoch=p['epoch'], last_actual_poll=data['poll_pin']['path'],
        last_actual_poll_sha256=data['poll_pin']['sha256'], last_current_progress_poll=data['poll_pin']['path'],
        last_current_progress_poll_epoch=p['epoch'], last_rank_observation=data['poll_pin']['path'],
        current_resource_observation=data['resource_pin'], last_confirmed_resource_epoch=data['resource']['epoch'],
        current_systemd=p['systemd'], latest_guard=p['latest_guard'], same_instance_confirmed=True,
        same_instance_repoll_required=False, current_observation_training_action='none',
        state_scope=f'Actual same-instance FULL/native{n} and next{n+1} entry; current remote state is known at the bound poll.',
        observed_current_training_phase=phase, current_next_outer_scope=f'Actual stage{n+1} entered; its completion/write/variance counts remain unknown.',
        expected_next_outer=n+1, expected_next_outer_scope='Actual stage entry, separate from saved native driver cursor.',
        current_latest_sampling_interval_log_growth=None,
        current_latest_sampling_interval_scope='The immutable two-observation proof establishes activity; no new next-outer sample/update counts are inferred.',
        training_action='none', training_terminal_proven=False, run_result=None, restart_or_submit_performed=False,
        last_nonterminal_service_CPU_seconds=data['cpu_seconds'],
        last_nonterminal_service_CPUUsageNSec=data['cpu_nsec'], final_service_CPU_seconds=None,
        actual_total_service_cpu_seconds=None, current_service_CPU_scope='Actual resource CPUUsageNSec/1e9 at this poll; nonterminal service total, not per-outer cost.',
        per_outer_costs_through=n, latest_cost_complete_outer=n, latest_cost_complete_scope=cost_scope,
        cumulative_cost_summary=data['cost_pin']['path'], cumulative_cost_summary_sha256=data['cost_pin']['sha256'],
        current_cost_summary_publication_pending_outer=None, current_cost_accounting_backfill_pending=None,
        current_candidate_cost_actual_rates=data['rates'], current_branch_fallback_rates_receipt=data['rates_pin']['path'],
        current_branch_fallback_rates_receipt_sha256=data['rates_pin']['sha256'],
        candidate_cost_extension_latest_complete_outer=n, candidate_cost_extension_latest_part=data['candidate_pin']['path'],
        candidate_cost_extension_latest_part_sha256=data['candidate_pin']['sha256'],
        candidate_cost_extension_registry=data['registry_pin']['path'],
        candidate_cost_extension_registry_sha256=data['registry_pin']['sha256'], current_candidate_cost_join_scope=cost_scope,
        current_cost_coverage=dict(aggregate_sampling_stage_cost_complete_through=n,
            immutable_cumulative_join_through=n, separate_candidate_supplement_outer=n,
            candidate_supplement_for_next_outer=n+1, scope=cost_scope, final_complete_cost_accepted=False),
        current_cost_local_readback=dict(immutable_cost=data['cost_pin'], candidate_supplement=data['candidate_pin'], scope=cost_scope),
        learning_curve_complete=data['curve_pin'], learning_curve_complete_through_outer=data['curve_end'],
        learning_curve_complete_scope='Existing hash-bound curve; not synthesized or extended by reconciliation.',
        effective_source_segments=data['costs']['effective_source_segments'],
        current_alias_reconciliation=receipt_pin, updated_at=now, recorded_at_utc=now,
        final200_51713_HF_tokenizer_acceptance_complete=False, final_evaluation_complete=False,
        independent_cold_reload_performed=False, exact_executor_resume=False,
        health=f'Healthy original PID{core.PID}/Invocation{core.INVOCATION}; FULL/native{n} complete and actual next{n+1} entered, next update details unknown.')
    for key, name in [('latest_actual_total_adv_population_variance','tree_variance/group_actual_adv_population_variance'),
                      ('latest_intra_adv_population_variance','tree_variance/intra_adv_population_variance'),
                      ('latest_within_root_raw_reward_population_variance','tree_variance/within_root_raw_reward_population_variance'),
                      ('latest_mixed_reward_question_fraction','tree_variance/mixed_reward_question_fraction')]:
        value[key] = metric[name]
    value[f'dev{n}_result_pending'] = False
    value[f'normal_dev512_outer{n}_pending'] = False
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', required=True, type=int)
    for name in ('poll', 'resource', 'proof', 'workspace'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    core, finalizer, dependency_paths = dependencies()
    workspace = args.workspace.resolve();base = workspace/core.BASE_REL
    inputs = Inputs(workspace, core)
    data = validate(args, core, finalizer, inputs, base);n = args.outer
    paths = [workspace/'outputs/STATUS.json', workspace/'outputs/tree-nested-branch/STATUS.json',
             base/'CURRENT_NATIVE_RECOVERY_FOLLOWUP.json', base/'CURRENT_OBSERVER_CONNECTION_STATE.json']
    previous = [(path, *inputs.read(path)) for path in paths]
    core.monotonic(previous, n, data['poll']['epoch'])
    monotonic_all(core, previous, n, data['poll']['epoch'])
    for _, value, _ in previous:
        core.need(value.get('source_commit', core.SOURCE) == core.SOURCE and value.get('goal_status') == 'active'
                  and value.get('goal_active', True) is not False,
                  'Current source/active goal differs', 'STALE_SNAPSHOT_NO_WRITE')
    now = datetime.now(timezone.utc).isoformat()
    receipt_path = base/f'CURRENT_ALIAS_RECONCILIATION_OUTER{n}.json'
    core.need(not receipt_path.exists(), 'Existing reconciliation receipt retained', 'RECEIPT_EXISTS_NO_WRITE')
    mirrors = [workspace/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',
               Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
    core.need(all(p.is_dir() for p in mirrors), 'Both existing local mirrors are required', 'REJECTED_INPUT_NO_WRITE')
    mirror_previous = []
    for mirror in mirrors:
        for name in ('STATUS.json', 'CURRENT_NATIVE_RECOVERY_FOLLOWUP.json', 'CURRENT_OBSERVER_CONNECTION_STATE.json'):
            if (mirror/name).is_file():
                value, pin = core.held(mirror/name)
                mirror_previous.append((mirror/name, value, pin));inputs.pins.append(pin)
    monotonic_all(core, mirror_previous, n, data['poll']['epoch'])
    snapshot = base/f'canonical_alias_snapshot_before_outer{n}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}'
    receipt = dict(schema='nested_full_current_alias_reconciliation_v1', status='ACTUAL_FULL_LOCAL_ALIASES_RECONCILED',
        created_at_utc=now, source_commit=core.SOURCE, outer=n, actual_next_outer_entered=n+1,
        full=data['full_pin'], proof=data['proof_pin'], native=data['native_pin'], cost=data['cost_pin'],
        curve=data['curve_pin'], poll=data['poll_pin'], resource=data['resource_pin'],
        pinned_core_sha256=CORE_SHA, pinned_finalizer_sha256=FINALIZER_SHA,
        actual_adam_per_rank=data['totals'], actual_child_adam_per_rank={k:v-100 for k,v in data['totals'].items()},
        actual_restore_outer=5, actual_restore_adam_per_rank={str(r):100 for r in range(8)},
        writer_complete_outer=n, current_dev_outer=n if data['actual_dev'] else None,
        latest_dev_outer=data['latest_dev']['outer'], next_sampling_complete=None, next_partial_adam=None,
        next_variance=None, CPUUsageNSec=data['cpu_nsec'], nonterminal_service_CPU_seconds=data['cpu_seconds'],
        final_service_CPU_seconds=None, exact_executor_resume=False, goal_status='active', goal_complete=False,
        prior_snapshot_manifest=str(snapshot/'SNAPSHOT_MANIFEST.json'),
        source_training_or_immutable_evidence_modified=False, remote_actions=False)
    receipt_raw = core.json_bytes(receipt)
    receipt_pin = dict(path=str(receipt_path), bytes=len(receipt_raw), sha256=core.digest(receipt_raw))
    common = aliases(core, n, data, receipt_pin, now)
    updated = []
    for path, old, _ in previous:
        value = copy.deepcopy(old)
        archive = value.setdefault('historical_current_alias_reconciliations', {})
        core.need(isinstance(archive, dict), 'Historical alias archive type differs', 'REJECTED_INPUT_NO_WRITE')
        archive[str(n)] = dict(scope='Prior generic aliases; historical after this FULL reconciliation.',
                              fields={k:old[k] for k,v in common.items() if k in old and old[k] != v})
        if old.get('current_active_rollout_progress') is not None:
            archive[str(n)]['retired_current_rollout_pointer'] = old['current_active_rollout_progress']
        value.update(copy.deepcopy(common));updated.append(value)
    # Both status aliases receive the richer existing root status, preserving its history.
    updated[1] = copy.deepcopy(updated[0])
    report = (f'Current nested Tree200: actual FULL/native{n}/200; next{n+1} entered in the original instance.\n\n'
              f'Actual cumulative Adam each rank {data["totals"]}; child counts subtract the actually restored100 each. '
              f'Cold load remains native5; writer-complete boundary is{n}. '
              f'Actual FULL{n} successful Adam writes per rank: {json.dumps(data["full"]["actual_outer_successful_writes_per_rank"])}. '
              f'Original per-rank processed opportunities: {json.dumps(data["full"]["actual_outer_processed_opportunities_per_rank"])}; '
              f'skipped opportunities: {json.dumps(data["full"]["actual_outer_skipped_opportunities_per_rank"])}. '
              f'Full metric actual successful Adam/processed/skipped: {data["metric"]["actor/optimizer_steps"]}/'
              f'{data["full"]["full_metric_processed_opportunities"]}/{data["full"]["full_metric_skipped_opportunities"]}. '
              'Missing original rank fields remain null.\n\n'
              f'Current full wall {data["metric"]["timing_s/step"]}s; dev wall {data["full"]["normal_dev512_wall_seconds"] if data["actual_dev"] else None}; '
              f'training/archive excluding current dev {data["full"]["training_archive_wall_seconds_excluding_current_dev"]}s. '
              f'Actual FULL{n} raw reward population variance (group): '
              f'{json.dumps(data["full"]["current_training_variances"].get("tree_variance/group_raw_reward_population_variance"))}; '
              f'raw reward population variance (within root): '
              f'{json.dumps(data["full"]["current_training_variances"].get("tree_variance/within_root_raw_reward_population_variance"))}; '
              f'intra advantage population variance: '
              f'{json.dumps(data["full"]["current_training_variances"].get("tree_variance/intra_adv_population_variance"))}; '
              f'total actual advantage population variance: '
              f'{json.dumps(data["full"]["current_training_variances"].get("tree_variance/group_actual_adv_population_variance"))}. '
              f'All are scoped to{n}; next{n+1} sampling completion, partial Adam and variance remain unknown.\n\n'
              f'Current dev outer {json.dumps(n if data["actual_dev"] else None)}, '
              f'F1_percent {json.dumps(100*data["actual_dev"]["metrics"]["val/f1/all"] if data["actual_dev"] else None)}, '
              f'EM_percent {json.dumps(100*data["actual_dev"]["metrics"]["val/em/all"] if data["actual_dev"] else None)} '
              '(non-dev outers have current dev=null). '
              f'Latest actual dev outer {data["latest_dev"]["outer"]}: '
              f'F1_percent {100*data["latest_dev"]["metrics"]["val/f1/all"]}, '
              f'EM_percent {100*data["latest_dev"]["metrics"]["val/em/all"]}.\n\n'
              f'Immutable cost boundary{n}; candidate{n} supplement remains separate for the next cumulative join. '
              f'Nonterminal service CPU {data["cpu_seconds"]}s is parsed from actual resource CPUUsageNSec/1e9; final service CPU remains null.\n\n'
              'Goal active/block0. Actual200, final51713, HF/tokenizer acceptance and cleanup remain pending. '
              'No new training, restore, evaluation, review, remote read or exact-resume claim.\n')
    outputs = {receipt_path:receipt_raw, **{path:core.json_bytes(value) for (path,_,_),value in zip(previous,updated)},
               workspace/'outputs/REPORT.md':report.encode(),workspace/'outputs/tree-nested-branch/REPORT.md':report.encode()}
    # Snapshot only the small mutable documents before any replacement.
    for mirror in mirrors:
        for source in (paths[1], paths[2], paths[3], workspace/'outputs/tree-nested-branch/REPORT.md', receipt_path):
            outputs[mirror/source.name] = outputs[source]
    prior_raw = {}
    for path in outputs:
        if path.exists():
            raw = path.read_bytes();core.need(len(raw) <= 12*1024**2, 'Mutable document exceeds small local limit')
            prior_raw[path] = raw
    with dependency_paths[0].open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        inputs.unchanged()
        core.need(all(core.digest(path.read_bytes()) == core.digest(raw) for path,raw in prior_raw.items()),
                  'Mutable report/mirror changed during preparation', 'CONCURRENT_UPDATE_NO_WRITE')
        core.monotonic(core.previous_snapshots(workspace,base), n, data['poll']['epoch'])
        core.need(not receipt_path.exists(), 'Another alias publisher completed', 'CONCURRENT_UPDATE_NO_WRITE')
        snapshot.mkdir();manifest=[]
        for index,(path,raw) in enumerate(prior_raw.items()):
            member = snapshot/f'{index:02d}-{path.name}.gz'
            member.write_bytes(gzip.compress(raw, mtime=0))
            stored = member.read_bytes()
            core.need(gzip.decompress(stored) == raw, 'Prior gzip snapshot does not restore exact original bytes')
            manifest.append(dict(source=str(path), original_bytes=len(raw), original_sha256=core.digest(raw),
                                 stored_gzip_path=str(member), stored_bytes=len(stored),
                                 stored_sha256=core.digest(stored), gzip_mtime=0,
                                 exact_original_bytes_readback=True))
        (snapshot/'SNAPSHOT_MANIFEST.json').write_bytes(core.json_bytes(dict(files=manifest, created_at_utc=now)))
        for path,raw in outputs.items():
            core.atomic_replace(path,raw)
        core.need(all(path.read_bytes() == raw for path,raw in outputs.items()), 'Local report/mirror readback differs')
    print(json.dumps(dict(status='LOCAL_FULL_CURRENT_ALIASES_AND_TWO_MIRRORS_PUBLISHED',outer=n,
                          receipt=receipt_pin, snapshot=str(snapshot), files=len(outputs), remote_actions=False)))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status = getattr(error, 'status', 'REJECTED_LOCAL_ALIAS_PUBLICATION')
        print(json.dumps(dict(status=status,error=str(error),remote_actions=False)),file=sys.stderr)
        raise SystemExit(2)
