"""Audit every real completed or fully skipped formal attempt, without fake writes.

This reader checks the owned driver, scheduler and worker receipt prefix. It
does not execute a model or authenticate physical GPU execution from JSON.
"""
import copy
import math
import os
import re
from pathlib import Path

from reproduction.common8.formal_contract import FORMAL_SAFEGUARDS, consumed_lr
from reproduction.common8.policy_calibration_plan import _strict_json, document_sha256
from reproduction.common8.topology import (
    actor_local_mini, actor_local_micro, formal_monitor_geometry, local_trajectories, monitor_scope, monitor_trajectories_per_rank, ranks, world_size,
)
from search_g1.monitor_forward_reuse import summarize_monitor_execution
from reproduction.common8.safety_calibration_evidence import (
    EvidenceError, GLOBAL_FIELDS, GRADIENT_CLIP_FLOAT32_TOLERANCE, GRADIENT_CLIP_LIMIT,
    QUANTILE_METHOD, objective_weight_definition, optimizer_gradient_evidence,
    _adjacent, _enforcement, _finite, _hash, _identity, _integer, _require,
    _row, _same_arithmetic, _statistics,
)


NUMERICAL_GUARD_STAGES = frozenset((
    'before_gradient_clipping', 'before_optimizer_step', 'before_dual_clip_exp',
    'before_backward', 'before_fp64_gradient_clipping', 'before_fp64_clip_write',
    'after_fp64_gradient_clipping', 'measured_after_gradient_clipping', 'after_optimizer_write',
))
INCOMPLETE_REASONS = frozenset((
    'nonfinite_global_totals', 'nonfinite_or_negative_effective_input',
    'delta_above_kl_log_bound', 'empty_effective_samples',
))


def _owned_rows(base, relative):
    path = base / relative
    _require(path.is_file() and not path.is_symlink() and path.resolve() == path
             and path.stat().st_uid == os.getuid(), 'Receipt is not an owned canonical file: ' + relative)
    return [_strict_json(line) for line in path.read_bytes().splitlines()]


def _attempt_identity(row):
    context = row.get('context') or row
    index = _integer(context['attempt_index'], 'attempt_index', minimum=1)
    attempt = context['attempt_id']
    _require(isinstance(attempt, str) and re.fullmatch('[0-9a-f]{32}', attempt) is not None,
             'Malformed actual attempt UUID')
    return attempt, index


def _monitor_statistics(row, diagnostic_only=False):
    if diagnostic_only:
        _require(row.get('finite_thresholds_diagnostic_only') is True,
                 'Actual monitor omitted the bound diagnostic-only threshold mode')
        _require(row.get('enforcement')==dict(negative_tail='observe_only',old_policy_kl='observe_only',
                 numerical_finite='enforced',dual_clip='enforced',log_ratio_bound='observe_only'),
                 'Diagnostic mode must retain numerical rejection and DualClip')
        reasons=[]
        if row['kl_estimate_complete']:
            if row['sampled_old_policy_kl']>FORMAL_SAFEGUARDS['target_old_policy_kl']: reasons.append('old_policy_kl')
        else:
            nonfinite_totals=any(row[name] is None for name in ('effective_weight_mass','weighted_k3_numerator',
                'negative_effective_weight_mass','tail_effective_weight_mass'))
            if nonfinite_totals: reasons.append('nonfinite_global_totals')
            if row['invalid_count']: reasons.append('nonfinite_or_negative_effective_input')
            if row['overflow_count']: reasons.append('delta_above_kl_log_bound')
            if row['effective_token_count']<=0 or (not nonfinite_totals and row['effective_weight_mass']<=0):
                reasons.append('empty_effective_samples')
        if row['negative_tail_weight_fraction']>FORMAL_SAFEGUARDS['negative_tail_limit']:
            reasons.append('negative_advantage_tail')
        thresholds={'delta_above_kl_log_bound','old_policy_kl','negative_advantage_tail'}
        warnings=[reason for reason in reasons if reason in thresholds]
        fatal=[reason for reason in reasons if reason not in thresholds]
        _require(row.get('diagnostic_warnings')==warnings and row.get('fatal_reasons')==fatal and
                 row['reasons']==fatal and row['allowed']==(not fatal),
                 'Monitor finite warnings/fatal numerical disposition differs')
        # Reuse unchanged scalar/count/arithmetic verification. Normalization is
        # only for the legacy reader's expectation, never an actual skip claim.
        _monitor_statistics(dict(row,reasons=reasons,allowed=not reasons,finite_thresholds_diagnostic_only=False))
        result={name:copy.deepcopy(row[name]) for name in GLOBAL_FIELDS}
        result.update(finite_thresholds_diagnostic_only=True,diagnostic_warnings=warnings,fatal_reasons=fatal)
        return result
    _require(row.get('finite_thresholds_diagnostic_only',False) is False,
             'Unbound diagnostic-only monitor mode')
    _require(type(row['allowed']) is bool and type(row['kl_estimate_complete']) is bool,
             'Malformed actual monitor decision')
    _require(isinstance(row['reasons'], list) and all(isinstance(x, str) and x for x in row['reasons']),
             'Malformed actual monitor reasons')
    if row['kl_estimate_complete']:
        statistics = _statistics(dict(row, allowed=True, reasons=[]))
        reasons = []
        if statistics['sampled_old_policy_kl'] > FORMAL_SAFEGUARDS['target_old_policy_kl']:
            reasons.append('old_policy_kl')
        if statistics['negative_tail_weight_fraction'] > FORMAL_SAFEGUARDS['negative_tail_limit']:
            reasons.append('negative_advantage_tail')
        _require(row['reasons'] == reasons and row['allowed'] == (not reasons),
                 'Actual monitor differs from the frozen prospective brake decision')
        statistics.update(allowed=row['allowed'], reasons=copy.deepcopy(reasons))
        return statistics
    _require(row['allowed'] is False and row['sampled_old_policy_kl'] is None and row['reasons']
             and any(reason in INCOMPLETE_REASONS for reason in row['reasons']),
             'Incomplete numerical estimate cannot claim true KL or an unexplained skip')
    for name in ('effective_token_count', 'negative_token_count', 'negative_tail_count', 'invalid_count', 'overflow_count'):
        _integer(row[name], name)
    count = row['effective_token_count']
    _require(0 <= row['negative_tail_count'] <= row['negative_token_count'] <= count
             and row['invalid_count'] <= count and row['overflow_count'] <= count,
             'Inconsistent incomplete numerical counts')
    nonfinite_totals = 'nonfinite_global_totals' in row['reasons']
    for name in ('effective_weight_mass', 'weighted_k3_numerator', 'negative_effective_weight_mass',
                 'tail_effective_weight_mass'):
        if nonfinite_totals:
            _require(row[name] is None, 'Nonfinite global total must remain an explicit unavailable diagnostic')
        else:
            _finite(row[name], name)
    for name in ('negative_tail_count_fraction', 'negative_tail_weight_fraction'):
        _finite(row[name], name)
    _same_arithmetic(row['negative_tail_count_fraction'],
        row['negative_tail_count'] / row['negative_token_count'] if row['negative_token_count'] else 0.,
        'incomplete negative tail count fraction')
    if nonfinite_totals:
        _require(row['negative_tail_weight_fraction'] == 0., 'Unavailable tail mass must remain a diagnostic fraction')
    else:
        mass, negative_mass, tail_mass = (row[name] for name in
            ('effective_weight_mass', 'negative_effective_weight_mass', 'tail_effective_weight_mass'))
        _require((negative_mass <= mass or math.isclose(negative_mass, mass, rel_tol=1e-12, abs_tol=1e-12)) and
                 (tail_mass <= negative_mass or math.isclose(tail_mass, negative_mass, rel_tol=1e-12, abs_tol=1e-12)),
                 'Inconsistent incomplete global masses')
        _same_arithmetic(row['negative_tail_weight_fraction'], tail_mass / negative_mass if negative_mass else 0.,
                         'incomplete negative tail weight fraction')
    saturated = row['dual_clip_saturated_objective_weight_fraction']
    if nonfinite_totals or row['effective_weight_mass'] == 0:
        _require(saturated is None, 'Unavailable saturation cannot claim a finite complete ratio')
    else:
        _finite(saturated, 'dual_clip_saturated_objective_weight_fraction')
        _same_arithmetic(saturated, row['tail_effective_weight_mass'] / row['effective_weight_mass'],
                         'incomplete saturated objective fraction')
    reasons = []
    if nonfinite_totals: reasons.append('nonfinite_global_totals')
    if row['invalid_count']: reasons.append('nonfinite_or_negative_effective_input')
    if row['overflow_count']: reasons.append('delta_above_kl_log_bound')
    if count <= 0 or (not nonfinite_totals and row['effective_weight_mass'] <= 0): reasons.append('empty_effective_samples')
    if row['negative_tail_weight_fraction'] > FORMAL_SAFEGUARDS['negative_tail_limit']:
        reasons.append('negative_advantage_tail')
    _require(row['reasons'] == reasons, 'Incomplete numerical guard differs from its actual diagnostic counts/ratios')
    _require(row['quantile_method'] == QUANTILE_METHOD, 'Unexpected incomplete quantile method')
    for name in ('negative_log_ratio_p95', 'negative_log_ratio_p99', 'negative_log_ratio_max'):
        if row[name] is not None:
            _finite(row[name], name, nonnegative=False)
    quantiles = [row[name] for name in ('negative_log_ratio_p95', 'negative_log_ratio_p99', 'negative_log_ratio_max')]
    _require((row['negative_token_count'] == 0 and quantiles == [None, None, None]) or
             (row['negative_token_count'] > 0 and all(value is not None for value in quantiles)
              and quantiles == sorted(quantiles)), 'Incomplete numerical quantiles differ from actual negative samples')
    return {name: copy.deepcopy(row[name]) for name in GLOBAL_FIELDS}


def _audit_transaction_rows(rows_by_rank, binding, attempt, rejection_before, *, run):
    """Audit only actual executed opportunities; rolled-back calls are not writes."""
    from search_g1.actor_performance import lean_run
    lean = lean_run(run)
    observations = {}; final_rejections = []; decisions_by_rank = []; executions = []; gradients = []
    extra_monitors = run.get('extra_policy_monitor_forwards',True)
    _require(type(extra_monitors) is bool and (extra_monitors or run.get('finite_thresholds_diagnostic_only') is True or lean),
             'Omitted monitor forwards require explicit diagnostic-only or native4 lean run binding')
    for rank in ranks(run):
        rows = rows_by_rank[rank]
        decisions = [row for row in rows if row['kind']=='OPPORTUNITY']
        monitors = [row for row in rows if row['kind']=='POLICY_MONITOR']
        execution = summarize_monitor_execution(monitors)
        executions.append(execution)
        writes = [row for row in rows if row['kind']=='OPTIMIZER_WRITE']
        _require([row['opportunity_index'] for row in decisions]==list(range(attempt['opportunities_processed'])),
                 'Transaction opportunities are missing or reordered')
        counter = attempt['optimizer_writes_before']; rejections = rejection_before
        identity = None
        for mini, decision in enumerate(decisions):
            _row(decision, rank, binding, 'OPPORTUNITY')
            context = dict(logical_update=attempt['logical_update'], minibatch_index=mini,
                prior_optimizer_writes=counter, attempt_id=attempt['attempt_id'],
                attempt_index=attempt['attempt_index'],
                behavior_probability_gate_sha256=attempt['behavior_probability_gate_sha256'])
            _require(decision['context']==context and decision['processed'] is True,
                     'Transaction opportunity identity differs')
            success = decision['status']=='written'
            rollback = decision['status']=='rolled_back_guard'
            _require(not lean or (not rollback and decision['optimizer_write_rolled_back'] is False),
                     'Lean execution has no per-write rollback transaction')
            _require(decision['status'] in ('written','rolled_back_guard','skipped_guard') and
                     decision['optimizer_write_succeeded'] is success and
                     decision['optimizer_write_rolled_back'] is rollback and
                     decision['optimizer_call_attempted'] is (success or rollback),
                     'Transaction call/write/rollback disposition differs')
            _require(decision['optimizer_writes_before']==counter and
                     decision['optimizer_writes_after']==counter+int(success), 'Transaction retained write counter differs')
            local_monitors = [row for row in monitors if row['context']['minibatch_index']==mini]
            stages = (['before_minibatch', 'after_minibatch'] if success or decision['guard_stage']=='after_minibatch' else ['before_minibatch']) if extra_monitors else []
            _require([row['stage'] for row in local_monitors]==stages,
                     'Transaction monitor execution differs from actual guard boundary')
            for row in local_monitors:
                _row(row, rank, binding, 'POLICY_MONITOR')
                _require(row['context']==context and row['calibration_sha256']==binding['calibration_sha256'] and
                         row['execution_role']=='production' and row['monitoring_scope']==monitor_scope(run),
                         'Transaction monitor policy identity differs')
                if run.get('easy2560_selection') is True:
                    _require(row['objective_weight_definition'] == objective_weight_definition(run),
                             'Transaction logical objective weighting differs')
                _identity(row['monitor_identity'], monitor_count=monitor_trajectories_per_rank(run), run=run)
                if identity is None: identity = row['monitor_identity']
                _require(identity==row['monitor_identity'], 'Transaction frozen sample changed')
                statistics = _monitor_statistics(row,run.get('finite_thresholds_diagnostic_only',False))
                rejections += int(not row['allowed'])
                _require(row['rejection_count']==rejections, 'Transaction rejection counter differs')
                _require(row['optimizer_writes']==counter+int(row['stage']=='after_minibatch'),
                         'Transaction monitor provisional write counter differs')
                key=(mini,row['stage'])
                if key in observations:
                    _require(observations[key]==statistics, 'Transaction ranks disagree on global statistics')
                else: observations[key]=statistics
                if not row['allowed']:
                    _require(decision['guard_stage']==row['stage'] and decision['guard_reasons']==row['reasons'],
                             'Transaction rejected monitor lacks corresponding guard')
            selected_writes=[row for row in writes if row['context']['minibatch_index']==mini]
            _require(len(selected_writes)==int(success), 'Rolled-back physical call cannot be a retained write receipt')
            if success:
                row=selected_writes[0]; _row(row,rank,binding,'OPTIMIZER_WRITE')
                _require(row['context']==context and row['write_count']==counter+1 and
                         row['learning_rates'] and all(rate==consumed_lr(attempt['logical_update']) for rate in row['learning_rates']),
                         'Transaction accepted Adam counter or LR differs')
                gradients.append(optimizer_gradient_evidence(row, run=run, clip_message='Transaction gradient clip differs'))
            else:
                _require(mini==len(decisions)-1 and decision['guard_reasons'] and
                         decision['guard_stage'] in NUMERICAL_GUARD_STAGES | {'before_minibatch','after_minibatch'},
                         'Guard must end stale rollout reuse immediately')
            _require(decision['status']==attempt['opportunity_statuses'][mini], 'Driver and transaction worker statuses differ')
            counter += int(success)
        terminal = rows[-1]
        if not extra_monitors:
            _require(not monitors and terminal.get('actual_monitor_forward_count')==0 and
                     terminal.get('reused_monitor_forward_count')==0 and terminal.get('extra_policy_monitor_forwards') is False,
                     'No-extra-forward execution must report actual zero monitor counts')
        _row(terminal, rank, binding, 'OUTER_BUDGET' if attempt['complete_outer'] else 'ABORTED_ATTEMPT')
        _require(terminal['opportunities_processed']==len(decisions) and terminal['optimizer_writes']==counter-attempt['optimizer_writes_before'] and
                 terminal['cumulative_writes']==counter==attempt['optimizer_writes_after'], 'Transaction terminal totals differ')
        for name in ('actual_monitor_forward_count','reused_monitor_forward_count'):
            if name in terminal: _require(terminal[name]==execution[name], 'Transaction monitor execution count differs')
            if name in attempt: _require(attempt[name]==execution[name], 'Driver monitor execution count differs')
        final_rejections.append(rejections)
        decisions_by_rank.append([{key:value for key,value in row.items() if key!='rank'} for row in decisions])
    _require(all(value==decisions_by_rank[0] for value in decisions_by_rank), 'Transaction rank dispositions differ')
    _require(len(set(final_rejections))==1, 'Transaction rank rejection counts differ')
    _require(all(value==executions[0] for value in executions), 'Rank monitor execution/reuse counts differ')
    for observation in observations.values():
        if observation['kl_estimate_complete']:
            _same_arithmetic(observation['effective_weight_mass'],
                formal_monitor_geometry(run)['expected_complete_effective_weight_mass'],
                'transaction global masked-mean weight mass')
    return dict(global_monitor_count=len(observations), rejection_count=final_rejections[0],
                actual_monitor_forward_count=executions[0]['actual_monitor_forward_count'],
                reused_monitor_forward_count=executions[0]['reused_monitor_forward_count'],
                gradient_norm_measurement=dict(actual_write_receipts_checked=len(gradients),
                    measured_post_clip_receipts=sum(row['gradient_norm_after_measured'] for row in gradients),
                    explicitly_unmeasured_post_clip_receipts=sum(not row['gradient_norm_after_measured'] for row in gradients),
                    all_post_clip_bounds_measured=bool(gradients) and all(row['measured_post_clip_bound_proven'] for row in gradients),
                    scope='Reported coefficient arithmetic is not a measured post-clip norm or GPU finite-scan proof'),
                guard_triggered_monitor_count=sum(not row['allowed'] for row in observations.values()))


def _audit_monitor_rows(rows_by_rank, binding, attempt, rejection_before, *, run):
    """Require40 logical pre/post monitors and20 dispositions for this real UUID."""
    if attempt.get('recovery_protocol_version',1)==2:
        return _audit_transaction_rows(rows_by_rank,binding,attempt,rejection_before,run=run)
    observations = {}; dispositions = {}; identities = {}; final_rejections = []
    outer = attempt['logical_update']; start = attempt['optimizer_writes_before']
    successes = attempt['optimizer_writes_this_attempt']; complete = attempt['complete_outer']
    for rank in ranks(run):
        rows = rows_by_rank[rank]; cursor = 0; counter = start; identity = None; frozen = None
        rejections = rejection_before
        for mini in range(20):
            _require(cursor < len(rows), 'Missing actual attempt pre-monitor')
            before = rows[cursor]; cursor += 1; write = None
            if cursor < len(rows) and rows[cursor]['kind'] == 'OPTIMIZER_WRITE':
                write = rows[cursor]; cursor += 1
            _require(cursor + 1 < len(rows), 'Missing actual attempt post-monitor or opportunity')
            after, decision = rows[cursor:cursor + 2]; cursor += 2
            succeeded = write is not None
            context = dict(logical_update=outer, minibatch_index=mini, prior_optimizer_writes=counter,
                attempt_id=attempt['attempt_id'], attempt_index=attempt['attempt_index'],
                behavior_probability_gate_sha256=attempt['behavior_probability_gate_sha256'])
            for event in [before, after, decision] + ([] if write is None else [write]):
                _require(event['context'] == context, 'Actual attempt opportunity context or gate differs')
            _row(decision, rank, binding, 'OPPORTUNITY')
            for name in ('opportunity_index', 'optimizer_writes_before', 'optimizer_writes_after'):
                _integer(decision[name], name)
            _require(decision['processed'] is True and decision['opportunity_index'] == mini,
                     'Actual attempt opportunities are missing or reordered')
            _require(type(decision['optimizer_write_succeeded']) is bool and
                     decision['optimizer_write_succeeded'] == succeeded,
                     'Actual opportunity success differs from the real write receipt')
            _require(decision['optimizer_writes_before'] == counter and
                     decision['optimizer_writes_after'] == counter + int(succeeded),
                     'Actual opportunity write counters differ')
            guards = decision['guard_events']
            _require(isinstance(guards, list), 'Missing actual opportunity guard events')
            for event in guards:
                _require(isinstance(event, dict) and set(event) == {'stage', 'reasons'} and
                         isinstance(event['reasons'], list) and event['reasons'] and
                         all(isinstance(x, str) and x for x in event['reasons']), 'Malformed actual guard event')
                _require(event['stage'] in NUMERICAL_GUARD_STAGES | {'before_minibatch', 'after_minibatch'},
                         'Fatal or unknown stage cannot claim a recoverable skip')
                if event['stage'] in NUMERICAL_GUARD_STAGES:
                    _require(event['reasons'] == ['nonfinite_or_empty_effective_numerical_input'],
                             'Unexpected actual numerical guard reason')
                _require(not succeeded or event['stage'] == 'after_minibatch',
                         'Rejected pre-write opportunity cannot claim an Adam write')
            expected_status = ('written_with_post_guard' if guards else 'written') if succeeded else 'skipped_guard'
            _require(decision['status'] == expected_status and (succeeded or bool(guards)),
                     'Actual skip/write disposition lacks its guard')
            _require(decision['guard_stage'] == (guards[0]['stage'] if guards else None) and
                     decision['guard_reasons'] == (guards[0]['reasons'] if guards else []),
                     'Actual primary guard differs from guard events')
            _require(decision['status'] == attempt['opportunity_statuses'][mini],
                     'Driver return differs from the real opportunity disposition')
            local_decision = {key: copy.deepcopy(value) for key, value in decision.items() if key != 'rank'}
            if mini in dispositions:
                _require(dispositions[mini] == local_decision, 'Actual ranks disagree on opportunity disposition')
            else:
                dispositions[mini] = local_decision
            for row, stage, writes in ((before, 'before_minibatch', counter),
                                      (after, 'after_minibatch', counter + int(succeeded))):
                _row(row, rank, binding, 'POLICY_MONITOR')
                _integer(row['optimizer_writes'], 'monitor optimizer_writes')
                _require(row['stage'] == stage and row['optimizer_writes'] == writes,
                         'Actual monitor/write order differs')
                _require(row['calibration_sha256'] == binding['calibration_sha256'] and
                         row['execution_role'] == 'production' and row['enforcement'] == dict(_enforcement('production'),**(dict(negative_tail='observe_only',old_policy_kl='observe_only',log_ratio_bound='observe_only') if run.get('finite_thresholds_diagnostic_only',False) else {})),
                         'Actual monitor prospective plan or enforced role differs')
                _require(row['objective_weight_definition'] == objective_weight_definition(run) and
                         row['monitoring_scope'] == monitor_scope(run), 'Actual monitor weight/scope differs')
                _identity(row['monitor_identity'], monitor_count=monitor_trajectories_per_rank(run), run=run)
                if identity is None:
                    identity = copy.deepcopy(row['monitor_identity']); identities[rank] = identity
                _require(identity == row['monitor_identity'], 'Actual frozen monitor identity changed')
                statistics = _monitor_statistics(row,run.get('finite_thresholds_diagnostic_only',False))
                rejections += int(not row['allowed'])
                _require(_integer(row['rejection_count'], 'rejection_count') == rejections,
                         'Actual cumulative policy rejection counter differs')
                if row['kl_estimate_complete']:
                    fixed = tuple(statistics[name] for name in ('effective_weight_mass', 'negative_effective_weight_mass',
                        'effective_token_count', 'negative_token_count'))
                    if frozen is None:
                        frozen = fixed
                    _require(fixed == frozen, 'Actual frozen monitor weights changed')
                if not row['allowed']:
                    _require(any(event['stage'] == stage and event['reasons'] == row['reasons'] for event in guards),
                             'Rejected actual policy monitor lacks its real disposition')
                    _require(stage != 'before_minibatch' or not succeeded,
                             'Rejected actual pre-monitor cannot enter Adam')
                key = (mini, stage)
                if key in observations:
                    _require(all(observations[key][name] == statistics[name] for name in GLOBAL_FIELDS),
                             'Actual rank replicas disagree on global attempt statistics')
                else:
                    observations[key] = statistics
            for event in guards:
                if event['stage'] in ('before_minibatch', 'after_minibatch'):
                    monitor_row = before if event['stage'] == 'before_minibatch' else after
                    _require(monitor_row['allowed'] is False and monitor_row['reasons'] == event['reasons'],
                             'Policy guard does not match its actual rejected monitor')
            if not succeeded:
                _adjacent(observations[(mini, 'before_minibatch')], observations[(mini, 'after_minibatch')])
            if write is not None:
                _row(write, rank, binding, 'OPTIMIZER_WRITE')
                _integer(write['write_count'], 'actual write_count', minimum=1)
                _require(write['write_count'] == counter + 1 and write['learning_rates'] and
                         all(type(rate) in (int, float) and rate == consumed_lr(outer) for rate in write['learning_rates']),
                         'Actual retained Adam write or consumed LR differs')
                optimizer_gradient_evidence(write, run=run, clip_message='Actual global L2 clip differs')
                counter += 1
        _require(cursor == len(rows) - 1, 'Missing or extra actual attempt monitor/write/terminal receipts')
        terminal = rows[cursor]
        _row(terminal, rank, binding, 'OUTER_BUDGET' if complete else 'ABORTED_ATTEMPT')
        for key, value in dict(logical_update=outer, attempt_id=attempt['attempt_id'], attempt_index=attempt['attempt_index'],
            opportunities_processed=20, optimizer_writes=successes, cumulative_writes=counter).items():
            if type(value) is int:
                _integer(terminal[key], key)
            _require(terminal[key] == value, 'Actual attempt terminal counter differs: ' + key)
        if complete:
            for key, value in dict(local_trajectories=local_trajectories(run), local_mini=actor_local_mini(run), local_micro=actor_local_micro(run),
                optimizer_writes_before=start, optimizer_writes_after=counter).items():
                _require(terminal[key] == value, 'Actual completed budget differs: ' + key)
        else:
            _require(terminal['context'] == context and terminal['writes_before'] == start and
                     terminal['behavior_probability_gate_sha256'] == context['behavior_probability_gate_sha256'] and
                     terminal['guard_stage'] == attempt['guard_stage'] and terminal['guard_reasons'] == attempt['guard_reasons'],
                     'Actual skipped terminal identity or guard differs')
            _require(terminal['recoverable_guard'] is True and terminal['complete_outer'] is False and
                     type(terminal['scheduler_advances']) is int and terminal['scheduler_advances'] == 0 and
                     terminal['retained_partial_adam_writes'] is True and terminal['no_automatic_rollback'] is True,
                     'Actual skipped terminal cannot advance or claim rollback')
        _require(counter == attempt['optimizer_writes_after'], 'Actual retained attempt write total differs')
        final_rejections.append(rejections)
    _require(len(set(final_rejections)) == 1, 'Actual ranks disagree on policy rejection totals')
    token_count = sum(sum(identity['actor_token_lengths']) for identity in identities.values())
    for (mini, stage), observation in observations.items():
        _require(observation['effective_token_count'] == token_count, 'Actual global32 token count differs')
        if observation['kl_estimate_complete']:
            _same_arithmetic(observation['effective_weight_mass'],
                formal_monitor_geometry(run)['expected_complete_effective_weight_mass'],
                'actual global masked-mean weight mass')
        if mini and stage == 'before_minibatch':
            _adjacent(observations[(mini - 1, 'after_minibatch')], observation)
    return dict(global_monitor_count=len(observations), rejection_count=final_rejections[0],
                guard_triggered_monitor_count=sum(not row['allowed'] for row in observations.values()))


def verify_attempt_history(base, run, complete_records, scheduler_receipts, *, last, plan_sha256):
    """Verify the sealed attempt prefix ending at the requested COMPLETE outer."""
    try:
        return _verify_attempt_history(base, run, complete_records, scheduler_receipts, last=last, plan_sha256=plan_sha256)
    except EvidenceError:
        raise
    except (AssertionError, KeyError, TypeError, OverflowError, ValueError, OSError) as failure:
        raise EvidenceError('Malformed actual formal attempt history: ' + str(failure)) from failure


def _verify_attempt_history(base, run, complete_records, scheduler_receipts, *, last, plan_sha256):
    from search_g1.actor_performance import lean_run
    lean = lean_run(run)
    from reproduction.common8.behavior_runtime import read_gate_receipt
    base = Path(base).resolve(); _integer(last, 'last', minimum=1); _hash(plan_sha256, 'prospective plan')
    _require(last <= 200 and run['policy_safety_mode'] == 'inline_first_five', 'Invalid formal attempt scope')
    from search_g1.full_state_restore import verified_runtime_prefix
    prefix=verified_runtime_prefix(base,run) if run.get('full_state_restore') else dict(completed_outer=0,writes=0,attempt_index=0,rejections=0,aborted=0)
    _require([row['step'] for row in complete_records] == list(range(prefix['completed_outer']+1, last + 1)), 'Incomplete actual completed selection')
    cutoff = _integer(complete_records[-1]['attempt_index'], 'cutoff attempt', minimum=1)
    _require(set(scheduler_receipts) == {str(rank) for rank in ranks(run)}, 'Need every actual scheduler receipt')
    all_attempts = _owned_rows(base, 'logs/actual-attempts.jsonl')
    attempts = [row for row in all_attempts if _attempt_identity(row)[1] <= cutoff]
    _require(last < 200 or len(attempts) == len(all_attempts), 'Actual driver contains attempts after final COMPLETE200')
    _require([row['attempt_index'] for row in attempts] == list(range(prefix['attempt_index']+1, cutoff + 1)),
             'Actual driver attempt prefix is omitted, duplicated or reordered')
    _require(len({row['attempt_id'] for row in attempts}) == cutoff-prefix['attempt_index'], 'Actual driver attempt UUID reused')
    rank_workers = {}; rank_schedulers = {}; selected_hashes = {}; scheduler_hashes = {}
    for rank in ranks(run):
        all_scheduler = _owned_rows(base, f'logs/scheduler-rank-{rank}.jsonl')
        scheduler = [row for row in all_scheduler if _attempt_identity(row)[1] <= cutoff]
        _require(last < 200 or len(scheduler) == len(all_scheduler), 'Actual scheduler continues after final COMPLETE200')
        _require(scheduler == scheduler_receipts[str(rank)]['selected_rows'],
                 'Owned scheduler prefix differs from independently checked receipts')
        all_workers = _owned_rows(base, f'logs/optimizer-rank-{rank}.jsonl')
        workers = [row for row in all_workers if _attempt_identity(row)[1] <= cutoff]
        _require(last < 200 or len(workers) == len(all_workers), 'Actual worker continues after final COMPLETE200')
        rank_workers[rank] = workers; rank_schedulers[rank] = scheduler
        selected_hashes[str(rank)] = document_sha256(workers)
        scheduler_hashes[str(rank)] = document_sha256(scheduler)
    binding = dict(run_id=run['run_id'], calibration_sha256=plan_sha256)
    completed=prefix['completed_outer'];writes=prefix['writes'];aborted=prefix['aborted'];rejections=prefix['rejections'];summaries=[]
    scheduler_cursors = {rank: 0 for rank in ranks(run)}; worker_cursors = {rank: 0 for rank in ranks(run)}
    job = rank_schedulers[0][0]['slurm_job_id']
    for ordinal, raw_attempt in enumerate(attempts, prefix['attempt_index']+1):
        attempt = copy.deepcopy(raw_attempt); outer = completed + 1
        for name in ('logical_update', 'optimizer_writes_before', 'optimizer_writes_after',
                     'completed_outer', 'cumulative_aborted_attempts'):
            _integer(attempt[name], name)
        _require(attempt['run_id'] == run['run_id'] and attempt['slurm_job_id'] == job and
                 attempt['logical_update'] == outer and attempt['all_ranks_agreed'] is True and
                 type(attempt['real_rank_count']) is int and attempt['real_rank_count'] == world_size(run),
                 'Actual driver attempt run/rank/outer identity differs')
        if run.get('execution_backend', 'slurm') == 'ssh_local':
            _require(job is None and attempt['execution_backend'] == 'ssh_local' and
                     attempt['execution_id'] == run['execution_id'], 'Actual driver local execution identity differs')
        count = _integer(attempt['optimizer_writes_this_attempt'], 'actual attempt writes')
        version=attempt.get('recovery_protocol_version',1)
        processed=_integer(attempt['opportunities_processed'], 'opportunities_processed', minimum=1)
        _require(count <= processed <= 20 and attempt['optimizer_writes_before'] == writes and
                 attempt['optimizer_writes_after'] == writes + count, 'Actual retained write counters differ')
        complete = count > 0; full_budget=count==20 and processed==20
        _require(type(attempt['complete_outer']) is bool and attempt['complete_outer'] is complete and
                 attempt['status'] == ('complete' if full_budget or (version==1 and complete) else 'partial' if complete else 'recoverable_guard') and
                 type(attempt['scheduler_advances']) is int and attempt['scheduler_advances'] == int(complete) and
                 attempt['completed_outer'] == completed + int(complete), 'Actual driver safe progress or warmup differs')
        statuses = attempt['opportunity_statuses']
        _require(isinstance(statuses, list) and len(statuses) == processed and
                 all(status in ('written', 'written_with_post_guard', 'skipped_guard','rolled_back_guard') for status in statuses) and
                 sum(status in ('written','written_with_post_guard') for status in statuses) == count,
                 'Actual driver opportunity returns differ')
        if version==2:
            _require(attempt['full_budget_complete'] is full_budget and attempt['opportunities_unattempted']==20-processed and
                     attempt['current_step_rollback'] is (not lean) and attempt['no_automatic_rollback'] is lean and
                     (not lean or attempt.get('optimizer_rollbacks_this_attempt',0)==0),
                     'Transaction driver full budget/rollback semantics differ')
        else:
            _require(processed==20 and attempt['guard_stage'] == (None if complete else 'all_update_opportunities_skipped') and
                     attempt['guard_reasons'] == ([] if complete else ['no_successful_adam_write']) and
                     attempt['retained_partial_adam_writes'] is (not complete) and attempt['no_automatic_rollback'] is True,
                     'Historical driver semantics differ')
        aborted += int(not complete)
        _require(attempt['cumulative_aborted_attempts'] == aborted, 'Actual driver retry counter differs')
        # A paired restore supplies only the current execution's suffix rows;
        # `completed` includes the authenticated parent prefix. Index relative
        # to that boundary rather than skipping real suffix receipts.
        record = complete_records[completed - prefix['completed_outer']] if complete else None
        if record is not None:
            _require(record['run_id'] == run['run_id'] and record['attempt_id'] == attempt['attempt_id'] and
                     record['attempt_index'] == ordinal and record['optimizer_writes_before'] == writes and
                     record['optimizer_writes_after'] == writes + count and record['opportunities_processed'] == processed,
                     'Actual completed update differs from the real driver attempt')
            _require(type(record['actual_optimizer_writes']) is int and record['actual_optimizer_writes'] == count and
                     record['metrics']['actor/optimizer_steps_this_update'] == count and
                     record['metrics']['actor/update_opportunities_this_attempt'] == processed and
                     record['metrics']['actor/skipped_update_opportunities'] == 20 - count,
                     'Actual scalar update ledger differs from retained Adam writes/opportunities')
            gate = read_gate_receipt(base, run, outer, update_record=record)
        else:
            gate = read_gate_receipt(base, run, outer, attempt_id=attempt['attempt_id'], attempt_index=ordinal)
        attempt['behavior_probability_gate_sha256'] = gate['proof_sha256']
        rows_by_rank = {}
        expected_kinds = ['BEGIN', 'COMMIT_PREPARED', 'COMPLETE'] if complete else ['BEGIN', 'ABORTED_ATTEMPT']
        for rank in ranks(run):
            cursor = scheduler_cursors[rank]; rows = rank_schedulers[rank]
            selected = rows[cursor:cursor + len(expected_kinds)]
            _require([row['kind'] for row in selected] == expected_kinds, 'Actual scheduler attempt is partial or reordered')
            for row in selected:
                _require(_attempt_identity(row) == (attempt['attempt_id'], ordinal) and row['run_id'] == run['run_id'] and
                         row['rank'] == rank and row['slurm_job_id'] == job and row['logical_update'] == outer and
                         row['batch_gate_sha256'] == gate['proof_sha256'], 'Actual scheduler attempt identity/gate differs')
                if run.get('execution_backend', 'slurm') == 'ssh_local':
                    _require(row['execution_backend'] == 'ssh_local' and row['execution_id'] == run['execution_id'],
                             'Actual scheduler local execution identity differs')
                end = row['kind'] != 'BEGIN'; advanced = row['kind'] == 'COMPLETE'
                _require(row['writes_before'] == writes and row['writes_after'] == writes + (count if end else 0) and
                         row['actual_optimizer_writes'] == (count if end else 0) and
                         row['opportunities_processed'] == (processed if end else 0) and row['optimizer_opportunities'] == 20 and
                         row['completed_outer_before'] == completed and row['completed_outer_after'] == completed + int(advanced) and
                         row['scheduler_advances'] == int(advanced), 'Actual scheduler differs from the real driver attempt')
            scheduler_cursors[rank] += len(selected)
            cursor = worker_cursors[rank]; rows = rank_workers[rank]; selected = []
            while cursor < len(rows) and _attempt_identity(rows[cursor]) == (attempt['attempt_id'], ordinal):
                selected.append(rows[cursor]); cursor += 1
            worker_cursors[rank] = cursor; rows_by_rank[rank] = selected
        audit = _audit_monitor_rows(rows_by_rank, binding, attempt, rejections, run=run)
        if 'actual_monitor_forward_count' not in audit:
            executions=[summarize_monitor_execution([row for row in rows_by_rank[rank]
                        if row['kind']=='POLICY_MONITOR']) for rank in ranks(run)]
            _require(all(value==executions[0] for value in executions), 'Rank monitor execution/reuse counts differ')
            audit.update(**{key:value for key,value in executions[0].items() if key!='logical_monitor_count'})
        if version==1: _require(audit['global_monitor_count'] == 40, 'Need40 historical monitors for every attempt')
        rejections = audit['rejection_count']
        summaries.append(dict(attempt_id=attempt['attempt_id'], attempt_index=ordinal, logical_update=outer,
            complete_outer=complete, full_budget_complete=full_budget, opportunities_processed=processed,
            actual_optimizer_writes=count, global_monitor_count=audit['global_monitor_count'],
            actual_monitor_forward_count=audit['actual_monitor_forward_count'],
            reused_monitor_forward_count=audit['reused_monitor_forward_count'],
            guard_triggered_monitor_count=audit['guard_triggered_monitor_count']))
        writes += count; completed += int(complete)
    _require(completed == last and attempts[-1]['complete_outer'] is True, 'Actual attempt prefix does not end at the requested COMPLETE')
    for rank in ranks(run):
        _require(scheduler_cursors[rank] == len(rank_schedulers[rank]) and worker_cursors[rank] == len(rank_workers[rank]),
                 'Extra or unbound actual scheduler/worker attempt receipts')
    selection = dict(plan_sha256=plan_sha256, driver_attempts=attempts,
                     selected_rank_rows_sha256=selected_hashes, selected_scheduler_rows_sha256=scheduler_hashes)
    return dict(schema_version=1, role='actual_formal_all_attempt_monitor_audit', run_id=run['run_id'],
        structural_accepted=True, completed_outer_count=completed, cutoff_attempt_index=cutoff,
        actual_attempt_count=len(attempts), source_completed_outer=prefix['completed_outer'],
        source_actual_optimizer_writes_per_rank=prefix['writes'], fully_skipped_attempt_count=aborted-prefix['aborted'],
        processed_opportunities_per_rank=sum(row['opportunities_processed'] for row in summaries), actual_optimizer_writes_per_rank=writes,
        completed_outer_global_monitor_count=sum(row['global_monitor_count'] for row in summaries if row['complete_outer']),
        retry_global_monitor_count=sum(row['global_monitor_count'] for row in summaries if not row['complete_outer']),
        unique_global_monitor_count=sum(row['global_monitor_count'] for row in summaries), cumulative_policy_rejections=rejections,
        actual_monitor_forward_count=sum(row['actual_monitor_forward_count'] for row in summaries),
        reused_monitor_forward_count=sum(row['reused_monitor_forward_count'] for row in summaries),
        guard_triggered_monitor_count=sum(row['guard_triggered_monitor_count'] for row in summaries),
        attempts=summaries, selected_driver_attempts_sha256=document_sha256(attempts),
        selected_rank_rows_sha256=selected_hashes, selected_scheduler_rows_sha256=scheduler_hashes,
        selected_evidence_sha256=document_sha256(selection), production_admission=False,
        scope='sealed actual attempt prefix; global policy-monitor rank replicas count once; '
              'actor loss keeps the original first-rank collector semantics; structural audit is not accuracy success')
