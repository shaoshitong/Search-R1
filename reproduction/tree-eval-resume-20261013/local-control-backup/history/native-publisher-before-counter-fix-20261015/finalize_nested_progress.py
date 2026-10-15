"""Finish local status/report publication from an immutable FULL and later poll.

No SSH, process actions, tensors, remote reads or training modifications. The
original c413 validation logic is unchanged; FULL receipts are retained byte-for-byte.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import re
import sys

import sync_nested_native_progress as core

ORIGINAL_VALIDATOR_SHA = 'c413fc582c2aaa46fefb00aa7d9cf1a368f3655ec5059a9192c7efba3d09d07a'
DEPENDENCY_SHA = '020bec1d2f1f06aa2149ad5cc1d1c72f41567c3b8cd10de10193b9a7d1a964c6'
need = core.need


def expand_cost_fields(costs, cost_pin, bound):
    """Rebuild only explicitly split top-level fields, retaining the root pin."""
    decoded = copy.deepcopy(costs)
    pins = []
    root = Path(cost_pin['path']).parent
    blocks = (root/'cumulative_cost_blocks').resolve()
    need(blocks.is_relative_to(root.resolve()), 'Cost blocks resolve outside root directory', 'REJECTED_INPUT_NO_WRITE')

    def unique_mapping(pairs):
        value = {}
        for key, item in pairs:
            need(key not in value, 'Duplicate JSON key in cost field part', 'REJECTED_INPUT_NO_WRITE')
            value[key] = item
        return value

    for field, descriptor in costs.items():
        if not isinstance(descriptor, dict) or descriptor.get('storage') != 'complete_original_field_parts':
            continue
        kind = descriptor.get('container_type');entries = descriptor.get('original_entries')
        parts = descriptor.get('parts')
        need(kind in ('mapping', 'list') and type(entries) is int and entries >= 0
             and isinstance(parts, list) and (parts or entries == 0),
             'Invalid split cost field descriptor: ' + field, 'REJECTED_INPUT_NO_WRITE')
        restored = {} if kind == 'mapping' else []
        for index, part_descriptor in enumerate(parts):
            need(isinstance(part_descriptor, dict) and isinstance(part_descriptor.get('file'), str)
                 and core.sha64(part_descriptor.get('sha256'))
                 and type(part_descriptor.get('bytes')) is int and part_descriptor['bytes'] > 0
                 and part_descriptor.get('json_selector') == 'value',
                 'Invalid cost field part descriptor: ' + field, 'REJECTED_INPUT_NO_WRITE')
            relative = Path(part_descriptor['file'])
            need(not relative.is_absolute() and '..' not in relative.parts
                 and len(relative.parts) > 1 and relative.parts[0] == 'cumulative_cost_blocks',
                 'Cost field part path outside blocks: ' + field, 'REJECTED_INPUT_NO_WRITE')
            path = (root/relative).resolve()
            need(path.is_relative_to(blocks) and path.suffix == '.json',
                 'Resolved cost field part outside blocks: ' + field, 'REJECTED_INPUT_NO_WRITE')
            _, pin = bound(dict(file=str(path), sha256=part_descriptor['sha256'], bytes=part_descriptor['bytes']))
            need(Path(pin['path']).is_relative_to(blocks), 'Held cost field part outside blocks', 'REJECTED_INPUT_NO_WRITE')
            raw = Path(pin['path']).read_bytes()
            need(core.digest(raw) == pin['sha256'], 'Cost field part changed during readback', 'CONCURRENT_UPDATE_NO_WRITE')
            part = json.loads(raw, object_pairs_hook=unique_mapping)
            value = part.get('value')
            need(part.get('schema') == 'cumulative_scalar_field_part_v1' and part.get('original_field') == field
                 and isinstance(value, dict if kind == 'mapping' else list),
                 'Cost field part schema/field/container differs: ' + field, 'REJECTED_INPUT_NO_WRITE')
            if kind == 'mapping':
                need(not (restored.keys() & value.keys()), 'Duplicate mapping key across cost field parts', 'REJECTED_INPUT_NO_WRITE')
                restored.update(value)
            else:
                restored.extend(value)
            pins.append(dict(pin, original_field=field, part_index=index, json_selector='value'))
        need(len(restored) == entries, 'Split cost field entry count differs: ' + field, 'REJECTED_INPUT_NO_WRITE')
        decoded[field] = restored
    return decoded, pins


def validate_next_activity(first, second, native, outer, growth, first_pin, second_pin):
    """Use measured log growth or a newly completed, poll-bound next rollout."""
    need(all(value is not None and value >= 0 for value in growth.values()),
         'Second observed log sizes are missing or decreased')
    evidence = dict(poll_bindings=[first_pin, second_pin], observed_epochs=[first['epoch'], second['epoch']],
                    next_outer=outer+1, optimizer_completion_inferred=False, additional_adam_credit=0)
    if any(value > 0 for value in growth.values()):
        return dict(evidence, kind='ACTUAL_SAME_INSTANCE_LOG_BYTE_GROWTH', log_byte_growth=growth)
    attempts = (first.get('progress') or {}).get('training_attempts_seen')
    need(core.integer(attempts) and attempts >= outer
         and (second.get('progress') or {}).get('training_attempts_seen') == attempts,
         'Original next rollout attempt boundary differs')
    expected_attempt = int(attempts)+1
    records = second.get('rollout_cost') or []
    need(isinstance(records, list), 'Second rollout cost records are invalid')
    candidates = [(index, row) for index, row in enumerate(records)
                  if isinstance(row, dict) and row.get('candidate_outer') == outer+1]
    need(len(candidates) == 1, 'Second actual sampling activity has not yet advanced')
    index, row = candidates[0]
    identity = native.get('identity') or {}
    need(row.get('status') == 'TRAINING_ROLLOUT_COST_COMPLETE' and row.get('error') is None
         and row.get('attempt_id') == expected_attempt and row.get('completed_outer_before') == outer
         and row.get('code_commit') == core.SOURCE and row.get('config_sha256') == core.CONFIG
         and row.get('job_id') == core.JOB and row.get('run_id') == Path(core.RUN).name
         and all(core.sha64(identity.get(key)) and row.get(key) == identity[key]
                 for key in ('data_identity', 'model_identity')),
         'Next completed rollout boundary/source/config/identity differs', 'REJECTED_INPUT_NO_WRITE')
    prior = first.get('rollout_cost') or []
    need(isinstance(prior, list) and not any(isinstance(item, dict) and item.get('attempt_id') == expected_attempt
                                           for item in prior), 'Next rollout was already present in the first observation')
    end = row.get('end_epoch');start = row.get('start_epoch')
    need(core.number(start) and core.number(end) and start < end
         and first['epoch'] < end < second['epoch']
         and core.number(row.get('wall_seconds')) and row['wall_seconds'] > 0,
         'Next rollout completion is not strictly between the two actual observations')
    costs = row.get('costs') or {}
    need(isinstance(costs, dict) and core.integer(row.get('question_rows')) and row['question_rows'] > 0
         and all(core.integer(costs.get(key)) and costs[key] > 0
                 for key in ('generated_tokens', 'prefill_tokens', 'retrieval_calls', 'trajectories'))
         and row.get('outer_credit_not_inferred') is True and row.get('actual_optimizer_writes') is None,
         'Next rollout lacks positive generation/retrieval costs or separates no optimizer credit')
    return dict(evidence, kind='ACTUAL_NEXT_ROLLOUT_COMPLETION_BETWEEN_POLLS',
                record_json_selector=f'rollout_cost[{index}]', record=copy.deepcopy(row),
                record_canonical_sha256=core.digest(core.json_bytes(row)),
                native_identity_binding=dict(data_identity=identity['data_identity'], model_identity=identity['model_identity']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--outer', type=int, required=True)
    parser.add_argument('--second-poll', type=Path, required=True)
    parser.add_argument('--second-resource', type=Path, required=True)
    for name in ('full', 'cost-through', 'candidate', 'rates', 'registry',
                 'curve', 'initial-supplement', 'gzip-receipt'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve();base = workspace/core.BASE_REL;n = args.outer
    need(core.RESTORE < n <= core.TARGET, 'Only actual child6..200 is allowed')
    dependency = Path(core.__file__).resolve()
    need(core.digest(dependency.read_bytes()) == DEPENDENCY_SHA, 'Pinned local publisher dependency changed', 'REJECTED_INPUT_NO_WRITE')
    loaded = []

    def read(path):
        path = path if path.is_absolute() else workspace/path
        value, pin = core.held(path);loaded.append((value, pin))
        return value, pin

    def bound(descriptor, default_root=workspace):
        need(isinstance(descriptor, dict) and core.sha64(descriptor.get('sha256')), 'Missing bound local descriptor')
        p = Path(descriptor.get('path') or descriptor.get('file') or '')
        need(str(p) != '.', 'Missing bound path')
        value, pin = read(p if p.is_absolute() else default_root/p)
        need(pin['sha256'] == descriptor['sha256'] and pin['bytes'] == descriptor.get('bytes'),
             'Bound local bytes differ: ' + str(p), 'REJECTED_INPUT_NO_WRITE')
        return value, pin

    full, full_pin = read(args.full or base/f'FULL_OUTER{n}_PROGRESS_RECEIPT.json')
    need(full.get('source_commit') == core.SOURCE and full.get('config_sha256') == core.CONFIG
         and full.get('unit') == core.UNIT and full.get('run') == core.RUN
         and full.get('restored_outer') == 5 and full.get('completed_outer') == n
         and full.get('actual_training_invocation_id') == core.INVOCATION,
         'FULL source/config/outer/original instance differs', 'REJECTED_INPUT_NO_WRITE')
    bindings = full.get('input_bindings') or {}
    first, first_pin = bound(bindings.get('poll'))
    native, native_pin = bound(bindings.get('native_readback'))
    resource, resource_pin = bound(bindings.get('resource'))
    # Reuse the unchanged FULL validator; no alternative weaker completion rule.
    core.validate_unit(first, resource, n)
    restore, provenance = core.validate_restore(first)
    progress = first.get('progress') or {}
    need(progress.get('actual_completed_outer_from_attempt_results') == n and not progress.get('errors'), 'Original credited progress differs')
    metric = core.validate_metric(first, n);ranks = core.validate_ranks(first, metric, n)
    native_record, guard, completed, totals = core.validate_native(native, progress, n)
    need(full.get('iteration_metrics') == metric and full.get('per_rank_successful_adam') == completed
         and full.get('per_rank_actual_total_adam') == totals
         and full.get('native_manifest_sha256') == native['manifest_sha256'], 'FULL disagrees with its original bound evidence')
    second, second_pin = read(args.second_poll)
    resource2, resource2_pin = read(args.second_resource)
    for poll in (first, second):
        need(poll.get('source_commit') == core.SOURCE and poll.get('unit') == core.UNIT
             and poll.get('job_id') == core.JOB and poll.get('run') == core.RUN and poll.get('node') == 'h100-10',
             'Poll identity differs', 'REJECTED_INPUT_NO_WRITE')
    ps, rs, active = core.validate_unit(second, resource2, n)
    core.validate_restore(second)
    watcher = resource2.get('state') or {}
    need(not watcher or (watcher.get('training_source_commit') == core.SOURCE
         and watcher.get('training_unit') == core.UNIT and watcher.get('run_directory') == core.RUN
         and watcher.get('expected_training_invocation_id') == core.INVOCATION), 'Second source-bound watcher identity differs')
    need(not watcher.get('last_errors') and not watcher.get('audit_error_count'), 'Second watcher has an audit observation')
    epoch = second.get('epoch')
    need(core.number(epoch) and epoch > first.get('epoch', epoch), 'Second real observation is not later')
    p2 = second.get('progress') or {}
    need(p2.get('actual_completed_outer_from_attempt_results') == n and not p2.get('errors'),
         'Second snapshot is pending/failed or already advanced past this selected boundary')
    need(core.counts(p2.get('completed_actual_optimizer_calls_per_rank'), 'Second credited counts') == completed,
         'Second cumulative actual calls differ')
    sizes1, sizes2 = first.get('own_output_files') or {}, second.get('own_output_files') or {}
    growth = {name:sizes2[name]-sizes1[name] if core.integer(sizes1.get(name)) and core.integer(sizes2.get(name)) else None
              for name in ('stage.log', 'retriever.log')}
    activity = None
    if n < core.TARGET:
        stage = (first.get('logs') or {}).get('stage.log') or ''
        need(active and second.get('run_result') is None
             and re.search(r'epoch\s+\d+,\s*step\s+' + str(n+1) + r'\b', stage), 'Original actual next-stage evidence missing')
        activity = validate_next_activity(first, second, native, n, growth, first_pin, second_pin)
        next_outer, final = n+1, None
    else:
        next_outer, final = None, core.final_phase(second, resource2, active)

    costs, cost_pin = read(args.cost_through or base/f'CUMULATIVE_COST_THROUGH{n}.json')
    costs, cost_field_part_pins = expand_cost_fields(costs, cost_pin, bound)
    candidate, candidate_pin = read(args.candidate or base/f'CANDIDATE_COST_EXTENSION_OUTER{n:04d}.json')
    rates, rates_pin = read(args.rates or base/f'BRANCH_FALLBACK_RATES_OUTER{n:04d}.json')
    registry, registry_pin = read(args.registry or base/'CANDIDATE_COST_EXTENSION_INPUTS.json')
    need(costs.get('through_effective_outer') == n and costs.get('target_outer') == core.TARGET
         and costs.get('origin') == 'fresh', 'Cost snapshot boundary/origin differs')
    cost_full = (costs.get('immutable_metadata_input_hashes') or {}).get(Path(full_pin['path']).name) or {}
    need(cost_full.get('sha256') == full_pin['sha256'] and cost_full.get('bytes') == full_pin['bytes'], 'Cost snapshot does not bind this FULL')
    rows = costs.get('rows_counted_once') or []
    need(len(rows) == n and sorted(r.get('outer') for r in rows) == list(range(1, n+1)), 'Cost effective outer coverage differs')
    need(all(r.get('source_commit') == ('0befbaf99fc44942105831b3fbec94e4bf1b8ccb' if r['outer'] == 1
             else 'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4' if r['outer'] <= 5 else core.SOURCE) for r in rows), 'Cost ancestry differs')
    accounting = costs.get('successful_optimizer_accounting') or {}
    need(core.counts(accounting.get(f'successful_calls_per_rank_through{n}'), 'Cost actual cumulative calls') == completed,
         'Cost snapshot actual Adam counts disagree')
    need(candidate.get('source_commit') == rates.get('source_commit') == core.SOURCE
         and candidate.get('config_sha256') == rates.get('config_sha256') == core.CONFIG
         and candidate.get('run_directory') == core.RUN and candidate.get('actual_completed_cost_outers') == [n]
         and rates.get('outer') == n, 'Current candidate/rate identity differs')
    parts = [v for v in candidate.get('parts', []) if v.get('outer') == n]
    need(len(parts) == 1, 'Current candidate part is missing')
    part, part_pin = bound(parts[0], base)
    need(part.get('status') == 'ACTUAL_COMPLETED_CANDIDATE_SCALARS_AGGREGATED'
         and part.get('source_commit') == core.SOURCE and part.get('config_sha256') == core.CONFIG
         and part.get('requested_outer') == n, 'Candidate scalar part is not actual complete')
    for descriptor, pin in ((rates.get('source_part'), part_pin), (rates.get('source_full_outer_receipt'), full_pin),
                            (rates.get('cumulative_snapshot'), cost_pin)):
        _, actual = bound(descriptor)
        need(actual['sha256'] == pin['sha256'], 'Rates join a different snapshot')
    indexes = [v for v in registry.get('indexes', []) if v.get('outer') == n]
    need(len(indexes) == 1 and indexes[0].get('sha256') == candidate_pin['sha256']
         and indexes[0].get('bytes') == candidate_pin['bytes'], 'Candidate registry entry is pending')

    previous = core.previous_snapshots(workspace, base);core.monotonic(previous, n, epoch)
    statuses = [(v.get('last_observed_epoch', v.get('observed_epoch', 0)), v) for p,v,_ in previous if p.name == 'STATUS.json']
    need(statuses, 'Existing current STATUS required')
    old_status = max(statuses, key=lambda item:item[0])[1]
    old_followup = next((v for p,v,_ in previous if p.name == 'CURRENT_NATIVE_RECOVERY_FOLLOWUP.json'), {})
    actual_dev, latest_dev, dev_seconds, _ = core.dev_evidence(first, old_status, metric, n)
    latest_scores = dict(outer=latest_dev['outer'] if latest_dev else None,
        f1_percent=100*latest_dev['metrics']['val/f1/all'] if latest_dev else None,
        em_percent=100*latest_dev['metrics']['val/em/all'] if latest_dev else None,
        current_outer_dev_complete=actual_dev is not None)
    wall = metric.get('timing_s/step') if metric is not None else None
    ordinary = wall-dev_seconds if dev_seconds is not None else wall if n % 5 else None
    variances = full.get('current_training_variances') or {}
    need(all(value is None or core.number(value) for value in variances.values()), 'Invalid original variance scalars')
    expected_dev = (n//5+1)*5 if n < core.TARGET else None
    proof_path = base/(f'OUTER{n}_FULL_NEXT{next_outer}_ACTUAL_PROGRESS.json' if next_outer
                       else 'OUTER200_TRAINING_COMPLETE_FINAL_PHASE_ACTUAL_PROGRESS.json')
    need(not proof_path.exists(), 'Existing immutable finalization proof retained', 'RECEIPT_EXISTS_NO_WRITE')
    attachments = {}
    for label, p in (('learning_curve_complete', args.curve), ('initial1_10_supplement', args.initial_supplement),
                     ('local_old_raw_log_gzip_archive', args.gzip_receipt)):
        if p is not None:
            _, attachments[label] = read(p)
    proof = dict(status=f'ACTUAL_FULL{n}_NATIVE{n}_TWO_OBSERVATIONS_LOCAL_SUMMARY_FINALIZED',
        source_commit=core.SOURCE,config_sha256=core.CONFIG,unit=core.UNIT,main_pid=int(rs['MainPID']),
        invocation_id=core.INVOCATION,observed_epochs=[first['epoch'],epoch],completed_outer=n,active_outer=next_outer,
        same_original_instance=True,log_byte_growth=growth,continued_progress_evidence=activity,
        log_growth_scope='Measured stage/retriever byte differences only; zero remains zero. Continued progress evidence is separate; no next-outer optimizer completion or partial Adam credit inferred.',
        full_outer_receipt=full_pin,snapshots=[first_pin,second_pin,resource_pin,resource2_pin],native_readback=native_pin,
        native_manifest_sha256=native['manifest_sha256'],successful_adam_per_rank=completed,
        current_dev512=actual_dev,latest_dev_outer=latest_scores['outer'],latest_actual_dev512_scores=latest_scores,
        current_full_outer_seconds=wall,current_dev_seconds=dev_seconds,current_training_archive_seconds_excluding_dev=ordinary,
        current_training_variances=variances,cumulative_cost=cost_pin,cumulative_cost_field_parts=cost_field_part_pins,
        candidate_index=candidate_pin,
        candidate_current_supplement=part_pin,current_branch_fallback_rates=rates_pin,candidate_registry=registry_pin,
        candidate_cost_join_scope=rates.get('cumulative_boundary'),attachments=attachments,final_phase_evidence=final,
        original_validator_sha256=ORIGINAL_VALIDATOR_SHA,current_validator_dependency_sha256=DEPENDENCY_SHA,
        finalizer_sha256=core.digest(Path(__file__).read_bytes()),
        loaded_native_restore_outer=5,no_new_restore_performed=True,missing_rank_opportunity_fields_remain_null=True,
        errors=[],run_result=second.get('run_result'),new_training_submitted=False,running_source_hot_modified=False,
        additional_review_or_test=False,exact_executor_resume=False,goal_status='active',goal_complete=False)
    rel = lambda pin:str(Path(pin['path']).relative_to(workspace)) if Path(pin['path']).is_relative_to(workspace) else pin['path']
    common = dict(status='TRAINING_RUNNING' if next_outer else 'TRAIN200_COMPLETE_FINAL_WORK_PENDING',
        phase='TRAINING_RUNNING' if next_outer else 'TRAIN200_COMPLETE_FINAL_WORK_PENDING',
        health=(f'Full{n}/native{n} complete; actual PID{rs["MainPID"]}/Invocation{rs["InvocationID"]}; '
            + (f'outer{next_outer} continues in the same instance; ' if next_outer else f'final phase {(final or {}).get("kind")}; ')
            + f'second progress outer{p2["actual_completed_outer_from_attempt_results"]}; '
            + f'progress errors={bool(p2.get("errors"))}; terminal observed={bool(not active or second.get("run_result") is not None or p2.get("terminal"))}.'),
        current_observation_status=(f'ACTUAL_FULL{n}_NEXT{next_outer}_PROGRESS_CONFIRMED' if next_outer
                                    else f'ACTUAL_FULL{n}_FINAL_PHASE_CONFIRMED'),
        current_remote_health_known=True,
        unit=core.UNIT,main_pid=int(rs['MainPID']),invocation_id=rs['InvocationID'],
        source_commit=core.SOURCE,current_outer=n,completed_outer=n,last_confirmed_outer=n,last_confirmed_chain_outer=n,
        actual_optimizer_completed_outer=n,new_completed_outer=n-5,last_durable_completed_outer=n,
        latest_native_checkpoint_outer=n,latest_native_manifest_sha256=native['manifest_sha256'],
        observed_epoch=epoch,last_observed_epoch=epoch,last_actual_poll=rel(second_pin),
        current_next_outer_progress=next_outer,second_snapshot=Path(second_pin['path']).name,
        current_next_outer_progress_evidence=activity,
        second_snapshot_sha256=second_pin['sha256'],last_actual_second_snapshot_epoch=epoch,
        actual_next_outer_entered=next_outer,active_outer_in_progress=next_outer,active_outer=next_outer,
        current_outer_full_metric=metric,current_iteration_metrics=metric,
        last_complete_full_metric_outer=n if metric is not None else old_status.get('last_complete_full_metric_outer'),
        current_outer_full_metric_pending=metric is None,current_full_outer_metric_pending=metric is None,
        full_iteration_log_pending=metric is None,current_outer_wall_seconds=wall,latest_full_outer_wall_seconds=wall,
        current_ordinary_training_seconds_excluding_dev=ordinary,latest_ordinary_training_seconds_excluding_dev=ordinary,
        latest_training_archive_wall_seconds_excluding_dev=ordinary,current_training_variances=variances,
        current_variances=variances,last_complete_training_variances=variances,latest_training_variance_outer=n,
        latest_intra_adv_population_variance=variances.get('tree_variance/intra_adv_population_variance'),
        latest_actual_total_adv_population_variance=variances.get('tree_variance/group_actual_adv_population_variance'),
        latest_within_root_raw_reward_population_variance=variances.get('tree_variance/within_root_raw_reward_population_variance'),
        latest_mixed_reward_question_fraction=variances.get('tree_variance/mixed_reward_question_fraction'),
        last_full_outer_progress_receipt=Path(full_pin['path']).name,last_full_receipt=rel(full_pin),
        last_full_receipt_sha256=full_pin['sha256'],last_full_outer_progress_receipt_sha256=full_pin['sha256'],
        current_full_outer_progress_receipt=Path(full_pin['path']).name,last_progress_receipt=Path(full_pin['path']).name,
        current_completed_progress_evidence=rel(dict(path=str(proof_path))),
        successful_adam_per_rank=completed,actual_total_adam_per_rank=totals,
        last_confirmed_chain_adam_per_rank=completed,
        child_successful_adam_per_rank={rank:value-100 for rank,value in completed.items()},last_rank_summaries=ranks,
        actual_outer_successful_writes_per_rank=full.get('actual_outer_successful_writes_per_rank'),
        actual_outer_skipped_opportunities_per_rank=full.get('actual_outer_skipped_opportunities_per_rank'),
        actual_outer_processed_opportunities_per_rank=full.get('actual_outer_processed_opportunities_per_rank'),
        loaded_native_restore_outer=5,native_restore_step=5,actual_current_restore_outer=5,
        actual_current_restore_adam_per_rank=restore['driver']['completed_optimizer_calls_per_rank'],
        actual_current_restore_scope=f'Actual cold-loaded boundary remains native5; latest writer-complete boundary is{n}. No new cold restore is claimed.',
        actual_dev512=actual_dev,current_dev512=actual_dev,normal_dev512_wall_seconds=dev_seconds,
        current_dev512_wall_seconds=dev_seconds,normal_dev512_pending=n % 5 == 0 and actual_dev is None,
        expected_next_dev_outer=expected_dev,latest_dev512=latest_dev,latest_actual_dev512_scores=latest_scores,
        latest_dev512_outer=latest_scores['outer'],latest_dev512_f1_percent=latest_scores['f1_percent'],
        latest_dev512_em_percent=latest_scores['em_percent'],
        dev_score_updated=bool(actual_dev and full.get('dev_score_updated_this_turn')),
        dev_score_updated_this_turn=bool(actual_dev and full.get('dev_score_updated_this_turn')),
        cumulative_cost_summary=rel(cost_pin),cumulative_cost_summary_sha256=cost_pin['sha256'],per_outer_costs_through=n,
        local_cost_append_latest_actual_outer=n,
        candidate_cost_extension_registry=rel(registry_pin),candidate_cost_extension_registry_sha256=registry_pin['sha256'],
        candidate_cost_extension_latest_complete_outer=n,candidate_cost_extension_latest_part=rel(part_pin),
        candidate_cost_extension_latest_part_sha256=part_pin['sha256'],initial_state_fallback_rates=rates,
        current_branch_fallback_rates_receipt=rel(rates_pin),current_branch_fallback_rates_receipt_sha256=rates_pin['sha256'],
        current_systemd=dict(ps, InvocationID=rs['InvocationID']),actual_training_progress=p2,finalized_progress_input_bindings=proof['snapshots'],
        recent_iteration_metrics=second.get('iteration_metrics'),latest_guard=second.get('latest_guard'),
        current_native_manifest_history=watcher or None,native_manifest_history=watcher or None,
        native_watcher_systemd=core.state(resource2['watcher_systemd']) if resource2.get('watcher_systemd') is not None else None,
        legacy_unbound_native_collector_observation=second.get('native_manifest_history'),
        current_candidate_cost_join_scope=rates.get('cumulative_boundary'),
        current_candidate_cost_scalars=full.get('current_candidate_cost_scalars'),goal_status='active',goal_complete=False,
        formal_effect_complete=False,submitted_this_turn=False,new_training_submitted=False,
        running_source_hot_modified=False,extra_prelaunch_review=False,exact_executor_resume=False)
    common[f'dev{n}_result_pending'] = common[f'normal_dev512_outer{n}_pending'] = common['normal_dev512_pending']
    common['score_updated'] = common['dev_score_updated_this_turn']
    common[f'current_outer{n}_completed_progress_evidence'] = common['current_completed_progress_evidence']
    if dev_seconds is not None:common['latest_completed_dev512_wall_seconds'] = dev_seconds
    updated = copy.deepcopy(old_status);updated.update(common)
    followup = copy.deepcopy(old_followup);followup.update(common)
    followup.update(unit=core.UNIT,main_pid=int(rs['MainPID']),invocation_id=core.INVOCATION,
        target_outer=200,restored_outer=5,normal_training_continues=next_outer is not None)
    for value in (updated, followup):
        value['local_finalizer_attachments'] = attachments
        for label, pin in attachments.items():
            value[label] = rel(pin);value[label+'_sha256'] = pin['sha256']
    report = (f'Current nested Tree200 on original H10018: actual complete outer{n}/200; '
        + ((f'outer{next_outer} rollout generation completed in the same original instance; optimizer completion remains uncredited.'
            if activity['kind'] == 'ACTUAL_NEXT_ROLLOUT_COMPLETION_BETWEEN_POLLS'
            else f'outer{next_outer} sampling in the same original instance.') if next_outer else 'training200 complete; final work remains.')
        + f'\n\nTwo actual observations at{first["epoch"]} and{epoch}; same PID{rs["MainPID"]}/Invocation{core.INVOCATION}. '
        + f'Observed log byte growth {growth}; this does not credit the next outer. '
        + f'Continued progress evidence {json.dumps(activity)}.\n\n'
        + f'Actual cumulative credited Adam/rank {completed}; actual totals including known partial writes {totals}. '
        + f'Original loaded native5 stays5; latest native writer boundary{n}. Missing per-rank opportunity fields remain null.\n\n'
        + f'Current dev512 {"complete" if actual_dev else "null"}; latest actual dev{latest_scores["outer"]}: '
        + f'F1 {latest_scores["f1_percent"]}%, EM {latest_scores["em_percent"]}%. '
        + f'Current outer wall {wall}s; current dev {dev_seconds}s; training/archive excluding current dev {ordinary}s.\n\n'
        + f'Original current training variances {json.dumps(variances)}.\n\n'
        + f'Actual cumulative cost through{n}: {rel(cost_pin)}; candidate{n} supplement {rel(part_pin)}; '
        + f'fallback rates {rel(rates_pin)}. {rates.get("cumulative_boundary", "")}\n\n'
        + f'B1/B2 initial-state fallback fractions: {json.dumps(rates.get("by_stage"))}. '
        + 'These are derived flag rates, not guaranteed same-reward or failure labels.\n\n'
        + f'Attachments {json.dumps(attachments)}.\n\n'
        + 'Origin0bef outer1, ba9 outer2–5 and778 outer6 onward keep their provenance. Failed original2/Ray6/eb6 costs stay separate without duplicate credit. '
        + 'Native PT writer metadata is not independent cold reload. Profile costs are separate. Final200/51713/HF/tokenizer/independent acceptance/cleanup remain separate; goal active. '
        + 'No new training, SSH, source/config change, review or test was performed by this local publisher.\n')
    proof_raw = core.json_bytes(proof);common_proof_sha = core.digest(proof_raw)
    for value in (updated, followup):
        value['current_completed_progress_sha256'] = common_proof_sha
        value[f'current_outer{n}_completed_progress_sha256'] = common_proof_sha
    outputs = {proof_path:proof_raw,base/f'FINALIZED_OUTER{n}_PROGRESS_REPORT.md':report.encode(),
        workspace/'outputs/STATUS.json':core.json_bytes(updated),workspace/'outputs/tree-nested-branch/STATUS.json':core.json_bytes(updated),
        workspace/'outputs/REPORT.md':report.encode(),workspace/'outputs/tree-nested-branch/REPORT.md':report.encode(),
        base/'CURRENT_NATIVE_RECOVERY_FOLLOWUP.json':core.json_bytes(followup)}
    # Use the original publisher's lock too; do not race a new FULL/status writer.
    with dependency.open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for _, pin in loaded + [(v,pin) for _,v,pin in previous]:
            need(core.digest(Path(pin['path']).read_bytes()) == pin['sha256'], 'Input/status changed during preparation', 'CONCURRENT_UPDATE_NO_WRITE')
        core.monotonic(core.previous_snapshots(workspace, base), n, epoch)
        need(not proof_path.exists(), 'Another publisher finalized this outer', 'CONCURRENT_UPDATE_NO_WRITE')
        for path, raw in outputs.items():
            if path == proof_path:
                with path.open('xb') as stream:stream.write(raw);stream.flush();core.os.fsync(stream.fileno())
            else:core.atomic_replace(path, raw)
        mirrors = []
        for mirror in (workspace/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',
                       Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')):
            if mirror.is_dir():
                for source in (workspace/'outputs/tree-nested-branch/STATUS.json',workspace/'outputs/tree-nested-branch/REPORT.md',
                               proof_path,base/f'FINALIZED_OUTER{n}_PROGRESS_REPORT.md',base/'CURRENT_NATIVE_RECOVERY_FOLLOWUP.json'):
                    core.atomic_replace(mirror/source.name, outputs[source])
                mirrors.append(str(mirror))
    print(json.dumps(dict(status='LOCAL_NESTED_PROGRESS_FINALIZED',outer=n,proof=str(proof_path),proof_sha256=common_proof_sha,
                          finalizer_sha256=proof['finalizer_sha256'],mirrors=mirrors,remote_actions=False)))


if __name__ == '__main__':
    try:main()
    except core.Observation as error:
        print(json.dumps(dict(status=error.status,reason=str(error),successful_receipt_written=False,status_or_report_written=False,remote_actions=False)))
        sys.exit(2)
    except Exception as error:
        print(json.dumps(dict(status='LOCAL_INPUT_OR_IO_ERROR',reason=repr(error),automatic_retry=False,remote_actions=False)))
        sys.exit(1)
