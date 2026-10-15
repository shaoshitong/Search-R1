"""Publish a local two-observation sampling-only verified wait after FULL6..199.

No collection, scheduling, training, evaluation, network, or pending-FULL repair.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ALIAS_SHA = 'f8775e2673d2844125a6683622880066cf870a3e6070a1ac847f5bb62ed8aafc'
CORE_SHA = '676dca88489fb4d3c84891603990fc96256a14fb7da8ac2bc5c2def053e016c1'


def dependency():
    path = HERE/'reconcile_nested_full_current_aliases.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != ALIAS_SHA:
        raise ValueError('Published787 alias dependency changed')
    spec = importlib.util.spec_from_file_location('published787_alias', path)
    alias = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alias)
    core, _, paths = alias.dependencies()
    return alias, core, paths


def sampling_observation(core, alias, poll, resource, native, full, outer):
    progress = alias.identity(core, poll, outer)
    ps, rs, active = core.validate_unit(poll, resource, outer)
    core.validate_restore(poll)
    metric = core.validate_metric(poll, outer)
    ranks = core.validate_ranks(poll, metric, outer)
    _, guard, completed, totals = core.validate_native(native, progress, outer)
    core.need(active and metric == full['iteration_metrics']
              and completed == full['per_rank_successful_adam']
              and totals == full['per_rank_actual_total_adam']
              and poll.get('direct_rank_records') == full['direct_rank_record_count'],
              'Actual FULL/rank/native boundary differs', 'REJECTED_INPUT_NO_WRITE')
    core.need(not any(m.get('step', 0) > outer for m in poll.get('iteration_metrics', []))
              and not any(r.get('outer_step', 0) > outer for r in poll.get('last_rank_summaries', [])),
              'Later metric or rank update exists; this is not sampling-only', 'SAMPLING_ONLY_PHASE_ENDED_NO_WRITE')
    rollouts = poll.get('rollout_cost')
    core.need(isinstance(rollouts, list) and all(isinstance(row, dict) for row in rollouts),
              'Original Controller ledger observation is required', 'REJECTED_INPUT_NO_WRITE')
    core.need(not any(row.get('candidate_outer', 0) > outer for row in rollouts),
              'Next Controller result exists; use its actual phase instead of sampling-only wait',
              'SAMPLING_ONLY_PHASE_ENDED_NO_WRITE')
    core.need(any(row.get('candidate_outer') == outer and row.get('status') == 'TRAINING_ROLLOUT_COST_COMPLETE'
                  for row in rollouts), 'Latest original Controller boundary is not confirmed', 'REJECTED_INPUT_NO_WRITE')
    stage = (poll.get('logs') or {}).get('stage.log') or ''
    entries = [int(step) for step in re.findall(r'epoch\s+\d+,\s*step\s+(\d+)\b', stage)]
    core.need(outer+1 in entries and max(entries) == outer+1,
              'Actual next-stage entry differs', 'SAMPLING_ONLY_PHASE_ENDED_NO_WRITE')
    watcher = resource.get('state') or {}
    core.need(not watcher or (watcher.get('training_source_commit') == core.SOURCE
              and watcher.get('training_unit') == core.UNIT and watcher.get('run_directory') == core.RUN
              and watcher.get('expected_training_invocation_id') == core.INVOCATION
              and not watcher.get('last_errors') and not watcher.get('audit_error_count')),
              'Original source-bound resource watcher differs', 'REJECTED_INPUT_NO_WRITE')
    cpu_ns, cpu_seconds = alias.service_cpu(core, rs)
    return dict(epoch=poll['epoch'], completed_outer=outer, training_attempts=progress['training_attempts_seen'],
        actual_adam_per_rank=totals, completed_adam_per_rank=completed,
        full_metric_outer=metric['step'], rank_outer=outer, rank_count=len(ranks),
        latest_original_Controller_outer=outer, next_Controller_result_present=False,
        actual_next_stage=outer+1, main_pid=ps['MainPID'], invocation_id=rs['InvocationID'],
        CPUUsageNSec=cpu_ns, service_CPU_seconds=cpu_seconds,
        direct_child_rank_record_count=poll['direct_rank_records'], all_skipped_attempts=guard['refused_attempts'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', required=True, type=int)
    for name in ('first-poll', 'first-resource', 'second-poll', 'second-resource', 'workspace'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    alias, core, lock_paths = dependency()
    n = args.outer;workspace = args.workspace.resolve();base = workspace/core.BASE_REL
    core.need(6 <= n <= 199, 'Only actual complete FULL6..199 is supported', 'REJECTED_INPUT_NO_WRITE')
    inputs = alias.Inputs(workspace, core)
    for path in (Path(__file__).resolve(),HERE/'reconcile_nested_full_current_aliases.py',*lock_paths):
        raw=path.read_bytes();inputs.pins.append(dict(path=str(path),bytes=len(raw),sha256=core.digest(raw)))
    first, first_pin = inputs.read(args.first_poll)
    resource1, resource1_pin = inputs.read(args.first_resource)
    second, second_pin = inputs.read(args.second_poll)
    resource2, resource2_pin = inputs.read(args.second_resource)
    full, full_pin = inputs.read(base/f'FULL_OUTER{n}_PROGRESS_RECEIPT.json')
    core.need(all(full.get(k) == v for k,v in dict(source_commit=core.SOURCE,config_sha256=core.CONFIG,
              unit=core.UNIT,run=core.RUN,restored_outer=5,completed_outer=n,
              actual_training_pid=int(core.PID),actual_training_invocation_id=core.INVOCATION).items())
              and full.get('full_iteration_metric_pending') is False
              and full.get('local_control_script_sha256') == CORE_SHA,
              'Already published complete FULL identity differs', 'REJECTED_INPUT_NO_WRITE')
    bindings = full.get('input_bindings') or {}
    original, original_pin = inputs.bound(bindings.get('poll'))
    original_resource, _ = inputs.bound(bindings.get('resource'))
    native, native_pin = inputs.bound(bindings.get('native_readback'))
    core.validate_unit(original, original_resource, n)
    core.validate_restore(original)
    original_metric = core.validate_metric(original, n)
    core.validate_ranks(original, original_metric, n)
    _, _, completed, totals = core.validate_native(native, alias.identity(core, original, n), n)
    core.need(original_metric == full['iteration_metrics']
              and completed == full['per_rank_successful_adam'] and totals == full['per_rank_actual_total_adam']
              and native['manifest_sha256'] == full['native_manifest_sha256'],
              'FULL is not bound to its original actual records', 'REJECTED_INPUT_NO_WRITE')
    facts1 = sampling_observation(core, alias, first, resource1, native, full, n)
    facts2 = sampling_observation(core, alias, second, resource2, native, full, n)
    core.need(original['epoch'] <= first['epoch'] < second['epoch'] and first_pin['path'] != second_pin['path']
              and core.number(resource1.get('epoch')) and core.number(resource2.get('epoch'))
              and resource1['epoch'] < resource2['epoch'] and resource1_pin['path'] != resource2_pin['path']
              and facts1['actual_adam_per_rank'] == facts2['actual_adam_per_rank']
              and facts2['CPUUsageNSec'] >= facts1['CPUUsageNSec'],
              'Two actual same-instance observations do not advance', 'REJECTED_INPUT_NO_WRITE')
    sizes1, sizes2 = first.get('own_output_files') or {}, second.get('own_output_files') or {}
    growth = {}
    for name in ('stage.log','retriever.log'):
        core.need(core.integer(sizes1.get(name)) and core.integer(sizes2.get(name))
                  and sizes2[name] > sizes1[name], 'Both actual logs must grow', 'REJECTED_INPUT_NO_WRITE')
        growth[name] = sizes2[name]-sizes1[name]
    stems = [Path(pin['path']).stem for pin in (first_pin, second_pin)]
    core.need(all(re.fullmatch(r'[A-Za-z0-9_.-]{1,90}', stem) for stem in stems),
              'Poll names cannot form a local receipt name', 'REJECTED_INPUT_NO_WRITE')
    receipt_path = base/('VERIFIED_WAIT_'+'_'.join(stems)+'.json')
    core.need(not receipt_path.exists(), 'Immutable same-pair wait receipt already exists', 'RECEIPT_EXISTS_NO_WRITE')
    canonical = [workspace/'outputs/STATUS.json',workspace/'outputs/tree-nested-branch/STATUS.json',
                 base/'CURRENT_NATIVE_RECOVERY_FOLLOWUP.json',base/'CURRENT_OBSERVER_CONNECTION_STATE.json']
    mirrors = [workspace/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',
               Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
    core.need(all(path.is_dir() for path in mirrors), 'Both existing local mirrors are required', 'REJECTED_INPUT_NO_WRITE')
    previous = [(path,*inputs.read(path)) for path in canonical]
    mirror_previous = []
    for mirror in mirrors:
        for name in ('STATUS.json','CURRENT_NATIVE_RECOVERY_FOLLOWUP.json','CURRENT_OBSERVER_CONNECTION_STATE.json'):
            value,pin = core.held(mirror/name);inputs.pins.append(pin)
            mirror_previous.append((mirror/name,value,pin))
    alias.monotonic_all(core, previous+mirror_previous, n, second['epoch'])
    for _,value,_ in previous+mirror_previous:
        core.need(value.get('goal_status') == 'active' and value.get('goal_active',True) is not False
                  and value.get('source_commit',core.SOURCE) == core.SOURCE,
                  'Current goal/source differs; never change a pause', 'STALE_SNAPSHOT_NO_WRITE')
        for key in ('completed_outer','current_outer','actual_optimizer_completed_outer','last_confirmed_native_saved_outer'):
            if key in value:
                core.need(value[key] == n, 'Complete FULL alias reconciliation is required first', 'REJECTED_INPUT_NO_WRITE')
    now = datetime.now(timezone.utc).isoformat()
    snapshot = base/f'verified_wait_prior_{stems[0]}_{stems[1]}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}'
    receipt = dict(schema='nested_actual_sampling_verified_wait_v1',status='ACTUAL_TWO_POLL_SAMPLING_VERIFIED_WAIT',
        created_at_utc=now,source_commit=core.SOURCE,config_sha256=core.CONFIG,unit=core.UNIT,
        main_pid=int(core.PID),invocation_id=core.INVOCATION,completed_outer=n,active_outer=n+1,
        observed_epochs=[first['epoch'],second['epoch']],polls=[first_pin,second_pin],
        resource_observed_epochs=[resource1['epoch'],resource2['epoch']],
        resources=[resource1_pin,resource2_pin],full=full_pin,original_full_poll=original_pin,native=native_pin,
        original_native_manifest_sha256=native['manifest_sha256'],actual_observations=[facts1,facts2],
        log_byte_growth=growth,log_byte_sizes=[{k:sizes1[k] for k in growth},{k:sizes2[k] for k in growth}],
        CPUUsageNSec=[facts1['CPUUsageNSec'],facts2['CPUUsageNSec']],
        service_CPU_seconds=[facts1['service_CPU_seconds'],facts2['service_CPU_seconds']],
        service_CPU_delta_seconds=(facts2['CPUUsageNSec']-facts1['CPUUsageNSec'])/1_000_000_000,
        next_outer_Controller_complete_observed=False,next_outer_credit_observed=False,
        next_outer_partial_Adam=None,next_outer_sampling_complete=None,next_outer_variance=None,next_outer_scoring=None,
        counter_scope='Actual FULL/nativeN and Adam unchanged across both polls; next-stage/log growth is not next outer completion or write credit.',
        goal_status='active',goal_turn_classification='verified_wait',consecutive_blocked_goal_turns=0,
        prior_snapshot_manifest=str(snapshot/'SNAPSHOT_MANIFEST.json'),
        source_script_sha256=core.digest(Path(__file__).read_bytes()),alias_dependency_sha256=ALIAS_SHA,
        core_dependency_sha256=CORE_SHA,finalizer_dependency_sha256=alias.FINALIZER_SHA,
        remote_actions=False,training_source_or_inputs_modified=False)
    receipt_raw = core.json_bytes(receipt)
    receipt_pin = dict(path=str(receipt_path),bytes=len(receipt_raw),sha256=core.digest(receipt_raw))
    label=f'ACTUAL_VERIFIED_WAIT_FULL{n}_NEXT{n+1}_SAMPLING'
    common = dict(status=label,phase=label,current_phase=label,training_phase=label,
        observed_epoch=second['epoch'],last_observed_epoch=second['epoch'],last_confirmed_epoch=second['epoch'],
        last_actual_second_snapshot_epoch=second['epoch'],
        last_actual_poll_epoch=second['epoch'],last_current_progress_poll_epoch=second['epoch'],
        last_actual_poll=second_pin['path'],last_actual_poll_sha256=second_pin['sha256'],
        last_current_progress_poll=second_pin['path'],last_rank_observation=second_pin['path'],
        current_resource_observation=resource2_pin,last_confirmed_resource_epoch=resource2['epoch'],
        current_systemd=second['systemd'],latest_guard=second['latest_guard'],
        current_remote_state='KNOWN_AT_TWO_ACTUAL_SAMPLING_POLLS',current_remote_health_known=True,
        current_observation_unknown=False,current_observation_error=None,current_failure_evidence=None,
        current_observation_failure_evidence=None,same_instance_confirmed=True,same_instance_repoll_required=False,
        current_observation_status=label,
        observed_current_training_phase=f'Actual next{n+1} stage with both logs growing; no new Controller result or next complete outer in either poll.',
        current_live_sampling_progress=receipt_pin,current_active_rollout_progress=receipt_pin,
        current_latest_sampling_interval_log_growth=growth,
        current_latest_sampling_interval_scope='Two actual same-instance stage/retriever increments; next complete credit/partial Adam/variance/scoring are not inferred.',
        active_outer_sampling_complete=None,active_outer_sampling_complete_confirmed=None,
        active_outer_actual_partial_Adam_calls=None,current_partial_optimizer_calls_per_rank=None,
        active_outer_advantage_variance=None,next_outer_sampling_complete=None,next_outer_scoring=None,
        current_active_outer_full_metric_pending=True,last_nonterminal_service_CPUUsageNSec=facts2['CPUUsageNSec'],
        last_nonterminal_service_CPU_seconds=facts2['service_CPU_seconds'],
        goal_turn_classification='verified_wait',whole_goal_turn_classification='verified_wait',
        consecutive_blocked_goal_turns=0,current_consecutive_blocked_goal_turns=0,blocked_audit_consecutive_goal_turns=0,
        current_observation_training_action='none',latest_verified_wait_receipt=receipt_pin,
        updated_at=now,recorded_at_utc=now)
    outputs = {receipt_path:receipt_raw}
    for path,old,_ in previous:
        value = copy.deepcopy(old);value.update(copy.deepcopy(common));outputs[path] = core.json_bytes(value)
    report_paths = [workspace/'outputs/REPORT.md',workspace/'outputs/tree-nested-branch/REPORT.md']
    for path in report_paths:
        raw = path.read_bytes()
        core.need(len(raw) <= 12*1024**2, 'Prior report exceeds small local limit', 'REJECTED_INPUT_NO_WRITE')
        inputs.pins.append(dict(path=str(path),bytes=len(raw),sha256=core.digest(raw)))
        note = (f'Latest verified wait: actual FULL/native{n} remains complete; next{n+1} sampling continues in original PID{core.PID}/Invocation{core.INVOCATION}. '
                f'Two actual epochs {first["epoch"]}/{second["epoch"]}: stage +{growth["stage.log"]}B, retriever +{growth["retriever.log"]}B. '
                f'Actual service CPUUsageNSec {facts2["CPUUsageNSec"]} /1e9 = {facts2["service_CPU_seconds"]}s; this is a nonterminal service observation. '
                'No next Controller completion, rank update or credit is published; next sampling completion/partial Adam/variance/scoring remain unknown. '
                'FULL counters, native boundary, cost, curve and dev below are unchanged. Goal active; verified_wait/block0.\n\n')
        outputs[path] = note.encode()+raw
    for mirror in mirrors:
        for source in (canonical[1],canonical[2],canonical[3],report_paths[1],receipt_path):
            outputs[mirror/source.name] = outputs[source]
    prior_raw = {}
    for path in outputs:
        if path.exists():
            raw = path.read_bytes();core.need(len(raw) <= 12*1024**2,'Mutable document exceeds small local limit')
            prior_raw[path] = raw
    core.need(len(prior_raw) == 14, 'Fourteen existing canonical/mirror documents required', 'REJECTED_INPUT_NO_WRITE')
    with lock_paths[0].open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inputs.unchanged()
        core.need(all(core.digest(path.read_bytes()) == core.digest(raw) for path,raw in prior_raw.items()),
                  'Concurrent canonical/mirror write', 'CONCURRENT_UPDATE_NO_WRITE')
        core.need(not any(path.exists() for path in (receipt_path,*[m/receipt_path.name for m in mirrors])),
                  'Same actual pair already published', 'RECEIPT_EXISTS_NO_WRITE')
        snapshot.mkdir();manifest=[]
        for index,(path,raw) in enumerate(prior_raw.items()):
            member=snapshot/f'{index:02d}-{path.name}.gz';member.write_bytes(gzip.compress(raw,mtime=0));stored=member.read_bytes()
            core.need(gzip.decompress(stored)==raw,'Prior gzip does not restore original bytes')
            manifest.append(dict(source=str(path),original_bytes=len(raw),original_sha256=core.digest(raw),
                stored_gzip_path=str(member),stored_bytes=len(stored),stored_sha256=core.digest(stored),
                gzip_mtime=0,exact_original_bytes_readback=True))
        (snapshot/'SNAPSHOT_MANIFEST.json').write_bytes(core.json_bytes(dict(files=manifest,created_at_utc=now)))
        for path,raw in outputs.items():core.atomic_replace(path,raw)
        core.need(all(path.read_bytes()==raw for path,raw in outputs.items()),'Local verified-wait readback differs')
    print(json.dumps(dict(status='LOCAL_SAMPLING_VERIFIED_WAIT_AND_TWO_MIRRORS_PUBLISHED',
                          receipt=receipt_pin,snapshot=str(snapshot),modified_files=len(outputs),remote_actions=False)))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status=getattr(error,'status','REJECTED_LOCAL_VERIFIED_WAIT'),
                              error=str(error),remote_actions=False)),file=sys.stderr)
        raise SystemExit(2)
