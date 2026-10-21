"""Explicitly invoked, read-only follower for a manifest-declared generation.

Default action is local plan only. once/follow never launches or stops remote
work and writes only its own local immutable observations/small summary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from . import guarded_rollover_control as source
from . import queued_generation_control as relay
from . import guarded_generation_read as observer

control = source.original.old
KNOWN_HISTORY = (
    ('node10-native-failures-20261021-v1', '2c3a7de77b48c08435bed4df6e045be5a44b39bfc276249925962e5848b6613d'),
    ('node10-primary-v1-pre-model-terminal-20261021', '1397fbde7efee668f13dc31581001a05b15652aa172dc7fb567e0b284bd64751'),
    ('node10-primary-v2-readiness-repair-20261021', '714aadbcaae9140a9c5cba7c0ae731677c12cc57699738ef5171ea06d31b8368'))


def registry(generation='owner-v2', instance='r1'):
    rows = []
    def add(kind, node, role, slot, manifest):
        rows.append(dict(key=kind + ':' + (slot or role), kind=kind, node=node, role=role, slot=slot,
            unit=manifest['unit'], directory=manifest['directory'], source_bundle_sha256=manifest['source_bundle_sha256']))
    add('source', 'h100-5', 'owner', None, source.bundle('owner', generation)[2])
    for slot in source.slots.DEFAULT_SLOTS:add('source', 'h100-5', 'model', slot, source.bundle('slot', generation, slot=slot, instance=instance)[2])
    for role in ('upstream', 'downstream'):
        add('relay', 'h100-5' if role == 'upstream' else 'h100-10', role, None, relay.bundle(role, generation)[2])
    for slot in relay.slots.DEFAULT_SLOTS:add('node10', 'h100-10', 'model', slot, relay.bundle('slot', generation, slot=slot, instance=instance)[2])
    assert len(rows) == len({row['unit'] for row in rows}) == 17
    return rows


def metadata_value(item):return (item or {}).get('value') or {}


def summarize(nodes, rows, minimum=1):
    attention, units, channel_errors = [], {}, {}
    node10 = nodes.get('h100-10') or {};bank = node10.get('global_sealed')
    finished = bool(bank and bank.get('questions') == 169028 and bank.get('trajectories') == 845140 and bank.get('sealed_batches') == 1322 and not bank.get('errors'))
    for expected in rows:
        raw = nodes.get(expected['node']) or {}
        if raw.get('observation_error'):
            channel_errors[expected['node']] = raw['observation_error'];continue
        item = (raw.get('units') or {}).get(expected['key']) or {};native = item.get('native') or {}
        if item.get('unit') != expected['unit'] or not native.get('Id'):
            channel_errors[expected['node']] = 'Native unit observation missing; no exit inferred';continue
        attempts = item.get('attempts') or []
        progress, complete, initialized, phases, failures, latest_seal, launches, typed_ooms = {}, {}, [], [], [], None, [], []
        allocation_sha = None
        for attempt in attempts:
            allocation = metadata_value(attempt.get('allocation'))
            if allocation.get('native_unit') != expected['unit']:continue
            allocation_sha = (attempt.get('allocation') or {}).get('sha256')
            progress = metadata_value(attempt.get('progress')) or progress
            complete = metadata_value(attempt.get('complete')) or complete
            latest_seal = attempt.get('latest_own_seal') or latest_seal
            if attempt.get('failure'):failures.append(attempt['failure'])
            launches += [x for x in attempt.get('launch_failures') or [] if x]
            typed_ooms += [x for x in attempt.get('typed_initialization_ooms') or [] if x]
            initialized += [dict(path=x['path'], sha256=x['sha256'], **{key:metadata_value(x).get(key) for key in
                ('actual_model_initialized', 'actual_compute_dtype', 'max_model_len', 'actual_num_gpu_blocks')}) for x in attempt.get('model_initialized') or [] if x]
            for claim in attempt.get('active_phases') or []:
                for phase, detail in (claim.get('phases') or {}).items():
                    value = metadata_value(detail);binding = value.get('device_binding') or {}
                    phases.append(dict(batch=claim['batch'], shard=claim['shard'], phase=phase,
                        completed=value.get('completed_trajectories', 0), utc=value.get('utc'),
                        actual_sampled_tokens=value.get('actual_sampled_tokens'), progress_path=(detail or {}).get('path'),
                        lock_target=claim['lock_target'], lock_owner=claim['child'],
                        current_allocation=binding.get('native_unit') == expected['unit']
                        and binding.get('physical_gpu_indices') == allocation.get('model_indices')
                        and binding.get('gpu_uuids') == allocation.get('model_cuda_uuids')))
        counts = progress or complete;own = counts.get('completed_own_batches', 0)
        previous_input = Path(progress.get('latest_input') or '')
        previous, previous_shard = previous_input.stem, previous_input.parent.name
        seal = metadata_value(latest_seal)
        sealed_own = bool(type(own) is int and own >= minimum and previous and seal.get('batch') == previous and seal.get('run_id') == control.RUN_ID)
        next_flow = [x for x in phases if x['current_allocation'] and (x['shard'], x['batch']) != (previous_shard, previous)
                     and type(x['completed']) is int and x['completed'] > 0]
        state = 'NOT_STARTED' if native.get('LoadState') == 'not-found' else native.get('ActiveState', 'UNKNOWN')
        if native.get('ActiveState') == 'failed' or failures:
            attention.append(dict(kind='ROOT_ATTENTION_NEW_NATIVE_FAILURE', key=expected['key'], unit=expected['unit'], native=native, failures=failures))
        elif expected['role'] == 'model' and complete and not finished:
            attention.append(dict(kind='ROOT_ATTENTION_RENEW_INSTANCE', key=expected['key'], unit=expected['unit'], complete=complete, global_count_known=bank is not None))
        elif native.get('LoadState') != 'not-found' and native.get('ActiveState') == 'inactive' and item.get('submitted') and not complete:
            attention.append(dict(kind='ROOT_ATTENTION_NATIVE_ENDED_WITHOUT_COMPLETE', key=expected['key'], unit=expected['unit'], native=native))
        if item.get('submitted') and not item.get('staged_manifest_matches'):
            attention.append(dict(kind='ROOT_ATTENTION_REGISTRY_STAGE_MISMATCH', key=expected['key'], unit=expected['unit']))
        units[expected['key']] = dict(unit=expected['unit'], role=expected['role'], state=state, native=native,
            allocation_sha256=allocation_sha, own_complete_batches=own, questions=counts.get('questions', 0), trajectories=counts.get('trajectories', 0),
            initialized=any(x['actual_model_initialized'] is True for x in initialized), model_initialized=initialized,
            latest_input=progress.get('latest_input'), latest_own_seal=latest_seal, lock_held_phases=phases,
            whole_and_next_observed=sealed_own and bool(next_flow), next_actual_batch_proofs=next_flow,
            native_complete=bool(complete), failures=failures, launch_failures=launches, typed_initialization_ooms=typed_ooms)
    history = node10.get('historical') or {}
    for error in history.get('proof_errors') or []:attention.append(dict(kind='ROOT_ATTENTION_HISTORICAL_PROOF_MISMATCH', detail=error))
    for failure in history.get('failures') or []:
        if failure['classification'] == 'UNKNOWN_FAILURE_REQUIRES_ROOT':attention.append(dict(kind='ROOT_ATTENTION_UNKNOWN_FAILURE', detail=failure))
    if bank and bank.get('errors'):attention.append(dict(kind='ROOT_ATTENTION_SEALED_METADATA_ERROR', errors=bank['errors']))
    if finished:attention.append(dict(kind='ROOT_ATTENTION_FINAL_DATA_AUDIT_REQUIRED', metadata_counts_only=True))
    return dict(status='READ_ONLY_GUARDED_GENERATION_FOLLOW', epoch=time.time(), run_id=control.RUN_ID,
        units=units, global_sealed=bank, historical=history, observation_channel_errors=channel_errors,
        model_units_expected=sum(x['role'] == 'model' for x in rows),
        models_initialized=sum(x['role'] == 'model' and x['initialized'] for x in units.values()),
        models_whole_and_next_observed=sum(x['role'] == 'model' and x['whole_and_next_observed'] for x in units.values()),
        attention=attention, global_goal_complete=False, remote_jobs_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'once', 'follow'), nargs='?', default='plan')
    parser.add_argument('--generation', default='owner-v2');parser.add_argument('--instance', default='r1')
    parser.add_argument('--hours', type=float, default=4);parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--own-batches', type=int, default=1)
    parser.add_argument('--history-proof', action='append', default=[], metavar='PATH=SHA256', help='Additional original terminal proof pin; known12 are included by default')
    args = parser.parse_args()
    assert 0 < args.hours <= 24 and args.interval >= 60 and args.own_batches > 0
    rows = registry(args.generation, args.instance)
    pins = [dict(path=str(Path(control.PHYSICAL) / 'preparation' / name / 'complete.json'), sha256=sha) for name, sha in KNOWN_HISTORY]
    for value in args.history_proof:
        path, sha = value.rsplit('=', 1);assert re.fullmatch('[0-9a-f]{64}', sha);pins.append(dict(path=path, sha256=sha))
    plan = dict(registry=rows, historical_proof_pins=pins, actual_running_claimed=False, remote_jobs_modified=False)
    if args.action == 'plan':print(json.dumps(plan, indent=2));return 0
    script = Path(observer.__file__).read_text();code_sha = hashlib.sha256(script.encode()).hexdigest()
    tag = control.digest(control.encoded(plan))[:16];root = control.ROOT / 'work' / ('qwen7b-guarded-follow-' + tag);root.mkdir(exist_ok=True)
    with(root / ('plan-%d.json' % time.time_ns())).open('x') as stream:json.dump(dict(plan, observer_sha256=code_sha), stream, indent=2)
    until = time.monotonic() + args.hours * 3600;accepted = {}
    while time.monotonic() < until:
        if args.action == 'follow' and until - time.monotonic() < 225:break
        cycle = time.monotonic();nodes, originals = {}, {}
        for node in ('h100-5', 'h100-10'):
            envelope = dict(node=node, epoch=time.time(), observer_sha256=code_sha)
            values = dict(REGISTRY=rows, NODE=node, BASE=control.BASE, PHYSICAL=control.PHYSICAL, RUN_ID=control.RUN_ID, HISTORY_PROOFS=pins)
            try:
                result = control.transport('\n'.join(key + '=' + repr(value) for key, value in values.items()) + '\n' + script, node)
                envelope.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                if result.returncode:raise RuntimeError('Read transport rc%d: %s' % (result.returncode, result.stderr[-700:]))
                raw = json.loads(result.stdout)
                if raw.get('status') != 'READ_ONLY_GUARDED_GENERATION' or raw.get('node') != node or raw.get('run_id') != control.RUN_ID:raise ValueError('Snapshot identity differs')
                nodes[node] = raw
            except Exception as error:
                envelope['observation_error'] = str(error)[-1200:];nodes[node] = dict(observation_error=envelope['observation_error'])
            actual = root / ('actual-%s-%d.json' % (node, time.time_ns()))
            with actual.open('x') as stream:json.dump(envelope, stream, indent=2)
            originals[node] = str(actual)
        summary = summarize(nodes, rows, args.own_batches);summary['actual'] = originals
        for key, row in summary['units'].items():
            if row['whole_and_next_observed']:accepted[key] = dict(unit=row['unit'], allocation_sha256=row['allocation_sha256'], own_complete_batches=row['own_complete_batches'], proofs=row['next_actual_batch_proofs'], actual=originals)
        summary['accepted_whole_and_next'] = accepted
        snapshot = root / ('summary-%d.json' % time.time_ns())
        with snapshot.open('x') as stream:json.dump(summary, stream, indent=2)
        temporary = root / ('latest.tmp-%d.json' % time.time_ns());temporary.write_text(json.dumps(summary, indent=2) + '\n');temporary.replace(root / 'latest-small.json')
        print(json.dumps(summary), flush=True)
        if summary['attention']:return 2
        if args.action == 'once':return 3 if summary['observation_channel_errors'] else 0
        time.sleep(max(0., min(args.interval - (time.monotonic() - cycle), until - time.monotonic())))
    print('ROOT_ATTENTION_READ_ONLY_FOLLOW_WINDOW_ENDED_NO_NATIVE_JOB_CHANGED', flush=True);return 3


if __name__ == '__main__':raise SystemExit(main())
