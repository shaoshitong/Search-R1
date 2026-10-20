"""Bounded local read-only follow of the exact mixed node10 primary cohort.

Inherits the caller's transport environment. Owns no remote process or master;
only immutable local observations and one local latest-small summary are saved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from . import shared_vram_control as control
from .operational_reads import node10_primary_current_compact as observer


def value(metadata):
    if isinstance(metadata, dict) and isinstance(metadata.get('value'), dict):
        return metadata['value']
    return {}


def summarize(raw, actual, minimum):
    slots, attention = {}, []
    for slot, instance in observer.INSTANCES.items():
        expected = 'sst-q7b-g5-primary10-' + slot + '-' + instance + '-' + control.RUN_ID[:12] + '.service'
        row = raw['units'].get(slot) or {}
        if row.get('unit') != expected:
            raise ValueError('Observed cohort unit differs for ' + slot)
        native = row.get('native') or {}
        attempts = row.get('attempts') or []
        phases, models, failures, launch_failures = [], [], [], []
        progress, complete = {}, {}
        for attempt in attempts:
            allocation = attempt.get('allocation') or {}
            if allocation.get('native_unit') != expected:
                continue
            progress = value(attempt.get('progress')) or progress
            complete = value(attempt.get('complete')) or complete
            failure = value(attempt.get('failure'))
            if failure:
                failures.append(dict(directory=attempt['directory'], failure=failure))
            for item in attempt.get('launch_failures') or []:
                if value(item):
                    # A constructor OOM can be handled by the existing finite
                    # retry. Preserve its receipt without calling a live,
                    # productive replacement sampler a native failure.
                    launch_failures.append(dict(path=item.get('path'), failure=value(item)))
            for item in attempt.get('model_initialized') or []:
                model = value(item)
                if model:
                    models.append(dict(path=item.get('path'), mtime_ns=item.get('mtime_ns'),
                        actual_model_initialized=model.get('actual_model_initialized'),
                        actual_compute_dtype=model.get('actual_compute_dtype'),
                        max_model_len=model.get('max_model_len'), actual_num_gpu_blocks=model.get('actual_num_gpu_blocks'),
                        actual_rank_profile_proofs=model.get('actual_rank_profile_proofs')))
            for claim in attempt.get('active_phases') or []:
                for phase, files in (claim.get('phases') or {}).items():
                    metadata = files.get('rank_progress') or {}
                    detail = value(metadata)
                    if not detail:
                        continue
                    binding = detail.get('device_binding') or {}
                    phases.append(dict(batch=claim['batch'], shard=claim.get('shard'), phase=phase,
                        completed=detail.get('completed_trajectories', 0), utc=detail.get('utc'),
                        stage=detail.get('stage'), actual_sampled_tokens=detail.get('actual_sampled_tokens'),
                        device_native_unit=binding.get('native_unit'), current_unit=binding.get('native_unit') == expected,
                        progress_path=metadata.get('path'), progress_mtime_ns=metadata.get('mtime_ns'),
                        child=claim.get('child'), fd=claim.get('fd'), lock_target=claim.get('lock_target')))
        counts = progress or complete
        own = counts.get('completed_own_batches', 0)
        previous = Path(progress.get('latest_input') or '').stem
        proofs = [phase for phase in phases if phase['current_unit'] and previous
                  and phase['batch'] != previous and type(phase['completed']) is int and phase['completed'] > 0]
        if native.get('ActiveState') == 'failed' or failures:
            attention.append(dict(slot=slot, unit=expected, native=native, failures=failures))
        elif complete and not (type(own) is int and own >= minimum and bool(proofs)):
            attention.append(dict(slot=slot, unit=expected, native=native,
                reason='NATIVE_ENDED_BEFORE_WHOLE_AND_NEXT_ACCEPTANCE', complete=complete))
        slots[slot] = dict(unit=expected, instance=instance, native=native,
            controller_pid=row.get('controller_pid'), historical_pid_from_immutable_receipt=row.get('historical_pid_from_immutable_receipt'),
            initialized=any(item['actual_model_initialized'] is True for item in models), model_proofs=models,
            own_complete_batches=own, questions=counts.get('questions', 0), trajectories=counts.get('trajectories', 0),
            latest_input=progress.get('latest_input'), latest_batch_seconds=progress.get('latest_batch_seconds'),
            active_phases=phases, next_actual_batch_proofs=proofs,
            whole_and_next_observed=type(own) is int and own >= minimum and bool(proofs),
            failures=failures, launch_failures=launch_failures, native_complete=bool(complete), actual=actual)
    cpu = raw['units'].get('cpu') or {}
    cpu_failures = [value(item.get('failure')) for item in cpu.get('attempts') or [] if value(item.get('failure'))]
    if (cpu.get('native') or {}).get('ActiveState') == 'failed' or cpu_failures:
        attention.append(dict(slot='cpu', unit=cpu.get('unit'), native=cpu.get('native'), failures=cpu_failures))
    return dict(status='READ_ONLY_PRIMARY_COHORT_FOLLOW', epoch=raw.get('epoch'), run_id=control.RUN_ID,
        slots=slots, cpu_native=cpu.get('native'), cpu_metadata=cpu.get('cpu_metadata'),
        global_sealed=raw.get('global_sealed'), attention=attention, actual=actual,
        global_goal_complete=False, remote_jobs_modified=False)


def save_latest(path, summary):
    temporary = path.with_name(path.name + '.tmp-' + str(time.time_ns()))
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=4)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--own-batches', type=int, default=1)
    args = parser.parse_args()
    if not (0 < args.hours <= 4 and args.interval >= 60 and args.own_batches >= 1):
        raise ValueError('Observer requires <=4 hours, >=60-second interval and >=1 own sealed batch')
    script = Path(observer.__file__).read_text()
    script_sha = hashlib.sha256(script.encode()).hexdigest()
    latest = control.ROOT / 'work/qwen7b-primary-cohort-follow-latest-small.json'
    until, accepted = time.monotonic() + args.hours * 3600, {}
    print(json.dumps(dict(status='READ_ONLY_PRIMARY_COHORT_FOLLOW_STARTED', instances=observer.INSTANCES,
        interval_seconds=args.interval, maximum_hours=args.hours, observer_source_sha256=script_sha,
        latest=str(latest), remote_jobs_modified=False)), flush=True)
    while time.monotonic() < until:
        # The existing transport has a 110s bound. Do not begin a call that
        # could extend this observer beyond its own bounded follow window.
        if until - time.monotonic() < 115:
            break
        cycle = time.monotonic()
        actual = control.ROOT / 'work' / ('qwen7b-primary-cohort-follow-actual-%d.json' % time.time_ns())
        envelope = dict(epoch=time.time(), observer_source_sha256=script_sha)
        try:
            result = control.transport(script, 'h100-10')
            envelope.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            if result.returncode:
                raise RuntimeError('Transport returned %d: %s' % (result.returncode, result.stderr[-1000:]))
            raw = json.loads(result.stdout)
            if raw.get('run_id') != control.RUN_ID or raw.get('status') != 'READ_ONLY_NODE10_PRIMARY_CURRENT_COHORT':
                raise ValueError('Unexpected read-only cohort observation identity')
            summary = summarize(raw, str(actual), args.own_batches)
            if raw.get('native_query_returncode') and not any((row.get('native') or {}).get('Id') for row in summary['slots'].values()):
                raise RuntimeError('Native observation command failed; no native state can be inferred')
            for slot, row in summary['slots'].items():
                if row['whole_and_next_observed']:
                    accepted[slot] = dict(unit=row['unit'], own_complete_batches=row['own_complete_batches'],
                        latest_input=row['latest_input'], proofs=row['next_actual_batch_proofs'], actual=str(actual), epoch=raw.get('epoch'))
            summary['accepted_whole_and_next'] = dict(accepted)
            summary['all_five_whole_and_next_accepted'] = len(accepted) == len(observer.INSTANCES)
        except Exception as error:
            envelope.update(observation_error_type=type(error).__name__, observation_error=str(error)[-1600:])
            summary = dict(status='OBSERVATION_CHANNEL_ERROR_NOT_NATIVE_EXIT', error_type=type(error).__name__,
                error=str(error)[-1600:], epoch=time.time(), actual=str(actual), accepted_whole_and_next=dict(accepted),
                global_goal_complete=False, remote_jobs_modified=False)
        with actual.open('x') as stream:
            stream.write(json.dumps(envelope, indent=2) + '\n')
        save_latest(latest, summary)
        print(json.dumps(summary), flush=True)
        if summary.get('attention'):
            print('ROOT_ATTENTION_ACTUAL_PRIMARY_OR_CPU_FAILURE_NO_REMOTE_ACTION', flush=True)
            return 2
        if summary.get('all_five_whole_and_next_accepted'):
            print('ALL_FIVE_CURRENT_PRIMARY_UNITS_REAL_WHOLE_AND_NEXT_ACCEPTED', flush=True)
            return 0
        time.sleep(max(0., min(args.interval - (time.monotonic() - cycle), until - time.monotonic())))
    print('ROOT_ATTENTION_READ_ONLY_FOLLOW_WINDOW_ENDED_NO_NATIVE_JOB_CHANGED', flush=True)
    return 3


if __name__ == '__main__':
    raise SystemExit(main())
