"""Reconcile this run's GPU reservation wall from retained native snapshots.

This local reader performs no remote action. Reservation CPU-core-hours are
capacity allocation, never measured CPU usage. Failed jobs may contain useful
committed outers, so their entire reservation is not labeled wasted compute.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re


TERMINAL = {'FAILED', 'COMPLETED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL'}


def seconds(value):
    days, rest = value.split('-', 1) if '-' in value else ('0', value)
    hours, minutes, secs = map(int, rest.split(':'))
    if min(int(days), hours, minutes, secs) < 0 or minutes >= 60 or secs >= 60:
        raise ValueError('Invalid actual native RunTime')
    return int(days)*86400 + hours*3600 + minutes*60 + secs


def native_rows(value, depth=0):
    if depth > 9:
        return
    if isinstance(value, dict):
        if 'JobId' in value and 'RunTime' in value:
            yield value
        for key, item in value.items():
            if key not in ('log_tails', 'logs', 'tail', 'head', 'gpu_actual'):
                yield from native_rows(item, depth+1)
    elif isinstance(value, list):
        for item in value:
            yield from native_rows(item, depth+1)
    elif isinstance(value, str):
        if value.lstrip().startswith('{'):
            try:
                decoded = json.loads(value)
            except ValueError:
                return
            yield from native_rows(decoded, depth+1)
        elif 'JobId=' in value and 'WorkDir=' in value:
            yield dict(re.findall(r'(?:^|\s)(\w+)=([^\s]*)', value))


def collect(receipts, run_id, jobs):
    directory = Path(receipts)
    if not re.fullmatch(r'[a-f0-9]{32}', run_id):
        raise ValueError('Logical run UUID is required')
    expected = '/mnt/data/shaoshitong/search-g1-prefix-diversity200-'
    candidates = {str(job): [] for job in jobs}
    for path in directory.glob('*.json'):
        payload = path.read_bytes()
        try:
            value = json.loads(payload)
        except ValueError:
            continue
        for native in native_rows(value):
            job = str(native['JobId'])
            workdir = native.get('WorkDir', '')
            if (job not in candidates or not workdir.startswith(expected)
                    or run_id[:12] not in workdir):
                continue
            candidates[job].append(dict(native=native, source_path=str(path),
                source_sha256=hashlib.sha256(payload).hexdigest()))
    result = {}
    for job, choices in candidates.items():
        terminal = [r for r in choices if r['native'].get('JobState') in TERMINAL]
        if not choices:
            raise ValueError('No retained authoritative native observation for job '+job)
        best = max(terminal or choices, key=lambda r: (
            seconds(r['native']['RunTime']), 'LATEST' not in r['source_path']))
        native = best['native']
        if (native.get('NumNodes') != '1' or native.get('NodeList') != 'h100-10'
                or native.get('TresPerNode') != 'gres:gpu:8'):
            raise ValueError('Reservation hardware differs from this run')
        wall = seconds(native['RunTime'])
        ended = native.get('JobState') in TERMINAL
        if ended:
            start = datetime.datetime.fromisoformat(native['StartTime'])
            end = datetime.datetime.fromisoformat(native['EndTime'])
            if (end-start).total_seconds() != wall:
                raise ValueError('Native start/end and RunTime disagree')
        result[job] = dict(best, actual_reservation_wall_seconds=wall,
            actual_gpu_count=8, reserved_gpu_seconds=wall*8,
            reserved_gpu_hours=wall*8/3600,
            reserved_cpu_core_hours=wall*int(native['NumCPUs'])/3600,
            native_terminal=ended, cost_is_final_for_job=ended,
            useful_committed_updates_may_be_present=True,
            entire_failed_job_reservation_is_not_wasted_compute=True)
    return dict(run_id=run_id, jobs=result,
        terminal_jobs_gpu_hours=sum(r['reserved_gpu_hours'] for r in result.values() if r['native_terminal']),
        observed_total_reserved_gpu_hours=sum(r['reserved_gpu_hours'] for r in result.values()),
        all_requested_jobs_terminal=all(r['native_terminal'] for r in result.values()),
        scope='native reserved eight-GPU wall including setup, QA and failures; running jobs are observation lower bounds',
        measured_cpu_usage=False, per_process_gpu_usage=False, extra_additive_phase_gpu_cost=False,
        remote_action=False, training_modified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--jobs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    value = collect(args.receipts, args.run_id, args.jobs.split(','))
    Path(args.output).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(jobs=list(value['jobs']), terminal_gpu_hours=value['terminal_jobs_gpu_hours'],
        observed_reserved_gpu_hours=value['observed_total_reserved_gpu_hours'],
        all_jobs_terminal=value['all_requested_jobs_terminal']), allow_nan=False))
