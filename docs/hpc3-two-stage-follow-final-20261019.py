"""Bounded read-only observation until the actual R4 outer200 joint.

This performs no submission, audit launch, training control or source change.
Outer200 is an alert to run the existing full CPU/QA/final/closure audits,
not whole-goal acceptance. The final probe remains at outer190.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from reproduction.hpc3_prefix5.control import OUTPUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=240)
    args = parser.parse_args()
    if not 1 <= args.samples <= 240:
        parser.error('Require a bounded foreground observation of1..240 samples')
    observer = Path(__file__).with_name('hpc3-two-stage-cp61-observe-20261018.py')
    target = OUTPUT / 'two-stage-follow-final-000200.jsonl'
    for sample in range(args.samples):
        try:
            result = subprocess.run([sys.executable, str(observer)], text=True,
                                    capture_output=True, timeout=40)
            if result.returncode:
                raise RuntimeError(result.stderr[-1500:])
            value = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, RuntimeError, ValueError) as error:
            print(json.dumps(dict(sample=sample,
                status='READ_FAILED_NOT_TRAINING_TERMINAL',
                detail=str(error)[-1500:])), flush=True)
            time.sleep(30)
            continue
        def counts(key):
            return [(row or {}).get('completed_trajectories')
                    for row in (value.get(key) or [])]
        row = dict(sample=sample, epoch=value['epoch'], job=value['job'],
            queue=value['queue'], stage=value.get('status'),
            complete_outer=value['complete_outer'], actor=value['actor'],
            probe=value['probe'], prefix=counts('prefix'),
            full_pool=counts('full_pool'), selected=value.get('selected'),
            updates=value.get('inflight_rank_updates'),
            failures=value.get('failed_jobs'), errors=value.get('error_lines'))
        with target.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if (value.get('failed_jobs') or value.get('error_lines')
                or any(value.get('rank_failures') or [])):
            print(json.dumps(dict(status='ACTUAL_PHASE_ERROR_INSPECT_NO_TRAINING_ACTION')), flush=True)
            return 1
        if value['complete_outer'] >= 200:
            print(json.dumps(dict(status=
                'ACTUAL200_JOINT_REACHED_FULL_CPU_QA_FINAL_CLOSURE_STILL_REQUIRED')),
                flush=True)
            return 0
        if value.get('queue_returncode') != 0:
            print(json.dumps(dict(status='QUEUE_QUERY_FAILED_NOT_TERMINAL',
                                  detail=value.get('queue_stderr'))), flush=True)
        elif not value['queue']:
            print(json.dumps(dict(status='QUEUE_EMPTY_INSPECT_ACCOUNTING_NO_RESTART',
                                  accounting=value.get('accounting'))), flush=True)
            return 1
        time.sleep(30)
    print(json.dumps(dict(status='BOUNDED_READ_ONLY_WAIT_EXPIRED_NOT_TRAINING_TERMINAL')),
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
