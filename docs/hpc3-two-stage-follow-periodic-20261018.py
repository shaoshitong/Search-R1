"""Bounded foreground reader until a real R4 P10 joint; never change training."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from reproduction.hpc3_prefix5.control import OUTPUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', type=int, choices=[n for n in range(70,200,10) if n not in (100,150)], required=True)
    parser.add_argument('--samples', type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.samples <= 240:
        raise ValueError('The foreground observation must be bounded')
    observer = Path(__file__).with_name('hpc3-two-stage-cp61-observe-20261018.py')
    periodic = Path(__file__).with_name('hpc3-two-stage-periodic-observe-20261018.py')
    target = OUTPUT / ('two-stage-follow-periodic-%06d.jsonl' % args.outer)
    for sample in range(args.samples):
        try:
            result = subprocess.run([sys.executable,str(observer)],text=True,capture_output=True,timeout=40)
            if result.returncode:
                raise RuntimeError(result.stderr[-1500:])
            v = json.loads(result.stdout)
        except (subprocess.TimeoutExpired,RuntimeError,ValueError) as error:
            print(json.dumps(dict(sample=sample,status='READ_FAILED_NOT_TRAINING_TERMINAL',detail=str(error)[-1500:])),flush=True)
            time.sleep(30)
            continue
        def counts(key):
            return [(r or {}).get('completed_trajectories') for r in (v.get(key) or [])]
        row = dict(sample=sample,epoch=v['epoch'],job=v['job'],queue=v['queue'],
            stage=v.get('status'),complete_outer=v['complete_outer'],actor=v['actor'],probe=v['probe'],
            prefix=counts('prefix'),full_pool=counts('full_pool'),selected=v.get('selected'),
            updates=v.get('inflight_rank_updates'),failures=v.get('failed_jobs'),errors=v.get('error_lines'))
        if (v.get('status') or {}).get('outer',0) >= args.outer:
            try:
                fitted = subprocess.run([sys.executable,str(periodic),'--outer',str(args.outer)],text=True,capture_output=True,timeout=40)
                row['periodic'] = json.loads(fitted.stdout) if fitted.returncode == 0 else dict(read_failed=fitted.stderr[-1000:])
            except (subprocess.TimeoutExpired,ValueError) as error:
                row['periodic'] = dict(read_failed=str(error)[-1000:])
        with target.open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if v.get('failed_jobs') or v.get('error_lines') or any(v.get('rank_failures') or []):
            print(json.dumps(dict(status='ACTUAL_PHASE_ERROR_NEEDS_INSPECTION_NO_TRAINING_ACTION_TAKEN')),flush=True)
            return 1
        if v.get('queue_returncode') != 0:
            print(json.dumps(dict(status='QUEUE_QUERY_FAILED_NOT_TERMINAL',detail=v.get('queue_stderr'))),flush=True)
        elif not v['queue']:
            print(json.dumps(dict(status='QUEUE_EMPTY_INSPECT_ACTUAL_ACCOUNTING_NO_RESTART',accounting=v.get('accounting'))),flush=True)
            return 1
        if v['complete_outer'] >= args.outer and v['probe']['completed_outer'] >= args.outer:
            print(json.dumps(dict(status='ACTUAL_P10_JOINT_REACHED_CPU_AND_NEXT_USE_ACCEPTANCE_STILL_REQUIRED',outer=args.outer)),flush=True)
            return 0
        time.sleep(30)
    print(json.dumps(dict(status='BOUNDED_READ_ONLY_WAIT_EXPIRED_NOT_TRAINING_TERMINAL')),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
