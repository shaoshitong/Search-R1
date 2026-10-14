"""Read-only terminal capture before Slurm's short job-retention window expires."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def fields(text):
    return dict(token.split('=', 1) for token in text.split() if '=' in token)


def terminal_record(detail, job_id):
    value = fields(detail)
    if value.get('JobId') != str(job_id):
        raise ValueError('Wrong predecessor job identity')
    if value.get('JobState') not in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'):
        return None
    return dict(job_id=job_id, state=value['JobState'], exit_code=value['ExitCode'],
                raw_scontrol=detail, observed_unix=time.time(),
                source='actual native Slurm terminal record; not child exit or cleanup')


def once(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def watch(base, job_id, duration=14400):
    base = Path(base).resolve()
    target = base / 'preparation/predecessor-exit0.json'
    if target.exists():
        raise RuntimeError('Terminal proof already exists; inspect, never overwrite')
    started = time.monotonic()
    last = None
    while time.monotonic() - started < duration:
        try:
            result = subprocess.run(['scontrol', 'show', 'job', str(job_id)],
                                    capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired:
            time.sleep(10)
            continue
        if result.returncode == 0 and result.stdout.strip():
            record = terminal_record(result.stdout, job_id)
            value = fields(result.stdout)
            last = dict(job_id=job_id, state=value.get('JobState'),
                        observed_unix=time.time(), raw_scontrol=result.stdout)
            if record is not None:
                # Failed terminal records are preserved here too; submit() rejects them.
                once(target, record)
                return record
        else:
            # Missing/timeout is not exit0. Preserve the query evidence and stop.
            record = dict(job_id=job_id, state='UNOBSERVED_TERMINAL',
                          query_returncode=result.returncode, stdout=result.stdout,
                          stderr=result.stderr, last_actual_observation=last,
                          observed_unix=time.time(), exit0_proven=False)
            once(base / 'preparation/predecessor-terminal-unobserved.json', record)
            return record
        time.sleep(10)
    record = dict(job_id=job_id, state='OBSERVER_DEADLINE', exit0_proven=False,
                  last_actual_observation=last, observed_unix=time.time())
    once(base / 'preparation/predecessor-observer-deadline.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--job-id', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(watch(args.base, args.job_id)))
