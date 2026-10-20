"""Fail closed when a control observation did not return a fresh snapshot."""
import json
import math


def accept_snapshot(returncode, stdout, previous_time=None):
    if returncode:
        raise RuntimeError('Control query failed with return code %d' % returncode)
    snapshots = []
    for line in stdout.splitlines():
        if line.startswith('{'):
            snapshots.append(json.loads(line))
    if len(snapshots) != 1:
        raise RuntimeError('Expected one current remote snapshot, received %d' % len(snapshots))
    result = snapshots[0]
    if not {'time', 'job_id', 'queue', 'accounting'} <= result.keys():
        raise RuntimeError('Remote snapshot lacks required status fields')
    if not isinstance(result['time'], (int, float)) or not math.isfinite(result['time']):
        raise RuntimeError('Remote snapshot has an invalid observation timestamp')
    if previous_time is not None and result['time'] <= previous_time:
        raise RuntimeError('Remote snapshot did not advance beyond the previous observation')
    if not str(result['job_id']).isdigit():
        raise RuntimeError('Remote snapshot has an invalid Slurm job id')
    if not all(isinstance(result[key], str) for key in ('queue', 'accounting')):
        raise RuntimeError('Remote scheduler output must be text')
    return result
