"""Read a scoped human-stop terminal state; never signal or restart a job."""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def inspect(base, job):
    base = Path(base)
    sys.path.insert(0, str(base / 'repo'))
    from reproduction.common8.native_allocation import fields
    from reproduction.prefix_only_correctness_ab.launch import birth_states
    path = base / 'preparation' / ('terminal%s-human-lr1e5-original-inode-full-sha.json' % job)
    if path.exists():
        return dict(existing=True, proof=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result = subprocess.run(['scontrol', 'show', 'job', str(job), '-o'], capture_output=True, text=True, timeout=30)
    native = fields(result.stdout) if result.returncode == 0 else dict(unavailable=result.stderr)
    ledger = json.loads((base / 'logs/process-ledger.json').read_text())
    births = [record for item in ledger['jobs'].values() for record in item['ledger'].values()]
    present = birth_states(births)
    cgroups = glob.glob('/sys/fs/cgroup/**/job_%s*' % job, recursive=True)
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    own_pids = {str(item['pid']) for item in births}
    owned_gpu = [line for line in gpu.stdout.splitlines() if line.split(',')[0].strip() in own_pids]
    # Slurm may purge a cancelled job before the following inspection. The
    # successful scoped scancel receipt, empty native queue and actual owned
    # birth/cgroup/GPU release prove stopping without inventing an exit code.
    stop = base / 'preparation/human-resume20-lr1e5-stop-once-v1/RECEIPT.json'
    queue = subprocess.run(['squeue', '-h', '-o', '%i'], capture_output=True, text=True, timeout=30)
    stopped = json.loads(stop.read_text()) if stop.exists() else {}
    purged_after_stop = (result.returncode != 0 and queue.returncode == 0
        and str(job) not in queue.stdout.split() and stopped.get('job') == str(job)
        and stopped.get('returncode') == 0
        and stopped.get('native_before', {}).get('Command') == str(base / 'preparation/native-launch.sh'))
    value = dict(job_id=str(job), run_id=ledger['run_id'], native=native,
                 actual_births_present=present, native_job_cgroup_paths=cgroups,
                 owned_gpu=owned_gpu, original_births=len(births), checked_unix=time.time(),
                 training_modified=False, human_stop=True, experiment_goal_paused=False,
                 native_purged_after_scoped_stop=purged_after_stop,
                 native_terminal_exit_code_observed=False if purged_after_stop else True)
    if ((not purged_after_stop and native.get('JobState') not in ('CANCELLED', 'FAILED', 'COMPLETED', 'TIMEOUT', 'OUT_OF_MEMORY'))
            or present or cgroups or owned_gpu or gpu.returncode):
        return dict(status='PENDING_ACTUAL_TERMINAL_RELEASE', **value)
    records = []
    for log in sorted(base.rglob('*.log')):
        if not log.is_file() or log.is_symlink():
            continue
        with log.open('rb') as handle:
            before = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b''):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if signature(before) != signature(after) or signature(after) != signature(log.stat()):
            raise ValueError('Original terminal log changed during read: ' + str(log))
        records.append(dict(path=str(log), dev=before.st_dev, inode=before.st_ino,
                            bytes=before.st_size, mtime_ns=before.st_mtime_ns, sha256=digest.hexdigest()))
    joint = base / 'latest-joint.json'
    value.update(status='HUMAN_STOP_ORIGINAL_INODE_FULL_SHA_OWNED_RELEASED', original_logs=records,
                 joint_complete_sha256=hashlib.sha256(joint.read_bytes()).hexdigest(),
                 latest_complete_outer=json.loads(joint.read_text())['outer'])
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    return dict(status=value['status'], proof=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                native=native, latest_complete_outer=value['latest_complete_outer'],
                original_logs=len(records), original_births=len(births), actual_births_present=[],
                native_job_cgroup_paths=[], owned_gpu=[])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.base, args.job)))
