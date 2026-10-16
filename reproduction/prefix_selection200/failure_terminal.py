"""Preserve actual failed-job log inodes and resource release, without signals."""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def inspect(base, job):
    base, job = Path(base), str(job)
    sys.path.insert(0, str(base / 'repo'))
    from reproduction.common8.native_allocation import fields
    from reproduction.prefix_only_correctness_ab.launch import birth_states
    proof = base / 'preparation' / ('terminal%s-actual-failure-original-inode-full-sha.json' % job)
    if proof.exists():
        return dict(existing=True, proof=str(proof), sha256=hashlib.sha256(proof.read_bytes()).hexdigest())
    def command(args):
        return subprocess.run(args, capture_output=True, text=True, timeout=30)
    query = command(['scontrol', 'show', 'job', job, '-o'])
    native = fields(query.stdout) if query.returncode == 0 else dict(unavailable=query.stderr)
    queue = command(['squeue', '-h', '-o', '%i'])
    cleanup = json.loads((base / 'logs/cleanup-status.json').read_text())
    ledger = json.loads((base / 'logs/process-ledger.json').read_text())
    births = [record for item in ledger['jobs'].values() for record in item['ledger'].values()]
    release_checks = []
    for _ in range(2):
        release_checks.append(dict(checked_unix=time.time(), actual_births_present=birth_states(births)))
        time.sleep(1)
    cgroups = glob.glob('/sys/fs/cgroup/**/job_%s*' % job, recursive=True)
    gpu = command(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'])
    own_pids = {str(item['pid']) for item in births}
    owned_gpu = [line for line in gpu.stdout.splitlines() if line.split(',')[0].strip() in own_pids]
    controller_lines = []
    for name in ('/var/log/slurm/slurmctld.log', '/var/log/slurmctld.log'):
        path = Path(name)
        if path.exists():
            with path.open('rb') as handle:
                handle.seek(max(0, path.stat().st_size - 1024 ** 2))
                controller_lines.extend(line for line in handle.read().decode(errors='replace').splitlines()
                                        if re.search(r'JobId=%s\b' % re.escape(job), line))
    exits = [int(match.group(1)) for line in controller_lines
             for match in [re.search(r'JobId=%s\s+WEXITSTATUS\s+(\d+)' % re.escape(job), line)] if match]
    known_failure = native.get('JobState') in ('FAILED', 'OUT_OF_MEMORY', 'TIMEOUT', 'NODE_FAIL')
    purged_failure = (query.returncode != 0 and queue.returncode == 0 and job not in queue.stdout.split()
                      and bool(exits) and exits[-1] != 0)
    if (not (known_failure or purged_failure)
            or any(check['actual_births_present'] for check in release_checks)
            or cgroups or owned_gpu or gpu.returncode):
        return dict(status='PENDING_ACTUAL_FAILURE_RELEASE', native=native, release_checks=release_checks,
                    native_job_cgroup_paths=cgroups, owned_gpu=owned_gpu, controller_lines=controller_lines)
    records = []
    for path in sorted(base.rglob('*.log')):
        if not path.is_file() or path.is_symlink():
            continue
        with path.open('rb') as handle:
            before = os.fstat(handle.fileno()); digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b''):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        signature = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        if signature(before) != signature(after) or signature(after) != signature(path.stat()):
            raise ValueError('Actual terminal log changed during full read: ' + str(path))
        records.append(dict(path=str(path), dev=before.st_dev, inode=before.st_ino,
                            bytes=before.st_size, mtime_ns=before.st_mtime_ns, sha256=digest.hexdigest()))
    joint = base / 'latest-joint.json'
    value = dict(status='ACTUAL_FAILURE_ORIGINAL_INODE_FULL_SHA_OWNED_RELEASED', job_id=job,
        run_id=ledger['run_id'], native=native, controller_lines=controller_lines,
        controller_observed_exit_status=exits[-1] if exits else None,
        actual_births_present=[], native_job_cgroup_paths=[], owned_gpu=[], release_checks=release_checks,
        original_cleanup_receipt=cleanup,
        original_births=len(births), original_logs=records, checked_unix=time.time(),
        latest_complete_outer=json.loads(joint.read_text())['outer'],
        joint_complete_sha256=hashlib.sha256(joint.read_bytes()).hexdigest(),
        training_modified=False, human_stop=False, experiment_goal_paused=False)
    with proof.open('x') as handle:
        json.dump(value, handle, sort_keys=True); handle.flush(); os.fsync(handle.fileno())
    return dict(status=value['status'], proof=str(proof), sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
                original_logs=len(records), original_births=len(births),
                latest_complete_outer=value['latest_complete_outer'], controller_observed_exit_status=value['controller_observed_exit_status'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True); parser.add_argument('--job', required=True)
    args = parser.parse_args(); print(json.dumps(inspect(args.base, args.job)))
