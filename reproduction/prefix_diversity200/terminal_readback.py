"""Capture terminal native/log/process/cost evidence without reading model tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def capture(base, job):
    from reproduction.prefix_only_correctness_ab.launch import birth_states
    from .actor import read, atomic_json
    base = Path(base)
    target = base / 'preparation' / ('terminal%s-original-inode-full-sha.json' % job)
    if target.exists():
        return read(target)
    native = subprocess.run(['scontrol', 'show', 'job', str(job), '-o'], capture_output=True, text=True, timeout=30)
    if native.returncode or 'JobState=FAILED' not in native.stdout or 'WorkDir=' + str(base) not in native.stdout:
        raise ValueError('Require the actual failed native allocation identity')
    queue = subprocess.check_output(['squeue', '-h', '-o', '%i|%j|%N'], text=True, timeout=30)
    name = read(base / 'preparation/submit-intent.json')['job_name']
    if any(row.split('|')[1] == name for row in queue.splitlines()):
        raise ValueError('A same-task native job is queued; reconcile before recovery')
    ledger = read(base / 'logs/process-ledger.json')
    births = [v for part in ledger['jobs'].values() for v in part['ledger'].values()]
    present = birth_states(births)
    cleanup = read(base / 'logs/cleanup-status.json')
    cgroups = [str(p) for p in Path('/sys/fs/cgroup').glob('**/job_' + str(job))]
    if present or not cleanup['owned_births_all_gone'] or cgroups:
        raise ValueError('Failed task still has original births or native cgroup')
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True, timeout=30)
    owned_pids = {int(v['pid']) for v in births}
    if any(int(row.split(',')[0]) in owned_pids for row in gpu.splitlines() if row.strip()):
        raise ValueError('Original owned task PID still holds GPU memory')
    paths = set((base / 'logs').glob('*.log'))
    paths.update((base / 'outers').glob('*/validation/*.log'))
    logs = []
    for path in sorted(paths):
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for part in iter(lambda: stream.read(2**20), b''):
                digest.update(part)
        after = path.stat()
        if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('Original terminal log changed while hashing')
        logs.append(dict(path=str(path), bytes=stat.st_size, inode=stat.st_ino, sha256=digest.hexdigest(),
                         tail=path.read_text(errors='replace')[-1800:] if 'validation' in str(path) else None))
    joint = read(base / 'latest-joint.json')
    result = dict(job_id=str(job), base=str(base), run_id=ledger['run_id'], observed_unix=time.time(),
        native=dict(returncode=native.returncode, stdout=native.stdout, stderr=native.stderr),
        queue=queue, actual_births_present=present, original_birth_count=len(births),
        native_job_cgroup_paths=cgroups, cleanup=cleanup, gpu_compute_processes=gpu,
        owned_gpu_processes_gone=True, original_logs=logs, complete_outer=joint['outer'],
        actor_adam_per_rank=joint['actor']['actual_adam_steps_per_rank'],
        probe_adam_per_rank=joint['probe']['actual_adam_steps_per_rank'],
        latest_joint_sha256=hashlib.sha256((base/'latest-joint.json').read_bytes()).hexdigest(),
        latest_phase_timings=joint['phase_timings'], historical_cost_evidence_retained=True,
        model_tensors_read=False, training_modified=False)
    atomic_json(target, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--base', required=True); parser.add_argument('--job', required=True)
    args = parser.parse_args(); print(json.dumps(capture(args.base, args.job), allow_nan=False))
