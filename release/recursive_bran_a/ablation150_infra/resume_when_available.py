"""CPU-only queue for an exact failed study's complete native boundary.

Does not signal processes, occupy GPUs, change science, or reinterpret OOM as
a numerical skip. The unchanged native loader verifies full state on restart.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p = Path(p)
    temp = p.with_name(p.name + '.pending')
    temp.write_text(json.dumps(value, sort_keys=True) + '\n')
    os.replace(temp, p)


def absent(owner):
    p = Path('/proc') / str(owner['pid'])
    try:
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[19]) != owner['tick'] or p.stat().st_uid != owner['uid']
    except FileNotFoundError:
        return True


def run(request_path):
    r = json.loads(Path(request_path).read_text())
    base = Path(r['base']); group = base / r['group']
    out = Path(r['queue_output']); out.mkdir(exist_ok=True, parents=True)
    lock = (out / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'launched.json').exists():
        raise RuntimeError('Already dispatched; inspect actual exit, never duplicate')
    entry_path = group / 'receipts/current-entry.json'
    old = json.loads(entry_path.read_text())
    assert all(old[k] == r['previous_entry'][k] for k in ('pid', 'uid', 'tick', 'run_id'))
    assert absent(old) and absent(r['previous_study'])
    wait_path = group / 'receipts/actual-parent-wait.json'
    wait = json.loads(wait_path.read_text())
    assert wait['pid'] == old['pid'] and wait['rc'] != 0
    terminal = json.loads((Path(old['output']) / 'entry-exit.json').read_text())
    assert terminal['pid'] == old['pid'] and terminal['rc'] != 0
    cp = Path(r['manifest']); assert sha(cp) == r['sha256']
    m = json.loads(cp.read_text()); identity = m['identity']
    assert cp.is_relative_to(group / 'checkpoints') and not cp.is_symlink()
    assert m['status'] == 'COMPLETE_SAME_WORLD_STATE_SAVED'
    assert identity['completed_step'] == r['completed_outer']
    assert identity['config_sha256'] == old['config_sha256']
    assert identity['code_commit'] == r['science_commit']
    assert len(m['worker_states']) == 8
    assert {x['identity']['rank'] for x in m['worker_states']} == set(range(8))
    for x in m['worker_states'] + [m['driver']]:
        p = Path(x['path']); assert p.is_file() and not p.is_symlink() and p.parent == cp.parent
    for name in ('current-entry.json', 'actual-parent-wait.json', 'explicit-recovery.json'):
        p = group / 'receipts' / name
        if p.exists() and not (out / name).exists():
            (out / name).write_bytes(p.read_bytes())
    recovery = dict(kind='COMPLETE_NATIVE_BOUNDARY', previous_pid=old['pid'],
        previous_tick=old['tick'], previous_uid=old['uid'], manifest=str(cp),
        sha256=r['sha256'], actual_prior_wait_rc=wait['rc'])
    write(group / 'receipts/explicit-recovery.json', recovery)
    write(out / 'started.json', dict(time=time.time(), pid=os.getpid(), uid=os.getuid(),
        tick=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        request_sha256=sha(request_path), scope='CPU resource queue; not resumed training'))
    while True:
        # Observe only the already-authorized allocation and its eight cards.
        job = subprocess.run(['scontrol', 'show', 'job', str(r['job']), '-o'],
                             capture_output=True, text=True, timeout=15)
        if job.returncode or 'JobState=RUNNING' not in job.stdout or 'NodeList=h100-11' not in job.stdout:
            raise RuntimeError('Original allocation no longer available; do not borrow another node')
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=15)
        mem = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=15)
        free = [int(x) for x in mem.stdout.split()] if mem.returncode == 0 else []
        ready = apps.returncode == 0 and not apps.stdout.strip() and len(free) == 8 and min(free) >= 76000
        write(out / 'resource-state.json', dict(time=time.time(), ready=ready,
            gpu_pids=apps.stdout, free_MiB=free, apps_rc=apps.returncode, memory_rc=mem.returncode))
        if ready:
            break
        time.sleep(30)
    # Recheck exact failure ownership and boundary immediately before dispatch.
    assert json.loads(entry_path.read_text()) == old and absent(old) and absent(r['previous_study'])
    assert sha(cp) == r['sha256']
    env = dict(os.environ, RBA_ABLATION_BASE=str(base),
        RBA_ABLATION_SOURCE_COMMIT=r['science_commit'],
        RBA_ABLATION_ENTRY_COMMIT=r['entry_commit'], RBA_H100_ALLOCATION_ID=str(r['job']))
    command = ['srun', '--jobid=' + str(r['job']), '--overlap', '--nodes=1', '--ntasks=1',
        '--gres=gpu:h100:8', '--cpus-per-task=96', '--mem=768G', 'bash',
        str(Path(r['training_source']) / 'release/recursive_bran_a/ablation150_v1/study.sh')]
    with (out / 'study.log').open('xb') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(out / 'launched.json', dict(time=time.time(), pid=child.pid, command=command,
            scope='Dispatch only; native restoration and complete outers must be observed'))
        rc = child.wait()
    write(out / 'actual-srun-wait.json', dict(time=time.time(), pid=child.pid, rc=rc))
    return rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True)
    raise SystemExit(run(parser.parse_args().request))
