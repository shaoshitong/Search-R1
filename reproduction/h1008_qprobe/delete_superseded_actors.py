"""Explicit, once-only deletion of two superseded actors and their partial copy.

No phase runs on import. Inputs, official weights, teacher labels/features,
initial probe heads, code and logs are outside the exact allowlisted paths.
An interrupted intent is inspection-only; it must never silently replay.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import time

RUN_ID = 'c025948ebc0947f69d86af84871a9237'
FRESH = Path('/mnt/data/shaoshitong/search-g1-qprobe3-stdloss-545d4cf-20261015-v2')
RESUME = Path('/mnt/data/shaoshitong/search-g1-qprobe3-resume5-fb68e88-20261015')
CP5 = FRESH / 'checkpoints/search-g1-common8/actor/global_step_5'
CP10 = RESUME / 'checkpoints/search-g1-common8/actor/global_step_10'
BACKUP = Path('/mnt/data/shaoshitong/search-g1-qprobe3-independent-backup-20261015') / (RUN_ID + '-step5-once-v1')
BACKUP_ACTOR = BACKUP / 'staging/payload' / str(CP5).lstrip('/')
PID, START_TICKS = 2122596, 1911592207
TERMINAL = {'CANCELLED', 'FAILED', 'COMPLETED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha_small(path):
    path = Path(path)
    assert path.stat().st_size < 64 * 1024**2, 'Only small audit metadata may be hashed'
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    assert Path(path).is_file() and not Path(path).is_symlink()
    return json.loads(Path(path).read_text())


def once(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def unaliased(path):
    path = Path(path)
    assert path.is_absolute()
    for component in (path, *path.parents):
        assert not component.is_symlink(), 'Symlink in deletion path: ' + str(component)
    assert path.resolve() == path
    return path


def inventory(path):
    path = unaliased(path)
    assert path.is_dir(), 'Expected explicit actor checkpoint directory'
    result = []
    for item in sorted(path.rglob('*')):
        assert not item.is_symlink(), 'Refuse checkpoint symlink: ' + str(item)
        if item.is_dir():
            continue
        assert item.is_file(), 'Nonregular checkpoint entry'
        st = item.stat()
        result.append(dict(path=str(item.relative_to(path)), bytes=st.st_size,
                           device=st.st_dev, inode=st.st_ino))
    assert result, 'Empty target requires separate inspection'
    return dict(path=str(path), files=result, file_count=len(result), bytes=sum(v['bytes'] for v in result))


def process():
    path = Path('/proc') / str(PID)
    if not path.exists():
        return None
    values = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=PID, start_ticks=int(values[19]), state=values[0],
                command=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'))


def backup_binding():
    assert socket.gethostname().split('.')[0] == 'h100-10'
    unaliased(BACKUP)
    intent, launch = read(BACKUP / 'INTENT.json'), read(BACKUP / 'LAUNCH.json')
    assert intent['run_id'] == RUN_ID
    assert int(launch['pid']) == PID and int(launch['start_ticks']) == START_TICKS
    plan_path = Path(intent['inventory_path'])
    assert str(plan_path).startswith('/mnt/shared/workspace/training_data_node_8/shaoshitong/')
    assert sha_small(plan_path) == intent['inventory_sha256']
    plan = read(plan_path)
    assert plan['run_id'] == RUN_ID
    candidates = [row for row in plan['files'] if Path(row['source_path']).is_relative_to(CP5)]
    assert candidates and all(Path(row['payload_path']) == Path(row['source_path'].lstrip('/')) for row in candidates)
    return dict(original_intent_sha256=sha_small(BACKUP / 'INTENT.json'),
                original_launch_sha256=sha_small(BACKUP / 'LAUNCH.json'),
                original_inventory_sha256=intent['inventory_sha256'], actor_inventory_files=len(candidates))


def begin(root, phase, binding, targets):
    root = unaliased(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent, receipt = root / (phase + '-INTENT.json'), root / (phase + '-RECEIPT.json')
    if intent.exists():
        if receipt.exists():
            result = read(receipt)
            assert result['phase'] == phase and result['run_id'] == RUN_ID
            print(json.dumps(dict(action='existing_receipt_no_replay', receipt=str(receipt), result=result)))
            return None
        raise RuntimeError('Existing deletion intent without receipt: inspect actual state; never replay')
    document = dict(phase=phase, run_id=RUN_ID, observed_utc=utc(),
                    user_instruction_original_time=None,
                    authorization='Latest explicit user deletion of superseded trained actors; exact allowlist only',
                    binding=binding, targets=targets, backup_complete_claimed=False)
    once(intent, document)
    return receipt


def finish(receipt, phase, **fields):
    result = dict(phase=phase, run_id=RUN_ID, observed_utc=utc(),
                  backup_complete_claimed=False, **fields)
    once(receipt, result)
    print(json.dumps(result))


def existing(root, phase):
    root = unaliased(root)
    intent, receipt = root / (phase + '-INTENT.json'), root / (phase + '-RECEIPT.json')
    if not intent.exists():
        return False
    assert read(intent)['run_id'] == RUN_ID
    if not receipt.exists():
        raise RuntimeError('Existing intent without receipt: inspect state; never replay')
    result = read(receipt)
    assert result['run_id'] == RUN_ID and result['phase'] == phase
    print(json.dumps(dict(action='existing_receipt_no_replay', receipt=str(receipt), result=result)))
    return True


def backup_stop(audit_root):
    if existing(audit_root, 'backup-stop'):
        return
    binding = backup_binding()
    before = process()
    if before is not None:
        assert before['start_ticks'] == START_TICKS, 'PID reused: no signal allowed'
        assert str(BACKUP / 'copy-worker.py') in before['command'] and str(BACKUP) in before['command']
    receipt = begin(audit_root, 'backup-stop', binding, [dict(pid=PID, start_ticks=START_TICKS)])
    if receipt is None:
        return
    signalled = False
    if before is not None and before['state'] != 'Z':
        fd = os.pidfd_open(PID)
        try:
            live = process()
            assert live and live['start_ticks'] == START_TICKS
            signal.pidfd_send_signal(fd, signal.SIGTERM)
            signalled = True
        finally:
            os.close(fd)
    deadline = time.monotonic() + 60
    while True:
        current = process()
        if current is None or current['start_ticks'] != START_TICKS or current['state'] == 'Z':
            break
        assert time.monotonic() < deadline, 'Own backup did not stop; no escalation or deletion'
        time.sleep(.25)
    finish(receipt, 'backup-stop', sigterm_sent=signalled, original_birth_live=False,
           pid=PID, start_ticks=START_TICKS, no_foreign_signal=True)


def delete_backup(audit_root):
    if existing(audit_root, 'delete-backup'):
        return
    binding = backup_binding()
    stop = read(Path(audit_root) / 'backup-stop-RECEIPT.json')
    assert stop['run_id'] == RUN_ID and stop['original_birth_live'] is False
    live = process()
    assert live is None or live['start_ticks'] != START_TICKS or live['state'] == 'Z'
    target = inventory(BACKUP_ACTOR)
    receipt = begin(audit_root, 'delete-backup', binding, [target])
    if receipt is None:
        return
    shutil.rmtree(BACKUP_ACTOR)
    assert not BACKUP_ACTOR.exists()
    finish(receipt, 'delete-backup', deleted=[target], remaining_staged_assets_preserved=True,
           official_model_and_teacher_assets_deleted=False)


def source_delete(audit_root, new_base):
    if existing(audit_root, 'source-delete'):
        return
    assert socket.gethostname().split('.')[0] == 'h100-8'
    new_base = unaliased(new_base)
    assert new_base not in (FRESH, RESUME)
    terminal_path = new_base / 'preparation/warmup10-old-terminal.json'
    terminal = read(terminal_path)
    assert terminal['job_id'] == 3400 and terminal['run_id'] == RUN_ID
    assert terminal['native_state'].split('+')[0] in TERMINAL
    assert terminal['original_births_live'] == 0
    assert terminal['cgroup_released'] and terminal['owned_gpu_contexts_released'] and terminal['cleanup_true']
    old_terminal = read(FRESH / 'preparation/terminal3399-once-v1/RESULT.json')
    assert old_terminal['job_id'] == 3399 and old_terminal['cleanup_true']
    assert old_terminal['cgroup_gone'] and old_terminal['original_processes_gone']
    for base in (FRESH, RESUME):
        unaliased(base)
        assert read(base / 'logs/run-manifest.json')['run_id'] == RUN_ID
    targets = [inventory(CP5), inventory(CP10)]
    receipt = begin(audit_root, 'source-delete', dict(terminal_path=str(terminal_path),
                    terminal_sha256=sha_small(terminal_path), source_jobs=[3399, 3400]), targets)
    if receipt is None:
        return
    for target in (CP5, CP10):
        shutil.rmtree(target)
        assert not target.exists()
    finish(receipt, 'source-delete', deleted=targets, official_model_and_teacher_assets_deleted=False,
           initial_probe_heads_features_labels_logs_code_preserved=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['backup-stop', 'delete-backup', 'source-delete'])
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--new-base', type=Path)
    args = parser.parse_args()
    if args.phase == 'backup-stop':
        backup_stop(args.audit_root)
    elif args.phase == 'delete-backup':
        delete_backup(args.audit_root)
    else:
        assert args.new_base is not None, 'New warmup10 base holding actual terminal evidence required'
        source_delete(args.audit_root, args.new_base)


if __name__ == '__main__':
    main()
