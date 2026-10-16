"""Preserve the requested complete20 state with same-filesystem hardlinks.

This runs outside the immutable training repo. It waits for the real joint20
commit, never changes training, and never claims an independent backup/hash/load.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time


def read(path):
    return json.loads(Path(path).read_text())


def small_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path, value):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as handle:
        json.dump(value, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def assets(base, joint):
    actor, probe = joint['actor'], joint['probe']
    if (joint['outer'] != 20 or actor['completed_outer'] != 20
            or actor['actual_adam_steps_per_rank'] != [400]*8
            or actor['scheduler_completed_outer'] != 20
            or probe['completed_outer'] != 20 or probe['actual_adam_steps_per_rank'] != [100]*8
            or probe['learning_rate'] != 1e-5):
        raise ValueError('Pin only this branch actual complete20/actor400/probe100')
    pool = read(joint['pool_snapshot'])
    if pool['next_outer'] != 21 or pool['consumed_candidates'] != 102400:
        raise ValueError('Retain the committed pool snapshot, not a live reservation')
    rows = []
    def add(source, relative, producer=None):
        source = Path(source)
        if (not source.is_absolute() or '..' in source.parts or base not in source.parents
                or source.resolve() != source or source.is_symlink()):
            raise ValueError('Only canonical current-run assets may be pinned')
        info = source.stat()
        if not stat.S_ISREG(info.st_mode) or (producer and info.st_size != producer['bytes']):
            raise ValueError('Committed source file type/size differs')
        rows.append(dict(source=str(source), relative=relative, bytes=info.st_size,
                         dev=info.st_dev, inode=info.st_ino, mtime_ns=info.st_mtime_ns,
                         producer_sha256=producer['sha256'] if producer else small_sha(source)))
    for name, producer in actor['hf_files'].items():
        if Path(name).name != name:
            raise ValueError('Invalid declared HF filename')
        add(Path(actor['hf_model_dir']) / name, 'actor/huggingface/' + name, producer)
    if (len(actor['rank_states']) != 8 or
            {Path(row['path']).name for row in actor['rank_states']} != {'rank-%d.pt' % r for r in range(8)}):
        raise ValueError('Missing complete8-rank actor state')
    for row in actor['rank_states']:
        add(row['path'], 'actor/' + Path(row['path']).name, row)
    add(actor['driver_state']['path'], 'actor/driver.pt', actor['driver_state'])
    add(joint['probe_snapshot'], 'probe/state.pt', probe)
    add(joint['pool_snapshot'], 'question-pool/state.json')
    add(base / 'question-pool/binding.json', 'question-pool/binding.json')
    add(base / 'preparation/launch-config.json', 'metadata/launch-config.json')
    add(base / 'preparation/source-files.json', 'metadata/source-files.json')
    add(base / 'outers/outer-000020/complete.json', 'metadata/original-joint.json')
    return rows


def relocate(value, mapping):
    if isinstance(value, dict):
        return {key: relocate(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item, mapping) for item in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def pin(base, root):
    base, root = Path(base), Path(root)
    result = root / 'RESULT.json'
    if result.exists():
        return read(result)
    joint_path = base / 'outers/outer-000020/complete.json'
    latest = base / 'latest-joint.json'
    if not joint_path.exists() or not latest.exists():
        return dict(status='PENDING_REAL_JOINT20_COMMIT', training_modified=False)
    # complete.json is written once, whereas latest-joint is atomically
    # published afterwards. Wait for that publication, not mere file creation.
    try:
        if read(latest)['outer'] < 20:
            return dict(status='PENDING_REAL_JOINT20_COMMIT', training_modified=False)
    except json.JSONDecodeError:
        return dict(status='PENDING_REAL_JOINT20_COMMIT', training_modified=False)
    config, joint = read(base / 'preparation/launch-config.json'), read(joint_path)
    if (joint['run_id'] != config['run_id'] or config['probe_learning_rate'] != 1e-5
            or joint['source_commit'] != config['source_commit']):
        raise ValueError('Do not pin another run/recipe')
    intent = root / 'PIN-INTENT.json'
    if intent.exists():
        return dict(status='EXISTING_PARTIAL_PIN_RECONCILE_NO_REPLAY', training_modified=False)
    rows = assets(base, joint)
    with intent.open('x') as handle:
        json.dump(dict(run_id=config['run_id'], outer=20, assets=rows,
                       source_joint_sha256=small_sha(joint_path)), handle, sort_keys=True)
        handle.flush(); os.fsync(handle.fileno())
    pinned, mapping = [], {}
    for row in rows:
        source, target = Path(row['source']), root / 'checkpoint' / row['relative']
        target.parent.mkdir(parents=True, exist_ok=True)
        if signature(source.stat()) != (row['dev'], row['inode'], row['bytes'], row['mtime_ns']):
            raise ValueError('Source identity changed after the committed inventory')
        os.link(source, target)
        if signature(target.stat()) != signature(source.stat()):
            raise ValueError('Pin did not preserve the actual source inode')
        mapping[str(source)] = str(target)
        pinned.append(dict(row, target=str(target), same_source_inode=True))
    actor = joint['actor']
    for key in ('state_path', 'state_dir'):
        mapping[actor[key]] = str(root / 'checkpoint/actor')
    for key in ('model_dir', 'hf_model_dir', 'actor_model_dir'):
        mapping[actor[key]] = str(root / 'checkpoint/actor/huggingface')
    record(root / 'checkpoint/JOINT.json', relocate(joint, mapping))
    record(root / 'checkpoint/RESTORE-MAPPING.json', mapping)
    directories = [p for p in (root / 'checkpoint').rglob('*') if p.is_dir()] + [root / 'checkpoint', root]
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    value = dict(status='COMPLETE20_SAME_FILESYSTEM_PINNED', run_id=config['run_id'], outer=20,
                 actor_adam_per_rank=[400]*8, probe_adam_per_rank=[100]*8,
                 probe_learning_rate=1e-5, committed_pool_next_outer=21,
                 source_joint_sha256=small_sha(joint_path), files=pinned,
                 declared_retained_bytes=sum(row['bytes'] for row in pinned),
                 finished_unix=time.time(), training_modified=False,
                 independent_backup=False, independent_full_model_sha_readback=False,
                 semantic_reload_performed=False, large_bytes_via_Mac=False)
    record(result, value)
    return value


def worker(base, root):
    base, root = Path(base), Path(root)
    intent = read(root / 'INTENT.json')
    if (base.resolve() != base or base.parent != Path('/mnt/data/shaoshitong')
            or not base.name.startswith('search-g1-prefix-selection200-')
            or root != base / 'preparation/retained-joint20-once-v1'
            or small_sha(base / 'preparation/launch-config.json') != intent['config_sha256']
            or small_sha(Path(__file__)) != intent['code_sha256']):
        raise ValueError('Explicit canonical source/code/configuration identity differs')
    os.nice(19)
    deadline = time.monotonic() + 6*3600
    record(root / 'PROGRESS.json', dict(status='WAITING_REAL_JOINT20', pid=os.getpid(), training_modified=False))
    try:
        while time.monotonic() < deadline:
            value = pin(base, root)
            if value['status'] != 'PENDING_REAL_JOINT20_COMMIT':
                return value
            time.sleep(10)
        value = dict(status='WAIT_EXPIRED_NO_TRAINING_CHANGE', training_modified=False)
        record(root / 'RESULT.json', value)
        return value
    except Exception as error:
        value = dict(status='PIN_FAILED_SOURCE_AND_PARTIAL_PRESERVED', error=repr(error), training_modified=False)
        record(root / 'RESULT.json', value)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(json.dumps(worker(args.base, args.root)))
