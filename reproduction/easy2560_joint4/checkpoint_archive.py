"""CPU-only full actor/state archive; no Prefix20 head or paired-head contract.

The source snapshot binds every actual HF shard, all four Adam/scheduler/RNG
files, driver stream state, source, data identities and grounding dependencies.
HKUST3 receives a tar stream directly from HPC3 and checks every member SHA.
An archive receipt is a byte-preservation claim, never an actual CUDA restore.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time

from .control import own_base, publish, sha

STEPS = (5, 50, 100, 150, 200)
STATUS = 'EASY2560_FULL_CHECKPOINT_SOURCE_ARCHIVE_SHA_PASS'


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def document(path):
    with Path(path).open() as f: return json.load(f)


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p) != name or not p.parts:
        raise ValueError('EASY2560_ARCHIVE_UNSAFE_NAME')
    return name


def checkpoint(base, step):
    if step not in STEPS: raise ValueError('EASY2560_ARCHIVE_FIXED_MILESTONE_REQUIRED')
    return own_base(base) / 'checkpoints/search-g1-common8/actor' / ('global_step_' + str(step))


def snapshot(base, commit, step):
    base = own_base(base); cp = checkpoint(base, step)
    root = base / 'preparation/checkpoint-archives' / ('global_step_' + str(step))
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'snapshot.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    target = root / 'source-snapshot.json'
    if target.exists(): return document(target)
    run = document(base / 'logs/run-manifest.json')
    if run['source_commit'] != commit or not run.get('easy2560_selection') or run['training_world_size'] != 4:
        raise ValueError('EASY2560_ARCHIVE_RUN_BINDING_DIFFERS')
    complete = document(cp / 'complete.json')
    required = {'training_state/rank-%d.pt' % rank for rank in range(4)} | {
        'training_state/driver.pt', 'training_state/manifest.json'}
    if (complete['run_id'] != run['run_id'] or complete['step'] != step or
            complete.get('finite_all_saved_tensors') is not True or
            not required.issubset(complete['files_sha256'])):
        raise ValueError('EASY2560_ARCHIVE_FULL_NATIVE4_STATE_NOT_READY')
    state = document(cp / 'training_state/manifest.json')
    if state['completed_outer'] != step or state.get('question_probe') is not None:
        raise ValueError('EASY2560_ARCHIVE_UNEXPECTED_QUESTION_HEAD_OR_STEP')
    raw_updates = (base / 'logs/actual-updates.jsonl').read_bytes()
    updates = [json.loads(line) for line in raw_updates.split(b'\n')[:-1] if line]
    if not any(r['step'] == step and r['run_id'] == run['run_id'] for r in updates):
        raise ValueError('EASY2560_ARCHIVE_CHECKPOINT_OUTER_NOT_COMPLETED')
    files = {}

    def add(name, path, expected=None):
        safe_name(name); path = Path(path)
        if name in files or not path.is_file(): raise ValueError('EASY2560_ARCHIVE_MISSING_OR_DUPLICATE_FILE')
        before = path.stat(); actual = sha(path); after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('EASY2560_ARCHIVE_SOURCE_CHANGED_WHILE_HASHING')
        if expected is not None and expected != actual: raise ValueError('EASY2560_ARCHIVE_PRODUCER_SHA_DIFFERS')
        files[name] = dict(path=str(path.resolve()), bytes=after.st_size, sha256=actual)

    for name, expected in sorted(complete['files_sha256'].items()): add('checkpoint/' + name, cp / name, expected)
    add('checkpoint/complete.json', cp / 'complete.json')
    # Preserve tokenizer/config accessories even if the producer hashes only
    # model shards. The shard set is driven by the actual saved HF index.
    for path in sorted(cp.iterdir()):
        name = 'checkpoint/' + path.name
        if path.is_file() and name not in files: add(name, path)
    add('metadata/run-manifest.json', base / 'logs/run-manifest.json')
    for name, expected in run['input_sha256'].items(): add('data/' + name, base / 'data' / name, expected)
    for name, expected in run['source_sha256'].items(): add('source/' + name, base / 'repo' / name, expected)
    versions = sorted((base / 'artifacts').glob('round_*/version-manifest.json'))
    for manifest in versions:
        value = document(manifest)
        if value.get('run_id') != run['run_id'] or value.get('round', step + 1) > step: continue
        folder = manifest.parent
        for path in sorted(folder.rglob('*')):
            if path.is_file(): add('grounding/' + folder.name + '/' + str(path.relative_to(folder)), path)
        # An encoder can be the official actor or a previous saved boundary.
        # Include the exact dependency bytes without creating another HPC copy.
        encoder = Path(value['encoder'])
        for name, expected in sorted(value['encoder_sha256'].items()):
            alias = 'encoders/round-%d/' % value['round'] + name
            add(alias, encoder / name, expected)
    output = dict(schema_version=1, role='easy2560_full_checkpoint_source_snapshot',
                  base=str(base), source_commit=commit, run_id=run['run_id'], step=step,
                  training_world_size=4, question_probe_saved=False, files=files,
                  completed_update=next(r for r in updates if r['step'] == step and r['run_id'] == run['run_id']),
                  total_bytes=sum(r['bytes'] for r in files.values()), checked_unix=time.time(),
                  actual_cuda_restore_accepted=False)
    publish(target, output)
    return output


class HashReader:
    def __init__(self, stream): self.stream = stream; self.digest = hashlib.sha256(); self.count = 0
    def read(self, n=-1):
        data = self.stream.read(n); self.digest.update(data); self.count += len(data); return data


def emit(snapshot_path, output):
    data = document(snapshot_path)
    with tarfile.open(fileobj=output, mode='w|') as tar:
        raw = encoded(data); info = tarfile.TarInfo('source-snapshot.json'); info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
        for name, record in sorted(data['files'].items()):
            safe_name(name); path = Path(record['path']); before = path.stat()
            if before.st_size != record['bytes']: raise ValueError('EASY2560_ARCHIVE_STREAM_SOURCE_SIZE_CHANGED')
            with path.open('rb') as src:
                reader = HashReader(src); info = tarfile.TarInfo(name); info.size = record['bytes']
                tar.addfile(info, reader)
            if reader.count != record['bytes'] or reader.digest.hexdigest() != record['sha256']:
                raise ValueError('EASY2560_ARCHIVE_STREAM_SOURCE_SHA_CHANGED')
    output.flush()


def receive(directory, stream):
    directory = Path(directory)
    if directory.exists(): raise ValueError('EASY2560_ARCHIVE_DESTINATION_EXISTS_INSPECT_RECEIPT_OR_PARTIAL')
    directory.mkdir(parents=True)
    seen = set(); snap = None; started = time.time()
    with tarfile.open(fileobj=stream, mode='r|') as tar:
        for member in tar:
            name = safe_name(member.name)
            if not member.isfile() or name in seen: raise ValueError('EASY2560_ARCHIVE_UNEXPECTED_MEMBER')
            seen.add(name); content = tar.extractfile(member)
            if name == 'source-snapshot.json':
                if len(seen) != 1 or member.size > 32 * 1024**2: raise ValueError('EASY2560_ARCHIVE_SNAPSHOT_FIRST_REQUIRED')
                raw = content.read(); snap = json.loads(raw); publish(directory / name, snap); continue
            if snap is None or name not in snap['files']: raise ValueError('EASY2560_ARCHIVE_UNLISTED_MEMBER')
            record = snap['files'][name]
            if member.size != record['bytes']: raise ValueError('EASY2560_ARCHIVE_MEMBER_SIZE_DIFFERS')
            target = directory / name; target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256(); count = 0
            with target.open('xb') as out:
                for block in iter(lambda: content.read(8 * 1024**2), b''):
                    out.write(block); h.update(block); count += len(block)
                out.flush(); os.fsync(out.fileno())
            if count != record['bytes'] or h.hexdigest() != record['sha256']:
                raise ValueError('EASY2560_ARCHIVE_RECEIVED_SHA_DIFFERS')
            publish_progress = dict(phase='RECEIVING', run_id=snap['run_id'], step=snap['step'],
                                    accepted_files=len(seen) - 1, accepted_bytes=sum(
                                        snap['files'][n]['bytes'] for n in seen if n in snap['files']),
                                    started_unix=started, checked_unix=time.time())
            temporary = directory / 'progress.json.partial'
            temporary.write_bytes(encoded(publish_progress)); temporary.replace(directory / 'progress.json')
    if snap is None or seen != set(snap['files']) | {'source-snapshot.json'}:
        raise ValueError('EASY2560_ARCHIVE_STREAM_INCOMPLETE')
    # Fresh destination readback, independent of the incoming streaming digest.
    for name, record in snap['files'].items():
        if sha(directory / name) != record['sha256']: raise ValueError('EASY2560_ARCHIVE_FRESH_DESTINATION_SHA_DIFFERS')
    for root, _, _ in os.walk(directory, topdown=False):
        fd = os.open(root, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    receipt = dict(status='EASY2560_ARCHIVE_RECEIVED_FRESH_SHA_FSYNC_PASS', run_id=snap['run_id'],
                   base=snap['base'], source_commit=snap['source_commit'], step=snap['step'],
                   snapshot_sha256=sha(directory / 'source-snapshot.json'),
                   file_count=len(snap['files']), total_bytes=snap['total_bytes'],
                   fresh_destination_sha_verified=True, destination_fsynced=True,
                   actual_cuda_restore_accepted=False, checked_unix=time.time())
    publish(directory / 'archive-receipt.json', receipt)
    return receipt


def accept(snapshot_path, receipt):
    snap = document(snapshot_path)
    for key in ('base', 'run_id', 'source_commit', 'step', 'total_bytes'):
        if receipt[key] != snap[key]: raise ValueError('EASY2560_ARCHIVE_ACK_IDENTITY_DIFFERS')
    if (receipt['status'] != 'EASY2560_ARCHIVE_RECEIVED_FRESH_SHA_FSYNC_PASS' or
            not receipt['fresh_destination_sha_verified'] or not receipt['destination_fsynced'] or
            receipt['snapshot_sha256'] != hashlib.sha256(encoded(snap)).hexdigest() or
            receipt['file_count'] != len(snap['files'])):
        raise ValueError('EASY2560_ARCHIVE_ACK_DIFFERS')
    for record in snap['files'].values():
        if Path(record['path']).stat().st_size != record['bytes'] or sha(record['path']) != record['sha256']:
            raise ValueError('EASY2560_ARCHIVE_FRESH_SOURCE_SHA_DIFFERS')
    result = dict(receipt, status=STATUS, fresh_source_after_archive_sha_verified=True,
                  source_snapshot=str(snapshot_path), actual_cuda_restore_accepted=False,
                  checked_unix=time.time())
    publish(Path(snapshot_path).parent / 'complete.json', result)
    return result


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='mode', required=True)
    s = sub.add_parser('snapshot'); s.add_argument('--base', required=True); s.add_argument('--commit', required=True); s.add_argument('--step', type=int, required=True)
    e = sub.add_parser('stream'); e.add_argument('--snapshot', required=True)
    r = sub.add_parser('receive'); r.add_argument('--directory', required=True)
    a = sub.add_parser('accept'); a.add_argument('--snapshot', required=True)
    args = p.parse_args()
    if args.mode == 'stream': emit(args.snapshot, sys.stdout.buffer); return
    if args.mode == 'snapshot': result = snapshot(args.base, args.commit, args.step)
    elif args.mode == 'receive': result = receive(args.directory, sys.stdin.buffer)
    else: result = accept(args.snapshot, json.load(sys.stdin))
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
