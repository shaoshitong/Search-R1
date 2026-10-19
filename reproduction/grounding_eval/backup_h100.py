"""Task-scoped, resumable HF export copy; never copies optimizer or credentials."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as stream:
        json.dump(obj, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_DIRECTORY)
    os.fsync(fd)
    os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    source, destination = Path(args.source), Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    lock = open(destination / '.copy.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    names = sorted(p.name for p in source.iterdir() if p.is_file() and (
        p.name.endswith('.safetensors') or p.name in {
            'config.json', 'generation_config.json', 'tokenizer.json',
            'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json',
            'merges.txt', 'added_tokens.json', 'model.safetensors.index.json',
            'complete.json'}))
    if len([x for x in names if x.endswith('.safetensors')]) != 3:
        raise RuntimeError('Expected exactly three source HF shards')
    status = dict(source=str(source), destination=str(destination),
                  pid=os.getpid(), started_at=time.time(), state='COPYING', files=[])
    publish(destination / 'backup-status.json', status)
    try:
        for name in names:
            src, dst = source / name, destination / name
            before = src.stat()
            if shutil.disk_usage(destination).free < before.st_size + (32 << 30):
                raise RuntimeError('Less than source file size plus 32 GiB free; source retained')
            source_hash = sha(src)
            if dst.exists():
                target_hash = sha(dst)
                if dst.stat().st_size != before.st_size or target_hash != source_hash:
                    raise RuntimeError('Existing destination mismatch: ' + name)
                copied = False
            else:
                part = destination / (name + '.partial')
                # A partial belongs solely to this lock-protected task; retain it
                # after failure, but only resume after checking the source prefix.
                offset = part.stat().st_size if part.exists() else 0
                if offset > before.st_size:
                    raise RuntimeError('Partial larger than source: ' + name)
                if offset:
                    a, b = hashlib.sha256(), hashlib.sha256()
                    with open(src, 'rb') as s, open(part, 'rb') as p:
                        left = offset
                        while left:
                            block = s.read(min(8 << 20, left)); a.update(block)
                            b.update(p.read(len(block))); left -= len(block)
                    if a.digest() != b.digest():
                        raise RuntimeError('Partial prefix mismatch: ' + name)
                with open(src, 'rb') as s, open(part, 'ab') as p:
                    s.seek(offset)
                    while True:
                        block = s.read(8 << 20)
                        if not block:
                            break
                        p.write(block)
                        status.update(current_file=name, current_bytes=p.tell(), updated_at=time.time())
                        if p.tell() % (256 << 20) < (8 << 20):
                            p.flush(); publish(destination / 'backup-status.json', status)
                    p.flush(); os.fsync(p.fileno())
                target_hash = sha(part)
                if part.stat().st_size != before.st_size or target_hash != source_hash:
                    raise RuntimeError('Read-back SHA mismatch: ' + name)
                os.replace(part, dst)
                copied = True
            after = src.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Source changed while copying: ' + name)
            status['files'].append(dict(name=name, bytes=before.st_size,
                                        source_sha256=source_hash, target_sha256=target_hash,
                                        copied=copied, verified_at=time.time()))
            status.update(updated_at=time.time())
            publish(destination / 'backup-status.json', status)
        status.update(state='COMPLETE', completed_at=time.time(),
                      total_bytes=sum(x['bytes'] for x in status['files']))
        publish(destination / 'backup-status.json', status)
    except Exception as exc:
        status.update(state='FAILED', error=str(exc), updated_at=time.time())
        publish(destination / 'backup-status.json', status)
        raise


if __name__ == '__main__':
    main()
