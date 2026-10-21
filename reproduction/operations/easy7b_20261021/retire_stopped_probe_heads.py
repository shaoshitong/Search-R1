"""Apply the exact human-approved old QProbe checkpoint list; preserve raw data."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

CONTROL = Path('/tmp/approved-old-qprobe-checkpoints-20261021')


def identity(p):
    s = p.stat()
    return dict(size=s.st_size, allocated=s.st_blocks*512, dev=s.st_dev,
                inode=s.st_ino, nlink=s.st_nlink, uid=s.st_uid, mtime_ns=s.st_mtime_ns)


def allowed(p):
    root = Path('/data/user/jxu957/experiments')
    rel = p.relative_to(root)
    assert re.fullmatch(r'qprobe-teacher-warmup200-[A-Za-z0-9._-]+', rel.parts[0])
    return (len(rel.parts) == 3 and rel.parts[1] == 'warmup'
            and p.name in ('fresh-head.pt','latest-state.pt','continued-head.pt')) or (
            rel.parts[1:-1] == ('artifacts','question-probe','continuous-warmup-adoption')
            and p.name == 'initial.pt')


def main():
    raw = sys.stdin.read(); plan = json.loads(raw)
    assert os.getuid() in (14090,21930)
    query = subprocess.run(['squeue','-u','jxu957,mou176','-h','-o','%i|%j|%T'],
                           capture_output=True,text=True,timeout=10)
    assert query.returncode == 0
    assert all('qprobe' not in line.lower() for line in query.stdout.splitlines()), 'An old QProbe job may be active'
    sha = hashlib.sha256(raw.encode()).hexdigest()
    CONTROL.mkdir(exist_ok=True)
    if CONTROL.stat().st_uid == os.getuid(): CONTROL.chmod(0o770)
    frozen = CONTROL/'plan.json'
    if frozen.exists(): assert frozen.read_text() == raw
    else:
        with frozen.open('x') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        frozen.chmod(0o640)
    claim = CONTROL/('apply-'+str(os.getuid())+'.json')
    assert not claim.exists(), 'Inspect this exact previous apply; never replay unknown results'
    previous = []
    for path in CONTROL.glob('unlink-*.jsonl'):
        previous += [json.loads(line) for line in path.read_text().splitlines() if line]
    removed = {x['path'] for x in previous}
    # Compare every extant target before making the first mutation.
    for row in plan['files']:
        p = Path(row['path']); assert allowed(p) and not p.is_symlink()
        assert p.resolve() == p
        if row['path'] in removed:
            assert not p.exists(); continue
        observed = identity(p)
        for key in ('size','dev','inode','uid','mtime_ns'):
            assert observed[key] == row[key], (str(p), key)
    with claim.open('x') as f:
        json.dump(dict(epoch=time.time(),uid=os.getuid(),plan_sha256=sha,queue=query.stdout),f)
        f.flush(); os.fsync(f.fileno())
    ledger = CONTROL/('unlink-'+str(os.getuid())+'.jsonl')
    done = []; skipped = []
    with ledger.open('x') as out:
        for row in plan['files']:
            p = Path(row['path'])
            if row['path'] in removed: continue
            if not os.access(p.parent,os.W_OK): skipped.append(str(p)); continue
            before = identity(p); p.unlink()
            fd = os.open(p.parent,os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
            assert not p.exists()
            result = dict(path=str(p),before=before,epoch=time.time(),
                          released_lastlink_allocated_bytes=before['allocated'] if before['nlink']==1 else 0)
            out.write(json.dumps(result)+'\n');out.flush();os.fsync(out.fileno());done.append(result)
    print(json.dumps(dict(epoch=time.time(),uid=os.getuid(),plan_sha256=sha,removed=len(done),
                          confirmed_lastlink_released_bytes=sum(x['released_lastlink_allocated_bytes']for x in done),
                          skipped_unwritable=skipped,raw_or_archives_deleted=False,training_changed=False)))


if __name__ == '__main__': main()
