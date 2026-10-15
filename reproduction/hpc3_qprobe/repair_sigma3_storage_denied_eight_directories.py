"""Apply prior shared-group authorization to eight actually denied directories.

Only the original MOU owner may add group read/traverse to these exact existing
mode700 inodes. No traversal, source edit, deletion or training control occurs.
Unknown/existing intents must be inspected and are never replayed.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
PARENTS=(ROOT.parent/'qprobe-teacher-warmup200-periodic25-thread5-20261016/tmp',ROOT.parent/'qprobe-teacher-warmup200-periodic25-thread5-20261016-r1/tmp')
CONTROL=ROOT/'preparation/storage-denied-eight-directory-permission-repair-20261016'


def publish(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def matches(s,row,mode=0o700):
    return (stat.S_ISDIR(s.st_mode) and s.st_uid==21930 and s.st_gid==21919 and
            stat.S_IMODE(s.st_mode)==mode and s.st_ino==row['inode'] and s.st_dev==row['device'])


def main():
    assert os.getuid()==21930 and 21919 in os.getgroups(),'original authorized MOU owner/group required'
    file=Path(__file__).resolve();repo=file.parents[2]
    assert repo.parent==ROOT/'source' and len(repo.name)==40
    proof=json.loads((ROOT/'preparation'/('source-'+repo.name+'-verification.json')).read_text())
    assert proof['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS' and proof['commit']==proof['published_remote_sha']==repo.name
    targets=file.with_name('storage_denied_eight_directory_targets_20261016.json');rows=json.loads(targets.read_text())
    for path in (file,targets):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==proof['files'][str(path.relative_to(repo))]
    assert len(rows)==8 and len({r['path'] for r in rows})==8
    fds=[]
    try:
        for row in rows:
            p=Path(row['path'])
            assert any(parent in p.parents for parent in PARENTS) and p.resolve()==p and not p.is_symlink()
            assert row['uid']==21930 and row['gid']==21919 and row['mode']=='0o700'
            assert matches(p.lstat(),row)
            fd=os.open(str(p),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            assert matches(os.fstat(fd),row);fds.append((fd,row))
        CONTROL.mkdir(mode=0o770)
        publish(CONTROL/'intent.json',dict(source_commit=repo.name,source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            targets=rows,authorization='existing explicit shared experiments group access; exact actually denied directories only',
            actual_uid=os.getuid(),created_unix=time.time(),old_mode='700',new_mode='750',recursive=False,training_actions=0))
        changed=[]
        for fd,row in fds:
            p=Path(row['path']);assert matches(p.lstat(),row) and matches(os.fstat(fd),row)
            os.fchmod(fd,0o750);os.fsync(fd)
            assert matches(os.fstat(fd),row,0o750) and matches(p.lstat(),row,0o750)
            changed.append(dict(path=row['path'],inode=row['inode'],before='700',after='750'))
            publish(CONTROL/('changed-'+str(len(changed))+'.json'),changed[-1])
        result=dict(status='EXACT_EIGHT_SHARED_DIRECTORY_GROUP_READ_TRAVERSE_REPAIRED',source_commit=repo.name,
            changed=changed,completed_unix=time.time(),recursive=False,training_actions=0)
        publish(CONTROL/'result.json',result);print(json.dumps(result),flush=True)
    finally:
        for fd,_ in fds:os.close(fd)


if __name__=='__main__':main()
