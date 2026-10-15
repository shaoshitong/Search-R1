"""Prior shared-experiments authority, exact actual denied MOU directories only.

Input must be an owned, terminal, published DU failure with unchanged SHA.
Only periodic25 temporary directory inodes reported by that failure and their
owned mode700 descendants may become750. No file contents/modes, models,
training control, signals, recursive chmod, or other-agent messages.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
RELATIVE='reproduction/hpc3_qprobe/repair_actual_storage_denials.py'
ALLOWED=re.compile(re.escape(str(ROOT.parent))+r'/qprobe-teacher-warmup200-periodic25-thread5-20261016(?:-r[0-9]+)?/tmp/(?:torchelastic_[A-Za-z0-9_]+|tmp[A-Za-z0-9_]+)(?:/[A-Za-z0-9_.-]+)*')


def need(ok,message):
    if not ok:raise ValueError('ACTUAL_STORAGE_DENIAL_REPAIR: '+message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document(path,owner):
    path=Path(path);s=path.lstat()
    need(path.resolve()==path and stat.S_ISREG(s.st_mode) and s.st_uid==owner and s.st_size<2**21,
         'unsafe metadata owner/path/size')
    raw=path.read_bytes();after=path.lstat()
    need((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==
         (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'metadata changed')
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def birth(pid):
    try:
        path=Path('/proc')/str(pid);fields=(path/'stat').read_text().rsplit(') ',1)[1].split()
        if fields[0]=='Z':return None
        return dict(pid=int(pid),start_ticks=int(fields[19]),sid=int(fields[3]),uid=path.stat().st_uid)
    except FileNotFoundError:return None


def publish(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(str(Path(path).parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def reported_roots(stderr):
    out=[]
    for line in stderr.splitlines():
        match=re.fullmatch(r"du: cannot read directory '([^'\n]+)': Permission denied",line)
        if match:
            path=Path(match.group(1))
            need(ALLOWED.fullmatch(str(path)) and '..' not in path.parts and path.resolve()==path,
                 'reported directory outside authorized temporary scope')
            out.append(path)
        else:
            need(line.startswith('du: ') and str(ROOT.parent)+'/' in line and
                 line.endswith(': No such file or directory'),'unrecognized/mixed failure type')
    need(out and len(out)==len(set(out)) and len(out)<=32,'missing/duplicate/excess denied roots')
    return out


def code_guard():
    need(os.getuid()==21930 and 21919 in os.getgroups(),'actual authorized MOU owner/group required')
    code=Path(__file__).resolve();repo=code.parents[2]
    need(repo.parent==ROOT/'source' and re.fullmatch('[0-9a-f]{40}',repo.name),'published immutable code required')
    proof,_=document(ROOT/'preparation'/('source-'+repo.name+'-verification.json'),14090)
    need(proof['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS' and proof['commit']==proof['published_remote_sha']==repo.name and
         sha(code)==proof['files'][RELATIVE],'executed helper publication differs')
    return repo.name


def failure(path,expected_sha):
    path=Path(path)
    need(ROOT/'observations' in path.parents and path.name=='result.json','only actual own storage failure receipt accepted')
    parts=path.relative_to(ROOT/'observations').parts
    need((len(parts)==2 and parts[0].startswith('storage-sigma3-')) or
         (len(parts)==3 and parts[0]=='storage-sigma3-after-actual-denial-repair-20261016' and
          re.fullmatch(r'attempt-(?:[1-9]|1[0-6])',parts[1])), 'unapproved failure control path')
    value,digest=document(path,14090)
    need(digest==expected_sha and value['status']=='ACTUAL_DU_FAILED_NO_USAGE_ACCEPTED' and
         value['actual_du_exit_code']==1 and value['actual_du_child_still_live'] is False and
         value['uid']==14090 and value['measure']==str(ROOT.parent) and
         value['stderr_truncated'] is False and value['stderr_bytes']==len(value['stderr'].encode()) and
         value['stderr_sha256']==hashlib.sha256(value['stderr'].encode()).hexdigest(),
         'actual terminal failure/source/complete error bytes differ')
    need(value['process']['uid']==value['du_process']['uid']==14090 and
         value['du_process']['sid']==value['process']['sid']==value['process']['pid'] and
         birth(value['process']['pid'])!=value['process'] and birth(value['du_process']['pid'])!=value['du_process'],
         'measurement process/held child must actually be terminal')
    source=value['source'];need(re.fullmatch('[0-9a-f]{40}',source['commit']),'invalid observer source commit')
    repo=ROOT/'source'/source['commit']
    proof,_=document(ROOT/'preparation'/('source-'+source['commit']+'-verification.json'),14090)
    need(proof['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS' and proof['commit']==proof['published_remote_sha']==source['commit'] and
         Path(source['path']).resolve()==Path(source['path']) and Path(source['path']).parent==repo/'reproduction/hpc3_qprobe' and
         Path(source['path']).name.startswith('storage_observer_sigma3_') and Path(source['path']).suffix=='.py' and repo in Path(source['path']).parents and
         sha(source['path'])==source['sha256']==proof['files'][str(Path(source['path']).relative_to(repo))],
         'actual failed observer published bytes differ')
    return value,reported_roots(value['stderr'])


def directory_row(path):
    path=Path(path);s=path.lstat()
    need(path.resolve()==path and ALLOWED.fullmatch(str(path)) and stat.S_ISDIR(s.st_mode) and
         s.st_uid==21930 and s.st_gid==21919,'unsafe directory type/owner/path')
    need(stat.S_IMODE(s.st_mode) in (0o700,0o750,0o770,0o2750,0o2770),'unexpected directory mode')
    return dict(path=str(path),inode=s.st_ino,device=s.st_dev,uid=s.st_uid,gid=s.st_gid,mode=stat.S_IMODE(s.st_mode))


def matches(s,row,mode):
    return stat.S_ISDIR(s.st_mode) and (s.st_dev,s.st_ino,s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode))==(
        row['device'],row['inode'],21930,21919,mode)


def existing_reported_roots(roots):
    """A genuinely removed temporary root needs no permission mutation."""
    existing=[];missing=[]
    for root in roots:
        try:directory_row(root)
        except FileNotFoundError:
            need(not root.exists() and not root.is_symlink(),'missing root changed or became a link')
            missing.append(str(root))
        else:existing.append(root)
    return existing,missing


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--failure',type=Path,required=True)
    parser.add_argument('--failure-sha',required=True);a=parser.parse_args();os.umask(0o007)
    commit=code_guard();value,roots=failure(a.failure,a.failure_sha)
    control=ROOT/'preparation'/('storage-actual-denial-repair-'+a.failure_sha[:24])
    need(not control.exists() and not control.is_symlink(),'existing/unknown intent is inspect-only; never replay')
    existing,missing=existing_reported_roots(roots)
    all_dirs={};targets={}
    for root in existing:
        todo=[root]
        while todo:
            path=todo.pop();row=directory_row(path);all_dirs[str(path)]=row
            if row['mode']==0o700:targets[str(path)]=row
            need(len(all_dirs)<=1024,'bounded actual denied tree exceeded')
            for entry in os.scandir(path):
                p=Path(entry.path);s=p.lstat()
                need(not stat.S_ISLNK(s.st_mode),'symlink in denied tree; refuse traversal')
                if stat.S_ISDIR(s.st_mode):todo.append(p)
                else:need(stat.S_ISREG(s.st_mode),'unexpected nonregular member; no file action')
    held=[]
    try:
        for row in targets.values():
            fd=os.open(row['path'],os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            held.append((fd,row))
            need(matches(os.fstat(fd),row,0o700) and matches(Path(row['path']).lstat(),row,0o700),'exact FD changed')
        for fd,row in held:need(matches(os.fstat(fd),row,0o700) and matches(Path(row['path']).lstat(),row,0o700),'pre-change FD differs')
        control.mkdir(mode=0o770)
        publish(control/'intent.json',dict(source_commit=commit,failure_path=str(a.failure),failure_sha256=a.failure_sha,
            original_stderr_sha256=value['stderr_sha256'],roots=[str(p) for p in roots],targets=list(targets.values()),
            missing_reported_roots=missing,
            authorization='prior explicit shared experiments mou176/jxu957 chmod-R770; minimum exact mode700 directory to750',
            actual_process=birth(os.getpid()),created_unix=time.time(),recursive_chmod=False,training_actions=0))
        changed=[]
        for fd,row in held:
            need(matches(os.fstat(fd),row,0o700) and matches(Path(row['path']).lstat(),row,0o700),'FD changed before fchmod')
            os.fchmod(fd,0o750);os.fsync(fd)
            need(matches(os.fstat(fd),row,0o750) and matches(Path(row['path']).lstat(),row,0o750),'post-change identity/mode differs')
            changed.append(dict(path=row['path'],inode=row['inode'],before=0o700,after=0o750))
            publish(control/('changed-'+str(len(changed))+'.json'),changed[-1])
        result=dict(status='ACTUAL_REPORTED_DENIED_SHARED_DIRECTORIES_EXACT_FD_REPAIRED',source_commit=commit,
            failure_sha256=a.failure_sha,changed=changed,inspected_directories=len(all_dirs),recursive_chmod=False,
            missing_reported_roots=missing,
            regular_file_actions=0,training_actions=0,completed_unix=time.time())
        publish(control/'result.json',result);print(json.dumps(dict(control=str(control),**result)),flush=True)
    finally:
        for fd,_ in held:os.close(fd)


if __name__=='__main__':main()
