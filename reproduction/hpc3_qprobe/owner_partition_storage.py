"""Complete two-owner GNU du coverage without changing private directory modes.

Every included number comes from a fresh complete held0/stderr0 scan. The
JXU complement excludes exactly the MOU subtrees measured by the bound owner
child. Cross-partition hard links can only overcount, so admission uses the
sum as a conservative allocated upper bound, never a precise quota claim.
No trace contents, permissions, training, models or credentials are changed.
This module also runs under the login node's Python3.6 for the MOU reader.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
import uuid

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
MEASURE=ROOT.parent
RUN='70827e85fe584f0c84453aae907918a4'
JXU=14090
MOU=21930
GID=21919
PROTOCOL='actual-full-owner-partition-du-v1'
MODULE='reproduction.hpc3_qprobe.owner_partition_storage'
RELATIVE='reproduction/hpc3_qprobe/owner_partition_storage.py'
INITIAL_FAILURE=ROOT/'observations/storage-sigma3-archive100-fourth-20261016-attempt-5f0065b98d1239c994addc19fd0b7e32f506d7fd-17/result.json'
INITIAL_SHA='505d03fab02302fe8ab893dbd7f2e1a6d7a648606f42548ff9d9369e8e76b6e4'
ALLOWED=re.compile(re.escape(str(MEASURE))+r'/qprobe-teacher-warmup200-periodic25-thread5-20261016(?:-r[0-9]+)?/(?:tmp/(?:torchelastic_[A-Za-z0-9_]+|tmp[A-Za-z0-9_]+)|artifacts/behavior-traces)(?:/[A-Za-z0-9_.-]+)*')
CAPS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
MAX_TRANSACTION_SECONDS=900
MAX_RESULT_AGE_SECONDS=600


def need(ok,message):
    if not ok:raise ValueError('OWNER_PARTITION_STORAGE: '+message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def read_bytes(path,owners,limit=4*2**20):
    path=Path(path);s=path.lstat()
    need(path.is_absolute() and path.resolve()==path and stat.S_ISREG(s.st_mode) and
         not path.is_symlink() and s.st_uid in owners and s.st_size<limit,'unsafe retained file')
    with os.fdopen(os.open(str(path),os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as stream:
        opened=os.fstat(stream.fileno())
        need(stat.S_ISREG(opened.st_mode) and opened.st_uid in owners and
             (opened.st_dev,opened.st_ino)==(s.st_dev,s.st_ino),'retained open identity differs')
        raw=stream.read(limit);after=os.fstat(stream.fileno())
    t=path.lstat()
    identity=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    need(identity(s)==identity(opened)==identity(after)==identity(t) and len(raw)==s.st_size,
         'retained bytes changed while reading')
    return raw


def document(path,owners=(JXU,MOU)):
    path=Path(path);raw=read_bytes(path,owners)
    return json.loads(raw),dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def publish(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with temporary.open('x') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    try:os.link(str(temporary),str(path))
    finally:temporary.unlink()
    fsync_dir(path.parent)


def fsync_dir(path):
    fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def replace_state(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with temporary.open('x') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(str(temporary),str(path));fsync_dir(path.parent)


def process(pid):
    try:
        path=Path('/proc')/str(pid);fields=(path/'stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=int(pid),start_ticks=int(fields[19]),sid=int(fields[3]),uid=path.stat().st_uid,state=fields[0])
    except (FileNotFoundError,ProcessLookupError):return None


def birth(row):return {key:row[key] for key in ('pid','start_ticks','sid','uid')}


def live(row):
    now=process(row['pid'])
    return now is not None and now['state']!='Z' and birth(now)==row


def host_guard():
    need(socket.gethostname()=='ACD-Login-1' and
         Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='a27512a3-24a0-4ca3-aa92-98d2a5256d9a' and
         os.readlink('/proc/self/ns/pid')=='pid:[4026531836]','actual pinned login kernel differs')


def publication(commit):
    need(re.fullmatch('[0-9a-f]{40}',commit) is not None,'exact source commit required')
    repo=ROOT/'source'/commit;proof,record=document(ROOT/'preparation'/('source-'+commit+'-verification.json'),(JXU,))
    need(proof['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS' and proof['commit']==proof['published_remote_sha']==commit,
         'full source publication missing')
    need(RELATIVE in proof['files'] and proof['files'][RELATIVE]==digest(repo/RELATIVE),
         'storage entry omitted from published source')
    from reproduction.common8.source_files import runtime_files
    need({str(path.relative_to(repo)) for path in runtime_files(repo)}<=set(proof['files']),
         'runtime source omitted from publication')
    for name,sha in proof['files'].items():
        path=repo/name
        need(path.resolve()==path and repo in path.parents and not path.is_symlink() and digest(path)==sha,
             'published source changed: '+name)
    return repo,dict(commit=commit,publication_path=record['path'],publication_sha256=record['sha256'])


def code_guard(commit,owner):
    need(os.getuid()==owner and GID in os.getgroups(),'actual owner/shared group required')
    host_guard();repo,source=publication(commit)
    need(Path(__file__).resolve()==repo/RELATIVE and os.environ.get('QPROBE_SOURCE')==str(repo),
         'loaded storage helper differs from publication')
    os.umask(0o007)
    for key in CAPS:os.environ[key]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    return repo,source


def permission_roots(stderr):
    roots=[];missing=[]
    for line in stderr.splitlines():
        match=re.fullmatch(r"du: cannot read directory '([^']+)': Permission denied",line)
        if match:
            path=Path(match.group(1))
            need(ALLOWED.fullmatch(str(path)) and '..' not in path.parts and path.resolve()==path,
                 'actual unreadable root is outside the read-only delegation scope')
            roots.append(path)
        else:
            need(line.startswith('du: ') and str(MEASURE)+'/' in line and line.endswith(': No such file or directory'),
                 'unrecognized storage scan error')
            missing.append(line)
    need(len(roots)==len(set(roots)) and len(roots)<=64,'duplicate or excessive roots')
    return roots,missing


def root_identity(path):
    path=Path(path);s=path.lstat()
    need(ALLOWED.fullmatch(str(path)) and path.resolve()==path and stat.S_ISDIR(s.st_mode) and
         s.st_uid==MOU and s.st_gid==GID,'delegated root is not the exact shared MOU directory')
    return dict(path=str(path),inode=s.st_ino,device=s.st_dev,uid=s.st_uid,gid=s.st_gid,mode=stat.S_IMODE(s.st_mode))


def frontier(paths):
    """Keep exactly the observed existing disjoint roots; never repair modes."""
    existing=[];absent=[]
    for path in sorted(set(map(Path,paths)),key=lambda p:(len(p.parts),str(p))):
        if any(parent==path or parent in path.parents for parent in existing):continue
        try:root_identity(path)
        except FileNotFoundError:
            need(not path.exists() and not path.is_symlink(),'reported absent root changed')
            absent.append(str(path))
        else:existing.append(path)
    rows=[root_identity(path) for path in sorted(existing)]
    need(len(rows)<=64,'delegated frontier exceeded')
    return rows,sorted(absent)


def validate_roots(rows,*,current=True):
    paths=[Path(row['path']) for row in rows]
    need(len(paths)==len(set(paths)) and len(paths)<=64,'duplicate or excessive delegated roots')
    need(not any(a!=b and a in b.parents for a in paths for b in paths),'delegated roots overlap')
    for row in rows:
        path=Path(row['path'])
        need(path.is_absolute() and ALLOWED.fullmatch(str(path)) and '..' not in path.parts and
             row['uid']==MOU and row['gid']==GID and type(row['inode']) is int and row['inode']>0 and
             type(row['device']) is int and type(row['mode']) is int and 0<=row['mode']<=0o7777,
             'invalid declared delegated root identity')
        if current:need(root_identity(path)==row,'delegated root identity or mode changed')


def parse_du(stdout,paths):
    """Every requested root must have exactly one complete GNU du result."""
    need(stdout.endswith('\n'),'incomplete du stdout')
    result={}
    for line in stdout.splitlines():
        match=re.fullmatch(r'([0-9]+)\t([^\n]+)',line)
        need(match is not None and match.group(2) not in result,'malformed/duplicate du output')
        result[match.group(2)]=int(match.group(1))
    need(set(result)==set(map(str,paths)) and len(result)==len(paths),'complete requested du root set differs')
    return result


def du_command(paths,excluded=()):
    # Each exclude is an exact canonical absolute path, not a glob or basename.
    need(all(not any(c in str(p) for c in '*?[]\n\r') for p in list(paths)+list(excluded)),
         'du root/exclusion contains pattern syntax')
    return ['du','-s','-B1']+['--exclude='+str(p) for p in excluded]+['--']+list(map(str,paths))


def run_du(directory,label,command,uid):
    """Own the real child until wait; an observation timeout never replays it."""
    need(os.getuid()==uid,'DU owner differs');directory=Path(directory)
    started=time.time();own=birth(process(os.getpid()))
    stdout_path=directory/(label+'-stdout');stderr_path=directory/(label+'-stderr')
    with stdout_path.open('xb') as out,stderr_path.open('xb') as err:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                               env=dict(os.environ,LC_ALL='C'))
        try:
            actual=process(child.pid)
            need(actual is not None and actual['uid']==uid and actual['sid']==own['sid'],'actual DU birth differs')
            du=birth(actual)
            publish(directory/(label+'-binding.json'),dict(actual_parent=own,actual_du=du,command=command,started_unix=started))
        finally:code=child.wait()  # Never orphan a real DU after a binding/publication error.
        out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno())
    stdout=stdout_path.read_bytes();stderr=stderr_path.read_bytes()
    need(len(stdout)<2**20 and len(stderr)<2**20 and not live(du),'DU output excessive or child still live')
    result=dict(actual_parent=own,actual_du=du,actual_du_wait_exit_code=code,command=command,
        stdout=stdout.decode(),stderr=stderr.decode(),stdout_sha256=hashlib.sha256(stdout).hexdigest(),
        stderr_sha256=hashlib.sha256(stderr).hexdigest(),stdout_bytes=len(stdout),stderr_bytes=len(stderr),
        started_unix=started,completed_unix=time.time(),uid=uid,training_actions=0,permission_changes=0)
    publish(directory/(label+'-result.json'),result)
    return result


def consume_du(directory,label,command,uid):
    value,proof=document(Path(directory)/(label+'-result.json'),(uid,))
    bound,_=document(Path(directory)/(label+'-binding.json'),(uid,))
    need(value['command']==bound['command']==command and value['uid']==uid and
         value['actual_parent']==bound['actual_parent'] and value['actual_du']==bound['actual_du'] and
         value['actual_du']['uid']==value['actual_parent']['uid']==uid and
         value['actual_du']['sid']==value['actual_parent']['sid'] and not live(value['actual_du']) and
         type(value['actual_du_wait_exit_code']) is int and value['training_actions']==value['permission_changes']==0 and
         value['started_unix']==bound['started_unix']<=value['completed_unix'],'actual held DU binding differs')
    for stream in ('stdout','stderr'):
        path=Path(directory)/(label+'-'+stream);raw=read_bytes(path,(uid,),2**20)
        need(len(raw)==value[stream+'_bytes'] and hashlib.sha256(raw).hexdigest()==value[stream+'_sha256'] and
             raw.decode()==value[stream],'retained complete DU bytes differ')
    return value,proof


def request_guard(path,commit,*,active=True):
    path=Path(path);value,proof=document(path,(JXU,))
    _,source=publication(commit)
    need(value['protocol']==PROTOCOL and value['source']==source and value['measure']==str(MEASURE) and
         value['run_id']==RUN and value['step'] in (150,200) and type(value['attempt']) is int and
         1<=value['attempt']<=16 and value['training_actions']==value['permission_changes']==0 and
         type(value['created_unix']) in (int,float),
         'owner request scope differs')
    archive=ROOT/'final-control'/RUN/('archive-'+str(value['step']))
    need(path==archive/('owner-storage-attempt-'+str(value['attempt']))/'request.json','owner request path differs')
    claim,claim_proof=document(archive/'submit-claim.json',(JXU,));task,_=document(archive/'task-binding.json',(JXU,))
    entered,_=document(archive/'task-entered.json',(JXU,))
    need(claim_proof['sha256']==value['archive_claim_sha256'] and claim['token']==value['archive_token'] and
         claim['storage_wait_protocol']==PROTOCOL and claim['source']==source and claim['step']==value['step'] and
         task['claim_sha256']==claim_proof['sha256'] and task['actual_process']==value['archive_child'] and
         value['archive_child']['uid']==JXU and entered['actual_process']==value['archive_child'] and
         entered['token']==claim['token'] and entered['claim_sha256']==claim_proof['sha256'] and
         entered['entered_unix']<=value['created_unix'],'owner request archive parent differs')
    if active:
        pointer,_=document(archive/'owner-storage-active.json',(JXU,))
        need(pointer['request_sha256']==proof['sha256'] and pointer['request_path']==str(path) and
             pointer['attempt']==value['attempt'] and pointer['archive_child']==value['archive_child'] and
             live(value['archive_child']),'request stale or archive child not live')
    validate_roots(value['roots'],current=active)
    need(value['excluded_paths']==[row['path'] for row in value['roots']],'exact exclusion/delegation sets differ')
    missing=value['absent_reported_roots']
    need(len(missing)==len(set(missing)) and all(ALLOWED.fullmatch(p) and '..' not in Path(p).parts for p in missing),
         'absent reported frontier scope differs')
    _,initial_proof=document(INITIAL_FAILURE,(JXU,))
    need(initial_proof==value['initial_failure'] and initial_proof['sha256']==INITIAL_SHA,
         'original failed frontier proof differs')
    return value,proof


def owner_command(interpreter,mode,path,commit,token):
    return [interpreter,'-m',MODULE,mode,'--request',str(path),'--commit',commit,'--token',token]


def owner_start(path,commit):
    repo,source=code_guard(commit,MOU);request,proof=request_guard(path,commit)
    need(request['roots'],'no owner scan is needed for an empty frontier')
    directory=Path(path).parent/'mou-control'
    need(not directory.exists() and not directory.is_symlink(),'existing owner claim is inspect-only; never replay')
    directory.mkdir(mode=0o770);token=uuid.uuid4().hex
    command=owner_command(sys.executable,'owner-supervise',path,commit,token)
    publish(directory/'claim.json',dict(role='read_only_MOU_partition_du',token=token,source=source,
        request=proof,command=command,controller=birth(process(os.getpid())),created_unix=time.time(),
        training_actions=0,permission_changes=0))
    with (directory/'worker.log').open('x') as output:
        child=subprocess.Popen(command,cwd=str(repo),env=dict(os.environ,PYTHONPATH=str(repo)),
            stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    actual=process(child.pid)
    need(actual is not None and actual['uid']==MOU and actual['sid']==child.pid,'owner supervisor birth differs')
    value=dict(actual_process=birth(actual),source=source,request=proof,token=token,checked_unix=time.time())
    publish(directory/'process.json',value);return value


def owner_binding(path,commit,token,kind):
    _,source=code_guard(commit,MOU);directory=Path(path).parent/'mou-control'
    receipt=directory/('process.json' if kind=='supervisor' else 'child.json')
    deadline=time.monotonic()+15
    while not receipt.exists():
        need(time.monotonic()<deadline,'owner launch outcome unknown; no replay');time.sleep(.05)
    own=birth(process(os.getpid()));bound,_=document(receipt,(MOU,));claim,_=document(directory/'claim.json',(MOU,))
    need(bound['actual_process']==own and bound['token']==claim['token']==token and bound['source']==claim['source']==source,
         'owner worker actual source/birth differs')
    need(claim['role']=='read_only_MOU_partition_du' and claim['training_actions']==claim['permission_changes']==0 and
         claim['command']==owner_command(claim['command'][0],'owner-supervise',path,commit,token),
         'owner supervisor command/role differs')
    request,proof=request_guard(path,commit)
    need(proof==bound['request']==claim['request'],'owner request changed')
    if kind=='task':
        supervisor,_=document(directory/'process.json',(MOU,))
        need(bound['supervisor']==supervisor['actual_process'] and live(bound['supervisor']) and
             own['sid']==bound['supervisor']['sid'] and own['pid']!=bound['supervisor']['pid'] and
             bound['command']==owner_command(claim['command'][0],'owner-worker',path,commit,token),
             'owner task is not the held original child')
    return directory,source,request,proof,claim,own


def owner_supervise(path,commit,token):
    directory,source,request,proof,claim,own=owner_binding(path,commit,token,'supervisor')
    need(own['pid']==own['sid'],'owner supervisor is not detached')
    command=owner_command(sys.executable,'owner-worker',path,commit,token)
    with (directory/'child.log').open('x') as output:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
        try:
            actual=process(child.pid)
            need(actual is not None and actual['uid']==MOU and actual['sid']==own['sid'],'owner task birth differs')
            task=birth(actual)
            publish(directory/'child.json',dict(actual_process=task,supervisor=own,source=source,request=proof,
                token=token,command=command,started_unix=time.time()))
        finally:code=child.wait()
    publish(directory/'worker-result.json',dict(actual_process=own,actual_task_birth=task,token=token,
        request=proof,source=source,actual_task_wait_exit_code=code,training_actions=0,permission_changes=0,
        completed_unix=time.time()))
    return code


def owner_worker(path,commit,token):
    directory,source,request,proof,claim,own=owner_binding(path,commit,token,'task')
    result=run_du(directory,'mou-du',du_command(request['excluded_paths']),MOU)
    validate_roots(request['roots'])
    need(result['actual_du_wait_exit_code']==0 and result['stderr']=='','owner full scan failed; no allocation accepted')
    parsed=parse_du(result['stdout'],request['excluded_paths'])
    publish(directory/'complete.json',dict(status='ACTUAL_MOU_SUBTREES_FULL_DU_HELD0',source=source,
        request=proof,actual_process=own,du_result_sha256=digest(directory/'mou-du-result.json'),
        roots=request['roots'],root_identities_after=[root_identity(row['path']) for row in request['roots']],
        allocated_by_root=parsed,allocated_bytes=sum(parsed.values()),
        private_modes_unchanged=True,training_actions=0,permission_changes=0,completed_unix=time.time()))
    return 0


def consume_owner(request_path,request,request_proof,*,current=True):
    directory=Path(request_path).parent/'mou-control'
    if not (directory/'worker-result.json').exists():
        if (directory/'process.json').exists():
            supervisor,_=document(directory/'process.json',(MOU,))
            need(live(supervisor['actual_process']),'owner supervisor dead without actual held wait; inspect no replay')
        return None
    if not (directory/'process.json').exists():return None  # Parent publication can trail a very fast child.
    wait,wait_proof=document(directory/'worker-result.json',(MOU,));bound,_=document(directory/'process.json',(MOU,))
    claim,_=document(directory/'claim.json',(MOU,))
    child,_=document(directory/'child.json',(MOU,))
    need(wait['actual_process']==bound['actual_process'] and wait['token']==bound['token']==claim['token'] and
         wait['request']==bound['request']==claim['request']==request_proof and
         wait['source']==bound['source']==claim['source']==request['source'] and
         wait['actual_process']['uid']==MOU and wait['actual_process']['pid']==wait['actual_process']['sid'] and
         wait['permission_changes']==wait['training_actions']==0 and child['actual_process']==wait['actual_task_birth'] and
         child['supervisor']==wait['actual_process'] and child['actual_process']['uid']==MOU and
         child['actual_process']['sid']==wait['actual_process']['sid'] and child['request']==request_proof and
         child['source']==request['source'] and child['token']==claim['token'] and
         claim['role']=='read_only_MOU_partition_du' and claim['training_actions']==claim['permission_changes']==0 and
         claim['command']==owner_command(claim['command'][0],'owner-supervise',request_path,request['source']['commit'],claim['token']) and
         child['command']==owner_command(claim['command'][0],'owner-worker',request_path,request['source']['commit'],claim['token']),
         'owner completion source/request/birth differs')
    if live(wait['actual_process']):return None
    need(type(wait['actual_task_wait_exit_code']) is int and wait['actual_task_wait_exit_code']==0 and
         not live(wait['actual_task_birth']),'owner helper failed; inspect no replay')
    complete,proof=document(directory/'complete.json',(MOU,));validate_roots(request['roots'],current=current)
    command=du_command(request['excluded_paths']);du,du_proof=consume_du(directory,'mou-du',command,MOU)
    parsed=parse_du(du['stdout'],request['excluded_paths'])
    need(du['actual_du_wait_exit_code']==0 and du['stderr']=='' and complete['status']=='ACTUAL_MOU_SUBTREES_FULL_DU_HELD0' and
         complete['source']==request['source'] and complete['request']==request_proof and complete['actual_process']==child['actual_process'] and
         du['actual_parent']==child['actual_process'] and request['created_unix']<=du['started_unix']<=du['completed_unix']<=
         complete['completed_unix']<=wait['completed_unix'] and complete['root_identities_after']==request['roots'] and
         complete['du_result_sha256']==du_proof['sha256'] and complete['roots']==request['roots'] and
         complete['allocated_by_root']==parsed and complete['allocated_bytes']==sum(parsed.values()) and
         complete['private_modes_unchanged'] is True and complete['training_actions']==complete['permission_changes']==0,
         'owner complete exact subtree coverage differs')
    return dict(allocated_bytes=sum(parsed.values()),complete=proof,worker_wait=wait_proof,
                du_result=du_proof,actual_process=wait['actual_process'],actual_task=wait['actual_task_birth'],
                actual_du=du['actual_du'],completed_unix=complete['completed_unix'])


def fresh(request,du,owner,now):
    ends=[du['completed_unix']]+([] if owner is None else [owner['completed_unix']])
    return (0<=now-request['created_unix']<=MAX_TRANSACTION_SECONDS and
            all(0<=now-end<=MAX_RESULT_AGE_SECONDS for end in ends))


def initial_frontier():
    value,proof=document(INITIAL_FAILURE,(JXU,))
    need(proof['sha256']==INITIAL_SHA and value['status']=='ACTUAL_DU_FAILED_NO_USAGE_ACCEPTED' and
         value['actual_du_exit_code']==1 and value['actual_du_child_still_live'] is False and
         value['stderr_truncated'] is False and value['measure']==str(MEASURE) and
         value['uid']==JXU and not live(value['process']) and not live(value['du_process']) and
         hashlib.sha256(value['stderr'].encode()).hexdigest()==value['stderr_sha256'],
         'initial actual terminal denied-frontier evidence differs')
    roots,_=permission_roots(value['stderr']);need(roots,'initial actual denied frontier missing')
    return roots,proof


def measure(claim_path,token,run_id,step):
    """Called only by the already-held archive child inside archive-storage.lock."""
    from reproduction.hpc3_qprobe import final_control as ctl
    claim_path=Path(claim_path);archive=claim_path.parent;claim,claim_proof=document(claim_path,(JXU,))
    commit=claim['source']['commit'];_,source=code_guard(commit,JXU)
    task,_=document(archive/'task-binding.json',(JXU,));own=birth(process(os.getpid()))
    need(step in (150,200) and run_id==RUN and claim['step']==step and claim['token']==token and
         claim['source']==source and claim['storage_wait_protocol']==PROTOCOL and task['actual_process']==own and
         task['claim_sha256']==claim_proof['sha256'],'actual owner-partition archive child differs')
    known,initial=initial_frontier();enoent=0
    for attempt in range(1,17):
        roots,absent=frontier(known);directory=archive/('owner-storage-attempt-'+str(attempt))
        need(not directory.exists() and not directory.is_symlink(),'existing measurement is inspect-only; never replay')
        directory.mkdir(mode=0o770)
        request=dict(protocol=PROTOCOL,source=source,measure=str(MEASURE),run_id=run_id,step=step,attempt=attempt,
            archive_claim_sha256=claim_proof['sha256'],archive_token=token,archive_child=own,roots=roots,
            excluded_paths=[row['path'] for row in roots],absent_reported_roots=absent,initial_failure=initial,
            training_actions=0,permission_changes=0,created_unix=time.time())
        request_path=directory/'request.json';publish(request_path,request)
        request_proof=dict(path=str(request_path),sha256=digest(request_path),bytes=request_path.stat().st_size)
        replace_state(archive/'owner-storage-active.json',dict(request_path=str(request_path),request_sha256=request_proof['sha256'],
            attempt=attempt,archive_child=own,phase='ACTUAL_FULL_COMPLEMENT_SCAN_AND_OWNER_REQUEST',checked_unix=time.time()))
        complement_command=du_command([MEASURE],request['excluded_paths'])
        complement=run_du(directory,'jxu-du',complement_command,JXU)
        owner=None
        if roots:
            while owner is None:
                need(live(own) and digest(claim_path)==claim_proof['sha256'],'archive parent/claim changed')
                owner=consume_owner(request_path,request,request_proof)
                if owner is None:time.sleep(5)
        validate_roots(roots)
        for path in absent:need(not Path(path).exists() and not Path(path).is_symlink(),'absent reported root reappeared')
        if complement['actual_du_wait_exit_code']!=0 or complement['stderr']!='':
            publish(directory/'failed.json',dict(status='ACTUAL_PARTITION_SCAN_FAILED_NO_USAGE_ACCEPTED',request=request_proof,
                jxu_result_sha256=digest(directory/'jxu-du-result.json'),owner=owner,usage_accepted=False,completed_unix=time.time()))
            need(complement['actual_du_wait_exit_code']==1,'complement scan failed with unrecognized exit')
            new,missing=permission_roots(complement['stderr'])
            if new:known=list(set(known+new))
            else:enoent+=1;need(missing and enoent<3,'pure ENOENT complete retry exhausted')
            continue
        parsed=parse_du(complement['stdout'],[MEASURE])
        if not fresh(request,complement,owner,time.time()):
            publish(directory/'stale.json',dict(status='ACTUAL_COMPLETE_PARTITIONS_STALE_NO_USAGE_ACCEPTED',
                request=request_proof,jxu_result_sha256=digest(directory/'jxu-du-result.json'),owner=owner,
                usage_accepted=False,checked_unix=time.time()))
            continue  # Both original children are terminal; a new numbered full transaction is required.
        upper=parsed[str(MEASURE)]+(0 if owner is None else owner['allocated_bytes'])
        complete=dict(status='ACTUAL_FULL_OWNER_PARTITION_DU_UPPER_BOUND',protocol=PROTOCOL,source=source,
            request=request_proof,archive_claim_sha256=claim_proof['sha256'],archive_child=own,
            jxu_result_sha256=digest(directory/'jxu-du-result.json'),jxu_allocated_bytes=parsed[str(MEASURE)],
            owner=owner,allocated_upper_bound_bytes=upper,complete_partition_coverage=True,
            root_identities_after=[root_identity(row['path']) for row in roots],absent_reported_roots_after=absent,
            cross_partition_hardlinks_may_overcount=True,atomic_usage_or_hardquota_claimed=False,
            private_modes_unchanged=True,training_actions=0,permission_changes=0,completed_unix=time.time())
        publish(directory/'complete.json',complete)
        pointer=dict(path=str(directory/'complete.json'),sha256=digest(directory/'complete.json'),
                     allocated_upper_bound_bytes=upper,protocol=PROTOCOL)
        publish(archive/'storage-owner-partition-accepted.json',pointer)
        # Reconsume actual complete child waits and bytes before copy admission.
        need(consume(archive,allow_active_archive_child=True)['allocated_upper_bound_bytes']==upper,
             'fresh complete partition consumption differs')
        return upper
    raise ValueError('OWNER_PARTITION_STORAGE: sixteen complete attempts exhausted; no usage accepted')


def consume(archive,allow_active_archive_child=False):
    archive=Path(archive);pointer,_=document(archive/'storage-owner-partition-accepted.json',(JXU,))
    complete,proof=document(pointer['path'],(JXU,))
    request,request_proof=request_guard(complete['request']['path'],complete['source']['commit'],active=allow_active_archive_child)
    claim,claim_proof=document(archive/'submit-claim.json',(JXU,))
    need(pointer['sha256']==proof['sha256'] and pointer['protocol']==complete['protocol']==request['protocol']==PROTOCOL and
         request_proof==complete['request'] and complete['source']==request['source']==claim['source'] and
         complete['archive_claim_sha256']==request['archive_claim_sha256']==claim_proof['sha256'] and
         complete['archive_child']==request['archive_child'] and request['archive_token']==claim['token'] and
         Path(proof['path']).parent==Path(request_proof['path']).parent and Path(proof['path']).parent.parent==archive,
         'complete partition pointer/source/request/archive differs')
    if not allow_active_archive_child:need(not live(complete['archive_child']),'archive child still live at final partition consumption')
    _,source=publication(claim['source']['commit']);need(source==claim['source'],'partition source publication changed')
    validate_roots(request['roots'],current=allow_active_archive_child)
    need(complete['root_identities_after']==request['roots'] and
         complete['absent_reported_roots_after']==request['absent_reported_roots'],
         'complete after-scan identities or absence coverage differs')
    if allow_active_archive_child:
        for path in request['absent_reported_roots']:
            need(not Path(path).exists() and not Path(path).is_symlink(),'absent reported root reappeared')
    need(request['excluded_paths']==[row['path'] for row in request['roots']],'partition exclusion set differs')
    directory=Path(proof['path']).parent
    command=du_command([MEASURE],request['excluded_paths']);du,du_proof=consume_du(directory,'jxu-du',command,JXU)
    need(du['actual_parent']==complete['archive_child'] and du['actual_du_wait_exit_code']==0 and du['stderr']=='' and
         complete['jxu_result_sha256']==du_proof['sha256'],'complete complement held0 is missing')
    need(request['created_unix']<=du['started_unix']<=du['completed_unix']<=complete['completed_unix'],
         'complete scan is outside this request window')
    parsed=parse_du(du['stdout'],[MEASURE]);owner=consume_owner(complete['request']['path'],request,request_proof,
        current=allow_active_archive_child) if request['roots'] else None
    need((not request['roots'] or owner is not None) and owner==complete['owner'],'complete owner subtree result differs')
    upper=parsed[str(MEASURE)]+(0 if owner is None else owner['allocated_bytes'])
    if allow_active_archive_child:
        need(fresh(request,du,owner,time.time()),'complete measurement became stale before copy admission')
    need(complete['status']=='ACTUAL_FULL_OWNER_PARTITION_DU_UPPER_BOUND' and
         upper==complete['allocated_upper_bound_bytes']==pointer['allocated_upper_bound_bytes'] and
         parsed[str(MEASURE)]==complete['jxu_allocated_bytes'] and complete['complete_partition_coverage'] is True and
         complete['cross_partition_hardlinks_may_overcount'] is True and complete['atomic_usage_or_hardquota_claimed'] is False and
         complete['private_modes_unchanged'] is True and complete['training_actions']==complete['permission_changes']==0,
         'complete partition arithmetic or safety semantics differ')
    return dict(protocol=PROTOCOL,allocated_upper_bound_bytes=upper,complete=proof,request=request_proof,
                jxu_result=du_proof,owner=owner,complete_partition_coverage=True,exact_usage_or_hardquota_claimed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('owner-start','owner-supervise','owner-worker'))
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--commit',required=True);parser.add_argument('--token')
    args=parser.parse_args()
    if args.mode=='owner-worker':sys.exit(owner_worker(args.request,args.commit,args.token))
    if args.mode=='owner-supervise':sys.exit(owner_supervise(args.request,args.commit,args.token))
    print(json.dumps(owner_start(args.request,args.commit)),flush=True)
