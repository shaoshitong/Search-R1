"""Complete two-owner GNU du coverage with explicit hidepid-safe owner proofs.

Every included number comes from a fresh complete held0/stderr0 scan. The
JXU complement excludes exactly the MOU subtrees measured by the bound owner
child. Cross-partition hard links can only overcount, so admission uses the
sum as a conservative allocated upper bound, never a precise quota claim.
No trace contents, permissions, training, models or credentials are changed.
This module also runs under the login node's Python3.6 for the MOU reader.
Fresh JXU0640 nonce grants admit MOU once; an independent MOU0640 confirmation
checks original own helper births after actual held waits. Foreign /proc is
never consulted as evidence of liveness or terminality.
"""
import argparse
from contextlib import contextmanager
import fcntl
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
PROTOCOL='actual-full-owner-partition-du-v2'
MODULE='reproduction.hpc3_qprobe.owner_partition_storage_v2'
RELATIVE='reproduction/hpc3_qprobe/owner_partition_storage_v2.py'
INITIAL_FAILURE=ROOT/'observations/storage-sigma3-archive100-fourth-20261016-attempt-5f0065b98d1239c994addc19fd0b7e32f506d7fd-17/result.json'
INITIAL_SHA='505d03fab02302fe8ab893dbd7f2e1a6d7a648606f42548ff9d9369e8e76b6e4'
ALLOWED=re.compile(re.escape(str(MEASURE))+r'/qprobe-teacher-warmup200-periodic25-thread5-20261016(?:-r[0-9]+)?/(?:tmp/(?:torchelastic_[A-Za-z0-9_]+|tmp[A-Za-z0-9_]+|ray-thread-smoke-[0-9a-f]{8})|artifacts/behavior-traces)(?:/[A-Za-z0-9_.-]+)*')
CAPS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
MAX_TRANSACTION_SECONDS=900
MAX_RESULT_AGE_SECONDS=600
AUTHORIZATION_TTL=120


def need(ok,message):
    if not ok:raise ValueError('OWNER_PARTITION_STORAGE: '+message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def read_bytes(path,owners,limit=4*2**20,protected=False):
    path=Path(path);s=path.lstat()
    need(path.is_absolute() and path.resolve()==path and stat.S_ISREG(s.st_mode) and
         not path.is_symlink() and s.st_uid in owners and s.st_size<limit,'unsafe retained file')
    with os.fdopen(os.open(str(path),os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as stream:
        opened=os.fstat(stream.fileno())
        need(stat.S_ISREG(opened.st_mode) and opened.st_uid in owners and
             (opened.st_dev,opened.st_ino)==(s.st_dev,s.st_ino),'retained open identity differs')
        if protected:need(stat.S_IMODE(opened.st_mode)==0o640 and opened.st_gid==GID,'protected owner mode/group differs')
        raw=stream.read(limit);after=os.fstat(stream.fileno())
    t=path.lstat()
    identity=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns,x.st_mode,x.st_uid,x.st_gid)
    need(identity(s)==identity(opened)==identity(after)==identity(t) and len(raw)==s.st_size,
         'retained bytes changed while reading')
    return raw


def document(path,owners=(JXU,MOU),protected=False):
    path=Path(path);raw=read_bytes(path,owners,protected=protected)
    return json.loads(raw),dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def publish(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with temporary.open('x') as stream:
        os.fchmod(stream.fileno(),0o640)
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
        os.fchmod(stream.fileno(),0o640)
        json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(str(temporary),str(path));fsync_dir(path.parent)


def process(pid):
    try:
        path=Path('/proc')/str(pid);fields=(path/'stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=int(pid),start_ticks=int(fields[19]),sid=int(fields[3]),uid=path.stat().st_uid,state=fields[0])
    except (FileNotFoundError,ProcessLookupError):return None


def birth(row):return {key:row[key] for key in ('pid','start_ticks','sid','uid')}


def live(row):
    need(row['uid']==os.getuid(),'foreign /proc visibility is unknown; use the owner confirmation')
    now=process(row['pid'])
    return now is not None and now['state']!='Z' and birth(now)==row


def host_guard():
    need(socket.gethostname()=='ACD-Login-1' and
         Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='a27512a3-24a0-4ca3-aa92-98d2a5256d9a' and
         os.readlink('/proc/self/ns/pid')=='pid:[4026531836]','actual pinned login kernel differs')


def host_binding():
    host_guard()
    return dict(hostname=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid_namespace=os.readlink('/proc/self/ns/pid'))


@contextmanager
def admission_lock(request_path):
    path=Path(request_path).parent/'owner-admission.lock'
    before=path.lstat();fd=os.open(str(path),os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        opened=os.fstat(fd)
        need(path.resolve()==path and stat.S_ISREG(opened.st_mode) and opened.st_uid==JXU and
             opened.st_gid==GID and stat.S_IMODE(opened.st_mode)==0o640 and
             (before.st_dev,before.st_ino)==(opened.st_dev,opened.st_ino),'unsafe owner admission lock')
        fcntl.flock(fd,fcntl.LOCK_EX);yield
    finally:os.close(fd)


def authorize_owner(path,commit):
    _,source=code_guard(commit,JXU)
    with admission_lock(path):
        refusal=unclaimed_admission_refusal(path,commit)
        if refusal is not None:return refusal
        request,proof=request_guard(path,commit)
        need(request['roots'] and not (Path(path).parent/'mou-control').exists() and
             not (Path(path).parent/'request-closed.json').exists(),'owner request is closed or already claimed')
        directory=Path(path).parent/'authorizations';directory.mkdir(mode=0o770,exist_ok=True)
        nonce=uuid.uuid4().hex;grant_path=directory/(nonce+'.json')
        grant=dict(protocol=PROTOCOL,source=source,request=proof,archive_child=request['archive_child'],
            archive_claim_sha256=request['archive_claim_sha256'],archive_token=request['archive_token'],
            host=host_binding(),nonce=nonce,authorized_by=birth(process(os.getpid())),created_unix=time.time(),
            ttl_seconds=AUTHORIZATION_TTL,training_actions=0,permission_changes=0)
        publish(grant_path,grant);_,grant_proof=document(grant_path,(JXU,),protected=True)
        replace_state(Path(path).parent/'authorization-active.json',grant_proof)
        return grant_proof


def authorization_guard(path,request,proof,*,claim=None):
    """TTL/current pointer gates only the first exclusive MOU claim."""
    if claim is None:
        ref,_=document(Path(path).parent/'authorization-active.json',(JXU,),protected=True)
    else:ref=claim['authorization']
    grant,grant_proof=document(ref['path'],(JXU,),protected=True)
    need(ref==grant_proof and Path(ref['path'])==Path(path).parent/'authorizations'/(grant['nonce']+'.json') and
         re.fullmatch('[0-9a-f]{32}',grant['nonce']) and grant['source']==request['source'] and
         grant['protocol']==PROTOCOL and grant['request']==proof and grant['archive_child']==request['archive_child'] and
         grant['archive_claim_sha256']==request['archive_claim_sha256'] and grant['archive_token']==request['archive_token'] and
         grant['host']==host_binding() and grant['authorized_by']['uid']==JXU and grant['ttl_seconds']==AUTHORIZATION_TTL and
         grant['training_actions']==grant['permission_changes']==0,'owner authorization source/request/nonce/host differs')
    when=time.time() if claim is None else claim['created_unix']
    need(0<=when-grant['created_unix']<=grant['ttl_seconds'],'owner authorization was expired at first claim')
    if claim is None:
        need(not (Path(path).parent/'request-closed.json').exists(),'owner request already closed')
    else:need(claim['authorization_nonce']==grant['nonce'],'owner claim authorization nonce differs')
    return grant,grant_proof


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
    publish(directory/(label+'-launch-intent.json'),dict(actual_parent=own,command=command,started_unix=started))
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


def consume_du(directory,label,command,uid,terminal_births=None):
    value,proof=document(Path(directory)/(label+'-result.json'),(uid,),protected=True)
    bound,bound_proof=document(Path(directory)/(label+'-binding.json'),(uid,),protected=True)
    launch,launch_proof=document(Path(directory)/(label+'-launch-intent.json'),(uid,),protected=True)
    if uid==os.getuid():need(not live(value['actual_du']),'original owner DU is still live')
    else:need(terminal_births is not None and value['actual_du'] in terminal_births,
              'foreign DU requires its exact protected owner terminal confirmation')
    need(value['command']==bound['command']==launch['command']==command and value['uid']==uid and
         value['actual_parent']==bound['actual_parent']==launch['actual_parent'] and value['actual_du']==bound['actual_du'] and
         value['actual_du']['uid']==value['actual_parent']['uid']==uid and
         value['actual_du']['sid']==value['actual_parent']['sid'] and
         type(value['actual_du_wait_exit_code']) is int and value['training_actions']==value['permission_changes']==0 and
         value['started_unix']==bound['started_unix']==launch['started_unix']<=value['completed_unix'],'actual held DU binding differs')
    for stream in ('stdout','stderr'):
        path=Path(directory)/(label+'-'+stream);raw=read_bytes(path,(uid,),2**20)
        need(len(raw)==value[stream+'_bytes'] and hashlib.sha256(raw).hexdigest()==value[stream+'_sha256'] and
             raw.decode()==value[stream],'retained complete DU bytes differ')
    return value,dict(proof,binding=bound_proof,launch_intent=launch_proof)


def request_guard(path,commit,*,active=True):
    path=Path(path);value,proof=document(path,(JXU,),protected=True)
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
        pointer,_=document(archive/'owner-storage-active.json',(JXU,),protected=True)
        need(pointer['request_sha256']==proof['sha256'] and pointer['request_path']==str(path) and
             pointer['attempt']==value['attempt'] and pointer['archive_child']==value['archive_child'],
             'request is not the actual active archive request')
        if os.getuid()==JXU:need(live(value['archive_child']),'own archive child not live')
        else:need(os.getuid()==MOU,'only the bound two owners may read active requests')
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
    repo,source=code_guard(commit,MOU)
    with admission_lock(path):
        refusal=unclaimed_admission_refusal(path,commit)
        if refusal is not None:return refusal
        request,proof=request_guard(path,commit);grant,grant_proof=authorization_guard(path,request,proof)
        need(request['roots'],'no owner scan is needed for an empty frontier')
        directory=Path(path).parent/'mou-control'
        need(not directory.exists() and not directory.is_symlink(),'existing owner claim is inspect-only; never replay')
        directory.mkdir(mode=0o770);token=uuid.uuid4().hex
        command=owner_command(sys.executable,'owner-supervise',path,commit,token)
        publish(directory/'claim.json',dict(role='read_only_MOU_partition_du',token=token,source=source,
            request=proof,command=command,controller=birth(process(os.getpid())),created_unix=time.time(),
            authorization=grant_proof,authorization_nonce=grant['nonce'],training_actions=0,permission_changes=0))
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
    own=birth(process(os.getpid()));bound,_=document(receipt,(MOU,),protected=True);claim,_=document(directory/'claim.json',(MOU,),protected=True)
    need(bound['actual_process']==own and bound['token']==claim['token']==token and bound['source']==claim['source']==source,
         'owner worker actual source/birth differs')
    need(claim['role']=='read_only_MOU_partition_du' and claim['training_actions']==claim['permission_changes']==0 and
         claim['command']==owner_command(claim['command'][0],'owner-supervise',path,commit,token),
         'owner supervisor command/role differs')
    request,proof=request_guard(path,commit)
    authorization_guard(path,request,proof,claim=claim)
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


def no_du_launched(directory,label):
    for suffix in ('launch-intent.json','binding.json','result.json','stdout','stderr'):
        path=Path(directory)/(label+'-'+suffix)
        need(not path.exists() and not path.is_symlink(),'DU launch outcome unknown without real wait; inspect no replay')


def held_owner_evidence(request_path,request,request_proof,*,current=True,terminal_births=None,require_usage=True):
    directory=Path(request_path).parent/'mou-control'
    if not (directory/'worker-result.json').exists() or not (directory/'process.json').exists():return None
    wait,wait_proof=document(directory/'worker-result.json',(MOU,),protected=True)
    bound,process_proof=document(directory/'process.json',(MOU,),protected=True)
    claim,claim_proof=document(directory/'claim.json',(MOU,),protected=True)
    child,child_proof=document(directory/'child.json',(MOU,),protected=True)
    _,authorization=authorization_guard(request_path,request,request_proof,claim=claim)
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
    helpers=[wait['actual_process'],wait['actual_task_birth']]
    if terminal_births is None:
        need(os.getuid()==MOU,'only MOU may directly check MOU helper terminality')
        if any(live(row) for row in helpers):return None
    else:need(all(row in terminal_births for row in helpers),'owner confirmation helper births differ')
    need(type(wait['actual_task_wait_exit_code']) is int,'actual owner wait missing')
    command=du_command(request['excluded_paths']);du=du_proof=None
    if (directory/'mou-du-result.json').exists():
        du,du_proof=consume_du(directory,'mou-du',command,MOU,terminal_births)
        need(du['actual_parent']==child['actual_process'] and request['created_unix']<=du['started_unix']<=
             du['completed_unix']<=wait['completed_unix'],'owner DU held parent/window differs')
        helpers.append(du['actual_du'])
    else:
        need(wait['actual_task_wait_exit_code']!=0,'successful owner has no retained DU result')
        no_du_launched(directory,'mou-du')
    if terminal_births is not None:need(terminal_births==helpers,'owner confirmation exact terminal birth set differs')
    common=dict(claim=claim_proof,process=process_proof,child=child_proof,worker_wait=wait_proof,
        du_result=du_proof,actual_process=wait['actual_process'],
        actual_task=wait['actual_task_birth'],actual_du=None if du is None else du['actual_du'],
        actual_terminal_births=helpers,authorization=authorization,authorization_nonce=claim['authorization_nonce'],
        actual_task_wait_exit_code=wait['actual_task_wait_exit_code'],
        actual_du_wait_exit_code=None if du is None else du['actual_du_wait_exit_code'])
    if wait['actual_task_wait_exit_code']!=0:
        need(not require_usage,'owner helper failed; no usage may be accepted')
        need(not (directory/'complete.json').exists(),'failed owner worker unexpectedly published success')
        return dict(common,status='ACTUAL_OWNER_FAILED_HELPERS_TERMINAL',allocated_bytes=None,complete=None,
                    completed_unix=wait['completed_unix'])
    complete,proof=document(directory/'complete.json',(MOU,),protected=True);validate_roots(request['roots'],current=current)
    need(du is not None,'successful owner worker has no actual held DU')
    parsed=parse_du(du['stdout'],request['excluded_paths'])
    need(du['actual_du_wait_exit_code']==0 and du['stderr']=='' and complete['status']=='ACTUAL_MOU_SUBTREES_FULL_DU_HELD0' and
         complete['source']==request['source'] and complete['request']==request_proof and complete['actual_process']==child['actual_process'] and
         du['actual_parent']==child['actual_process'] and request['created_unix']<=du['started_unix']<=du['completed_unix']<=
         complete['completed_unix']<=wait['completed_unix'] and complete['root_identities_after']==request['roots'] and
         complete['du_result_sha256']==du_proof['sha256'] and complete['roots']==request['roots'] and
         complete['allocated_by_root']==parsed and complete['allocated_bytes']==sum(parsed.values()) and
         complete['private_modes_unchanged'] is True and complete['training_actions']==complete['permission_changes']==0,
         'owner complete exact subtree coverage differs')
    return dict(common,status='ACTUAL_OWNER_FULL_PARTITION_HELD0',allocated_bytes=sum(parsed.values()),
                complete=proof,completed_unix=complete['completed_unix'])


def owner_confirm(path,commit):
    _,source=code_guard(commit,MOU);request,proof=request_guard(path,commit,active=False)
    directory=Path(path).parent/'mou-control';confirmation=directory/'terminal-confirmation.json'
    if confirmation.exists():
        need(consume_owner(path,request,proof,current=False,require_usage=False) is not None,
             'existing owner confirmation cannot replace its original held graph')
        return document(confirmation,(MOU,),protected=True)[1]
    evidence=held_owner_evidence(path,request,proof,current=False,require_usage=False)
    if evidence is None:return dict(status='ACTUAL_OWN_HELPERS_NOT_YET_TERMINAL',request=proof)
    reporter=birth(process(os.getpid()))
    need(reporter['uid']==MOU and reporter not in evidence['actual_terminal_births'],'owner confirmation reporter differs')
    publish(confirmation,dict(status='ACTUAL_MOU_ORIGINAL_HELPERS_TERMINAL_CONFIRMED',source=source,
        request=proof,host=host_binding(),actual_reporter=reporter,owner_evidence=evidence,
        confirmed_unix=time.time(),training_actions=0,permission_changes=0))
    return document(confirmation,(MOU,),protected=True)[1]


def consume_owner(request_path,request,request_proof,*,current=True,require_usage=True):
    """JXU never guesses a MOU process state from hidden /proc entries."""
    directory=Path(request_path).parent/'mou-control';path=directory/'terminal-confirmation.json'
    if not path.exists():return None
    terminal,proof=document(path,(MOU,),protected=True)
    expected=terminal['owner_evidence']
    need(terminal['status']=='ACTUAL_MOU_ORIGINAL_HELPERS_TERMINAL_CONFIRMED' and
         terminal['source']==request['source'] and terminal['request']==request_proof and terminal['host']==host_binding() and
         terminal['actual_reporter']['uid']==MOU and terminal['actual_reporter'] not in expected['actual_terminal_births'] and
         terminal['training_actions']==terminal['permission_changes']==0 and
         expected['completed_unix']<=terminal['confirmed_unix'],'protected original owner terminal report differs')
    actual=held_owner_evidence(request_path,request,request_proof,current=current,
        terminal_births=expected['actual_terminal_births'],require_usage=require_usage)
    need(actual is not None and actual==expected,'owner terminal confirmation does not match original held bytes')
    return dict(actual,terminal_confirmation=proof)


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
        with (directory/'owner-admission.lock').open('xb') as lock:
            os.fchmod(lock.fileno(),0o640);lock.flush();os.fsync(lock.fileno())
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
        if complement['actual_du_wait_exit_code']!=0 or complement['stderr']!='':
            # Serialize closure with the first exclusive MOU admission. No
            # owner claim means no useless owner scan or cross-owner wait.
            with admission_lock(request_path):
                has_owner=(directory/'mou-control').exists()
                need(not (directory/'mou-control').is_symlink(),'owner control is a symlink')
                publish(directory/'request-closed.json',dict(status='ACTUAL_FAILED_COMPLEMENT_REQUEST_CLOSED',
                    source=source,request=request_proof,owner_claim_present=has_owner,closed_unix=time.time(),
                    training_actions=0,permission_changes=0))
            if has_owner:
                while owner is None:
                    owner=consume_owner(request_path,request,request_proof,current=False,require_usage=False)
                    if owner is None:time.sleep(5)
            publish(directory/'failed.json',dict(status='ACTUAL_PARTITION_SCAN_FAILED_NO_USAGE_ACCEPTED',request=request_proof,
                jxu_result_sha256=digest(directory/'jxu-du-result.json'),owner=owner,usage_accepted=False,completed_unix=time.time()))
            need(complement['actual_du_wait_exit_code']==1,'complement scan failed with unrecognized exit')
            new,missing=permission_roots(complement['stderr'])
            if new:known=list(set(known+new))
            else:enoent+=1;need(missing and enoent<3,'pure ENOENT complete retry exhausted')
            continue
        if roots:
            while owner is None:
                need(live(own) and digest(claim_path)==claim_proof['sha256'],'archive parent/claim changed')
                owner=consume_owner(request_path,request,request_proof)
                if owner is None:time.sleep(5)
        validate_roots(roots)
        for path in absent:need(not Path(path).exists() and not Path(path).is_symlink(),'absent reported root reappeared')
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


def unclaimed_admission_refusal(path,commit):
    """Caller holds the admission lock; only proven zero-claim races are benign."""
    path=Path(path);request,proof=request_guard(path,commit,active=False)
    directory=path.parent/'mou-control'
    need(not directory.exists() and not directory.is_symlink(),
         'existing owner claim is inspect-only; never replay')
    archive=path.parent.parent
    pointer,pointer_proof=document(archive/'owner-storage-active.json',(JXU,),protected=True)
    active_path=Path(pointer['request_path']);active,active_proof=request_guard(active_path,commit,active=False)
    need(pointer['request_sha256']==active_proof['sha256'] and pointer['attempt']==active['attempt'] and
         pointer['archive_child']==active['archive_child'] and active['step']==request['step'],
         'active pointer cannot prove the current source-bound request')
    closed=path.parent/'request-closed.json';closed_proof=None
    if closed.exists():
        value,closed_proof=document(closed,(JXU,),protected=True)
        need(value['status']=='ACTUAL_FAILED_COMPLEMENT_REQUEST_CLOSED' and value['source']==request['source'] and
             value['request']==proof and value['owner_claim_present'] is False,
             'closed request is not a proven zero-owner-claim refusal')
    if active_path==path and closed_proof is None:return None
    need(not directory.exists() and not directory.is_symlink(),'owner claim appeared while admission lock held')
    return dict(status='ACTUAL_OWNER_REQUEST_NOT_ADMITTED',protocol=PROTOCOL,source=request['source'],request=proof,
        active_request=active_proof,active_pointer=pointer_proof,closed_request=closed_proof,
        reason='closed_or_superseded_request_with_no_owner_claim',owner_claim_created=False,
        training_actions=0,permission_changes=0,checked_unix=time.time())


def inspect_storage(step,commit):
    """JXU read-only action selection; foreign process visibility is never used."""
    _,source=code_guard(commit,JXU)
    need(type(step) is int and step in (150,200),'inspection is scoped to150/200')
    from reproduction.hpc3_qprobe import final_control as ctl
    archive=ROOT/'final-control'/RUN/('archive-'+str(step))
    value=dict(step=step,source=source,protocol=PROTOCOL,checked_unix=time.time(),
               training_actions=0,permission_changes=0,goal_complete=False)
    if not archive.exists():return dict(value,action='WAIT_FOR_ARCHIVE_CLAIM')
    need(not archive.is_symlink(),'archive control is a symlink')
    if not (archive/'submit-claim.json').exists():return dict(value,action='UNKNOWN_ARCHIVE_ADMISSION')
    claim,_=document(archive/'submit-claim.json',(JXU,))
    need(claim['source']==source and claim['step']==step and claim['storage_wait_protocol']==PROTOCOL and
         claim['role']=='detached_cpu_pair_archive' and claim['producer']['run_id']==RUN and
         claim['producer']['producer_job']=='721485' and claim['training_permitted'] is False,
         'inspected archive claim source/scope differs')
    state=ctl.controller_state(archive);value['archive_state']=state
    code=state['actual_task_exit_code']
    if code is not None:
        need(type(code) is int,'archive actual wait is not an integer')
        return dict(value,action='ARCHIVE_COMPLETED' if code==0 and
            state['status']=='ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0' else 'ARCHIVE_FAILED')
    active=archive/'owner-storage-active.json'
    if not active.exists():return dict(value,action='WAIT_FOR_STORAGE_REQUEST')
    pointer,_=document(active,(JXU,),protected=True)
    path=Path(pointer['request_path']);request,proof=request_guard(path,commit,active=False)
    need(pointer['request_sha256']==proof['sha256'] and pointer['attempt']==request['attempt'] and
         pointer['archive_child']==request['archive_child'],'inspected active request identity differs')
    value.update(request=proof,attempt=request['attempt'])
    directory=path.parent/'mou-control'
    if directory.exists():
        need(not directory.is_symlink(),'owner control is a symlink')
        if not (directory/'claim.json').exists():return dict(value,action='UNKNOWN_OWNER_ADMISSION')
        owner,_=document(directory/'claim.json',(MOU,),protected=True)
        need(owner['source']==source and owner['request']==proof and owner['role']=='read_only_MOU_partition_du' and
             owner['training_actions']==owner['permission_changes']==0,'inspected owner claim differs')
        authorization_guard(path,request,proof,claim=owner)
        if not (directory/'terminal-confirmation.json').exists():return dict(value,action='CONFIRM_OWNER')
        actual=consume_owner(path,request,proof,current=False,require_usage=False)
        need(actual is not None,'owner confirmation without original held evidence')
        return dict(value,action='OWNER_CONFIRMED',owner_status=actual['status'])
    if (path.parent/'request-closed.json').exists():return dict(value,action='WAIT_FOR_NEXT_REQUEST')
    need(state.get('task_alive') is True and state.get('actual_task')==request['archive_child'],
         'own archive child not live; original outcome unknown')
    if not request['roots']:return dict(value,action='WAIT_FOR_JXU_ONLY_SCAN')
    if not (path.parent/'authorization-active.json').exists():return dict(value,action='AUTHORIZE_OWNER')
    try:authorization_guard(path,request,proof)
    except ValueError as error:
        if str(error).endswith('owner authorization was expired at first claim'):
            return dict(value,action='AUTHORIZE_OWNER')
        raise
    return dict(value,action='START_OWNER')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('authorize-owner','owner-start','owner-supervise','owner-worker','owner-confirm','inspect-storage'))
    parser.add_argument('--request',type=Path);parser.add_argument('--step',type=int,choices=(150,200));parser.add_argument('--commit',required=True);parser.add_argument('--token')
    args=parser.parse_args()
    if args.mode=='inspect-storage':
        need(args.step is not None and args.request is None and args.token is None,'explicit read-only step required')
        print(json.dumps(inspect_storage(args.step,args.commit),allow_nan=False),flush=True);sys.exit(0)
    need(args.request is not None and args.step is None,'explicit owner request required')
    if args.mode=='owner-worker':sys.exit(owner_worker(args.request,args.commit,args.token))
    if args.mode=='owner-supervise':sys.exit(owner_supervise(args.request,args.commit,args.token))
    action={'authorize-owner':authorize_owner,'owner-confirm':owner_confirm,'owner-start':owner_start}[args.mode]
    print(json.dumps(action(args.request,args.commit)),flush=True)
