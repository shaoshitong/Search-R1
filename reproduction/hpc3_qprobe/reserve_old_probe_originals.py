"""One non-destructive HPC3 -> HKUST3 reserve of old protected artifacts.

Copies only the fixed stopped-run plan. Credentials arrive on stdin and remain
in process RAM. No actor/model import, training action, source unlink, signals,
or GPU allocation. Existing or unknown target claims are inspect-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import sys
import time
import uuid

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
BASE=ROOT.parent/'qprobe-teacher-warmup200-sigma3-20261015'
TARGET=Path('/backup/shaoshitong/qprobe-old-original-reserve-20261016')
CONTROL=TARGET/'control'
PAYLOAD=TARGET/'payload'
PLAN_NAME='old_original_reserve_plan_20261016.json'
PLAN_SHA='0c9a27ca36509f885ad2c0d0d5ff357a3c4f7055ce485ec5ff607ec9dc6d75f2'
RELATIVE='reproduction/hpc3_qprobe/reserve_old_probe_originals.py'
CAPS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
STATUS='OLD_PROTECTED_ORIGINALS_REMOTE_RESERVE_SOURCE_BEFORE_AFTER_TARGET_FULL_SHA_PASS'


def need(ok,message):
    if not ok:raise ValueError('OLD_ORIGINAL_RESERVE: '+message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def below(path,root):
    try:Path(path).relative_to(root);return True
    except ValueError:return False


def validate_plan(p):
    need(p.get('schema_version')==1 and p.get('old_job')=='719225' and
         p.get('old_run_id')=='c3856315d3f345d7920ebf5725a083b4' and
         p.get('new_run_id')=='70827e85fe584f0c84453aae907918a4' and
         p.get('deletion_authorized') is False and p.get('source_cleanup_performed') is False and
         p.get('training_actions')==0,'fixed non-destructive scope differs')
    need([(g['step'],g['role']) for g in p['groups']]==[(5,'source'),(50,'source'),(100,'archive')],
         'fixed three groups differ')
    paths=[r['path'] for r in p['files']]
    need(len(paths)==118 and len(set(paths))==118 and sum(r['bytes'] for r in p['files'])==121296682925==p['total_bytes'],
         'fixed file set/byte count differs')
    for r in p['files']:
        path=Path(r['path'])
        need(path.is_absolute() and '..' not in path.parts and '\n' not in str(path) and '\0' not in str(path) and
             below(path,ROOT.parent) and not below(path,BASE) and
             type(r['bytes']) is int and r['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',r['sha256']) and
             r['uid'] in (14090,21930),'unsafe/unapproved source member')
    for group in p['groups']:
        need(set(group['members'])<=set(paths),'group has an unplanned member')
    need(str(TARGET)==p['target_root'] and p['source_uid']==14090 and p['target_uid']==1002 and
         p['source_login']=='10.120.48.31' and p['source_account']=='jxu957' and
         p['personal_target_budget_bytes']==10**12,'fixed endpoint/budget differs')
    return p


def plan():
    path=Path(__file__).with_name(PLAN_NAME)
    need(digest(path)==PLAN_SHA,'immutable fixed plan SHA differs')
    return validate_plan(json.loads(path.read_text()))


def caps():
    for k in CAPS:os.environ[k]='1'
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.dont_write_bytecode=True


def identity():
    return dict(hostname=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid_namespace=os.readlink('/proc/self/ns/pid'))


def birth(pid):
    try:
        p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(') ',1)[1].split()
        if fields[0]=='Z':return None
        return dict(pid=int(pid),start_ticks=int(fields[19]),sid=int(fields[3]),uid=p.stat().st_uid)
    except (FileNotFoundError,ProcessLookupError):return None


def live(row):return isinstance(row,dict) and birth(row['pid'])==row


def publish(path,value,replace=False):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    if replace:os.replace(str(temporary),str(path))
    else:
        try:os.link(str(temporary),str(path))
        finally:temporary.unlink()
    fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def read_owned(path,uid):
    path=Path(path);s=path.lstat()
    need(path.resolve()==path and stat.S_ISREG(s.st_mode) and s.st_uid==uid and s.st_size<4*2**20,
         'unsafe control metadata')
    return json.loads(path.read_text())


def code_guard(commit,site):
    p=plan();need(identity()==p[site+'_host'] and os.getuid()==p[site+'_uid'],'actual host/owner differs')
    repo=Path(__file__).resolve().parents[2]
    root=ROOT if site=='source' else TARGET
    need(re.fullmatch('[0-9a-f]{40}',commit) and repo==root/'source'/commit,'immutable code path differs')
    proof_path=root/'preparation'/('source-'+commit+'-verification.json')
    proof=read_owned(proof_path,p[site+'_uid'])
    need(proof['commit']==proof['published_remote_sha']==commit and proof['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS',
         'published source header differs')
    for path in (Path(__file__).resolve(),Path(__file__).with_name(PLAN_NAME)):
        need(digest(path)==proof['files'][str(path.relative_to(repo))],'loaded code/plan bytes differ')
    return p,repo


def row_stat(row):
    path=Path(row['path']);s=path.lstat()
    need(path.resolve()==path and stat.S_ISREG(s.st_mode) and
         (s.st_ino,s.st_dev,s.st_size,s.st_mtime_ns,s.st_uid,s.st_gid,s.st_nlink)==
         tuple(row[k] for k in ('inode','device','bytes','mtime_ns','uid','gid','nlink')),'original file identity changed')
    return s


def source_manifest(commit):
    p,_=code_guard(commit,'source')
    accounting=subprocess.check_output(['sacct','-nP','-X','-j','719225','--format=JobID,State%64,ExitCode,UID'],
                                      text=True,timeout=30)
    rows=[line.split('|') for line in accounting.splitlines() if line.strip()]
    need(len(rows)==1 and rows[0][0]=='719225' and rows[0][1].split()[0]=='CANCELLED' and rows[0][3]=='14090',
         'old human-cancelled job is not currently terminal')
    by_path={r['path']:r for r in p['files']}
    for row in p['files']:row_stat(row)
    for group in p['groups']:
        step=group['step'];directory=ROOT/'archives'/p['old_run_id']/('global_step_'+str(step))
        proof_path=directory/'complete.json';need(digest(proof_path)==by_path[str(proof_path)]['sha256'],
                                                'original transport metadata changed')
        transport=read_owned(proof_path,14090)
        need(transport['run_id']==p['old_run_id'] and transport['step']==step,'old pair run/step differs')
        receipt=Path(transport['source_receipt']['path'])
        need(digest(receipt)==transport['source_receipt']['sha256']==by_path[str(receipt)]['sha256'],
             'original source receipt changed')
        expected={r[group['role']+'_path']:(r['bytes'],r['sha256']) for r in transport['files']}
        if group['role']=='archive':
            q=transport['archive_receipt'];expected[q['path']]=(by_path[q['path']]['bytes'],q['sha256'])
        for path,(size,sha) in expected.items():
            need(path in group['members'] and (by_path[path]['bytes'],by_path[path]['sha256'])==(size,sha),
                 'backup loses a recorded payload member')
        expected.update({str(proof_path):(by_path[str(proof_path)]['bytes'],by_path[str(proof_path)]['sha256']),
                         str(receipt):(by_path[str(receipt)]['bytes'],by_path[str(receipt)]['sha256'])})
        need(set(expected)==set(group['members']),'backup group has omitted/extra metadata')
        tree=Path(group['checkpoint'])
        actual=set()
        for path in tree.rglob('*'):
            s=path.lstat();need(stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode),'link/device in old tree')
            if stat.S_ISREG(s.st_mode):actual.add(str(path))
        need(actual=={q for q in group['members'] if below(Path(q),tree)},'old protected tree file set changed')
    value=dict(schema_version=1,source_commit=commit,plan_sha256=PLAN_SHA,old_run_id=p['old_run_id'],
               groups=p['groups'],files=p['files'],total_bytes=p['total_bytes'],source_host=p['source_host'],
               source_uid=14090,source_cleanup_performed=False,deletion_authorized=False)
    return value,accounting


def source_verify(commit,token,phase,expected_manifest):
    need(re.fullmatch('[0-9a-f]{32}',token) and phase in ('before','after'),'invalid fixed reserve phase')
    manifest,accounting=source_manifest(commit)
    need(hashlib.sha256(canonical(manifest)).hexdigest()==expected_manifest,'source manifest differs')
    root=ROOT/'observations/old-original-reserve-20261016'/token
    root.mkdir(parents=True,exist_ok=True)
    directory=root/phase
    need(not directory.exists(),'source phase already attempted; inspect, never replay')
    directory.mkdir(mode=0o700)
    process=birth(os.getpid());publish(directory/'process.json',process)
    verified=[];count=0
    with (directory/'progress.jsonl').open('x') as progress:
        for row in manifest['files']:
            row_stat(row);sha=digest(row['path']);row_stat(row)
            need(sha==row['sha256'],'actual original full SHA differs')
            count+=row['bytes'];verified.append(dict(path=row['path'],bytes=row['bytes'],sha256=sha))
            progress.write(json.dumps(dict(files=len(verified),bytes=count,epoch=time.time()))+'\n');progress.flush()
    after,_=source_manifest(commit)
    need(after==manifest,'source manifest changed during full read')
    result=dict(status='ORIGINAL_SOURCE_FULL_SHA_PASS',phase=phase,token=token,actual_process=process,
        source_commit=commit,manifest_sha256=expected_manifest,verified_files=len(verified),verified_bytes=count,
        files=verified,actual_old_cancelled_accounting=accounting,completed_unix=time.time(),
        source_cleanup_performed=False,actor_model_loaded=False,training_performed=False)
    publish(directory/'result.json',result);return result


def remote_command(commit,mode,token=None,phase=None,manifest_sha=None):
    source=ROOT/'source'/commit/RELATIVE
    args=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1']+[k+'=1' for k in CAPS]
    args += [str(BASE/'env/bin/python'),str(source),mode,'--commit',commit]
    if token is not None:args+=['--token',token]
    if phase is not None:args+=['--phase',phase]
    if manifest_sha is not None:args+=['--manifest-sha',manifest_sha]
    return ' '.join(shlex.quote(s) for s in args)


def ssh_args(p):
    need(digest(p['known_hosts_path'])==p['known_hosts_sha256'],'existing known-hosts bytes changed')
    return ['/usr/bin/sshpass','-e','/usr/bin/ssh','-T','-oStrictHostKeyChecking=yes',
        '-oControlMaster=no','-S','none','-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn',
        '-oUserKnownHostsFile='+p['known_hosts_path'],'-oConnectTimeout=8']


def child_wait(name,args,password,directory=None):
    directory=CONTROL if directory is None else Path(directory)
    env=dict(os.environ,SSHPASS=password)
    with (directory/(name+'.stdout')).open('xb') as stdout,(directory/(name+'.stderr')).open('xb') as stderr:
        child=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env)
        row=birth(child.pid);need(row is not None and row['uid']==1002,'actual backup child owner differs')
        publish(directory/(name+'.child.json'),dict(actual_process=row,command=args,started_unix=time.time()))
        publish(directory/'state.json',dict(phase=name,actual_child=row,epoch=time.time()),replace=True)
        code=child.wait()
    publish(directory/(name+'.wait.json'),dict(actual_process=row,actual_exit_code=code,completed_unix=time.time()))
    need(code==0,'actual backup child failed: '+name)
    return (directory/(name+'.stdout')).read_bytes()


def source_call(p,commit,mode,password,token,phase=None,manifest_sha=None,directory=None):
    args=ssh_args(p)+[p['source_account']+'@'+p['source_login'],remote_command(commit,mode,token,phase,manifest_sha)]
    return json.loads(child_wait('source-'+(phase or mode),args,password,directory))


def filelist_at(path,files):
    with Path(path).open('xb') as f:
        f.write(b''.join(r['path'].lstrip('/').encode()+b'\0' for r in files));f.flush();os.fsync(f.fileno())


def copy_command(p,commit,token,manifest_sha,filelist,destination,dry_run=False):
    rsync_path=remote_command(commit,'rsync-sender',token,manifest_sha=manifest_sha)+' --'
    rsh=' '.join(shlex.quote(s) for s in ssh_args(p))
    return ['/usr/bin/rsync','--recursive','--relative','--times','--no-links','--protect-args',
            '--from0','--files-from='+str(filelist),'--partial-dir=.rsync-partial','--info=progress2','--stats',
            '--outbuf=L']+(['--dry-run'] if dry_run else [])+['-e',rsh,'--rsync-path='+rsync_path,
            p['source_account']+'@'+p['source_login']+':/',str(destination)+'/']


def preflight(commit,password):
    p,_=code_guard(commit,'target')
    need(password and not CONTROL.exists() and not PAYLOAD.exists(),'missing RAM credential or existing main writer')
    directory=TARGET/'preparation'/('preflight-'+commit)
    need(not directory.exists(),'preflight already attempted; inspect its output')
    directory.mkdir(mode=0o700)
    token=uuid.uuid4().hex
    manifest=source_call(p,commit,'source-manifest',password,token,directory=directory)
    need(manifest['files']==p['files'] and manifest['total_bytes']==p['total_bytes'],'actual source preflight differs')
    manifest_sha=hashlib.sha256(canonical(manifest)).hexdigest()
    filelist=directory/'files.from0';filelist_at(filelist,manifest['files'])
    command=copy_command(p,commit,token,manifest_sha,filelist,directory/'payload-check',True)
    output=child_wait('rsync-dry-run',command,password,directory).decode()
    fs=os.statvfs(str(TARGET));need(fs.f_bavail*fs.f_frsize>2*p['total_bytes']+2**30,'target capacity insufficient')
    value=dict(status='ACTUAL_REMOTE_AUTHENTICATED_READONLY_RSYNC_PREFLIGHT_PASS',source_commit=commit,
        manifest_sha256=manifest_sha,files=len(p['files']),bytes=p['total_bytes'],actual_rsync_dry_run=True,
        actual_free_bytes=fs.f_bavail*fs.f_frsize,copy_performed=False,source_cleanup_performed=False,
        completed_unix=time.time(),output_sha256=hashlib.sha256(output.encode()).hexdigest())
    publish(directory/'result.json',value);return value


def start(commit,password):
    p,repo=code_guard(commit,'target')
    need(password and '\n' not in password and not CONTROL.exists() and not PAYLOAD.exists(),
         'missing RAM credential or existing claim/payload; inspect only')
    fs=os.statvfs(str(TARGET));need(fs.f_bavail*fs.f_frsize>2*p['total_bytes']+2**30 and p['total_bytes']<10**12,
                                'actual target free space/budget insufficient')
    admission=read_owned(TARGET/'preparation'/('preflight-'+commit)/'result.json',1002)
    need(admission['status']=='ACTUAL_REMOTE_AUTHENTICATED_READONLY_RSYNC_PREFLIGHT_PASS' and
         admission['source_commit']==commit and admission['bytes']==p['total_bytes'],
         'actual readonly protocol preflight required before the one writer')
    CONTROL.mkdir(mode=0o700);token=uuid.uuid4().hex
    command=[sys.executable,str(repo/RELATIVE),'supervise','--commit',commit,'--token',token]
    publish(CONTROL/'claim.json',dict(source_commit=commit,plan_sha256=PLAN_SHA,token=token,command=command,
        target_host=identity(),target_uid=os.getuid(),free_bytes=fs.f_bavail*fs.f_frsize,created_unix=time.time(),
        deletion_authorized=False,source_cleanup_performed=False))
    with (CONTROL/'supervisor.log').open('xb') as log:
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        actual=birth(child.pid);need(actual and actual['sid']==actual['pid'] and actual['uid']==1002,'backup worker birth differs')
        publish(CONTROL/'process.json',dict(actual_process=actual,source_commit=commit,token=token))
        child.stdin.write((password+'\n').encode());child.stdin.close()
    return dict(status='REMOTE_RESERVE_STARTED_ONCE',actual_process=actual,token=token,source_commit=commit,
                target=str(TARGET),source_cleanup_performed=False)


def worker(commit,token,password):
    p,repo=code_guard(commit,'target');claim=read_owned(CONTROL/'claim.json',1002)
    binding=read_owned(CONTROL/'worker-process.json',1002)
    need(claim['token']==binding['token']==token and claim['source_commit']==binding['source_commit']==commit and
         binding['actual_process']==birth(os.getpid()),'actual once worker differs')
    started=time.time()
    try:
        manifest=source_call(p,commit,'source-manifest',password,token)
        need(manifest['files']==p['files'] and manifest['total_bytes']==p['total_bytes'] and
             manifest['source_commit']==commit and manifest['plan_sha256']==PLAN_SHA,'actual source manifest differs')
        manifest_sha=hashlib.sha256(canonical(manifest)).hexdigest();publish(CONTROL/'source-manifest.json',manifest)
        before=source_call(p,commit,'source-verify',password,token,'before',manifest_sha)
        expected=[dict(path=r['path'],bytes=r['bytes'],sha256=r['sha256']) for r in manifest['files']]
        need(before['status']=='ORIGINAL_SOURCE_FULL_SHA_PASS' and before['files']==expected and
             before['manifest_sha256']==manifest_sha and before['phase']=='before','actual source pre-copy SHA differs')
        PAYLOAD.mkdir(mode=0o700)
        filelist=CONTROL/'files.from0'
        filelist_at(filelist,manifest['files'])
        command=copy_command(p,commit,token,manifest_sha,filelist,PAYLOAD)
        child_wait('rsync-copy',command,password)
        after=source_call(p,commit,'source-verify',password,token,'after',manifest_sha)
        need(after['status']=='ORIGINAL_SOURCE_FULL_SHA_PASS' and after['files']==expected and
             after['manifest_sha256']==manifest_sha and after['phase']=='after','actual source post-copy SHA differs')
        verified=[]
        publish(CONTROL/'state.json',dict(phase='target-fresh-full-sha',epoch=time.time()),replace=True)
        for row in manifest['files']:
            target=PAYLOAD/row['path'].lstrip('/');s=target.lstat()
            need(target.resolve()==target and stat.S_ISREG(s.st_mode) and s.st_uid==1002 and
                 s.st_size==row['bytes'],'actual target file identity differs')
            sha=digest(target);after_stat=target.lstat()
            need((s.st_ino,s.st_size,s.st_mtime_ns)==(after_stat.st_ino,after_stat.st_size,after_stat.st_mtime_ns) and
                 sha==row['sha256'],'actual target full SHA differs')
            verified.append(dict(source_path=row['path'],target_path=str(target),bytes=row['bytes'],sha256=sha,
                                 inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns))
            publish(CONTROL/'target-progress.json',dict(files=len(verified),bytes=sum(r['bytes'] for r in verified),
                                                      epoch=time.time()),replace=True)
        actual={str(q) for q in PAYLOAD.rglob('*') if q.is_file()}
        need(actual=={v['target_path'] for v in verified},'actual target fileset has missing/extra members')
        value=dict(status=STATUS,source_commit=commit,plan_sha256=PLAN_SHA,manifest_sha256=manifest_sha,
            files=verified,verified_files=len(verified),verified_bytes=sum(r['bytes'] for r in verified),
            source_before=before,source_after=after,started_unix=started,completed_unix=time.time(),
            actual_process=birth(os.getpid()),actor_model_loaded=False,training_performed=False,
            source_cleanup_performed=False,deletion_authorized=False)
        publish(CONTROL/'complete.json',value)
        publish(CONTROL/'worker-result.json',dict(status=STATUS,exit_code=0,complete_sha256=digest(CONTROL/'complete.json'),
            actual_process=birth(os.getpid()),completed_unix=time.time(),source_cleanup_performed=False))
        return 0
    except Exception as exc:
        publish(CONTROL/'worker-result.json',dict(status='REMOTE_RESERVE_FAILED_RETAIN_ALL_PARTIALS',exit_code=1,
            error=repr(exc),actual_process=birth(os.getpid()),completed_unix=time.time(),source_cleanup_performed=False))
        raise


def supervise(commit,token,password):
    p,repo=code_guard(commit,'target');claim=read_owned(CONTROL/'claim.json',1002)
    parent=read_owned(CONTROL/'process.json',1002)
    need(claim['token']==parent['token']==token and parent['actual_process']==birth(os.getpid()) and
         claim['source_commit']==parent['source_commit']==commit,'actual supervisor differs')
    command=[sys.executable,str(repo/RELATIVE),'worker','--commit',commit,'--token',token]
    with (CONTROL/'worker.log').open('xb') as log:
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
        actual=birth(child.pid);need(actual and actual['uid']==1002 and actual['sid']==os.getsid(0),'held reserve worker birth differs')
        publish(CONTROL/'worker-process.json',dict(actual_process=actual,source_commit=commit,token=token))
        child.stdin.write((password+'\n').encode());child.stdin.close()
        code=child.wait()
    result=dict(status='ACTUAL_HELD_REMOTE_RESERVE_WORKER_FAILED',actual_child_exit_code=code,
        exit_code=1,actual_supervisor=birth(os.getpid()),actual_worker=actual,completed_unix=time.time(),
        source_cleanup_performed=False)
    try:
        if code==0:
            returned=read_owned(CONTROL/'worker-result.json',1002)
            need(returned['actual_process']==actual and returned['exit_code']==0 and returned['status']==STATUS and
                 returned['source_cleanup_performed'] is False and digest(CONTROL/'complete.json')==returned['complete_sha256'],
                 'actual held worker result/bytes differ')
            result.update(status='ACTUAL_HELD_REMOTE_RESERVE_WORKER_COMPLETED0',exit_code=0,
                          complete_sha256=returned['complete_sha256'])
    except Exception as exc:
        result['validation_error']=repr(exc)
    publish(CONTROL/'result.json',result)
    return result['exit_code']


def inspect(commit):
    p,_=code_guard(commit,'target');binding=read_owned(CONTROL/'process.json',1002)
    claim=read_owned(CONTROL/'claim.json',1002)
    need(binding['source_commit']==claim['source_commit']==commit and binding['token']==claim['token'],
         'reserve original claim/process differs')
    files={};allocated=0
    if PAYLOAD.exists():
        for q in PAYLOAD.rglob('*'):
            try:s=q.lstat()
            except FileNotFoundError:continue
            need(stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode),'unsafe target progress member')
            if stat.S_ISREG(s.st_mode):
                need(s.st_uid==1002,'foreign target progress file');files[(s.st_dev,s.st_ino)]=s
        allocated=sum(s.st_blocks*512 for s in files.values())
    result=read_owned(CONTROL/'result.json',1002) if (CONTROL/'result.json').exists() else None
    state=read_owned(CONTROL/'state.json',1002) if (CONTROL/'state.json').exists() else None
    out=dict(epoch=time.time(),actual_live_birth=live(binding['actual_process']),process=binding,
        state=state,result=result,allocated_payload_bytes=allocated,payload_regular_files=len(files),
        planned_bytes=p['total_bytes'],source_cleanup_performed=False)
    worker_file=CONTROL/'worker-process.json'
    if worker_file.exists():
        child=read_owned(worker_file,1002);out['worker']=dict(process=child['actual_process'],live=live(child['actual_process']))
    for name in ('source-before','rsync-copy','source-after'):
        path=CONTROL/(name+'.child.json')
        if path.exists():
            child=read_owned(path,1002);out[name]=dict(process=child['actual_process'],live=live(child['actual_process']))
    for name in ('supervisor.log','worker.log','rsync-copy.stdout','rsync-copy.stderr'):
        path=CONTROL/name
        if path.exists():
            with path.open('rb') as f:f.seek(max(0,path.stat().st_size-4096));tail=f.read().decode(errors='replace')
            out[name]=dict(bytes=path.stat().st_size,tail=tail)
    if result and result['exit_code']==0:
        need(result['status']=='ACTUAL_HELD_REMOTE_RESERVE_WORKER_COMPLETED0' and
             result['actual_child_exit_code']==0 and result['actual_supervisor']==binding['actual_process'] and
             result['actual_worker']==out.get('worker',{}).get('process') and
             digest(CONTROL/'complete.json')==result['complete_sha256'],'held complete result or bytes differ')
        complete=read_owned(CONTROL/'complete.json',1002)
        need(complete['status']==STATUS and complete['verified_bytes']==p['total_bytes'] and
             complete['verified_files']==len(p['files']) and complete['source_commit']==commit and
             complete['plan_sha256']==PLAN_SHA and complete['actual_process']==result['actual_worker'] and
             complete['source_cleanup_performed'] is False and complete['deletion_authorized'] is False,
             'terminal reserve scope incomplete')
        out['complete_sha256']=result['complete_sha256']
        out['actual_terminal_accepted']=not out['actual_live_birth'] and not out.get('worker',{}).get('live',True)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('source-manifest','source-verify','rsync-sender','preflight','start','supervise','worker','inspect'))
    parser.add_argument('--commit',required=True);parser.add_argument('--token');parser.add_argument('--phase');parser.add_argument('--manifest-sha')
    a,remainder=parser.parse_known_args();caps();os.umask(0o077)
    if a.mode=='source-manifest':value=source_manifest(a.commit)[0]
    elif a.mode=='source-verify':value=source_verify(a.commit,a.token,a.phase,a.manifest_sha)
    elif a.mode=='rsync-sender':
        manifest,_=source_manifest(a.commit)
        need(hashlib.sha256(canonical(manifest)).hexdigest()==a.manifest_sha and
             remainder and remainder[0]=='--' and '--server' in remainder and '--sender' in remainder and
             not any(x.startswith(('--delete','--remove-source')) for x in remainder),'only read-only rsync sender allowed')
        os.execv('/usr/bin/rsync',['rsync']+remainder[1:]);return
    elif a.mode=='inspect':value=inspect(a.commit)
    else:
        need(not remainder,'unexpected reserve arguments');password=sys.stdin.readline().rstrip('\n')
        if a.mode=='worker':sys.exit(worker(a.commit,a.token,password))
        if a.mode=='supervise':sys.exit(supervise(a.commit,a.token,password))
        value=preflight(a.commit,password) if a.mode=='preflight' else start(a.commit,password)
    print(json.dumps(value,allow_nan=False),flush=True)


if __name__=='__main__':main()
