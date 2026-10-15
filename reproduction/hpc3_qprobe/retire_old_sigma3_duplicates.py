"""Explicitly authorized retirement of exactly five stopped-run duplicate trees.

No training control, model load or job submission. The immutable proposed plan
is fixed by SHA. Every original source file, deleted duplicate, and retained
archive100 is fully hashed before the first unlink. Only exact file names are
unlinked through directory FDs; no glob, recursive delete or scope expansion.
A detached CPU supervisor retains the actual worker until wait returns.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

from reproduction.hpc3_qprobe import final_control as ctl
from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as reader

ROOT=ctl.ROOT
OLD_RUN='c3856315d3f345d7920ebf5725a083b4'
OLD_JOB='719225'
OLD_SOURCE='1843babdd96b387dd10d91e3207c9ea9489b6941'
ARCHIVES=ROOT/'archives'/OLD_RUN
DIRECTORY=ROOT/'observations/retire-five-old-sigma3-duplicates-20261015'
MODULE='reproduction.hpc3_qprobe.retire_old_sigma3_duplicates'
PAIRS=((5,'target'),(50,'target'),(100,'target'),(5,'archive'),(50,'archive'))
TOTAL_BYTES=204223139168
PLAN_NAME='old_duplicate_retirement_plan_sigma3_20261015.json'
PLAN_SHA='a7847c7e6eddb9964ce6ef3877c824f0cffbb8c7c3eccdfb92c4199d548eea98'
HUMAN_NAME='old_duplicate_retirement_human_stop719225.json'
HUMAN_SHA='3b6fd925af6a8acf7e128257d92d04ecc877bd2c256e7bdb9d2098d9b7ad63e0'
STATUS='ACTUAL_EXACT_FIVE_OLD_DUPLICATES_RETIRED_RETAINED_SOURCE_AND_ARCHIVE100_FULL_SHA_PASS'
HELD_STATUS='ACTUAL_HELD_EXACT_FIVE_DUPLICATE_RETIREMENT_CHILD_COMPLETED0'


def need(ok,message):ctl.need(ok,'EXACT_FIVE_DUPLICATE_RETIREMENT: '+message)


def caps():
    ctl.owner()
    for name in ctl.CAPS:os.environ[name]='1'
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.dont_write_bytecode=True


def birth_ok(value):return reader.birth_ok(value)


def baseline():
    folder=Path(__file__).resolve().parent
    payload,proof=reader.read_file(folder/PLAN_NAME,expected_sha=PLAN_SHA,capture=True)
    wrapper=json.loads(payload);need(wrapper['returncode']==0 and wrapper['stderr']=='','original plan command did not succeed')
    value=json.loads(wrapper['stdout'])
    need(value['scope']=='read-only proposed old duplicate retirement; NO deletion' and
         value['total_bytes']==TOTAL_BYTES and [(r['step'],r['role']) for r in value['directories']]==list(PAIRS) and
         sum(r['bytes'] for r in value['directories'])==TOTAL_BYTES,'authorized plan scope differs')
    for row in value['directories']:
        directory=ARCHIVES/('global_step_'+str(row['step']))/row['role']
        need(row['path']==str(directory) and row['files']==len(row['file_snapshot'])==len(row['expected_content_hashes'])+1 and
             sum(f['bytes'] for f in row['file_snapshot'])==row['bytes'] and
             len({f['path'] for f in row['file_snapshot']})==row['files'],'authorized paths/fileset/count differs')
    return value,proof


def stopped():
    folder=Path(__file__).resolve().parent
    raw,proof=reader.read_file(folder/HUMAN_NAME,expected_sha=HUMAN_SHA,capture=True)
    wrapper=json.loads(raw);value=json.loads(wrapper['stdout'])
    need(wrapper['exit']==0 and wrapper['stderr']=='' and value['old_job']==OLD_JOB and
         value['old_run_id']==OLD_RUN and value['old_source_commit']==OLD_SOURCE and
         value['human_cancel'] is True and value['autonomous_recovery_permitted'] is False and
         value['cancel_returncode']==0,'original explicit human cancellation proof differs')
    actual=subprocess.check_output(['sacct','-nP','-X','-j',OLD_JOB,
        '--format=JobID,State%64,ExitCode,User,UID,Account,QOS,Partition,NodeList'],text=True,timeout=30)
    rows=[r.split('|') for r in actual.splitlines() if r.strip()]
    need(len(rows)==1 and rows[0][0]==OLD_JOB and rows[0][1].split()[0]=='CANCELLED' and
         rows[0][3:]==['jxu957','14090','jxu957','formal-user','emergency_acd','ACD1-56'],
         'old719225 must actually remain human CANCELLED; healthy/unknown/other failure refuses')
    return dict(original_human_stop=proof,actual_cancelled_accounting=actual)


def safe_directory(path):
    path=Path(path);s=path.lstat()
    need(path.is_absolute() and path.resolve()==path and not path.is_symlink() and stat.S_ISDIR(s.st_mode) and
         s.st_uid==ctl.OWNER,'unsafe directory owner/link: '+str(path))
    return dict(device=s.st_dev,inode=s.st_ino,uid=s.st_uid,gid=s.st_gid,mode=stat.S_IMODE(s.st_mode))


def tree(path):
    """Walk only the exact approved tree; never follow links or devices."""
    path=Path(path);directories={'':safe_directory(path)};files={}
    todo=[path]
    while todo:
        folder=todo.pop()
        with os.scandir(folder) as scan:entries=sorted(scan,key=lambda x:x.name)
        for entry in entries:
            child=folder/entry.name;s=child.lstat()
            need(child.resolve()==child and not stat.S_ISLNK(s.st_mode) and s.st_uid==ctl.OWNER,
                 'unsafe tree member owner/link: '+str(child))
            relative=str(child.relative_to(path))
            if stat.S_ISDIR(s.st_mode):directories[relative]=safe_directory(child);todo.append(child)
            else:
                need(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'unsafe duplicate file type/link count')
                files[str(child)]=dict(path=str(child),bytes=s.st_size,inode=s.st_ino,mtime_ns=s.st_mtime_ns,
                    device=s.st_dev,uid=s.st_uid,gid=s.st_gid,mode=stat.S_IMODE(s.st_mode))
    return dict(files=files,directories=directories)


def progress(stage,**fields):print(json.dumps(dict(stage=stage,epoch=time.time(),**fields)),flush=True)


def hash_bound(path,sha,size,*,shared=False):
    _,proof=reader.read_file(path,expected_sha=sha,expected_bytes=size,shared=shared)
    proof['shared_asset_allowed']=shared
    progress('actual_full_sha_pass',path=str(path),bytes=size)
    return proof


def preflight(plan,plan_proof):
    cancellation=stopped();transports={};retained={};targets=[]
    # Complete original transport records remain intact, explicitly retired
    # only by the new receipt. The old three-copy proofs are never overwritten.
    for step in (5,50,100):
        path=ARCHIVES/('global_step_'+str(step))/'complete.json'
        value,proof=reader.document(path)
        need(value['status']=='QPROBE_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS' and
             value['run_id']==OLD_RUN and value['step']==step and value['world_size']==4,
             'original transport identity differs')
        row=next(r for r in plan['directories'] if r['step']==step)
        need(value['files']==row['expected_content_hashes'] and value['source_receipt']==row['remaining_original_receipt'],
             'original transport/source receipt differs from authorized plan')
        source,source_proof=reader.document(Path(value['source_receipt']['path']))
        need(source_proof['sha256']==value['source_receipt']['sha256'] and source['run_id']==OLD_RUN and source['step']==step,
             'remaining original pair receipt changed')
        retained[source_proof['path']]=source_proof;retained[proof['path']]=proof
        transports[step]=value
        for f in value['files']:
            need(not any(Path(f['source_path']).is_relative_to(ARCHIVES/('global_step_'+str(s))/role) for s,role in PAIRS),
                 'retained source aliases a deleted duplicate')
            key=f['source_path']
            if key in retained:
                need(retained[key]['sha256']==f['sha256'] and retained[key]['bytes']==f['bytes'],'conflicting source records')
            else:retained[key]=hash_bound(Path(key),f['sha256'],f['bytes'],shared=True)
    # Retain the independent archive100 in addition to ALL original sources.
    keep=ARCHIVES/'global_step_100/archive';keep_tree=tree(keep);transport=transports[100]
    expected={f['archive_path']:(f['sha256'],f['bytes']) for f in transport['files']}
    expected[transport['archive_receipt']['path']]=(transport['archive_receipt']['sha256'],keep_tree['files'][transport['archive_receipt']['path']]['bytes'])
    need(set(keep_tree['files'])==set(transport['archive_file_set'])==set(expected),'retained archive100 complete fileset differs')
    for path,(sha,size) in expected.items():retained[path]=hash_bound(Path(path),sha,size)
    for row in plan['directories']:
        directory=Path(row['path']);snapshot=tree(directory);transport=transports[row['step']]
        observed=snapshot['files'];authorized={f['path']:f for f in row['file_snapshot']}
        need(set(observed)==set(authorized)==set(transport[row['role']+'_file_set']),
             'target full fileset differs; no deletion')
        expected={f[row['role']+'_path']:(f['sha256'],f['bytes']) for f in row['expected_content_hashes']}
        receipt=transport[row['role']+'_receipt']
        expected[receipt['path']]=(receipt['sha256'],authorized[receipt['path']]['bytes'])
        need(set(expected)==set(observed),'target content SHA records incomplete')
        for path,record in observed.items():
            need(all(record[k]==authorized[path][k] for k in ('path','bytes','inode','mtime_ns')),
                 'target original snapshot identity changed')
            sha,size=expected[path];record['sha256']=hash_bound(Path(path),sha,size)['sha256']
        targets.append(dict(path=str(directory),files=observed,directories=snapshot['directories'],bytes=row['bytes']))
    # Every original byte and target has passed before any destructive action.
    for record in retained.values():reader.read_file(record['path'],expected_sha=record['sha256'],expected_bytes=record['bytes'],shared=record.get('shared_asset_allowed',False))
    reader.read_file(plan_proof['path'],expected_sha=plan_proof['sha256'],expected_bytes=plan_proof['bytes'])
    stopped()
    return dict(plan=plan_proof,cancellation=cancellation,targets=targets,retained_files=list(retained.values()),
        total_apparent_bytes=TOTAL_BYTES,all_guards_before_first_unlink=True,created_unix=time.time())


def open_relative(rootfd,parts,directories):
    fd=os.dup(rootfd)
    try:
        accumulated=[]
        for name in parts:
            accumulated.append(name);child=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=child;s=os.fstat(fd);expected=directories['/'.join(accumulated)]
            need((s.st_dev,s.st_ino,s.st_uid)==(expected['device'],expected['inode'],ctl.OWNER),'directory identity changed before unlink')
        return fd
    except BaseException:
        os.close(fd);raise


def exact_delete(target):
    directory=Path(target['path']);snapshot=tree(directory)
    need(snapshot['directories']==target['directories'] and set(snapshot['files'])==set(target['files']),
         'duplicate tree changed after all guards')
    rootfd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        s=os.fstat(rootfd);expected=target['directories']['']
        need((s.st_dev,s.st_ino,s.st_uid)==(expected['device'],expected['inode'],ctl.OWNER),'duplicate root changed before delete')
        for path,record in sorted(target['files'].items()):
            relative=Path(path).relative_to(directory);fd=open_relative(rootfd,relative.parts[:-1],target['directories'])
            try:
                # The same parent FD supplies open, fstatat and unlink. Equal
                # NFS metadata is insufficient: rehash the complete held FD.
                filefd=os.open(relative.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
                with os.fdopen(filefd,'rb') as stream:
                    before=os.fstat(stream.fileno());h=hashlib.sha256()
                    need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_uid==ctl.OWNER and
                         (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
                         (record['device'],record['inode'],record['bytes'],record['mtime_ns']),'file identity changed before unlink')
                    for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
                    after=os.fstat(stream.fileno());current=os.stat(relative.name,dir_fd=fd,follow_symlinks=False)
                    need(reader.signature(before)==reader.signature(after)==reader.signature(current) and
                         h.hexdigest()==record['sha256'],'duplicate content changed before unlink')
                    os.unlink(relative.name,dir_fd=fd);os.fsync(fd)
                progress('actual_exact_file_unlinked',path=path,bytes=record['bytes'])
            finally:os.close(fd)
        for relative in sorted((k for k in target['directories'] if k),key=lambda x:(-len(Path(x).parts),x)):
            p=Path(relative);fd=open_relative(rootfd,p.parts[:-1],target['directories'])
            try:
                s=os.stat(p.name,dir_fd=fd,follow_symlinks=False);expected=target['directories'][relative]
                need((s.st_dev,s.st_ino,s.st_uid)==(expected['device'],expected['inode'],ctl.OWNER),'directory changed before rmdir')
                os.rmdir(p.name,dir_fd=fd);os.fsync(fd)
            finally:os.close(fd)
    finally:os.close(rootfd)
    parentfd=os.open(directory.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        s=os.stat(directory.name,dir_fd=parentfd,follow_symlinks=False);expected=target['directories']['']
        need((s.st_dev,s.st_ino,s.st_uid)==(expected['device'],expected['inode'],ctl.OWNER),'duplicate root replaced before rmdir')
        os.rmdir(directory.name,dir_fd=parentfd);os.fsync(parentfd)
    finally:os.close(parentfd)


def worker(commit):
    caps();repository,source=ctl.source(commit);own=ctl.birth(ctl.process(os.getpid()));claim=ctl.read(DIRECTORY/'claim.json')
    need(Path(__file__).resolve()==repository/(MODULE.replace('.','/')+'.py') and claim['source']==source and
         claim['exact_five_destructive_scope_authorized'] is True,'actual worker immutable source/authorization differs')
    deadline=time.monotonic()+15
    while not (DIRECTORY/'child.json').exists():
        need(time.monotonic()<deadline,'actual child birth binding missing; unknown no replay');time.sleep(.05)
    need(ctl.read(DIRECTORY/'child.json')==own,'only actually held worker may delete')
    plan,proof=baseline();need(claim['plan']==proof,'authorized plan bytes changed')
    guards=preflight(plan,proof);ctl.publish(DIRECTORY/'preflight.json',guards)
    for target in guards['targets']:exact_delete(target)
    for record in guards['retained_files']:
        reader.read_file(record['path'],expected_sha=record['sha256'],expected_bytes=record['bytes'],shared=record.get('shared_asset_allowed',False))
    stopped()
    need(all(not Path(r['path']).exists() and not Path(r['path']).is_symlink() for r in guards['targets']),
         'an authorized duplicate remained or reappeared')
    receipt=dict(status=STATUS,source=source,claim_sha256=ctl.digest(DIRECTORY/'claim.json'),actual_process=own,
        preflight_sha256=ctl.digest(DIRECTORY/'preflight.json'),retired_directories=[r['path'] for r in guards['targets']],
        retired_files=sum(len(r['files']) for r in guards['targets']),retired_apparent_bytes=TOTAL_BYTES,
        actual_allocated_bytes_freed_not_measured=True,retained_original_sources_and_archive100_full_sha_pass=True,
        original_transport_proofs_not_overwritten=True,training_control_actions=0,model_or_gpu_accesses=0,completed_unix=time.time())
    ctl.publish(DIRECTORY/'retired-receipt.json',receipt);print(json.dumps(receipt),flush=True)


def start(commit,*,authorize_exact_five=False):
    caps()
    if DIRECTORY.exists() or DIRECTORY.is_symlink():return inspect(commit)
    need(authorize_exact_five is True,'explicit exact-five authorization required')
    repository,source=ctl.source(commit);_,proof=baseline();stopped()
    need(DIRECTORY.parent.resolve()==DIRECTORY.parent and DIRECTORY.parent.stat().st_uid==ctl.OWNER,'unsafe observations parent')
    try:DIRECTORY.mkdir(mode=0o770)
    except FileExistsError:return inspect(commit)
    command=[sys.executable,'-m',MODULE,'supervise','--commit',commit]
    ctl.publish(DIRECTORY/'claim.json',dict(source=source,plan=proof,command=command,cwd=str(repository),
        exact_five_destructive_scope_authorized=True,training_control_permitted=False,created_unix=time.time()))
    with (DIRECTORY/'supervisor.stdout').open('xb') as out,(DIRECTORY/'supervisor.stderr').open('xb') as err:
        child=subprocess.Popen(command,cwd=repository,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    birth=ctl.birth(ctl.process(child.pid));need(birth_ok(birth) and birth['pid']==birth['sid'],'actual detached supervisor birth differs')
    ctl.publish(DIRECTORY/'process.json',birth)
    return dict(status='ACTUAL_EXACT_FIVE_RETIREMENT_CPU_STARTED_NOT_ACCEPTED',process=birth)


def supervise(commit):
    caps();repository,source=ctl.source(commit);own=ctl.birth(ctl.process(os.getpid()))
    process=None;child=None;child_exit=None;code=1;error=None;started=time.time();receipt_sha=None;preflight_sha=None
    try:
        deadline=time.monotonic()+15
        while not (DIRECTORY/'process.json').exists():
            need(time.monotonic()<deadline,'launcher birth missing; unknown no replay');time.sleep(.05)
        claim=ctl.read(DIRECTORY/'claim.json')
        need(ctl.read(DIRECTORY/'process.json')==own and claim['source']==source and claim['cwd']==str(repository) and
             claim['exact_five_destructive_scope_authorized'] is True,'held supervisor original intent/birth differs')
        with (DIRECTORY/'worker.stdout').open('xb') as out,(DIRECTORY/'worker.stderr').open('xb') as err:
            process=subprocess.Popen([sys.executable,'-m',MODULE,'worker','--commit',commit],cwd=repository,
                stdin=subprocess.DEVNULL,stdout=out,stderr=err)
            child=ctl.birth(ctl.process(process.pid));need(birth_ok(child) and child['sid']==own['sid'] and child['pid']!=own['pid'],'held CPU worker birth differs')
            ctl.publish(DIRECTORY/'child.json',child);child_exit=process.wait()
            for stream in (out,err):stream.flush();os.fsync(stream.fileno())
        need(child_exit==0 and not ctl.live(child) and (DIRECTORY/'worker.stderr').stat().st_size==0,'held worker did not actually return clean0')
        receipt=ctl.read(DIRECTORY/'retired-receipt.json')
        need(receipt['status']==STATUS and receipt['actual_process']==child and receipt['source']==source and
             receipt['claim_sha256']==ctl.digest(DIRECTORY/'claim.json'),'worker returned without authentic retired receipt')
        receipt_sha=ctl.digest(DIRECTORY/'retired-receipt.json');preflight_sha=ctl.digest(DIRECTORY/'preflight.json')
        need(receipt.get('preflight_sha256')==preflight_sha,'worker retired receipt is not the actual complete preflight')
        code=0
    except BaseException as exc:
        error=repr(exc)
        import traceback;traceback.print_exc()
        if process is not None and child_exit is None:child_exit=process.wait()
    outputs={}
    for name in ('stdout','stderr'):
        path=DIRECTORY/('worker.'+name)
        if path.exists():outputs[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=ctl.digest(path))
    ctl.publish(DIRECTORY/'result.json',dict(source=source,claim_sha256=ctl.digest(DIRECTORY/'claim.json'),process=own,child=child,
        actual_child_exit_code=child_exit,supervisor_exit_code=code,error=error,retired_receipt_sha256=receipt_sha,preflight_sha256=preflight_sha,
        started_unix=started,completed_unix=time.time(),**outputs))
    return code


def consume(commit):
    _,source=ctl.source(commit);claim=ctl.read(DIRECTORY/'claim.json');result=ctl.read(DIRECTORY/'result.json')
    process=ctl.read(DIRECTORY/'process.json');child=ctl.read(DIRECTORY/'child.json')
    need(claim['source']==result['source']==source and result['claim_sha256']==ctl.digest(DIRECTORY/'claim.json') and
         claim['exact_five_destructive_scope_authorized'] is True and result['process']==process and result['child']==child and
         birth_ok(process) and birth_ok(child) and child['sid']==process['sid']==process['pid'] and child['pid']!=process['pid'] and
         not ctl.live(process) and not ctl.live(child) and type(result['actual_child_exit_code']) is int and result['actual_child_exit_code']==0 and
         type(result['supervisor_exit_code']) is int and result['supervisor_exit_code']==0 and result['error'] is None and
         result['completed_unix']>=result['started_unix']>=claim['created_unix'],'actual terminal held wait/source/birth differs')
    for name in ('stdout','stderr'):
        row=result[name];path=DIRECTORY/('worker.'+name)
        need(row['path']==str(path) and row['bytes']==path.stat().st_size and row['sha256']==ctl.digest(path),'held worker output bytes changed')
    need(result['stderr']['bytes']==0,'held worker stderr is not empty')
    receipt=ctl.read(DIRECTORY/'retired-receipt.json');guards=ctl.read(DIRECTORY/'preflight.json')
    need(result.get('retired_receipt_sha256')==ctl.digest(DIRECTORY/'retired-receipt.json') and
         result.get('preflight_sha256')==ctl.digest(DIRECTORY/'preflight.json') and claim.get('training_control_permitted') is False and
         claim['plan']['sha256']==PLAN_SHA and guards['plan']==claim['plan'] and
         guards['all_guards_before_first_unlink'] is True and guards['total_apparent_bytes']==TOTAL_BYTES and
         [target['path'] for target in guards['targets']]==[str(ARCHIVES/('global_step_'+str(step))/role) for step,role in PAIRS],
         'held retired receipt/preflight/authorized scope bytes differ')
    need(receipt['status']==STATUS and receipt['source']==source and receipt['actual_process']==child and
         receipt['claim_sha256']==result['claim_sha256'] and receipt['preflight_sha256']==ctl.digest(DIRECTORY/'preflight.json') and
         receipt['retired_directories']==[str(ARCHIVES/('global_step_'+str(step))/role) for step,role in PAIRS] and
         receipt['retired_apparent_bytes']==TOTAL_BYTES and receipt['retired_files']==sum(len(t['files']) for t in guards['targets']) and
         receipt['retained_original_sources_and_archive100_full_sha_pass'] is True and
         receipt['original_transport_proofs_not_overwritten'] is True and receipt['training_control_actions']==0 and receipt['model_or_gpu_accesses']==0 and
         result['completed_unix']>=receipt['completed_unix']>=result['started_unix'] and
         all(not Path(p).exists() and not Path(p).is_symlink() for p in receipt['retired_directories']), 'actual exact-five retired receipt differs')
    # Inspect stays light: full weights were re-read in the held worker after
    # deletion. Fresh hashes here cover the unmodified small transport proofs.
    for record in guards['retained_files']:
        if Path(record['path']).name in ('complete.json','source-receipt.json'):
            reader.read_file(record['path'],expected_sha=record['sha256'],expected_bytes=record['bytes'],shared=record.get('shared_asset_allowed',False))
    return dict(status=HELD_STATUS,result_sha256=ctl.digest(DIRECTORY/'result.json'),
        retired_receipt_sha256=ctl.digest(DIRECTORY/'retired-receipt.json'),actual_child_exit_code=0,
        retired_apparent_bytes=TOTAL_BYTES,actual_allocated_bytes_freed_not_measured=True,
        scope='exact five authorized old duplicate trees; originals and archive100 preserved')


def inspect(commit):
    ctl.owner()
    if not DIRECTORY.exists():return dict(status='NOT_STARTED')
    safe_directory(DIRECTORY);claim=ctl.read(DIRECTORY/'claim.json');need(claim['source']['commit']==commit,'existing once intent source differs')
    process=ctl.read(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else None
    child=ctl.read(DIRECTORY/'child.json') if (DIRECTORY/'child.json').exists() else None
    result=ctl.read(DIRECTORY/'result.json') if (DIRECTORY/'result.json').exists() else None
    alive=bool(process and ctl.live(process))
    if result and result['supervisor_exit_code']==0 and not alive:return consume(commit)
    return dict(status='ACTUAL_CPU_RETIREMENT_LIVE' if alive else 'UNKNOWN_OR_FAILED_NO_REPLAY',
        process=process,child=child,result=result,actual_child_live=bool(child and ctl.live(child)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('start','supervise','worker','inspect'))
    p.add_argument('--commit',required=True);p.add_argument('--authorize-exact-five',action='store_true');a=p.parse_args()
    if a.mode=='supervise':sys.exit(supervise(a.commit))
    if a.mode=='worker':worker(a.commit)
    else:print(json.dumps(start(a.commit,authorize_exact_five=a.authorize_exact_five) if a.mode=='start' else inspect(a.commit)),flush=True)
