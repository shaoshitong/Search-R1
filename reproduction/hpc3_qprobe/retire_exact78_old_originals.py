"""Human-authorized exact78 retirement; no training control or tensor imports.

All candidate FDs remain open after their full SHA checks until unlink through
their bound parent FDs. Old100's 37 retained files are fully read before AND
after retirement. A once-only supervisor retains the actual child through wait.
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
from reproduction.hpc3_qprobe import reserve_old_probe_originals as reserve
from reproduction.hpc3_qprobe import retire_old_sigma3_duplicates as old
from reproduction.hpc3_qprobe import verify_retained_old100 as keep100
from reproduction.hpc3_qprobe import independent_goal_acceptance719225 as reader

OWNER=14090
ROOT=reserve.ROOT
DIRECTORY=ROOT/'observations/retire-exact78-old-originals-20261016'
ADMISSION=ROOT/'observations/retire-exact78-old-originals-admission-20261016.json'
RELATIVE='reproduction/hpc3_qprobe/retire_exact78_old_originals.py'
PLAN_NAME='old_original_retirement_plan_20261016.json'
PLAN_SHA='a04121d8ddfc4d552330d666143f73f43d1a8b17a2c716f00ed298e59d8e08bb'
HUMAN_NAME='old_original_retirement_authorization_20261016.json'
HUMAN_SHA='e5fcfea443213760bf4af2246f3fc212ad926e5e44f0e4b425faf73e6e3d6337'
BACKUP_SHA='ba51bc0dc2511117bfef6072832d3ee28f475b0828fb6ff2037d1d52b2f98739'
DURABLE_SHA='1410139a59035dd45a228fce4ac6b8bc4f67591420dd07ef4de1a25bb1b202d1'
KEEP_SHA='48a249ab7516d391468227cc32defa05e451ed48b6918c66f2a39815d2dd733f'
STATUS='EXACT78_RETIRED_RETAINED_OLD100_37_FULL_SHA_PASS'
HELD_STATUS='ACTUAL_HELD_EXACT78_RETIREMENT_COMPLETED0'
TOTAL=115100222440


def need(ok,message):reserve.need(ok,'EXACT78: '+message)


def document(path,sha):
    raw,proof=reader.read_file(path,expected_sha=sha,capture=True)
    return json.loads(raw),proof


def baseline():
    p,proof=document(Path(__file__).with_name(PLAN_NAME),PLAN_SHA)
    a,authorization=document(Path(__file__).with_name(HUMAN_NAME),HUMAN_SHA)
    need(a['authorized'] is True and a['source']=='direct human reply in current thread' and
         a['proposal_sha256']==PLAN_SHA and a['exact_files']==78 and a['exact_logical_bytes']==TOTAL,
         'exact human authorization differs')
    paths=[r['path'] for r in p['candidate_files']]
    roots=[g['checkpoint'] for g in reserve.plan()['groups']]
    kept=[str(Path(x)/'complete.json') for x in roots[:2]]
    need(p['prospective_trees']==roots and len(paths)==len(set(paths))==p['candidate_count']==78 and
         sum(r['bytes'] for r in p['candidate_files'])==p['candidate_logical_bytes']==TOTAL and
         [r['path'] for r in p['retained_source100_dependency_metadata']]==kept and
         not set(paths)&set(kept),'approved complete scope differs')
    originals={r['path']:r for r in reserve.plan()['files']}
    for row in p['candidate_files']+p['retained_source100_dependency_metadata']:
        path=Path(row['path'])
        need(path.is_absolute() and path.resolve()==path and any(reserve.below(path,Path(x)) for x in roots) and
             row['uid']==OWNER and row['nlink']==1 and row['path'] in originals and
             all(row[k]==originals[row['path']][k] for k in ('bytes','sha256','inode','device','mtime_ns','uid','gid','nlink')),
             'scope, owner, or fixed original snapshot differs')
    need(set(paths)|set(kept)=={path for path in originals if any(
        reserve.below(Path(path),Path(x)) for x in roots)},'candidate plus keep union differs')
    return p,proof,authorization


def code_guard(commit):
    _,repo=reserve.code_guard(commit,'source')
    repository,source=ctl.source(commit)
    need(repo==repository and Path(__file__).resolve()==repo/RELATIVE,'actual code differs')
    return source


def admission(expected_sha,*,fresh=False):
    a,proof=document(ADMISSION,expected_sha)
    need(a['status']=='AUTHENTICATED_HKUST3_TERMINAL_RESERVE_AND_DURABILITY_PASS' and
         a['actual_ssh_exit_code']==0 and a['backup_complete_sha256']==BACKUP_SHA and
         a['durability_sha256']==DURABLE_SHA and a['backup_files']==118 and
         a['backup_bytes']==121296682925 and a['target_metadata_full_set_verified'] is True and
         a['actual_terminal_accepted'] is True and a['target_host']==reserve.plan()['target_host'],
         'authenticated backup/durability admission differs')
    if fresh:need(0<=time.time()-a['observed_unix']<=1800,'backup admission stale before first unlink')
    return proof


def consumers(plan):
    stopped=old.stopped()
    run,binding,accounting=ctl.producer(reserve.BASE)
    need(binding['producer_job']=='721485' and run['run_id']=='70827e85fe584f0c84453aae907918a4' and
         run['source_commit']==ctl.SIGMA3_PRODUCER_SOURCE and '|RUNNING|0:0|' in accounting,
         'current native4 producer is not the expected healthy run')
    jobs=subprocess.check_output(['squeue','-h','-u','jxu957,mou176','-o','%i|%u|%T|%Z|%o'],text=True,timeout=30)
    rows=[r.split('|') for r in jobs.splitlines() if r.strip()]
    need(len(rows)==1 and rows[0][:3]==['721485','jxu957','RUNNING'],'other active job requires dependency re-audit')
    inspected=[]
    for path in (reserve.BASE/'logs/run-manifest.json',reserve.BASE/'logs/resolved-training-config.json',
                 reserve.BASE/'artifacts/question-probe/teacher-warmup-ready.json'):
        # Exact runtime names are validated by local preflight; fail closed if absent.
        value,proof=reader.document(path)
        def strings(v):
            if isinstance(v,str):yield v
            elif isinstance(v,dict):
                for k,x in v.items():yield from strings(k);yield from strings(x)
            elif isinstance(v,list):
                for x in v:yield from strings(x)
        for s in strings(value):
            for root in plan['prospective_trees']:
                need(root not in s and not (s.startswith('/') and reserve.below(Path(s).resolve(),Path(root))),
                     'active runtime references a prospective tree')
        inspected.append(proof)
    return dict(stopped=stopped,producer=binding,accounting=accounting,active_jobs=jobs,runtime=inspected)


def retained_full(phase):
    previous,_=document(keep100.DIRECTORY/'complete.json',KEEP_SHA)
    result=reserve.read_owned(keep100.DIRECTORY/'result.json',OWNER)
    need(previous['verified_files']==37 and previous['verified_bytes']==38371213245 and
         result['complete_sha256']==KEEP_SHA and result['actual_child_exit_code']==result['exit_code']==0 and
         result['actual_worker']==previous['actual_process'] and not reserve.live(result['actual_worker']) and
         not reserve.live(result['actual_supervisor']),'original retained100 held full SHA proof differs')
    verified=[]
    for row in previous['files']:
        _,proof=reader.read_file(row['path'],expected_sha=row['sha256'],expected_bytes=row['bytes'],shared=True)
        need(proof['stat_identity']==row['stat_identity'],'retained100 original identity differs')
        verified.append(proof);progress(phase,files=len(verified),bytes=sum(r['bytes'] for r in verified))
    # Full source100 checkpoint file set, including metadata dependencies outside it.
    checkpoint=keep100.OLD_BASE/'checkpoints/search-g1-common8/actor/global_step_100'
    actual={str(x) for x in checkpoint.rglob('*') if x.is_file()}
    need(actual=={r['path'] for r in verified if reserve.below(Path(r['path']),checkpoint)},'retained100 fileset differs')
    return verified


def progress(phase,**kwargs):
    print(json.dumps(dict(epoch=time.time(),phase=phase,**kwargs)),flush=True)


def snapshots(plan,*,after=False):
    snapshots={};actual=set()
    for root in plan['prospective_trees']:
        snapshots[root]=old.tree(Path(root));actual.update(snapshots[root]['files'])
    candidates={r['path'] for r in plan['candidate_files']}
    kept={r['path'] for r in plan['retained_source100_dependency_metadata']}
    need(actual==(kept if after else candidates|kept),'approved three-tree fileset differs')
    for row in plan['retained_source100_dependency_metadata']:
        reserve.row_stat(row);reader.read_file(row['path'],expected_sha=row['sha256'],expected_bytes=row['bytes'])
    return snapshots


def held_check(item):
    row,parent,fd,signature=item
    s=os.fstat(fd);name=Path(row['path']).name
    current=os.stat(name,dir_fd=parent,follow_symlinks=False)
    need(reader.signature(s)==signature==reader.signature(current) and s.st_uid==current.st_uid==OWNER and
         s.st_nlink==current.st_nlink==1,'held candidate changed before unlink')
    need(Path(row['path']).resolve()==Path(row['path']) and
         reader.signature(Path(row['path']).lstat())==signature,'parent name detached or replaced')


def open_hash(row,root,snapshot):
    rootfd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    parent=None;fd=None
    try:
        s=os.fstat(rootfd);expected=snapshot['directories']['']
        need((s.st_dev,s.st_ino,s.st_uid)==(expected['device'],expected['inode'],OWNER),'root FD changed')
        relative=Path(row['path']).relative_to(root)
        parent=old.open_relative(rootfd,relative.parts[:-1],snapshot['directories'])
        fd=os.open(relative.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
        s=os.fstat(fd)
        need(stat.S_ISREG(s.st_mode) and (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_uid,s.st_gid,s.st_nlink)==
             tuple(row[k] for k in ('device','inode','bytes','mtime_ns','uid','gid','nlink')),'candidate snapshot changed')
        signature=reader.signature(s);h=hashlib.sha256()
        while True:
            block=os.read(fd,8*2**20)
            if not block:break
            h.update(block)
        item=(row,parent,fd,signature);held_check(item)
        need(h.hexdigest()==row['sha256'],'candidate full SHA differs')
        return item
    except BaseException:
        if fd is not None:os.close(fd)
        if parent is not None:os.close(parent)
        raise
    finally:os.close(rootfd)


def unlink_held(item):
    held_check(item)
    row,parent,fd,_=item;allocated=os.fstat(fd).st_blocks*512
    os.unlink(Path(row['path']).name,dir_fd=parent);os.fsync(parent)
    need(not os.path.lexists(row['path']),'approved file name remains after unlink')
    return allocated


def worker(commit):
    source=code_guard(commit);claim=reserve.read_owned(DIRECTORY/'claim.json',OWNER)
    binding=reserve.read_owned(DIRECTORY/'worker-process.json',OWNER)
    need(claim['source']==source and binding['actual_process']==reserve.birth(os.getpid()),'actual held worker differs')
    plan,proof,authorization=baseline()
    need(claim['plan']==proof and claim['authorization']==authorization,'immutable approved bytes differ')
    admission(claim['admission_sha256'],fresh=True);consumers(plan)
    snapshot=snapshots(plan);before=retained_full('retained100-before')
    held=[];retired=[];allocated=0
    try:
        for row in plan['candidate_files']:
            root=next(Path(r) for r in plan['prospective_trees'] if reserve.below(Path(row['path']),Path(r)))
            held.append(open_hash(row,root,snapshot[str(root)]))
            progress('candidate-full-sha',files=len(held),bytes=sum(x[0]['bytes'] for x in held))
        snapshots(plan);current=consumers(plan);admission(claim['admission_sha256'],fresh=True)
        for item in held:held_check(item)
        reserve.publish(DIRECTORY/'preflight.json',dict(plan=proof,authorization=authorization,
            admission_sha256=claim['admission_sha256'],retained100=before,consumers=current,
            all78_full_sha_before_first_unlink=True,candidates=[dict(row=x[0],stat_identity=x[3]) for x in held],files=78,bytes=TOTAL,epoch=time.time()))
        for i,item in enumerate(held):
            allocated+=unlink_held(item);row,parent,fd,_=item
            os.close(fd);os.close(parent);held[i]=None
            retired.append(row['path']);progress('unlinked',files=len(retired),bytes=sum(
                r['bytes'] for r in plan['candidate_files'] if r['path'] in retired))
    finally:
        for item in held:
            if item is not None:os.close(item[2]);os.close(item[1])
    after=retained_full('retained100-after');snapshots(plan,after=True);old.stopped()
    need(len(retired)==78 and all(not os.path.lexists(p) for p in retired),'retired member remained or returned')
    receipt=dict(status=STATUS,source=source,actual_process=reserve.birth(os.getpid()),
        claim_sha256=reserve.digest(DIRECTORY/'claim.json'),plan_sha256=PLAN_SHA,human_sha256=HUMAN_SHA,
        preflight_sha256=reserve.digest(DIRECTORY/'preflight.json'),retired_paths=retired,
        retired_files=78,retired_logical_bytes=TOTAL,unlinked_inode_allocated_bytes=allocated,
        global_quota_reduction_measured=False,retained100_full_sha=after,retained100_files=37,
        retained100_bytes=38371213245,kept_metadata=plan['retained_source100_dependency_metadata'],
        training_actions=0,model_loads=0,gpu_allocations=0,completed_unix=time.time())
    reserve.publish(DIRECTORY/'complete.json',receipt)


def supervise(commit):
    code_guard(commit);parent=reserve.read_owned(DIRECTORY/'process.json',OWNER)
    need(parent['actual_process']==reserve.birth(os.getpid()),'supervisor birth differs')
    child=None;actual=None;code=None;error=None
    try:
        with (DIRECTORY/'worker.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--commit',commit],
                stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
            try:
                actual=reserve.birth(child.pid)
                need(actual and actual['sid']==parent['actual_process']['sid'] and actual['uid']==OWNER,'worker birth differs')
                reserve.publish(DIRECTORY/'worker-process.json',dict(source_commit=commit,actual_process=actual))
                child.stdin.write(b'ready\n');child.stdin.close()
            except BaseException as exc:
                error=exc
                try:child.stdin.close()
                except Exception:pass
            finally:code=child.wait()
            log.flush();os.fsync(log.fileno())
    except BaseException as exc:
        error=exc
        if child is not None and code is None:code=child.wait()
    result=dict(status='EXACT78_FAILED_OR_UNKNOWN_NO_REPLAY',source_commit=commit,
        actual_supervisor=parent['actual_process'],actual_worker=actual,actual_child_exit_code=code,
        exit_code=1,error=repr(error) if error else None,completed_unix=time.time())
    try:
        if code==0 and error is None:
            receipt=reserve.read_owned(DIRECTORY/'complete.json',OWNER)
            need(receipt['status']==STATUS and receipt['actual_process']==actual,'held child receipt differs')
            result.update(status=HELD_STATUS,exit_code=0,complete_sha256=reserve.digest(DIRECTORY/'complete.json'))
    except BaseException as exc:result['validation_error']=repr(exc)
    reserve.publish(DIRECTORY/'result.json',result);return result['exit_code']


def start(commit,admission_sha,authorized):
    source=code_guard(commit)
    if DIRECTORY.exists():return inspect(commit)
    need(authorized is True,'explicit exact-plan authorization flag required')
    plan,proof,authorization=baseline();admission(admission_sha,fresh=True);consumers(plan);snapshots(plan)
    DIRECTORY.mkdir(mode=0o700)
    reserve.publish(DIRECTORY/'claim.json',dict(source=source,source_commit=commit,plan=proof,
        authorization=authorization,admission_sha256=admission_sha,created_unix=time.time()))
    with (DIRECTORY/'supervisor.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'supervise','--commit',commit],
            stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            actual=reserve.birth(child.pid)
            need(actual and actual['uid']==OWNER and actual['sid']==actual['pid'],'detached supervisor birth differs')
            reserve.publish(DIRECTORY/'process.json',dict(source_commit=commit,actual_process=actual))
            child.stdin.write(b'ready\n');child.stdin.close()
        except BaseException:
            try:child.stdin.close()
            except Exception:pass
            child.wait();raise
    return dict(status='EXACT78_ONCE_CPU_STARTED_NOT_ACCEPTED',actual_process=actual,source_commit=commit)


def inspect(commit):
    source=code_guard(commit);claim=reserve.read_owned(DIRECTORY/'claim.json',OWNER)
    need(claim['source']==source,'once claim immutable source differs')
    parent=reserve.read_owned(DIRECTORY/'process.json',OWNER) if (DIRECTORY/'process.json').exists() else None
    child=reserve.read_owned(DIRECTORY/'worker-process.json',OWNER) if (DIRECTORY/'worker-process.json').exists() else None
    result=reserve.read_owned(DIRECTORY/'result.json',OWNER) if (DIRECTORY/'result.json').exists() else None
    live=bool(parent and reserve.live(parent['actual_process']))
    out=dict(epoch=time.time(),process=parent,worker=child,actual_live=live,result=result,accepted=False)
    log=DIRECTORY/'worker.log'
    if log.exists():
        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-3000));out['tail']=f.read().decode(errors='replace')
    if result and result['exit_code']==0 and not live:
        need(parent and child and not reserve.live(child['actual_process']) and
             result['source_commit']==commit and result['status']==HELD_STATUS and result['actual_child_exit_code']==0 and
             result['error'] is None and result['actual_supervisor']==parent['actual_process'] and
             result['actual_worker']==child['actual_process'] and child['actual_process']['sid']==parent['actual_process']['sid'] and
             result['complete_sha256']==reserve.digest(DIRECTORY/'complete.json'),'actual held wait proof differs')
        receipt=reserve.read_owned(DIRECTORY/'complete.json',OWNER);plan,proof,authorization=baseline()
        need(receipt['status']==STATUS and receipt['source']==source and receipt['actual_process']==result['actual_worker'] and
             receipt['claim_sha256']==reserve.digest(DIRECTORY/'claim.json') and claim['plan']==proof and
             claim['authorization']==authorization and receipt['preflight_sha256']==reserve.digest(DIRECTORY/'preflight.json') and
             receipt['retired_paths']==[r['path'] for r in plan['candidate_files']] and
             receipt['retired_files']==78 and receipt['retired_logical_bytes']==TOTAL and
             receipt['retained100_files']==len(receipt['retained100_full_sha'])==37 and receipt['retained100_bytes']==sum(r['bytes'] for r in receipt['retained100_full_sha'])==38371213245 and
             receipt['training_actions']==receipt['model_loads']==receipt['gpu_allocations']==0,
             'actual completed cleanup scope differs')
        snapshots(plan,after=True)
        for row in receipt['retained100_full_sha']:
            need(reader.signature(Path(row['path']).lstat())==tuple(row['stat_identity']),'retained100 post-read identity changed')
        out.update(accepted=True,retired_files=78,retired_logical_bytes=TOTAL,
            unlinked_inode_allocated_bytes=receipt['unlinked_inode_allocated_bytes'],retained100_files=37,
            complete_sha256=result['complete_sha256'])
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('start','worker','supervise','inspect'))
    parser.add_argument('--commit',required=True);parser.add_argument('--admission-sha')
    parser.add_argument('--authorize-exact-plan',action='store_true');args=parser.parse_args()
    reserve.caps();os.umask(0o077)
    if args.mode in ('worker','supervise'):
        need(sys.stdin.readline()=='ready\n','actual parent readiness missing')
        if args.mode=='worker':worker(args.commit)
        else:sys.exit(supervise(args.commit))
    else:print(json.dumps(start(args.commit,args.admission_sha,args.authorize_exact_plan) if args.mode=='start'
                          else inspect(args.commit)),flush=True)


if __name__=='__main__':main()
