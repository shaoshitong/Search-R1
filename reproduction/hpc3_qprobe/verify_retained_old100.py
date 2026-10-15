"""Once-only full SHA read of the original old100 retained on HPC3.

No tensor/model import, training control, deletion, or GPU allocation. This is
preparation evidence only; it grants no authority to retire protected originals.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from reproduction.hpc3_qprobe import reserve_old_probe_originals as reserve
from reproduction.hpc3_qprobe import independent_goal_acceptance719225 as reader

ROOT=reserve.ROOT
DIRECTORY=ROOT/'observations/old100-retention-readcheck-20261016'
RELATIVE='reproduction/hpc3_qprobe/verify_retained_old100.py'
OLD_BASE=ROOT.parent/'qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014'
STATUS='RETAINED_ORIGINAL_OLD100_FULL_SOURCE_SHA_AND_DEPENDENCY_PASS'


def need(ok,message):reserve.need(ok,'RETAINED_OLD100: '+message)


def code_guard(commit):
    p,repo=reserve.code_guard(commit,'source')
    proof=reserve.read_owned(ROOT/'preparation'/('source-'+commit+'-verification.json'),14090)
    need(Path(__file__).resolve()==repo/RELATIVE and reserve.digest(__file__)==proof['files'][RELATIVE],
         'actual immutable readcheck source differs')
    for name in ('reproduction/hpc3_qprobe/independent_goal_acceptance719225.py',):
        need(reserve.digest(repo/name)==proof['files'][name],'actual file reader source differs')
    return p,repo


def records(value):
    if isinstance(value,dict):
        if all(k in value for k in ('path','inode','bytes','mtime_ns','full_read_sha256')):yield value
        else:
            for item in value.values():yield from records(item)
    elif isinstance(value,(list,tuple)):
        for item in value:yield from records(item)


def unique_records(value):
    out={}
    for row in records(value):
        path=Path(row['path'])
        need(path.is_absolute() and '..' not in path.parts and reserve.below(path,ROOT.parent) and
             not reserve.below(path,reserve.BASE),'unsafe retained source path')
        if row['path'] in out:
            need(all(out[row['path']][k]==row[k] for k in ('inode','bytes','mtime_ns','full_read_sha256')),
                 'conflicting retained source records')
        else:out[row['path']]=row
    return out


def expected_metadata(p,transport,receipt):
    need(transport['status']=='QPROBE_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS' and
         transport['run_id']==receipt['run_id']==p['old_run_id'] and transport['step']==receipt['step']==100 and
         transport['world_size']==receipt['world_size']==4 and
         receipt['status']=='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS' and
         receipt['source_base']==str(OLD_BASE) and receipt['checkpoint']==str(
             OLD_BASE/'checkpoints/search-g1-common8/actor/global_step_100'),'old100 pair identity differs')
    by_path=unique_records(receipt)
    planned={r['source_path']:(r['bytes'],r['sha256']) for r in transport['files']}
    need(len(planned)==len(transport['files']) and set(by_path)==set(planned),'old100 complete source file set differs')
    for path,row in by_path.items():
        need((row['bytes'],row['full_read_sha256'])==planned[path],'old100 source SHA binding differs')
    dependencies={path for path in by_path if any(reserve.below(Path(path),Path(g['checkpoint'])) for g in p['groups'])}
    required={str(Path(g['checkpoint'])/'complete.json') for g in p['groups'][:2]}
    need(dependencies==required,'old100 aliases prospective payload beyond two retained complete manifests')
    return by_path,sorted(required)


def verify(commit):
    p,_=code_guard(commit)
    reserve.source_manifest(commit)  # Actual cancelled old719225 and original file identities.
    plan_files={x['path']:x for x in p['files']}
    tp=ROOT/'archives'/p['old_run_id']/'global_step_100/complete.json'
    raw,tp_proof=reader.read_file(tp,expected_sha=plan_files[str(tp)]['sha256'],capture=True)
    transport=json.loads(raw);sp=Path(transport['source_receipt']['path'])
    raw,sp_proof=reader.read_file(sp,expected_sha=transport['source_receipt']['sha256'],capture=True)
    receipt=json.loads(raw);by_path,required=expected_metadata(p,transport,receipt)
    checkpoint=Path(receipt['checkpoint']);actual={str(x) for x in checkpoint.rglob('*') if x.is_file()}
    need(actual=={s for s in by_path if reserve.below(Path(s),checkpoint)},'old100 checkpoint file set changed')
    verified=[]
    with (DIRECTORY/'progress.jsonl').open('x') as log:
        for path,row in sorted(by_path.items()):
            _,proof=reader.read_file(path,expected_sha=row['full_read_sha256'],expected_bytes=row['bytes'],
                                     shared=True)
            proof.update(device=proof['stat_identity'][0],inode=proof['stat_identity'][1],
                         mtime_ns=proof['stat_identity'][3])
            need(proof['inode']==row['inode'] and proof['mtime_ns']==row['mtime_ns'],
                 'retained original identity differs')
            verified.append(proof)
            log.write(json.dumps(dict(epoch=time.time(),files=len(verified),bytes=sum(v['bytes'] for v in verified)))+'\n')
            log.flush()
    for row in verified:
        s=Path(row['path']).lstat()
        need(reader.signature(s)==tuple(row['stat_identity']),
             'old100 file changed during complete read')
    value=dict(status=STATUS,source_commit=commit,transport=tp_proof,source_receipt=sp_proof,
        files=verified,verified_files=len(verified),verified_bytes=sum(v['bytes'] for v in verified),
        required_retained_metadata_paths=required,actual_process=reserve.birth(os.getpid()),
        completed_unix=time.time(),source_cleanup_performed=False,deletion_authorized=False,
        actor_model_loaded=False,training_performed=False,gpu_allocations=0)
    reserve.publish(DIRECTORY/'complete.json',value)
    return value


def worker(commit):
    code_guard(commit);binding=reserve.read_owned(DIRECTORY/'worker-process.json',14090)
    need(binding['source_commit']==commit and binding['actual_process']==reserve.birth(os.getpid()),'actual held worker differs')
    try:
        verify(commit)
        value=dict(status=STATUS,exit_code=0,complete_sha256=reserve.digest(DIRECTORY/'complete.json'))
    except Exception as exc:value=dict(status='OLD100_READCHECK_FAILED_RETAIN_EVIDENCE',exit_code=1,error=repr(exc))
    value.update(actual_process=reserve.birth(os.getpid()),completed_unix=time.time(),source_cleanup_performed=False)
    reserve.publish(DIRECTORY/'worker-result.json',value);return value['exit_code']


def supervise(commit):
    _,repo=code_guard(commit);binding=reserve.read_owned(DIRECTORY/'process.json',14090)
    need(binding['actual_process']==reserve.birth(os.getpid()) and binding['source_commit']==commit,'actual supervisor differs')
    actual=None;error=None
    with (DIRECTORY/'worker.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(repo/RELATIVE),'worker','--commit',commit],
            stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
        try:
            actual=reserve.birth(child.pid);need(actual and actual['uid']==14090 and actual['sid']==os.getsid(0),'actual child birth differs')
            reserve.publish(DIRECTORY/'worker-process.json',dict(source_commit=commit,actual_process=actual))
            child.stdin.write(b'ready\n')
        except Exception as exc:error=exc
        finally:
            try:child.stdin.close()
            except Exception as exc:
                if error is None:error=exc
            finally:code=child.wait()
    result=dict(status='ACTUAL_HELD_OLD100_READCHECK_FAILED',exit_code=1,actual_child_exit_code=code,
        actual_supervisor=binding['actual_process'],actual_worker=actual,error=repr(error) if error else None,
        completed_unix=time.time(),source_cleanup_performed=False)
    try:
        if code==0 and error is None:
            returned=reserve.read_owned(DIRECTORY/'worker-result.json',14090)
            need(returned['actual_process']==actual and returned['status']==STATUS and returned['exit_code']==0 and
                 returned['complete_sha256']==reserve.digest(DIRECTORY/'complete.json'),'actual held readcheck result differs')
            result.update(status='ACTUAL_HELD_OLD100_READCHECK_COMPLETED0',exit_code=0,
                          complete_sha256=returned['complete_sha256'])
    except Exception as exc:result['validation_error']=repr(exc)
    reserve.publish(DIRECTORY/'result.json',result);return result['exit_code']


def start(commit):
    _,repo=code_guard(commit);need(not DIRECTORY.exists(),'readcheck already attempted; inspect only')
    DIRECTORY.mkdir(mode=0o700)
    reserve.publish(DIRECTORY/'claim.json',dict(source_commit=commit,plan_sha256=reserve.PLAN_SHA,
        created_unix=time.time(),source_cleanup_performed=False,deletion_authorized=False))
    with (DIRECTORY/'supervisor.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(repo/RELATIVE),'supervise','--commit',commit],
            stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        actual=None
        try:
            actual=reserve.birth(child.pid);need(actual and actual['uid']==14090 and actual['sid']==actual['pid'],'actual supervisor birth differs')
            reserve.publish(DIRECTORY/'process.json',dict(source_commit=commit,actual_process=actual))
            child.stdin.write(b'ready\n');child.stdin.close()
        except Exception:
            try:child.stdin.close()
            except Exception:pass
            child.wait();raise
    return dict(status='OLD100_READCHECK_STARTED_ONCE',actual_process=actual,source_commit=commit)


def inspect(commit):
    code_guard(commit);parent=reserve.read_owned(DIRECTORY/'process.json',14090)
    need(parent['source_commit']==commit,'original readcheck source differs')
    result=reserve.read_owned(DIRECTORY/'result.json',14090) if (DIRECTORY/'result.json').exists() else None
    out=dict(epoch=time.time(),process=parent,actual_live_birth=reserve.live(parent['actual_process']),result=result,
             source_cleanup_performed=False)
    if (DIRECTORY/'worker-process.json').exists():
        child=reserve.read_owned(DIRECTORY/'worker-process.json',14090)
        out['worker']=dict(actual_process=child['actual_process'],live=reserve.live(child['actual_process']))
    if (DIRECTORY/'progress.jsonl').exists():
        rows=(DIRECTORY/'progress.jsonl').read_bytes().split(b'\n')[:-1]
        out['progress']=json.loads(rows[-1]) if rows else None
    if result and result['exit_code']==0:
        need(result['actual_child_exit_code']==0 and result['actual_supervisor']==parent['actual_process'] and
             result['actual_worker']==out.get('worker',{}).get('actual_process') and
             result['complete_sha256']==reserve.digest(DIRECTORY/'complete.json'),'held readcheck receipt differs')
        value=reserve.read_owned(DIRECTORY/'complete.json',14090)
        for row in value['files']:
            s=Path(row['path']).lstat()
            need(reader.signature(s)==tuple(row['stat_identity']),'retained source metadata changed')
        out['verified_files']=value['verified_files'];out['verified_bytes']=value['verified_bytes']
        out['required_retained_metadata_paths']=value['required_retained_metadata_paths']
        out['accepted']=not out['actual_live_birth'] and not out['worker']['live']
    for name in ('supervisor.log','worker.log'):
        path=DIRECTORY/name
        if path.exists():
            with path.open('rb') as f:f.seek(max(0,path.stat().st_size-2000));out[name]=f.read().decode(errors='replace')
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','supervise','worker','inspect'));p.add_argument('--commit',required=True);a=p.parse_args()
    reserve.caps();os.umask(0o077)
    if a.mode in ('supervise','worker'):
        need(sys.stdin.readline()=='ready\n','actual parent readiness/birth binding missing')
        sys.exit(supervise(a.commit) if a.mode=='supervise' else worker(a.commit))
    print(json.dumps(start(a.commit) if a.mode=='start' else inspect(a.commit)),flush=True)


if __name__=='__main__':main()
