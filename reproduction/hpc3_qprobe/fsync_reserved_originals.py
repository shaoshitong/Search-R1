"""Terminal durability audit of the completed old-original reserve, no deletion.

Consumes the actual held full-SHA backup receipt, then fsyncs those exact owned
files and directories. It does not claim another full-content readback, reload
models, or modify the pinned backup worker/source or any HPC3 artifact.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import time

TARGET=Path('/backup/shaoshitong/qprobe-old-original-reserve-20261016')
AUDIT=TARGET.with_name('qprobe-old-original-reserve-durability-20261016')
BACKUP_COMMIT='afda9f708ad0d93312c162d3aa8452816c420f35'
RELATIVE='reproduction/hpc3_qprobe/reserve_old_probe_originals.py'


def need(ok,message):
    if not ok:raise ValueError('RESERVE_DURABILITY: '+message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def signature(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_uid)


def flush_file(path,row,owner):
    path=Path(path);before=path.lstat()
    need(path.resolve()==path and stat.S_ISREG(before.st_mode) and before.st_uid==owner and
         (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
         tuple(row[k] for k in ('device','inode','bytes','mtime_ns')),'owned target identity differs')
    fd=os.open(str(path),os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        need(signature(os.fstat(fd))==signature(before),'target changed before fsync')
        os.fsync(fd)
        need(signature(os.fstat(fd))==signature(before)==signature(path.lstat()),'target changed during fsync')
    finally:os.close(fd)
    return dict(path=str(path),device=before.st_dev,inode=before.st_ino,bytes=before.st_size,
                mtime_ns=before.st_mtime_ns,sha256_from_completed_full_read=row['sha256'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True)
    parser.add_argument('--expected-complete-sha',required=True);args=parser.parse_args()
    need(re.fullmatch('[0-9a-f]{40}',args.commit) and re.fullmatch('[0-9a-f]{64}',args.expected_complete_sha),
         'explicit published code and actual backup receipt SHA required')
    os.umask(0o077)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]='1'
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    code=Path(__file__).resolve();proof_path=AUDIT/('code-'+args.commit+'-verification.json')
    proof=json.loads(proof_path.read_text())
    need(os.getuid()==1002 and code==AUDIT/('code-'+args.commit+'.py') and
         code.lstat().st_uid==proof_path.lstat().st_uid==1002 and proof_path.resolve()==proof_path and
         proof['status']=='PUBLISHED_SINGLE_FILE_FULL_SHA_PASS' and
         proof['commit']==proof['published_remote_sha']==args.commit and
         proof['code_sha256']==digest(code),'actual separately published durability code differs')
    original_path=TARGET/'source'/BACKUP_COMMIT/RELATIVE
    original_proof=json.loads((TARGET/'preparation'/('source-'+BACKUP_COMMIT+'-verification.json')).read_text())
    need(original_path.resolve()==original_path and original_path.lstat().st_uid==1002 and
         original_proof['commit']==original_proof['published_remote_sha']==BACKUP_COMMIT and
         original_proof['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS' and
         digest(original_path)==original_proof['files'][RELATIVE],'pinned original verifier bytes differ')
    spec=importlib.util.spec_from_file_location('original_reserve',str(original_path))
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    p,_=r.code_guard(BACKUP_COMMIT,'target')
    observed=r.inspect(BACKUP_COMMIT)
    need(observed.get('actual_terminal_accepted') is True and
         observed.get('current_target_metadata_full_set_verified') is True and
         observed['complete_sha256']==args.expected_complete_sha,'backup not actually terminal/full-SHA accepted')
    complete=r.read_owned(r.CONTROL/'complete.json',1002)
    directory=AUDIT/'final';need(not directory.exists(),'durability audit already attempted; inspect, never replay')
    directory.mkdir(mode=0o700)
    process=r.birth(os.getpid());r.publish(directory/'claim.json',dict(actual_process=process,source_commit=args.commit,
        backup_source_commit=BACKUP_COMMIT,backup_complete_sha256=args.expected_complete_sha,started_unix=time.time(),
        source_cleanup_performed=False,deletion_authorized=False))
    files=[];folders={TARGET,r.PAYLOAD,AUDIT,directory}
    with (directory/'progress.jsonl').open('x') as log:
        for row in complete['files']:
            path=Path(row['target_path']);need(r.below(path,r.PAYLOAD),'fsync path outside fixed payload')
            files.append(flush_file(path,row,1002))
            parent=path.parent
            while r.below(parent,r.PAYLOAD):folders.add(parent);parent=parent.parent
            log.write(json.dumps(dict(files=len(files),bytes=sum(x['bytes'] for x in files),epoch=time.time()))+'\n');log.flush()
    for path in sorted(folders,key=lambda x:(-len(x.parts),str(x))):
        before=path.lstat();need(path.resolve()==path and stat.S_ISDIR(before.st_mode) and before.st_uid==1002,'unsafe owned directory')
        fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            need((os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(before.st_dev,before.st_ino),'directory changed before fsync')
            os.fsync(fd)
        finally:os.close(fd)
    final=r.inspect(BACKUP_COMMIT)
    need(final.get('actual_terminal_accepted') is True and final['complete_sha256']==args.expected_complete_sha and
         len(files)==118 and sum(x['bytes'] for x in files)==p['total_bytes']==121296682925,
         'completed payload or receipt changed during durability audit')
    value=dict(status='ACTUAL_COMPLETED_FULL_SHA_BACKUP_OWNED_FILES_AND_DIRECTORIES_FSYNC_PASS',
        source_commit=args.commit,code_sha256=proof['code_sha256'],backup_source_commit=BACKUP_COMMIT,
        backup_complete_sha256=args.expected_complete_sha,actual_process=process,verified_files=len(files),
        verified_bytes=sum(x['bytes'] for x in files),synced_directories=len(folders),files=files,
        scope='Consume completed full SHA and verify current identities before/after fsync; no new full-content rehash',
        completed_unix=time.time(),source_cleanup_performed=False,deletion_authorized=False,actor_model_loaded=False)
    r.publish(directory/'complete.json',value)
    print(json.dumps(dict(status=value['status'],receipt_path=str(directory/'complete.json'),
        receipt_sha256=digest(directory/'complete.json'),files=value['verified_files'],bytes=value['verified_bytes'],
        synced_directories=len(folders),completed_unix=value['completed_unix'],source_cleanup_performed=False)),flush=True)


if __name__=='__main__':main()
