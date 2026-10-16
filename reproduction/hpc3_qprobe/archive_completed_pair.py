"""Copy all real paired checkpoint files to independent archive and target.

Only receipt path/inode metadata is relocated. Original HF, Adam, scheduler,
RNG, head, replay and probe-manifest bytes stay unchanged in both copies.
This CPU file transport cannot accept an actual CUDA restore.
"""
import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time

from reproduction.hpc3_qprobe.control import publish
from search_g1.qprobe_features import sha_file

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
STATUS='QPROBE_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS'


def file_records(value):
    if isinstance(value,dict):
        if all(key in value for key in ('path','inode','bytes','mtime_ns','full_read_sha256')):
            yield value
        else:
            for item in value.values():yield from file_records(item)
    elif isinstance(value,(list,tuple)):
        for item in value:yield from file_records(item)


def destination(original, source_base, target_base,checkpoint=None):
    original=Path(original).resolve();source_base=Path(source_base).resolve()
    if original.is_relative_to(source_base):return target_base/original.relative_to(source_base)
    if checkpoint is not None and original.is_relative_to(Path(checkpoint).resolve()):
        # Retained foreign CP5 is one complete tree, including training_state.
        # Hashing each parent separately would detach Adam/head from HF files.
        return target_base/'restored-checkpoint'/original.relative_to(Path(checkpoint).resolve())
    # Keep external encoder files together, so HF index relative paths work.
    group=hashlib.sha256(str(original.parent).encode()).hexdigest()
    return target_base/'external'/group/original.name


def copy_checked(source, target, expected_sha, expected_bytes):
    source=Path(source);target=Path(target)
    if source.is_symlink() or not source.is_file() or source.stat().st_size!=expected_bytes:
        raise ValueError('Archive source file/size changed')
    before=source.stat()
    if sha_file(source)!=expected_sha:raise ValueError('Fresh source fullSHA differs')
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.partial')
    if target.exists() or temporary.exists():raise ValueError('Copy already attempted; inspect partial state')
    h=hashlib.sha256();count=0
    with source.open('rb') as reader, temporary.open('xb') as writer:
        while block:=reader.read(8*2**20):
            writer.write(block);h.update(block);count+=len(block)
        writer.flush();os.fsync(writer.fileno())
    after=source.stat()
    if ((before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns) or
            count!=expected_bytes or h.hexdigest()!=expected_sha or sha_file(temporary)!=expected_sha):
        raise ValueError('Copy bytes or source changed; retain partial evidence')
    os.link(temporary,target);temporary.unlink()
    fd=os.open(target.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    if sha_file(target)!=expected_sha:raise ValueError('Published archive fullSHA differs')


def relocate(receipt, target_base):
    value=copy.deepcopy(receipt);source_base=Path(receipt['source_base']).resolve()
    for row in file_records(value):
        path=destination(row['path'],source_base,target_base,receipt['checkpoint']);st=path.stat()
        if st.st_size!=row['bytes'] or sha_file(path)!=row['full_read_sha256']:
            raise ValueError('Relocated receipt does not match actual destination bytes')
        row.update(path=str(path),inode=st.st_ino,mtime_ns=st.st_mtime_ns)
    for name in ('checkpoint','question_probe_state_path','head_path'):
        value[name]=str(destination(value[name],source_base,target_base,receipt['checkpoint']))
    value['source_base']=str(target_base)
    value['actual_cuda_restore_accepted']=False
    # Do not stamp a new producer, state, counters, or scientific acceptance.
    return value


def measured_owner_storage(run_id, step):
    """Only a complete zero-error walk admits copies; retain every failure.

    Live cache entries can disappear during du. Retry at most three complete
    walks for that exact ENOENT case, never subtract/ignore inaccessible files.
    Permission, quota, malformed output and other failures remain failures.
    """
    command=['du','-s','-B1',str(ROOT.parent)]
    for attempt in range(1,4):
        started=time.time()
        try:
            measured=subprocess.run(command,capture_output=True,text=True,timeout=1800,
                                    env=dict(os.environ,LC_ALL='C'))
        except subprocess.TimeoutExpired as error:
            print(json.dumps(dict(stage='archive_owner_storage_measurement_timeout',run_id=run_id,
                step=step,attempt=attempt,started_unix=started,completed_unix=time.time(),
                stdout=repr(error.stdout),stderr=repr(error.stderr),accepted=False)),flush=True)
            raise
        match=re.fullmatch(r'([0-9]+)\t'+re.escape(str(ROOT.parent))+r'\n?',measured.stdout)
        accepted=measured.returncode==0 and measured.stderr=='' and match is not None
        print(json.dumps(dict(stage='archive_owner_storage_measurement',run_id=run_id,step=step,
            attempt=attempt,started_unix=started,completed_unix=time.time(),
            actual_returncode=measured.returncode,stdout=measured.stdout,stderr=measured.stderr,
            accepted=accepted)),flush=True)
        if accepted:return int(match.group(1))
        lines=measured.stderr.splitlines()
        raced=measured.returncode==1 and bool(lines) and all(
            line.startswith('du: ') and str(ROOT.parent)+'/' in line and
            line.endswith(': No such file or directory') for line in lines)
        if not raced or attempt==3:
            raise ValueError('Fresh owner storage measurement failed: rc='+str(measured.returncode)+
                             '; stderr='+measured.stderr+'; stdout='+measured.stdout)
    raise AssertionError('unreachable storage admission')


def archive(receipt_path,*,storage_measurement=None,storage_semantics='complete_single_owner_allocated_measurement'):
    if os.getuid()!=14090:raise ValueError('Actual jxu957 archive owner required')
    from search_g1.qprobe_checkpoint import read_source
    original,receipt,_,origin,parent=read_source(receipt_path)
    if receipt['world_size']!=4 or receipt['step'] not in (5,50,100,150,200):
        raise ValueError('Only the actual native4 milestone pair is accepted')
    if parent.get('execution_site')!='hpc3' or not str(origin).startswith('/data/user/jxu957/experiments/qprobe-teacher-warmup200-'):
        raise ValueError('Archive is outside this own HPC3 experiment')
    directory=ROOT/'archives'/receipt['run_id']/('global_step_'+str(receipt['step']))
    if directory.exists():raise ValueError('Archive already attempted; inspect its durable claim')
    rows={}
    for row in file_records(receipt):
        path=str(Path(row['path']).resolve())
        if path in rows and (rows[path]['bytes'],rows[path]['full_read_sha256'])!=(row['bytes'],row['full_read_sha256']):
            raise ValueError('Conflicting source records')
        rows[path]=row
    measure=measured_owner_storage if storage_measurement is None else storage_measurement
    allocated=measure(receipt['run_id'],receipt['step'])
    if type(allocated) is not int or allocated<0 or storage_semantics not in (
            'complete_single_owner_allocated_measurement','complete_two_owner_allocated_upper_bound'):
        raise ValueError('Invalid complete storage measurement or semantics')
    required=2*sum(row['bytes'] for row in rows.values())+2**30
    if allocated+required>2*10**12:raise ValueError('Explicit own2TB archive/target allowance exceeded')
    os.umask(0o007);directory.mkdir(parents=True,mode=0o770)
    claim=dict(source_receipt=dict(path=str(original),sha256=sha_file(original)),run_id=receipt['run_id'],
        step=receipt['step'],world_size=4,uid=os.getuid(),pid=os.getpid(),
        measured_allocated_bytes=allocated,required_copy_bytes=required,storage_semantics=storage_semantics,
        allocated_upper_bound_bytes=allocated if storage_semantics=='complete_two_owner_allocated_upper_bound' else None,
        started_unix=time.time(),actual_cuda_restore_accepted=False)
    publish(directory/'claim.json',claim)
    archive_base=directory/'archive';target_base=directory/'target'
    archive_base.mkdir();target_base.mkdir()
    files=[]
    with (directory/'progress.jsonl').open('x') as progress:
        for source,row in sorted(rows.items()):
            paths=[destination(source,origin,b,receipt['checkpoint']) for b in (archive_base,target_base)]
            for target in paths:copy_checked(source,target,row['full_read_sha256'],row['bytes'])
            record=dict(source_path=source,archive_path=str(paths[0]),target_path=str(paths[1]),
                        bytes=row['bytes'],sha256=row['full_read_sha256'])
            files.append(record);progress.write(json.dumps(dict(record,completed_files=len(files),unix=time.time()))+'\n')
            progress.flush();os.fsync(progress.fileno())
    archive_receipt=archive_base/'relocated-source-receipt.json'
    target_receipt=target_base/'relocated-source-receipt.json'
    publish(archive_receipt,relocate(receipt,archive_base));publish(target_receipt,relocate(receipt,target_base))
    # Both consumer readbacks parse the real head and CPU saved-state bindings.
    read_source(archive_receipt);read_source(target_receipt)
    proof=dict(schema_version=1,status=STATUS,run_id=receipt['run_id'],step=receipt['step'],world_size=4,
        source_manifest_sha256=receipt['source_run_manifest']['full_read_sha256'],
        source_receipt=claim['source_receipt'],archive_receipt=dict(path=str(archive_receipt),sha256=sha_file(archive_receipt)),
        target_receipt=dict(path=str(target_receipt),sha256=sha_file(target_receipt)),files=files,
        archive_file_set=sorted([x['archive_path'] for x in files]+[str(archive_receipt)]),
        target_file_set=sorted([x['target_path'] for x in files]+[str(target_receipt)]),
        actual_cuda_restore_accepted=False,checked_unix=time.time(),
        scope='receipt paths/inode identity relocated; original checkpoint/state/reward manifests unchanged')
    publish(directory/'complete.json',proof)
    return dict(path=str(directory/'complete.json'),sha256=sha_file(directory/'complete.json'),
                status=STATUS,step=receipt['step'],files=len(files),bytes=sum(x['bytes'] for x in files),
                actual_cuda_restore_accepted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);a=p.parse_args()
    print(json.dumps(archive(a.receipt)),flush=True)
