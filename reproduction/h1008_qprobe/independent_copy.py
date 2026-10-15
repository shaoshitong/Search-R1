"""Copy a pinned checkpoint closure to h100-10 ext4; never load models or GPUs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

EXPORT=Path('/mnt/shared/workspace/training_data_node_8')
BACKUPS=Path('/mnt/data/shaoshitong/search-g1-qprobe3-independent-backup-20261015')
RATE=20*1024**2
RESERVE=128*1024**3


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()
def identity(path):
    s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
def record(root,name,value):
    p=root/(name+'.partial')
    with p.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(p,root/name)
    fd=os.open(root,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def once(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def copy_and_verify(source,target,expected_sha,expected_bytes,*,rate=RATE,reserve=RESERVE,progress=None):
    source=Path(source);target=Path(target);before=identity(source)
    if before[2]!=expected_bytes:raise ValueError('SOURCE_SIZE_CHANGED')
    target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_name(target.name+'.partial')
    started=time.monotonic();total=0;h=hashlib.sha256()
    with source.open('rb') as src,partial.open('xb') as dst:
        while True:
            data=src.read(8*1024**2)
            if not data:break
            if shutil.disk_usage(target.parent).free<len(data)+reserve:raise ValueError('ORDINARY_FREE_RESERVE_REQUIRED')
            dst.write(data);h.update(data);total+=len(data)
            if rate:
                delay=total/rate-(time.monotonic()-started)
                if delay>0:time.sleep(delay)
            if progress:progress(total)
        dst.flush();os.fsync(dst.fileno())
    if identity(source)!=before:raise ValueError('SOURCE_MUTATED_DURING_FULL_READ')
    source_sha=h.hexdigest()
    if total!=expected_bytes or source_sha!=expected_sha:raise ValueError('SOURCE_FULL_SHA_DIFFERS')
    if progress:progress(total,phase='independent_target_full_read')
    target_sha=digest(partial)
    if target_sha!=expected_sha:raise ValueError('INDEPENDENT_TARGET_FULL_SHA_DIFFERS')
    # Publication cannot overwrite an earlier result or another writer.
    os.link(partial,target);partial.unlink()
    fd=os.open(target.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(bytes=total,expected_sha256=expected_sha,source_full_read_sha256=source_sha,target_full_read_sha256=target_sha)


def validate_environment(root):
    if socket.gethostname().split('.')[0]!='h100-10':raise ValueError('RETAINED_H10010_ONLY')
    root=root.resolve()
    if not root.is_relative_to(BACKUPS.resolve()):raise ValueError('TASK_BACKUP_NAMESPACE_REQUIRED')
    root.parent.mkdir(parents=True,exist_ok=True)
    fs=subprocess.check_output(['findmnt','-n','-T',str(root.parent),'-o','FSTYPE'],text=True).strip()
    if fs!='ext4':raise ValueError('INDEPENDENT_LOCAL_EXT4_REQUIRED')
    return root


def worker(root):
    root=validate_environment(Path(root));intent=json.loads((root/'INTENT.json').read_text())
    inventory_path=Path(intent['inventory_path'])
    if digest(inventory_path)!=intent['inventory_sha256']:raise ValueError('PINNED_INVENTORY_CHANGED')
    plan=json.loads(inventory_path.read_text())
    if plan['status']!='BACKUP_INVENTORY_ONLY' or plan['run_id']!=intent['run_id'] or plan['step'] not in (5,50,100,150,200):raise ValueError('CHECKPOINT_INVENTORY_IDENTITY_DIFFERS')
    files=plan['files'];total=sum(r['bytes'] for r in files)
    if shutil.disk_usage(root).free<total+RESERVE:raise ValueError('FULL_CLOSURE_PLUS_128GIB_ORDINARY_RESERVE_REQUIRED')
    staging=root/'staging';staging.mkdir(exist_ok=False);results=[];names=set()
    for index,row in enumerate(files):
        relative=Path(row['payload_path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in names:raise ValueError('INVALID_OR_DUPLICATE_PAYLOAD_PATH')
        names.add(str(relative))
        exported=Path(row.get('exported_source_path',row['source_path']))
        if not exported.is_absolute() or not exported.is_relative_to('/mnt/data'):raise ValueError('SOURCE8_EXPORTED_ALIAS_REQUIRED')
        source=(EXPORT/exported.relative_to('/mnt/data')).resolve()
        if not source.is_relative_to(EXPORT.resolve()) or source.stat().st_dev==root.stat().st_dev:raise ValueError('INDEPENDENT_SOURCE_FILESYSTEM_REQUIRED')
        target=staging/'payload'/relative
        def progress(count,phase='source_copy'):record(root,'PROGRESS.json',dict(utc=utc(),phase=phase,file_index=index,files_total=len(files),completed_files=len(results),current_file_bytes=count,completed_bytes=sum(x['bytes'] for x in results)))
        proof=copy_and_verify(source,target,row['expected_sha256'],row['bytes'],progress=progress)
        results.append(dict(payload_path=str(relative),source_path=row['source_path'],exported_source_path=str(source),roles=row['roles'],**proof))
        record(root,'FILES.json',results)
    if digest(inventory_path)!=intent['inventory_sha256']:raise ValueError('INVENTORY_CHANGED_BEFORE_PUBLICATION')
    with (staging/'inventory.json').open('xb') as f:f.write(inventory_path.read_bytes());f.flush();os.fsync(f.fileno())
    for directory in sorted([staging]+[p for p in staging.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
        fd=os.open(directory,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    final=root/'checkpoint-closure';os.rename(staging,final)
    result=dict(status='INDEPENDENT_INVENTORY_FULL_SHA_PASS',utc=utc(),run_id=plan['run_id'],step=plan['step'],files=len(results),bytes=sum(x['bytes'] for x in results),inventory_sha256=intent['inventory_sha256'],final_directory=str(final),semantic_reload_performed=False,all_experiment_backup_complete=False)
    once(root/'RESULT.json',result);return result


def launch(root,inventory_path,inventory_sha,run_id):
    root=validate_environment(Path(root))
    if root.exists():return dict(action='inspect existing intent/receipt; never replay',existing_root=str(root))
    if digest(inventory_path)!=inventory_sha:raise ValueError('INVENTORY_SHA_DIFFERS')
    plan=json.loads(Path(inventory_path).read_text())
    if plan.get('run_id')!=run_id:raise ValueError('WRONG_RUN_INVENTORY')
    root.mkdir(mode=0o700);source=Path(__file__).read_bytes();script=root/'copy-worker.py'
    with script.open('xb') as f:f.write(source);f.flush();os.fsync(f.fileno())
    once(root/'INTENT.json',dict(run_id=run_id,inventory_path=str(Path(inventory_path).resolve()),inventory_sha256=inventory_sha,worker_source_sha256=hashlib.sha256(source).hexdigest(),rate_bytes_per_second=RATE,ordinary_reserve_bytes=RESERVE,created_utc=utc()))
    with (root/'worker.log').open('xb') as log:
        child=subprocess.Popen(['nice','-n','19',sys.executable,str(script),'--root',str(root),'--worker'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
    ticks=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    receipt=dict(pid=child.pid,start_ticks=ticks,started_utc=utc(),gpu_use=False)
    once(root/'LAUNCH.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--inventory');p.add_argument('--inventory-sha');p.add_argument('--run-id');a=p.parse_args()
    if a.worker:
        try:result=worker(a.root)
        except Exception as error:
            record(Path(a.root),'ERROR.json',dict(utc=utc(),error=repr(error),status='NOT_COMPLETE'));raise
    else:result=launch(a.root,a.inventory,a.inventory_sha,a.run_id)
    print(json.dumps(result))


if __name__=='__main__':main()
