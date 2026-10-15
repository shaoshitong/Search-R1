"""Stage immutable root-filesystem official inputs into the node8 NFS export.

This is a same-host staging copy, never an independent persistent backup or a
semantic reload. Originals, active training and its source tree are untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

RATE=20*1024**2
RESERVE=128*1024**3


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value,exclusive=False):
    path=Path(path);temporary=path if exclusive else path.with_suffix('.tmp')
    with temporary.open('x' if exclusive else 'w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    if not exclusive:os.replace(temporary,path)


def full_read(path,output=None,*,rate=RATE,progress=None):
    path=Path(path);before=path.stat();identity=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
    sha=hashlib.sha256();count=0;start=time.monotonic();reported=start
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):
            if output is not None:output.write(block)
            sha.update(block);count+=len(block)
            if rate:time.sleep(max(0,count/rate-(time.monotonic()-start)))
            if progress and time.monotonic()-reported>=10:progress(count);reported=time.monotonic()
    after=path.stat()
    if identity!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) or count!=before.st_size:
        raise ValueError('Source changed during complete read')
    if progress:progress(count)
    return count,sha.hexdigest()


def copy_file(source,destination,expected_sha,*,rate=RATE,progress=None):
    with Path(destination).open('xb') as output:
        count,sha=full_read(source,output,rate=rate,progress=lambda n:progress('copy_source_full_read',n) if progress else None)
        output.flush();os.fsync(output.fileno())
    if sha!=expected_sha:raise ValueError('Official source full SHA differs')
    size,target_sha=full_read(destination,rate=rate,progress=lambda n:progress('same_host_staged_full_read',n) if progress else None)
    if size!=count or target_sha!=sha:raise ValueError('Staged complete read differs')
    return dict(bytes=count,expected_sha256=expected_sha,source_full_read_sha256=sha,staged_full_read_sha256=target_sha)


def worker(base,root):
    os.nice(19);base=Path(base).resolve();root=Path(root).resolve();intent=json.loads((root/'INTENT.json').read_text())
    if intent['worker_source_sha256']!=digest(__file__):raise ValueError('Worker differs from published intent')
    run=json.loads((base/'logs/run-manifest.json').read_text())
    if run['run_id']!=intent['run_id'] or run['source_commit']!=intent['frozen_training_source']:raise ValueError('Source run differs')
    source=Path(run['initial_model']).resolve();expected=run['initial_model_sha256']
    if str(source)!=intent['source_model'] or expected!=intent['files_sha256']:raise ValueError('Official inputs differ')
    partial=root/'model.partial';partial.mkdir();rows=[]
    try:
        for name,sha in sorted(expected.items()):
            if Path(name).name!=name:raise ValueError('Unsafe official file name')
            src=source/name;dst=partial/name;fs=os.statvfs(root)
            if fs.f_bavail*fs.f_frsize<src.stat().st_size+RESERVE:raise ValueError('Ordinary staging reserve exhausted')
            def progress(phase,count):write(root/'PROGRESS.json',dict(pid=os.getpid(),run_id=run['run_id'],completed_files=len(rows),file=name,phase=phase,current_bytes=count,unix=time.time(),independent_backup=False))
            row=copy_file(src,dst,sha,progress=progress)
            rows.append(dict(row,source_path=str(src),staged_path=str(root/'model'/name)))
        fd=os.open(partial,os.O_RDONLY);os.fsync(fd);os.close(fd)
        os.rename(partial,root/'model');fd=os.open(root,os.O_RDONLY);os.fsync(fd);os.close(fd)
        result=dict(status='SAME_HOST_OFFICIAL_MODEL_STAGE_FULL_SHA_PASS',run_id=run['run_id'],files=rows,
            bytes=sum(x['bytes'] for x in rows),source_model=str(source),staged_model=str(root/'model'),
            independent_backup=False,semantic_reload_performed=False,unix=time.time())
        write(root/'RESULT.json',result,True)
    except BaseException as error:
        write(root/'RESULT.json',dict(status='FAILED_PRESERVE_SOURCE_AND_PARTIAL',error=repr(error),run_id=run['run_id'],independent_backup=False,unix=time.time()),True);raise


def launch(base,expected_run):
    base=Path(base).resolve()
    if not str(base).startswith('/mnt/data/shaoshitong/search-g1-qprobe3-h1008-'):raise ValueError('Remote source namespace required')
    run=json.loads((base/'logs/run-manifest.json').read_text())
    if run['run_id']!=expected_run or run['model_revision']!='aa8e72537993ba99e69dfaafa59ed015b17504d1' or run['initial_model']!=run['reference_model']:
        raise ValueError('Official fresh experiment identity differs')
    root=base/'backup-dependencies/official-model-once-v1';root.parent.mkdir(exist_ok=True)
    if root.exists():
        print(json.dumps(dict(action='inspect existing outcome; no replay',root=str(root),result=json.loads((root/'RESULT.json').read_text()) if (root/'RESULT.json').exists() else None)));return
    need=sum((Path(run['initial_model'])/name).stat().st_size for name in run['initial_model_sha256']);fs=os.statvfs(base)
    if fs.f_bavail*fs.f_frsize<need+RESERVE:raise ValueError('Insufficient ordinary staging capacity')
    root.mkdir()
    write(root/'INTENT.json',dict(run_id=run['run_id'],frozen_training_source=run['source_commit'],source_model=str(Path(run['initial_model']).resolve()),files_sha256=run['initial_model_sha256'],worker_source_sha256=digest(__file__),rate=RATE,reserve=RESERVE,independent_backup=False,unix=time.time()),True)
    with (root/'worker.log').open('xb') as log:
        child=subprocess.Popen(['/usr/bin/python3',str(Path(__file__).resolve()),'--worker','--base',str(base),'--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    stat=Path('/proc')/str(child.pid)/'stat';ticks=stat.read_text().rsplit(')',1)[1].split()[19]
    receipt=dict(pid=child.pid,start_ticks=ticks,root=str(root),gpu_use=False,independent_backup=False,unix=time.time());write(root/'LAUNCH.json',receipt,True);print(json.dumps(receipt))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--expected-run');p.add_argument('--worker',action='store_true');p.add_argument('--root');a=p.parse_args()
    if a.worker:worker(a.base,a.root)
    else:launch(a.base,a.expected_run)

if __name__=='__main__':main()
