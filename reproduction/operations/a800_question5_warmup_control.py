"""Stage, launch once and observe the fresh merged question+5 EMA probe job.

Uses only the existing A800 physical SSH route. All inputs/features/checkpoints
stay on that host; the Mac keeps source and bounded metadata only. No actor,
teacher generation, retriever, global Ray operation or foreign signal is used.
"""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
REPO = ROOT/'repo'
PYTHON = '/data/search-g1/env/bin/python'


def route(code):
    path = Path(__file__).with_name('a800_qprobe_prefix_scale20.py')
    spec = importlib.util.spec_from_file_location('existing_question5_route', path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.route().ssh_command(code)


COMMON = r'''
import csv,hashlib,io,json,os,pathlib,signal,socket,subprocess,sys,time
p=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base']);repo=b/'repo'
assert str(b).startswith('/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-')
assert b.resolve()==b and len(b.name.rsplit('-',1)[-1])==32
def read(path):return json.loads(pathlib.Path(path).read_text())
def sha(path):
 d=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):d.update(x)
 return d.hexdigest()
def write(path,value,exclusive=False):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 target=path if exclusive else path.with_name(path.name+'.partial')
 with target.open('x' if exclusive else 'w') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
 if not exclusive:os.replace(target,path)
def birth(pid):
 q=pathlib.Path('/proc')/str(pid);s=(q/'stat').read_text();v=s[s.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(v[19]),sid=int(v[3]),uid=q.stat().st_uid)
def alive(x):
 try:
  s=(pathlib.Path('/proc')/str(x['pid'])/'stat').read_text();return birth(x['pid'])==x and s[s.rfind(')')+2:].split()[0] not in ('Z','X')
 except (FileNotFoundError,ProcessLookupError):return False
def source():
 assert read(b/'preparation/deployment.json')==h
 for name,digest in h['source_files'].items():
  q=repo/name;assert q.resolve()==q and sha(q)==digest,'Deployed source mismatch: '+name
 assert sha(b/'preparation/source-plan.json')==h['source_plan_sha256']
'''

STAGE = COMMON+r'''
import base64,tarfile
assert not (b/'preparation/deployment.json').exists()
b.mkdir(mode=0o700,exist_ok=True);(b/'logs').mkdir(exist_ok=True);(b/'preparation').mkdir(exist_ok=True)
archive=base64.b64decode(p['source_archive_b64']);assert hashlib.sha256(archive).hexdigest()==h['source_archive_sha256']
repo.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:*') as tar:
 for member in tar:
  path=repo/member.name
  assert not pathlib.PurePosixPath(member.name).is_absolute() and '..' not in pathlib.PurePosixPath(member.name).parts
  assert path.resolve().is_relative_to(repo) and not member.issym() and not member.islnk()
  if member.isdir():path.mkdir(parents=True,exist_ok=True)
  else:
   assert member.isfile() and member.size<=64*1024**2
   path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('xb') as f:f.write(tar.extractfile(member).read());f.flush();os.fsync(f.fileno())
for name,value in p['side_files'].items():
 path=b/'preparation'/name;assert path.parent==b/'preparation'
 raw=base64.b64decode(value)
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
write(b/'preparation/deployment.json',h,True);source()
v=dict(stage='QUESTION5_EXACT_SOURCE_ARCHIVE_STAGED',checked_unix=time.time(),source_commit=h['source_commit'],
 source_tree=h['source_tree'],source_files_verified=len(h['source_files']),source_archive_sha256=h['source_archive_sha256'],
 base=str(b),teacher_or_actor_started=False,training_started=False)
write(b/'preparation/source-receipt.json',v,True);print(json.dumps(v))
'''

LAUNCH = COMMON+r'''
source();assert not (b/'logs/launch.json').exists() and not (b/'logs/launch-once.json').exists()
write(b/'logs/launch-once.json',dict(run_id=h['run_id'],execution_id=h['execution_id'],claimed_unix=time.time()),True)
env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='',
 OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HOME=str(b/'cache/hf'),
 XDG_CACHE_HOME=str(b/'cache'),TMPDIR=str(b/'tmp'))
for name in ('tmp','cache/hf','cache/triton','cache/inductor'):(b/name).mkdir(parents=True,exist_ok=True)
for k in tuple(env):
 if k.startswith('SLURM_') or k in ('RAY_ADDRESS','PYTHONHOME','LD_LIBRARY_PATH'):env.pop(k,None)
command=['/data/search-g1/env/bin/python',str(repo/'reproduction/operations/a800_question5_warmup_control.py'),
 '--worker','--handle',str(b/'preparation/deployment.json')]
with (b/'logs/wrapper.log').open('xb') as log:
 child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(stage='QUESTION5_PIPELINE_LAUNCHED_ONCE',process=birth(child.pid),run_id=h['run_id'],execution_id=h['execution_id'],
 source_commit=h['source_commit'],checked_unix=time.time(),training_progress_proven=False)
write(b/'logs/launch.json',v,True);print(json.dumps(v))
'''

OBSERVE = COMMON+r'''
source();docs={}
names=['logs/launch.json','logs/status.json','logs/exit.json','logs/failure.json','logs/gpu-binding.json',
 'data/PREPARE-COMPLETE.json','features/cache-complete.json','features/complete.json','features/progress.json',
 'train/status.json','train/complete.json','train/failure.json','train/latest-metrics.json','train/latest.json','train/run-claim.json']
for name in names:
 q=b/name
 if q.exists():
  assert q.stat().st_size<=256*1024
  docs[name]=read(q)
launch=docs.get('logs/launch.json');docs['wrapper_alive']=alive(launch['process']) if launch else None
jobs=[]
for job in docs.get('logs/status.json',{}).get('jobs',[]):
 row=dict(job,alive=alive(job['process']));jobs.append(row)
progress=[]
for q in sorted((b/'train').glob('**/progress-rank-*.json'))[-8:]:
 if q.stat().st_size<=64*1024:progress.append(dict(path=str(q.relative_to(b)),value=read(q)))
import re
errors=[]
for q in sorted((b/'logs').glob('*.log')):
 with q.open('rb') as f:f.seek(max(0,q.stat().st_size-16384));tail=f.read(16384).decode(errors='replace')
 classes=re.findall(r'^\s*((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception))(?::|$)',tail,re.M)
 frames=[dict(file=pathlib.Path(name).name,line=int(line)) for name,line in re.findall(r'File "([^"\n]+)", line (\d+)',tail)]
 if classes:errors.append(dict(log=q.name,error_classes=classes[-3:],frames=frames[-5:]))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
v=dict(stage='QUESTION5_BOUNDED_EXISTING_METADATA_OBSERVATION',checked_unix=time.time(),run_id=h['run_id'],
 execution_id=h['execution_id'],source_commit=h['source_commit'],metadata=docs,jobs=jobs,rank_progress=progress,gpu_samples=gpu,error_metadata=errors,
 teacher_or_actor_updates=0,read_only=True)
raw=json.dumps(v,allow_nan=False);assert len(raw.encode())<=128*1024;print(raw)
'''


def worker(handle):
    import csv
    import fcntl
    import signal
    import socket
    import traceback
    h=json.loads(handle.read_text());b=Path(h['base']);repo=b/'repo'
    jobs=[];children=[];leases=[]
    def birth(pid):
        q=Path('/proc')/str(pid);s=(q/'stat').read_text();v=s[s.rfind(')')+2:].split()
        return dict(pid=pid,start_ticks=int(v[19]),sid=int(v[3]),uid=q.stat().st_uid)
    def publish(path,value):
        target=path.with_name(path.name+'.partial');target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('w') as f:json.dump(value,f,allow_nan=False,sort_keys=True);f.flush();os.fsync(f.fileno())
        os.replace(target,path)
    def status(stage,**extra):
        publish(b/'logs/status.json',dict(stage=stage,checked_unix=time.time(),run_id=h['run_id'],
            execution_id=h['execution_id'],source_commit=h['source_commit'],jobs=jobs,**extra))
    env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    def start(name,args,child_env=None,pass_fds=()):
        log=(b/'logs'/(name+'.log')).open('xb')
        child=subprocess.Popen([PYTHON]+args,cwd=repo,env=child_env or env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=pass_fds);log.close();children.append(child)
        row=dict(name=name,process=birth(child.pid),exit_code=None);jobs.append(row)
        status('CHILD_STARTED',current_job=name);return child,row
    def wait(child,row):
        code=child.wait();row['exit_code']=code;status('CHILD_EXITED',current_job=row['name'])
        if code:raise RuntimeError('Owned child exited: '+row['name']+' code='+str(code))
    def stop(signum=None,_frame=None):
        for child,row in zip(children,jobs):
            if child.poll() is None:
                try:
                    if birth(child.pid)==row['process'] and row['process']['sid']==child.pid:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        if signum:raise InterruptedError('Human/native signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for rel,digest in h['source_files'].items():
            assert hashlib.sha256((repo/rel).read_bytes()).hexdigest()==digest
        plan=json.loads((b/'preparation/source-plan.json').read_text())
        while True:
            parts=plan['reference_archive']['parts']
            ready=[x for x in parts if Path(x['path']).is_file() and Path(x['path']).with_name(Path(x['path']).name+'.verified.json').is_file()]
            if len(ready)==len(parts):break
            status('WAITING_IMMUTABLE_REFERENCE_ARCHIVE',verified_parts=len(ready),expected_parts=len(parts))
            time.sleep(60)
        status('CPU_PREPARING_MERGED_ORIGINAL_PREFIX5_EVENTS')
        child,row=start('prepare',['-m','reproduction.question5_ema_warmup.prepare','--config',str(b/'preparation/source-plan.json'),'--output',str(b/'data')]);wait(child,row)
        # No GPU is requested until the accepted CPU data manifest exists.
        manifest=b/'data/composite-manifest.json';assert manifest.is_file()
        while True:
            rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15))))
            apps=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15))))
            busy={x[0].strip() for x in apps if x};eligible=[]
            for index,uuid,memory,utilization in rows:
                if int(index) not in (5,6) and int(memory)<=64 and int(utilization)<=5 and uuid.strip() not in busy:eligible.append((int(index),uuid.strip()))
            selected=eligible[:4]
            if len(selected)==4:
                root=Path('/data/shaoshitong/Search-G1-qprobe-shared/token-feature-gpu-locks');root.mkdir(parents=True,exist_ok=True)
                try:
                    for _,uuid in selected:
                        f=(root/(uuid+'.lock')).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);leases.append(f)
                    break
                except BlockingIOError:
                    for f in leases:f.close()
                    leases=[]
            status('WAITING_FOUR_CURRENTLY_IDLE_TRAINING_GPUS',idle_count=len(eligible));time.sleep(1800)
        binding=dict(run_id=h['run_id'],execution_id=h['execution_id'],checked_unix=time.time(),indices=[x[0] for x in selected],
            uuids=[x[1] for x in selected],auxiliary_gpu_used=False,inventory=rows,foreign_tasks_signalled=False)
        publish(b/'logs/gpu-binding.json',binding);status('BUILDING_ONCE_FROZEN_PREFIX5_FEATURE_CACHE')
        active=[]
        for rank,(physical,uuid) in enumerate(selected):
            fd=leases[rank].fileno()
            child_env=dict(env,CUDA_VISIBLE_DEVICES=uuid,QUESTION5_GPU_LEASE_FD=str(fd))
            child,row=start('cache-%d'%rank,['-m','reproduction.question5_ema_warmup.cache','--action','build-cache',
                '--input-manifest',str(manifest),'--output-dir',str(b/'features'),'--model-dir',h['model_dir'],
                '--worker-index',str(rank),'--world-size','4','--physical-index',str(physical),
                '--micro-batch-size','128','--token-budget','65536'],child_env,pass_fds=(fd,));active.append((child,row))
        for child,row in active:wait(child,row)
        child,row=start('cache-finalize',['-m','reproduction.question5_ema_warmup.cache','--action','finalize-cache',
            '--input-manifest',str(manifest),'--output-dir',str(b/'features'),'--model-dir',h['model_dir'],
            '--worker-index','0','--world-size','4']);wait(child,row)
        cache=b/'features/manifest.json';assert cache.is_file()
        counts=json.loads(cache.read_text())['groups_by_split']
        assert counts==dict(train=445440,test=512),'Actual complete merged source count differs'
        fit=dict(h['fit_config'],feature_manifest=str(cache),feature_manifest_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
            model_dir=h['model_dir'],output_dir=str(b/'train'),run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],
            physical_gpu_uuids=[x[1] for x in selected],expected_train_groups=counts['train'],expected_test_groups=counts['test'])
        publish(b/'preparation/fit-config.json',fit);status('FORMAL_PROBE_WARMUP200_STARTING')
        child_env=dict(env,CUDA_VISIBLE_DEVICES=','.join(x[1] for x in selected),NCCL_NVLS_ENABLE='0')
        child,row=start('probe-train',['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',
            '-m','reproduction.question5_ema_warmup.train','--config',str(b/'preparation/fit-config.json')],child_env);wait(child,row)
        status('PROBE_WARMUP200_CHILD_RETURNED_ZERO')
        publish(b/'logs/exit.json',dict(exit_code=0,checked_unix=time.time(),run_id=h['run_id'],execution_id=h['execution_id'],
            training_complete_requires_native_200_metrics_and_checkpoint_audit=True))
    except BaseException as exc:
        stop()
        for child,row in zip(children,jobs):
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    if birth(child.pid)==row['process'] and row['process']['sid']==child.pid:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=30)
            row['exit_code']=child.returncode
        publish(b/'logs/failure.json',dict(error_class=type(exc).__name__,checked_unix=time.time(),run_id=h['run_id'],execution_id=h['execution_id']))
        publish(b/'logs/exit.json',dict(exit_code=1,checked_unix=time.time(),run_id=h['run_id'],execution_id=h['execution_id']))
        traceback.print_exc();raise
    finally:
        for f in leases:f.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['stage','launch','observe'])
    parser.add_argument('--handle',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--payload',type=Path)
    args=parser.parse_args()
    if args.worker:worker(args.handle);return
    h=json.loads(args.handle.read_text());payload=dict(handle=h)
    if args.action=='stage':payload.update(json.loads(args.payload.read_text()))
    code={'stage':STAGE,'launch':LAUNCH,'observe':OBSERVE}[args.action]
    command=route(code);started=time.time()
    result=subprocess.run(command,input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
    record=dict(action=args.action,checked_unix=time.time(),elapsed_seconds=time.time()-started,exit_code=result.returncode,
        stdout=result.stdout.decode(),stderr=result.stderr.decode(),source_commit=h['source_commit'],run_id=h['run_id'],execution_id=h['execution_id'])
    out=ROOT/'preparation'/('question5-ema-'+args.action+'-'+str(time.time_ns())+'.json');out.write_text(json.dumps(record,indent=2)+'\n')
    if result.returncode:raise RuntimeError('Action failed; retained receipt '+str(out)+': '+record['stderr'][:1000])
    assert len(result.stdout)<=128*1024
    print(json.dumps(dict(receipt=str(out),result=json.loads(result.stdout))))


if __name__=='__main__':main()
