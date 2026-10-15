"""Build a larger frozen-teacher prefix dataset on the A800 host.

One immutable deployment owns one teacher invocation. A failed or unknown
launch is observed through the same handle; this operator never launches PPO,
fits a probe, stops foreign processes, or silently regenerates completed data.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
OPS = Path(__file__).resolve().parents[2]
SOURCE = 'fcb77204f48d5cb8ec21c68df7d4a2c937b3557a'
FORMAL = ROOT/'preparation/qprobe-qwen-native-selection-metrics-fix-deployment.json'
DIAGNOSTIC = ROOT/'preparation/qprobe-native-new512-diagnostic-deployment-20261016.json'
HUMAN = ROOT/'preparation/qprobe-prefix-scale20-dataset-human-intent-20261016.json'
PYTHON = '/data/search-g1/env/bin/python'
PREFIX = '/data/shaoshitong/Search-G1-a800-fresh200-'
FILES = ('a800_qprobe_prefix_scale20.py', 'qprobe_prefix_teacher_scale20.py',
         'a800_qprobe_native_new512_diagnostic.py', 'qprobe_prefix_correctness_features.py',
         'qprobe_prefix_scale20_encode.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def route():
    spec = importlib.util.spec_from_file_location('scale20_existing_route',
        OPS/'reproduction/operations/a800_qprobe_flex_observer.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.runtime_module(ROOT/'qprobe-qwen-native-selection-metrics-fix', read(FORMAL))


def publication():
    commit = subprocess.check_output(['git', '-C', str(OPS), 'rev-parse', 'HEAD'], text=True).strip()
    mapping = {}
    for name in FILES:
        rel = 'reproduction/operations/'+name
        raw = subprocess.check_output(['git', '-C', str(OPS), 'show', commit+':'+rel])
        assert raw == (OPS/rel).read_bytes(), 'Publish exact source first: '+name
        mapping[name] = hashlib.sha256(raw).hexdigest()
    actual = subprocess.check_output(['git', '-C', str(OPS), 'ls-remote', 'origin',
                                     'refs/heads/search-g1'], text=True).split()[0]
    assert actual == commit, 'Latest source has not reached the remote branch'
    return commit, mapping


COMMON = r'''
import base64,csv,hashlib,io,json,os,pathlib,subprocess,sys,time
p=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base']);repo=b/'repo'
assert str(b).startswith('/data/shaoshitong/Search-G1-a800-fresh200-') and b.resolve()==b
assert len(b.name.rsplit('-',1)[-1])==32
def read(q):return json.loads(pathlib.Path(q).read_text())
def digest(q):return hashlib.sha256(pathlib.Path(q).read_bytes()).hexdigest()
def once(q,v):
 with pathlib.Path(q).open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
def birth(pid):
 q=pathlib.Path('/proc')/str(pid);s=(q/'stat').read_text();v=s[s.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(v[19]),sid=int(v[3]),uid=q.stat().st_uid)
def alive(q):
 try:
  a=birth(q['pid']);s=(pathlib.Path('/proc')/str(q['pid'])/'stat').read_text()
  return a==q and s[s.rfind(')')+2:].split()[0] not in ('Z','X')
 except (FileNotFoundError,ProcessLookupError):return False
def source():
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==h['runtime_source']
 assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip()
 for name,value in h['operator_sha256'].items():assert digest(b/name)==value
 assert read(b/'intent.json')==h
'''

STAGE = COMMON+r'''
assert not b.exists();b.mkdir(mode=0o700)
for name in ('logs','tmp','cache/hf','cache/triton','cache/inductor'):(b/name).mkdir(parents=True,exist_ok=True)
assert subprocess.check_output(['git','-C',h['source_runtime_repo'],'rev-parse','HEAD'],text=True).strip()==h['runtime_source']
subprocess.run(['git','clone','--quiet','--no-hardlinks',h['source_runtime_repo'],str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','--quiet','--detach',h['runtime_source']],check=True)
for name,value in p['files'].items():
 assert name in h['operator_sha256']
 raw=base64.b64decode(value);assert hashlib.sha256(raw).hexdigest()==h['operator_sha256'][name]
 with (b/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
once(b/'intent.json',h);once(b/'human-intent.json',p['human']);source()
import shutil
free=shutil.disk_usage(b).free
assert free>=600*1024**3, 'Remote capacity insufficient for sealed prefix features'
args=['/data/search-g1/env/bin/python',str(b/'qprobe_prefix_teacher_scale20.py'),
 '--action','stage','--runtime-repo',str(repo),'--formal-base',h['formal_base'],
 '--output-dir',str(b),'--dataset-dir',h['dataset_dir'],
 '--raw-policy','compact-after-verified','--execution-profile','fast',
 '--train-indices',','.join(map(str,h['preferred_training_indices']))]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo))
with (b/'logs/stage.log').open('xb') as log:
 result=subprocess.run(args,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
assert result.returncode==0, 'Dataset stage failed; inspect remote stage.log without retrying launch'
v=dict(stage='PREFIX_SCALE20_STAGED',checked_unix=time.time(),base=str(b),
 operator_commit=h['operator_commit'],runtime_source=h['runtime_source'],remote_free_bytes=free,
 train_questions=102400,total_trajectories=512000,reused_trajectories=25600,
 new_trajectories=486400,formal_training_started=False,head_fitting_started=False,
 producer_stage_sha256=digest(b/'stage-complete.json'))
once(b/'logs/staged.json',v);print(json.dumps(v),flush=True)
'''

LAUNCH = COMMON+r'''
source();assert read(b/'logs/staged.json')['new_trajectories']==486400
assert not (b/'logs/launch-once.json').exists() and not (b/'logs/launch.json').exists()
rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))))
rows={int(x[0]):[q.strip() for q in x] for x in rows}
apps=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True))))
busy={x[0].strip() for x in apps if x}
def idle(index):
 g=rows[index];return int(g[2])<=64 and int(g[3])<=5 and g[1] not in busy
selected=h['preferred_training_indices']
assert len(selected)==len(set(selected))==4 and not set(selected).intersection((5,6,7))
assert all(idle(index) for index in selected+[5,6]), 'Assigned training or auxiliary GPUs are currently busy'
once(b/'logs/launch-once.json',dict(claimed_unix=time.time(),training_indices=selected,
 training_uuids=[rows[index][1] for index in selected],auxiliary_indices=[5,6],
 operator_commit=h['operator_commit'],actor_training_budget=0,head_training_budget=0))
env=dict(os.environ)
for key in tuple(env):
 if key.startswith('SLURM_') or key in ('PYTHONHOME','RAY_ADDRESS','LD_LIBRARY_PATH','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(repo),PATH='/data/search-g1/env/bin:'+env.get('PATH','/usr/bin:/bin'),
 PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
 TMPDIR=str(b/'tmp'),HF_HOME=str(b/'cache/hf'),XDG_CACHE_HOME=str(b/'cache'),
 TRITON_CACHE_DIR=str(b/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(b/'cache/inductor'))
args=['/data/search-g1/env/bin/python',str(b/'a800_qprobe_prefix_scale20.py'),'--worker','--handle',str(b/'intent.json')]
with (b/'logs/worker.log').open('xb') as log:
 child=subprocess.Popen(args,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(stage='PREFIX_SCALE20_LAUNCHED',process=birth(child.pid),training_indices=selected,
 launched_unix=time.time(),base=str(b),actor_training_budget=0,head_training_budget=0,no_automatic_retry=True)
once(b/'logs/launch.json',v);print(json.dumps(v),flush=True)
'''

OBSERVE = COMMON+r'''
source();docs={}
for name in ('logs/staged.json','logs/launch.json','logs/status.json','logs/exit.json',
 'teacher-exit.json','teacher-failure.json','teacher-complete.json','diagnostic-driver-route.json',
 'diagnostic-driver-exit.json','diagnostic-driver-failure.json','dataset/progress.json',
 'dataset/complete.json','features/complete.json'):
 q=b/name
 if q.exists():
  assert q.stat().st_size<=128*1024, 'Observation document exceeds small metadata bound'
  docs[name]=read(q)
if 'logs/launch.json' in docs:docs['wrapper_alive']=alive(docs['logs/launch.json']['process'])
if 'logs/status.json' in docs and 'child' in docs['logs/status.json']:
 docs['child_alive']=alive(docs['logs/status.json']['child'])
encoders=[]
for item in docs.get('logs/status.json',{}).get('encoder_children',[]):
 worker=b/'features'/('worker-%02d'%item['worker'])
 row=dict(item,alive=alive(item['process']))
 for name in ('worker-complete.json','worker-failure.json'):
  q=worker/name
  if q.exists():
   value=read(q);value.pop('shards',None);value.pop('reused_cache',None);row[name]=value
 q=worker/'progress.jsonl'
 if q.exists():
  with q.open('rb') as f:f.seek(max(0,q.stat().st_size-8000));tail=f.read()
  values=[]
  for line in tail.splitlines():
   try:values.append(json.loads(line))
   except ValueError:pass
  row['latest_progress']=values[-1] if values else None
 encoders.append(row)
docs['encoders']=encoders
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
errors=[];markers=[]
for name in ('teacher.log','worker.log','stage.log'):
 q=b/'logs'/name
 if not q.exists():continue
 with q.open('rb') as f:f.seek(max(0,q.stat().st_size-16000));tail=f.read()
 for line in tail.decode(errors='replace').splitlines():
  if line.startswith(('AssertionError','ValueError','RuntimeError','ImportError','ModuleNotFoundError','TypeError','KeyError','AttributeError','torch.cuda.OutOfMemoryError')):
   errors.append(dict(log=name,error_class=line.split(':',1)[0]))
  try:
   x=json.loads(line)
   if isinstance(x,dict) and 'stage' in x:
    markers.append({k:v for k,v in x.items() if k in ('stage','completed_chunks','completed_questions','completed_trajectories','new_trajectories','elapsed_seconds','checked_unix','observed_unix')})
  except ValueError:pass
print(json.dumps(dict(stage='PREFIX_SCALE20_OBSERVATION',checked_unix=time.time(),metadata=docs,
 latest_markers=markers[-3:],error_classes=errors[-4:],gpu_inventory=gpu)),flush=True)
'''


def worker(handle_path):
    h = read(handle_path); b = Path(h['base']); repo = b/'repo'
    assert Path(handle_path) == b/'intent.json' and h['runtime_source'] == SOURCE
    for name, value in h['operator_sha256'].items():
        assert sha(b/name) == value
    status = dict(stage='PREFIX_SCALE20_WORKER',started_unix=time.time(),
                  actual_actor_optimizer_updates=0,actual_head_optimizer_updates=0)

    def publish():
        path = b/'logs/status.json'; temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status,sort_keys=True,allow_nan=False)+'\n'); temporary.replace(path)

    try:
        args = [PYTHON,str(b/'qprobe_prefix_teacher_scale20.py'),'--action','teacher',
                '--runtime-repo',str(repo),'--formal-base',h['formal_base'],'--output-dir',str(b)]
        status.update(stage='GENERATING_FROZEN_TEACHER_DATASET'); publish()
        with (b/'logs/teacher.log').open('xb') as log:
            child = subprocess.Popen(args,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            info = Path('/proc')/str(child.pid); text = (info/'stat').read_text(); fields = text[text.rfind(')')+2:].split()
            status['child'] = dict(pid=child.pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=info.stat().st_uid)
            publish(); code = child.wait()
        status.update(teacher_exit_code=code); publish()
        assert code == 0, 'Teacher dataset process exited; preserve completed chunks and inspect evidence'
        complete = read(Path(h['dataset_dir'])/'complete.json')
        assert complete.get('questions',complete.get('question_count'))==102400 and complete['trajectories']==512000
        status.update(stage='PREFIX_SCALE20_TEACHER_DATASET_COMPLETE',
                      complete_sha256=sha(Path(h['dataset_dir'])/'complete.json'))
        publish()
        encode_dataset(h,status,publish)
        status.update(stage='PREFIX_SCALE20_DATASET_AND_FEATURES_COMPLETE',finished_unix=time.time(),exit_code=0)
    except BaseException as error:
        status.update(stage='PREFIX_SCALE20_EXITED',error=repr(error),exit_code=1,finished_unix=time.time())
        raise
    finally:
        publish(); once(b/'logs/exit.json',status)
    print(json.dumps(status),flush=True)


def encode_dataset(h,status,publish):
    """Encode sealed new shards after teacher cleanup, without a probe fit."""
    import csv
    import io
    b = Path(h['base']); repo = b/'repo'; root = Path(h['dataset_dir'])
    compact = read(root/'compact-manifest.json')
    assert compact['train_trajectories']==512000 and compact['new_trajectories']==486400
    assert compact['reuse']['train_trajectories']==25600 and compact['reuse']['test_trajectories']==2560
    rows = list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))))
    rows = {int(row[0]):[x.strip() for x in row] for row in rows}
    apps = list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True))))
    busy = {row[0].strip() for row in apps if row}
    selected = h['preferred_training_indices']
    assert all(int(rows[i][2])<=64 and int(rows[i][3])==0 and rows[i][1] not in busy for i in selected), \
        'Teacher dataset is complete; encoder cards are busy, preserve data and wait for actual idle admission'
    output = b/'features'; output.mkdir(mode=0o700,exist_ok=False)
    once(output/'encoding-admission.json',dict(physical_indices=selected,
        gpu_uuids=[rows[i][1] for i in selected],checked_unix=time.time(),foreign_processes_modified=False))
    processes = []
    status.update(stage='ENCODING_NEW_PREFIX_FEATURES',encoder_children=[]);publish()
    try:
        for ordinal,index in enumerate(selected):
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=rows[index][1])
            args = [PYTHON,str(b/'qprobe_prefix_scale20_encode.py'),'--runtime-repo',str(repo),
                '--input-dir',str(root),'--output-dir',str(output),'--compact-manifest-sha256',sha(root/'compact-manifest.json'),
                '--shard-index',str(ordinal),'--world-size','4','--physical-index',str(index),
                '--micro-batch-size','128','--token-budget','65536']
            log = (b/'logs'/('encoder-%02d.log'%ordinal)).open('xb')
            child = subprocess.Popen(args,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            log.close()
            info = Path('/proc')/str(child.pid); text = (info/'stat').read_text(); fields = text[text.rfind(')')+2:].split()
            birth = dict(pid=child.pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=info.stat().st_uid)
            processes.append((child,birth));status['encoder_children'].append(dict(worker=ordinal,physical_index=index,process=birth));publish()
        for child,birth in processes:
            assert child.wait()==0, 'A prefix encoder failed; sealed data and completed feature shards are retained'
        workers = [read(output/('worker-%02d'%i)/'worker-complete.json') for i in range(4)]
        assert sum(item['completed_trajectories'] for item in workers)==486400
        shards = [item for worker in workers for item in worker['shards']]
        assert len(shards)==190 and len({item['ordinal'] for item in shards})==190
        for worker in workers:
            assert worker['compact_manifest_sha256']==sha(root/'compact-manifest.json')
            assert worker['reused_features_reencoded'] is False and worker['actor_updated'] is False and worker['probe_trained'] is False
        for item in shards:
            assert sha(Path(item['path'])/'manifest.json')==item['manifest_sha256']
        summary = dict(stage='PREFIX_SCALE20_COMPOSITE_FEATURES_COMPLETE',train_questions=102400,
            train_trajectories=512000,new_trajectories=486400,reused_train_trajectories=25600,
            fixed_test_trajectories=2560,reuse=compact['reuse'],spec=compact['spec'],shards=sorted(shards,key=lambda item:item['ordinal']),
            compact_manifest_sha256=sha(root/'compact-manifest.json'),no_large_feature_consolidation=True,
            actual_actor_optimizer_updates=0,actual_head_optimizer_updates=0,checked_unix=time.time())
        once(output/'composite-manifest.json',summary)
        once(output/'complete.json',dict(stage=summary['stage'],train_questions=102400,train_trajectories=512000,
            fixed_test_trajectories=2560,composite_manifest_sha256=sha(output/'composite-manifest.json'),
            all_four_encoder_processes_exited=True,actual_actor_optimizer_updates=0,actual_head_optimizer_updates=0,
            checked_unix=time.time()))
    finally:
        from reproduction.a800.owned_processes import same
        for child,birth in processes:
            if child.poll() is None and same(birth):
                child.terminate()
        for child,birth in processes:
            if child.poll() is None:
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if same(birth):child.kill();child.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('stage','run-once','observe'))
    parser.add_argument('--handle',required=True); parser.add_argument('--worker',action='store_true')
    parser.add_argument('--train-indices',default='0,2,3,4')
    args = parser.parse_args()
    if args.worker:
        worker(args.handle); return
    path = Path(args.handle)
    if args.action == 'stage':
        assert not path.exists()
        commit,mapping = publication(); diagnostic = read(DIAGNOSTIC); human = read(HUMAN)
        indices = [int(x) for x in args.train_indices.split(',')]
        assert len(indices)==len(set(indices))==4 and set(indices)<=set(range(5))
        base = PREFIX+uuid.uuid4().hex
        h = dict(role='frozen_teacher_scale20_prefix_dataset_only',base=base,dataset_dir=base+'/dataset',
            runtime_repo=base+'/repo',source_runtime_repo=diagnostic['runtime_repo'],runtime_source=SOURCE,
            operator_commit=commit,operator_sha256=mapping,human_intent_sha256=sha(HUMAN),
            formal_base='/data/shaoshitong/Search-G1-a800-fresh200-e9c4c584107b4624bcc9787aaf24b7a2',
            preferred_training_indices=indices,auxiliary_indices=[5,6],train_questions=102400,
            train_trajectories=512000,reused_train_questions=5120,new_train_questions=97280,
            actor_training_budget=0,head_training_budget=0,formal_training_remains_stopped=True,
            raw_policy='compact-after-verified',execution_profile='fast',created_unix=time.time(),
            no_automatic_retry=True)
        once(path,h)
        payload = dict(handle=h,human=human,files={name:base64.b64encode((OPS/'reproduction/operations'/name).read_bytes()).decode() for name in FILES})
        code = STAGE
    else:
        h = read(path); assert h['human_intent_sha256'] == sha(HUMAN)
        payload = dict(handle=h); code = LAUNCH if args.action=='run-once' else OBSERVE
    command = route().ssh_command(code)
    command[-1] = PYTHON+' -c '+command[-1][len('python3 -c '):]
    started = time.time()
    try:
        result = subprocess.run(command,input=json.dumps(payload).encode(),capture_output=True,
                                timeout=370 if args.action=='stage' else 65)
        value = dict(action=args.action,return_code=result.returncode,started_unix=started,
            finished_unix=time.time(),stdout=result.stdout.decode(errors='replace'),
            stderr_sha256=hashlib.sha256(result.stderr).hexdigest())
        if result.returncode:
            value['stderr_tail'] = result.stderr.decode(errors='replace')[-1500:]
    except subprocess.TimeoutExpired:
        value = dict(action=args.action,status='UNKNOWN_SSH_TIMEOUT',started_unix=started,
                     finished_unix=time.time(),retry_launch_forbidden=True)
    receipt = ROOT/'preparation'/('qprobe-prefix-scale20-'+args.action+'-'+str(time.time_ns())+'.json')
    once(receipt,value); print(json.dumps(dict(receipt=str(receipt),**value)),flush=True)
    if value.get('return_code',1):
        raise SystemExit(1)


if __name__=='__main__':
    main()
