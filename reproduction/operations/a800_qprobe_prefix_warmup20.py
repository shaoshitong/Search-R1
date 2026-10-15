"""Publish and run a balanced trajectory-prefix probe warmup; never starts PPO.

The worker reuses the completed prefix cache, then fits one correctness logit
with dynamically balanced batches for20 epochs and a fixed new512-question test.
All large inputs and outputs remain on the A800 host. Unknown launch outcomes
must be observed through the same handle, never retried automatically.
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
DIAGNOSTIC = ROOT/'preparation/qprobe-native-new512-diagnostic-deployment-20261016.json'
FORMAL = ROOT/'preparation/qprobe-qwen-native-selection-metrics-fix-deployment.json'
HUMAN = ROOT/'preparation/qprobe-prefix-correctness-balanced-warmup20-human-intent-20261016.json'
PREFIX = '/data/shaoshitong/Search-G1-qprobe-prefix-correctness-balanced20-'
PYTHON = '/data/search-g1/env/bin/python'
FILES = ('qprobe_prefix_correctness_features.py', 'qprobe_prefix_correctness_fit.py',
         'a800_qprobe_prefix_warmup20.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    path = Path(path)
    with path.open('x') as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n'); out.flush(); os.fsync(out.fileno())


def route():
    spec = importlib.util.spec_from_file_location('prefix_warmup_existing_route',
        OPS/'reproduction/operations/a800_qprobe_flex_observer.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.runtime_module(ROOT/'qprobe-qwen-native-selection-metrics-fix', read(FORMAL))


def publication():
    commit = subprocess.check_output(['git', '-C', str(OPS), 'rev-parse', 'HEAD'], text=True).strip()
    mapping = {}
    for name in FILES:
        rel = 'reproduction/operations/'+name
        raw = subprocess.check_output(['git', '-C', str(OPS), 'show', commit+':'+rel])
        assert raw == (OPS/rel).read_bytes(), 'Publish exact source before deployment: '+name
        mapping[name] = hashlib.sha256(raw).hexdigest()
    assert subprocess.check_output(['git', '-C', str(OPS), 'ls-remote', 'origin', 'refs/heads/search-g1'],
        text=True).split()[0] == commit, 'Exact operations source must be pushed first'
    return commit, mapping


COMMON = r'''
import base64,csv,hashlib,io,json,os,pathlib,subprocess,sys,time
p=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base']);repo=pathlib.Path(h['runtime_repo'])
assert str(b).startswith('/data/shaoshitong/Search-G1-qprobe-prefix-correctness-balanced20-')
assert len(b.name.rsplit('-',1)[-1])==32 and b.resolve()==b
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
for name,value in p['files'].items():
 assert name in h['operator_sha256']
 raw=base64.b64decode(value);assert hashlib.sha256(raw).hexdigest()==h['operator_sha256'][name]
 with (b/name).open('xb') as f:f.write(raw)
once(b/'intent.json',h);once(b/'human-intent.json',p['human']);source()
assert read(pathlib.Path(h['test_base'])/'teacher-exit.json')['exit_code']==0
assert read(pathlib.Path(h['test_base'])/'inference/new512-complete.json')['new_random512_test'] is True
feature_root=pathlib.Path(h['feature_output_dir'])
assert feature_root.resolve()==feature_root and str(feature_root).startswith('/data/shaoshitong/Search-G1-qprobe-prefix-correctness-warmup20-')
qualified=read(feature_root/'features-complete.json')
assert qualified['count']==28160 and qualified['G5_regenerated'] is False
assert qualified['manifest_sha256']==h['feature_manifest_sha256']==digest(feature_root/'cache/manifest.json')
assert qualified['actor_trained'] is False and qualified['head_trained'] is False
assert h['feature_cache_dir']==str(feature_root/'cache')
assert h['balanced_sampling'] is True and h['batch_class_counts']==dict(correct=128,incorrect=128)
assert digest(feature_root.parent/'qprobe_prefix_correctness_features.py')==h['operator_sha256']['qprobe_prefix_correctness_features.py']
import shutil
free=shutil.disk_usage(b).free
assert free>=80*1024**3, 'Need remote capacity for packed features and epoch checkpoints'
v=dict(stage='PREFIX_WARMUP20_STAGED',checked_unix=time.time(),base=str(b),source_commit=h['operator_commit'],
 runtime_source=h['runtime_source'],remote_free_bytes=free,formal_training_started=False,
 teacher_generation_repeated=False,prefix_encoding_repeated=False,feature_manifest_sha256=h['feature_manifest_sha256'],
 balanced_sampling=True,batch_class_counts=h['batch_class_counts'],train_questions=5120,test_questions=512,epochs=20,expected_head_adam=2000)
once(b/'logs/staged.json',v);print(json.dumps(v),flush=True)
'''

LAUNCH = COMMON+r'''
source();assert read(b/'logs/staged.json')['expected_head_adam']==2000
assert not (b/'logs/launch-once.json').exists() and not (b/'logs/launch.json').exists()
rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))))
rows={int(x[0]):[q.strip() for q in x] for x in rows};g=rows[h['physical_index']]
assert h['physical_index'] in (0,1,2,3,4) and int(g[2])<=64 and int(g[3])<=5, 'Selected GPU is busy'
gpu_uuid=g[1]
apps=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True))))
assert not [x for x in apps if x and x[0].strip()==gpu_uuid], 'Selected GPU has a compute process'
once(b/'logs/launch-once.json',dict(claimed_unix=time.time(),physical_index=h['physical_index'],gpu_uuid=gpu_uuid,
 source_commit=h['operator_commit'],formal_training_forbidden=True))
env=dict(os.environ)
for key in tuple(env):
 if key.startswith('SLURM_') or key in ('PYTHONHOME','RAY_ADDRESS','LD_LIBRARY_PATH','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
env.update(CUDA_VISIBLE_DEVICES=gpu_uuid,PYTHONPATH=str(repo),PATH='/data/search-g1/env/bin:'+env.get('PATH','/usr/bin:/bin'),
 PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
 TMPDIR=str(b/'tmp'),HF_HOME=str(b/'cache/hf'),XDG_CACHE_HOME=str(b/'cache'),
 TRITON_CACHE_DIR=str(b/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(b/'cache/inductor'))
args=['/data/search-g1/env/bin/python',str(b/'a800_qprobe_prefix_warmup20.py'),'--worker','--handle',str(b/'intent.json')]
with (b/'logs/worker.log').open('xb') as log:
 child=subprocess.Popen(args,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(stage='PREFIX_WARMUP20_LAUNCHED',process=birth(child.pid),gpu_uuid=gpu_uuid,physical_index=h['physical_index'],
 launched_unix=time.time(),formal_training_started=False,base=str(b),no_automatic_retry=True)
once(b/'logs/launch.json',v);print(json.dumps(v),flush=True)
'''

OBSERVE = COMMON+r'''
source();docs={}
def compact_epoch(v):
 out={k:v[k] for k in ('epoch','actual_head_adam_updates','sampling') if k in v}
 for split in ('train','test'):
  if split not in v:continue
  x=v[split];names=('samples','questions','positive_fraction','binary_accuracy','balanced_accuracy','binary_bce','majority_baseline_accuracy','constant_training_prior_probability','constant_training_prior_nll','prior_corrected_logit_shift')
  out[split]={k:x[k] for k in names if k in x}
  if 'prior_corrected' in x:out[split]['prior_corrected']={k:x['prior_corrected'][k] for k in names if k in x['prior_corrected']}
 if 'checkpoint' in v:out['checkpoint_sha256']=v['checkpoint']['sha256']
 return out
for name in ('logs/staged.json','logs/launch.json','logs/status.json','logs/exit.json',
 'warmup/ready.json','warmup/failure.json','warmup/partial-state.json'):
 q=b/name
 if q.exists():
  assert q.stat().st_size<=512*1024
  value=read(q)
  if name=='warmup/ready.json':
   value={k:v for k,v in value.items() if k!='binding'}
  docs[name]=value
if 'logs/launch.json' in docs:docs['worker_alive']=alive(docs['logs/launch.json']['process'])
logs={}
for name in ('worker','features','fit'):
 q=b/'logs'/(name+'.log')
 if not q.exists():continue
 with q.open('rb') as f:f.seek(max(0,q.stat().st_size-12000));raw=f.read()
 markers=[];errors=[]
 for line in raw.decode(errors='replace').splitlines():
  try:
   x=json.loads(line)
   if isinstance(x,dict):markers.append(dict(stage=x.get('stage'),**compact_epoch(x)) if 'train' in x and 'test' in x else x)
  except ValueError:
   if line.startswith(('AssertionError','ValueError','RuntimeError','ImportError','ModuleNotFoundError','TypeError','KeyError','AttributeError','torch.cuda.OutOfMemoryError')):errors.append(line)
 logs[name]=dict(bytes=q.stat().st_size,mtime=q.stat().st_mtime,markers=markers[-2:],errors=errors[-3:])
epochs=[]
for q in sorted((b/'warmup').glob('epoch-*.json')):
 assert q.stat().st_size<=32*1024
 epochs.append(compact_epoch(read(q)))
docs['epochs']=epochs
metrics=b/'warmup/minibatch-metrics.jsonl'
if metrics.exists():
 with metrics.open('rb') as f:f.seek(max(0,metrics.stat().st_size-8000));raw=f.read()
 rows=[]
 for line in raw.splitlines():
  try:rows.append(json.loads(line))
  except ValueError:pass
 docs['latest_minibatch']=rows[-1] if rows else None
print(json.dumps(dict(stage='PREFIX_WARMUP20_OBSERVATION',checked_unix=time.time(),metadata=docs,logs=logs)),flush=True)
'''


def worker(handle_path):
    h = read(handle_path); base = Path(h['base']); repo = Path(h['runtime_repo'])
    assert Path(handle_path) == base/'intent.json' and base.resolve() == base
    assert h['runtime_source'] == SOURCE
    for name, expected in h['operator_sha256'].items():
        assert sha(base/name) == expected
    claim = read(base/'logs/launch-once.json')
    born = dict(x.split(b'=', 1) for x in Path('/proc/self/environ').read_bytes().split(b'\0') if b'=' in x)
    assert born[b'CUDA_VISIBLE_DEVICES'].decode() == os.environ['CUDA_VISIBLE_DEVICES'] == claim['gpu_uuid']
    status = dict(stage='PREFIX_WARMUP20_WORKER',started_unix=time.time(),actor_adam_updates=0)

    def publish():
        path = base/'logs/status.json'; temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(status, sort_keys=True, allow_nan=False)+'\n'); temp.replace(path)

    try:
        actions = [
            ('fit', [PYTHON, str(base/FILES[1]), '--runtime-repo', str(repo), '--cache-dir', h['feature_cache_dir'],
                '--cache-manifest-sha256',h['feature_manifest_sha256'], '--output-dir', str(base/'warmup'),
                '--physical-index', str(h['physical_index']), '--epochs', '20']),
        ]
        for action, args in actions:
            status.update(stage='RUNNING_'+action.upper(), action=action); publish()
            with (base/'logs'/(action+'.log')).open('xb') as log:
                child = subprocess.Popen(args, cwd=repo, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                info = Path('/proc')/str(child.pid); text = (info/'stat').read_text(); fields = text[text.rfind(')')+2:].split()
                status['child'] = dict(pid=child.pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=info.stat().st_uid)
                publish(); code = child.wait()
            status.update(action_exit_code=code); publish()
            if code:
                raise RuntimeError('PREFIX_WARMUP20_'+action.upper()+'_FAILED: '+str(code))
        ready=read(base/'warmup/ready.json')
        assert ready['actual_head_adam_updates']==2000 and ready['completed_epochs']==20
        assert ready['epoch_records']==21 and ready['final_epoch']==20
        assert ready['balanced_sampling'] is True and ready['per_class_per_batch']==128 and ready['replace'] is True
        assert ready['actor_updated'] is False and ready['formal_training_started'] is False
        status.update(stage='PREFIX_WARMUP20_COMPLETE',exit_code=0,head_adam_updates=2000,finished_unix=time.time())
    except BaseException as error:
        status.update(stage='PREFIX_WARMUP20_FAILED',exit_code=1,error=repr(error),finished_unix=time.time())
        raise
    finally:
        publish(); once(base/'logs/exit.json',status)
    print(json.dumps(status),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('stage','run-once','observe'))
    parser.add_argument('--handle',required=True); parser.add_argument('--worker',action='store_true')
    parser.add_argument('--reuse-features-from',type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.handle); return
    assert args.action in ('stage','run-once','observe')
    path = Path(args.handle)
    if args.action == 'stage':
        assert not path.exists()
        commit,mapping = publication(); diagnostic = read(DIAGNOSTIC); human = read(HUMAN)
        assert diagnostic['runtime_source'] == SOURCE and human['fit']['epochs'] == 20
        assert args.reuse_features_from is not None
        feature_output=args.reuse_features_from
        assert feature_output.is_absolute() and str(feature_output).startswith('/data/shaoshitong/Search-G1-qprobe-prefix-correctness-warmup20-')
        # The already saved, bounded parent observation authenticates the completed cache.
        parent=read(ROOT/'preparation/qprobe-prefix-encoding-completed-for-balanced20-20261016.json')
        assert parent['feature_output_dir']==str(feature_output) and parent['encoder_released'] is True
        h = dict(role='trajectory_prefix_correctness_balanced_warmup20_only',base=PREFIX+uuid.uuid4().hex,
            runtime_repo=diagnostic['runtime_repo'],runtime_source=SOURCE,operator_commit=commit,operator_sha256=mapping,
            human_intent_sha256=sha(HUMAN),teacher_cache='/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v4-sigma3-fullouter',
            test_base=diagnostic['base'],physical_index=0,epochs=20,train_questions=5120,test_questions=512,
            train_trajectories=25600,test_trajectories=2560,expected_head_adam_updates=2000,
            actor_training_budget=0,formal_training_remains_stopped=True,created_unix=time.time(),no_automatic_retry=True)
        h.update(feature_output_dir=str(feature_output),feature_cache_dir=str(feature_output/'cache'),
            feature_manifest_sha256=parent['manifest_sha256'],feature_parent_receipt_sha256=sha(ROOT/'preparation/qprobe-prefix-encoding-completed-for-balanced20-20261016.json'),
            balanced_sampling=True,batch_class_counts=dict(correct=128,incorrect=128),
            sample_with_replacement=True,fixed_test_distribution_unchanged=True,
            original_prior_probability_correction='corrected_logit=balanced_logit+logit(full_training_positive_prior)')
        once(path,h)
        payload = dict(handle=h,human=human,files={name:base64.b64encode((OPS/'reproduction/operations'/name).read_bytes()).decode() for name in FILES})
        code=STAGE
    else:
        h=read(path); assert h['human_intent_sha256']==sha(HUMAN)
        payload=dict(handle=h);code=LAUNCH if args.action=='run-once' else OBSERVE
    command=route().ssh_command(code)
    command[-1]=PYTHON+' -c '+command[-1][len('python3 -c '):]
    result=subprocess.run(command,input=json.dumps(payload).encode(),capture_output=True,timeout=65)
    value=dict(action=args.action,exit_code=result.returncode,checked_unix=time.time(),
        stderr_sha256=hashlib.sha256(result.stderr).hexdigest())
    if result.returncode:
        value['stderr_tail']=result.stderr.decode(errors='replace')[-3000:]
    else:
        assert len(result.stdout)<=128*1024
        value['observation']=json.loads(result.stdout)
    receipt=ROOT/'preparation'/('qprobe-prefix-warmup20-'+args.action+'-'+str(time.time_ns())+'.json')
    once(receipt,value);print(json.dumps(dict(receipt=str(receipt),**value)),flush=True)
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
