"""Once-only finite h100-10 TP1 slots using the current V5 local relay.

Plan is local; stage copies only small code to this run's own NAS10 directory.
Submit observes current owner births, primary locks and two NVML windows before
one immutable intent. It never replaces or stops an existing unit or relay.
"""
import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import time

from . import shared_vram_control as old

ROOT, BASE, PHYSICAL, RUN_ID = old.ROOT, old.BASE, old.PHYSICAL, old.RUN_ID
SLOTS = {'gpu1': 1, 'gpu2': 2, 'gpu3': 3, 'gpu5': 5}
LIMITS = dict(seqs=64, tokens=8192, blocks=4096, reserve=1.0)
MAX_SECONDS = 14400
SOURCES = ('relay_tp1_control.py', 'relay_tp1_worker.py', 'test_relay_tp1_control.py',
           'test_relay_tp1_worker.py', 'relay_consumer_worker.py', 'shared_vram_worker.py',
           'shared_vram_vllm_adapter.py', 'pool_worker.py', 'pool_paths.sh', 'retrieval_relay.py')


def names(slot):
    if slot not in SLOTS:
        raise ValueError('Only the four current-cohort gpu1/gpu2/gpu3/gpu5 slots are permitted')
    return ('sst-q7b-g5-consumer10-tp1-' + slot + '-v1-' + RUN_ID[:12] + '.service',
            'relay-consumer10-tp1-' + slot + '-v1')


def bootstrap(slot):
    unit, _ = names(slot)
    return ('BASE=' + repr(BASE) + '\nUNIT=' + repr(unit) + '\nMODEL=' + repr(SLOTS[slot])
            + '\nFROZEN_REPO=' + repr(old.FROZEN_REPO) + '\nLIMITS=' + repr(LIMITS) + '\n' + r'''
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-10'and os.geteuid()==0
stage=Path(__file__).resolve().parent;root=stage.parent
manifest=json.loads((stage/'manifest.json').read_text())
for name,row in manifest['files'].items():
 data=(stage/name).read_bytes();assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],name
intent=json.loads((root/'native-submit-intent.json').read_text())
path=root/'slot-context.json';raw=path.read_bytes();context=json.loads(raw)
assert intent['unit']==UNIT and intent['source_bundle_sha256']==manifest['source_bundle_sha256']
assert hashlib.sha256(raw).hexdigest()==intent['dependency_context_sha256']
assert context['unit']==UNIT and context['run_id']==manifest['run_id']and context['model_indices']==[MODEL]
assert context['model_limits']==LIMITS and context['model_deadline_epoch']<=context['relay_owner']['deadline_epoch']-600
assert context['model_deadline_epoch']<=context['lease_owner']['deadline_epoch']-600
assert context['model_deadline_epoch']<=context['source']['source_sampling_deadline_epoch']-600
run_seconds=min(14400,int(context['model_deadline_epoch']-time.time()))
assert run_seconds>1260
assert os.stat('/proc/self/ns/mnt').st_ino!=os.stat('/proc/1/ns/mnt').st_ino
assert os.stat('/proc/self/ns/net').st_ino==os.stat('/proc/1/ns/net').st_ino
launch=Path(BASE)/'relay-consumer10-tp1/launch'/('native-'+str(os.getpid()));launch.mkdir(parents=True)
log=(launch/'bootstrap.log').open('ab',buffering=0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
detail=subprocess.check_output(['systemctl','show',UNIT,'-p','Id','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','RuntimeMaxUSec','-p','MemoryMax'],text=True)
props=dict(line.split('=',1)for line in detail.splitlines()if'='in line)
assert props['Id']==UNIT and props['MainPID']==str(os.getpid())and props['ActiveState']=='active'
assert props['ControlGroup']=='/system.slice/'+UNIT and props['ControlGroup']in Path('/proc/self/cgroup').read_text()
evidence=launch/'allocation-evidence.json'
evidence.write_text(json.dumps(dict(node='h100-10',unit=UNIT,main_pid=os.getpid(),unit_show=detail,
 cgroup=props['ControlGroup'],captured_epoch=time.time(),resource_limits=dict(cpu_quota_percent=800,
 memory_max_bytes=103079215104,runtime_max_seconds=14400),original_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))))
os.chmod(evidence,0o400)
env=dict(os.environ,PYTHONPATH=FROZEN_REPO+':/dataset/shaoshitong/Search-G1-common8/retriever-gpu-lib',
 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',SEARCH_G1_B_PREFIX_TOKENS='20',
 SEARCH_G1_PHYSICAL_WORLD_SIZE='4',SEARCH_G1_BASE=BASE,HF_HOME=BASE+'/cache/huggingface',
 TORCH_EXTENSIONS_DIR=BASE+'/cache/torch_extensions',TMPDIR=str(launch/'tmp'),
 CUDA_VISIBLE_DEVICES=str(MODEL),NCCL_NVLS_ENABLE='0',NCCL_BUFFSIZE='1048576',NCCL_MAX_NCHANNELS='4',
 NCCL_DEBUG='INFO',VLLM_ATTENTION_BACKEND='XFORMERS',VLLM_HOST_IP='127.0.0.1',
 Q7B_NODE10_TP1_CONTEXT=str(path),Q7B_NODE10_TP1_CONTEXT_SHA256=hashlib.sha256(raw).hexdigest(),
 CPATH='/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include:'
 '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/python3.10:'
 '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/x86_64-linux-gnu/python3.10')
for key in('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):env.pop(key,None)
for key in('HF_HOME','TORCH_EXTENSIONS_DIR','TMPDIR'):Path(env[key]).mkdir(parents=True,exist_ok=True)
os.chdir(FROZEN_REPO)
command=['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-u',str(stage/'relay_tp1_worker.py'),
 '--base',BASE,'--allocation-evidence',str(evidence),'--pool-worker',str(stage/'pool_worker.py'),
 '--relay-script',str(stage/'retrieval_relay.py'),'--upstream-ready',context['upstream_ready'],
 '--reuse-local-relay-ready',context['reuse_local_relay_ready'],
 '--reuse-local-relay-sha256',context['reuse_local_relay_sha256'],
 '--reuse-local-relay-unit',context['relay_owner']['unit'],'--model-gpus',str(MODEL),'--tp','1',
 '--model-max-seqs','64','--model-batch-tokens','8192','--model-kv-block-cap','4096','--model-reserve-gib','1.0',
 '--run-seconds',str(run_seconds),'--batch-reserve-seconds','1200']
print(json.dumps(dict(status='NODE10_FINITE_TP1_BOOTSTRAP',unit=UNIT,pid=os.getpid(),model_gpus=[MODEL],
 relay_owned=False,primary_owner_owned=False,model_deadline_epoch=context['model_deadline_epoch'],epoch=time.time())),flush=True)
os.execvpe(command[0],command,env)
''').encode()


def bundle(slot):
    unit, directory = names(slot)
    files = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCES}
    rows = {name: dict(bytes=len(data), sha256=old.digest(data)) for name, data in files.items()}
    identity = old.digest(old.encoded(dict(mode=directory, source_files=rows, model_limits=LIMITS)))
    files.update({'relay_tp1_bootstrap.py': bootstrap(slot),
                  'consumer_paths.sh': old.native_paths_source(files['pool_paths.sh'])})
    manifest = dict(source_bundle_sha256=identity, run_id=RUN_ID, seed=20261021, slot=slot,
        unit=unit, directory=directory, model_gpus=[SLOTS[slot]], tensor_parallel_size=1,
        model_limits=dict(LIMITS), max_model_len=4096, maximum_run_seconds=MAX_SECONDS, batch_reserve_seconds=1200,
        owner_deadline_cutoff_seconds=600, local_relay_owned=False, primary_owner_owned=False,
        upstream_retriever_owned=False, no_actor_or_probe_updates=True,
        original_full_fp16_index_and_top3_preserved=True,
        source_files=rows, files={name: dict(bytes=len(data), sha256=old.digest(data)) for name, data in files.items()})
    return identity, files, manifest


REMOTE = r'''
import base64,fcntl,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-10'and os.geteuid()==0
base=Path(PHYSICAL);root=base/'preparation'/MANIFEST['directory'];stage=root/IDENTITY;UNIT=MANIFEST['unit']
def read(p):return json.loads(p.read_text())if p.exists()else None
def immutable(p,value):
 raw=json.dumps(value,sort_keys=True,indent=2).encode();temporary=p.with_name(p.name+'.tmp-%d-%d'%(os.getpid(),time.time_ns()))
 try:
  with temporary.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
  try:os.link(temporary,p)
  except FileExistsError:assert p.read_bytes()==raw
 finally:temporary.unlink(missing_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def query(unit):
 r=subprocess.run(['systemctl','show',unit,'-p','LoadState','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','Result','-p','ExecMainStatus'],text=True,capture_output=True,timeout=15)
 return dict(returncode=r.returncode,properties=dict(line.split('=',1)for line in r.stdout.splitlines()if'='in line),stderr=r.stderr)
def verify():
 assert read(stage/'manifest.json')==MANIFEST
 for name,row in MANIFEST['files'].items():
  data=(stage/name).read_bytes();assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],name
 assert(read(stage/'stage-receipt.json')or{}).get('status')=='NODE10_TP1_SMALL_CODE_FULL_SHA_PASS'
value=dict(action=ACTION,slot=MANIFEST['slot'],unit=UNIT,bundle_sha256=IDENTITY,epoch=time.time())
intent=root/'native-submit-intent.json';receipt=root/'native-submit-receipt.json'
if ACTION=='stage':
 assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 root.mkdir(parents=True,exist_ok=True);stage.mkdir(exist_ok=True)
 for name,encoded in FILES.items():
  path=stage/name;raw=base64.b64decode(encoded)
  if not path.exists():
   with path.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
  assert sha(path)==MANIFEST['files'][name]['sha256'];os.chmod(path,0o555)
 if not(stage/'manifest.json').exists():immutable(stage/'manifest.json',MANIFEST)
 if not(stage/'stage-receipt.json').exists():immutable(stage/'stage-receipt.json',dict(status='NODE10_TP1_SMALL_CODE_FULL_SHA_PASS',publication=PUBLICATION,files=MANIFEST['files'],epoch=time.time()))
 verify();value.update(status='NODE10_TP1_CODE_ONLY_STAGED',stage=str(stage),files=MANIFEST['files'])
elif ACTION=='submit':
 with(root/'native-submit.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if intent.exists():value.update(status='EXISTING_INTENT_NO_START_REPLAY',intent=read(intent),receipt=read(receipt))
  else:
   verify();assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
   observed=query(UNIT);value['observed']=observed
   assert observed['returncode']==0 and observed['properties'].get('LoadState')=='not-found'
   sys.path.insert(0,str(stage));import relay_tp1_worker as worker
   first=worker.capture_context(base,MANIFEST['slot'],BASE);time.sleep(3)
   context=worker.capture_context(base,MANIFEST['slot'],BASE)
   for key in('lease_owner','relay_owner','relay','source','primary_gpu_lock','model_cuda_uuids','reuse_local_relay_sha256','upstream_ready_sha256','source_context_sha256'):
    assert context[key]==first[key],'Dependency or physical GPU changed across submit windows: '+key
   context['model_deadline_epoch']=min(context['model_deadline_epoch'],first['model_deadline_epoch'])
   context['unit']=UNIT;context['window_observations']=[dict(epoch=row['captured_epoch'],gpu=row['initial_gpu_memory'])for row in(first,context)]
   assert context['window_observations'][1]['epoch']-context['window_observations'][0]['epoch']>=3
   immutable(root/'slot-context.json',context)
   shared_stage=Path(BASE)/'preparation'/MANIFEST['directory']/IDENTITY
   command=['systemd-run','--unit='+UNIT,'--property=CPUQuota=800%','--property=MemoryMax=96G',
    '--property=RuntimeMaxSec=14400','--property=KillMode=control-group','--property=TimeoutStopSec=60',
    '--setenv=Q7B_NATIVE_UNIT_INSTANCE='+UNIT,'unshare','--mount','--propagation','private','/bin/bash','-c',
    base64.b64decode(FILES['consumer_paths.sh']).decode(),'consumer-private-paths','h100-10',
    '/usr/bin/python3',str(shared_stage/'relay_tp1_bootstrap.py')]
   immutable(intent,dict(intent_id=INTENT,unit=UNIT,run_id=RUN_ID,source_bundle_sha256=IDENTITY,
    argv=command,dependency_context_sha256=sha(root/'slot-context.json'),dependency=context,publication=PUBLICATION,epoch=time.time()))
   try:
    r=subprocess.run(command,text=True,capture_output=True,timeout=45)
    result=dict(status='OWN_NODE10_TP1_STARTED_ONCE'if r.returncode==0 else'OWN_NODE10_TP1_START_FAILED_NO_REPLAY',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,epoch=time.time())
   except subprocess.TimeoutExpired as error:result=dict(status='OWN_START_OUTCOME_UNKNOWN_QUERY_SAME_UNIT',error=str(error),epoch=time.time())
   result['observed']=query(UNIT);immutable(receipt,result);value.update(status=result['status'],receipt=result,dependency=context)
else:
 saved=read(intent);attempts=[]
 observed=query(UNIT);pid=observed['properties'].get('MainPID','0')
 if pid=='0':pid=((read(receipt)or{}).get('observed')or{}).get('properties',{}).get('MainPID','0')
 if pid!='0':
  for job in sorted((base/'pool/jobs'/('job-relay10-tp1-'+pid)).glob('attempt-*')):
   row=dict(directory=str(job),progress=read(job/'progress.json'),failure=read(job/'failure.json'),complete=read(job/'complete.json'),
    model_initialized=[read(p)for p in(job/'sampler-launches').glob('launch-*/model-initialized.json')],active_phases=[])
   allocation=read(job/'allocation.json')or{};children=read(job/'children.json')or{}
   for child in children.get('own_process_groups',[]):
    try:
     group=allocation.get('native_cgroup');assert group and group in Path('/proc/%d/cgroup'%child).read_text()
     for fd in Path('/proc/%d/fd'%child).iterdir():
      try:target=os.readlink(fd)
      except FileNotFoundError:continue
      if '/pool/batch-locks/'not in target:continue
      prefix,relative=target.split('/pool/batch-locks/',1);tail=Path(relative)
      if not prefix.endswith('/shaoshitong/'+base.name)or len(tail.parts)!=2 or tail.suffix!='.lock':continue
      for phase in('prefix','full'):
       detail=read(base/'sampling'/tail.parent/'batches'/tail.stem/'outers/outer-000001'/phase/'rank-0-progress.json')
       if detail:row['active_phases'].append(dict(batch=tail.stem,phase=phase,metadata=detail))
    except(FileNotFoundError,ProcessLookupError,AssertionError):continue
   attempts.append(row)
 value.update(status='READ_ONLY_NODE10_TP1_STATUS',observed=observed,intent=saved,receipt=read(receipt),
  dependency=read(root/'slot-context.json'),attempts=attempts,global_goal_complete=False)
print(json.dumps(value))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'stage', 'submit', 'status'))
    parser.add_argument('--slot', choices=tuple(SLOTS), required=True)
    parser.add_argument('--intent-id')
    parser.add_argument('--published-commit')
    parser.add_argument('--published-ref')
    args = parser.parse_args()
    identity, files, manifest = bundle(args.slot)
    if args.action == 'plan':
        print(json.dumps(manifest, indent=2))
        return
    publication = None
    if args.action != 'status':
        if not re.fullmatch(r'[A-Za-z0-9._-]{1,120}', args.intent_id or ''):
            raise ValueError('An explicit immutable intent identity is required')
        publication = old.verify_publication(args.published_commit, args.published_ref, manifest['source_files'])
    values = dict(BASE=BASE, PHYSICAL=PHYSICAL, IDENTITY=identity, ACTION=args.action, INTENT=args.intent_id,
                  PUBLICATION=publication, MANIFEST=manifest,
                  FILES={name: base64.b64encode(data).decode() for name, data in files.items()})
    script = '\n'.join(key + '=' + repr(value) for key, value in values.items()) + '\n' + REMOTE
    out = ROOT / 'work' / ('qwen7b-node10-tp1-%s-%s-%d.json' % (args.slot, args.action, time.time_ns()))
    try:
        result = old.transport(script, 'h100-10')
    except subprocess.TimeoutExpired:
        out.write_text(json.dumps(dict(action=args.action, status='TRANSPORT_UNKNOWN_QUERY_SAME_INTENT_NO_REPLAY', epoch=time.time())))
        raise
    value = dict(action=args.action, slot=args.slot, returncode=result.returncode, stdout=result.stdout,
                 stderr=result.stderr, receipt=str(out), epoch=time.time())
    out.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
