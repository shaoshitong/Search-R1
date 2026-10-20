"""Once-only independent node10 primary consumers of the new CPU relay.

Stage is small code only. Submit binds the real CPU ready, its dependencies,
selected physical UUIDs and two bounded observations before one native intent.
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
TP1 = {'gpu%d-tp1' % i: dict(indices=[i], tp=1, seqs=64, tokens=8192, blocks=4096, reserve=1.) for i in (1, 2, 3, 5)}
TOPOLOGIES = {
    'tp4': dict(TP1, **{'tp4-0-4-6-7': dict(indices=[0, 4, 6, 7], tp=4, seqs=8, tokens=4096, blocks=512, reserve=.125)}),
    'tp2-pairs': dict(TP1, **{'tp2-0-4': dict(indices=[0, 4], tp=2, seqs=16, tokens=4096, blocks=1024, reserve=.5),
        'tp2-6-7': dict(indices=[6, 7], tp=2, seqs=16, tokens=4096, blocks=1024, reserve=.5)})}
SLOTS = tuple(dict.fromkeys(slot for rows in TOPOLOGIES.values() for slot in rows))
SOURCES = ('relay_primary_control.py', 'relay_primary_worker.py', 'test_relay_primary.py',
           'legacy_relay_owner.py', 'relay_consumer_worker.py', 'shared_vram_worker.py', 'shared_vram_vllm_adapter.py',
           'draining_pool_worker.py', 'pool_worker.py', 'pool_paths.sh', 'retrieval_relay.py', 'retrieval_relay_backpressure.py')


def names(slot):
    if slot not in SLOTS: raise ValueError('Unknown finite primary slot')
    return ('sst-q7b-g5-primary10-' + slot + '-v1-' + RUN_ID[:12] + '.service', 'relay-primary10-' + slot + '-v1')


def bootstrap(slot, topology='tp4'):
    config = TOPOLOGIES[topology][slot]
    return ('BASE=' + repr(BASE) + '\nUNIT=' + repr(names(slot)[0]) + '\nCONFIG=' + repr(config)
            + '\nFROZEN_REPO=' + repr(old.FROZEN_REPO) + '\n' + r'''
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-10'and os.geteuid()==0
stage=Path(__file__).resolve().parent;root=stage.parent
manifest=json.loads((stage/'manifest.json').read_text())
for name,row in manifest['files'].items():
 data=(stage/name).read_bytes();assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],name
intent=json.loads((root/'native-submit-intent.json').read_text())
path=root/'slot-context.json';raw=path.read_bytes();context=json.loads(raw);cpu=context['cpu_ready']
assert intent['unit']==UNIT and intent['source_bundle_sha256']==manifest['source_bundle_sha256']
assert hashlib.sha256(raw).hexdigest()==intent['dependency_context_sha256']
assert context['unit']==UNIT and context['run_id']==manifest['run_id']and context['slot_configuration']==CONFIG
assert context['model_deadline_epoch']<=cpu['consumer_deadline_epoch']<=cpu['deadline_epoch']-600
assert context['model_deadline_epoch']<=context['source']['source_sampling_deadline_epoch']-600
run_seconds=min(14400,int(context['model_deadline_epoch']-time.time()));assert run_seconds>1260
assert os.stat('/proc/self/ns/mnt').st_ino!=os.stat('/proc/1/ns/mnt').st_ino
assert os.stat('/proc/self/ns/net').st_ino==os.stat('/proc/1/ns/net').st_ino
launch=Path(BASE)/'relay-primary10/launch'/('native-'+str(os.getpid()));launch.mkdir(parents=True)
log=(launch/'bootstrap.log').open('ab',buffering=0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
detail=subprocess.check_output(['systemctl','show',UNIT,'-p','Id','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','RuntimeMaxUSec','-p','MemoryMax'],text=True)
props=dict(line.split('=',1)for line in detail.splitlines()if'='in line)
assert props['Id']==UNIT and props['MainPID']==str(os.getpid())and props['ActiveState']=='active'
assert props['ControlGroup']=='/system.slice/'+UNIT and props['ControlGroup']in Path('/proc/self/cgroup').read_text()
birth=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
evidence=launch/'allocation-evidence.json'
evidence.write_text(json.dumps(dict(node='h100-10',unit=UNIT,main_pid=os.getpid(),controller_birth=birth,boot_id=boot_id,
 unit_show=detail,cgroup=props['ControlGroup'],captured_epoch=time.time(),resource_limits=dict(cpu_quota_percent=800,
 memory_max_bytes=103079215104,runtime_max_seconds=14400),original_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))))
os.chmod(evidence,0o400)
env=dict(os.environ,PYTHONPATH=FROZEN_REPO+':/dataset/shaoshitong/Search-G1-common8/retriever-gpu-lib',
 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',SEARCH_G1_B_PREFIX_TOKENS='20',
 SEARCH_G1_PHYSICAL_WORLD_SIZE='4',SEARCH_G1_BASE=BASE,HF_HOME=BASE+'/cache/huggingface',
 TORCH_EXTENSIONS_DIR=BASE+'/cache/torch_extensions',TMPDIR=str(launch/'tmp'),
 CUDA_VISIBLE_DEVICES=','.join(map(str,CONFIG['indices'])),NCCL_NVLS_ENABLE='0',NCCL_BUFFSIZE='1048576',NCCL_MAX_NCHANNELS='4',
 NCCL_DEBUG='INFO',VLLM_ATTENTION_BACKEND='XFORMERS',VLLM_HOST_IP='127.0.0.1',
 Q7B_NODE10_PRIMARY_CONTEXT=str(path),Q7B_NODE10_PRIMARY_CONTEXT_SHA256=hashlib.sha256(raw).hexdigest(),
 Q7B_SLOT_OWNER_DRAIN=context['owner_drain_path'],
 CPATH='/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include:'
 '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/python3.10:'
 '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/x86_64-linux-gnu/python3.10')
for key in('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):env.pop(key,None)
for key in('HF_HOME','TORCH_EXTENSIONS_DIR','TMPDIR'):Path(env[key]).mkdir(parents=True,exist_ok=True)
os.chdir(FROZEN_REPO)
command=['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-u',str(stage/'relay_primary_worker.py'),
 '--base',BASE,'--allocation-evidence',str(evidence),'--pool-worker',str(stage/'draining_pool_worker.py'),
 '--relay-script',str(stage/'retrieval_relay_backpressure.py'),'--upstream-ready',context['upstream_ready'],
 '--reuse-local-relay-ready',context['reuse_local_relay_ready'],'--reuse-local-relay-sha256',context['reuse_local_relay_sha256'],
 '--reuse-local-relay-unit',cpu['owner']['unit'],'--model-gpus',','.join(map(str,CONFIG['indices'])),'--tp',str(CONFIG['tp']),
 '--model-max-seqs',str(CONFIG['seqs']),'--model-batch-tokens',str(CONFIG['tokens']),
 '--model-kv-block-cap',str(CONFIG['blocks']),'--model-reserve-gib',str(CONFIG['reserve']),
 '--run-seconds',str(run_seconds),'--batch-reserve-seconds','1200']
print(json.dumps(dict(status='NODE10_PRIMARY_BOOTSTRAP',unit=UNIT,pid=os.getpid(),controller_birth=birth,boot_id=boot_id,
 model_gpus=CONFIG['indices'],relay_owned=False,model_deadline_epoch=context['model_deadline_epoch'],epoch=time.time())),flush=True)
os.execvpe(command[0],command,env)
''').encode()


def bundle(slot, topology='tp4'):
    if topology not in TOPOLOGIES or slot not in TOPOLOGIES[topology]: raise ValueError('Slot is outside chosen topology')
    unit, directory = names(slot)
    config = TOPOLOGIES[topology][slot]
    files = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCES}
    rows = {name: dict(bytes=len(data), sha256=old.digest(data)) for name, data in files.items()}
    identity = old.digest(old.encoded(dict(mode=directory, topology=topology, source_files=rows, model_configuration=config)))
    files.update({'relay_primary_bootstrap.py': bootstrap(slot, topology),
                  'consumer_paths.sh': old.native_paths_source(files['pool_paths.sh'])})
    manifest = dict(source_bundle_sha256=identity, run_id=RUN_ID, seed=20261021, slot=slot, topology=topology,
        slots=TOPOLOGIES[topology], unit=unit, directory=directory, model_gpus=config['indices'], tensor_parallel_size=config['tp'],
        model_configuration=config, max_model_len=4096, maximum_run_seconds=14400, batch_reserve_seconds=1200,
        owner_deadline_cutoff_seconds=600, local_relay_owned=False, primary_uuid_locks_owned=True,
        upstream_retriever_owned=False, original_full_fp16_index_and_top3_preserved=True,
        source_files=rows, files={name: dict(bytes=len(data), sha256=old.digest(data)) for name, data in files.items()})
    return identity, files, manifest


REMOTE = r'''
import base64,fcntl,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-10'and os.geteuid()==0
base=Path(PHYSICAL);root=base/'preparation'/MANIFEST['directory'];stage=root/IDENTITY;UNIT=MANIFEST['unit']
def read(p):return json.loads(p.read_text())if p.exists()else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def immutable(p,value):
 raw=json.dumps(value,sort_keys=True,indent=2).encode();temporary=p.with_name(p.name+'.tmp-%d-%d'%(os.getpid(),time.time_ns()))
 try:
  with temporary.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
  try:os.link(temporary,p)
  except FileExistsError:assert p.read_bytes()==raw
 finally:temporary.unlink(missing_ok=True)
def query(unit):
 r=subprocess.run(['systemctl','show',unit,'-p','LoadState','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','Result','-p','ExecMainStatus'],text=True,capture_output=True,timeout=15)
 return dict(returncode=r.returncode,properties=dict(line.split('=',1)for line in r.stdout.splitlines()if'='in line),stderr=r.stderr)
def verify():
 assert read(stage/'manifest.json')==MANIFEST
 for name,row in MANIFEST['files'].items():
  data=(stage/name).read_bytes();assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],name
 assert(read(stage/'stage-receipt.json')or{}).get('status')=='NODE10_PRIMARY_SMALL_CODE_FULL_SHA_PASS'
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
 if not(stage/'stage-receipt.json').exists():immutable(stage/'stage-receipt.json',dict(status='NODE10_PRIMARY_SMALL_CODE_FULL_SHA_PASS',publication=PUBLICATION,files=MANIFEST['files'],epoch=time.time()))
 verify();value.update(status='NODE10_PRIMARY_CODE_ONLY_STAGED',stage=str(stage),files=MANIFEST['files'])
elif ACTION=='submit':
 with(root/'native-submit.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if intent.exists():value.update(status='EXISTING_INTENT_NO_START_REPLAY',intent=read(intent),receipt=read(receipt))
  else:
   verify();assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
   observed=query(UNIT);value['observed']=observed
   assert observed['returncode']==0 and observed['properties'].get('LoadState')=='not-found'
   sys.path.insert(0,str(stage));import relay_primary_worker as worker
   first=worker.capture_context(base,MANIFEST['slot'],BASE,CPU_READY,CPU_SHA,MANIFEST['topology']);time.sleep(3)
   context=worker.capture_context(base,MANIFEST['slot'],BASE,CPU_READY,CPU_SHA,MANIFEST['topology'])
   for key in('cpu_ready','cpu_ready_sha256','source','model_cuda_uuids','reuse_local_relay_sha256','upstream_ready_sha256','source_context_sha256'):
    assert context[key]==first[key],'Dependency or physical GPU changed across submit windows: '+key
   context['model_deadline_epoch']=min(context['model_deadline_epoch'],first['model_deadline_epoch'])
   context['window_observations']=[dict(epoch=row['captured_epoch'],gpus=row['initial_gpu_memory'])for row in(first,context)]
   assert context['window_observations'][1]['epoch']-context['window_observations'][0]['epoch']>=3
   path=root/'slot-context.json'
   if path.exists():
    previous=read(path);assert previous['run_id']==RUN_ID and previous['unit']==UNIT
    archived=root/('preintent-context-'+sha(path)+'.json')
    if archived.exists():assert archived.read_bytes()==path.read_bytes()
    else:os.link(path,archived)
    path.unlink()
   immutable(path,context)
   command=['systemd-run','--unit='+UNIT,'--property=CPUQuota=800%','--property=MemoryMax=96G',
    '--property=RuntimeMaxSec=14400','--property=KillMode=control-group','--property=TimeoutStopSec=60',
    '--setenv=Q7B_NATIVE_UNIT_INSTANCE='+UNIT,'unshare','--mount','--propagation','private','/bin/bash',
    str(stage/'consumer_paths.sh'),'h100-10','/usr/bin/python3',str(Path(BASE)/'preparation'/MANIFEST['directory']/IDENTITY/'relay_primary_bootstrap.py')]
   immutable(intent,dict(intent_id=INTENT,unit=UNIT,run_id=RUN_ID,source_bundle_sha256=IDENTITY,
    argv=command,dependency_context_sha256=sha(path),dependency=context,publication=PUBLICATION,epoch=time.time()))
   try:
    r=subprocess.run(command,text=True,capture_output=True,timeout=45)
    result=dict(status='OWN_NODE10_PRIMARY_STARTED_ONCE'if r.returncode==0 else'OWN_NODE10_PRIMARY_START_FAILED_NO_REPLAY',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,epoch=time.time())
   except subprocess.TimeoutExpired as error:result=dict(status='OWN_START_OUTCOME_UNKNOWN_QUERY_SAME_UNIT',error=str(error),epoch=time.time())
   result['observed']=query(UNIT);immutable(receipt,result);value.update(status=result['status'],receipt=result,dependency=context)
else:
 saved=read(intent);observed=query(UNIT);pid=observed['properties'].get('MainPID','0')
 if pid=='0':pid=((read(receipt)or{}).get('observed')or{}).get('properties',{}).get('MainPID','0')
 attempts=[]
 if pid!='0':
  for job in sorted((base/'pool/jobs'/('job-relay10-primary-'+pid)).glob('attempt-*')):
   attempts.append(dict(directory=str(job),allocation=read(job/'allocation.json'),progress=read(job/'progress.json'),
    failure=read(job/'failure.json'),complete=read(job/'complete.json'),children=read(job/'children.json'),
    model_initialized=[read(p)for p in(job/'sampler-launches').glob('launch-*/model-initialized.json')]))
 value.update(status='READ_ONLY_NODE10_PRIMARY_STATUS',observed=observed,intent=saved,receipt=read(receipt),
  dependency=read(root/'slot-context.json'),attempts=attempts,global_goal_complete=False)
print(json.dumps(value))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'stage', 'submit', 'status'))
    parser.add_argument('--slot', choices=SLOTS, required=True)
    parser.add_argument('--topology', choices=tuple(TOPOLOGIES), default='tp4')
    parser.add_argument('--cpu-ready', default=BASE + '/preparation/legacy-relay10-independent-v1/owner-ready.json')
    parser.add_argument('--cpu-ready-sha256')
    parser.add_argument('--intent-id')
    parser.add_argument('--published-commit')
    parser.add_argument('--published-ref')
    args = parser.parse_args()
    identity, files, manifest = bundle(args.slot, args.topology)
    if args.action == 'plan':
        print(json.dumps(manifest, indent=2)); return
    publication = None
    if args.action != 'status':
        if not re.fullmatch(r'[A-Za-z0-9._-]{1,120}', args.intent_id or ''): raise ValueError('Explicit immutable intent identity required')
        publication = old.verify_publication(args.published_commit, args.published_ref, manifest['source_files'])
    if args.action == 'submit' and not re.fullmatch(r'[0-9a-f]{64}', args.cpu_ready_sha256 or ''):
        raise ValueError('Actual new CPU-owner ready SHA required')
    values = dict(BASE=BASE, PHYSICAL=PHYSICAL, RUN_ID=RUN_ID, IDENTITY=identity, ACTION=args.action, INTENT=args.intent_id,
        CPU_READY=args.cpu_ready, CPU_SHA=args.cpu_ready_sha256, PUBLICATION=publication, MANIFEST=manifest,
        FILES={name: base64.b64encode(data).decode() for name, data in files.items()})
    script = '\n'.join(key + '=' + repr(value) for key, value in values.items()) + '\n' + REMOTE
    out = ROOT / 'work' / ('qwen7b-node10-primary-%s-%s-%d.json' % (args.slot, args.action, time.time_ns()))
    try: result = old.transport(script, 'h100-10')
    except subprocess.TimeoutExpired:
        out.write_text(json.dumps(dict(action=args.action, status='TRANSPORT_UNKNOWN_QUERY_SAME_INTENT_NO_REPLAY', epoch=time.time())))
        raise
    value = dict(action=args.action, slot=args.slot, returncode=result.returncode, stdout=result.stdout,
        stderr=result.stderr, receipt=str(out), epoch=time.time())
    out.write_text(json.dumps(value, indent=2) + '\n'); print(json.dumps(value, indent=2))
    if result.returncode: raise SystemExit(result.returncode)


if __name__ == '__main__': main()
