"""Once-only future CPU relay owners and h100-10 slots; plan is local only.

Use retrieval_remote_guard_control for the new source owner. This control
never adopts current V6/V5, changes Slurm, or returns remote token contents.
"""
import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import time
from . import shared_vram_control as old
from .retrieval_relay_owner import SLOTS, DEFAULT_SLOTS

ROOT, BASE, PHYSICAL, RUN_ID = old.ROOT, old.BASE, old.PHYSICAL, old.RUN_ID
SOURCES = ('retrieval_relay_generation_control.py', 'retrieval_relay_owner.py', 'relay_slot_consumer_worker.py',
    'retrieval_remote_guard_owner.py', 'retrieval_only_owner.py', 'source_slot_consumer_worker.py',
    'draining_pool_worker.py', 'pool_worker.py', 'relay_consumer_worker.py', 'retrieval_relay.py',
    'shared_vram_worker.py', 'shared_vram_vllm_adapter.py', 'pool_paths.sh')


def bootstrap(role, unit, slot):
    return ('BASE=' + repr(BASE) + '\nFROZEN_REPO=' + repr(old.FROZEN_REPO) + '\nUNIT=' + repr(unit)
        + '\nROLE=' + repr(role) + '\nSLOT=' + repr(slot) + '\n' + r'''
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
stage=Path(__file__).resolve().parent;root=stage.parent
manifest=json.loads((stage/'manifest.json').read_text());node='h100-5'if ROLE=='upstream'else'h100-10'
assert os.geteuid()==0 and socket.gethostname().split('.')[0]==node
assert os.stat('/proc/self/ns/mnt').st_ino!=os.stat('/proc/1/ns/mnt').st_ino
assert os.stat('/proc/self/ns/net').st_ino==os.stat('/proc/1/ns/net').st_ino
for name,row in manifest['files'].items():
 raw=(stage/name).read_bytes();assert len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['sha256'],name
context=json.loads((root/'submit-context.json').read_text())
detail=subprocess.check_output(['systemctl','show',UNIT,'-p','Id','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','MemoryMax','-p','ActiveEnterTimestampMonotonic'],text=True)
props=dict(line.split('=',1)for line in detail.splitlines()if'='in line)
assert props['Id']==UNIT and props['MainPID']==str(os.getpid())and props['ActiveState']=='active'
assert any(line.split(':',2)[-1]==props['ControlGroup']for line in Path('/proc/self/cgroup').read_text().splitlines())
hard=28800 if ROLE=='slot'else 90000;memory=103079215104 if ROLE=='slot'else 1073741824
assert int(props['MemoryMax'])==memory
remaining=hard-(time.monotonic()-int(props['ActiveEnterTimestampMonotonic'])/1000000);assert 0<remaining<=hard
launch=Path(BASE)/'retrieval-relay-generation/launch'/('native-'+str(os.getpid()));launch.mkdir(parents=True)
log=(launch/'bootstrap.log').open('ab',buffering=0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
evidence=launch/'allocation-evidence.json'
evidence.write_text(json.dumps(dict(mode='NATIVE_RELAY_SLOT'if ROLE=='slot'else'NATIVE_CPU_RELAY_OWNER',
 node=node,unit=UNIT,main_pid=os.getpid(),cgroup=props['ControlGroup'],unit_show=detail,
 captured_epoch=time.time(),unit_hard_end_epoch=time.time()+remaining,
 resource_limits=dict(runtime_max_seconds=hard,cpu_quota_percent=800 if ROLE=='slot'else 100,memory_max_bytes=memory),
 original_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))));os.chmod(evidence,0o400)
env=dict(os.environ,PYTHONPATH=FROZEN_REPO+':/dataset/shaoshitong/Search-G1-common8/retriever-gpu-lib',
 PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',SEARCH_G1_B_PREFIX_TOKENS='20',
 SEARCH_G1_BASE=BASE,HF_HOME=BASE+'/cache/huggingface',TORCH_EXTENSIONS_DIR=BASE+'/cache/torch_extensions',
 TMPDIR=str(launch/'tmp'),CUDA_VISIBLE_DEVICES='',VLLM_HOST_IP='127.0.0.1',VLLM_ATTENTION_BACKEND='XFORMERS',
 NCCL_NVLS_ENABLE='0',NCCL_BUFFSIZE='1048576',NCCL_MAX_NCHANNELS='4',NCCL_DEBUG='INFO',
 CPATH='/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include:'
 '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/python3.10:'
 '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/x86_64-linux-gnu/python3.10')
for key in('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):env.pop(key,None)
for key in('HF_HOME','TORCH_EXTENSIONS_DIR','TMPDIR'):Path(env[key]).mkdir(parents=True,exist_ok=True)
python='/dataset/shaoshitong/Search-G1-common8/env/bin/python'
if ROLE!='slot':
 command=[python,'-u',str(stage/'retrieval_relay_owner.py'),ROLE,'--base',BASE,'--allocation-evidence',str(evidence),
  '--source-owner-ready',context['source_owner_ready'],'--source-owner-ready-sha256',context['source_owner_ready_sha256'],
  '--pool-worker',str(stage/'pool_worker.py'),'--relay-script',str(stage/'retrieval_relay.py'),
  '--token-file',manifest['token_file'],'--ready-file',str(root/'owner-ready.json'),'--owner-seconds','86400']
 if ROLE=='upstream':command+=['--listen-host',context['listen_host']]
 else:command+=['--upstream-owner-ready',context['upstream_owner_ready'],
  '--upstream-owner-ready-sha256',context['upstream_owner_ready_sha256'],
  '--predecessor-proof',context['predecessor_proof'],'--predecessor-sha256',context['predecessor_sha256'],
  '--slot-manifest',str(stage/'slots.json')]
else:
 path=Path(context['downstream_owner_ready']);raw=path.read_bytes()
 assert hashlib.sha256(raw).hexdigest()==context['downstream_owner_ready_sha256'];ready=json.loads(raw);slot=ready['slots'][SLOT]
 run_seconds=min(28500,int(remaining-90),int(ready['consumer_deadline_epoch']-time.time()-60));assert run_seconds>1260
 up=json.loads(Path(ready['upstream_owner_ready']).read_text())
 env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,slot['indices'])),SEARCH_G1_PHYSICAL_WORLD_SIZE=str(slot['tp']),
  Q7B_RELAY_SLOT_NAME=SLOT,Q7B_SLOT_OWNER_DRAIN=str(Path(ready['job_ready_path']).parent/'drain-request.json'))
 command=[python,'-u',str(stage/'relay_slot_consumer_worker.py'),'--base',BASE,'--allocation-evidence',str(evidence),
  '--pool-worker',str(stage/'draining_pool_worker.py'),'--relay-script',str(stage/'retrieval_relay.py'),
  '--upstream-ready',up['transport_ready_path'],'--reuse-local-relay-ready',str(path),
  '--reuse-local-relay-sha256',context['downstream_owner_ready_sha256'],'--reuse-local-relay-unit',ready['owner']['unit'],
  '--model-gpus',','.join(map(str,slot['indices'])),'--tp',str(slot['tp']),
  '--model-max-seqs',str(slot['seqs']),'--model-batch-tokens',str(slot['tokens']),
  '--model-kv-block-cap',str(slot['blocks']),'--model-reserve-gib',str(slot['reserve']),
  '--run-seconds',str(run_seconds),'--batch-reserve-seconds','1200']
os.chdir(FROZEN_REPO)
print(json.dumps(dict(status='NEW_INDEPENDENT_RELAY_GENERATION_BOOTSTRAP',role=ROLE,unit=UNIT,epoch=time.time())),flush=True)
os.execvpe(command[0],command,env)
''').encode()


def bundle(role='upstream', generation='owner-v1', slot=None, instance='r1', enabled_slots=DEFAULT_SLOTS):
    assert role in ('upstream', 'downstream', 'slot') and re.fullmatch(r'[a-z0-9-]{1,25}', generation)
    assert re.fullmatch(r'[a-z0-9-]{1,16}', instance) and (role != 'slot' or slot in SLOTS)
    assert set(DEFAULT_SLOTS) <= set(enabled_slots) <= set(SLOTS)
    declared = {name: SLOTS[name] for name in enabled_slots}
    name = generation + '-' + role + ('-' + slot if slot else '') + '-' + instance
    unit = 'sst-q7b-g5-relaygen-' + name + '-' + RUN_ID[:12] + '.service'
    files = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCES}
    rows = {name: dict(bytes=len(raw), sha256=old.digest(raw)) for name, raw in files.items()}
    identity = old.digest(old.encoded(dict(role=role, generation=generation, slot=slot, instance=instance, enabled_slots=sorted(enabled_slots), source_files=rows)))
    files.update({'relay_generation_bootstrap.py': bootstrap(role, unit, slot),
        'native_paths.sh': old.native_paths_source(files['pool_paths.sh']), 'slots.json': old.encoded(declared)})
    compile(files['relay_generation_bootstrap.py'], 'relay_generation_bootstrap.py', 'exec')
    manifest = dict(role=role, generation=generation, unit=unit, instance=instance, slot=slot, run_id=RUN_ID,
        directory='relay-generation-' + name, source_bundle_sha256=identity, source_files=rows,
        files={name: dict(bytes=len(raw), sha256=old.digest(raw)) for name, raw in files.items()},
        token_file=BASE + '/preparation/relay-generation-' + generation + '/relay-secret',
        owner_hard_seconds=90000, slot_hard_seconds=28800, finite_node10_slots=declared,
        no_slurm_changes=True, original_protocol_unchanged=True, secret_remote_only=True)
    return identity, files, manifest


REMOTE = r'''
import base64,fcntl,hashlib,json,os,secrets,socket,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
node='h100-10'if ACTION=='stage'or MANIFEST['role']!='upstream'else'h100-5'
assert os.geteuid()==0 and socket.gethostname().split('.')[0]==node
base=Path(PHYSICAL if ACTION=='stage'else BASE);root=base/'preparation'/MANIFEST['directory'];stage=root/IDENTITY
unit=MANIFEST['unit'];role=MANIFEST['role'];intent=root/'submit-intent.json';receipt=root/'submit-receipt.json'
def read(p):return json.loads(Path(p).read_text())if Path(p).exists()else None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 with Path(p).open('x')as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def query():
 r=subprocess.run(['systemctl','show',unit,'-p','LoadState','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','Result'],text=True,capture_output=True,timeout=20)
 return dict(returncode=r.returncode,properties=dict(line.split('=',1)for line in r.stdout.splitlines()if'='in line),stderr=r.stderr)
def verify(expected=MANIFEST,where=stage):
 assert read(where/'manifest.json')==expected
 for name,row in expected['files'].items():assert sha(where/name)==row['sha256']and(where/name).stat().st_size==row['bytes'],name
 assert read(where/'stage-receipt.json')['status']=='EXACT_RELAY_GENERATION_STAGE_SHA_PASS'
value=dict(action=ACTION,role=role,unit=unit,epoch=time.time())
if ACTION=='stage':
 assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 root.mkdir(parents=True,exist_ok=True);stage.mkdir(exist_ok=True)
 for name,encoded in FILES.items():
  p=stage/name;raw=base64.b64decode(encoded)
  if not p.exists():
   with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  assert sha(p)==MANIFEST['files'][name]['sha256'];os.chmod(p,0o555)
 if not(stage/'manifest.json').exists():put(stage/'manifest.json',MANIFEST)
 secret=Path(PHYSICAL)/Path(MANIFEST['token_file']).relative_to(BASE);secret.parent.mkdir(parents=True,exist_ok=True)
 if not secret.exists():
  fd=os.open(secret,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(fd,'w')as f:f.write(secrets.token_hex(32));f.flush();os.fsync(f.fileno())
 assert not secret.is_symlink()and secret.is_file()and secret.stat().st_mode&0o777==0o600
 if not(stage/'stage-receipt.json').exists():put(stage/'stage-receipt.json',dict(status='EXACT_RELAY_GENERATION_STAGE_SHA_PASS',publication=PUBLICATION,epoch=time.time()))
 verify();value.update(status='NEW_RELAY_CODE_STAGED_SECRET_NOT_RETURNED',stage=str(stage))
elif ACTION=='submit':
 verify();assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 with(root/'submit.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);observed=query()
  if intent.exists():value.update(status='EXISTING_INTENT_NO_START_REPLAY',intent=read(intent),receipt=read(receipt),observed=observed)
  else:
   assert observed['properties'].get('LoadState')=='not-found'
   sys.path.insert(0,str(stage));import retrieval_relay_owner as runtime
   pool=SimpleNamespace(read=read,sha=sha,RUN_ID=RUN_ID)
   if role=='slot':
    ready=runtime.bound_read(pool,DOWN_READY,DOWN_SHA);owner=ready['owner']
    assert ready['role']=='downstream'and ready['status']=='RETRIEVAL_RELAY_OWNER_READY'and ready['run_id']==RUN_ID
    runtime.same_process(owner);runtime.same_process(ready['relay']);runtime.check_terminal(pool,ready['job_ready_path'])
    runtime.bound_read(pool,ready['allocation_path'],ready['allocation_sha256'])
    assert MANIFEST['slot']in ready['slots']and ready['consumer_deadline_epoch']>time.time()+1320
    assert not(Path(ready['job_ready_path']).parent/'drain-request.json').exists()
    context=dict(downstream_owner_ready=DOWN_READY,downstream_owner_ready_sha256=DOWN_SHA)
    target=['nsenter','--target',str(owner['pid']),'--mount','--net','--','/usr/bin/python3',str(stage/'relay_generation_bootstrap.py')]
   else:
    arguments=SimpleNamespace(base=BASE,source_owner_ready=SOURCE_READY,source_owner_ready_sha256=SOURCE_SHA,
     predecessor_proof=PREDECESSOR,predecessor_sha256=PREDECESSOR_SHA)
    ready=runtime.source_ready(pool,arguments,local=role=='upstream')
    assert ready['consumer_deadline_epoch']>time.time()+1320
    context=dict(source_owner_ready=SOURCE_READY,source_owner_ready_sha256=SOURCE_SHA)
    if role=='upstream':
     import ipaddress
     address=ipaddress.ip_address(LISTEN_HOST);assert address.version==4 and address.is_private and not address.is_loopback
     context['listen_host']=LISTEN_HOST
    else:
     upstream=runtime.bound_read(pool,UP_READY,UP_SHA)
     assert upstream['role']=='upstream'and upstream['run_id']==RUN_ID and upstream['source_owner']==ready['owner']
     runtime.check_terminal(pool,upstream['job_ready_path']);runtime.verify_predecessor(pool,arguments)
     context.update(upstream_owner_ready=UP_READY,upstream_owner_ready_sha256=UP_SHA,
      predecessor_proof=PREDECESSOR,predecessor_sha256=PREDECESSOR_SHA)
    target=['unshare','--mount','--propagation','private','/bin/bash','-c',base64.b64decode(FILES['native_paths.sh']).decode(),
     'relay-generation-paths',node,'/usr/bin/python3',str(stage/'relay_generation_bootstrap.py')]
   put(root/'submit-context.json',context)
   command=['systemd-run','--unit='+unit,'--property=KillMode=control-group','--property=TimeoutStopSec=60','--property=Restart=no',
    '--property=CPUQuota='+('800%'if role=='slot'else'100%'),'--property=MemoryMax='+('96G'if role=='slot'else'1G'),
    '--property=RuntimeMaxSec='+('28800'if role=='slot'else'90000'),'--setenv=Q7B_NATIVE_UNIT_INSTANCE='+unit,*target]
   put(intent,dict(intent_id=INTENT,unit=unit,manifest=MANIFEST,context=context,argv=command,publication=PUBLICATION,epoch=time.time()))
   try:
    r=subprocess.run(command,text=True,capture_output=True,timeout=45)
    result=dict(status='OWN_RELAY_GENERATION_STARTED_ONCE'if r.returncode==0 else'OWN_START_FAILED_NO_REPLAY',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
   except subprocess.TimeoutExpired as error:result=dict(status='OWN_START_OUTCOME_UNKNOWN_QUERY_SAME_UNIT',error=str(error))
   result.update(observed=query(),epoch=time.time());put(receipt,result);value.update(status=result['status'],receipt=result)
elif ACTION=='drain':
 saved=read(intent);assert saved and saved['unit']==unit
 verify(saved['manifest'],root/saved['manifest']['source_bundle_sha256'])
 observed=query();pid=int(observed['properties'].get('MainPID','0'));assert pid and observed['properties'].get('ActiveState')=='active'
 if role=='slot':
  jobs=list((base/'pool/jobs'/('job-relay-slot-'+MANIFEST['slot']+'-'+str(pid))).glob('attempt-*'));assert len(jobs)==1
  job=jobs[0];allocation=read(job/'allocation.json');marker=job/'drain.json'
  request=dict(status='DRAIN_AT_WHOLE_BATCH_BOUNDARY',run_id=RUN_ID,job_id=allocation['job_id'],allocation_sha256=sha(job/'allocation.json'),epoch=time.time())
 else:
  ready=read(root/'owner-ready.json');assert ready['owner']['pid']==pid and ready['owner']['unit']==unit
  job=Path(ready['job_ready_path']).parent;allocation=read(job/'allocation.json');marker=job/'drain-owner.json'
  request=dict(status='DRAIN_RETRIEVAL_OWNER',run_id=RUN_ID,owner_pid=pid,owner_allocation_sha256=sha(job/'allocation.json'),epoch=time.time())
 assert allocation['controller_pid']==pid and allocation['native_unit']==unit
 if not marker.exists():put(marker,request)
 value.update(status='BOUND_WHOLE_BATCH_DRAIN_RECORDED_NO_SIGNAL',marker=str(marker),request=read(marker))
else:value.update(status='READ_ONLY_RELAY_GENERATION_STATUS',observed=query(),intent=read(intent),receipt=read(receipt),owner_ready=read(root/'owner-ready.json'))
print(json.dumps(value))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=('upstream', 'downstream', 'slot'))
    parser.add_argument('action', choices=('plan', 'stage', 'submit', 'status', 'drain'))
    parser.add_argument('--generation', default='owner-v1')
    parser.add_argument('--instance', default='r1')
    parser.add_argument('--slot', choices=tuple(SLOTS))
    parser.add_argument('--enabled-slots', default=','.join(DEFAULT_SLOTS))
    for key in ('source-owner-ready', 'source-owner-ready-sha256', 'upstream-owner-ready', 'upstream-owner-ready-sha256',
                'downstream-owner-ready', 'downstream-owner-ready-sha256', 'predecessor-proof', 'predecessor-sha256',
                'listen-host', 'intent-id', 'published-commit', 'published-ref'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    identity, files, manifest = bundle(args.role, args.generation, args.slot, args.instance, tuple(args.enabled_slots.split(',')))
    if args.action == 'plan':print(json.dumps(manifest, indent=2));return
    publication = None
    if args.action in ('stage', 'submit'):
        assert re.fullmatch(r'[A-Za-z0-9._-]{1,120}', args.intent_id or '')
        publication = old.verify_publication(args.published_commit, args.published_ref, manifest['source_files'])
    values = dict(BASE=BASE, PHYSICAL=PHYSICAL, RUN_ID=RUN_ID, MANIFEST=manifest, IDENTITY=identity,
        ACTION=args.action, INTENT=args.intent_id, PUBLICATION=publication,
        SOURCE_READY=args.source_owner_ready, SOURCE_SHA=args.source_owner_ready_sha256,
        UP_READY=args.upstream_owner_ready, UP_SHA=args.upstream_owner_ready_sha256,
        DOWN_READY=args.downstream_owner_ready, DOWN_SHA=args.downstream_owner_ready_sha256,
        PREDECESSOR=args.predecessor_proof, PREDECESSOR_SHA=args.predecessor_sha256, LISTEN_HOST=args.listen_host,
        FILES={name: base64.b64encode(raw).decode() for name, raw in files.items()})
    script = '\n'.join(key + '=' + repr(value) for key, value in values.items()) + '\n' + REMOTE
    node = 'h100-10' if args.action == 'stage' or args.role != 'upstream' else 'h100-5'
    if args.action != 'stage':
        command = ['unshare', '--mount', '--propagation', 'private', '/bin/bash', '-c', files['native_paths.sh'].decode(),
                   'relay-control-paths', node, '/usr/bin/python3', '-']
        script = 'import os,subprocess,sys\nr=subprocess.run(' + repr(command) + ',input=' + repr(script) + \
            ',env=dict(os.environ,Q7B_NATIVE_UNIT_INSTANCE=' + repr(manifest['unit']) + '),text=True,capture_output=True,timeout=100)\nprint(r.stdout,end="");print(r.stderr,end="",file=sys.stderr);raise SystemExit(r.returncode)\n'
    output = ROOT / 'work' / ('qwen7b-relay-generation-%s-%d.json' % (args.action, time.time_ns()))
    try:
        result = old.transport(script, node)
    except subprocess.TimeoutExpired:
        output.write_text(json.dumps(dict(status='TRANSPORT_UNKNOWN_QUERY_SAME_UNIT_NO_REPLAY', unit=manifest['unit'])));raise
    value = dict(action=args.action, role=args.role, unit=manifest['unit'], returncode=result.returncode,
                 stdout=result.stdout, stderr=result.stderr, epoch=time.time())
    output.write_text(json.dumps(value, indent=2) + '\n');print(json.dumps(value, indent=2))
    if result.returncode:raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
