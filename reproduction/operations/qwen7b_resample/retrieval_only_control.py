"""Independent next-generation retrieval/slot control; plan is local only.

Actual owner submit requires an externally reviewed predecessor release proof.
No action edits old stages, resubmits old intents or changes Slurm queues.
"""
import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import time
from . import shared_vram_control as old
from .retrieval_only_owner import SLOTS, DEFAULT_SLOTS

ROOT, BASE, PHYSICAL, RUN_ID = old.ROOT, old.BASE, old.PHYSICAL, old.RUN_ID
SOURCES = ('retrieval_only_owner.py', 'retrieval_only_control.py', 'source_slot_consumer_worker.py',
           'source_slot_control.py', 'draining_pool_worker.py', 'shared_vram_worker.py',
           'shared_vram_retriever.py', 'shared_vram_vllm_adapter.py', 'pool_worker.py',
           'relay_consumer_worker.py', 'retrieval_relay.py', 'pool_paths.sh')


def bootstrap(role, unit, slot):
    return ('BASE=' + repr(BASE) + '\nFROZEN_REPO=' + repr(old.FROZEN_REPO) + '\nUNIT=' + repr(unit)
        + '\nROLE=' + repr(role) + '\nSLOT=' + repr(slot) + '\n' + r'''
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
stage=Path(__file__).resolve().parent;root=stage.parent
assert os.geteuid()==0 and socket.gethostname().split('.')[0]=='h100-5'
manifest=json.loads((stage/'manifest.json').read_text())
for name,row in manifest['files'].items():
 data=(stage/name).read_bytes();assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],name
detail=subprocess.check_output(['systemctl','show',UNIT,'-p','Id','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','MemoryMax','-p','RuntimeMaxUSec','-p','ActiveEnterTimestampMonotonic'],text=True)
props=dict(line.split('=',1)for line in detail.splitlines()if'='in line)
assert props['Id']==UNIT and props['MainPID']==str(os.getpid())and props['ActiveState']=='active'
assert props['ControlGroup']in Path('/proc/self/cgroup').read_text()
launch=Path(BASE)/'retrieval-owner-generation/launch'/('native-'+str(os.getpid()));launch.mkdir(parents=True,exist_ok=False)
log=(launch/'bootstrap.log').open('ab',buffering=0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
hard=90000 if ROLE=='owner'else 28800;memory=171798691840 if ROLE=='owner'else 103079215104
assert int(props['MemoryMax'])==memory
start_us=int(props['ActiveEnterTimestampMonotonic']);assert start_us>0
remaining=hard-(time.monotonic()-start_us/1000000);assert 0<remaining<=hard
evidence=launch/'allocation-evidence.json'
evidence.write_text(json.dumps(dict(mode='NATIVE_SHARED_VRAM'if ROLE=='owner'else'NATIVE_DECLARED_MODEL_SLOT',
 node='h100-5',unit=UNIT,main_pid=os.getpid(),unit_show=detail,cgroup=props['ControlGroup'],captured_epoch=time.time(),
 unit_hard_end_epoch=time.time()+remaining,
 resource_limits=dict(cpu_quota_percent=1600 if ROLE=='owner'else 800,memory_max_bytes=memory,runtime_max_seconds=hard),
 source_bundle_sha256=manifest['source_bundle_sha256'],original_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))));os.chmod(evidence,0o400)
env=dict(os.environ,PYTHONPATH=FROZEN_REPO+':/dataset/shaoshitong/Search-G1-common8/retriever-gpu-lib',
 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',SEARCH_G1_B_PREFIX_TOKENS='20',
 SEARCH_G1_PHYSICAL_WORLD_SIZE='4',SEARCH_G1_BASE=BASE,HF_HOME=BASE+'/cache/huggingface',
 TORCH_EXTENSIONS_DIR=BASE+'/cache/torch_extensions',TMPDIR=str(launch/'tmp'),
 VLLM_ATTENTION_BACKEND='XFORMERS',VLLM_HOST_IP='127.0.0.1',NCCL_NVLS_ENABLE='0',NCCL_BUFFSIZE='1048576',
 NCCL_MAX_NCHANNELS='4',NCCL_DEBUG='INFO',
 CPATH='/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include:'
 '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/python3.10:'
 '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/x86_64-linux-gnu/python3.10')
for key in('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):env.pop(key,None)
for key in('HF_HOME','TORCH_EXTENSIONS_DIR','TMPDIR'):Path(env[key]).mkdir(parents=True,exist_ok=True)
python='/dataset/shaoshitong/Search-G1-common8/env/bin/python'
if ROLE=='owner':
 env['CUDA_VISIBLE_DEVICES']='3,4,6,7'
 command=['unshare','--net','/bin/bash','-c','set -euo pipefail; ip link set lo up; exec "$@"','retrieval-owner-net',
  python,'-u',str(stage/'retrieval_only_owner.py'),'--base',BASE,'--allocation-evidence',str(evidence),
  '--slot-manifest',str(stage/'slots.json'),'--owner-seconds','86400','--pool-worker',str(stage/'pool_worker.py'),
  '--retriever-script',str(stage/'shared_vram_retriever.py')]
else:
 context=json.loads((root/'submit-context.json').read_text());path=Path(context['owner_ready'])
 raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==context['owner_ready_sha256']
 ready=json.loads(raw);slot=ready['slots'][SLOT]
 run_seconds=min(28500,int(ready['consumer_deadline_epoch']-time.time()-60),int(remaining-90));assert run_seconds>1260
 env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,slot['indices'])),Q7B_SLOT_NAME=SLOT,
  Q7B_SLOT_OWNER_READY=str(path),Q7B_SLOT_OWNER_READY_SHA256=context['owner_ready_sha256'],
  Q7B_SLOT_OWNER_DRAIN=str(path.parent/'drain-request.json'))
 command=[python,'-u',str(stage/'source_slot_consumer_worker.py'),'--base',BASE,'--allocation-evidence',str(evidence),
  '--pool-worker',str(stage/'draining_pool_worker.py'),'--relay-script',str(stage/'retrieval_relay.py'),
  '--upstream-ready',str(path.parent/'retriever-source-ready.json'),'--reuse-local-relay-ready',str(path),
  '--reuse-local-relay-sha256',context['owner_ready_sha256'],'--reuse-local-relay-unit',ready['owner']['unit'],
  '--model-gpus',','.join(map(str,slot['indices'])),'--tp',str(slot['tp']),
  '--model-max-seqs',str(slot['seqs']),'--model-batch-tokens',str(slot['tokens']),
  '--model-kv-block-cap',str(slot['blocks']),'--model-reserve-gib',str(slot['reserve']),
  '--run-seconds',str(run_seconds),'--batch-reserve-seconds','1200']
os.chdir(FROZEN_REPO)
print(json.dumps(dict(status='EXACT_NEW_GENERATION_BOOTSTRAP',role=ROLE,unit=UNIT,epoch=time.time())),flush=True)
os.execvpe(command[0],command,env)
''').encode()


def bundle(role='owner', generation='owner-v1', enabled_slots=DEFAULT_SLOTS, slot=None, instance='r1'):
    assert re.fullmatch(r'[a-z0-9-]{1,30}', generation) and re.fullmatch(r'[a-z0-9-]{1,20}', instance)
    assert role in ('owner', 'slot') and set(enabled_slots) <= set(SLOTS) and enabled_slots
    assert role == 'owner' or slot in SLOTS
    declared_slots = enabled_slots if role == 'owner' else (slot,)
    name = generation if role == 'owner' else generation + '-' + slot + '-' + instance
    directory = 'retrieval-generation-' + name
    unit = 'sst-q7b-g5-' + name + '-' + RUN_ID[:12] + '.service'
    files = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCES}
    rows = {name: dict(bytes=len(raw), sha256=old.digest(raw)) for name, raw in files.items()}
    identity = old.digest(old.encoded(dict(role=role, generation=generation, slot=slot, instance=instance,
                                         enabled_slots=sorted(declared_slots), source_files=rows)))
    files.update({'generation_bootstrap.py': bootstrap(role, unit, slot),
                  'native_paths.sh': old.native_paths_source(files['pool_paths.sh']),
                  'slots.json': old.encoded({name: SLOTS[name] for name in declared_slots})})
    compile(files['generation_bootstrap.py'], 'generation_bootstrap.py', 'exec')
    manifest = dict(source_bundle_sha256=identity, role=role, generation=generation, directory=directory,
        unit=unit, slot=slot, instance=instance, run_id=RUN_ID, seed=20261021,
        slots={name: SLOTS[name] for name in declared_slots}, source_files=rows,
        files={name: dict(bytes=len(raw), sha256=old.digest(raw)) for name, raw in files.items()},
        owner_seconds=86400, owner_hard_seconds=90000, consumer_max_seconds=28500,
        consumer_owner_margin_seconds=1800, no_slurm_changes=True, old_stages_unchanged=True)
    assert sum(len(raw) for raw in files.values()) < 256 * 1024
    return identity, files, manifest


REMOTE = r'''
import base64,fcntl,hashlib,json,os,socket,subprocess,time
from pathlib import Path
assert os.geteuid()==0 and socket.gethostname().split('.')[0]==('h100-10'if ACTION=='stage'else'h100-5')
base=Path(PHYSICAL if ACTION=='stage'else BASE);root=base/'preparation'/MANIFEST['directory'];stage=root/IDENTITY
UNIT=MANIFEST['unit'];ROLE=MANIFEST['role']
def read(p):return json.loads(Path(p).read_text())if Path(p).exists()else None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def birth(pid):
 try:return Path('/proc/%d/stat'%int(pid)).read_text().rsplit(')',1)[1].split()[19]
 except(FileNotFoundError,ProcessLookupError):return None
def query(unit):
 r=subprocess.run(['systemctl','show',unit,'-p','Id','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','LoadState','-p','Result','-p','ExecMainStatus'],text=True,capture_output=True,timeout=20)
 return dict(returncode=r.returncode,properties=dict(x.split('=',1)for x in r.stdout.splitlines()if'='in x),stderr=r.stderr)
def verify(expected=MANIFEST,source_stage=stage):
 assert read(source_stage/'manifest.json')==expected
 for name,row in expected['files'].items():
  raw=(source_stage/name).read_bytes();assert len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['sha256'],name
 assert(read(source_stage/'stage-receipt.json')or{}).get('status')=='EXACT_OWNER_GENERATION_STAGE_SHA_PASS'
value=dict(action=ACTION,unit=UNIT,role=ROLE,epoch=time.time())
intent=root/'submit-intent.json';receipt=root/'submit-receipt.json'
if ACTION=='stage':
 assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 root.mkdir(parents=True,exist_ok=True);stage.mkdir(exist_ok=True)
 for name,text in FILES.items():
  p=stage/name;raw=base64.b64decode(text)
  if not p.exists():
   with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  assert sha(p)==MANIFEST['files'][name]['sha256'];os.chmod(p,0o555)
 if not(stage/'manifest.json').exists():put(stage/'manifest.json',MANIFEST)
 if not(stage/'stage-receipt.json').exists():put(stage/'stage-receipt.json',dict(status='EXACT_OWNER_GENERATION_STAGE_SHA_PASS',publication=PUBLICATION,epoch=time.time()))
 verify();value.update(status='NEW_GENERATION_CODE_STAGED',stage=str(stage))
elif ACTION=='submit':
 verify();assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 with(root/'submit.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);observed=query(UNIT)
  if intent.exists():value.update(status='EXISTING_INTENT_NO_START_REPLAY',intent=read(intent),receipt=read(receipt),observed=observed)
  else:
   assert observed['properties'].get('LoadState')=='not-found'
   context={}
   if ROLE=='owner':
    assert PREDECESSOR and re.fullmatch(r'[0-9a-f]{64}',PREDECESSOR_SHA or'')and sha(PREDECESSOR)==PREDECESSOR_SHA
    proof=read(PREDECESSOR);assert proof['status']=='OWN_PREDECESSOR_GENERATION_TERMINAL'and proof['run_id']==RUN_ID
    assert proof['source_unit']==OLD_SOURCE_UNIT and proof['source_pid']==1490214 and proof['source_birth']=='6807965'
    assert '1490214'in proof['original_owned_births']and'1490465'in proof['original_owned_births']
    for pid,ticks in proof['original_owned_births'].items():assert birth(pid)!=str(ticks),'An original generation process is still live'
    for row in proof['units']:
     q=query(row['unit']);p=q['properties'];assert q['returncode']==0 and p.get('MainPID')=='0'and p.get('ActiveState')in('inactive','failed')
    assert any(row['unit']==OLD_SOURCE_UNIT for row in proof['units'])
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=20)
    assert not set(proof['original_owned_births'])&set(apps.split()),'Original generation GPU context remains'
    context=dict(predecessor_proof=PREDECESSOR,predecessor_sha256=PREDECESSOR_SHA)
    target=['unshare','--mount','--propagation','private','/bin/bash','-c',base64.b64decode(FILES['native_paths.sh']).decode(),
     'owner-generation-paths','h100-5','/usr/bin/python3',str(stage/'generation_bootstrap.py')]
   else:
    assert OWNER_READY and sha(OWNER_READY)==OWNER_READY_SHA
    ready=read(OWNER_READY);owner=ready['owner'];q=query(owner['unit'])
    assert ready['status']=='RETRIEVAL_ONLY_OWNER_READY'and ready['run_id']==RUN_ID
    assert q['properties'].get('MainPID')==str(owner['pid'])and q['properties'].get('ActiveState')=='active'
    assert birth(owner['pid'])==owner['birth']and sha(ready['allocation_path'])==ready['allocation_sha256']
    assert MANIFEST['slot']in ready['slots']and ready['consumer_deadline_epoch']>time.time()+1320
    assert not(Path(OWNER_READY).parent/'drain-request.json').exists()
    context=dict(owner_ready=OWNER_READY,owner_ready_sha256=OWNER_READY_SHA,slot=MANIFEST['slot'])
    target=['nsenter','--target',str(owner['pid']),'--mount','--net','--','/usr/bin/python3',str(stage/'generation_bootstrap.py')]
   put(root/'submit-context.json',context)
   command=['systemd-run','--unit='+UNIT,'--property=KillMode=control-group','--property=TimeoutStopSec=60','--property=Restart=no',
    '--property=CPUQuota='+('1600%'if ROLE=='owner'else'800%'),'--property=MemoryMax='+('160G'if ROLE=='owner'else'96G'),
    '--property=RuntimeMaxSec='+('90000'if ROLE=='owner'else'28800'),'--setenv=Q7B_NATIVE_UNIT_INSTANCE='+UNIT,*target]
   put(intent,dict(intent_id=INTENT,unit=UNIT,argv=command,manifest=MANIFEST,publication=PUBLICATION,context=context,epoch=time.time()))
   try:
    r=subprocess.run(command,text=True,capture_output=True,timeout=45)
    result=dict(status='OWN_GENERATION_UNIT_STARTED_ONCE'if r.returncode==0 else'OWN_START_FAILED_NO_REPLAY',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
   except subprocess.TimeoutExpired as error:result=dict(status='OWN_START_OUTCOME_UNKNOWN_QUERY_SAME_UNIT',error=str(error))
   result.update(observed=query(UNIT),epoch=time.time());put(receipt,result);value.update(status=result['status'],receipt=result)
elif ACTION=='drain':
 saved=read(intent);assert saved and saved['unit']==UNIT and saved['manifest']['role']==ROLE
 verify(saved['manifest'],root/saved['manifest']['source_bundle_sha256'])
 observed=query(UNIT);pid=observed['properties'].get('MainPID','0');assert pid!='0'and observed['properties'].get('ActiveState')=='active'
 pattern='job-retrieval-owner-'+pid if ROLE=='owner'else'job-source-slot-'+MANIFEST['slot']+'-'+pid
 attempts=list((base/'pool/jobs'/pattern).glob('attempt-*'));assert len(attempts)==1
 job=attempts[0];allocation=read(job/'allocation.json');assert allocation['controller_pid']==int(pid)and allocation['native_unit']==UNIT
 if ROLE=='owner':
  marker=job/'drain-owner.json';request=dict(status='DRAIN_RETRIEVAL_OWNER',run_id=RUN_ID,owner_pid=int(pid),owner_allocation_sha256=sha(job/'allocation.json'),epoch=time.time())
 else:
  marker=job/'drain.json';request=dict(status='DRAIN_AT_WHOLE_BATCH_BOUNDARY',run_id=RUN_ID,job_id=allocation['job_id'],allocation_sha256=sha(job/'allocation.json'),epoch=time.time())
 if not marker.exists():put(marker,request)
 value.update(status='DRAIN_REQUEST_RECORDED_NO_SIGNAL',marker=str(marker),request=read(marker))
else:
 value.update(status='READ_ONLY_GENERATION_STATUS',observed=query(UNIT),intent=read(intent),receipt=read(receipt),context=read(root/'submit-context.json'))
print(json.dumps(value))
'''


def main(role='owner'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'stage', 'submit', 'status', 'drain'))
    parser.add_argument('--generation', default='owner-v1')
    parser.add_argument('--enabled-slots', default=','.join(DEFAULT_SLOTS))
    parser.add_argument('--slot', choices=tuple(SLOTS))
    parser.add_argument('--instance', default='r1')
    parser.add_argument('--owner-ready')
    parser.add_argument('--owner-ready-sha256')
    parser.add_argument('--predecessor-proof')
    parser.add_argument('--predecessor-sha256')
    parser.add_argument('--intent-id')
    parser.add_argument('--published-commit')
    parser.add_argument('--published-ref')
    args = parser.parse_args()
    identity, files, manifest = bundle(role, args.generation, tuple(args.enabled_slots.split(',')), args.slot, args.instance)
    if args.action == 'plan':
        print(json.dumps(manifest, indent=2));return
    publication = None
    if args.action in ('stage', 'submit'):
        assert re.fullmatch(r'[A-Za-z0-9._-]{1,120}', args.intent_id or '')
        publication = old.verify_publication(args.published_commit, args.published_ref, manifest['source_files'])
    values = dict(PHYSICAL=PHYSICAL, BASE=BASE, RUN_ID=RUN_ID, MANIFEST=manifest, IDENTITY=identity,
        ACTION=args.action, INTENT=args.intent_id, PUBLICATION=publication, OLD_SOURCE_UNIT=old.UNIT,
        PREDECESSOR=args.predecessor_proof, PREDECESSOR_SHA=args.predecessor_sha256,
        OWNER_READY=args.owner_ready, OWNER_READY_SHA=args.owner_ready_sha256,
        FILES={key: base64.b64encode(raw).decode() for key, raw in files.items()})
    script = '\n'.join(key + '=' + repr(value) for key, value in values.items()) + '\nimport re\n' + REMOTE
    node = 'h100-10' if args.action == 'stage' else 'h100-5'
    if node == 'h100-5':
        command = ['unshare', '--mount', '--propagation', 'private', '/bin/bash', '-c', files['native_paths.sh'].decode(),
                   'generation-control-paths', 'h100-5', '/usr/bin/python3', '-']
        script = 'import os,subprocess,sys\nr=subprocess.run(' + repr(command) + ',input=' + repr(script) + \
            ',env=dict(os.environ,Q7B_NATIVE_UNIT_INSTANCE=' + repr(manifest['unit']) + '),text=True,capture_output=True,timeout=95)\nprint(r.stdout,end="");print(r.stderr,end="",file=sys.stderr);raise SystemExit(r.returncode)\n'
    record = ROOT / 'work' / ('qwen7b-generation-%s-%d.json' % (args.action, time.time_ns()))
    try:
        result = old.transport(script, node)
    except subprocess.TimeoutExpired:
        record.write_text(json.dumps(dict(status='TRANSPORT_UNKNOWN_QUERY_SAME_UNIT_NO_REPLAY', unit=manifest['unit'])));raise
    value = dict(action=args.action, unit=manifest['unit'], returncode=result.returncode,
                 stdout=result.stdout, stderr=result.stderr, receipt=str(record), epoch=time.time())
    record.write_text(json.dumps(value, indent=2) + '\n');print(json.dumps(value, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
