"""Once-only additional h100-10 consumer; preserves the healthy h100-5 unit.

Only code and small receipts cross the Mac. A remote-only secret authenticates
same-cluster retrieval forwarding; the original URL, index, seed and batches
are unchanged. The four-card model allocation is proven only by real sampling.
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
DIRECTORY = 'relay-consumer10-control-v1'
UNIT = 'sst-q7b-g5-consumer10-v3-' + RUN_ID[:12] + '.service'
UPSTREAM_UNIT = 'sst-q7b-retrieval-relay5-v1-' + RUN_ID[:12] + '.service'
SOURCE_UNIT = 'sst-q7b-g5-sharedvram-v6-' + RUN_ID[:12] + '.service'
SOURCES = ('relay_consumer_control.py', 'relay_consumer_worker.py', 'retrieval_relay.py',
           'shared_vram_worker.py', 'shared_vram_vllm_adapter.py', 'pool_worker.py', 'pool_paths.sh')


def bootstrap():
    return ('BASE=' + repr(BASE) + '\nUNIT=' + repr(UNIT) + '\nFROZEN_REPO=' + repr(old.FROZEN_REPO) + '\n' + r'''
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
stage=Path(__file__).resolve().parent;root=stage.parent
manifest=json.loads((stage/'manifest.json').read_text())
assert os.geteuid()==0 and socket.gethostname().split('.')[0]=='h100-10'
assert os.stat('/proc/self/ns/mnt').st_ino!=os.stat('/proc/1/ns/mnt').st_ino
assert os.stat('/proc/self/ns/net').st_ino==os.stat('/proc/1/ns/net').st_ino
for name,row in manifest['files'].items():
 p=stage/name;data=p.read_bytes();assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],name
context=json.loads((root/'source-context.json').read_text())
run_seconds=min(28500,int(context['source_sampling_deadline_epoch']-time.time()-600))
assert run_seconds>900,'Too little original retriever runtime remains for a new consumer'
launch=Path(BASE)/'relay-consumer10/launch'/('native-'+str(os.getpid()));launch.mkdir(parents=True,exist_ok=False)
log=(launch/'bootstrap.log').open('ab',buffering=0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
detail=subprocess.check_output(['systemctl','show',UNIT,'-p','Id','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','MemoryMax','-p','RuntimeMaxUSec'],text=True)
values=dict(x.split('=',1)for x in detail.splitlines()if'='in x)
assert values['Id']==UNIT and int(values['MainPID'])==os.getpid() and values['ControlGroup']in Path('/proc/self/cgroup').read_text()
evidence=launch/'allocation-evidence.json'
evidence.write_text(json.dumps(dict(mode='NATIVE_MODEL_ONLY_SHARED_VRAM',node='h100-10',unit=UNIT,
 main_pid=os.getpid(),cgroup=values['ControlGroup'],unit_show=detail,captured_epoch=time.time(),
 resource_limits=dict(cpu_quota_percent=800,memory_max_bytes=103079215104,runtime_max_seconds=28800),
 communication_environment=dict(NCCL_NVLS_ENABLE='0',NCCL_DEBUG='INFO'),
 original_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))));os.chmod(evidence,0o400)
env=dict(os.environ,PYTHONPATH=FROZEN_REPO+':/dataset/shaoshitong/Search-G1-common8/retriever-gpu-lib',
 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',SEARCH_G1_B_PREFIX_TOKENS='20',
 SEARCH_G1_PHYSICAL_WORLD_SIZE='4',SEARCH_G1_BASE=BASE,HF_HOME=BASE+'/cache/huggingface',
 TORCH_EXTENSIONS_DIR=BASE+'/cache/torch_extensions',TMPDIR=str(launch/'tmp'),
 CUDA_VISIBLE_DEVICES='0,1,2,3',NCCL_NVLS_ENABLE='0',NCCL_DEBUG='INFO',VLLM_ATTENTION_BACKEND='XFORMERS',VLLM_HOST_IP='127.0.0.1',
 CPATH='/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include:'
 '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/python3.10:'
 '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/x86_64-linux-gnu/python3.10')
for key in('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):env.pop(key,None)
for key in('HF_HOME','TORCH_EXTENSIONS_DIR','TMPDIR'):Path(env[key]).mkdir(parents=True,exist_ok=True)
os.chdir(FROZEN_REPO)
command=['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-u',str(stage/'relay_consumer_worker.py'),
 '--base',BASE,'--allocation-evidence',str(evidence),'--pool-worker',str(stage/'pool_worker.py'),
 '--relay-script',str(stage/'retrieval_relay.py'),'--token-file',str(root/'relay-secret'),
 '--upstream-ready',str(root/'upstream-ready.json'),'--model-gpus','0,1,2,3',
 '--model-max-seqs','8','--model-batch-tokens','4096','--model-kv-block-cap','1024','--model-reserve-gib','.25',
 '--run-seconds',str(run_seconds),'--batch-reserve-seconds','300']
print(json.dumps(dict(status='MODEL_ONLY_CONSUMER_BOOTSTRAP',actual_pid=os.getpid(),source_context=context,
 model_gpus=[0,1,2,3],tensor_parallel_size=4,source_retriever_owned=False,epoch=time.time())),flush=True)
os.execvpe(command[0],command,env)
''').encode()


def bundle():
    files = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCES}
    rows = {name: dict(bytes=len(data), sha256=old.digest(data)) for name, data in files.items()}
    identity = old.digest(old.encoded(dict(mode=DIRECTORY, source_files=rows)))
    files.update({'relay_consumer_bootstrap.py': bootstrap(),
                  'consumer_paths.sh': old.native_paths_source(files['pool_paths.sh'])})
    manifest = dict(source_bundle_sha256=identity, run_id=RUN_ID, seed=20261021,
        unit=UNIT, upstream_unit=UPSTREAM_UNIT, unchanged_source_unit=SOURCE_UNIT,
        source_files=rows, files={k: dict(bytes=len(v), sha256=old.digest(v)) for k, v in files.items()},
        same_cluster_retrieval_only=True, original_index_reused_without_reencoding=True,
        model_gpus=[0,1,2,3], tensor_parallel_size=4, no_actor_or_probe_updates=True,
        communication_environment=dict(NCCL_NVLS_ENABLE='0',NCCL_DEBUG='INFO'))
    return identity, files, manifest


REMOTE = r'''
import base64,fcntl,hashlib,ipaddress,json,os,secrets,socket,subprocess,time
from pathlib import Path
base=Path(BASE if ACTION=='start-upstream'else PHYSICAL);root=base/'preparation'/DIRECTORY;stage=root/IDENTITY
def read(p):return json.loads(p.read_text())if p.exists()else None
def put(p,v):
 t=p.with_name(p.name+'.tmp-'+str(time.time_ns()))
 with t.open('x')as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def query(unit):
 r=subprocess.run(['systemctl','show',unit,'-p','LoadState','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','Result','-p','ExecMainStatus'],capture_output=True,text=True,timeout=20)
 return dict(unit=unit,returncode=r.returncode,properties=dict(x.split('=',1)for x in r.stdout.splitlines()if'='in x),stderr=r.stderr)
def verify():
 assert read(stage/'manifest.json')==MANIFEST
 for name,row in MANIFEST['files'].items():
  p=stage/name;data=p.read_bytes();assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],name
 assert(read(stage/'stage-receipt.json')or{}).get('status')=='EXACT_SMALL_RELAY_CODE_FULL_SHA_PASS'
value=dict(action=ACTION,unit=UNIT,upstream_unit=UPSTREAM_UNIT,bundle_sha256=IDENTITY,epoch=time.time())
if ACTION=='stage':
 assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 root.mkdir(parents=True,exist_ok=True);stage.mkdir(exist_ok=True)
 for name,encoded in FILES.items():
  p=stage/name;data=base64.b64decode(encoded)
  if not p.exists():
   with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  assert hashlib.sha256(p.read_bytes()).hexdigest()==MANIFEST['files'][name]['sha256'];os.chmod(p,0o555)
 if not(stage/'manifest.json').exists():put(stage/'manifest.json',MANIFEST)
 secret=root/'relay-secret'
 if not secret.exists():
  fd=os.open(secret,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
  with os.fdopen(fd,'w')as f:f.write(secrets.token_hex(32));f.flush();os.fsync(f.fileno())
 assert secret.is_file()and not secret.is_symlink()and secret.stat().st_mode&0o777==0o600
 put(stage/'stage-receipt.json',dict(status='EXACT_SMALL_RELAY_CODE_FULL_SHA_PASS',files=MANIFEST['files'],publication=PUBLICATION,epoch=time.time()))
 verify();value.update(status='CODE_STAGED_REMOTE_SECRET_NOT_RETURNED',stage=str(stage),files=MANIFEST['files'])
elif ACTION in('start-upstream','submit'):
 verify();assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 source=ACTION=='start-upstream';unit=UPSTREAM_UNIT if source else UNIT
 name='upstream'if source else'native-v3';intent=root/(name+'-submit-intent.json');receipt=root/(name+'-submit-receipt.json')
 with(root/(name+'-submit.lock')).open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);old_intent=read(intent);observed=query(unit)
  value['observed']=observed
  if old_intent:
   value.update(status='EXISTING_INTENT_NO_START_REPLAY',intent=old_intent,receipt=read(receipt))
  else:
   assert observed['properties'].get('LoadState')=='not-found','Unique ownunit is not confirmed absent'
   shared_stage=Path(BASE)/'preparation'/DIRECTORY/IDENTITY;shared_root=shared_stage.parent
   if source:
    original=query(SOURCE_UNIT);assert original['properties'].get('MainPID')=='1490214'and original['properties'].get('ActiveState')=='active'
    def birth(pid):return Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
    assert SOURCE_UNIT in Path('/proc/1490465/cgroup').read_text()
    context=dict(source_unit=SOURCE_UNIT,source_main_pid=1490214,source_main_birth=birth(1490214),
     source_pid=1490465,source_birth=birth(1490465),source_netns_inode=os.stat('/proc/1490465/ns/net').st_ino)
    attempts=list((Path(BASE)/'pool/jobs/job-shared5-1490214').glob('attempt-*'));assert len(attempts)==1
    source_progress=read(attempts[0]/'progress.json');assert source_progress['completed_own_batches']>=2
    context['source_sampling_deadline_epoch']=source_progress['deadline_epoch'];put(root/'source-context.json',context)
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM)as probe:
     probe.connect((socket.gethostbyname('h100-10'),2049));listen=probe.getsockname()[0]
    address=ipaddress.ip_address(listen);assert address.is_private and not address.is_loopback
    command=['systemd-run','--unit='+unit,'--property=CPUQuota=100%','--property=MemoryMax=512M',
     '--property=RuntimeMaxSec=28800','--property=KillMode=control-group','--setenv=CUDA_VISIBLE_DEVICES=',
     'nsenter','--target','1490214','--mount','--','/usr/bin/python3','-u',str(shared_stage/'retrieval_relay.py'),
     'upstream','--listen-host',listen,'--listen-port','0','--log-file',str(shared_root/'upstream.log'),
     '--source-pid','1490465','--source-birth',context['source_birth'],'--source-netns-inode',str(context['source_netns_inode']),
     '--token-file',str(shared_root/'relay-secret'),'--ready-file',str(shared_root/'upstream-ready.json')]
   else:
    previous=query('sst-q7b-g5-consumer10-v2-'+RUN_ID[:12]+'.service')
    assert previous['properties'].get('MainPID')=='0'and previous['properties'].get('ActiveState')in('inactive','failed')
    terminal=read(root/'v2-terminal/complete.json')
    assert terminal and terminal['status']=='OWN_CONSUMER_V2_TERMINAL_FULL_SHA_RELEASE_PASS'and terminal['controller_pid']==2327023
    ready=read(root/'upstream-ready.json');assert ready and ready['role']=='upstream'and ready['source_pid']==1490465
    assert read(root/'source-context.json')['source_sampling_deadline_epoch']>time.time()+1800
    command=['systemd-run','--unit='+unit,'--property=CPUQuota=800%','--property=MemoryMax=96G',
     '--property=RuntimeMaxSec=28800','--property=KillMode=control-group','--property=TimeoutStopSec=60',
     '--setenv=Q7B_NATIVE_UNIT_INSTANCE='+UNIT,'unshare','--mount','--propagation','private','/bin/bash','-c',
     base64.b64decode(FILES['consumer_paths.sh']).decode(),'consumer-private-paths','h100-10',
     '/usr/bin/python3',str(shared_stage/'relay_consumer_bootstrap.py')]
   proposed=dict(intent_id=INTENT,unit=unit,run_id=RUN_ID,source_bundle_sha256=IDENTITY,argv=command,publication=PUBLICATION,epoch=time.time())
   with intent.open('x')as f:json.dump(proposed,f,indent=2);f.flush();os.fsync(f.fileno())
   try:
    r=subprocess.run(command,text=True,capture_output=True,timeout=45)
    result=dict(status='OWN_UNIT_STARTED_ONCE'if r.returncode==0 else'OWN_UNIT_START_FAILED_NO_REPLAY',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,epoch=time.time())
   except subprocess.TimeoutExpired as error:result=dict(status='OWN_START_OUTCOME_UNKNOWN_QUERY_SAME_UNIT',error=str(error),epoch=time.time())
   result['observed']=query(unit);put(receipt,result);value.update(status=result['status'],receipt=result)
else:
 value.update(status='READ_ONLY_CONSUMER_AND_RELAY_STATUS',observed=query(UNIT),
  upstream_intent=read(root/'upstream-submit-intent.json'),upstream_receipt=read(root/'upstream-submit-receipt.json'),
  upstream_ready=read(root/'upstream-ready.json'),intent=read(root/'native-v3-submit-intent.json'),receipt=read(root/'native-v3-submit-receipt.json'),
  attempts=[dict(directory=str(p),progress=read(p/'progress.json'),failure=read(p/'failure.json'),complete=read(p/'complete.json'))for p in sorted((base/'pool/jobs').glob('job-relay10-*/attempt-*'))],
  global_goal_complete=False)
print(json.dumps(value))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('plan','stage','start-upstream','submit','status'))
    p.add_argument('--intent-id')
    p.add_argument('--published-commit')
    p.add_argument('--published-ref')
    a = p.parse_args()
    identity, files, manifest = bundle()
    if a.action == 'plan':
        print(json.dumps(manifest, indent=2))
        return
    publication = None
    if a.action != 'status':
        assert re.fullmatch(r'[A-Za-z0-9._-]{1,120}', a.intent_id or '')
        publication = old.verify_publication(a.published_commit, a.published_ref, manifest['source_files'])
    values = dict(PHYSICAL=PHYSICAL, BASE=BASE, DIRECTORY=DIRECTORY, RUN_ID=RUN_ID, UNIT=UNIT,
        UPSTREAM_UNIT=UPSTREAM_UNIT, SOURCE_UNIT=SOURCE_UNIT, IDENTITY=identity, ACTION=a.action,
        INTENT=a.intent_id, PUBLICATION=publication, MANIFEST=manifest,
        FILES={k:base64.b64encode(v).decode()for k,v in files.items()})
    script = '\n'.join(k+'='+repr(v)for k,v in values.items())+'\n'+REMOTE
    node = 'h100-5' if a.action == 'start-upstream' else 'h100-10'
    if node == 'h100-5':
        # Enter only the existing mount view. This control process and the new
        # helper initially keep hostNET; the helper itself performs the setns.
        command=['nsenter','--target','1490214','--mount','--','/usr/bin/python3','-']
        script='import subprocess,sys\nr=subprocess.run('+repr(command)+',input='+repr(script)+',text=True,capture_output=True,timeout=85)\nprint(r.stdout,end="");print(r.stderr,end="",file=sys.stderr);raise SystemExit(r.returncode)\n'
    out=ROOT/'work'/('qwen7b-relay-consumer-control-%s-%d.json'%(a.action,time.time_ns()))
    try:
        r=old.transport(script,node)
    except subprocess.TimeoutExpired:
        out.write_text(json.dumps(dict(action=a.action,status='TRANSPORT_UNKNOWN_QUERY_SAME_INTENT_NO_REPLAY',epoch=time.time())))
        raise
    value=dict(action=a.action,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,receipt=str(out),epoch=time.time())
    out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))
    if r.returncode:
        raise SystemExit(r.returncode)


if __name__ == '__main__':
    main()
