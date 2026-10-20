"""Additional TP1 models on newly released h100-5 VRAM with explicit nested leases.

No existing unit, primary lease, retriever or namespace is changed or owned.
Only immutable small code/context is staged in the own shared NAS10 directory.
"""
import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import time
from . import shared_vram_control as old

ROOT, BASE, RUN_ID = old.ROOT, old.BASE, old.RUN_ID
SOURCE_UNIT = old.UNIT
PARENT_PID, PARENT_BIRTH = 1490214, '6807965'
PEER_PID = 2013773
PEER_UNIT = 'sst-q7b-g5-source5-extra-retrieval-v1-' + RUN_ID[:12] + '.service'
SOURCES = ('source_tp1_control.py', 'source_sublease_worker.py', 'relay_consumer_worker.py',
           'shared_vram_worker.py', 'shared_vram_vllm_adapter.py', 'pool_worker.py', 'retrieval_relay.py')
GROUPS = {'gpu2': [2], 'gpu1': [1]}
LIMITS = {'gpu2': dict(seqs=128, tokens=16384, blocks=16384, reserve=1.5),
          'gpu1': dict(seqs=64, tokens=8192, blocks=8192, reserve=1.0)}


def bootstrap(unit, group):
    return ('BASE=' + repr(BASE) + '\nUNIT=' + repr(unit) + '\nFROZEN_REPO=' + repr(old.FROZEN_REPO)
            + '\nMODEL=' + repr(GROUPS[group]) + '\nLIMITS=' + repr(LIMITS[group]) + '\n' + r'''
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-5'and os.geteuid()==0
stage=Path(__file__).resolve().parent;root=stage.parent
manifest=json.loads((stage/'manifest.json').read_text())
for name,row in manifest['files'].items():
 data=(stage/name).read_bytes();assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],name
context_path=root/'sublease-context.json';raw=context_path.read_bytes();context=json.loads(raw)
assert context['run_id']==manifest['run_id']and context['model_indices']==MODEL
assert os.stat('/proc/self/ns/mnt').st_ino==context['parent']['mntns_inode']
assert os.stat('/proc/self/ns/net').st_ino==context['parent']['netns_inode']
run_seconds=min(28500,int(context['source_sampling_deadline_epoch']-time.time()-600))
assert run_seconds>1260
launch=Path(BASE)/'source-sublease/launch'/('native-'+str(os.getpid()));launch.mkdir(parents=True)
log=(launch/'bootstrap.log').open('ab',buffering=0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
detail=subprocess.check_output(['systemctl','show',UNIT,'-p','Id','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','MemoryMax','-p','RuntimeMaxUSec'],text=True)
props=dict(x.split('=',1)for x in detail.splitlines()if'='in x)
assert props['Id']==UNIT and props['MainPID']==str(os.getpid())and props['ActiveState']=='active'
assert props['ControlGroup']in Path('/proc/self/cgroup').read_text()
evidence=launch/'allocation-evidence.json'
evidence.write_text(json.dumps(dict(node='h100-5',unit=UNIT,main_pid=os.getpid(),unit_show=detail,
 cgroup=props['ControlGroup'],captured_epoch=time.time(),resource_limits=dict(cpu_quota_percent=800,memory_max_bytes=103079215104,runtime_max_seconds=28800),
 original_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))));os.chmod(evidence,0o400)
# Retain the historical rollout utility import profile (4); canonical sampling
# config.world_size=1 and the actual tensor topology --tp=1 remain independent.
env=dict(os.environ,PYTHONPATH=FROZEN_REPO+':/dataset/shaoshitong/Search-G1-common8/retriever-gpu-lib',
 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',SEARCH_G1_B_PREFIX_TOKENS='20',
 SEARCH_G1_PHYSICAL_WORLD_SIZE='4',SEARCH_G1_BASE=BASE,HF_HOME=BASE+'/cache/huggingface',
 TORCH_EXTENSIONS_DIR=BASE+'/cache/torch_extensions',TMPDIR=str(launch/'tmp'),
 CUDA_VISIBLE_DEVICES=','.join(map(str,MODEL)),NCCL_NVLS_ENABLE='0',NCCL_BUFFSIZE='1048576',NCCL_MAX_NCHANNELS='4',
 NCCL_DEBUG='INFO',VLLM_ATTENTION_BACKEND='XFORMERS',VLLM_HOST_IP='127.0.0.1',
 Q7B_SOURCE_SUBLEASE_CONTEXT=str(context_path),
 CPATH='/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include:'
 '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/python3.10:'
 '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/x86_64-linux-gnu/python3.10')
for key in('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):env.pop(key,None)
for key in('HF_HOME','TORCH_EXTENSIONS_DIR','TMPDIR'):Path(env[key]).mkdir(parents=True,exist_ok=True)
os.chdir(FROZEN_REPO)
command=['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-u',str(stage/'source_sublease_worker.py'),
 '--base',BASE,'--allocation-evidence',str(evidence),'--pool-worker',str(stage/'pool_worker.py'),
 '--relay-script',str(stage/'retrieval_relay.py'),'--upstream-ready',str(root/'retriever-source-ready.json'),
 '--reuse-local-relay-ready',str(context_path),'--reuse-local-relay-sha256',hashlib.sha256(raw).hexdigest(),
 '--reuse-local-relay-unit',context['parent']['unit'],'--model-gpus',','.join(map(str,MODEL)),
 '--tp',str(len(MODEL)),'--model-max-seqs',str(LIMITS['seqs']),
 '--model-batch-tokens',str(LIMITS['tokens']),'--model-kv-block-cap',str(LIMITS['blocks']),
 '--model-reserve-gib',str(LIMITS['reserve']),
 '--run-seconds',str(run_seconds),'--batch-reserve-seconds','1200']
print(json.dumps(dict(status='EXPLICIT_SAME_RUN_SOURCE5_SUBLEASE_STARTING',unit=UNIT,model_gpus=MODEL,
 parent_owned=False,retriever_owned=False,epoch=time.time())),flush=True)
os.execvpe(command[0],command,env)
''').encode()


def bundle(group):
    unit = 'sst-q7b-g5-source5-extra-' + group + '-v3-' + RUN_ID[:12] + '.service'
    directory = 'source5-model-sublease-' + group + '-v3'
    files = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCES}
    rows = {name: dict(bytes=len(data), sha256=old.digest(data)) for name, data in files.items()}
    identity = old.digest(old.encoded(dict(mode=directory, source_files=rows)))
    files['source_sublease_bootstrap.py'] = bootstrap(unit, group)
    manifest = dict(source_bundle_sha256=identity, run_id=RUN_ID, seed=20261021,unit=unit,
        directory=directory, model_gpus=GROUPS[group], tensor_parallel_size=1,
        source_unit=SOURCE_UNIT, source_files=rows,
        files={name:dict(bytes=len(data),sha256=old.digest(data))for name,data in files.items()},
        borrowed_source_never_signalled=True, no_actor_or_probe_updates=True)
    return identity, files, manifest


REMOTE = r'''
import base64,fcntl,hashlib,json,os,re,socket,subprocess,time
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-5'and os.geteuid()==0
base=Path(BASE);root=base/'preparation'/MANIFEST['directory'];stage=root/IDENTITY
UNIT=MANIFEST['unit']
def read(p):return json.loads(p.read_text())if p.exists()else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def birth(pid):return Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
def query(unit):
 r=subprocess.run(['systemctl','show',unit,'-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','LoadState','-p','ExecMainStatus','-p','Result'],text=True,capture_output=True,timeout=15)
 return dict(returncode=r.returncode,properties=dict(x.split('=',1)for x in r.stdout.splitlines()if'='in x),stderr=r.stderr)
def verify():
 assert read(stage/'manifest.json')==MANIFEST
 for name,row in MANIFEST['files'].items():
  data=(stage/name).read_bytes();assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],name
 assert(read(stage/'stage-receipt.json')or{})['status']=='SOURCE_SUBLEASE_CODE_ALL_FILES_FULL_SHA_PASS'
value=dict(action=ACTION,unit=UNIT,bundle_sha256=IDENTITY,epoch=time.time())
intent=root/'submit-intent.json';receipt=root/'submit-receipt.json'
if ACTION=='stage':
 assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 root.mkdir(parents=True,exist_ok=True);stage.mkdir(exist_ok=True)
 for name,encoded in FILES.items():
  data=base64.b64decode(encoded);p=stage/name
  if not p.exists():
   with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  assert sha(p)==MANIFEST['files'][name]['sha256'];os.chmod(p,0o555)
 if not(stage/'manifest.json').exists():put(stage/'manifest.json',MANIFEST)
 proof=dict(status='SOURCE_SUBLEASE_CODE_ALL_FILES_FULL_SHA_PASS',files=MANIFEST['files'],publication=PUBLICATION,epoch=time.time())
 if not(stage/'stage-receipt.json').exists():put(stage/'stage-receipt.json',proof)
 verify();value.update(status=proof['status'],stage=str(stage),files=MANIFEST['files'])
elif ACTION=='submit':
 verify();assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 with(root/'submit.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);observed=query(UNIT);value['observed']=observed
  if intent.exists():value.update(status='EXISTING_INTENT_NO_START_REPLAY',intent=read(intent),receipt=read(receipt))
  else:
   assert observed['properties'].get('LoadState')=='not-found'
   parent_query=query(SOURCE_UNIT);assert parent_query['properties'].get('MainPID')==str(PARENT_PID)and parent_query['properties'].get('ActiveState')=='active'
   assert birth(PARENT_PID)==PARENT_BIRTH
   attempts=list((base/'pool/jobs'/('job-shared5-'+str(PARENT_PID))).glob('attempt-*'));assert len(attempts)==1
   attempt=attempts[0];allocation=read(attempt/'allocation.json');progress=read(attempt/'progress.json')
   assert allocation['controller_pid']==PARENT_PID and allocation['native_unit']==SOURCE_UNIT
   assert read(base/'config.json')['run_id']==RUN_ID
   parent_group=Path('/proc/%d/cgroup'%PARENT_PID).read_text();assert SOURCE_UNIT in parent_group
   parent=dict(pid=PARENT_PID,birth=PARENT_BIRTH,unit=SOURCE_UNIT,cgroup=parent_group,
    netns_inode=os.stat('/proc/%d/ns/net'%PARENT_PID).st_ino,mntns_inode=os.stat('/proc/%d/ns/mnt'%PARENT_PID).st_ino,
    allocation_path=str(attempt/'allocation.json'),allocation_sha256=sha(attempt/'allocation.json'))
   retriever=dict(pid=1490465,birth=birth(1490465),cgroup=Path('/proc/1490465/cgroup').read_text(),netns_inode=os.stat('/proc/1490465/ns/net').st_ino)
   assert retriever['birth']=='6808057'and retriever['cgroup']==parent_group and retriever['netns_inode']==parent['netns_inode']
   inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=20)
   gpu={int(row[0]):dict(index=int(row[0]),uuid=row[1].strip(),free_mib=int(row[2]),total_mib=int(row[3]))for row in [x.split(',')for x in inventory.splitlines()]}
   model=MANIFEST['model_gpus'];uuids=[gpu[i]['uuid']for i in model]
   assert set(uuids)<=set(allocation['actual_cuda_runtime_uuids'])
   locks={}
   for fd in Path('/proc/%d/fd'%PARENT_PID).iterdir():
    try:target=os.readlink(fd)
    except FileNotFoundError:continue
    for uuid in uuids:
     primary=base/'pool/gpu-locks'/(uuid+'.lock')
     if target!=str(primary):continue
     actual=fd.stat();expected=primary.stat();assert(actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino)
     info=Path('/proc/%d/fdinfo/%s'%(PARENT_PID,fd.name)).read_text()
     assert re.search(r'POSIX\s+ADVISORY\s+WRITE\s+%d\s+'%PARENT_PID,info)
     locks[uuid]=dict(fd=int(fd.name),device=actual.st_dev,inode=actual.st_ino)
   assert set(locks)==set(uuids)
   assert progress['deadline_epoch']>time.time()+1860
   peer_query=query(PEER_UNIT);assert peer_query['properties'].get('MainPID')==str(PEER_PID)and peer_query['properties'].get('ActiveState')=='active'
   peer_attempts=list((base/'pool/jobs'/('job-source5-model-'+str(PEER_PID))).glob('attempt-*'));assert len(peer_attempts)==1
   peer_allocation_path=peer_attempts[0]/'allocation.json';peer_allocation=read(peer_allocation_path)
   assert peer_allocation['controller_pid']==PEER_PID and peer_allocation['native_unit']==PEER_UNIT
   assert peer_allocation['source_parent']==parent
   peer_context_path=Path(peer_allocation['source_context']);assert sha(peer_context_path)==peer_allocation['source_context_sha256']
   peer_context=read(peer_context_path);assert peer_context['run_id']==RUN_ID and peer_context['parent']==parent
   peer_group=Path('/proc/%d/cgroup'%PEER_PID).read_text();assert PEER_UNIT in peer_group
   peer=dict(pid=PEER_PID,birth=birth(PEER_PID),unit=PEER_UNIT,cgroup=peer_group,
    netns_inode=os.stat('/proc/%d/ns/net'%PEER_PID).st_ino,mntns_inode=os.stat('/proc/%d/ns/mnt'%PEER_PID).st_ino,
    allocation_path=str(peer_allocation_path),allocation_sha256=sha(peer_allocation_path),gpu_locks={})
   assert peer['netns_inode']==parent['netns_inode']and peer['mntns_inode']==parent['mntns_inode']
   for uuid in uuids:
    peer_lease=base/'pool/gpu-subleases'/RUN_ID/('%d-%s'%(PARENT_PID,PARENT_BIRTH))/(uuid+'.lock')
    assert peer_allocation['gpu_sublease_paths'][uuid]==str(peer_lease)
    for fd in Path('/proc/%d/fd'%PEER_PID).iterdir():
     try:target=os.readlink(fd)
     except FileNotFoundError:continue
     if target!=str(peer_lease):continue
     actual=fd.stat();expected=peer_lease.stat();assert(actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino)
     info=Path('/proc/%d/fdinfo/%s'%(PEER_PID,fd.name)).read_text()
     assert re.search(r'POSIX\s+ADVISORY\s+WRITE\s+%d\s+'%PEER_PID,info)
     peer['gpu_locks'][uuid]=dict(fd=int(fd.name),device=actual.st_dev,inode=actual.st_ino)
   assert set(peer['gpu_locks'])==set(uuids)
   context=dict(schema_version=2,tensor_parallel_size=1,peer_sublease=peer,mode='SOURCE5_SAME_RUN_GPU_SUBLEASE',run_id=RUN_ID,
    model_indices=model,model_cuda_uuids=uuids,parent=parent,retriever=retriever,parent_gpu_locks=locks,
    source_sampling_deadline_epoch=progress['deadline_epoch'],initial_gpu_memory=[gpu[i]for i in model],epoch=time.time())
   put(root/'sublease-context.json',context)
   put(root/'retriever-source-ready.json',dict(role='upstream',source_pid=1490465,listen_url='http://127.0.0.1:28831',source_owned=False))
   command=['systemd-run','--unit='+UNIT,'--property=CPUQuota=800%','--property=MemoryMax=96G',
    '--property=RuntimeMaxSec=28800','--property=KillMode=control-group','--property=TimeoutStopSec=60',
    'nsenter','--target',str(PARENT_PID),'--mount','--net','--','/usr/bin/python3',str(stage/'source_sublease_bootstrap.py')]
   proposed=dict(intent_id=INTENT,unit=UNIT,run_id=RUN_ID,source_bundle_sha256=IDENTITY,argv=command,publication=PUBLICATION,context=context,epoch=time.time());put(intent,proposed)
   try:
    r=subprocess.run(command,text=True,capture_output=True,timeout=45)
    result=dict(status='OWN_SOURCE_SUBLEASE_UNIT_STARTED_ONCE'if r.returncode==0 else'OWN_SOURCE_SUBLEASE_START_FAILED_NO_REPLAY',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,epoch=time.time())
   except subprocess.TimeoutExpired as error:result=dict(status='OWN_START_OUTCOME_UNKNOWN_QUERY_SAME_UNIT',error=str(error),epoch=time.time())
   result['observed']=query(UNIT);put(receipt,result);value.update(status=result['status'],receipt=result,context=context)
else:
 observed=query(UNIT);saved=read(receipt)or{}
 pid=observed['properties'].get('MainPID','0')
 if pid=='0':pid=(saved.get('observed')or{}).get('properties',{}).get('MainPID','0')
 value.update(status='READ_ONLY_SOURCE_SUBLEASE_STATUS',observed=observed,parent=query(SOURCE_UNIT),intent=read(intent),receipt=saved,context=read(root/'sublease-context.json'))
 attempts=list((base/'pool/jobs'/('job-source5-model-'+pid)).glob('attempt-*'))
 assert len(attempts)<=1
 if attempts:
  job=attempts[0];allocation=read(job/'allocation.json')or{};children=read(job/'children.json')or{}
  value['sampler']=dict(directory=str(job),progress=read(job/'progress.json'),failure=read(job/'failure.json'),complete=read(job/'complete.json'),
   model_initialized=[read(p)for p in(job/'sampler-launches').glob('launch-*/model-initialized.json')],active_phases=[])
  group=allocation.get('native_cgroup')
  for child in children.get('own_process_groups',[]):
   try:
    if not group or group not in Path('/proc/%d/cgroup'%child).read_text():continue
    for fd in Path('/proc/%d/fd'%child).iterdir():
     try:target=os.readlink(fd)
     except FileNotFoundError:continue
     if '/pool/batch-locks/'not in target:continue
     prefix,relative=target.split('/pool/batch-locks/',1)
     if not prefix.endswith('/shaoshitong/'+base.name):continue
     tail=Path(relative)
     if len(tail.parts)!=2 or tail.suffix!='.lock':continue
     for phase in('prefix','full'):
      q=base/'sampling'/tail.parent/'batches'/tail.stem/'outers/outer-000001'/phase/'rank-0-progress.json'
      detail=read(q)
      if detail:value['sampler']['active_phases'].append(dict(batch=tail.stem,phase=phase,metadata=detail))
   except (FileNotFoundError,ProcessLookupError):continue
  if value['sampler']['failure']:
   q=job/'sampler.log'
   if q.exists():
    with q.open('rb')as f:f.seek(max(0,q.stat().st_size-5000));value['sampler']['sampler_log_tail']=f.read().decode(errors='replace')
print(json.dumps(value))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('plan','stage','submit','status'))
    p.add_argument('--group',choices=tuple(GROUPS),default='gpu2')
    p.add_argument('--intent-id');p.add_argument('--published-commit');p.add_argument('--published-ref')
    a=p.parse_args();identity,files,manifest=bundle(a.group)
    if a.action=='plan':print(json.dumps(manifest,indent=2));return
    publication=None
    if a.action!='status':
        assert re.fullmatch(r'[A-Za-z0-9._-]{1,120}',a.intent_id or '')
        publication=old.verify_publication(a.published_commit,a.published_ref,manifest['source_files'])
    values=dict(BASE=BASE,RUN_ID=RUN_ID,SOURCE_UNIT=SOURCE_UNIT,PARENT_PID=PARENT_PID,PARENT_BIRTH=PARENT_BIRTH,
        PEER_PID=PEER_PID,PEER_UNIT=PEER_UNIT,IDENTITY=identity,FILES={n:base64.b64encode(v).decode()for n,v in files.items()},MANIFEST=manifest,
        ACTION=a.action,INTENT=a.intent_id,PUBLICATION=publication)
    script='\n'.join(k+'='+repr(v)for k,v in values.items())+'\n'+REMOTE
    wrapper='import subprocess,sys\nr=subprocess.run('+repr(['nsenter','--target',str(PARENT_PID),'--mount','--net','--','/usr/bin/python3','-'])+',input='+repr(script)+',text=True,capture_output=True,timeout=85)\nprint(r.stdout,end="");print(r.stderr,end="",file=sys.stderr);raise SystemExit(r.returncode)\n'
    stamp=time.time_ns();out=ROOT/'work'/('qwen7b-source-sublease-%s-%s-%d.json'%(a.group,a.action,stamp))
    try:r=old.transport(wrapper,'h100-5')
    except subprocess.TimeoutExpired:
        out.write_text(json.dumps(dict(status='TRANSPORT_UNKNOWN_QUERY_SAME_INTENT_NO_REPLAY',epoch=time.time())));raise
    value=dict(action=a.action,group=a.group,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,receipt=str(out),epoch=time.time())
    out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
    if r.returncode:raise SystemExit(r.returncode)


if __name__=='__main__':main()
