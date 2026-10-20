"""Independent TP1 replacements for only the terminal extra-model TP4 lease.

Plan is local. Stage/submit are separate explicit operations after publication
and a retained controlled-stop proof. This never stops the retired or original
source units. The original V6/retriever can return earlier than their deadline;
new models stop at batch boundaries with an absolute parent-deadline-minus-hour
cap and a four-hour maximum. Partial data always remains on the NAS.
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
RETIRED_PID, RETIRED_BIRTH = 2021698, '7430669'
RETIRED_UNIT = 'sst-q7b-g5-source5-extra-model-v1-' + RUN_ID[:12] + '.service'
TERMINAL_PROOF = BASE + '/preparation/source5-model-tp4-to-tp1-v1/complete.json'
SOURCE_CUTOFF_SECONDS = 3600
GROUPS = {'gpu3': [3], 'gpu4': [4], 'gpu6': [6], 'gpu7': [7]}
LIMITS = {name: dict(seqs=32, tokens=8192, blocks=4096, reserve=1.0) for name in GROUPS}
# GPU3's twice-observed 17.73GiB residual requires a smaller actual KV profile.
# Context length and full trajectories stay 4096; only concurrency/cache shrink.
LIMITS['gpu3'] = dict(seqs=8, tokens=4096, blocks=512, reserve=1.0)
SOURCES = ('source_tp1_replacement_control.py', 'source_sublease_worker.py', 'relay_consumer_worker.py',
           'shared_vram_worker.py', 'shared_vram_vllm_adapter.py', 'pool_worker.py', 'retrieval_relay.py')


def bootstrap(unit, group, run_seconds=14400):
    return ('BASE=' + repr(BASE) + '\nUNIT=' + repr(unit) + '\nFROZEN_REPO=' + repr(old.FROZEN_REPO)
            + '\nMAX_RUN_SECONDS=' + repr(run_seconds) + '\nMODEL=' + repr(GROUPS[group]) + '\nLIMITS=' + repr(LIMITS[group]) + '\n' + r'''
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-5'and os.geteuid()==0
stage=Path(__file__).resolve().parent;root=stage.parent
manifest=json.loads((stage/'manifest.json').read_text())
for name,row in manifest['files'].items():
 data=(stage/name).read_bytes();assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],name
context_path=root/'sublease-context.json';raw=context_path.read_bytes();context=json.loads(raw)
assert context['run_id']==manifest['run_id']and context['model_indices']==MODEL and context['schema_version']==3
assert os.stat('/proc/self/ns/mnt').st_ino==context['parent']['mntns_inode']
assert os.stat('/proc/self/ns/net').st_ino==context['parent']['netns_inode']
assert context['model_deadline_epoch']<=context['source_sampling_deadline_epoch']-3600
run_seconds=min(MAX_RUN_SECONDS,int(context['model_deadline_epoch']-time.time()))
assert run_seconds>1260
launch=Path(BASE)/'source-sublease/launch'/('native-'+str(os.getpid()));launch.mkdir(parents=True)
log=(launch/'bootstrap.log').open('ab',buffering=0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
detail=subprocess.check_output(['systemctl','show',UNIT,'-p','Id','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','MemoryMax','-p','RuntimeMaxUSec'],text=True)
props=dict(x.split('=',1)for x in detail.splitlines()if'='in x)
assert props['Id']==UNIT and props['MainPID']==str(os.getpid())and props['ActiveState']=='active'
assert props['ControlGroup']in Path('/proc/self/cgroup').read_text()
evidence=launch/'allocation-evidence.json'
evidence.write_text(json.dumps(dict(node='h100-5',unit=UNIT,main_pid=os.getpid(),unit_show=detail,
 cgroup=props['ControlGroup'],captured_epoch=time.time(),resource_limits=dict(cpu_quota_percent=800,memory_max_bytes=103079215104,runtime_max_seconds=MAX_RUN_SECONDS+120),
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


def bundle(group, terminal_path=TERMINAL_PROOF, terminal_sha256=None, run_seconds=14400):
    if group not in GROUPS or not 1260 < run_seconds <= 14400:
        raise ValueError('Unknown immutable replacement group or invalid conservative runtime')
    if terminal_sha256 is not None and not re.fullmatch(r'[0-9a-f]{64}', terminal_sha256):
        raise ValueError('Retained terminal proof needs an exact SHA256')
    physical = old.PHYSICAL
    if terminal_path.startswith(physical + '/'):
        terminal_path = BASE + terminal_path[len(physical):]
    if terminal_path != TERMINAL_PROOF:
        raise ValueError('Only the fixed controlled TP4 terminal proof is accepted')
    unit = 'sst-q7b-g5-source5-tp1-replacement-' + group + '-v1-' + RUN_ID[:12] + '.service'
    directory = 'source5-tp1-replacement-' + group + '-v1'
    files = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCES}
    rows = {name: dict(bytes=len(data), sha256=old.digest(data)) for name, data in files.items()}
    retirement = dict(path=terminal_path, sha256=terminal_sha256)
    identity = old.digest(old.encoded(dict(mode=directory, source_files=rows, retired_terminal=retirement,
                                          max_run_seconds=run_seconds)))
    files['source_sublease_bootstrap.py'] = bootstrap(unit, group, run_seconds)
    manifest = dict(source_bundle_sha256=identity, run_id=RUN_ID, seed=20261021, unit=unit,
        directory=directory, model_gpus=GROUPS[group], tensor_parallel_size=1, source_unit=SOURCE_UNIT,
        context_schema_version=3, retired_terminal=retirement, source_deadline_reserve_seconds=3600,
        maximum_run_seconds=run_seconds, model_limits=LIMITS[group], source_parent_may_return_early=True,
        source_files=rows, files={name: dict(bytes=len(data), sha256=old.digest(data)) for name, data in files.items()},
        borrowed_source_never_signalled=True, no_actor_or_probe_updates=True)
    return identity, files, manifest



REMOTE = r'''
import base64,fcntl,hashlib,json,os,re,socket,subprocess,sys,time
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-5'and os.geteuid()==0
base=Path(BASE);root=base/'preparation'/MANIFEST['directory'];stage=root/IDENTITY
UNIT=MANIFEST['unit']
def read(p):return json.loads(p.read_text())if p.exists()else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(p,raw):
 tmp=p.with_name(p.name+'.tmp-%d-%d'%(os.getpid(),time.time_ns()))
 try:
  with tmp.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  try:os.link(tmp,p)
  except FileExistsError:assert p.read_bytes()==raw
  fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 finally:tmp.unlink(missing_ok=True)
def put(p,v):once(p,(json.dumps(v,indent=2)+'\n').encode())
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
 assert sha(Path(MANIFEST['retired_terminal']['path']))==MANIFEST['retired_terminal']['sha256']
 assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 root.mkdir(parents=True,exist_ok=True);stage.mkdir(exist_ok=True)
 for name,encoded in FILES.items():
  data=base64.b64decode(encoded);p=stage/name
  if not p.exists():once(p,data)
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
   assert progress['deadline_epoch']-3600>time.time()+1260
   sys.path.insert(0,str(stage));import source_sublease_worker as worker;import pool_worker as pool
   for uuid in uuids:worker.parent_lock(base,parent,uuid,locks[uuid])
   initial={row['uuid']:row for row in pool.nvml().values()}
   assert all(initial[uuid]['free_bytes']>1024**3 for uuid in uuids)
   now=time.time();deadline=min(now+MANIFEST['maximum_run_seconds'],progress['deadline_epoch']-3600)
   context=dict(schema_version=3,base=str(base),tensor_parallel_size=1,mode='SOURCE5_SAME_RUN_GPU_SUBLEASE',run_id=RUN_ID,
    model_indices=model,model_cuda_uuids=uuids,parent=parent,retriever=retriever,parent_gpu_locks=locks,
    retired_terminal=MANIFEST['retired_terminal'],source_sampling_deadline_epoch=progress['deadline_epoch'],
    model_deadline_epoch=deadline,source_deadline_reserve_seconds=3600,model_limits=MANIFEST['model_limits'],
    source_parent_may_return_early=True,initial_gpu_memory=[initial[uuid]for uuid in uuids],epoch=now)
   retirement=worker.validate_retired_terminal(pool,context,parent,model,uuids)
   context['retired_actual_release_at_submit']=worker.retired_release(retirement)
   directory=base/'pool/gpu-subleases'/RUN_ID/('%d-%s'%(PARENT_PID,PARENT_BIRTH))
   available=[]
   try:
    for uuid in uuids:
     lease=directory/(uuid+'.lock');fd=pool.try_lock(lease);assert fd is not None,'Root-child GPU lease is not available'
     available.append(fd)
    assert birth(PARENT_PID)==PARENT_BIRTH and birth(1490465)=='6808057'
   finally:
    for fd in available:os.close(fd)
   put(root/'sublease-context.json',context)
   put(root/'retriever-source-ready.json',dict(role='upstream',source_pid=1490465,listen_url='http://127.0.0.1:28831',source_owned=False))
   command=['systemd-run','--unit='+UNIT,'--property=CPUQuota=800%','--property=MemoryMax=96G',
    '--property=RuntimeMaxSec='+str(MANIFEST['maximum_run_seconds']+120),'--property=KillMode=control-group','--property=TimeoutStopSec=60',
    'nsenter','--target',str(PARENT_PID),'--mount','--net','--','/usr/bin/python3',str(stage/'source_sublease_bootstrap.py')]
   proposed=dict(intent_id=INTENT,unit=UNIT,run_id=RUN_ID,source_bundle_sha256=IDENTITY,retired_terminal=MANIFEST['retired_terminal'],argv=command,publication=PUBLICATION,context=context,epoch=time.time());put(intent,proposed)
   try:
    r=subprocess.run(command,text=True,capture_output=True,timeout=45)
    result=dict(status='OWN_SOURCE_TP1_REPLACEMENT_UNIT_STARTED_ONCE'if r.returncode==0 else'OWN_SOURCE_SUBLEASE_START_FAILED_NO_REPLAY',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,epoch=time.time())
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'stage', 'submit', 'status'))
    parser.add_argument('--group', choices=tuple(GROUPS), default='gpu4')
    parser.add_argument('--retired-terminal-proof', default=TERMINAL_PROOF)
    parser.add_argument('--retired-terminal-sha256')
    parser.add_argument('--run-seconds', type=int, default=14400)
    parser.add_argument('--intent-id')
    parser.add_argument('--published-commit')
    parser.add_argument('--published-ref')
    args = parser.parse_args()
    if args.action in ('stage', 'submit') and not args.retired_terminal_sha256:
        parser.error('stage/submit require the actual retained controlled-terminal proof SHA256')
    identity, files, manifest = bundle(args.group, args.retired_terminal_proof,
                                       args.retired_terminal_sha256, args.run_seconds)
    if args.action == 'plan':
        print(json.dumps(manifest, indent=2))
        return
    publication = None
    if args.action != 'status':
        if not re.fullmatch(r'[A-Za-z0-9._-]{1,120}', args.intent_id or ''):
            parser.error('an explicit immutable intent-id is required')
        publication = old.verify_publication(args.published_commit, args.published_ref, manifest['source_files'])
    values = dict(BASE=BASE, RUN_ID=RUN_ID, SOURCE_UNIT=SOURCE_UNIT, PARENT_PID=PARENT_PID,
        PARENT_BIRTH=PARENT_BIRTH, IDENTITY=identity, MANIFEST=manifest, ACTION=args.action,
        FILES={name: base64.b64encode(data).decode() for name, data in files.items()},
        INTENT=args.intent_id, PUBLICATION=publication)
    script = '\n'.join(key + '=' + repr(value) for key, value in values.items()) + '\n' + REMOTE
    wrapper = ('import subprocess,sys\nr=subprocess.run(' + repr(['nsenter', '--target', str(PARENT_PID),
        '--mount', '--net', '--', '/usr/bin/python3', '-']) + ',input=' + repr(script)
        + ',text=True,capture_output=True,timeout=85)\nprint(r.stdout,end="");'
        + 'print(r.stderr,end="",file=sys.stderr);raise SystemExit(r.returncode)\n')
    output = ROOT / 'work' / ('qwen7b-source-tp1-replacement-%s-%s-%d.json' % (args.group, args.action, time.time_ns()))
    try:
        result = old.transport(wrapper, 'h100-5')
    except subprocess.TimeoutExpired:
        output.write_text(json.dumps(dict(status='TRANSPORT_UNKNOWN_QUERY_SAME_INTENT_NO_REPLAY', epoch=time.time())))
        raise
    value = dict(action=args.action, group=args.group, returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr, receipt=str(output), epoch=time.time())
    output.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

