"""Once-only second h100-10 TP4 consumer borrowing the active V5 local relay.

This has its own immutable code stage, intent and bounded unit. The original
consumer, its local relay and the h100-5 retriever are external dependencies.
"""
import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import time

from . import relay_consumer_control as original

old = original.old
ROOT, BASE, PHYSICAL, RUN_ID = original.ROOT, original.BASE, original.PHYSICAL, original.RUN_ID
DIRECTORY = 'relay-consumer10-parallel-v1'
UNIT = 'sst-q7b-g5-consumer10-parallel-v1-' + RUN_ID[:12] + '.service'
OWNER_UNIT, OWNER_PID, RELAY_PID = original.UNIT, 2371322, 2371352
RELAY_DIRECTORY = original.DIRECTORY
SOURCES = ('relay_parallel_control.py', 'test_relay_parallel_control.py', *original.SOURCES)


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Original consumer bootstrap changed: expected one ' + repr(before))
    return source.replace(before, after, 1)


def bootstrap():
    source = original.bootstrap().decode()
    source = replace_once(source, 'BASE=' + repr(original.BASE), 'BASE=' + repr(BASE))
    source = replace_once(source, 'UNIT=' + repr(original.UNIT), 'UNIT=' + repr(UNIT))
    source = replace_once(source, 'stage=Path(__file__).resolve().parent;root=stage.parent',
        "stage=Path(__file__).resolve().parent;control_root=stage.parent\n"
        "root=Path(BASE)/'preparation'/" + repr(RELAY_DIRECTORY))
    source = replace_once(source,
        "context=json.loads((root/'source-context.json').read_text())\n"
        "run_seconds=min(28500,int(context['source_sampling_deadline_epoch']-time.time()-600))",
        "context_bytes=(root/'source-context.json').read_bytes()\n"
        "context=json.loads(context_bytes)\n"
        "dependency=json.loads((control_root/'submit-context.json').read_text())\n"
        "assert dependency['unit']==UNIT and dependency['owner_unit']==" + repr(OWNER_UNIT) + "\n"
        "assert dependency['owner_main_pid']==2371322 and dependency['relay_pid']==2371352\n"
        "assert hashlib.sha256(context_bytes).hexdigest()==dependency['source_context_sha256']\n"
        "assert context['source_sampling_deadline_epoch']==dependency['source_sampling_deadline_epoch']\n"
        "dependency_deadline=min(context['source_sampling_deadline_epoch']-600,dependency['owner_deadline_epoch']-300)\n"
        "run_seconds=min(28500,int(dependency_deadline-time.time()))")
    source = replace_once(source, "Path(BASE)/'relay-consumer10/launch'",
                          "Path(BASE)/'relay-consumer10-parallel/launch'")
    source = replace_once(source, "CUDA_VISIBLE_DEVICES='0,1,2,3'", "CUDA_VISIBLE_DEVICES='4,5,6,7'")
    source = replace_once(source,
        "'--upstream-ready',str(root/'upstream-ready.json'),'--model-gpus','0,1,2,3',",
        "'--upstream-ready',str(root/'upstream-ready.json'),'--model-gpus','4,5,6,7',\n"
        " '--reuse-local-relay-ready',dependency['reuse_local_relay_ready'],\n"
        " '--reuse-local-relay-sha256',dependency['reuse_local_relay_sha256'],\n"
        " '--reuse-local-relay-unit',dependency['reuse_local_relay_unit'],")
    source = replace_once(source, "status='MODEL_ONLY_CONSUMER_BOOTSTRAP'",
                          "status='PARALLEL_MODEL_ONLY_CONSUMER_BOOTSTRAP'")
    source = replace_once(source, 'model_gpus=[0,1,2,3]',
                          'model_gpus=[4,5,6,7],local_relay_owned=False,dependency=dependency')
    compile(source, 'relay_parallel_bootstrap.py', 'exec')
    return source.encode()


def bundle():
    files = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCES}
    rows = {name: dict(bytes=len(data), sha256=old.digest(data)) for name, data in files.items()}
    identity = old.digest(old.encoded(dict(mode=DIRECTORY, source_files=rows)))
    files.update({'relay_parallel_bootstrap.py': bootstrap(),
                  'consumer_paths.sh': old.native_paths_source(files['pool_paths.sh'])})
    manifest = dict(source_bundle_sha256=identity, run_id=RUN_ID, seed=20261021,
        mode='REUSE_EXISTING_V5_LOCAL_RELAY', unit=UNIT, owner_unit=OWNER_UNIT,
        owner_main_pid=OWNER_PID, local_relay_pid=RELAY_PID, original_control_directory=RELAY_DIRECTORY,
        source_files=rows, files={k: dict(bytes=len(v), sha256=old.digest(v)) for k, v in files.items()},
        same_cluster_retrieval_only=True, original_index_reused_without_reencoding=True,
        model_gpus=[4, 5, 6, 7], tensor_parallel_size=4, no_actor_or_probe_updates=True,
        local_relay_owned=False, upstream_retriever_owned=False,
        communication_environment=dict(NCCL_NVLS_ENABLE='0', NCCL_BUFFSIZE='1048576',
                                       NCCL_MAX_NCHANNELS='4', NCCL_DEBUG='INFO'))
    return identity, files, manifest


REMOTE = r'''
import base64,fcntl,hashlib,json,os,socket,stat,subprocess,time
from pathlib import Path
assert os.geteuid()==0 and socket.gethostname().split('.')[0]=='h100-10'
base=Path(PHYSICAL);root=base/'preparation'/DIRECTORY;stage=root/IDENTITY
relay_root=base/'preparation'/RELAY_DIRECTORY
def read(p):return json.loads(p.read_text())if p.exists()else None
def put(p,v):
 t=p.with_name(p.name+'.tmp-'+str(time.time_ns()))
 with t.open('x')as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def immutable(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def birth(pid):return Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
def canonical(p):return str(Path(BASE)/p.relative_to(base))
def query(unit):
 r=subprocess.run(['systemctl','show',unit,'-p','LoadState','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','Result','-p','ExecMainStatus'],capture_output=True,text=True,timeout=20)
 return dict(unit=unit,returncode=r.returncode,properties=dict(x.split('=',1)for x in r.stdout.splitlines()if'='in x),stderr=r.stderr)
def verify():
 assert read(stage/'manifest.json')==MANIFEST
 for name,row in MANIFEST['files'].items():
  p=stage/name;data=p.read_bytes();assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],name
 assert(read(stage/'stage-receipt.json')or{}).get('status')=='EXACT_SMALL_PARALLEL_CONSUMER_CODE_FULL_SHA_PASS'
value=dict(action=ACTION,unit=UNIT,owner_unit=OWNER_UNIT,bundle_sha256=IDENTITY,epoch=time.time())
intent=root/'native-submit-intent.json';receipt=root/'native-submit-receipt.json'
if ACTION=='stage':
 assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 root.mkdir(parents=True,exist_ok=True);stage.mkdir(exist_ok=True)
 for name,encoded in FILES.items():
  p=stage/name;data=base64.b64decode(encoded)
  if not p.exists():
   with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  assert sha(p)==MANIFEST['files'][name]['sha256'];os.chmod(p,0o555)
 if not(stage/'manifest.json').exists():immutable(stage/'manifest.json',MANIFEST)
 put(stage/'stage-receipt.json',dict(status='EXACT_SMALL_PARALLEL_CONSUMER_CODE_FULL_SHA_PASS',files=MANIFEST['files'],publication=PUBLICATION,epoch=time.time()))
 verify();value.update(status='PARALLEL_CODE_STAGED_EXISTING_RELAY_UNCHANGED',stage=str(stage),files=MANIFEST['files'])
elif ACTION=='submit':
 verify();assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 with(root/'native-submit.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);old_intent=read(intent);observed=query(UNIT);value['observed']=observed
  if old_intent:
   value.update(status='EXISTING_INTENT_NO_START_REPLAY',intent=old_intent,receipt=read(receipt))
  else:
   assert observed['properties'].get('LoadState')=='not-found','Unique own unit is not confirmed absent'
   owner=query(OWNER_UNIT);props=owner['properties']
   assert owner['returncode']==0 and props.get('MainPID')==str(OWNER_PID)and props.get('ActiveState')=='active'
   owner_birth=birth(OWNER_PID);owner_group=props['ControlGroup'];assert owner_group and owner_group.endswith('/'+OWNER_UNIT)
   assert owner_group in Path('/proc/%d/cgroup'%OWNER_PID).read_text()
   attempts=list((base/'pool/jobs'/('job-relay10-'+str(OWNER_PID))).glob('attempt-*'));assert len(attempts)==1
   attempt=attempts[0];progress=read(attempt/'progress.json');allocation=read(attempt/'allocation.json')
   assert allocation['controller_pid']==OWNER_PID and allocation['native_unit']==OWNER_UNIT
   ready_path=attempt/'relay-ready.json';raw_ready=ready_path.read_bytes();ready=json.loads(raw_ready)
   assert ready['status']=='RELAY_LISTENING'and ready['role']=='downstream'
   assert ready['listen']==dict(host='127.0.0.1',port=28831)and ready['listen_url']=='http://127.0.0.1:28831'
   process=ready['process'];assert process['pid']==RELAY_PID and ready['source_pid']==1490465
   relay_birth=birth(RELAY_PID);relay_group=Path('/proc/%d/cgroup'%RELAY_PID).read_text()
   relay_netns=os.stat('/proc/%d/ns/net'%RELAY_PID).st_ino
   assert relay_birth==str(process['birth'])and relay_group==process['cgroup']and owner_group in relay_group
   assert relay_netns==process['netns_inode']==ready['listener_netns_inode']==os.stat('/proc/1/ns/net').st_ino
   assert ready['code_sha256']==sha(stage/'retrieval_relay.py')
   sockets={os.readlink(p)for p in Path('/proc/%d/fd'%RELAY_PID).iterdir()if p.is_symlink()}
   listeners=[x.split()for x in Path('/proc/net/tcp').read_text().splitlines()[1:]]
   assert any(x[1]=='0100007F:709F'and x[3]=='0A'and'socket:['+x[9]+']'in sockets for x in listeners),'Borrowed relay does not own the expected listener'
   source_bytes=(relay_root/'source-context.json').read_bytes();source_context=json.loads(source_bytes)
   upstream=read(relay_root/'upstream-ready.json')
   assert upstream['role']=='upstream'and upstream['source_pid']==ready['source_pid']
   assert ready['upstream_proof']['sha256']==sha(relay_root/'upstream-ready.json')
   assert ready['source_birth']==upstream['source_birth']and ready['source_netns_inode']==upstream['source_netns_inode']
   secret=relay_root/'relay-secret';info=secret.lstat()
   assert stat.S_ISREG(info.st_mode)and stat.S_IMODE(info.st_mode)==0o600 and info.st_uid==os.geteuid()
   dependency_deadline=min(source_context['source_sampling_deadline_epoch']-600,progress['deadline_epoch']-300)
   assert dependency_deadline>time.time()+900,'Too little borrowed relay owner runtime remains'
   assert birth(OWNER_PID)==owner_birth and birth(RELAY_PID)==relay_birth
   context=dict(unit=UNIT,owner_unit=OWNER_UNIT,owner_main_pid=OWNER_PID,owner_birth=owner_birth,
    relay_pid=RELAY_PID,relay_birth=relay_birth,relay_cgroup=relay_group,relay_netns_inode=relay_netns,
    reuse_local_relay_ready=canonical(ready_path),reuse_local_relay_sha256=hashlib.sha256(raw_ready).hexdigest(),
    reuse_local_relay_unit=OWNER_UNIT,owner_deadline_epoch=progress['deadline_epoch'],
    owner_progress=canonical(attempt/'progress.json'),owner_progress_sha256=sha(attempt/'progress.json'),
    source_sampling_deadline_epoch=source_context['source_sampling_deadline_epoch'],
    source_context_sha256=hashlib.sha256(source_bytes).hexdigest(),dependency_deadline_epoch=dependency_deadline,
    local_relay_owned=False,source_retriever_owned=False,epoch=time.time())
   immutable(root/'submit-context.json',context)
   shared_stage=Path(BASE)/'preparation'/DIRECTORY/IDENTITY
   command=['systemd-run','--unit='+UNIT,'--property=CPUQuota=800%','--property=MemoryMax=96G',
    '--property=RuntimeMaxSec=28800','--property=KillMode=control-group','--property=TimeoutStopSec=60',
    '--setenv=Q7B_NATIVE_UNIT_INSTANCE='+UNIT,'unshare','--mount','--propagation','private','/bin/bash','-c',
    base64.b64decode(FILES['consumer_paths.sh']).decode(),'consumer-private-paths','h100-10',
    '/usr/bin/python3',str(shared_stage/'relay_parallel_bootstrap.py')]
   proposed=dict(intent_id=INTENT,unit=UNIT,run_id=RUN_ID,source_bundle_sha256=IDENTITY,argv=command,
    dependency=context,publication=PUBLICATION,epoch=time.time());immutable(intent,proposed)
   try:
    r=subprocess.run(command,text=True,capture_output=True,timeout=45)
    result=dict(status='OWN_UNIT_STARTED_ONCE'if r.returncode==0 else'OWN_UNIT_START_FAILED_NO_REPLAY',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,epoch=time.time())
   except subprocess.TimeoutExpired as error:result=dict(status='OWN_START_OUTCOME_UNKNOWN_QUERY_SAME_UNIT',error=str(error),epoch=time.time())
   result['observed']=query(UNIT);put(receipt,result);value.update(status=result['status'],receipt=result,dependency=context)
else:
 value.update(status='READ_ONLY_PARALLEL_CONSUMER_STATUS',observed=query(UNIT),owner=query(OWNER_UNIT),
  intent=read(intent),receipt=read(receipt),dependency=read(root/'submit-context.json'),
  attempts=[dict(directory=str(p),progress=read(p/'progress.json'),failure=read(p/'failure.json'),complete=read(p/'complete.json'))for p in sorted((base/'pool/jobs').glob('job-relay10-*/attempt-*'))],global_goal_complete=False)
print(json.dumps(value))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'stage', 'submit', 'status'))
    parser.add_argument('--intent-id')
    parser.add_argument('--published-commit')
    parser.add_argument('--published-ref')
    args = parser.parse_args()
    identity, files, manifest = bundle()
    if args.action == 'plan':
        print(json.dumps(manifest, indent=2))
        return
    publication = None
    if args.action != 'status':
        assert re.fullmatch(r'[A-Za-z0-9._-]{1,120}', args.intent_id or '')
        publication = old.verify_publication(args.published_commit, args.published_ref, manifest['source_files'])
    values = dict(PHYSICAL=PHYSICAL, BASE=BASE, DIRECTORY=DIRECTORY, RELAY_DIRECTORY=RELAY_DIRECTORY,
        RUN_ID=RUN_ID, UNIT=UNIT, OWNER_UNIT=OWNER_UNIT, OWNER_PID=OWNER_PID, RELAY_PID=RELAY_PID,
        IDENTITY=identity, ACTION=args.action, INTENT=args.intent_id, PUBLICATION=publication, MANIFEST=manifest,
        FILES={k: base64.b64encode(v).decode() for k, v in files.items()})
    script = '\n'.join(k + '=' + repr(v) for k, v in values.items()) + '\n' + REMOTE
    out = ROOT / 'work' / ('qwen7b-relay-parallel-control-%s-%d.json' % (args.action, time.time_ns()))
    try:
        result = old.transport(script, 'h100-10')
    except subprocess.TimeoutExpired:
        out.write_text(json.dumps(dict(action=args.action,
            status='TRANSPORT_UNKNOWN_QUERY_SAME_INTENT_NO_REPLAY', epoch=time.time())))
        raise
    value = dict(action=args.action, returncode=result.returncode, stdout=result.stdout,
                 stderr=result.stderr, receipt=str(out), epoch=time.time())
    out.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
