"""Once-only independent node10 CPU relay for the explicitly retained old V6."""
import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import time
from . import shared_vram_control as old
from .legacy_relay_owner import TOPOLOGIES

ROOT, BASE, PHYSICAL, RUN_ID = old.ROOT, old.BASE, old.PHYSICAL, old.RUN_ID
SOURCES = ('legacy_relay_control.py', 'legacy_relay_owner.py', 'retrieval_relay_backpressure.py',
           'retrieval_relay.py', 'pool_worker.py', 'pool_paths.sh')
RETAINED_ROOT = BASE + '/preparation/relay-consumer10-control-v1'


def bootstrap(unit):
    return ('BASE=' + repr(BASE) + '\nUNIT=' + repr(unit) + '\n' + r'''
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
stage=Path(__file__).resolve().parent;root=stage.parent
assert os.geteuid()==0 and socket.gethostname().split('.')[0]=='h100-10'
assert os.stat('/proc/self/ns/mnt').st_ino!=os.stat('/proc/1/ns/mnt').st_ino
assert os.stat('/proc/self/ns/net').st_ino==os.stat('/proc/1/ns/net').st_ino
manifest=json.loads((stage/'manifest.json').read_text())
for name,row in manifest['files'].items():
 raw=(stage/name).read_bytes();assert len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['sha256'],name
context=json.loads((root/'submit-context.json').read_text())
detail=subprocess.check_output(['systemctl','show',UNIT,'-p','Id','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','MemoryMax','-p','ActiveEnterTimestampMonotonic'],text=True)
props=dict(line.split('=',1)for line in detail.splitlines()if'='in line)
assert props['Id']==UNIT and props['MainPID']==str(os.getpid())and props['ActiveState']=='active'and int(props['MemoryMax'])==1073741824
assert any(line.split(':',2)[-1]==props['ControlGroup']for line in Path('/proc/self/cgroup').read_text().splitlines())
remaining=14400-(time.monotonic()-int(props['ActiveEnterTimestampMonotonic'])/1000000);assert 0<remaining<=14400
launch=Path(BASE)/'legacy-cpu-relay/launch'/('native-'+str(os.getpid()));launch.mkdir(parents=True)
log=(launch/'bootstrap.log').open('ab',buffering=0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
evidence=launch/'allocation-evidence.json'
evidence.write_text(json.dumps(dict(mode='NATIVE_INDEPENDENT_CPU_RELAY',node='h100-10',unit=UNIT,main_pid=os.getpid(),
 cgroup=props['ControlGroup'],unit_show=detail,captured_epoch=time.time(),unit_hard_end_epoch=time.time()+remaining,
 resource_limits=dict(cpu_quota_percent=100,memory_max_bytes=1073741824,runtime_max_seconds=14400))));os.chmod(evidence,0o400)
command=['/usr/bin/python3','-u',str(stage/'legacy_relay_owner.py'),'--base',BASE,
 '--source-context',context['source_context'],'--source-context-sha256',context['source_context_sha256'],
 '--upstream-ready',context['upstream_ready'],'--upstream-ready-sha256',context['upstream_ready_sha256'],
 '--token-file',context['token_file'],'--allocation-evidence',str(evidence),
 '--predecessor-proof',context['predecessor_proof'],'--predecessor-sha256',context['predecessor_sha256'],
 '--slot-manifest',str(stage/'slots.json'),'--ready-file',str(root/'owner-ready.json'),
 '--pool-worker',str(stage/'pool_worker.py'),'--relay-script',str(stage/'retrieval_relay_backpressure.py'),'--owner-seconds','14400']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
for key in('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):env.pop(key,None)
print(json.dumps(dict(status='INDEPENDENT_OLD_V6_CPU_RELAY_BOOTSTRAP',unit=UNIT,epoch=time.time())),flush=True)
os.execvpe(command[0],command,env)
''').encode()


def bundle(topology='tp4', instance='v1'):
    assert topology in TOPOLOGIES and re.fullmatch(r'[a-z0-9-]{1,20}', instance)
    directory = 'legacy-relay10-independent-' + instance
    unit = 'sst-q7b-g5-cpu-relay10-' + instance + '-' + RUN_ID[:12] + '.service'
    files = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCES}
    rows = {name: dict(bytes=len(raw), sha256=old.digest(raw)) for name, raw in files.items()}
    identity = old.digest(old.encoded(dict(mode='current-oldV6-source', topology=topology, instance=instance, source_files=rows)))
    files.update({'legacy_cpu_bootstrap.py': bootstrap(unit), 'native_paths.sh': old.native_paths_source(files['pool_paths.sh']),
                  'slots.json': old.encoded(TOPOLOGIES[topology])})
    compile(files['legacy_cpu_bootstrap.py'], 'legacy_cpu_bootstrap.py', 'exec')
    manifest = dict(source_bundle_sha256=identity, directory=directory, unit=unit, instance=instance,
        role='current-oldV6-source', run_id=RUN_ID, topology=topology, slots=TOPOLOGIES[topology],
        retained_source_context=RETAINED_ROOT + '/source-context.json', retained_upstream_ready=RETAINED_ROOT + '/upstream-ready.json',
        retained_token_file=RETAINED_ROOT + '/relay-secret', source_files=rows,
        files={name: dict(bytes=len(raw), sha256=old.digest(raw)) for name, raw in files.items()},
        gpu_processes_owned=False, gpu_primary_leases_owned=False, source5_unchanged=True,
        downstream_workers=8, queued_requests=128, queue_wait_seconds=45, request_budget_seconds=59, backend_timeout_seconds=50,
        source_deadline_margin_seconds=900, consumer_margin_seconds=600, hard_seconds=14400)
    return identity, files, manifest


REMOTE = r'''
import base64,fcntl,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
assert os.geteuid()==0 and socket.gethostname().split('.')[0]=='h100-10'
base=Path(PHYSICAL if ACTION=='stage'else BASE);root=base/'preparation'/MANIFEST['directory'];stage=root/IDENTITY
unit=MANIFEST['unit'];intent=root/'submit-intent.json';receipt=root/'submit-receipt.json'
def read(p):return json.loads(Path(p).read_text())if Path(p).exists()else None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 with Path(p).open('x')as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def query():
 r=subprocess.run(['systemctl','show',unit,'-p','LoadState','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','Result'],text=True,capture_output=True,timeout=15)
 return dict(returncode=r.returncode,properties=dict(line.split('=',1)for line in r.stdout.splitlines()if'='in line),stderr=r.stderr)
def verify(expected=MANIFEST,where=stage):
 assert read(where/'manifest.json')==expected
 for name,row in expected['files'].items():assert sha(where/name)==row['sha256']and(where/name).stat().st_size==row['bytes'],name
 assert read(where/'stage-receipt.json')['status']=='EXACT_LEGACY_CPU_RELAY_STAGE_SHA_PASS'
value=dict(action=ACTION,unit=unit,role='current-oldV6-source',epoch=time.time())
if ACTION=='stage':
 assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 root.mkdir(parents=True,exist_ok=True);stage.mkdir(exist_ok=True)
 for name,encoded in FILES.items():
  p=stage/name;raw=base64.b64decode(encoded)
  if not p.exists():
   with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  assert sha(p)==MANIFEST['files'][name]['sha256'];os.chmod(p,0o555)
 if not(stage/'manifest.json').exists():put(stage/'manifest.json',MANIFEST)
 if not(stage/'stage-receipt.json').exists():put(stage/'stage-receipt.json',dict(status='EXACT_LEGACY_CPU_RELAY_STAGE_SHA_PASS',publication=PUBLICATION,epoch=time.time()))
 verify();value.update(status='INDEPENDENT_CPU_RELAY_CODE_STAGED_NO_SECRET_COPY',stage=str(stage))
elif ACTION=='submit':
 verify();assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 with(root/'submit.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);observed=query()
  if intent.exists():value.update(status='EXISTING_INTENT_NO_REPLAY',intent=read(intent),receipt=read(receipt),observed=observed)
  else:
   assert observed['properties'].get('LoadState')=='not-found'
   sys.path.insert(0,str(stage));import legacy_relay_owner as runtime
   context=dict(source_context=MANIFEST['retained_source_context'],upstream_ready=MANIFEST['retained_upstream_ready'],
    token_file=MANIFEST['retained_token_file'],predecessor_proof=PREDECESSOR,predecessor_sha256=PREDECESSOR_SHA)
   context.update(source_context_sha256=sha(context['source_context']),upstream_ready_sha256=sha(context['upstream_ready']))
   arguments=SimpleNamespace(**context,relay_script=str(stage/'retrieval_relay_backpressure.py'))
   pool=SimpleNamespace(read=read,sha=sha,RUN_ID=RUN_ID)
   source,upstream=runtime.source_binding(pool,arguments);runtime.verify_release(pool,arguments)
   assert source['source_sampling_deadline_epoch']>time.time()+2700
   with socket.socket()as check:check.bind(('127.0.0.1',28831))
   put(root/'submit-context.json',context)
   command=['systemd-run','--unit='+unit,'--property=CPUQuota=100%','--property=MemoryMax=1G','--property=RuntimeMaxSec=14400',
    '--property=KillMode=control-group','--property=TimeoutStopSec=60','--property=Restart=no','--setenv=Q7B_NATIVE_UNIT_INSTANCE='+unit,
    'unshare','--mount','--propagation','private','/bin/bash','-c',base64.b64decode(FILES['native_paths.sh']).decode(),
    'legacy-cpu-relay-paths','h100-10','/usr/bin/python3',str(stage/'legacy_cpu_bootstrap.py')]
   put(intent,dict(intent_id=INTENT,unit=unit,manifest=MANIFEST,context=context,argv=command,publication=PUBLICATION,epoch=time.time()))
   try:
    r=subprocess.run(command,text=True,capture_output=True,timeout=45)
    result=dict(status='INDEPENDENT_CPU_RELAY_STARTED_ONCE'if r.returncode==0 else'OWN_START_FAILED_NO_REPLAY',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
   except subprocess.TimeoutExpired as error:result=dict(status='OWN_START_OUTCOME_UNKNOWN_QUERY_SAME_UNIT',error=str(error))
   result.update(observed=query(),epoch=time.time());put(receipt,result);value.update(status=result['status'],receipt=result)
elif ACTION=='drain':
 saved=read(intent);assert saved and saved['unit']==unit
 verify(saved['manifest'],root/saved['manifest']['source_bundle_sha256'])
 observed=query();pid=int(observed['properties'].get('MainPID','0'));ready=read(root/'owner-ready.json')
 assert pid and observed['properties'].get('ActiveState')=='active'and ready['owner']['pid']==pid and ready['owner']['unit']==unit
 sys.path.insert(0,str(root/saved['manifest']['source_bundle_sha256']));import legacy_relay_owner as runtime
 assert all(runtime.process(pid)[key]==ready['owner'][key]for key in('birth','cgroup','netns_inode','mntns_inode'))
 job=Path(ready['job_ready_path']).parent;assert sha(ready['allocation_path'])==ready['allocation_sha256']
 marker=job/'drain-owner.json';request=dict(status='DRAIN_RETRIEVAL_OWNER',run_id=RUN_ID,owner_pid=pid,owner_allocation_sha256=ready['allocation_sha256'],epoch=time.time())
 if not marker.exists():put(marker,request)
 value.update(status='BOUND_DRAIN_REQUEST_RECORDED_NO_SIGNAL',marker=str(marker))
else:value.update(status='READ_ONLY_INDEPENDENT_CPU_RELAY_STATUS',observed=query(),intent=read(intent),receipt=read(receipt),ready=read(root/'owner-ready.json'))
print(json.dumps(value))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'stage', 'submit', 'status', 'drain'))
    parser.add_argument('--instance', default='v1')
    parser.add_argument('--topology', choices=tuple(TOPOLOGIES), default='tp4')
    for name in ('intent-id', 'published-commit', 'published-ref', 'predecessor-proof', 'predecessor-sha256'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    identity, files, manifest = bundle(args.topology, args.instance)
    if args.action == 'plan':print(json.dumps(manifest, indent=2));return
    publication = None
    if args.action in ('stage', 'submit'):
        assert re.fullmatch(r'[A-Za-z0-9._-]{1,120}', args.intent_id or '')
        publication = old.verify_publication(args.published_commit, args.published_ref, manifest['source_files'])
    values = dict(BASE=BASE, PHYSICAL=PHYSICAL, RUN_ID=RUN_ID, MANIFEST=manifest, IDENTITY=identity, ACTION=args.action,
        INTENT=args.intent_id, PUBLICATION=publication, PREDECESSOR=args.predecessor_proof, PREDECESSOR_SHA=args.predecessor_sha256,
        FILES={name: base64.b64encode(raw).decode() for name, raw in files.items()})
    script = '\n'.join(key + '=' + repr(value) for key, value in values.items()) + '\n' + REMOTE
    if args.action != 'stage':
        command = ['unshare', '--mount', '--propagation', 'private', '/bin/bash', '-c', files['native_paths.sh'].decode(),
                   'legacy-cpu-control-paths', 'h100-10', '/usr/bin/python3', '-']
        script = 'import os,subprocess,sys\nr=subprocess.run(' + repr(command) + ',input=' + repr(script) + \
            ',env=dict(os.environ,Q7B_NATIVE_UNIT_INSTANCE=' + repr(manifest['unit']) + '),text=True,capture_output=True,timeout=100)\nprint(r.stdout,end="");print(r.stderr,end="",file=sys.stderr);raise SystemExit(r.returncode)\n'
    output = ROOT / 'work' / ('qwen7b-legacy-cpu-relay-%s-%d.json' % (args.action, time.time_ns()))
    try:result = old.transport(script, 'h100-10')
    except subprocess.TimeoutExpired:
        output.write_text(json.dumps(dict(status='TRANSPORT_UNKNOWN_QUERY_SAME_UNIT_NO_REPLAY', unit=manifest['unit'])));raise
    value = dict(action=args.action, unit=manifest['unit'], returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, epoch=time.time())
    output.write_text(json.dumps(value, indent=2) + '\n');print(json.dumps(value, indent=2))
    if result.returncode:raise SystemExit(result.returncode)


if __name__ == '__main__':main()
