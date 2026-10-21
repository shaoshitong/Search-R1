"""Isolated future live-plan capture and terminal-only collector; no stop API."""
import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import time
from . import shared_vram_control as old
from .rollover_terminal import PRIMARY_INSTANCES
from . import rollover_paths

ROOT, BASE, PHYSICAL, RUN_ID = old.ROOT, old.BASE, old.PHYSICAL, old.RUN_ID
SOURCES = ('rollover_terminal_control.py', 'rollover_terminal.py', 'rollover_paths.py', 'predecessor_generation_terminal.py', 'pool_paths.sh')


def bundle(node='h100-10', generation='rollover-v1', collector_instance=None):
    assert node in ('h100-5', 'h100-10') and re.fullmatch(r'[a-z0-9-]{1,25}', generation)
    assert collector_instance is None or (node == 'h100-10' and re.fullmatch(r'[a-z][a-z0-9-]{0,23}', collector_instance) and collector_instance != 'original')
    instance = collector_instance or 'original'
    suffix = '-' + collector_instance if collector_instance else ''
    stem = 'collector' + suffix
    files = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCES}
    rows = {name: dict(bytes=len(raw), sha256=old.digest(raw)) for name, raw in files.items()}
    identity = old.digest(old.encoded(dict(node=node, generation=generation, collector_instance=instance, source_files=rows)))
    files['native_paths.sh'] = old.native_paths_source(files['pool_paths.sh'])
    manifest = dict(node=node, generation=generation, source_bundle_sha256=identity,
        directory='cohort-' + generation + '-' + node, unit='sst-q7b-terminal-' + generation + '-' + node + suffix + '-' + RUN_ID[:12] + '.service',
        collector_instance=instance, collector_intent=stem + '-intent.json', collector_receipt=stem + '-receipt.json', collector_log=stem + '.log',
        run_id=RUN_ID, source_files=rows, files={name: dict(bytes=len(raw), sha256=old.digest(raw)) for name, raw in files.items()},
        no_stop_or_signal_action=True, no_slurm_changes=True)
    return identity, files, manifest


REMOTE = r'''
import base64,fcntl,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
assert os.geteuid()==0 and socket.gethostname().split('.')[0]==('h100-10'if ACTION in('stage','status')else MANIFEST['node'])
base=Path(PHYSICAL if ACTION in('stage','status')else BASE);root=base/'preparation'/MANIFEST['directory'];stage=root/IDENTITY
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def immutable(p,v):
 with Path(p).open('x')as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def verify():
 assert read(stage/'manifest.json')==MANIFEST
 for name,row in MANIFEST['files'].items():assert sha(stage/name)==row['sha256']and(stage/name).stat().st_size==row['bytes']
 assert read(stage/'stage-receipt.json')['status']=='EXACT_TERMINAL_COLLECTOR_STAGE_SHA_PASS'
value=dict(action=ACTION,node=MANIFEST['node'],generation=MANIFEST['generation'],collector_instance=MANIFEST['collector_instance'],unit=MANIFEST['unit'],epoch=time.time())
if ACTION=='stage':
 assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 root.mkdir(parents=True,exist_ok=True);stage.mkdir(exist_ok=True)
 for name,encoded in FILES.items():
  p=stage/name;raw=base64.b64decode(encoded)
  if not p.exists():
   with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  assert sha(p)==MANIFEST['files'][name]['sha256'];os.chmod(p,0o555)
 if not(stage/'manifest.json').exists():immutable(stage/'manifest.json',MANIFEST)
 if not(stage/'stage-receipt.json').exists():immutable(stage/'stage-receipt.json',dict(status='EXACT_TERMINAL_COLLECTOR_STAGE_SHA_PASS',publication=PUBLICATION,epoch=time.time()))
 verify();value.update(status='TERMINAL_COLLECTOR_CODE_STAGED_NO_SIGNALS',stage=str(stage))
elif ACTION=='capture':
 verify();assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 sys.path.insert(0,str(stage));import rollover_terminal as terminal
 planpath=root/'plan.json'
 if planpath.exists():
  saved=terminal.validate_plan(read(planpath));value.update(status='EXISTING_IMMUTABLE_PLAN_NO_RECAPTURE',plan=str(planpath),sha256=sha(planpath))
 else:
  node=MANIFEST['node'];rows=[]
  def allocation(pattern,pid):
   found=list((base/'pool/jobs'/(pattern+str(pid))).glob('attempt-*/allocation.json'));assert len(found)==1
   return found[0]
  def add(name,pattern,prep,borrows=(),path=None,kind='model-or-cpu-owner'):
   props=terminal.core.query(name);pid=int(props['MainPID']);p=Path(path)if path else allocation(pattern,pid)
   evidence=read(p).get('allocation_evidence')if kind!='retained-upstream-ready'else None
   roots=[p.parent,base/'preparation'/prep]
   if evidence:roots.append(Path(evidence).parent)
   rows.append(terminal.captured_row(name,node,p,roots,borrows,kind))
  common=dict(schema_version=2,run_id=RUN_ID,node=node,original_partial_outputs_retained=True,collector_signals_no_processes=True)
  if node=='h100-10':
   assert sha(CPU_READY)==CPU_SHA;ready=read(CPU_READY);cpu=ready['owner']
   assert ready['status']=='LEGACY_SOURCE_RELAY_OWNER_READY'and ready['role']=='current-oldV6-source'and ready['run_id']==RUN_ID
   instances=terminal.primary_instances(PRIMARY_INSTANCES,ready['slots'])
   for slot,instance in instances.items():
    add(terminal.unit('primary10-'+slot+'-'+instance),'job-relay10-primary-','relay-primary10-'+slot+'-'+instance,(cpu['unit'],))
   add(cpu['unit'],'job-legacy-relay-owner-','legacy-relay10-independent-v1',path=ready['allocation_path'])
   assert rows[-1]['pid']==cpu['pid']and rows[-1]['birth']==cpu['birth']
   common.update(cohort_kind='legacy-cpu-and-primary-mixed',primary_instances=instances,cpu_owner_ready=ready,cpu_owner_ready_path=CPU_READY,cpu_owner_ready_sha256=CPU_SHA,
    prior_failed_proof_path=FAILED_PROOF,prior_failed_proof_sha256=FAILED_SHA,prior_failed_proof_status=terminal.FAILED)
   terminal.verify_prior_failed(common)
  else:
   parent=terminal.core.SOURCE['unit']
   pairs=[('source5-extra-gpu1-v3','source5-model-sublease-gpu1-v3'),('source5-extra-gpu2-v3','source5-model-sublease-gpu2-v3')]
   pairs += [('source5-tp1-replacement-gpu%d-v1'%i,'source5-tp1-replacement-gpu%d-v1'%i)for i in(3,4,6,7)]
   pairs += [('source5-extra-retrieval-v1','source5-model-sublease-retrieval-v1')]
   for name,prep in pairs:add(terminal.unit(name),'job-source5-model-',prep,(parent,))
   add('sst-q7b-retrieval-relay5-v1-'+RUN_ID[:12]+'.service','unused','relay-consumer10-control-v1',(parent,),
    path=base/'preparation/relay-consumer10-control-v1/upstream-ready.json',kind='retained-upstream-ready')
   add(parent,'job-shared5-','shared-vram-native-control-v6')
   assert rows[-1]['pid']==1490214 and rows[-1]['birth']=='6807965'
   assert sha(REMOTE_PLAN)==REMOTE_PLAN_SHA
   remote=terminal.validate_plan(read(REMOTE_PLAN));assert remote['node']=='h100-10'
   common.update(cohort_kind='source5-v6-eight-models',source_mount_ns_inode=os.stat('/proc/1490214/ns/mnt').st_ino,
    source_net_ns_inode=os.stat('/proc/1490214/ns/net').st_ino,remote_node10_plan_path=REMOTE_PLAN,
    remote_node10_plan_sha256=REMOTE_PLAN_SHA,remote_node10_proof_path=str(Path(REMOTE_PLAN).parent/'complete.json'))
  common.update(source={key:rows[-1][key]for key in('unit','pid','birth')},ordered_units=rows,
   captured_epoch=time.time(),source_code_sha256=sha(stage/'rollover_terminal.py'))
  terminal.validate_plan(common);immutable(planpath,common)
  value.update(status='ACTUAL_LIVE_COHORT_CAPTURED_NO_SIGNALS',plan=str(planpath),sha256=sha(planpath),units=len(rows))
elif ACTION=='collect':
 verify();assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
 sys.path.insert(0,str(stage));import rollover_terminal as terminal
 planpath=root/'plan.json';plan=terminal.validate_plan(read(planpath));assert sha(planpath)==PLAN_SHA
 intent=root/MANIFEST['collector_intent'];receipt=root/MANIFEST['collector_receipt']
 with(root/'collector-submit.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if intent.exists():value.update(status='EXISTING_COLLECTOR_INTENT_NO_REPLAY',intent=read(intent),receipt=read(receipt)if receipt.exists()else None)
  else:
   prior=[]
   if MANIFEST['collector_instance']!='original':
    assert(root/'collector-intent.json').exists(),'Repair requires the consumed original collector intent'
    seen=set()
    for previous in sorted(root.glob('collector*-intent.json')):
     saved=read(previous);unit=saved['unit'];previous_receipt=previous.with_name(previous.name.replace('-intent.json','-receipt.json'))
     assert unit.startswith('sst-q7b-terminal-'+MANIFEST['generation']+'-'+MANIFEST['node']+'-')and unit.endswith(RUN_ID[:12]+'.service')
     assert unit not in seen and unit!=MANIFEST['unit']and saved['plan_sha256']==PLAN_SHA,'Prior collector unit/plan binding differs'
     seen.add(unit)
     assert previous_receipt.exists(),'Prior collector start is unknown: receipt missing'
     outcome=read(previous_receipt)
     assert outcome.get('status')in('TERMINAL_ONLY_COLLECTOR_STARTED_ONCE','COLLECTOR_START_FAILED_NO_REPLAY'),'Prior collector start is unknown'
     detail=subprocess.check_output(['systemctl','show',unit,*[item for prop in('Id','LoadState','MainPID','ActiveState','SubState','Result','ExecMainStatus','ControlGroup')for item in('-p',prop)]],text=True,timeout=15)
     observed=dict(line.split('=',1)for line in detail.splitlines()if'='in line)
     assert observed.get('Id')==unit and observed.get('LoadState')=='loaded'and observed.get('MainPID')=='0'and observed.get('ActiveState')in('inactive','failed'),'Prior collector is active or its terminal/start state is unknown'
     prior.append(dict(unit=unit,intent=str(previous),intent_sha256=sha(previous),receipt=str(previous_receipt),receipt_sha256=sha(previous_receipt),observed=observed,unit_show=detail,epoch=time.time()))
   query=subprocess.check_output(['systemctl','show',MANIFEST['unit'],'-p','LoadState'],text=True,timeout=15);assert'LoadState=not-found'in query
   command=['/usr/bin/python3',str(stage/'rollover_terminal.py'),'--plan',str(planpath),'--plan-sha256',PLAN_SHA,'--collect-terminal','--watch-seconds',str(WATCH),'--log-file',str(root/MANIFEST['collector_log'])]
   if MANIFEST['node']=='h100-5':command=['nsenter','--target','1490214','--mount','--net','--',*command]
   else:command=['unshare','--mount','--propagation','private','/bin/bash',terminal.core.physical(stage/'native_paths.sh'),'h100-10',*command]
   command=['systemd-run','--unit='+MANIFEST['unit'],'--property=CPUQuota=100%','--property=MemoryMax=1G',
    '--property=RuntimeMaxSec='+str(WATCH+300),'--property=KillMode=control-group','--property=Restart=no',
    *(['--setenv=Q7B_NATIVE_UNIT_INSTANCE='+MANIFEST['unit']]if MANIFEST['node']=='h100-10'else[]),*command]
   assert not any('\n'in item or'\r'in item for item in command),'Collector systemd argv must contain no multiline arguments'
   immutable(intent,dict(unit=MANIFEST['unit'],collector_instance=MANIFEST['collector_instance'],argv=command,plan_sha256=PLAN_SHA,publication=PUBLICATION,prior_collectors_terminal=prior,signals_predecessors=False,epoch=time.time()))
   try:
    r=subprocess.run(command,text=True,capture_output=True,timeout=45)
    result=dict(status='TERMINAL_ONLY_COLLECTOR_STARTED_ONCE'if r.returncode==0 else'COLLECTOR_START_FAILED_NO_REPLAY',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
   except subprocess.TimeoutExpired as error:result=dict(status='COLLECTOR_START_UNKNOWN_QUERY_SAME_UNIT',error=str(error))
   immutable(receipt,result);value.update(result)
else:
 value.update(status='READ_ONLY_TERMINAL_COLLECTOR_STATUS',plan=read(root/'plan.json')if(root/'plan.json').exists()else None,
  proof=read(root/'complete.json')if(root/'complete.json').exists()else None,
  proof_sha256=sha(root/'complete.json')if(root/'complete.json').exists()else None)
print(json.dumps(value))
'''
REMOTE = rollover_paths.remote_reads(REMOTE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'stage', 'capture', 'collect', 'status'))
    parser.add_argument('--node', choices=('h100-5', 'h100-10'), required=True)
    parser.add_argument('--generation', default='rollover-v1')
    parser.add_argument('--collector-instance', help='Distinct node10 repair instance, e.g. repair1; original default unit/intent names stay unchanged')
    parser.add_argument('--watch-seconds', type=int, default=7200)
    parser.add_argument('--primary-instances', default=','.join(name + '=' + instance for name, instance in PRIMARY_INSTANCES.items()),
        help='Exact live slot=instance mapping; capture binds each actual MainPID/birth/allocation')
    for name in ('published-commit', 'published-ref', 'cpu-ready', 'cpu-ready-sha256', 'failed-proof', 'failed-proof-sha256',
                 'remote-node10-plan', 'remote-node10-plan-sha256', 'plan-sha256'):parser.add_argument('--' + name)
    args = parser.parse_args();assert 0 <= args.watch_seconds <= 7200
    pairs = [item.split('=') for item in args.primary_instances.split(',')]
    assert all(len(pair) == 2 for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs)
    from .rollover_terminal import primary_instances
    instances = primary_instances(dict(pairs), PRIMARY_INSTANCES)
    identity, files, manifest = bundle(args.node, args.generation, args.collector_instance)
    if args.action == 'plan':print(json.dumps(manifest, indent=2));return
    publication = None
    if args.action != 'status':publication = old.verify_publication(args.published_commit, args.published_ref, manifest['source_files'])
    values = dict(BASE=BASE, PHYSICAL=PHYSICAL, RUN_ID=RUN_ID, MANIFEST=manifest, IDENTITY=identity, ACTION=args.action,
        PUBLICATION=publication, CPU_READY=args.cpu_ready, CPU_SHA=args.cpu_ready_sha256,
        PRIMARY_INSTANCES=instances,
        FAILED_PROOF=args.failed_proof, FAILED_SHA=args.failed_proof_sha256,
        REMOTE_PLAN=args.remote_node10_plan, REMOTE_PLAN_SHA=args.remote_node10_plan_sha256,
        PLAN_SHA=args.plan_sha256, WATCH=args.watch_seconds, FILES={name: base64.b64encode(raw).decode() for name, raw in files.items()})
    script = '\n'.join(key + '=' + repr(value) for key, value in values.items()) + '\n' + REMOTE
    node = 'h100-10' if args.action in ('stage', 'status') else args.node
    if args.action not in ('stage', 'status'):
        target = ['nsenter', '--target', '1490214', '--mount', '--net', '--', '/usr/bin/python3', '-'] if node == 'h100-5' else [
            'unshare', '--mount', '--propagation', 'private', '/bin/bash', '-c', files['native_paths.sh'].decode(),
            'terminal-control-paths', node, '/usr/bin/python3', '-']
        script = 'import os,subprocess,sys\nr=subprocess.run(' + repr(target) + ',input=' + repr(script) + ',text=True,capture_output=True,timeout=100,env=dict(os.environ,Q7B_NATIVE_UNIT_INSTANCE=' + repr(manifest['unit']) + '))\nprint(r.stdout,end="");print(r.stderr,end="",file=sys.stderr);raise SystemExit(r.returncode)\n'
    output = ROOT / 'work' / ('qwen7b-rollover-terminal-%s-%d.json' % (args.action, time.time_ns()))
    try:result = old.transport(script, node)
    except subprocess.TimeoutExpired:
        output.write_text(json.dumps(dict(status='TRANSPORT_UNKNOWN_OBSERVE_SAME_COLLECTOR_NO_REPLAY', unit=manifest['unit'])));raise
    value = dict(action=args.action, node=args.node, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, epoch=time.time())
    output.write_text(json.dumps(value, indent=2) + '\n');print(json.dumps(value, indent=2))
    if result.returncode:raise SystemExit(result.returncode)


if __name__ == '__main__':main()
