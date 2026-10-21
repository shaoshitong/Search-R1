"""Local finite-model observation control; no stop/drain/submit actions.

Plan is local. Stage writes two small published sources on NAS10; capture,
status and collect run only their CPU observer on the selected model's node.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time
from . import finite_instance_terminal as terminal
from . import shared_vram_control as transport

ROOT = transport.ROOT
SOURCES = ('finite_instance_terminal.py', 'finite_instance_terminal_control.py')
REGISTRY = ROOT / 'work/qwen7b-owner-v2-eleven-model-current-cohort-registry.json'
RENEWAL = ROOT / 'work/qwen7b-owner-v2-eleven-model-r2-renewal-plan.json'


def selection(key, registry_path=REGISTRY, renewal_path=RENEWAL):
    raw_registry, raw_plan = Path(registry_path).read_bytes(), Path(renewal_path).read_bytes()
    terminal.require(terminal.digest(raw_registry) == terminal.REGISTRY_SHA
        and terminal.digest(raw_plan) == terminal.RENEWAL_SHA, 'Exact reviewed registry/renewal plan SHA differs')
    registry, plan = json.loads(raw_registry), json.loads(raw_plan)
    models = [row for row in registry['registry'] if row['key'] == key and row['role'] == 'model']
    planned = [row for row in plan['rows'] if row['key'] == key]
    terminal.require(len(models) == len(planned) == 1, 'Select one exact declared model key')
    row, renew = models[0], planned[0];original = renew['current_r1']
    terminal.require(all(row[name] == original[name] for name in ('unit', 'directory'))
        and row['source_bundle_sha256'] == original['frozen_source_bundle_sha256'], 'Registry and original r1 identity disagree')
    owners = [r for r in registry['registry'] if r['key'] == renew['owner_registry_key']]
    terminal.require(len(owners) == 1 and owners[0]['role'] in ('owner', 'downstream')
        and owners[0]['node'] == row['node'], 'Wrong resource-owner registry row')
    spec = renew['model_specification']
    value = dict(schema_version=1, run_id=terminal.RUN_ID, key=key, node=row['node'], unit=row['unit'],
        instance='r1', slot=row['slot'], directory=row['directory'], source_bundle_sha256=row['source_bundle_sha256'],
        submit_intent=original['submit_intent'], owner_unit=owners[0]['unit'],
        registry_sha256=terminal.REGISTRY_SHA, renewal_plan_sha256=terminal.RENEWAL_SHA,
        model_indices=spec['indices'], tensor_parallel_size=spec['tp'])
    return terminal.validate_selection(value)


def bundle():
    files = {name:Path(__file__).with_name(name).read_bytes() for name in SOURCES}
    rows = {name:dict(bytes=len(raw), sha256=terminal.digest(raw)) for name,raw in files.items()}
    identity = terminal.digest(terminal.encoded(dict(contract=terminal.STATUS, source_files=rows)))
    return identity, files, dict(source_bundle_sha256=identity, contract=terminal.STATUS,
        source_files=rows, files=rows, directory=terminal.DIRECTORY, model_runtime_modified=False)


REMOTE = r'''
import base64,hashlib,importlib.util,json,os,socket,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
assert os.geteuid()==0 and socket.gethostname().split('.')[0]==('h100-10'if ACTION=='stage'else SELECTION['node'])
base=Path(BASE);physical=Path(PHYSICAL)
if ACTION=='stage':view=physical
else:
 unit=SELECTION['owner_unit']
 q=subprocess.run(['systemctl','show',unit,'-p','Id','-p','LoadState','-p','MainPID','-p','ActiveState','-p','ControlGroup'],text=True,capture_output=True,timeout=10)
 p=dict(line.split('=',1)for line in q.stdout.splitlines()if'='in line)
 assert q.returncode==0 and p.get('Id')==unit and p.get('LoadState')=='loaded'and p.get('ActiveState')=='active','Borrowed owner is unavailable; no model action taken'
 pid=int(p['MainPID']);assert pid>1 and p.get('ControlGroup')=='/system.slice/'+unit
 assert any(line.split(':',2)[-1]==p['ControlGroup']for line in Path('/proc/%d/cgroup'%pid).read_text().splitlines())
 candidates=[Path('/proc/%d/root'%pid)/base.relative_to('/'),base,physical]
 view=next((path for path in candidates if(path/'config.json').is_file()),None)
 assert view is not None,'Same-run NAS10 view is unavailable'
assert json.loads((view/'config.json').read_text())['run_id']==RUN_ID
root=view/'preparation'/MANIFEST['directory'];stage=root/'code'/IDENTITY
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 raw=(json.dumps(v,sort_keys=True,separators=(',',':'))+'\n').encode();temp=p.with_name(p.name+'.tmp-%d-%d'%(os.getpid(),time.time_ns()))
 try:
  with temp.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  os.link(temp,p)
 finally:temp.unlink(missing_ok=True)
if ACTION=='stage':
 assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'and PUBLICATION['source_files']==MANIFEST['source_files']
 stage.mkdir(parents=True,exist_ok=True)
 for name,encoded in FILES.items():
  target=stage/name;raw=base64.b64decode(encoded)
  if not target.exists():
   temporary=target.with_name(name+'.tmp-%d-%d'%(os.getpid(),time.time_ns()))
   try:
    with temporary.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
    try:os.link(temporary,target)
    except FileExistsError:pass
   finally:temporary.unlink(missing_ok=True)
  assert sha(target)==MANIFEST['files'][name]['sha256'];os.chmod(target,0o555)
 if not(stage/'manifest.json').exists():put(stage/'manifest.json',MANIFEST)
 if not(stage/'stage-receipt.json').exists():put(stage/'stage-receipt.json',dict(status='FINITE_INSTANCE_OBSERVER_CODE_STAGE_SHA_PASS',publication=PUBLICATION,epoch=time.time()))
assert json.loads((stage/'manifest.json').read_text())==MANIFEST
for name,row in MANIFEST['files'].items():
 raw=(stage/name).read_bytes();assert len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['sha256'],name
receipt=json.loads((stage/'stage-receipt.json').read_text())
assert receipt['status']=='FINITE_INSTANCE_OBSERVER_CODE_STAGE_SHA_PASS'and receipt['publication']['source_files']==MANIFEST['source_files']
if ACTION=='stage':value=dict(status='FINITE_INSTANCE_OBSERVER_CODE_STAGED',identity=IDENTITY,physical_stage=str(physical/'preparation'/MANIFEST['directory']/'code'/IDENTITY))
else:
 spec=importlib.util.spec_from_file_location('q7b_finite_instance_observer',stage/'finite_instance_terminal.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 value=module.operate(ACTION,SELECTION,module.View(view),IDENTITY)
value.update(action=ACTION,key=SELECTION['key'],node=SELECTION['node'],unit=SELECTION['unit'],code_identity=IDENTITY)
print(json.dumps(value,sort_keys=True))
raise SystemExit(2 if value['status'].startswith('ROOT_ATTENTION')else 3 if value['status'].startswith('OBSERVATION_UNKNOWN')else 0)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'stage', 'capture', 'status', 'collect'))
    parser.add_argument('--key', required=True)
    parser.add_argument('--registry', default=str(REGISTRY))
    parser.add_argument('--renewal-plan', default=str(RENEWAL))
    parser.add_argument('--published-commit')
    parser.add_argument('--published-ref')
    args = parser.parse_args();chosen = selection(args.key, args.registry, args.renewal_plan)
    identity, files, manifest = bundle()
    if args.action == 'plan':
        print(json.dumps(dict(status='LOCAL_FINITE_INSTANCE_OBSERVER_PLAN',selection=chosen,manifest=manifest,
            capture=str(terminal.root_for(chosen)/'capture.json'),proof=str(terminal.root_for(chosen)/'complete.json')),indent=2));return
    publication = transport.verify_publication(args.published_commit,args.published_ref,manifest['source_files']) if args.action=='stage' else None
    values = dict(ACTION=args.action,SELECTION=chosen,IDENTITY=identity,MANIFEST=manifest,PUBLICATION=publication,
        BASE=str(terminal.BASE),PHYSICAL=str(terminal.PHYSICAL),RUN_ID=terminal.RUN_ID,
        FILES={name:base64.b64encode(raw).decode() for name,raw in files.items()} if args.action=='stage' else {})
    script='\n'.join(key+'='+repr(value) for key,value in values.items())+'\n'+REMOTE
    record=ROOT/'work'/('qwen7b-finite-instance-%s-%d.json'%(args.action,time.time_ns()))
    try:
        result=transport.transport(script,'h100-10' if args.action=='stage' else chosen['node'])
        value=dict(action=args.action,key=args.key,unit=chosen['unit'],code_identity=identity,
            returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,epoch=time.time())
    except (OSError,subprocess.TimeoutExpired) as error:
        value=dict(status='TRANSPORT_UNKNOWN_NO_NATIVE_EXIT_INFERRED',action=args.action,key=args.key,error=str(error),epoch=time.time())
        with record.open('x') as stream:json.dump(value,stream,indent=2)
        print(json.dumps(dict(record=str(record),**value),indent=2));raise
    with record.open('x') as stream:json.dump(value,stream,indent=2)
    print(json.dumps(dict(record=str(record),**value),indent=2))
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
