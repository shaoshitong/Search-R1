"""Prepare/start once or read status; fixed assets never pass through the Mac."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

from reproduction.h1008_qprobe.management_transport import remote
from .fixed_assets_backup import PARENT, fixed_namespace
from .checkpoint_backup_native import run_identity
from .checkpoint_backup_native_control import status_script

REPO = Path(__file__).resolve().parents[2]


def published_bundle():
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files = {}
    for name in ('fixed_assets_backup.py','checkpoint_backup_native.py'):
        path = 'reproduction/prefix_selection200/'+name
        code = subprocess.check_output(['git','show',commit+':'+path],cwd=REPO)
        if code != (REPO/path).read_bytes(): raise ValueError('Publish backup code before use')
        files[name] = dict(code=base64.b64encode(code).decode(),sha256=hashlib.sha256(code).hexdigest())
    return commit, files


def prepare_script(base, commit, files):
    return r'''
import base64,hashlib,importlib.util,json,sys
from pathlib import Path
base=Path(BASE);files=FILES
assert base.parent==Path('/mnt/data/shaoshitong') and base.name.startswith('search-g1-prefix-selection200-actorhidden20-')
control=base/'preparation/fixed-assets-control'/files['fixed_assets_backup.py']['sha256'];control.mkdir(parents=True,exist_ok=True)
for name,item in files.items():
 p=control/name;code=base64.b64decode(item['code']);assert hashlib.sha256(code).hexdigest()==item['sha256']
 if p.exists():assert p.read_bytes()==code
 else:
  with p.open('xb')as f:f.write(code)
sys.path.insert(0,str(control))
spec=importlib.util.spec_from_file_location('fixed_assets_backup',control/'fixed_assets_backup.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
path=base/'preparation/fixed-recovery-assets-inventory-once-v1.json'
if path.exists():
 plan=module.read(path)
 assert plan['run_id']==module.run_identity(module.read(base/'preparation/launch-config.json')['run_id']) and plan['source_base']==str(base) and plan['inventory_worker_sha256']==files['fixed_assets_backup.py']['sha256']
 assert plan['source_config_sha256']==module.sha(base/'preparation/launch-config.json')
 replay=False
else:
 plan=module.inventory(base);plan.update(inventory_worker_sha256=files['fixed_assets_backup.py']['sha256'],inventory_common_sha256=files['checkpoint_backup_native.py']['sha256'],inventory_tool_commit=COMMIT)
 module.record(path.parent,path.name,plan);replay=False
print(json.dumps(dict(status=plan['status'],inventory_path=str(path),inventory_sha256=module.sha(path),files=len(plan['files']),bytes=plan['bytes'],existing_inventory_rehashed_or_replayed=replay,independent_backup_performed=False,training_modified=False)))
'''.replace('BASE',repr(base)).replace('FILES',repr(files)).replace('COMMIT',repr(commit))


def start_script(inventory_path, inventory_sha, commit, files, run_id):
    run_id = run_identity(run_id)
    return r'''
import base64,fcntl,hashlib,json,os,socket,subprocess,time
from pathlib import Path
parent=Path(PARENT_PATH);root=parent/ROOT_NAME;files=FILES
assert socket.gethostname()=='h100-10' and parent.resolve()==parent
with(parent/'fixed-assets-start.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if root.exists():
  print(json.dumps(dict(status='EXISTING_NAMESPACE_RECONCILE_NO_REPLAY',namespace=str(root),files=[p.name for p in root.iterdir()])));raise SystemExit
 source=Path('/mnt/shared/workspace/training_data_node_8')/Path(INVENTORY_PATH).relative_to('/mnt/data')
 assert hashlib.sha256(source.read_bytes()).hexdigest()==INVENTORY_SHA
 plan=json.loads(source.read_text());assert plan['run_id']==RUN_ID and plan['status']=='FIXED_RECOVERY_ASSETS_INVENTORY_ONLY'
 assert plan['inventory_worker_sha256']==files['fixed_assets_backup.py']['sha256'] and plan['inventory_common_sha256']==files['checkpoint_backup_native.py']['sha256']
 fs=os.statvfs(parent);assert fs.f_bavail*fs.f_frsize>=plan['bytes']+128*1024**3
 assert subprocess.check_output(['findmnt','-n','-T',str(parent),'-o','FSTYPE'],text=True).strip()=='ext4'
 dependency=parent/(RUN_ID[:12]+'-outer-0050-once-v1');assert(dependency/'RECEIPT.json').exists()
 root.mkdir(mode=0o700)
 def once(path,value):
  with path.open('x')as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 for name,item in files.items():
  code=base64.b64decode(item['code']);assert hashlib.sha256(code).hexdigest()==item['sha256']
  with(root/name).open('xb')as f:f.write(code);f.flush();os.fsync(f.fileno())
 once(root/'INTENT.json',dict(run_id=RUN_ID,inventory_path=INVENTORY_PATH,inventory_sha256=INVENTORY_SHA,worker_sha256=files['fixed_assets_backup.py']['sha256'],common_sha256=files['checkpoint_backup_native.py']['sha256'],code_commit=COMMIT,created_unix=time.time(),expected_files=len(plan['files']),expected_bytes=plan['bytes'],wait_for_existing_cp50=True))
 with(root/'backup.log').open('xb')as log:
  child=subprocess.Popen(['/usr/bin/python3','-E','-B','-S',str(root/'fixed_assets_backup.py'),'--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C'})
 state=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
 receipt=dict(status='FIXED_ASSET_COPY_STARTED_OR_WAITING_NOT_COMPLETE',run_id=RUN_ID,namespace=str(root),pid=child.pid,start_ticks=int(state[19]),sid=os.getsid(child.pid),expected_files=len(plan['files']),expected_bytes=plan['bytes'],ordinary_free_bytes=fs.f_bavail*fs.f_frsize,bytes_per_second_limit=20*1024**2,wait_for_existing_cp50=True,semantic_reload_performed=False,training_modified=False,model_bytes_transit_Mac=False)
 once(root/'RECEIPT.json',receipt);print(json.dumps(receipt))
'''.replace('PARENT_PATH',repr(str(PARENT))).replace('ROOT_NAME',repr(fixed_namespace(run_id).name)).replace('FILES',repr(files)).replace('INVENTORY_PATH',repr(inventory_path)).replace('INVENTORY_SHA',repr(inventory_sha)).replace('RUN_ID',repr(run_id)).replace('COMMIT',repr(commit))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','start','status'])
    p.add_argument('--run-id');p.add_argument('--base');p.add_argument('--inventory-path');p.add_argument('--inventory-sha');p.add_argument('--receipts',required=True);a=p.parse_args()
    target='h100-10'
    if a.phase == 'status':
        if not a.run_id:p.error('Status needs the complete --run-id')
        script=status_script(str(fixed_namespace(a.run_id)),a.run_id).replace("print(json.dumps(out))", "error=root/'ERROR.json';out['error']=json.loads(error.read_text()) if error.exists() else None;print(json.dumps(out))")
    else:
        commit,files=published_bundle()
        if a.phase=='prepare':
            if not a.base:p.error('Prepare needs current base')
            script=prepare_script(a.base,commit,files);target='h100-8'
        else:
            if not(a.inventory_path and a.inventory_sha):p.error('Start needs the actual prepared inventory path/SHA')
            if not a.run_id:p.error('Start needs the complete --run-id matching its inventory')
            script=start_script(a.inventory_path,a.inventory_sha,commit,files,a.run_id)
    path=Path(a.receipts)/('actorhidden-fixed-assets-'+a.phase+'-'+str(time.time_ns())+'-transport.json')
    result=json.loads(remote(script,path,control_path=None,target=target));actual=path.with_name(path.name.replace('-transport','-actual'))
    actual.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(path=str(actual),result=result)))


if __name__=='__main__':main()
