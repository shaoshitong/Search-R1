"""One-shot launch/status for the independent CPU/NFS checkpoint copier."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .checkpoint_backup_native import PARENT, RUN
from reproduction.h1008_qprobe.management_transport import remote

REPO = Path(__file__).resolve().parents[2]


def start_script(spec, code, commit):
    payload = dict(spec, code=base64.b64encode(code).decode(), code_sha256=hashlib.sha256(code).hexdigest(), code_commit=commit)
    return r'''
import pathlib,json,base64,hashlib,os,socket,subprocess,time,fcntl
p=PAYLOAD;root=pathlib.Path(p['namespace']);parent=root.parent
assert socket.gethostname()=='h100-10' and str(parent)==EXPECTED_PARENT and root.name.startswith(RUN_PREFIX)
parent.mkdir(mode=0o700,exist_ok=True);assert parent.resolve()==parent
with(parent/'start-control.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if root.exists():
  print(json.dumps(dict(status='EXISTING_NAMESPACE_RECONCILE_NO_REPLAY',namespace=str(root),files=[x.name for x in root.iterdir()])));raise SystemExit
 code=base64.b64decode(p.pop('code'));assert hashlib.sha256(code).hexdigest()==p['code_sha256']
 scope={'__name__':'checkpoint_backup_admission'};exec(compile(code,'checkpoint_backup.py','exec'),scope)
 assets,proof=scope['inventory'](p)
 fs=os.statvfs(parent);need=sum(row['bytes']for row in assets)
 assert fs.f_bavail*fs.f_frsize>=need+scope['RESERVE'],'Insufficient ordinary backup destination free space'
 assert subprocess.check_output(['findmnt','-n','-T',str(parent),'-o','FSTYPE'],text=True).strip()=='ext4'
 root.mkdir(mode=0o700)
 def once(path,value):
  with path.open('x')as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 once(root/'INTENT.json',dict(p,operation='INDEPENDENT_JOINT_CHECKPOINT_COPY_ONLY',created_unix=time.time(),expected_bytes=need,expected_files=len(assets)))
 script=root/'checkpoint_backup.py'
 with script.open('xb')as f:f.write(code);f.flush();os.fsync(f.fileno())
 assert hashlib.sha256(script.read_bytes()).hexdigest()==p['code_sha256']
 with(root/'backup.log').open('xb')as output:
  child=subprocess.Popen(['/usr/bin/python3','-E','-B','-S',str(script),'--namespace',str(root)],stdin=subprocess.DEVNULL,stdout=output,stderr=output,start_new_session=True,env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C'})
 tail=(pathlib.Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
 receipt=dict(status='INDEPENDENT_COPY_STARTED_NOT_COMPLETE',namespace=str(root),pid=child.pid,start_ticks=int(tail[19]),sid=os.getsid(child.pid),uid=os.getuid(),run_id=p['run_id'],outer=p['outer'],code_commit=p['code_commit'],code_sha256=p['code_sha256'],expected_files=len(assets),expected_bytes=need,ordinary_free_bytes=fs.f_bavail*fs.f_frsize,bytes_per_second_limit=scope['RATE'],semantic_reload_performed=False,training_modified=False,model_bytes_transit_Mac=False)
 once(root/'RECEIPT.json',receipt);print(json.dumps(receipt))
'''.replace('PAYLOAD', repr(payload)).replace('EXPECTED_PARENT', repr(str(PARENT))).replace('RUN_PREFIX',repr(RUN[:12]))


def status_script(namespace):
    return r'''
import pathlib,json,os,hashlib,socket
root=pathlib.Path(NAMESPACE);out=dict(hostname=socket.gethostname(),namespace=str(root),exists=root.exists(),read_only=True)
if root.exists():
 for name in('INTENT.json','RECEIPT.json','PROGRESS.json','RESULT.json'):
  path=root/name
  if path.exists():out[name]=json.loads(path.read_text())
 receipt=out.get('RECEIPT.json')
 if receipt:
  path=pathlib.Path('/proc')/str(receipt['pid'])/'stat'
  if path.exists():
   tail=path.read_text().rsplit(')',1)[1].split();out['process']=dict(pid=receipt['pid'],start_ticks=int(tail[19]),same_birth=int(tail[19])==receipt['start_ticks'],state=tail[0],sid=os.getsid(receipt['pid']))
  else:out['process']=dict(pid=receipt['pid'],gone=True)
 out['final_directory_exists']=(root/'checkpoint').is_dir()
 out['staging_directory_exists']=(root/'checkpoint.partial').is_dir()
 if(root/'backup.log').exists():
  with(root/'backup.log').open('rb')as f:f.seek(max(0,os.fstat(f.fileno()).st_size-1500));out['log_tail']=f.read().decode(errors='replace')
print(json.dumps(out))
'''.replace('NAMESPACE',repr(namespace))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('start','status'))
    parser.add_argument('--outer',type=int,required=True);parser.add_argument('--source-base');parser.add_argument('--cpu-proof')
    parser.add_argument('--cpu-proof-sha256');parser.add_argument('--receipts',required=True)
    args=parser.parse_args();namespace=str(PARENT/(RUN[:12]+'-outer-%04d-once-v1'%args.outer))
    if args.phase=='start':
        if not all((args.source_base,args.cpu_proof,args.cpu_proof_sha256)):parser.error('Start needs actual source/proof identity')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        code=subprocess.check_output(['git','show',commit+':reproduction/prefix_selection200/checkpoint_backup_native.py'],cwd=REPO)
        if code!=(REPO/'reproduction/prefix_selection200/checkpoint_backup_native.py').read_bytes():raise ValueError('Publish current backup code before starting')
        script=start_script(dict(namespace=namespace,run_id=RUN,outer=args.outer,source_base=args.source_base,
            cpu_proof=args.cpu_proof,cpu_proof_sha256=args.cpu_proof_sha256),code,commit)
    else:script=status_script(namespace)
    path=Path(args.receipts)/('prefix-selection200-backup-'+args.phase+'-'+str(time.time_ns())+'.json')
    result=json.loads(remote(script,path,target='h100-10'));actual=path.with_name(path.stem+'-actual.json')
    actual.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(path=str(actual),result=result)))


if __name__=='__main__':main()
