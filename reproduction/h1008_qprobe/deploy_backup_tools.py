"""Install published CPU backup controls outside the frozen training checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from reproduction.h1008_qprobe.management_transport import remote

REMOTE='''import pathlib,json,hashlib,os,socket,subprocess
P=pathlib.Path;meta=META;target=meta['target'];assert socket.gethostname().split('.')[0]==target
if target=='h100-8':
 base=P(meta['base']);run=json.loads((base/'logs/run-manifest.json').read_text())
 assert run['run_id']==meta['run_id'] and run['question_probe']['loss_version']in ('std_class_ce_v1','binary_mix_variance_v1')
 root=base/'preparation/backup-tools'/meta['commit'];assert root.resolve().is_relative_to(base.resolve()/'preparation')
else:
 root=P('/mnt/data/shaoshitong/search-g1-qprobe3-independent-backup-20261015/published-tools')/meta['commit']
 root.parent.mkdir(parents=True,exist_ok=True)
 assert subprocess.check_output(['findmnt','-n','-T',str(root.parent),'-o','FSTYPE'],text=True).strip()=='ext4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
if root.exists():
 receipt=root/'RECEIPT.json'
 if not receipt.exists():print(json.dumps(dict(status='EXISTING_INTENT_INSPECT_REQUIRED',root=str(root))));raise SystemExit
 old=json.loads(receipt.read_text());assert old['commit']==meta['commit'] and old['run_id']==meta['run_id']
 for name,row in meta['files'].items():assert sha(root/name)==row['sha256']
 print(json.dumps(dict(old,action='existing code verified; no replay')));raise SystemExit
root.mkdir(mode=0o700,parents=True)
once(root/'INTENT.json',dict(commit=meta['commit'],run_id=meta['run_id'],target=target,files={name:row['sha256'] for name,row in meta['files'].items()},gpu_use=False,copy_launched=False))
for name,row in meta['files'].items():
 assert P(name).name==name and hashlib.sha256(row['source'].encode()).hexdigest()==row['sha256']
 path=root/name;compile(row['source'],str(path),'exec')
 with path.open('x') as f:f.write(row['source']);f.flush();os.fsync(f.fileno())
 assert sha(path)==row['sha256']
result=dict(status='CPU_BACKUP_CONTROL_SOURCE_FULL_SHA_PASS',target=target,root=str(root),commit=meta['commit'],run_id=meta['run_id'],files={name:row['sha256'] for name,row in meta['files'].items()},frozen_runtime_changed=False,copy_launched=False)
once(root/'RECEIPT.json',result);print(json.dumps(result))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--base',required=True);p.add_argument('--run-id',required=True);p.add_argument('--control-path',required=True);p.add_argument('--evidence-prefix',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip();proof=json.loads(a.proof.read_text())
    assert proof['published_control_commit']==commit and proof['all_blobs_full_sha_match']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo)
    paths={'h100-8':('inventory.py','reproduction/h1008_qprobe/backup_inventory.py'),'h100-10':('copy-worker.py','reproduction/h1008_qprobe/independent_copy.py')}
    for target,(name,relative) in paths.items():
        source=(repo/relative).read_text();sha=hashlib.sha256(source.encode()).hexdigest();assert sha==proof['files'][relative]
        meta=dict(target=target,base=a.base,run_id=a.run_id,commit=commit,files={name:dict(source=source,sha256=sha)})
        result=json.loads(remote(REMOTE.replace('META',repr(meta)),a.evidence_prefix+'-'+target+'.json',a.control_path,target=target))
        print(json.dumps(result),flush=True)
        assert result['status']=='CPU_BACKUP_CONTROL_SOURCE_FULL_SHA_PASS'


if __name__=='__main__':main()
