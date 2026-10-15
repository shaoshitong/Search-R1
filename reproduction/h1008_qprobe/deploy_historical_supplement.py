"""Deploy supplement CODE ONLY to independent directories on8 and10; no prepare/copy."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from reproduction.h1008_qprobe.management_transport import remote

REMOTE='''import pathlib,json,hashlib,os,socket
meta=META;P=pathlib.Path
assert socket.gethostname().split('.')[0]==meta['target']
root=P('/mnt/data/shaoshitong/search-g1-backup-supplement-control-'+meta['commit'][:7]+'-20261015')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
if root.exists():raise ValueError('EXISTING_CONTROL_INSPECT_DO_NOT_REPLAY')
root.mkdir(mode=0o700)
once(root/'INTENT.json',dict(commit=meta['commit'],target=meta['target'],code_only=True,files={n:v['sha256'] for n,v in meta['files'].items()}))
for name,row in meta['files'].items():
 assert P(name).name==name and hashlib.sha256(row['source'].encode()).hexdigest()==row['sha256']
 if name.endswith('.py'):compile(row['source'],name,'exec')
 with (root/name).open('x') as f:f.write(row['source']);f.flush();os.fsync(f.fileno())
 assert sha(root/name)==row['sha256']
receipt=dict(status='HISTORICAL_SUPPLEMENT_CODE_ONLY_FULL_SHA_PASS',commit=meta['commit'],target=meta['target'],root=str(root),files={n:v['sha256'] for n,v in meta['files'].items()},source_prepared=False,copy_launched=False,training_modified=False)
once(root/'RECEIPT.json',receipt);print(json.dumps(receipt))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--proof',required=True);p.add_argument('--control-path',required=True);p.add_argument('--evidence-prefix',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();proof=json.loads(Path(a.proof).read_text())
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo) or proof.get('published_control_commit')!=commit or proof.get('all_blobs_full_sha_match') is not True:raise ValueError('CLEAN_PUBLISHED_FULL_SHA_CONTROL_REQUIRED')
    names={'historical-supplement.py':'historical_supplement.py','historical-supplement-plan.json':'historical_supplement_plan.json','test-historical-supplement.py':'test_historical_supplement.py','deploy-historical-supplement.py':'deploy_historical_supplement.py'}
    files={}
    for name,relative in names.items():
        path='reproduction/h1008_qprobe/'+relative;source=(repo/path).read_text();sha=hashlib.sha256(source.encode()).hexdigest()
        if proof['files'].get(path)!=sha:raise ValueError('CONTROL_FILE_DIFFERS_FROM_PUBLISHED_BLOB')
        files[name]=dict(source=source,sha256=sha)
    for target in ('h100-8','h100-10'):
        print(remote(REMOTE.replace('META',repr(dict(commit=commit,target=target,files=files))),a.evidence_prefix+'-'+target+'.json',a.control_path,target=target),flush=True)

if __name__=='__main__':main()
