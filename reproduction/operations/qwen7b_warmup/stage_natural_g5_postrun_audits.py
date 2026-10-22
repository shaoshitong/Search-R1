"""Stage post-run audit code separately, never change an active training entry.

Local CPU preparation emits a small remote deployment script. Root may execute
it through the existing private control after network permission is restored.
It verifies the publication externally before running this stage script. No
network or remote action is performed by this local generator itself.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

FILES=('audit_natural_g5_online_checkpoint.py','compare_natural_g5_online_matched.py')
PACKAGE='reproduction/operations/qwen7b_warmup/'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-natural-g5-8192-20261022-v1'
REMOTE=r'''import pathlib,json,base64,gzip,hashlib,time
base=pathlib.Path(V['base']);assert base.is_dir();root=base/('postrun-audit-code-'+V['commit'][:12]);root.mkdir(exist_ok=True)
refs={}
for name,item in V['files'].items():
 raw=gzip.decompress(base64.b64decode(item['gzip64']));assert len(raw)==item['bytes']and hashlib.sha256(raw).hexdigest()==item['sha256'];p=root/name
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 compile(raw,name,'exec');refs[name]=dict(path=str(p),bytes=len(raw),sha256=item['sha256'])
receipt=dict(status='POSTRUN_AUDIT_CODE_SEPARATELY_STAGED_NOT_EXECUTED',commit=V['commit'],files=refs,epoch=time.time(),training_entries_or_locks_or_Slurm_modified=False)
p=root/'source-publication.json'
if p.exists():assert json.loads(p.read_text())['files']==refs
else:p.write_text(json.dumps(receipt,sort_keys=True)+'\n')
print(json.dumps(receipt))
'''


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();repo=Path(__file__).resolve().parents[3]
 payload={}
 for name in FILES:
  raw=subprocess.check_output(['git','show',args.commit+':'+PACKAGE+name],cwd=repo)
  payload[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),gzip64=base64.b64encode(gzip.compress(raw,mtime=0)).decode())
 source='V='+repr(dict(base=BASE,commit=args.commit,files=payload))+'\n'+REMOTE
 compile(source,'stage-remote-postrun-audits','exec');args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(source)
 print(json.dumps(dict(status='LOCAL_AUDIT_STAGE_SCRIPT_CREATED_NO_REMOTE_ACTION',path=str(args.output),bytes=len(source.encode()),commit=args.commit)))


if __name__=='__main__':main()
