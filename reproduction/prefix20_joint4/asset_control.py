"""Mac control only: publish public asset code; send HF credential over stdin."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT='/data/shaoshitong/qprobe-prefix20-hpc3-4gpu-20261016/assets'

REMOTE = r'''
import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);token=p.pop('token');root=Path(p['root'])
root.mkdir(parents=True,exist_ok=True)
if (root/'launch-claim.json').exists():raise ValueError('EXISTING_ASSET_CLAIM_INSPECT_BEFORE_START')
files=[]
for name,value in p['files'].items():
 raw=base64.b64decode(value);target=root/name;digest=hashlib.sha256(raw).hexdigest()
 if target.exists():
  if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('DEPLOYED_SOURCE_DIFFERS')
 else:
  with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 files.append(dict(path=str(target),sha256=digest,bytes=len(raw)))
publication=dict(source_commit=p['source_commit'],remote_source_files=files,checked_unix=time.time())
with (root/'source-publication.json').open('x') as f:json.dump(publication,f,indent=2)
with (root/'launch-claim.json').open('x') as f:json.dump(dict(source_commit=p['source_commit'],created_unix=time.time()),f)
argv=['/usr/bin/python3',str(root/'assets.py'),'--root',str(root),'--plan',str(root/'probe-asset-plan.json'),'--proxy','http://127.0.0.1:17898']
with (root/'download.log').open('ab') as out:
 child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=out,stderr=out,start_new_session=True)
 child.stdin.write((token+'\n').encode());child.stdin.close()
start_ticks=int(Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
receipt=dict(publication,pid=child.pid,start_ticks=start_ticks,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(root),token_persisted=False,training_job_submitted=False)
with (root/'launch.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-commit',required=True)
    p.add_argument('--receipt',required=True);a=p.parse_args()
    here=Path(__file__).resolve().parent
    head=subprocess.check_output(['git','-C',str(here),'rev-parse','HEAD'],text=True).strip()
    remote=subprocess.check_output(['git','-C',str(here),'ls-remote',
        'git@github.com:shaoshitong/Search-R1.git',
        'refs/heads/codex/qprobe-prefix20-hpc3-4gpu-20261016'],text=True).split()[0]
    if head!=a.source_commit or remote!=head:raise ValueError('LATEST_SOURCE_NOT_PUBLISHED')
    token=Path.home().joinpath('.cache/huggingface/token').read_text().strip()
    payload=dict(token=token,root=ROOT,source_commit=head,
                 files={name:base64.b64encode((here/name).read_bytes()).decode()
                        for name in ('assets.py','probe-asset-plan.json')})
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'hkust2','python3 -c '+shlex.quote(REMOTE)],input=json.dumps(payload),
        text=True,capture_output=True,timeout=40)
    if result.returncode:
        raise RuntimeError('REMOTE_ASSET_LAUNCH_FAILED; inspect remote owned claim before retry')
    receipt=json.loads(result.stdout)
    Path(a.receipt).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
