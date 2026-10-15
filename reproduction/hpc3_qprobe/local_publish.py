"""Publish verified immutable Git source through the existing RAM-only bridge."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from reproduction.hpc3_qprobe.local_account_bridge import run

REMOTE=r'''import base64,hashlib,io,json,os,tarfile,time
from pathlib import Path
v=json.load(__import__('sys').stdin);base=Path(v['base']);sha=v['commit']
assert str(base)=='/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014'
assert len(sha)==40 and sha==v['published_remote_sha'];os.umask(0o007)
raw=base64.b64decode(v.pop('archive'));assert hashlib.sha256(raw).hexdigest()==v['archive_sha256']
source=base/'source'/sha;partial=source.with_suffix('.partial');source.parent.mkdir(parents=True,exist_ok=True)
if not source.exists():
 assert not partial.exists(),'Inspect incomplete source transport before repeating'
 partial.mkdir()
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
  for m in t.getmembers():assert not m.name.startswith('/') and '..' not in Path(m.name).parts and (m.isdir() or m.isfile())
  t.extractall(str(partial))
 for n,h in v['files'].items():assert hashlib.sha256((partial/n).read_bytes()).hexdigest()==h,n
 partial.rename(source)
for n,h in v['files'].items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h,n
receipt=base/'preparation'/('source-'+sha+'-verification.json')
out=dict(v,status='PUBLISHED_SOURCE_FULL_SHA_PASS',source=str(source),file_count=len(v['files']),checked_unix=time.time())
if receipt.exists():
 old=json.loads(receipt.read_text());assert old['files']==v['files'] and old['commit']==sha and old['status']==out['status'];out=old
else:
 with receipt.open('x') as f:json.dump(out,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(out))
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control-work',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--login-host')
    parser.add_argument('--host-identity-json',type=json.loads)
    parser.add_argument('--branch',default='codex/qprobe-teacher-warmup-hpc3-20261014');args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','git@github.com:shaoshitong/Search-R1.git',
        'refs/heads/'+args.branch],text=True).split()[0]
    assert remote==sha,'Latest implementation has not been pushed'
    archive=subprocess.check_output(['git','-C',str(repo),'archive','--format=tar.gz',sha])
    assert len(archive)<30*1024**2
    files={}
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as t:
        for m in t.getmembers():
            assert not m.name.startswith('/') and '..' not in Path(m.name).parts and (m.isdir() or m.isfile())
            if m.isfile():files[m.name]=hashlib.sha256(t.extractfile(m).read()).hexdigest()
    payload=dict(base='/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014',commit=sha,
        published_remote_sha=remote,archive_sha256=hashlib.sha256(archive).hexdigest(),files=files,
        archive=base64.b64encode(archive).decode())
    # The account's actual shell access has been restored. New deployment
    # receipts must be authored by JXU for the later native restore audit.
    binding={}
    if args.login_host is not None:binding['login_host']=args.login_host
    if args.host_identity_json is not None:binding['expected_host_identity']=args.host_identity_json
    result=run('jxu957','python3 -c '+shlex.quote(REMOTE),json.dumps(payload),args.control_work,**binding)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout)
    assert value['commit']==sha and value['file_count']==len(files) and value['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS'
    print(json.dumps({k:value[k] for k in ('status','commit','source','file_count','archive_sha256','checked_unix')}),flush=True)
    args.audit_dir.mkdir(exist_ok=True,parents=True)
    (args.audit_dir/('deployment-'+sha+'.json')).write_text(json.dumps(value,indent=2))


if __name__=='__main__':main()
