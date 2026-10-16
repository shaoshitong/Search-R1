"""Publish this complete committed source to the pinned HPC3 login node."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

BASE='/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-20261016'
PYTHON='/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015/env/bin/python'
IDENTITY=dict(hostname='ACD-Login-1',boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',pid_namespace='pid:[4026531836]')
REMOTE=r'''
import sys,json,base64,tarfile,io,hashlib,time
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['base'])/'source'/p['commit'];root.mkdir(parents=True,exist_ok=False);seen=set()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(p['tar_gz'])),mode='r:gz') as tar:
 for m in tar:
  if m.isdir():continue
  if not m.isfile() or m.name not in p['manifest'] or m.name in seen:raise ValueError('SOURCE_ARCHIVE_MEMBER_DIFFERS')
  seen.add(m.name);target=root/m.name;target.parent.mkdir(parents=True,exist_ok=True);data=tar.extractfile(m).read();rec=p['manifest'][m.name]
  if len(data)!=rec['bytes'] or hashlib.sha256(data).hexdigest()!=rec['sha256']:raise ValueError('SOURCE_ARCHIVE_SHA_DIFFERS')
  target.write_bytes(data)
assert seen==set(p['manifest'])
assert all((root/n).stat().st_size==r['bytes'] and hashlib.sha256((root/n).read_bytes()).hexdigest()==r['sha256'] for n,r in p['manifest'].items())
proof=dict(source_commit=p['commit'],files=p['manifest'],fresh_source_all_files_sha_verified=True,checked_unix=time.time())
(root/'source-publication.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(dict(source_commit=p['commit'],source_root=str(root),verified_files=len(seen),verified_bytes=sum(r['bytes'] for r in p['manifest'].values()),fresh_source_all_files_sha_verified=True,checked_unix=proof['checked_unix'])))
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bridge-source',required=True)
    parser.add_argument('--control-work',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--base',default=BASE)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True):
        raise ValueError('SOURCE_HAS_UNPUBLISHED_CHANGES')
    remote=subprocess.check_output(['git','-C',str(repo),'ls-remote','git@github.com:shaoshitong/Search-R1.git',
        'refs/heads/codex/qprobe-prefix20-hpc3-4gpu-20261016'],text=True).split()[0]
    if remote!=commit:raise ValueError('LATEST_SOURCE_NOT_PUSHED')
    names=[n for n in subprocess.check_output(['git','-C',str(repo),'ls-files','-z']).decode().split('\0') if n]
    if any((repo/n).is_symlink() for n in names):raise ValueError('SOURCE_SYMLINKS_NOT_SUPPORTED')
    manifest={n:dict(sha256=hashlib.sha256((repo/n).read_bytes()).hexdigest(),bytes=(repo/n).stat().st_size) for n in names}
    raw=subprocess.check_output(['git','-C',str(repo),'archive','--format=tar.gz','HEAD'])
    spec=importlib.util.spec_from_file_location('bound_account_bridge',args.bridge_source)
    bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    from reproduction.prefix20_joint4.control import own_base
    payload=json.dumps(dict(base=str(own_base(args.base)),commit=commit,manifest=manifest,tar_gz=base64.b64encode(raw).decode()))
    a=bridge.run('jxu957',PYTHON+' -c '+shlex.quote(REMOTE),payload,args.control_work,
        login_host='10.120.48.31',observation_timeout=90,expected_host_identity=IDENTITY)
    result=dict(returncode=a.returncode,stdout=a.stdout,stderr=a.stderr,checked_unix=time.time())
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));raise SystemExit(a.returncode)


if __name__=='__main__':main()
