"""Control-only SSH transport: source/configuration in, small summaries out."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
REPO=ROOT/'work/dynrpo7b-formal-20261023'
BASE='/data/user/jxu957/experiments/dynrpo7b-formal-20261023'


def remote(script):
    memory=Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text()
    password=memory.split('账号 mou176；密码 ')[1].splitlines()[0]
    args=['sshpass','-e','ssh','-oStrictHostKeyChecking=yes',
        '-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn',
        '-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts',
        'mou176@10.120.81.30','python3 -']
    # Send code through stdin on both SSH hops, avoiding shell/argument limits.
    worker='import os,subprocess;os.environ["SSHPASS"]='+repr(password)+';raise SystemExit(subprocess.run('+repr(args)+',input='+repr(script)+',universal_newlines=True).returncode)'
    result=subprocess.run(['ssh','-oConnectTimeout=10','hkust3','python3 -'],input=worker,
                          text=True,capture_output=True,timeout=180)
    if result.stdout: print(result.stdout,end='')
    if result.stderr: print(result.stderr,end='')
    if result.returncode: raise SystemExit(result.returncode)


def deploy(commit):
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=REPO,text=True).splitlines()
    names=[n for n in names if Path(n).suffix in ('.py','.sh','.json','.yaml','.yml') or
           n in ('verl/version/version','verl/single_controller/version/version')]
    # Bundle committed source only. Model/tensor/cache files cannot enter it.
    archive=subprocess.check_output(['git','archive','--format=tar',commit,*names],cwd=REPO)
    records={}
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source:
            if member.isfile():
                data=source.extractfile(member).read()
                records[member.name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    import gzip
    packed=base64.b64encode(gzip.compress(archive)).decode()
    script='''import base64,gzip,hashlib,io,json,pathlib,tarfile
base=pathlib.Path(BASE); destination=base/'source'/COMMIT
destination.mkdir(parents=True,exist_ok=True)
raw=gzip.decompress(base64.b64decode(PAYLOAD))
with tarfile.open(fileobj=io.BytesIO(raw)) as source:
 for member in source:
  name=pathlib.PurePosixPath(member.name)
  if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()): raise ValueError('Unsafe source member')
 source.extractall(str(destination))
for name,record in FILES.items():
 data=(destination/name).read_bytes()
 if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']: raise ValueError('Source SHA differs: '+name)
receipt=dict(source_commit=COMMIT,fresh_source_all_files_sha_verified=True,files=FILES)
(destination/'source-publication.json').write_text(json.dumps(receipt,sort_keys=True))
print(json.dumps(dict(status='REMOTE_SOURCE_ALL_SHA_PASS',commit=COMMIT,files=len(FILES),source=str(destination))))
'''
    remote('BASE='+repr(BASE)+'\nCOMMIT='+repr(commit)+'\nFILES='+repr(records)+'\nPAYLOAD='+repr(packed)+'\n'+script)


def command(module,arguments):
    remote('import subprocess\nraise SystemExit(subprocess.call('+repr([
        '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021/env/bin/python',
        '-B','-m',module,*arguments])+',cwd='+repr(BASE+'/repo')+'))')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['source','python','script'])
    p.add_argument('--commit');p.add_argument('--module');p.add_argument('arguments',nargs='*');a=p.parse_args()
    if a.mode=='source':deploy(a.commit)
    elif a.mode=='script':
        import sys
        remote(sys.stdin.read())
    else:command(a.module,a.arguments)
