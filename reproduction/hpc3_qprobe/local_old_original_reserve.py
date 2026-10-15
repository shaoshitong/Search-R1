"""Publish/control only the fixed non-destructive remote reserve; RAM credentials."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time

from reproduction.hpc3_qprobe import reserve_old_probe_originals as r


REMOTE_PUBLISH=r'''import base64,hashlib,io,json,os,socket,stat,tarfile,time
from pathlib import Path
v=json.load(__import__('sys').stdin);root=Path(v['base']);sha=v['commit'];os.umask(0o077)
assert str(root)=='/backup/shaoshitong/qprobe-old-original-reserve-20261016'
actual=dict(hostname=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pid_namespace=os.readlink('/proc/self/ns/pid'))
assert os.getuid()==1002 and actual==v['target_host']
assert len(sha)==40 and all(c in '0123456789abcdef' for c in sha) and sha==v['published_remote_sha']
parent=root.parent;s=parent.lstat();assert parent.resolve()==parent and stat.S_ISDIR(s.st_mode) and s.st_uid==1002
if not root.exists():root.mkdir(mode=0o700)
s=root.lstat();assert root.resolve()==root and stat.S_ISDIR(s.st_mode) and s.st_uid==1002
assert not (root/'control').exists(),'Existing backup writer must retain its pinned source'
fs=os.statvfs(str(root));assert fs.f_bavail*fs.f_frsize>2*v['planned_payload_bytes']+2**30
raw=base64.b64decode(v.pop('archive'));assert len(raw)<30*2**20 and hashlib.sha256(raw).hexdigest()==v['archive_sha256']
source=root/'source'/sha;partial=source.with_suffix('.partial');source.parent.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
 members=t.getmembers();names=[]
 for m in members:
  assert not m.name.startswith('/') and '..' not in Path(m.name).parts and (m.isdir() or m.isfile())
  if m.isfile():names.append(m.name)
 assert len(names)==len(set(names)) and set(names)==set(v['files'])
 if not source.exists():
  assert not partial.exists(),'Inspect preserved partial deployment; do not overwrite'
  partial.mkdir()
  t.extractall(str(partial))
  for n,h in v['files'].items():assert hashlib.sha256((partial/n).read_bytes()).hexdigest()==h,n
  partial.rename(source)
assert source.resolve()==source
for n,h in v['files'].items():
 p=source/n;s=p.lstat();assert p.resolve()==p and stat.S_ISREG(s.st_mode) and s.st_uid==1002
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n
preparation=root/'preparation';preparation.mkdir(exist_ok=True)
receipt=preparation/('source-'+sha+'-verification.json')
out=dict(v,status='PUBLISHED_SOURCE_FULL_SHA_PASS',source=str(source),file_count=len(v['files']),checked_unix=time.time(),actual_host=actual,actual_uid=os.getuid(),source_cleanup_performed=False)
if receipt.exists():
 old=json.loads(receipt.read_text());assert old['files']==v['files'] and old['commit']==sha and old['status']==out['status'];out=old
else:
 with receipt.open('x') as f:json.dump(out,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
fd=os.open(str(preparation),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(out))
'''


def ssh(command,payload):
    return subprocess.run(['/usr/bin/ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes',
        '-oConnectTimeout=8','-oControlMaster=no','-S','none','hkust3',command],
        input=payload,text=True,capture_output=True,timeout=55)


def credential():
    text=Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text()
    line=next(s for s in text.splitlines() if s.startswith('账号 jxu957'))
    password=re.split(r'密码\s*[:：]?\s*',line,1)[1].strip().strip('`')
    text=line=None
    r.need(bool(password) and '\n' not in password,'missing RAM credential')
    return password


def payload(repo,commit,branch):
    remote=subprocess.check_output(['git','ls-remote','git@github.com:shaoshitong/Search-R1.git',
                                   'refs/heads/'+branch],text=True).split()[0]
    r.need(remote==commit,'latest source has not been pushed')
    archive=subprocess.check_output(['git','-C',str(repo),'archive','--format=tar.gz',commit])
    r.need(len(archive)<30*2**20,'code archive unexpectedly large')
    files={}
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as t:
        for m in t.getmembers():
            r.need(not m.name.startswith('/') and '..' not in Path(m.name).parts and (m.isdir() or m.isfile()),
                   'unsafe source archive')
            if m.isfile():files[m.name]=hashlib.sha256(t.extractfile(m).read()).hexdigest()
    plan=r.plan()
    r.need(files[r.RELATIVE]==hashlib.sha256((repo/r.RELATIVE).read_bytes()).hexdigest() and
           files['reproduction/hpc3_qprobe/'+r.PLAN_NAME]==r.PLAN_SHA,'local implementation differs from publication')
    return dict(base=str(r.TARGET),commit=commit,published_remote_sha=remote,target_host=plan['target_host'],
        planned_payload_bytes=plan['total_bytes'],archive_sha256=hashlib.sha256(archive).hexdigest(),
        files=files,archive=base64.b64encode(archive).decode())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('publish','preflight','start','inspect'))
    parser.add_argument('--commit',required=True);parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--branch',default='codex/qprobe-sigma3-loss-20261015');a=parser.parse_args()
    r.need(re.fullmatch('[0-9a-f]{40}',a.commit),'explicit immutable source required')
    repo=Path(__file__).resolve().parents[2]
    if a.mode=='publish':
        command='python3 -c '+shlex.quote(REMOTE_PUBLISH)
        data=json.dumps(payload(repo,a.commit,a.branch))
    else:
        args=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1']+[k+'=1' for k in r.CAPS]
        args+=['python3',str(r.TARGET/'source'/a.commit/r.RELATIVE),a.mode,'--commit',a.commit]
        command=' '.join(shlex.quote(x) for x in args)
        data=credential()+'\n' if a.mode in ('preflight','start') else ''
    result=ssh(command,data);data=None
    observation=dict(mode=a.mode,source_commit=a.commit,returncode=result.returncode,
        stdout=result.stdout,stderr=result.stderr,epoch=time.time(),source_cleanup_performed=False)
    a.audit_dir.mkdir(parents=True,exist_ok=True)
    audit=a.audit_dir/('old-original-reserve-'+a.mode+'-'+str(time.time_ns())+'.json')
    audit.write_text(json.dumps(observation,ensure_ascii=False,indent=2)+'\n')
    r.need(result.returncode==0,'actual remote '+a.mode+' failed; evidence '+str(audit))
    value=json.loads(result.stdout)
    if a.mode=='publish':value={k:value[k] for k in ('status','commit','source','file_count','archive_sha256','checked_unix')}
    print(json.dumps(dict(audit=str(audit),observed=value),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
