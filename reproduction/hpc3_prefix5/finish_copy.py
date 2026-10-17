"""Finish accepted A800->HKUST2->HKUST3->HPC3 copy without Mac asset bytes."""
import json,pathlib,shlex,subprocess,time
from .direct_control import password_for
ROOT=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
RECEIVER=r'''
import sys,json,pathlib,hashlib,os,fcntl,tarfile,time
p=json.loads(sys.stdin.buffer.readline());b=pathlib.Path(p['target']);b.mkdir(parents=True,exist_ok=True)
with (b/'receive.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (b/'receive-complete.json').exists():raise ValueError('Already received: inspect existing receipt, do not stream twice')
 part=b/'reference-assets.tar.part';target=b/'reference-assets.tar';h=hashlib.sha256();count=0
 if part.exists() or target.exists():raise ValueError('Existing target needs explicit resume reconciliation')
 with part.open('xb') as out:
  while True:
   chunk=sys.stdin.buffer.read(4*1024**2)
   if not chunk:break
   count+=len(chunk)
   if count>p['proof']['archive_bytes']:raise ValueError('Oversized archive input')
   out.write(chunk);out.flush();h.update(chunk)
   (b/'receive-progress.json').write_text(json.dumps(dict(epoch=time.time(),bytes=count,total=p['proof']['archive_bytes'],mac_model_data=False)))
  os.fsync(out.fileno())
 if count!=p['proof']['archive_bytes'] or h.hexdigest()!=p['proof']['archive_sha256']:raise ValueError('Actual received source archive size/SHA differs')
 part.replace(target);(b/'source-export.json').write_text(json.dumps(p['proof']))
 if p.get('extract'):
  assets=b/'reference-assets';assets.mkdir(exist_ok=True)
  files={x['path']:x for x in p['proof']['manifest']['files']}
  with tarfile.open(target,'r') as tar:
   members=tar.getmembers()
   if {m.name for m in members}!=set(files)|{'manifest.json'} or any(not m.isfile() for m in members):raise ValueError('Archive inventory differs')
   for m in members:
    rel=pathlib.Path(m.name)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe archive path')
    dest=assets/rel;dest.parent.mkdir(exist_ok=True,parents=True)
    with tar.extractfile(m) as source,dest.open('xb') as out:
     for chunk in iter(lambda:source.read(8*1024**2),b''):out.write(chunk)
  for name,expected in files.items():
   q=assets/name;h=hashlib.sha256()
   with q.open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
   if q.stat().st_size!=expected['bytes'] or h.hexdigest()!=expected['sha256']:raise ValueError('Extracted target file SHA differs')
 result=dict(status='REFERENCE_ASSETS_ALL_TARGET_SHA_VERIFIED' if p.get('extract') else 'INTERMEDIATE_ARCHIVE_SHA_VERIFIED',bytes=count,archive_sha256=p['proof']['archive_sha256'],files=len(p['proof']['manifest']['files']),epoch=time.time(),mac_model_data=False)
 (b/'receive-complete.json').write_text(json.dumps(result));print(json.dumps(result))
'''

def run2():
 code=r'''
import subprocess,pathlib,json,sys,shlex
p=json.load(sys.stdin);b=pathlib.Path('/data/shaoshitong/thread5-prefix5-hpc3-sync-20261017');proof=json.loads((b/'copy-complete.json').read_text())
assert proof['status']=='SOURCE_HKUST2_ARCHIVE_SHA_VERIFIED'
command=['ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8','shaoshitong@10.120.17.122','python3 -c '+shlex.quote(p['receiver'])]
child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
child.stdin.write((json.dumps(dict(target='/backup/shaoshitong/thread5-prefix5-hpc3-sync-20261017',proof=proof))+'\n').encode())
with (b/'reference-assets.tar').open('rb') as f:
 for chunk in iter(lambda:f.read(4*1024**2),b''):child.stdin.write(chunk)
child.stdin.close();out=child.stdout.read();err=child.stderr.read();code=child.wait();print(out.decode());print(err.decode(),file=sys.stderr);raise SystemExit(code)
'''
 return subprocess.run(['ssh','-oConnectTimeout=8','hkust2','python3 -c '+shlex.quote(code)],input=json.dumps(dict(receiver=RECEIVER)).encode(),capture_output=True,timeout=90)


def run3():
 code=r'''
import subprocess,pathlib,json,sys,shlex,os
p=json.load(sys.stdin);password=p.pop('password');b=pathlib.Path('/backup/shaoshitong/thread5-prefix5-hpc3-sync-20261017');proof=json.loads((b/'source-export.json').read_text());accepted=json.loads((b/'receive-complete.json').read_text());assert accepted['archive_sha256']==proof['archive_sha256']
env=dict(os.environ,SSHPASS=password)
command=['sshpass','-e','ssh','-T','-oStrictHostKeyChecking=yes','-oControlMaster=no','-S','none','-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn','-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts','-oConnectTimeout=10','mou176@10.120.81.30',p['python']+' -c '+shlex.quote(p['receiver'])]
child=subprocess.Popen(command,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
child.stdin.write((json.dumps(dict(target='/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017',proof=proof,extract=True))+'\n').encode())
with (b/'reference-assets.tar').open('rb') as f:
 for chunk in iter(lambda:f.read(4*1024**2),b''):child.stdin.write(chunk)
child.stdin.close();out=child.stdout.read();err=child.stderr.read();code=child.wait();print(out.decode());print(err.decode(),file=sys.stderr);raise SystemExit(code)
'''
 request=dict(receiver=RECEIVER,password=password_for('mou176').decode(),python='/data/user/jxu957/experiments/topd-20261008/environment/topd-py310/bin/python')
 return subprocess.run(['ssh','-oConnectTimeout=8','hkust3','python3 -c '+shlex.quote(code)],input=json.dumps(request).encode(),capture_output=True,timeout=90)


def main():
 out=ROOT/'outputs/hpc3-prefix5-switch-20261017'
 for phase,fn in [('hkust3',run2),('hpc3',run3)]:
  path=out/(phase+'-receive.json')
  if path.exists():
   if json.loads(path.read_text())['returncode']!=0:raise RuntimeError('Prior failed/unknown receive needs reconciliation')
   continue
  result=fn();record=dict(returncode=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode(),epoch=time.time(),mac_model_data=False)
  path.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
  if result.returncode:raise SystemExit(result.returncode)

if __name__=='__main__':main()
