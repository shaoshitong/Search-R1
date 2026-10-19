"""Install the already published small sidecar helper; never start its archive run."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

SCRIPT = 'reproduction/easy2560_joint4/code_backup/micro8-after731063-20261018/current-sidecar-archive734689.py'
PUBLISHED = '8428e661808b00363f0a8c08128c41194aa730d3'
SHA = '1b5d161ad1dd0360977446bc4c212cc63b348a76642c191f42526437cb101f73'
TARGET = '/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/postprocessing-operators/job-734689/current-sidecar-archive734689.py'
INSTALL = r'''import hashlib,json,os,socket,stat,sys,time
from pathlib import Path
p=json.load(sys.stdin)
assert socket.gethostname()=='xielab3' and os.getuid()==1002
target=Path(p['target']);raw=p['source'].encode()
assert str(target)=='/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/postprocessing-operators/job-734689/current-sidecar-archive734689.py'
assert hashlib.sha256(raw).hexdigest()==p['sha256']
target.parent.mkdir(parents=True,exist_ok=True)
assert target.parent.resolve()==target.parent
created=False
if target.exists() or target.is_symlink():
 s=target.lstat();assert stat.S_ISREG(s.st_mode) and not target.is_symlink()
 assert target.read_bytes()==raw
else:
 with target.open('xb') as f:
  f.write(raw);f.flush();os.fsync(f.fileno())
 created=True
fd=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
assert target.read_bytes()==raw
print(json.dumps(dict(status='SMALL_SIDECAR_INSTALLED_SHA_PASS',path=str(target),sha256=p['sha256'],bytes=len(raw),created=created,checked_unix=time.time(),training_modified=False,archive_started=False)))
'''
VERIFY = r'''import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);target=Path(p['target'])
assert socket.gethostname()=='xielab3' and os.getuid()==1002
assert target.resolve()==target and hashlib.sha256(target.read_bytes()).hexdigest()==p['sha256']
child=subprocess.run(['/usr/bin/python3','-B',str(target),'--step','50'],capture_output=True,text=True,timeout=15)
assert child.returncode==0,child.stderr
plan=json.loads(child.stdout)
assert plan['status']=='DRY_RUN_NO_CREDENTIAL_READ_NO_SSH' and plan['source_job']=='734689' and plan['step']==50
print(json.dumps(dict(status='SMALL_SIDECAR_INDEPENDENT_SHA_AND_DRY_RUN_PASS',path=str(target),sha256=p['sha256'],bytes=target.stat().st_size,plan=plan,checked_unix=time.time(),training_modified=False,archive_started=False)))
'''


def main():
    w = Path.cwd()
    out = w.parents[1] / 'outputs/easy2560-co-rollout-hpc3-4gpu-20261017'
    source = subprocess.check_output(['git', 'show', PUBLISHED + ':' + SCRIPT], cwd=w)
    assert hashlib.sha256(source).hexdigest() == SHA and (w / SCRIPT).read_bytes() == source
    spec = importlib.util.spec_from_file_location('existing_hk_transport', w / 'reproduction/easy2560_joint4/code_backup/recovery5-after730912-hk-source-8033-20261018/deploy_exact8033_HK_source_once.py')
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    tag = 'current734689-small-sidecar-HK-' + str(int(time.time()))
    payload = dict(target=TARGET, sha256=SHA, source=source.decode())
    installed = transport.run_ssh(INSTALL, payload, out / (tag + '-install.json'))
    verified = transport.run_ssh(VERIFY, dict(target=TARGET, sha256=SHA), out / (tag + '-independent-readback.json'))
    result = dict(status='PUBLISHED_SIDECAR_HK_DEPLOYED_AND_INDEPENDENTLY_VERIFIED', published_source_commit=PUBLISHED, installed=installed, verified=verified, training_modified=False, archive_started=False, credentials_written=False)
    transport.save(out / (tag + '-complete.json'), result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
