"""Forward only the two freshly verified canonical inputs HKUST2 to HKUST3."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

try:
    from reproduction.prefix20_joint4.assets import sha,atomic
except ModuleNotFoundError:
    from assets import sha,atomic

SOURCE=Path('/data/shaoshitong/qprobe-prefix20-hpc3-4gpu-20261016/canonical-data')
TARGET='/backup/shaoshitong/qprobe-prefix20-hpc3-4gpu-20261016/canonical-data'
PEER='shaoshitong@10.120.17.122'


def remote(code):
    p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',
        PEER,'python3 -c '+shlex.quote(code)],text=True,capture_output=True,timeout=90)
    if p.returncode:raise RuntimeError('CANONICAL_FORWARD_REMOTE_FAILED_INSPECT_EXISTING_CLAIM')
    return json.loads(p.stdout)


def main():
    if socket.gethostname()!='xielab2':raise ValueError('CANONICAL_FORWARD_REQUIRES_HKUST2')
    with (SOURCE/'forward-hkust3.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof=json.loads((SOURCE/'complete.json').read_text())
        if proof['status']!='CANONICAL_SELECTED_MEMBERS_FRESH_SHA_VERIFIED' or not proof['full_compressed_archive_verified']:
            raise ValueError('CANONICAL_SOURCE_NOT_FRESH_VERIFIED')
        for row in proof['files']:
            path=SOURCE/'selected'/row['path']
            if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
                raise ValueError('CANONICAL_SOURCE_CHANGED_BEFORE_FORWARD')
        preflight=remote("import json,os,socket,time;from pathlib import Path;r=Path("+repr(TARGET)+
            ");assert socket.gethostname()=='xielab3';r.mkdir(parents=True,exist_ok=True);(r/'selected/canonical').mkdir(parents=True,exist_ok=True);"+
            "f=(r/'forward-claim.json').open('x');json.dump(dict(created_unix=time.time(),source_host='xielab2'),f);f.close();"+
            "s=os.statvfs(r);assert s.f_bavail*s.f_frsize>1024**3;print(json.dumps(dict(host=socket.gethostname(),available_bytes=s.f_bavail*s.f_frsize)))")
        atomic(SOURCE/'canonical-forward-start.json',dict(pid=os.getpid(),checked_unix=time.time(),target=TARGET,preflight=preflight))
        transport='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10'
        subprocess.run(['rsync','-a','--partial','--append-verify','--no-owner','--no-group','-e',transport,
            str(SOURCE/'selected/canonical/train.parquet'),str(SOURCE/'selected/canonical/eligible-question-pool.json'),
            PEER+':'+TARGET+'/selected/canonical/'],check=True,timeout=180)
        code="""import hashlib,json,time
from pathlib import Path
root=Path(TARGET_VALUE);source=SOURCE_PROOF
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
for row in source['files']:
 path=root/'selected'/row['path']
 assert path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
proof=dict(source,status='HKUST3_SELECTED_CANONICAL_INPUTS_FRESH_SHA_VERIFIED',route_stage='HKUST3_SELECTED_INPUTS_ONLY',
 compressed_archive_copied=False,historical_rollouts_copied=False,checked_unix=time.time())
with (root/'complete.json').open('x') as f:json.dump(proof,f,indent=2)
print(json.dumps(proof))
""".replace('TARGET_VALUE',repr(TARGET)).replace('SOURCE_PROOF',repr(proof))
        accepted=remote(code)
        atomic(SOURCE/'hkust3-canonical-verified.json',accepted)
        print(json.dumps(accepted),flush=True)


if __name__=='__main__':main()
