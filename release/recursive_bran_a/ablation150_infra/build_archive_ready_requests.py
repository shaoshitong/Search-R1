"""Build tiny-object server-read and exact same-cluster readiness finalization."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['server', 'finalize'], required=True)
    parser.add_argument('--pending', type=Path, required=True)
    parser.add_argument('--server', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pending = json.loads(args.pending.read_text())
    assert pending['protocol'] == 'ablation150_existing_same_cluster_storage_pending_server_v1'
    assert pending['probe']['bytes'] == 64 and pending['probe']['write_rc'] == 0
    assert pending['mount']['source'] == 'h100-10:/mnt/data'
    if args.phase == 'server':
        code = '''import hashlib,json,os,time
from pathlib import Path
r=PENDING;p=Path(r['probe']['server_path'])
root=Path('/mnt/data/shaoshitong/recursive-ablation150-archive-20261014')
assert p.is_relative_to(root) and not p.is_symlink() and p.name.startswith('.readiness-probe-')
data=p.read_bytes();sha=hashlib.sha256(data).hexdigest()
assert len(data)==64 and sha==r['probe']['sha256']
print(json.dumps(dict(time=time.time(),server_hostname=os.uname().nodename,server_path=str(p),
 server_read_rc=0,server_sha256=sha,bytes=len(data),GPU_action='NONE')),flush=True)
'''.replace('PENDING', repr(pending))
    else:
        assert args.server is not None
        server = json.loads(args.server.read_text())
        transport = json.loads(Path(str(args.server)+'.record.json').read_text())
        assert transport['exit_code'] == 0 and json.loads(transport['stdout']) == server
        assert server['server_read_rc'] == 0 and server['bytes'] == 64
        assert server['server_sha256'] == pending['probe']['sha256']
        code = '''import hashlib,json,os,time
from pathlib import Path
r=PENDING;server=SERVER
probe=Path(r['probe']['path']);data=probe.read_bytes();sha=hashlib.sha256(data).hexdigest()
assert len(data)==64 and sha==r['probe']['sha256']==server['server_sha256']
assert server['server_path']==r['probe']['server_path']
mount=Path(r['mount']['mountpoint']);fs=os.statvfs(mount)
assert fs.f_bavail*fs.f_frsize>2100000000000
identities=[]
for line in Path('/proc/self/mountinfo').read_text().splitlines():
 left,right=line.split(' - ',1);fields=left.split();filesystem=right.split()
 if fields[4]==str(mount) and filesystem[0]=='nfs4':identities.append(filesystem[1])
assert identities==['h100-10:/mnt/data']
ready=dict(r,protocol='ablation150_existing_same_cluster_storage_ready_v1',ready=True,
 verified_time=time.time(),actual_free_bytes=fs.f_bavail*fs.f_frsize)
ready['probe']=dict(r['probe'],local_sha256=sha,server_sha256=server['server_sha256'],server_read_rc=0,
 server_read_time=server['time'],server_hostname=server['server_hostname'])
path=Path(r['study_root'])/'analysis/archive-storage-ready.json'
if path.exists():
 old=json.loads(path.read_text());assert old['probe']==ready['probe'];ready=old
else:
 with path.open('x') as stream:
  stream.write(json.dumps(ready,indent=2));stream.flush();os.fsync(stream.fileno())
print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),receipt=ready)),flush=True)
'''.replace('PENDING', repr(pending)).replace('SERVER', repr(server))
    args.output.write_text(code)
    print(json.dumps(dict(phase=args.phase, output=str(args.output), bytes=len(code.encode()))))


if __name__ == '__main__':
    main()
