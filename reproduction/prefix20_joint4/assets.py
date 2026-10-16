"""Remote-only pinned HF probe download and full-member SHA verification.

Token is read from stdin and retained only in process RAM. No signed URL,
authorization header, or credential is included in operational receipts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import socket
import sys
import tarfile
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


class PartsReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths); self.current = None
    def readable(self): return True
    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None: return 0
                self.current = path.open('rb')
            count = self.current.readinto(buffer)
            if count: return count
            self.current.close(); self.current = None
    def close(self):
        if self.current is not None: self.current.close()
        super().close()


def unpack(root, archive, manifest):
    destination = root/'probe'
    destination.mkdir(exist_ok=True)
    expected = {r['path']:r for r in manifest['files']}
    seen = set(); archive_hash = hashlib.sha256()
    paths = [root/r['name'] for r in archive['parts']]
    for path in paths:
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024), b''): archive_hash.update(block)
    if archive_hash.hexdigest() != archive['concatenated_sha256']:
        raise ValueError('CONCATENATED_ARCHIVE_SHA_MISMATCH')
    with io.BufferedReader(PartsReader(paths)) as stream, tarfile.open(fileobj=stream,mode='r|gz') as tar:
        for member in tar:
            name = str(PurePosixPath(member.name))
            if name in ('', '.') and member.isdir(): continue
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
                raise ValueError('UNSAFE_ARCHIVE_MEMBER')
            if member.isdir(): continue
            if not member.isfile() or name in seen or name not in set(expected)|{'MANIFEST.json'}:
                raise ValueError('UNEXPECTED_DUPLICATE_OR_NONREGULAR_ARCHIVE_MEMBER')
            seen.add(name); target = destination/name
            target.parent.mkdir(parents=True,exist_ok=True)
            if any(p.is_symlink() for p in [destination,target,*target.parents]):
                raise ValueError('SYMLINK_IN_ASSET_TARGET')
            data = tar.extractfile(member)
            temporary = target.with_name(target.name+'.extract-part')
            if temporary.exists(): raise ValueError('UNCOMMITTED_EXTRACT_REQUIRES_INSPECTION')
            h=hashlib.sha256(); count=0
            with temporary.open('xb') as f:
                for block in iter(lambda:data.read(1024*1024),b''):
                    f.write(block);h.update(block);count+=len(block)
                f.flush();os.fsync(f.fileno())
            if name in expected:
                if count!=expected[name]['bytes'] or h.hexdigest()!=expected[name]['sha256']:
                    raise ValueError('ARCHIVE_MEMBER_SHA_MISMATCH')
            elif json.loads(temporary.read_text()) != manifest:
                raise ValueError('EMBEDDED_MANIFEST_DIFFERS')
            if target.exists():
                if target.stat().st_size!=count or sha(target)!=h.hexdigest():
                    raise ValueError('EXISTING_ASSET_DIFFERS')
                temporary.unlink()
            else:os.replace(temporary,target)
    if set(expected)-seen:raise ValueError('MISSING_ARCHIVE_MEMBERS')
    # Fresh full reads after extraction, independent from extraction streams.
    files=[]
    for name,r in sorted(expected.items()):
        path=destination/name
        if path.stat().st_size!=r['bytes'] or sha(path)!=r['sha256']:
            raise ValueError('FRESH_EXTRACTED_SHA_MISMATCH')
        files.append(dict(r))
    proof=dict(status='PROBE_ALL_MEMBERS_FRESH_SHA_VERIFIED',snapshot_epoch=68,
               archive_sha256=archive_hash.hexdigest(),files=files,checked_unix=time.time())
    atomic(root/'probe-verified.json',proof)
    return proof


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--plan',required=True);parser.add_argument('--proxy',required=True)
    args=parser.parse_args();root=Path(args.root).resolve()
    if socket.gethostname() not in ('xielab2','xielab3') or not str(root).startswith(('/data/shaoshitong/','/backup/shaoshitong/')):
        raise ValueError('REMOTE_TASK_STORAGE_REQUIRED')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'assets.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=json.loads(Path(args.plan).read_text()); token=sys.stdin.readline().strip()
        if not token:raise ValueError('RAM_TOKEN_REQUIRED')
        import requests
        session=requests.Session();session.trust_env=False
        session.proxies={'http':args.proxy,'https':args.proxy}
        session.headers['Authorization']='Bearer '+token
        api='https://huggingface.co/api/datasets/'+plan['repo']
        parts=plan['archive']['parts']
        minimum=2*sum(p['bytes'] for p in parts)+2*1024**3
        fs=os.statvfs(root)
        if fs.f_bavail*fs.f_frsize<minimum:raise ValueError('REMOTE_STAGING_SPACE_INSUFFICIENT')
        probe=root/'write-probe'
        with probe.open('xb') as f:f.write(b'probe');f.flush();os.fsync(f.fileno())
        probe.unlink()
        for attempt in range(8):
            try:
                info=session.post(api+'/paths-info/'+plan['revision'],json={'paths':[p['name'] for p in parts]},timeout=(10,30))
                info.raise_for_status();inventory={r['path']:r for r in info.json()}
                break
            except requests.RequestException as error:
                status=getattr(getattr(error,'response',None),'status_code',None)
                atomic(root/'inventory-error.json',dict(attempt=attempt+1,error_type=type(error).__name__,
                    http_status=status,observed_unix=time.time()))
                if status in (401,403,404):raise RuntimeError('HF_ACCESS_REJECTED') from None
                if attempt==7:raise RuntimeError('BOUNDED_INVENTORY_RETRIES_EXHAUSTED') from None
                time.sleep(min(10,2+attempt))
        for part in parts:
            entry=inventory[part['name']]
            if entry['size']!=part['bytes'] or entry['lfs']['oid']!=part['sha256']:
                raise ValueError('OFFICIAL_HF_LFS_SHA_MISMATCH')
        atomic(root/'hf-lfs-verified.json',dict(repo=plan['repo'],revision=plan['revision'],parts=parts,checked_unix=time.time()))
        atomic(root/'controller.json',dict(pid=os.getpid(),host=socket.gethostname(),started_unix=time.time(),token_persisted=False))
        def download(part):
            # Each thread owns its connection pool and unique file lock target.
            client=requests.Session();client.trust_env=False;client.proxies=session.proxies.copy()
            client.headers['Authorization']='Bearer '+token
            target=root/part['name'];temporary=root/(part['name']+'.'+part['sha256']+'.part')
            progress=root/(part['name']+'.progress.json')
            if target.exists():
                if target.stat().st_size!=part['bytes'] or sha(target)!=part['sha256']:
                    raise ValueError('EXISTING_DOWNLOADED_PART_DIFFERS')
                return
            started=time.time()
            for attempt in range(8):
                offset=temporary.stat().st_size if temporary.exists() else 0
                if offset>part['bytes']:raise ValueError('PART_TOO_LARGE')
                try:
                    if offset<part['bytes']:
                        url='https://huggingface.co/datasets/'+plan['repo']+'/resolve/'+plan['revision']+'/'+part['name']
                        with client.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(10,30)) as response:
                            response.raise_for_status()
                            if offset and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                                raise ValueError('RESUME_RANGE_NOT_HONORED')
                            last=0
                            with temporary.open('ab') as f:
                                for block in response.iter_content(1024*1024):
                                    if not block:continue
                                    if offset+len(block)>part['bytes']:raise ValueError('DOWNLOAD_OVERSIZED')
                                    f.write(block);offset+=len(block)
                                    if time.time()-last>=2:
                                        f.flush();atomic(progress,dict(bytes=offset,total_bytes=part['bytes'],observed_unix=time.time(),started_unix=started,retry=attempt,stage='DOWNLOADING'));last=time.time()
                                f.flush();os.fsync(f.fileno())
                    if offset!=part['bytes']:raise ValueError('DOWNLOAD_SHORT_READ')
                    if sha(temporary)!=part['sha256']:raise ValueError('DOWNLOADED_PART_SHA_MISMATCH')
                    os.replace(temporary,target)
                    atomic(progress,dict(bytes=offset,total_bytes=part['bytes'],observed_unix=time.time(),started_unix=started,stage='FULL_SHA_VERIFIED'))
                    return
                except requests.RequestException as error:
                    status=getattr(getattr(error,'response',None),'status_code',None)
                    atomic(root/(part['name']+'.error.json'),dict(attempt=attempt+1,error_type=type(error).__name__,http_status=status,observed_unix=time.time()))
                    if status in (401,403,404):raise RuntimeError('HF_ACCESS_REJECTED') from None
                    time.sleep(min(10,2+attempt))
            raise RuntimeError('BOUNDED_DOWNLOAD_RETRIES_EXHAUSTED')
        with ThreadPoolExecutor(max_workers=2) as workers:list(workers.map(download,parts))
        proof=unpack(root,plan['archive'],plan['member_manifest'])
        atomic(root/'complete.json',dict(proof,repo=plan['repo'],revision=plan['revision'],route_stage='HKUST2_DOWNLOAD_ONLY'))


if __name__=='__main__':
    try:main()
    except Exception as error:
        # Do not allow requests exceptions to expose authenticated redirect URLs.
        print(json.dumps(dict(status='ASSET_COMPONENT_FAILED',error_type=type(error).__name__)),file=sys.stderr)
        sys.exit(1)
