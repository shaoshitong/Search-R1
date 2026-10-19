"""Run on mgmt-host: retained18 -> bounded RAM -> an existing private HF repo.

Preparation alone never uploads. --execute is required, and the token is read
only from the existing task-private file. No model bytes are staged on disk.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import selectors
import shlex
import subprocess
import time


ALLOWED = {
    'added_tokens.json', 'complete.json', 'config.json',
    'generation_config.json', 'merges.txt', 'model.safetensors.index.json',
    'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json',
    'vocab.json', 'model-00001-of-00003.safetensors',
    'model-00002-of-00003.safetensors', 'model-00003-of-00003.safetensors',
}
MAX_READ = 8 << 20


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def publish(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def ssh_source(cfg, command):
    # Authentication stays on management host. Never copy an SSH identity.
    if cfg['targets']['h100-10'] != cfg['gateway_ip']:
        raise ValueError('Expected retained18 to be the existing gateway')
    args = ['env', '-u', 'LD_LIBRARY_PATH', 'ssh', '-T', '-i',
            cfg['management_identity_file'], '-o', 'IdentitiesOnly=yes',
            '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=8',
            '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=10',
            '-o', 'ServerAliveCountMax=2']
    keys = cfg.get('trusted_target_host_keys', {}).get('h100-10')
    if keys:
        args += ['-o', 'KnownHostsCommand=' + shlex.join(['/bin/echo', '\n'.join(keys)])]
    return args + [cfg['user'] + '@' + cfg['gateway_ip'], command]


def inspect_source(cfg, source_root, records, timeout):
    # Full source SHA, size, and stable metadata are fresh; output is only JSON.
    spec = [(str(Path(source_root) / r['name']), r['bytes'], r['sha256']) for r in records]
    code = '''import pathlib,json,hashlib
out=[]
for name,size,expected in SPEC:
 p=pathlib.Path(name);a=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 z=p.stat()
 assert (a.st_size,a.st_mtime_ns,a.st_ino)==(z.st_size,z.st_mtime_ns,z.st_ino)
 assert a.st_size==size and h.hexdigest()==expected
 out.append(dict(path=name,bytes=size,sha256=expected,mtime_ns=z.st_mtime_ns,inode=z.st_ino))
print(json.dumps(out))
'''.replace('SPEC', repr(spec))
    p = subprocess.run(ssh_source(cfg, 'python3 -c ' + shlex.quote(code)),
                       capture_output=True, timeout=timeout, check=False)
    if p.returncode:
        raise RuntimeError('Source manifest validation failed')
    return json.loads(p.stdout)


class RemoteFile(io.BufferedIOBase):
    """Seekable bounded SSH stream with a deadline for each source read."""

    def __init__(self, cfg, record, timeout=60, progress=None, command_factory=None):
        super().__init__()
        self.cfg, self.record, self.timeout = cfg, record, timeout
        self.position, self.process = 0, None
        self.progress, self.command_factory = progress, command_factory

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def _stop(self):
        if self.process is not None:
            p, self.process = self.process, None
            p.stdout.close()
            if p.poll() is None:
                p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=3)

    def seek(self, offset, whence=0):
        base = 0 if whence == 0 else self.position if whence == 1 else self.record['bytes'] if whence == 2 else None
        if base is None or not isinstance(offset, int):
            raise ValueError('Invalid seek')
        target = base + offset
        if not 0 <= target <= self.record['bytes']:
            raise ValueError('Seek outside source')
        if target != self.position:
            self._stop()
            self.position = target
        return target

    def _start(self):
        r = self.record
        code = '''import pathlib,sys
p=pathlib.Path(PATH);a=p.stat()
assert (a.st_size,a.st_mtime_ns,a.st_ino)==(SIZE,MTIME,INODE)
with p.open('rb') as f:
 f.seek(OFFSET)
 for b in iter(lambda:f.read(1048576),b''):sys.stdout.buffer.write(b)
sys.stdout.buffer.flush();z=p.stat()
assert (z.st_size,z.st_mtime_ns,z.st_ino)==(SIZE,MTIME,INODE)
'''
        for k, v in {'PATH': r['path'], 'SIZE': r['bytes'], 'MTIME': r['mtime_ns'],
                     'INODE': r['inode'], 'OFFSET': self.position}.items():
            code = code.replace(k, repr(v))
        args = self.command_factory(self.position) if self.command_factory else ssh_source(
            self.cfg, 'python3 -c ' + shlex.quote(code))
        self.process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self, size=-1):
        remaining = self.record['bytes'] - self.position
        if not remaining:
            return b''
        if size is None or size < 0:
            if remaining > MAX_READ:
                raise ValueError('Unbounded model read forbidden')
            size = remaining
        size = min(size, remaining, MAX_READ)
        if size == 0:
            return b''
        if self.process is None:
            self._start()
        result = bytearray()
        deadline = time.monotonic() + self.timeout
        with selectors.DefaultSelector() as sel:
            sel.register(self.process.stdout, selectors.EVENT_READ)
            while len(result) < size:
                left = deadline - time.monotonic()
                if left <= 0 or not sel.select(left):
                    self._stop()
                    raise TimeoutError('Bounded source read deadline')
                block = os.read(self.process.stdout.fileno(), min(65536, size - len(result)))
                if not block:
                    self._stop()
                    raise IOError('Short source stream')
                result.extend(block)
        self.position += len(result)
        if self.progress:
            self.progress(self.position)
        if self.position == self.record['bytes']:
            try:
                rc = self.process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self._stop()
                raise TimeoutError('Source final validation deadline')
            if rc:
                self._stop()
                raise IOError('Source changed or stream failed')
        return bytes(result)

    def close(self):
        self._stop()
        super().close()


def file_value(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def metadata_match(meta, record, git_blob=None):
    if file_value(meta, 'size') != record['bytes']:
        return False
    lfs = file_value(meta, 'lfs')
    if lfs:
        return file_value(lfs, 'sha256') == record['sha256']
    return git_blob is not None and file_value(meta, 'blob_id') == git_blob


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--task-dir', required=True)
    parser.add_argument('--route-json', required=True)
    parser.add_argument('--token-file', required=True)
    parser.add_argument('--read-timeout', type=float, default=60)
    parser.add_argument('--source-hash-timeout', type=float, default=900)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    records = plan['files']
    names = [r['name'] for r in records]
    if set(names) != ALLOWED or len(names) != len(ALLOWED):
        raise ValueError('Require exactly the thirteen audited Easy HF files')
    for r in records:
        if len(r['sha256']) != 64 or any(c not in '0123456789abcdef' for c in r['sha256']) or r['bytes'] < 0:
            raise ValueError('Invalid file binding')
    if plan['source_root'] != '/mnt/data/shaoshitong/grounding-eval-paper-strict-v1-20261019/assets/easy-selection-iter200':
        raise ValueError('Only the verified retained18 Easy backup is allowed')
    if plan['repo'] != 'Sutoonq/TMP_MODEL' or not plan['prefix'].startswith('grounding-eval-20261019/'):
        raise ValueError('Require authorized repository and dedicated task prefix')
    if any(x in plan['prefix'].split('/') for x in ('..', '')) or plan['prefix'].startswith('/'):
        raise ValueError('Invalid destination prefix')
    if not args.execute:
        print(json.dumps(dict(state='PREPARED_NOT_UPLOADED', files=names,
                              bytes=sum(r['bytes'] for r in records), plan_sha256=digest(plan))))
        return
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    from huggingface_hub import HfApi, CommitOperationAdd
    from huggingface_hub.lfs import UploadInfo
    root = Path(args.task_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'upload.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    token_path = Path(args.token_file)
    if token_path.stat().st_mode & 0o077:
        raise ValueError('Task token permissions must be private')
    token = token_path.read_text().strip()
    api = HfApi(token=token, endpoint='https://huggingface.co')
    if not api.model_info(plan['repo']).private:
        raise RuntimeError('Private repository required')
    cfg = json.loads(Path(args.route_json).read_text())
    fresh = inspect_source(cfg, plan['source_root'], records, args.source_hash_timeout)
    status = dict(state='UPLOADING', pid=os.getpid(), started_at=time.time(),
                  plan_sha256=digest(plan), source_manifest=fresh, receipts=[],
                  verified_bytes=0,
                  model_disk_staging=False, mac_in_data_route=False)
    publish(root / 'status.json', status)
    try:
        for record, bound in zip(records, fresh):
            path = plan['prefix'].rstrip('/') + '/' + record['name']
            def progress(position):
                if position == bound['bytes'] or position // (64 << 20) != status.get('progress_bucket', -1):
                    status.update(current_file=record['name'], source_read_position_bytes=position,
                                  progress_bucket=position // (64 << 20), updated_at=time.time())
                    publish(root / 'status.json', status)
            with RemoteFile(cfg, bound, args.read_timeout, progress) as stream:
                sample = stream.read(min(512, bound['bytes']))
                stream.seek(0)
                # Git's identity is required for non-LFS metadata, not a SHA256
                # claim on a repository Git blob. Largest metadata file is12MB.
                git_blob = None
                if bound['bytes'] < (32 << 20):
                    h = hashlib.sha1(('blob %d\0' % bound['bytes']).encode())
                    while stream.tell() < bound['bytes']:
                        h.update(stream.read(min(MAX_READ, bound['bytes'] - stream.tell())))
                    git_blob = h.hexdigest()
                    stream.seek(0)
                existing = api.get_paths_info(plan['repo'], [path], repo_type='model')
                if existing and not metadata_match(existing[0], bound, git_blob):
                    raise RuntimeError('Existing destination conflicts with source')
                if existing:
                    revision = api.model_info(plan['repo']).sha
                    skipped = True
                else:
                    operation = CommitOperationAdd(path_in_repo=path, path_or_fileobj=sample)
                    operation.path_or_fileobj = stream
                    operation.upload_info = UploadInfo(size=bound['bytes'], sha256=bytes.fromhex(bound['sha256']), sample=sample)
                    revision = api.create_commit(plan['repo'], operations=[operation],
                                commit_message='Preserve Easy Selection iter200 for grounding evaluation', repo_type='model').oid
                    skipped = False
                meta = api.get_paths_info(plan['repo'], [path], revision=revision, repo_type='model')
                if len(meta) != 1 or not metadata_match(meta[0], bound, git_blob):
                    raise RuntimeError('HF full size/LFS SHA or Git blob verification failed')
                status['receipts'].append(dict(name=record['name'], path=path, bytes=bound['bytes'],
                    sha256=bound['sha256'], git_blob_id=git_blob, commit=revision,
                    already_present=skipped, verified_at=time.time()))
                status.update(updated_at=time.time(), verified_bytes=sum(
                    r['bytes'] for r in status['receipts']))
                publish(root / 'status.json', status)
        # Fresh metadata for all files at one final immutable revision.
        revision = api.model_info(plan['repo']).sha
        paths = [plan['prefix'] + '/' + r['name'] for r in records]
        final = {x.path: x for x in api.get_paths_info(plan['repo'], paths, revision=revision, repo_type='model')}
        for r, receipt in zip(records, status['receipts']):
            if not metadata_match(final.get(receipt['path']), r, receipt['git_blob_id']):
                raise RuntimeError('Final pinned HF snapshot verification failed')
        # Preserve the source binding even when matching files were skipped or
        # an HTTP client opened and abandoned a source stream before EOF.
        check = 'import pathlib,json\nrows=' + repr(fresh) + '\nfor r in rows:\n s=pathlib.Path(r["path"]).stat()\n assert (s.st_size,s.st_mtime_ns,s.st_ino)==(r["bytes"],r["mtime_ns"],r["inode"])\nprint("SOURCE_METADATA_STABLE")'
        stable = subprocess.run(ssh_source(cfg, 'python3 -c ' + shlex.quote(check)),
                                capture_output=True, timeout=args.read_timeout)
        if stable.returncode or stable.stdout.strip() != b'SOURCE_METADATA_STABLE':
            raise RuntimeError('Source metadata changed during upload')
        status.update(state='HF_FILES_VERIFIED', completed_at=time.time(), final_revision=revision)
        publish(root / 'status.json', status)
        print(json.dumps(dict(state=status['state'], files=len(records), revision=revision)))
    except Exception as exc:
        status.update(state='FAILED_SOURCE_RETAINED', error_type=type(exc).__name__, updated_at=time.time())
        publish(root / 'status.json', status)
        # Do not print HTTP exception bodies, request headers, signed URLs,
        # token values, or a traceback from a third-party upload client.
        print(json.dumps(dict(state=status['state'], error_type=type(exc).__name__)))
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps(dict(state='FAILED_BEFORE_UPLOAD_OR_INITIALIZATION',
                              error_type=type(exc).__name__)))
        raise SystemExit(1)
