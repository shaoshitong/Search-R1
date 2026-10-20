"""H100 sealed raw snapshot -> management bounded RAM stream -> private HF.

Run on mgmt-host, never Mac. Token only encrypted stdin/process RAM. Source
archives stay on our NAS10 directory. No sampling or training process changes.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
REPO = 'Sutoonq/Qwen25-7B-Probe-H100-Transfer-20261020'
ROUTE = Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json')
SOURCE_PARENT = Path('/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021')


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.%d.tmp'%os.getpid())
    with tmp.open('w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write(chr(10)); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def source_args(command, cfg=None):
    cfg = cfg if cfg is not None else json.loads(ROUTE.read_text())
    target = 'h100-10'
    keys = cfg.get('trusted_target_host_keys', {}).get(target)
    prefix = ['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],
        '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
        '-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1',
        '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3']
    result = list(prefix)
    if cfg['targets'][target] != cfg['gateway_ip']:
        result += ['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',cfg['user']+'@'+cfg['gateway_ip']])]
    # Some existing route targets use management's already populated known_hosts
    # instead of the optional inline public-key map. Strict checking stays on;
    # an absent/unrecognized host key is an SSH failure, never auto-accepted.
    if keys:
        result += ['-o','KnownHostsCommand='+shlex.join(['/bin/echo',chr(10).join(keys)])]
    return result+[cfg['user']+'@'+cfg['targets'][target],command]


def validate_snapshot(value, source):
    source = Path(source)
    if (not source.is_absolute() or source.parent != SOURCE_PARENT or
            not re.fullmatch('[A-Za-z0-9_-]{8,80}', source.name) or
            value['status'] != 'SEALED_CANONICAL_NEWSEED_PREFIX20_G5_SNAPSHOT_EXPORT_COMPLETE' or
            value['source_node'] != 'h100-10' or value['revision'] != REVISION or
            value['snapshot_id'] != source.name or value['actor_updates'] != 0 or
            value['source_run_id'] != '70188b8e50fd4b45bc420ce58014f46e' or value['seed'] != 20261021 or
            value['hidden_exported'] is not False or not value['all_archives_fresh_SHA_and_member_SHA_verified'] or
            not value['fresh_all_source_files_SHA256']):
        raise ValueError('Sealed own-NAS10 official7B raw-only source snapshot required')
    names = set(); paths = set()
    for group in value['groups']:
        name = group['group']; archive = Path(group['archive'])
        if (not re.fullmatch('[A-Za-z0-9_-]+', name) or name in names or
                archive.parent not in (source,source/'archives') or not (archive.suffix == '.tar' or archive.suffixes[-2:] == ['.tar','.gz']) or
                not re.fullmatch('[0-9a-f]{64}', group['archive_sha256']) or
                type(group['archive_bytes']) is not int or group['archive_bytes'] < 1):
            raise ValueError('Distinct complete source archives required')
        names.add(name)
        for item in group['files']:
            path = Path(item['path'])
            if (path.is_absolute() or '..' in path.parts or item['path'] in paths or
                    path.suffix in ('.bin','.pt') or type(item['bytes']) is not int or item['bytes'] < 0 or
                    not re.fullmatch('[0-9a-f]{64}', item['sha256'])):
                raise ValueError('Distinct safe raw-only source members required')
            paths.add(item['path'])
        if sum(f['bytes'] for f in group['files']) != group['source_bytes']:
            raise ValueError('Group source size differs')
    if (not names or len(paths) != value['files'] or
            sum(g['source_bytes'] for g in value['groups']) != value['payload_bytes'] or
            sum(g['questions'] for g in value['groups']) != value['questions'] or
            sum(g['trajectories'] for g in value['groups']) != value['trajectories'] or
            value['trajectories'] != 5*value['questions']):
        raise ValueError('Actual source group/file/question/trajectory totals differ')
    return value


class RemoteFile(io.BufferedIOBase):
    """Seekable source with an explicit <=8MiB read/HTTP memory bound."""
    def __init__(self, record, argv=source_args):
        self.record = record; self.position = 0; self.process = None; self.argv = argv

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def stop_stream(self):
        if self.process is None: return
        self.process.stdout.close()
        if self.process.poll() is None: self.process.terminate()
        try: self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill(); self.process.wait(timeout=5)
        self.process = None

    def seek(self, offset, whence=0):
        if whence not in (0,1,2): raise ValueError('Invalid seek origin')
        position = offset if whence == 0 else (self.position+offset if whence == 1 else self.record['archive_bytes']+offset)
        if type(position) is not int or not 0 <= position <= self.record['archive_bytes']:
            raise ValueError('Source seek outside sealed archive')
        if position != self.position: self.stop_stream(); self.position = position
        return position

    def read(self, size=-1):
        remaining = self.record['archive_bytes']-self.position
        if not remaining: return b''
        if size < 0:
            if remaining > 8*1024**2: raise ValueError('Unbounded source archive read forbidden')
            size = remaining
        size = min(size, remaining, 8*1024**2)
        if not size: return b''
        if self.process is None:
            code = ('import pathlib,sys; p=pathlib.Path(%r); assert p.stat().st_size==%d; '
                'f=p.open("rb"); f.seek(%d)\n'
                'for block in iter(lambda:f.read(1048576),b""):sys.stdout.buffer.write(block)\n') % (
                    self.record['archive'], self.record['archive_bytes'], self.position)
            self.process = subprocess.Popen(self.argv('python3 -c '+shlex.quote(code)),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        data = self.process.stdout.read(size)
        if len(data) != size: raise IOError('Short sealed source stream')
        self.position += len(data)
        return data

    def close(self):
        self.stop_stream(); super().close()


def remote_snapshot(source):
    code = 'import pathlib,sys; p=pathlib.Path(%r); assert p.stat().st_size<=16*1024**2; sys.stdout.buffer.write(p.read_bytes())' % str(Path(source)/'complete.json')
    result = subprocess.run(source_args('python3 -c '+shlex.quote(code)), stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, timeout=45)
    if result.returncode: raise RuntimeError('Source snapshot read rejected')
    return validate_snapshot(json.loads(result.stdout), source)


def run(base, source, token):
    if (not str(base).startswith('/home/lichen/qwen7b-newseed-to-hpc3-20261021/') or
            not token or not Path('/home/lichen').is_dir()):
        raise ValueError('Run only on management host with RAM credentials')
    base.mkdir(parents=True, exist_ok=True)
    with (base/'uploader.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX|fcntl.LOCK_NB)
        from huggingface_hub import HfApi, CommitOperationAdd, configure_http_backend
        from huggingface_hub.lfs import UploadInfo
        import requests
        class BoundedSession(requests.Session):
            def request(self, method, url, **kwargs):
                if kwargs.get('timeout') is None: kwargs['timeout'] = (15,60)
                return super().request(method, url, **kwargs)
        configure_http_backend(backend_factory=BoundedSession)
        api = HfApi(token=token)
        info = api.repo_info(REPO, repo_type='dataset', timeout=30)
        if not info.private or info.id.lower() != REPO.lower(): raise ValueError('Exact authorized private repository required')
        plan = remote_snapshot(source)
        old = base/'source-plan.json'
        if old.exists() and json.loads(old.read_text()) != plan: raise ValueError('Immutable source snapshot changed')
        if not old.exists(): atomic(old, plan)
        prefix = 'peer-newseed-to-hpc3-20261021/'+plan['snapshot_id']
        atomic(base/'controller.json', dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            source=str(source),repo=REPO,repo_type='dataset',private=True,prefix=prefix,
            token_saved=False,source_staging_only=True,management_data_disk_staging=False,Mac_data_transit=False,epoch=time.time()))
        accepted = []
        for group in plan['groups']:
            done = base/'uploaded'/(group['group']+'.json')
            archive_path = prefix+'/archives/'+Path(group['archive']).name
            # Cached receipts alone do not prove current HF content; always inspect.
            existing = api.get_paths_info(REPO, paths=[archive_path], repo_type='dataset')
            commit = info.sha
            if existing:
                item = existing[0]; lfs = item.lfs
                oid = lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
                if item.size != group['archive_bytes'] or oid != group['archive_sha256']:
                    raise ValueError('Existing immutable private HF archive differs')
            else:
                for attempt in range(1,9):
                    stream = RemoteFile(group)
                    try:
                        sample = stream.read(min(512, group['archive_bytes'])); stream.seek(0)
                        operation = CommitOperationAdd(path_in_repo=archive_path, path_or_fileobj=sample)
                        operation.path_or_fileobj = stream
                        operation.upload_info = UploadInfo(size=group['archive_bytes'],sha256=bytes.fromhex(group['archive_sha256']),sample=sample)
                        result = api.create_commit(repo_id=REPO, repo_type='dataset', operations=[operation],
                            commit_message='Accept sealed independent 7B raw trajectories '+group['group'])
                        commit = result.oid
                        break
                    except Exception as error:
                        atomic(base/'retry.json',dict(group=group['group'],attempt=attempt,error_type=type(error).__name__,epoch=time.time()))
                        if attempt == 8: raise
                        time.sleep(min(20,2*attempt))
                    finally: stream.close()
                item = api.get_paths_info(REPO, paths=[archive_path], repo_type='dataset')[0]
                lfs = item.lfs; oid = lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
                if item.size != group['archive_bytes'] or oid != group['archive_sha256']:
                    raise ValueError('Official HF LFS size/SHA verification failed')
            receipt = dict(group, status='PEER_SOURCE_PRIVATE_HF_ARCHIVE_SHA_ACCEPTED',
                repo=REPO,repo_type='dataset',private=True,hf_archive_path=archive_path,hf_archive_commit=commit,
                source_revision=REVISION,snapshot_id=plan['snapshot_id'],epoch=time.time())
            manifest_path = prefix+'/receipts/'+group['group']+'.json'
            m = api.upload_file(path_or_fileobj=json.dumps(receipt,sort_keys=True).encode(),path_in_repo=manifest_path,
                repo_id=REPO,repo_type='dataset',commit_message='Source payload SHA manifest '+group['group'])
            receipt.update(hf_receipt_path=manifest_path,hf_receipt_commit=m.oid)
            atomic(done,receipt); accepted.append(receipt)
            atomic(base/'progress.json',dict(groups=len(accepted),total_groups=len(plan['groups']),
                HF_archive_bytes=sum(r['archive_bytes'] for r in accepted),total_archive_bytes=sum(g['archive_bytes'] for g in plan['groups']),
                HPC3_complete=False,epoch=time.time()))
        catalog = dict(status='ALL_PEER_NEWSEED_SOURCE_PRIVATE_HF_ARCHIVES_SHA_ACCEPTED_NOT_HPC3_COMPLETE',
            snapshot_id=plan['snapshot_id'],repo=REPO,repo_type='dataset',private=True,prefix=prefix,
            source_snapshot=plan,groups=accepted,Mac_data_transit=False,epoch=time.time())
        result = api.upload_file(path_or_fileobj=json.dumps(catalog,sort_keys=True).encode(),
            path_in_repo=prefix+'/catalog.json',repo_id=REPO,repo_type='dataset',commit_message='Sealed peer data migration catalog')
        catalog.update(catalog_path=prefix+'/catalog.json',catalog_commit=result.oid)
        atomic(base/'complete.json',catalog)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    secret=json.loads(sys.stdin.readline())
    try: run(a.base,a.source,secret['hf_token'])
    except BaseException as error:
        atomic(a.base/'failure.json',dict(error_type=type(error).__name__,epoch=time.time()))
        raise SystemExit(1)


if __name__=='__main__':main()
