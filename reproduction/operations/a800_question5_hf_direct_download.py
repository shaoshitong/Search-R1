"""CPU-only HF CDN -> A800 download of eight pinned dataset archive parts.

Signed URLs arrive only on stdin and stay in memory. No HF token is read,
stored or forwarded. The Mac stages only this small code/metadata; dataset
payloads go directly to the existing A800 host. No training job is changed.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
REPO = ROOT / 'repo'
REL = 'reproduction/operations/a800_question5_hf_direct_download.py'
PREFIX = '/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-'
HF_REPO = 'Sutoonq/Search-G1-prefix20-dataset-probe-epoch68-20261016-111a2d'
HF_REVISION = 'd794759ab9f28cf184128b227fb38f40b59fcedc'
CONCAT_SHA = '085b039051359f341bcc123a952e2ac9500fd890331dc4f7cd2e367c81d39ced'
TOTAL = 8389298946
PYTHON = '/data/search-g1/env/bin/python'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path if exclusive else path.with_name(path.name + '.partial-%d' % os.getpid())
    with temporary.open('xb') as handle:
        handle.write(canonical(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    if not exclusive:
        os.replace(temporary, path)


def manifest_value(value):
    repo = value.get('repo_id', value.get('repo'))
    revision = value.get('revision', value.get('commit'))
    if repo != HF_REPO or revision != HF_REVISION or value.get('concatenated_sha256') != CONCAT_SHA:
        raise ValueError('Require the exact pinned original dataset repository/revision/archive SHA')
    parts = sorted(value['parts'], key=lambda part: part['name'])
    if len(parts) != 8 or sum(part['bytes'] for part in parts) != TOTAL:
        raise ValueError('Require exactly eight dataset-only parts totaling8389298946 bytes')
    for i, part in enumerate(parts):
        if (part['name'] != 'dataset-through-epoch-0068.tar.gz.part%03d' % i
                or type(part['bytes']) is not int or part['bytes'] <= 0
                or re.fullmatch('[0-9a-f]{64}', part['sha256']) is None):
            raise ValueError('Invalid dataset-only archive part identity')
    return dict(schema_version=1, repo_id=repo, revision=revision,
                parts=[dict(name=p['name'], bytes=p['bytes'], sha256=p['sha256']) for p in parts],
                total_bytes=TOTAL, concatenated_sha256=CONCAT_SHA,
                model_or_probe_archives_included=False)


def signed_urls(value, manifest):
    urls = value.get('urls', value) if isinstance(value, dict) else value
    if isinstance(urls, list):
        if len(urls) != 8:
            raise ValueError('Require exactly eight in-memory signed URLs')
        urls = dict(zip([p['name'] for p in manifest['parts']], urls))
    if not isinstance(urls, dict) or set(urls) != {p['name'] for p in manifest['parts']}:
        raise ValueError('Signed URL names differ from the eight pinned parts')
    for url in urls.values():
        if not isinstance(url, str) or len(url) > 32768 or any(c in url for c in ('\n', '\r', '\0')):
            raise ValueError('Invalid in-memory signed URL')
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Only direct HTTPS signed URLs without userinfo are accepted')
    return urls


def namespace(base):
    if re.fullmatch(re.escape(PREFIX) + '[0-9a-f]{32}', str(base)) is None:
        raise ValueError('Use only the current question5 experiment base')
    return Path(base) / 'hf-direct-source'


def identity(pid):
    process = Path('/proc') / str(pid)
    try:
        fields = (process / 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=pid, birth=int(fields[19]), sid=int(fields[3]), uid=process.stat().st_uid, state=fields[0])
    except FileNotFoundError:
        return None


class CurlFailure(Exception):
    def __init__(self, code, stderr_sha256, http_status):
        self.code, self.stderr_sha256, self.http_status = code, stderr_sha256, http_status
        super().__init__('CurlFailure')


def worker(base):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = namespace(base)
    payload = json.load(sys.stdin)
    manifest = manifest_value(payload['manifest'])
    urls = signed_urls(payload['urls'], manifest)
    source = json.loads((root / 'stage.json').read_text())
    if sha(Path(__file__)) != source['worker_sha256'] or sha_bytes(canonical(manifest)) != source['manifest_sha256']:
        raise ValueError('Staged worker or pinned manifest identity differs')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Direct download must be CPU-only')
    started = time.time()
    owner = identity(os.getpid())
    atomic(root / 'worker-start.json', dict(identity=owner, started_unix=started,
        worker_sha256=source['worker_sha256'], publication=source['publication'],
        manifest_sha256=source['manifest_sha256'], cpu_only=True, actor_updates=0), exclusive=True)
    archive = root / 'archive'
    states, lock = {}, threading.Lock()
    exitcode = 1

    def update(name, **values):
        with lock:
            states.setdefault(name, {}).update(values)

    def fetch(part):
        name = part['name']
        target = archive / name
        partial = archive / (name + '.partial')
        if target.is_symlink() or partial.is_symlink():
            raise ValueError('Owned download paths may not be symlinks')
        if target.exists():
            if target.stat().st_size != part['bytes'] or sha(target) != part['sha256']:
                raise ValueError('Existing sealed file differs from the pinned original data')
            update(name, stage='EXISTING_FULL_SHA_VERIFIED', bytes=part['bytes'], sha256=part['sha256'])
            return
        offset = partial.stat().st_size if partial.exists() else 0
        if not 0 <= offset <= part['bytes']:
            raise ValueError('Owned partial file exceeds expected bytes')
        update(name, stage='DIRECT_CDN_DOWNLOADING', resume_offset=offset, expected_bytes=part['bytes'])
        if offset < part['bytes']:
            command = ['curl', '--disable', '--config', '-', '--proto', '=https', '--proto-redir', '=https',
                       '--fail', '--location', '--silent', '--show-error', '--connect-timeout', '15',
                       '--speed-time', '60', '--speed-limit', '1', '--retry', '0',
                       '--continue-at', str(offset), '--output', str(partial), '--write-out', '%{http_code}']
            # Neither argv nor the environment contains a token or signed URL.
            url = urls[name]
            curl_config = 'url = "' + url.replace('\\', '\\\\').replace('"', '\\"') + '"\n'
            beginning = time.monotonic()
            process = subprocess.run(command, input=curl_config.encode(), capture_output=True)
            status = process.stdout.decode('ascii', 'replace').strip()
            status = int(status) if re.fullmatch('[0-9]{3}', status) else None
            metadata = dict(curl_returncode=process.returncode, http_status=status,
                            curl_seconds=time.monotonic() - beginning, stderr_sha256=sha_bytes(process.stderr))
            update(name, **metadata)
            if process.returncode:
                raise CurlFailure(process.returncode, metadata['stderr_sha256'], status)
        update(name, stage='FULL_FILE_SHA_VERIFYING')
        with partial.open('rb') as handle:
            os.fsync(handle.fileno())
        actual_size, actual_sha = partial.stat().st_size, sha(partial)
        if actual_size != part['bytes'] or actual_sha != part['sha256']:
            raise ValueError('Completed direct download size or full SHA differs')
        os.replace(partial, target)
        target.chmod(0o444)
        receipt = dict(stage='ORIGINAL_DATA_PART_FULL_SHA_VERIFIED', name=name, bytes=actual_size,
                       sha256=actual_sha, verified_unix=time.time(), resumed_from_bytes=offset,
                       repo_id=HF_REPO, revision=HF_REVISION)
        atomic(target.with_name(name + '.verified.json'), receipt, exclusive=True)
        directory = os.open(archive, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        update(name, stage='SEALED_FULL_SHA_VERIFIED', bytes=actual_size, sha256=actual_sha)

    def progress(stage):
        rows, observed, durable = [], 0, 0
        for part in manifest['parts']:
            name = part['name']
            final, partial = archive / name, archive / (name + '.partial')
            count, synced = 0, 0
            path = final if final.exists() else partial
            try:
                with path.open('rb') as handle:
                    before = os.fstat(handle.fileno()).st_size
                    os.fsync(handle.fileno())
                    count = os.fstat(handle.fileno()).st_size
                    synced = before  # Later concurrent writes are not called durable.
            except FileNotFoundError:
                pass
            if count > part['bytes']:
                raise ValueError('Observed partial grew beyond the pinned file size')
            with lock:
                row = dict(states.get(name, {}))
            row.update(name=name, expected_bytes=part['bytes'], observed_bytes=count, durable_bytes=synced)
            rows.append(row)
            observed += count
            durable += synced
        atomic(root / 'progress.json', dict(stage=stage, identity=owner, observed_unix=time.time(),
            elapsed_seconds=time.time() - started, observed_bytes=observed, durable_bytes=durable,
            expected_bytes=TOTAL, parts=rows, parallel_downloads=4, cpu_only=True,
            direct_hf_cdn_to_a800=True, h100_or_mac_payload_relay=False, actor_updates=0))

    try:
        progress('DIRECT_CDN_DOWNLOAD_STARTING')
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch, part) for part in manifest['parts']]
            while not all(future.done() for future in futures):
                progress('DIRECT_CDN_DOWNLOAD_ACTIVE')
                time.sleep(5)
            errors = []
            for future in futures:
                try:
                    future.result()
                except BaseException as error:
                    row = dict(error_type=type(error).__name__)
                    if isinstance(error, CurlFailure):
                        row.update(curl_returncode=error.code, stderr_sha256=error.stderr_sha256,
                                   http_status=error.http_status)
                    errors.append(row)
            if errors:
                atomic(root / 'failure.json', dict(error_type='DIRECT_DOWNLOAD_PART_FAILED', errors=errors,
                    identity=owner, observed_unix=time.time(), partials_preserved=True), exclusive=True)
                raise RuntimeError('DIRECT_DOWNLOAD_PART_FAILED')
        progress('ALL_PARTS_SEALED_VERIFYING_CONCATENATED_SHA')
        whole = hashlib.sha256()
        total, proofs = 0, []
        for part in manifest['parts']:
            path = archive / part['name']
            if path.stat().st_size != part['bytes']:
                raise ValueError('Sealed original dataset size changed')
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(8 * 1024**2), b''):
                    whole.update(block)
                    total += len(block)
            proofs.append(dict(part))
        if total != TOTAL or whole.hexdigest() != CONCAT_SHA:
            raise ValueError('All-eight concatenated original dataset SHA differs')
        atomic(root / 'complete.json', dict(status='DIRECT_HF_ORIGINAL_DATA_COMPLETE', identity=owner,
            repo_id=HF_REPO, revision=HF_REVISION, parts=proofs, bytes=total,
            concatenated_sha256=whole.hexdigest(), manifest_sha256=source['manifest_sha256'],
            worker_sha256=source['worker_sha256'], publication=source['publication'],
            started_unix=started, finished_unix=time.time(), cpu_only=True,
            direct_hf_cdn_to_a800=True, h100_or_mac_payload_relay=False,
            signed_urls_persisted=False, actor_updates=0), exclusive=True)
        progress('DIRECT_HF_ORIGINAL_DATA_COMPLETE')
        exitcode = 0
    except BaseException as error:
        if not (root / 'failure.json').exists():
            atomic(root / 'failure.json', dict(error_type=type(error).__name__, identity=owner,
                observed_unix=time.time(), partials_preserved=True), exclusive=True)
        raise
    finally:
        urls.clear()
        atomic(root / 'exit.json', dict(exit_code=exitcode, identity=owner, finished_unix=time.time(),
            elapsed_seconds=time.time() - started, cpu_only=True, actor_updates=0), exclusive=True)


STAGE = r'''
import base64,hashlib,json,os,pathlib,re,resource,subprocess,sys,time
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
p=json.load(sys.stdin);base=pathlib.Path(p['base']);root=base/'hf-direct-source'
assert re.fullmatch('/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-[0-9a-f]{32}',str(base))
assert base.resolve(strict=True)==base and base.stat().st_uid==os.getuid()
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def once(path,value):
 with path.open('x')as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
def ident(pid):
 q=pathlib.Path('/proc')/str(pid)
 try:
  v=(q/'stat').read_text().rsplit(') ',1)[1].split();return dict(pid=pid,birth=int(v[19]),sid=int(v[3]),uid=q.stat().st_uid,state=v[0])
 except FileNotFoundError:return None
if root.exists():
 assert root.resolve(strict=True)==root and root.stat().st_uid==os.getuid()
 stage=json.loads((root/'stage.json').read_text())
 assert stage['worker_sha256']==p['worker_sha256']and stage['manifest_sha256']==p['manifest_sha256']
 assert sha(root/'worker.py')==p['worker_sha256']
 if(root/'launch-claim.json').exists():
  print(json.dumps(dict(stage='ALREADY_ONCE_CLAIMED_NO_RELAUNCH',launch=json.loads((root/'launch.json').read_text())if(root/'launch.json').exists()else None)));raise SystemExit
else:
 root.mkdir(mode=0o700);(root/'archive').mkdir(mode=0o700)
 raw=base64.b64decode(p['source']);assert hashlib.sha256(raw).hexdigest()==p['worker_sha256']
 with(root/'worker.py').open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 (root/'worker.py').chmod(0o444)
 once(root/'manifest.json',p['manifest'])
 once(root/'stage.json',dict(worker_sha256=p['worker_sha256'],manifest_sha256=p['manifest_sha256'],
  input_manifest_file_sha256=p['input_manifest_file_sha256'],publication=p['publication'],
  staged_unix=time.time(),signed_urls_persisted=False))
once(root/'launch-claim.json',dict(claimed_unix=time.time(),worker_sha256=p['worker_sha256'],manifest_sha256=p['manifest_sha256']))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
 PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
with(root/'worker.log').open('xb')as log:
 child=subprocess.Popen(['/usr/bin/nice','-n','19',p['python'],str(root/'worker.py'),'--worker','--base',str(base)],
  stdin=subprocess.PIPE,stdout=log,stderr=log,env=env,cwd=str(root),start_new_session=True)
 packet=json.dumps(dict(manifest=p['manifest'],urls=p['urls'])).encode()
 child.stdin.write(packet);child.stdin.close()
result=dict(stage='ONE_DIRECT_HF_CPU_WORKER_LAUNCHED',identity=ident(child.pid),launch_unix=time.time(),
 base=str(base),namespace=str(root),worker_sha256=p['worker_sha256'],manifest_sha256=p['manifest_sha256'],
 cpu_only=True,direct_hf_cdn_to_a800=True,h100_or_mac_payload_relay=False,signed_urls_persisted=False)
once(root/'launch.json',result);print(json.dumps(result))
'''


OBSERVE = r'''
import hashlib,json,os,pathlib,re,sys,time
p=json.load(sys.stdin);base=pathlib.Path(p['base']);root=base/'hf-direct-source'
assert re.fullmatch('/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-[0-9a-f]{32}',str(base))
assert base.resolve(strict=True)==base and root.resolve(strict=True)==root and root.stat().st_uid==os.getuid()
def read(name):
 q=root/name
 if not q.exists():return None
 assert q.is_file()and not q.is_symlink()and q.stat().st_size<131072
 return json.loads(q.read_text())
stage=read('stage.json');assert hashlib.sha256((root/'worker.py').read_bytes()).hexdigest()==stage['worker_sha256']
launch=read('launch.json');start=read('worker-start.json');owner=start['identity']if start else launch['identity']if launch else None
alive=False
if owner:
 q=pathlib.Path('/proc')/str(owner['pid'])
 try:
  v=(q/'stat').read_text().rsplit(') ',1)[1].split()
  alive=(int(v[19]),int(v[3]),q.stat().st_uid)==(owner['birth'],owner['sid'],owner['uid'])and v[0]not in('Z','X')
 except FileNotFoundError:pass
progress=read('progress.json');complete=read('complete.json');exit=read('exit.json');failure=read('failure.json')
files=[]
for part in read('manifest.json')['parts']:
 final=root/'archive'/part['name'];partial=final.with_name(final.name+'.partial')
 files.append(dict(name=part['name'],final_bytes=final.stat().st_size if final.exists()else None,
  partial_bytes=partial.stat().st_size if partial.exists()else None,verified=read('archive/'+part['name']+'.verified.json')))
print(json.dumps(dict(stage='DIRECT_HF_DOWNLOAD_OBSERVATION',base=str(base),checked_unix=time.time(),
 worker_alive=alive,launch=launch,start=start,progress=progress,complete=complete,exit=exit,failure=failure,files=files,
 cpu_only=True,model_or_original_qa_readback=False),allow_nan=False))
'''


def target_command(code):
    path = Path(__file__).with_name('a800_question5_source_transfer.py')
    spec = importlib.util.spec_from_file_location('question5_existing_target_route', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.target_command(code)


def publication():
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    source = Path(__file__).read_bytes()
    if subprocess.check_output(['git', '-C', str(REPO), 'show', commit + ':' + REL]) != source:
        raise ValueError('Publish the exact direct-download helper before execution')
    remote = subprocess.check_output(['git', '-C', str(REPO), 'ls-remote', 'origin', 'refs/heads/search-g1'], text=True)
    if remote.split()[0] != commit:
        raise ValueError('Latest helper source has not reached the published branch')
    return dict(source_commit=commit, helper_sha256=sha_bytes(source)), source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('stage-launch', 'observe'), default='stage-launch')
    parser.add_argument('--handle', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--urls-stdin', action='store_true')
    parser.add_argument('--python', default=PYTHON)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--base', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.base)
        return
    if args.handle is None:
        parser.error('--handle is required')
    handle = json.loads(args.handle.read_text())
    namespace(handle['base'])
    payload = dict(base=handle['base'])
    code = OBSERVE
    if args.action == 'stage-launch':
        if args.manifest is None or not args.urls_stdin:
            parser.error('stage-launch requires --manifest and --urls-stdin')
        manifest = manifest_value(json.loads(args.manifest.read_text()))
        urls = signed_urls(json.load(sys.stdin), manifest)
        published, source = publication()
        payload.update(manifest=manifest, urls=urls, python=args.python, publication=published,
                       worker_sha256=sha_bytes(source), source=base64.b64encode(source).decode(),
                       manifest_sha256=sha_bytes(canonical(manifest)),
                       input_manifest_file_sha256=sha_bytes(args.manifest.read_bytes()))
        code = STAGE
    started = time.time()
    process = subprocess.run(target_command(code), input=json.dumps(payload).encode(), capture_output=True, timeout=65)
    # Never print or persist SSH stderr; only its hash and a generic failure type.
    record = dict(action=args.action, started_unix=started, finished_unix=time.time(),
                  returncode=process.returncode, stderr_sha256=sha_bytes(process.stderr),
                  base=handle['base'], namespace=str(namespace(handle['base'])), signed_urls_persisted=False)
    if process.returncode == 0:
        if len(process.stdout) > 131072:
            raise ValueError('Metadata response exceeded128KiB')
        record['result'] = json.loads(process.stdout)
    else:
        record['error_type'] = 'REMOTE_DIRECT_DOWNLOAD_OPERATOR_NONZERO'
        record['stdout_sha256'] = sha_bytes(process.stdout)
    receipt = args.handle.with_name(args.handle.stem + '-hf-direct-' + args.action + '-' + str(time.time_ns()) + '.json')
    atomic(receipt, record, exclusive=True)
    print(json.dumps(dict(receipt=str(receipt), **record), allow_nan=False))
    if process.returncode:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
