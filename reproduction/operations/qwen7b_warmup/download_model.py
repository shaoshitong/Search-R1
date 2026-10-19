"""Public official HF snapshot, resumable on HKUST storage; no credentials."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def fingerprints(path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    git = hashlib.sha1(('blob %d\0' % size).encode())
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            sha.update(block)
            git.update(block)
    return dict(bytes=size, sha256=sha.hexdigest(), git_blob_sha1=git.hexdigest())


def verify(path, item):
    actual = fingerprints(path)
    assert actual['bytes'] == item['size'], 'Snapshot file size differs'
    if item.get('lfs'):
        assert actual['sha256'] == item['lfs']['sha256'], 'Official LFS SHA differs'
    else:
        assert actual['git_blob_sha1'] == item['blobId'], 'Official Git blob SHA differs'
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--proxy', default='http://127.0.0.1:17894')
    parser.add_argument('--workers', type=int, default=11)
    parser.add_argument('--weight-endpoint', default='https://hf-mirror.com')
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    proxies = {} if args.proxy == 'direct' else {'https': args.proxy, 'http': args.proxy}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    atomic(root / 'controller.json', dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        epoch=time.time(), model=MODEL, revision=REVISION, proxy=args.proxy, mac_data_used=False))
    inventory_path = root / 'official-inventory.json'
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
    else:
        for attempt in range(30):
            try:
                with opener.open('https://huggingface.co/api/models/%s/revision/%s?blobs=true' % (MODEL, REVISION), timeout=20) as handle:
                    inventory = json.load(handle)
                assert inventory['sha'] == REVISION
                atomic(inventory_path, inventory)
                break
            except Exception as error:
                atomic(root / 'api-retry.json', dict(attempt=attempt+1, error=type(error).__name__, epoch=time.time()))
                if attempt == 29:
                    raise
                time.sleep(min(20, attempt+2))
    assert inventory['sha'] == REVISION
    names = {'config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json',
        'vocab.json', 'merges.txt', 'model.safetensors.index.json'}
    items = [item for item in inventory['siblings'] if item['rfilename'] in names or item['rfilename'].endswith('.safetensors')]
    items.sort(key=lambda item: (item['rfilename'].endswith('.safetensors'), item['rfilename']))
    assert all('/' not in item['rfilename'] and item.get('size', 0) > 0 for item in items)
    assert len([item for item in items if item['rfilename'].endswith('.safetensors')]) == 4
    old_state = json.loads((root / 'progress.json').read_text()) if (root / 'progress.json').exists() else {}
    state = dict(model=MODEL, revision=REVISION, total_bytes=sum(item['size'] for item in items),
        files={}, errors=old_state.get('errors', 0), status='DOWNLOADING', mac_data_used=False)
    mutex = threading.Lock()

    def update(name, **fields):
        with mutex:
            state['files'].setdefault(name, {}).update(fields)
            state['epoch'] = time.time()
            atomic(root / 'progress.json', state)

    def download(item):
        name = item['rfilename']
        final = root / name
        partial = root / (name + '.part')
        if final.exists():
            actual = verify(final, item)
            update(name, status='SHA_VERIFIED', **actual)
            return actual
        for attempt in range(80):
            offset = partial.stat().st_size if partial.exists() else 0
            try:
                if offset != item['size']:
                    assert offset < item['size'], 'Partial larger than official file'
                    weights = name.endswith('.safetensors')
                    endpoint = args.weight_endpoint
                    route = 'raw' if endpoint == 'https://hf-mirror.com' and not weights and attempt % 2 == 0 else 'resolve'
                    url = '%s/%s/%s/%s/%s' % (endpoint, MODEL, route, REVISION, name)
                    connection = direct if endpoint == 'https://hf-mirror.com' else opener
                    headers = {'Accept-Encoding': 'identity', 'User-Agent': 'Mozilla/5.0'}
                    if weights or offset:
                        headers['Range'] = 'bytes=%d-' % offset
                    request = urllib.request.Request(url, headers=headers)
                    with connection.open(request, timeout=25) as response:
                        if response.status == 206:
                            assert response.headers['Content-Range'].startswith('bytes %d-' % offset)
                            mode = 'ab'
                        else:
                            assert response.status == 200
                            offset, mode = 0, 'wb'
                        last = 0
                        with partial.open(mode) as handle:
                            while True:
                                block = response.read(1024**2)
                                if not block:
                                    break
                                handle.write(block)
                                offset += len(block)
                                assert offset <= item['size']
                                if time.monotonic()-last >= 4:
                                    handle.flush()
                                    update(name, status='DOWNLOADING', bytes=offset, total_bytes=item['size'], attempt=attempt+1)
                                    last = time.monotonic()
                            handle.flush()
                            os.fsync(handle.fileno())
                assert partial.stat().st_size == item['size'], 'Short download'
                update(name, status='VERIFYING_SHA', bytes=item['size'], total_bytes=item['size'])
                actual = verify(partial, item)
                partial.replace(final)
                update(name, status='SHA_VERIFIED', **actual)
                return actual
            except Exception as error:
                with mutex:
                    state['errors'] += 1
                update(name, status='RETRY', bytes=partial.stat().st_size if partial.exists() else 0,
                    total_bytes=item['size'], attempt=attempt+1, error=type(error).__name__, http_status=getattr(error, 'code', None))
                if attempt == 79 or isinstance(error, AssertionError):
                    raise
                time.sleep(min(20, attempt+2))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(download, items))
    # Fresh complete read, independent of download-time verification.
    manifest = {item['rfilename']: verify(root / item['rfilename'], item) for item in items}
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    assert set(index['weight_map'].values()) == {item['rfilename'] for item in items if item['rfilename'].endswith('.safetensors')}
    atomic(root / 'verified-manifest.json', dict(model=MODEL, revision=REVISION, files=manifest, total_bytes=state['total_bytes'], epoch=time.time()))
    state.update(status='OFFICIAL_SNAPSHOT_ALL_FILES_FRESH_SHA_VERIFIED', epoch=time.time())
    atomic(root / 'progress.json', state)
    atomic(root / 'complete.json', state)


if __name__ == '__main__':
    main()
