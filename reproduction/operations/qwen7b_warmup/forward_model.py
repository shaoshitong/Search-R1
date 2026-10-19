"""HKUST2 -> HKUST3 verified files; no model data through the control host."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from download_model import atomic, verify, REVISION, MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    dest = args.destination
    assert dest.startswith('/backup/shaoshitong/qwen25-7b-instruct-official-20261019')
    lock = (root / 'forward.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ssh = ['ssh', '-T', '-oBatchMode=yes', '-oStrictHostKeyChecking=yes', '-oConnectTimeout=10', 'shaoshitong@10.120.17.122']
    atomic(root / 'forward-controller.json', dict(pid=os.getpid(), epoch=time.time(), destination=dest, mac_data_used=False))
    subprocess.run(ssh + ['mkdir -p ' + shlex.quote(dest)], check=True)
    done = {}
    if (root / 'forward-progress.json').exists():
        done = json.loads((root / 'forward-progress.json').read_text())['files']
    while True:
        if not (root / 'progress.json').exists() or not (root / 'official-inventory.json').exists():
            time.sleep(5)
            continue
        progress = json.loads((root / 'progress.json').read_text())
        inventory = json.loads((root / 'official-inventory.json').read_text())
        assert inventory['sha'] == REVISION
        entries = {item['rfilename']: item for item in inventory['siblings']}
        for name, row in progress['files'].items():
            if name in done or row['status'] != 'SHA_VERIFIED':
                continue
            actual = verify(root / name, entries[name])
            assert actual['sha256'] == row['sha256']
            atomic(root / 'forward-current.json', dict(file=name, epoch=time.time(), status='TRANSFERRING'))
            target = dest + '/' + name
            subprocess.run(['rsync', '--partial', '--append-verify', '-e', shlex.join(ssh[:-1]),
                str(root / name), 'shaoshitong@10.120.17.122:' + target + '.part'], check=True)
            code = """import pathlib,hashlib,json,os,time
p=pathlib.Path(TARGET+'.part');h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8388608),b''):h.update(b)
assert p.stat().st_size==EXPECTED['bytes'] and h.hexdigest()==EXPECTED['sha256']
p.replace(pathlib.Path(TARGET));r=dict(EXPECTED,status='HKUST3_FILE_SHA_VERIFIED',epoch=time.time())
q=pathlib.Path(TARGET+'.forward-verified.json');t=q.with_suffix('.tmp');t.write_text(json.dumps(r));os.replace(str(t),str(q));print(json.dumps(r))
""".replace('TARGET', repr(target)).replace('EXPECTED', repr(actual))
            result = subprocess.run(ssh + ['python3 -c ' + shlex.quote(code)], check=True, stdout=subprocess.PIPE, universal_newlines=True)
            done[name] = json.loads(result.stdout)
            atomic(root / 'forward-progress.json', dict(files=done, bytes=sum(v['bytes'] for v in done.values()), epoch=time.time()))
        if (root / 'complete.json').exists():
            manifest = json.loads((root / 'verified-manifest.json').read_text())
            if set(done) == set(manifest['files']):
                payload = json.dumps(dict(model=MODEL, revision=REVISION, files=done, total_bytes=manifest['total_bytes'], status='HKUST3_ALL_FILES_SHA_VERIFIED', epoch=time.time()))
                code = 'import pathlib;pathlib.Path(' + repr(dest + '/verified-manifest.json') + ').write_text(' + repr(payload) + ')'
                subprocess.run(ssh + ['python3 -c ' + shlex.quote(code)], check=True)
                atomic(root / 'forward-complete.json', json.loads(payload))
                return
        time.sleep(5)


if __name__ == '__main__':
    main()
