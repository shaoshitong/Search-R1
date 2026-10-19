"""HKUST3 -> HPC3 with RAM-only password and fresh whole-snapshot SHA."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from download_model import atomic, fingerprints, MODEL, REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    root, dest = Path(args.root), args.destination
    assert dest == '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/assets/qwen25-7b-instruct-a09a35458c70'
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'target.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    password = sys.stdin.readline().strip()
    assert password and '\x00' not in password
    env = dict(os.environ, SSHPASS=password)
    password = None
    ssh = ['sshpass', '-e', 'ssh', '-T', '-oBatchMode=no', '-oStrictHostKeyChecking=yes',
        '-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn',
        '-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts',
        '-oConnectTimeout=10', '-oServerAliveInterval=15', '-oServerAliveCountMax=3', 'mou176@10.120.81.30']
    atomic(root / 'target-controller.json', dict(pid=os.getpid(), epoch=time.time(), destination=dest, mac_data_used=False))

    def remote(code):
        value = subprocess.run(ssh + ['python3 -c ' + shlex.quote(code)], env=env,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        return json.loads(value.stdout)

    remote("import pathlib,json; p=pathlib.Path(%r);p.mkdir(parents=True,exist_ok=True); print(json.dumps({'ready':True}))" % dest)
    done, errors = {}, 0
    if (root / 'target-progress.json').exists():
        old = json.loads((root / 'target-progress.json').read_text())
        done, errors = old['files'], old['errors']
    while True:
        try:
            for receipt in sorted(root.glob('*.forward-verified.json')):
                name = receipt.name[:-len('.forward-verified.json')]
                if name in done:
                    continue
                item = json.loads(receipt.read_text())
                assert item['status'] == 'HKUST3_FILE_SHA_VERIFIED'
                source = root / name
                actual = fingerprints(source)
                assert actual['bytes'] == item['bytes'] and actual['sha256'] == item['sha256']
                atomic(root / 'target-current.json', dict(file=name, bytes=item['bytes'], epoch=time.time(), status='TRANSFERRING'))
                target = dest + '/' + name
                subprocess.run(['rsync', '--partial', '--append-verify', '-e', shlex.join(ssh[:-1]),
                    str(source), 'mou176@10.120.81.30:' + target + '.part'], check=True, env=env)
                code = """import pathlib,hashlib,json,os,time
p=pathlib.Path(TARGET+'.part');h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8388608),b''):h.update(b)
assert p.stat().st_size==EXPECTED['bytes'] and h.hexdigest()==EXPECTED['sha256']
p.replace(pathlib.Path(TARGET));print(json.dumps(dict(EXPECTED,epoch=time.time(),status='HPC3_FILE_SHA_VERIFIED')))
""".replace('TARGET', repr(target)).replace('EXPECTED', repr(actual))
                done[name] = remote(code)
                atomic(root / 'target-progress.json', dict(files=done, bytes=sum(v['bytes'] for v in done.values()), errors=errors, epoch=time.time()))
            if (root / 'verified-manifest.json').exists():
                source_manifest = json.loads((root / 'verified-manifest.json').read_text())
                assert source_manifest['revision'] == REVISION
                if set(source_manifest['files']) == set(done):
                    code = """import pathlib,hashlib,json,os,time
root=pathlib.Path(DEST);manifest=MANIFEST
for name,item in manifest['files'].items():
 p=root/name; h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 assert p.stat().st_size==item['bytes'] and h.hexdigest()==item['sha256'],name
result=dict(manifest,status='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED',fresh_all_files_SHA256=True,epoch=time.time(),route='OfficialHF->HKUST2->HKUST3->HPC3',mac_data_used=False)
p=root/'complete.json';q=root/'complete.json.tmp';q.write_text(json.dumps(result,indent=2));os.replace(str(q),str(p));print(json.dumps(result))
""".replace('DEST', repr(dest)).replace('MANIFEST', repr(source_manifest))
                    result = remote(code)
                    atomic(root / 'target-complete.json', result)
                    return
            time.sleep(5)
        except (subprocess.CalledProcessError, OSError) as error:
            errors += 1
            atomic(root / 'target-retry.json', dict(errors=errors, error=type(error).__name__, epoch=time.time()))
            time.sleep(min(30, errors+2))


if __name__ == '__main__':
    main()
