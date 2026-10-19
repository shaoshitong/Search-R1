"""Bounded, resumable private HF asset fetch. Secrets are accepted on stdin only.

No HF client cache, environment modification, signed-URL logging, or GPU use.
A public job binds each selected file to its source SHA256. The token lives only
in this process and curl's configuration pipe. Each file has exactly one writer.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def validate_job(job: dict) -> None:
    assert job['repo'] == 'Sutoonq/TMP_MODEL'
    assert len(job['revision']) == 40 and all(c in '0123456789abcdef' for c in job['revision'])
    assert len(job['files']) == 12
    assert not Path(job['prefix']).is_absolute() and '..' not in Path(job['prefix']).parts
    for name, entry in job['files'].items():
        assert Path(name).name == name and name not in ('.', '..')
        assert not any(x in name for x in ['training_state', '.pt', 'optimizer', 'environment'])
        assert len(entry['sha256']) == 64 and int(entry['size']) > 0
    assert job.get('concurrency', 3) in (1, 2, 3)
    for item in job.get('resolve', []):
        assert item.startswith('huggingface.co:443:') and len(item.split(':')) == 3


def run(job: dict, token: str) -> dict:
    validate_job(job)
    assert token and '\n' not in token and '\r' not in token and '"' not in token
    root = Path(job['output']); root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.asset-migration.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = {k: v for k, v in job.items() if k != 'initial_attempt'}
    bpath = root / 'asset-binding.json'
    if bpath.exists():
        assert json.loads(bpath.read_text()) == binding, 'Immutable asset binding differs'
    else: write_json(bpath, binding)
    total = sum(int(v['size']) for v in job['files'].values())
    required = sum(max(0, int(v['size']) - ((root / n).stat().st_size if (root / n).exists() else 0)) for n,v in job['files'].items())
    sv = os.statvfs(root)
    assert sv.f_bavail * sv.f_frsize >= required + (2 << 30), 'Insufficient filesystem reserve'
    state = {'status': 'DOWNLOADING', 'pid': os.getpid(), 'host': os.uname().nodename,
             'started_epoch': time.time(), 'total_bytes': total, 'files': {},
             'user_quota': 'UNKNOWN; filesystem free space is not user quota',
             'max_buffer_bytes': (4 << 20) * job.get('concurrency', 3),
             'route': job['route'], 'credentials': 'stdin/process RAM only'}
    mutex = threading.Lock(); done = threading.Event()
    def publish():
        with mutex:
            present = 0
            for name, info in job['files'].items():
                final = root/name; part = root/(name+'.'+info['sha256']+'.part')
                q = final if final.exists() else part
                try: present += min(q.stat().st_size, int(info['size']))
                except FileNotFoundError:
                    if final.exists(): present += min(final.stat().st_size, int(info['size']))
            state['observed_bytes'] = present; state['observed_epoch'] = time.time()
            write_json(root/'asset-status.json', state)
    def watcher():
        while not done.wait(5): publish()
    thread = threading.Thread(target=watcher, daemon=True); thread.start(); publish()
    def fetch(name: str, info: dict):
        final = root/name; part = root/(name+'.'+info['sha256']+'.part')
        if final.exists():
            assert final.stat().st_size == info['size'] and digest(final) == info['sha256'], 'Existing final identity differs'
            with mutex: state['files'][name] = {'status':'SOURCE_SHA_VERIFIED','bytes':info['size']}
            return
        if part.exists(): assert part.stat().st_size <= info['size'], 'Oversized partial preserved'
        url = 'https://huggingface.co/'+job['repo']+'/resolve/'+job['revision']+'/'+job['prefix'].strip('/')+'/'+name
        cfg = 'url = "'+url+'"\nheader = "Authorization: Bearer '+token+'"\n'
        env = {k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
        last = None
        for attempt in range(3):
            args = ['curl','--silent','--fail','--location','--proto','=https','--proto-redir','=https',
                    '--max-redirs','5','--noproxy','*','--connect-timeout','10','--max-time','600',
                    '--speed-limit','65536','--speed-time','30','--max-filesize',str(info['size']),'--continue-at','-',
                    '--output',str(part),'--config','-']
            for item in job.get('resolve', []): args += ['--resolve',item]
            with mutex: state['files'][name] = {'status':'FETCHING','attempt':attempt+1}
            process = subprocess.run(args,input=cfg.encode(),stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL,env=env,timeout=615)
            last = process.returncode
            if last == 0: break
            with mutex: state['files'][name] = {'status':'RETRYABLE_TRANSFER_ERROR','curl_exit':last,'attempt':attempt+1}
            publish()
            if attempt < 2: time.sleep(3)
        assert last == 0, 'Bounded curl transfer failed; partial retained'
        assert part.stat().st_size == info['size'], 'Downloaded size differs; partial retained'
        assert digest(part) == info['sha256'], 'Downloaded SHA differs; partial retained'
        with part.open('rb') as f: os.fsync(f.fileno())
        os.replace(part,final)
        with mutex: state['files'][name] = {'status':'SOURCE_SHA_VERIFIED','bytes':info['size']}
        publish()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=job.get('concurrency',3)) as pool:
            futures = [pool.submit(fetch,n,v) for n,v in job['files'].items()]
            for future in futures: future.result()
        with mutex: state['status'] = 'FRESH_SHARED_OR_TARGET_READBACK'
        publish()
        receipt = {}
        for n, info in job['files'].items():
            q=root/n; actual=digest(q)
            assert q.stat().st_size == info['size'] and actual == info['sha256']
            receipt[n] = {'size':q.stat().st_size,'sha256':actual}
        result = {'status':'MIGRATION_ALL_SELECTED_ASSETS_VERIFIED','files':receipt,
                  'verified_bytes':total,'repo':job['repo'],'revision':job['revision'],
                  'prefix':job['prefix'],'output':str(root),'route':job['route'],
                  'fresh_readback_epoch':time.time(),'source_receipt_sha256':job['source_receipt_sha256']}
        write_json(root/'complete.json',result)
        with mutex: state['status']=result['status'];state['finished_epoch']=time.time()
        publish(); return result
    except BaseException as error:
        with mutex: state['status']='FAILED_PARTIALS_PRESERVED';state['error_type']=type(error).__name__;state['finished_epoch']=time.time()
        publish(); raise RuntimeError('Asset transfer did not complete; inspect sanitized asset-status.json') from None
    finally:
        done.set();thread.join(timeout=6);lock.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);args=ap.parse_args()
    token=sys.stdin.readline().strip()
    result=run(json.loads(Path(args.job).read_text()),token)
    print(json.dumps({'status':result['status'],'verified_bytes':result['verified_bytes']}))

if __name__=='__main__': main()
