"""Acquire pinned public NQ dev on HKUST2; optionally forward to HKUST3.

No dataset bytes, URLs with credentials, or tokens are sent to a Mac. Expected
LFS SHA and length came from the official HF API at the pinned dataset commit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

REVISION = 'bcafb8dd07d453be3cbeeeb3f78be1841bddf92c'
EXPECTED_SHA = '10d5ee25e813f3e2092247a5d2797793dd47ef5853959989cb34bf80a110d607'
EXPECTED_BYTES = 1073443
NAME = 'nq-dev-' + REVISION[:12] + '.jsonl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data/shaoshitong/qwen7b-probe-heldout2048-20261020'))
    parser.add_argument('--proxy', default='http://127.0.0.1:17894')
    parser.add_argument('--forward-hkust3', action='store_true')
    args = parser.parse_args()
    source = args.root / 'source'
    source.mkdir(parents=True, exist_ok=True)
    assert os.statvfs(str(source)).f_bavail * os.statvfs(str(source)).f_frsize > 16 * 1024 * 1024
    path = source / NAME
    if path.exists():
        assert path.stat().st_size == EXPECTED_BYTES and sha(path) == EXPECTED_SHA
    else:
        part = path.with_suffix('.jsonl.part')
        env = {k: v for k, v in os.environ.items() if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
        attempts = []
        for proxy in dict.fromkeys([args.proxy, 'http://127.0.0.1:17897']):
            for endpoint in ('https://huggingface.co', 'https://hf-mirror.com'):
                url = endpoint + '/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/' + REVISION + '/nq/dev.jsonl?download=true'
                command = ['curl', '--ipv4', '--http1.1', '--fail', '--location', '--silent', '--show-error',
                           '--connect-timeout', '8', '--max-time', '45', '--noproxy', '', '--proxy', proxy,
                           '--continue-at', '-', '--output', str(part), url]
                result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        universal_newlines=True, timeout=55)
                attempts.append(dict(endpoint=endpoint, proxy=proxy, code=result.returncode,
                                     error=result.stderr[-300:], bytes=part.stat().st_size if part.exists() else 0))
                (source / 'download-attempts.json').write_text(json.dumps(attempts, indent=2))
                print(json.dumps(dict(stage='BOUNDED_REMOTE_DOWNLOAD_ATTEMPT', **attempts[-1])), flush=True)
                if part.exists() and part.stat().st_size == EXPECTED_BYTES and sha(part) == EXPECTED_SHA:
                    break
            if part.exists() and part.stat().st_size == EXPECTED_BYTES and sha(part) == EXPECTED_SHA:
                break
        assert part.stat().st_size == EXPECTED_BYTES and sha(part) == EXPECTED_SHA
        os.rename(str(part), str(path))
    rows = [json.loads(line) for line in path.open()]
    result = dict(status='OFFICIAL_PINNED_NQ_DEV_SHA_VERIFIED', revision=REVISION,
                  path=str(path), sha256=EXPECTED_SHA, bytes=EXPECTED_BYTES,
                  rows=len(rows), fields=sorted(rows[0]), epoch_time=time.time())
    (source / 'download-verified.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if args.forward_hkust3:
        destination = '/backup/shaoshitong/qwen7b-probe-heldout2048-20261020/source'
        options = ['-oBatchMode=yes', '-oConnectTimeout=5', '-oStrictHostKeyChecking=yes']
        target = 'shaoshitong@10.120.17.122'
        subprocess.run(['ssh'] + options + [target, 'mkdir -p -- ' + shlex.quote(destination)], check=True, timeout=15)
        subprocess.run(['scp'] + options + [str(path), target + ':' + destination + '/' + NAME], check=True, timeout=60)
        code = 'import pathlib,hashlib,json;p=pathlib.Path(%r);d=p.read_bytes();assert len(d)==%d and hashlib.sha256(d).hexdigest()==%r;print(json.dumps(dict(path=str(p),bytes=len(d),sha256=hashlib.sha256(d).hexdigest())))' % (destination + '/' + NAME, EXPECTED_BYTES, EXPECTED_SHA)
        check = subprocess.check_output(['ssh'] + options + [target, 'python3 -c ' + shlex.quote(code)], universal_newlines=True, timeout=15)
        result['hkust3'] = json.loads(check)
        result['status'] = 'HKUST2_AND_HKUST3_OFFICIAL_NQ_DEV_SHA_VERIFIED'
        (source / 'forward-verified.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
