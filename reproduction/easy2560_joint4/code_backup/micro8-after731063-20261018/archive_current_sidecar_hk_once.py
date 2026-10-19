"""One explicit milestone sidecar transfer; credentials stay in RAM/stdin."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import time

SCRIPT = '/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/postprocessing-operators/job-734689/current-sidecar-archive734689.py'
SHA = '1b5d161ad1dd0360977446bc4c212cc63b348a76642c191f42526437cb101f73'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True, choices=(50, 100, 150, 200))
    args = parser.parse_args()
    w = Path(__file__).resolve().parents[4]
    out = w.parents[1] / 'outputs/easy2560-co-rollout-hpc3-4gpu-20261017'
    claim = out / ('current734689-sidecar-HK-step%d-execution-claim.json' % args.step)
    with claim.open('x') as stream:
        json.dump(dict(job='734689', step=args.step, script=SCRIPT, script_sha256=SHA,
                       checked_unix=time.time(), automatic_retry=False,
                       training_modified=False, credentials_persisted=False), stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    entry = w / 'reproduction/easy2560_joint4/code_backup/recovery5-after730912-hk-source-8033-20261018/deploy_exact8033_HK_source_once.py'
    spec = importlib.util.spec_from_file_location('existing_single_transport', entry)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    secret = Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text()
    line = next(x for x in secret.splitlines() if x.startswith('账号 jxu957'))
    password = re.split(r'密码\s*[:：]?\s*', line, 1)[1].strip().strip('`')
    secret = line = None
    payload = dict(password=password)
    password = None
    code = ('import hashlib,runpy,sys;from pathlib import Path;'
            'p=Path(' + repr(SCRIPT) + ');'
            'assert hashlib.sha256(p.read_bytes()).hexdigest()==' + repr(SHA) + ';'
            'sys.argv=[str(p),"--step",' + repr(str(args.step)) + ',"--execute"];'
            'runpy.run_path(str(p),run_name="__main__")')
    try:
        result = transport.run_ssh(code, payload,
            out / ('current734689-sidecar-HK-step%d-transfer-result.json' % args.step))
    finally:
        payload.clear()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
