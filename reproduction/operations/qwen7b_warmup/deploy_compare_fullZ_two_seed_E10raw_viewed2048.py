"""Publish-bound, CPU-only analysis of existing exploratory predictions."""
import argparse
import base64
import hashlib
import importlib.util
import json
import shlex
import subprocess
import zlib
from pathlib import Path

R = Path(__file__).resolve().parents[3]
CONTROL = R.parents[1]
BRANCH = 'codex/q7b-fullZ-two-seed-E10raw-dev-comparison-20261023'
NAME = 'compare_fullZ_two_seed_E10raw_viewed2048.py'
REMOTE = '''import pathlib,json,base64,hashlib,subprocess,os
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
full=A/'online-packed-full169028-M3to10-P20-20261022-v1';pub=json.loads((full/'source-publication.json').read_bytes());assert(full/'fixture/fixture-complete.json').is_file()
b=A/'viewed2048-fullZ-two-seed-E10raw-comparison-20261023-v1';b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
helper=code/'compare_full169028_packed_variance.py';helperraw=base64.b64decode(v['helper']);assert hashlib.sha256(helperraw).hexdigest()==v['helper_sha256']
if helper.exists():assert helper.read_bytes()==helperraw
else:helper.write_bytes(helperraw);helper.chmod(0o444)
raw=base64.b64decode(v['source']);assert hashlib.sha256(raw).hexdigest()==v['sha256'];path=code/v['name']
if path.exists():assert path.read_bytes()==raw
else:path.write_bytes(raw);path.chmod(0o444)
rule=base64.b64decode(v['rule']);assert hashlib.sha256(rule).hexdigest()==v['rule_sha256'];rp=code/'predeclared-seed1-rule.json'
if rp.exists():assert rp.read_bytes()==rule
else:rp.write_bytes(rule);rp.chmod(0o444)
compile(raw,str(path),'exec');runtime=pathlib.Path(pub['code_root'])/'reproduction/operations/qwen7b_warmup/runtime.sh'
env=dict(os.environ,QWEN7B_CODE_ROOT=pub['code_root'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run(['bash',str(runtime),'-u',str(path)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=50)
if r.returncode:raise RuntimeError(r.stdout[-1000:]+r.stderr[-2500:])
print(r.stdout.strip())
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    remote_sha = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + BRANCH], cwd=R, text=True).split()[0]
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', args.commit, remote_sha], cwd=R)
    source = subprocess.check_output(['git', 'show', args.commit + ':reproduction/operations/qwen7b_warmup/' + NAME], cwd=R)
    spec = importlib.util.spec_from_file_location('private_control', CONTROL / 'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    ctl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctl)
    address = subprocess.check_output(['/usr/sbin/ipconfig', 'getifaddr', 'en0'], text=True).strip()
    helper = subprocess.check_output(['git','show',args.commit+':reproduction/operations/qwen7b_warmup/compare_full169028_packed_variance.py'],cwd=R)
    rule=subprocess.check_output(['git','show',args.commit+':reproduction/reports/qwen7b-algorithm-improvement-20261021/leading-Z-replication/ACTUAL-SEED1-SELECTION-BY-PREDECLARED-RULE.json'],cwd=R)
    values = dict(rule=base64.b64encode(rule).decode(),rule_sha256=hashlib.sha256(rule).hexdigest(),helper=base64.b64encode(helper).decode(),helper_sha256=hashlib.sha256(helper).hexdigest(),commit=args.commit, name=NAME, source=base64.b64encode(source).decode(), sha256=hashlib.sha256(source).hexdigest())
    script = 'v=' + repr(values) + '\n' + REMOTE
    packed = base64.b64encode(zlib.compress(script.encode())).decode()
    cmd = 'python3 -c ' + shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(packed) + ')))')
    reply = ctl.run(cmd, account='mou176', source_address=address, timeout=60)
    if reply.returncode:
        raise RuntimeError(reply.stdout[-1200:] + reply.stderr[-2500:])
    result = json.loads(reply.stdout.splitlines()[-1])
    out = R / 'reproduction/reports/qwen7b-algorithm-improvement-20261021/viewed2048-fullZ-two-seed-E10raw-comparison'
    out.mkdir(exist_ok=True)
    (out / ('actual-%d.json' % result['result']['time'])).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (out / 'actual-latest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(receipt=result['receipt'], seconds=result['result']['seconds'],
                         comparisons=result['result']['paired_comparison']), ensure_ascii=False))


if __name__ == '__main__':
    main()
