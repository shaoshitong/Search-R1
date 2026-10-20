"""Publish an isolated CPU auditor without changing the active fit code."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess


REMOTE = r'''
import pathlib,json,hashlib,shutil,base64,os,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/balanced-rank1-from-epoch1-prefix20-20261020')
pub=json.loads((b/'preparation/source-publication.json').read_text())
code=b/('control-tools/checkpoint-audit-'+commit[:12])
for name,sha in pub['files'].items():assert hashlib.sha256((pathlib.Path(pub['code_root'])/name).read_bytes()).hexdigest()==sha,name
if not code.exists():shutil.copytree(pub['code_root'],str(code))
own=base64.b64decode(encoded);assert hashlib.sha256(own).hexdigest()==digest
destination=code/rel
if destination.exists():assert hashlib.sha256(destination.read_bytes()).hexdigest()==digest
else:destination.write_bytes(own)
assert hashlib.sha256(destination.read_bytes()).hexdigest()==digest
receipt=json.loads((pathlib.Path(receipt_path) if receipt_path else b/'probe/latest.json').read_text())
assert receipt['steps']>=10464 and receipt['actor_updates']==0
rp=b/'preparation'/('independent-audit-receipt-'+str(receipt['steps'])+'.json')
rp.write_text(json.dumps(receipt,indent=2))
record=dict(commit=commit,code_root=str(code),sha256=digest,relative_path=rel,receipt=str(rp),time=time.time(),active_training_code_unchanged=True)
(b/'preparation/independent-audit-source.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'deployed':record}),flush=True)
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','reproduction.operations.qwen7b_warmup.audit_balanced_checkpoint','--base',str(b),'--receipt',str(rp)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=150)
(b/'preparation/independent-audit.log').write_text(r.stdout+r.stderr)
print(r.stdout,flush=True)
if r.returncode:print(r.stderr,flush=True);raise SystemExit(r.returncode)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control-root', type=Path, required=True)
    parser.add_argument('--receipt', help='Optional remote immutable epoch receipt')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    for source in (Path(__file__), Path(__file__).with_name('audit_balanced_checkpoint.py')):
        rel_source=str(source.relative_to(repo))
        assert source.read_bytes()==subprocess.check_output(['git','show',commit+':'+rel_source],cwd=repo), 'Unpublished audit source'
    assert subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/codex/qwen7b-probe-warmup-20261019'], cwd=repo, text=True).split()[0] == commit
    rel = 'reproduction/operations/qwen7b_warmup/audit_balanced_checkpoint.py'
    data = (repo / rel).read_bytes()
    spec = importlib.util.spec_from_file_location('control', args.control_root / 'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    address = subprocess.check_output(['/usr/sbin/ipconfig', 'getifaddr', 'en0'], text=True).strip()
    inputs = dict(commit=commit, rel=rel, digest=hashlib.sha256(data).hexdigest(), encoded=base64.b64encode(data).decode(), receipt_path=args.receipt)
    code = ''.join(key + '=' + repr(value) + '\n' for key, value in inputs.items()) + REMOTE
    result = control.run('python3 -c ' + shlex.quote(code), account='mou176', source_address=address, timeout=175)
    out = args.control_root / 'outputs/qwen7b-warmup-20261019/balanced-independent-cpu-audit.txt'
    out.write_bytes(result.stdout)
    print('returncode', result.returncode)
    print(result.stdout.decode())
    if result.returncode:
        print(result.stderr.decode())
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
