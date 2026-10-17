"""Direct one-shot HK SSH -> authenticated HPC CPU check; no route replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

ROOT = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6')
OUT = ROOT/'outputs/easy2560-co-rollout-hpc3-4gpu-20261017'
HERE = Path(__file__).resolve().parent
OLD = '/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018'
NEW = '/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730226-20261018'
AUDIT = NEW+'/preparation/d915-independent-cpu-reader-check-20261018'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--execute', action='store_true')
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--parent-fixture-retest', action='store_true'); args = parser.parse_args()
    if sum((args.execute,args.observe,args.parent_fixture_retest))>1: parser.error('choose one operation')
    if args.execute or args.parent_fixture_retest:
        source = (HERE/('retest_visible_parent.py' if args.parent_fixture_retest else 'remote_cpu_check.py')).read_text()
        command = shlex.join(['/usr/bin/env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',
            'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',OLD+'/env/bin/python','-B','-c',source])
    else:
        source = """import hashlib,json,time
from pathlib import Path
p=Path(AUDIT)
result=dict(checked_unix=time.time(),audit=str(p),exists=p.exists(),files={})
if p.exists():
 for name in ('started.json','actual-first-five-reader.json','targeted-tests.json','complete.json','visible-parent-fixture-retest.json','visible-parent-fixture-wrapper-retest.json'):
  f=p/name
  if f.exists():
   b=f.read_bytes();result['files'][name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),value=json.loads(b))
print(json.dumps(result,sort_keys=True))
""".replace('AUDIT', repr(AUDIT))
        command = shlex.join(['python3','-B','-c',source])
    record = dict(kind='parent-fixture-retest' if args.parent_fixture_retest else ('execute' if args.execute else 'observe'), command=command,
        script_sha256=hashlib.sha256(source.encode()).hexdigest(), audit=AUDIT,
        credentials_persisted=False, automatic_retry=False, checked_unix=time.time())
    if not args.execute and not args.observe and not args.parent_fixture_retest:
        print(json.dumps(dict(status='LOCAL_DRY_RUN_NO_CREDENTIAL_READ', **record),sort_keys=True)); return
    # No password in argv, files, receipts, source code, or console.
    secret = Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text()
    line = next(value for value in secret.splitlines() if value.startswith('账号 jxu957'))
    password = re.split(r'密码\s*[:：]?\s*', line, 1)[1].strip().strip('`')
    secret = line = None
    hpc_ssh = ['sshpass','-e','ssh','-T','-oStrictHostKeyChecking=yes','-oControlMaster=no','-S','none',
        '-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn',
        '-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts',
        '-oConnectTimeout=8','jxu957@10.120.48.31',command]
    hk_code = "import os,sys,subprocess;env=dict(os.environ,SSHPASS=sys.stdin.readline().rstrip('\\n'));r=subprocess.run("+repr(hpc_ssh)+",input='',env=env,capture_output=True,text=True,timeout=600);print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr);sys.exit(r.returncode)"
    argv = ['ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8','hkust3','python3 -B -c '+shlex.quote(hk_code)]
    try:
        result = subprocess.run(argv,input=password+'\n',capture_output=True,text=True,timeout=620)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                      outcome_uncertain=result.returncode==255)
    except subprocess.TimeoutExpired as error:
        record.update(returncode=124,stdout=(error.stdout or b'').decode(errors='replace') if isinstance(error.stdout,bytes) else error.stdout,
                      stderr='Timeout: outcome uncertain; do not replay', outcome_uncertain=True)
    finally:
        password = None
    record['completed_unix']=time.time()
    path=OUT/('recovery5-d915-independent-cpu-'+record['kind']+'-'+str(int(time.time()))+'.json')
    with path.open('x') as output: json.dump(record,output,sort_keys=True,indent=2);output.write('\n')
    print(json.dumps(dict(receipt=str(path),returncode=record['returncode'],stdout=record.get('stdout'),stderr=record.get('stderr'),
                         outcome_uncertain=record['outcome_uncertain']),sort_keys=True))


if __name__=='__main__':main()
