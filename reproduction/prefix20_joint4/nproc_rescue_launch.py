"""Publish/run the reviewed fixed-job repair with an actual parent wait receipt."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

from reproduction.prefix20_joint4 import raise_owned_nproc as repair

REMOTE = r'''
import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);plan=p['plan'];base=Path(plan['base']);commit=plan['operator_publication_commit']
assert plan['job']=='727251' and plan['training_commit']=='78074b669a0d47648ad7c5b2a687159af00cd090'
root=base/'preparation/nproc-rescue-code'/commit
assert root.resolve()==root
root.mkdir(parents=True,exist_ok=False)
raw=base64.b64decode(p['code'],validate=True)
assert hashlib.sha256(raw).hexdigest()==plan['operator_sha256']
def save(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
code=root/'raise_owned_nproc.py'
with code.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
assert digest(code)==plan['operator_sha256']
path=root/'public-plan.json';save(path,plan)
command=['srun','--jobid=727251','--overlap','--immediate=10','--ntasks=1','--cpus-per-task=1','--mem=0','--gres=none','--time=00:05:00',
 'env','CUDA_VISIBLE_DEVICES=',str(base/'env/bin/python'),'-B',str(code),'--plan',str(path)]
if p['apply']:command.append('--apply')
save(root/'launch-claim.json',dict(command=command,operator_sha256=plan['operator_sha256'],
 source_commit=commit,apply=p['apply'],started_unix=time.time(),retry_allowed=False,credentials_persisted=False))
out=root/'child.stdout';err=root/'child.stderr'
with out.open('xb') as stdout,err.open('xb') as stderr:
 child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
 stat=Path('/proc/%d/stat'%child.pid).read_text().rsplit(')',1)[1].split()
 save(root/'child-birth.json',dict(pid=child.pid,start_ticks=int(stat[19]),sid=int(stat[3]),
  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),uid=os.getuid()))
 result=child.wait()
 stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
value=dict(actual_parent_wait=True,actual_child_exit_code=result,command=command,apply=p['apply'],
 stdout_file=str(out),stderr_file=str(err),stdout_sha256=digest(out),stderr_sha256=digest(err),checked_unix=time.time())
save(root/'actual-wait.json',value)
print(json.dumps(dict(value,stdout=out.read_text(),stderr=err.read_text())),flush=True)
sys.exit(result)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--bridge-source', required=True)
    parser.add_argument('--control-work', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    plan = repair.validate_plan(json.loads(args.plan.read_text()))
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if plan['operator_publication_commit'] != commit or subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True):
        raise ValueError('NPROC_LAUNCH_REQUIRES_CLEAN_PUBLISHED_OPERATOR_COMMIT')
    remote = subprocess.check_output(['git', '-C', str(repo), 'ls-remote', 'git@github.com:shaoshitong/Search-R1.git',
        'refs/heads/codex/qprobe-prefix20-hpc3-4gpu-20261016'], text=True).split()[0]
    if remote != commit:
        raise ValueError('NPROC_LAUNCH_LATEST_SOURCE_NOT_PUSHED')
    raw = (repo/'reproduction/prefix20_joint4/raise_owned_nproc.py').read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan['operator_sha256']:
        raise ValueError('NPROC_LAUNCH_OPERATOR_SHA_DIFFERS')
    spec = importlib.util.spec_from_file_location('existing_bridge', args.bridge_source)
    bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
    payload = json.dumps(dict(plan=plan, code=base64.b64encode(raw).decode(), apply=args.apply))
    result = bridge.run('jxu957', str(Path(plan['base'])/'env/bin/python')+' -B -c '+shlex.quote(REMOTE),
        payload, args.control_work, login_host='10.120.48.31', observation_timeout=600,
        expected_host_identity=dict(hostname='ACD-Login-1', boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a', pid_namespace='pid:[4026531836]'))
    value = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        operator_commit=commit, operator_sha256=plan['operator_sha256'], checked_unix=time.time(),
        outcome_unknown_no_retry=result.returncode==124, credentials_persisted=False)
    with args.output.open('x') as out:json.dump(value,out,indent=2)
    print(json.dumps(value), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
