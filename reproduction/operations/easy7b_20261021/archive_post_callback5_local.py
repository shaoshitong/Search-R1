"""Publish-bound local launcher/observer for the independent CP5 CPU supplement."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


MODULE_COMMIT = 'b87a0c3d7a7ca2d9f6751a99179897d8d8aca56d'
MODULE_SHA = '082201e54ab06626e877ea66e70f881ff9f6e69cc81a2eb03b6e98ced51b2c99'
BRANCH = 'codex/easy2560-qwen7b-20261021'
PREFIX = 'reproduction/operations/easy7b_20261021/'
RUNNER = 'archive_post_callback5_hk.py'
OBSERVER = 'observe_post_callback5_hk.py'
MAX_PAYLOAD = 128 * 1024


REMOTE_LAUNCH = r'''import fcntl,hashlib,importlib.util,json,os,re,socket,subprocess,sys,time,uuid
from pathlib import Path
assert socket.gethostname()=='xielab3' and os.getuid()==1002
raw=sys.stdin.buffer.read(131073)
assert 0<len(raw)<=131072
p=json.loads(raw);raw=None
assert set(p)=={'launcher_commit','remote_head','origin_ref','files','runner_envelope'}
commit=p['launcher_commit'];assert re.fullmatch('[0-9a-f]{40}',commit) and p['remote_head']==commit
assert p['origin_ref']=='refs/heads/codex/easy2560-qwen7b-20261021'
assert set(p['files'])=={'archive_post_callback5_hk.py','observe_post_callback5_hk.py'}
hk=Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017')
root=hk/'post-callback-boundary-control/job-745371/global_step_5'
tools=hk/'post-callback-boundary-launch-tools'/commit
def canonical(path):
 assert path.is_absolute() and path.resolve()==path and not path.is_symlink()
 return path
def digest(path):return hashlib.sha256(canonical(path).read_bytes()).hexdigest()
def syncdir(path):
 fd=os.open(str(canonical(path)),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def publish(path,value):
 canonical(path);data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as out:out.write(data);out.flush();os.fsync(out.fileno())
 syncdir(path.parent)
tools.mkdir(parents=True,exist_ok=True);canonical(tools)
files={}
for name,row in p['files'].items():
 assert set(row)=={'source','sha256','bytes'} and isinstance(row['source'],str)
 data=row['source'].encode('utf-8')
 assert 0<len(data)<=65536 and row['bytes']==len(data) and hashlib.sha256(data).hexdigest()==row['sha256']
 target=canonical(tools/name)
 if target.exists():assert target.is_file() and target.stat().st_size==len(data) and digest(target)==row['sha256']
 else:
  tmp=tools/('.'+name+'.partial-'+uuid.uuid4().hex)
  fd=os.open(str(tmp),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(fd,'wb') as out:out.write(data);out.flush();os.fsync(out.fileno())
  assert digest(tmp)==row['sha256'];os.link(str(tmp),str(target));tmp.unlink();syncdir(tools)
 assert digest(target)==row['sha256']
 files[name]=dict(path=str(target),sha256=row['sha256'],bytes=len(data))
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('published_post_callback5_runner',str(tools/'archive_post_callback5_hk.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
assert runner.MODULE_COMMIT=='b87a0c3d7a7ca2d9f6751a99179897d8d8aca56d'
assert runner.MODULE_SHA=='082201e54ab06626e877ea66e70f881ff9f6e69cc81a2eb03b6e98ced51b2c99'
assert runner.CONTROL==root and runner.JOB=='745371' and runner.STEP==5
runner.validate_envelope(p['runner_envelope'])
core_before=runner.ready_core()
root.mkdir(parents=True,exist_ok=True);canonical(root)
lockfd=os.open(str(root/'launch.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
with os.fdopen(lockfd,'a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not any((root/name).exists() for name in ('launch-claim.json','launch-receipt.json','launch-failure.json',
  'runner.stdout.log','claim.json','result.json','failure.json')), 'Inspect prior launch; replay forbidden'
 claim=dict(base=str(runner.BASE),source_job=runner.JOB,run_id=runner.RUN,step=5,training_commit=runner.TRAINING,
  module_commit=runner.MODULE_COMMIT,module_sha256=runner.MODULE_SHA,launcher_commit=commit,remote_head=p['remote_head'],
  origin_ref=p['origin_ref'],tools=str(tools),public_code=files,core_before=core_before,credentials_persisted=False,
  training_modified=False,healthy_controller_modified=False,jobs_submitted=0,created_unix=time.time())
 publish(root/'launch-claim.json',claim)
 child=None;secret=p['runner_envelope']['password']
 try:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');env.pop('SSHPASS',None)
  command=[sys.executable,'-B',str(tools/'archive_post_callback5_hk.py')]
  with (root/'runner.stdout.log').open('xb') as output:
   child=subprocess.Popen(command,cwd=str(tools),env=env,stdin=subprocess.PIPE,stdout=output,
    stderr=subprocess.STDOUT,start_new_session=True)
  process=Path('/proc')/str(child.pid);fields=(process/'stat').read_text().rsplit(')',1)[1].split()
  birth=dict(pid=child.pid,start_ticks=int(fields[19]),state=fields[0],uid=process.stat().st_uid,
   boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),hostname=socket.gethostname(),sid=int(fields[3]))
  payload=json.dumps(p['runner_envelope']).encode();assert 0<len(payload)<=262144
  child.stdin.write(payload);child.stdin.close();payload=None;p['runner_envelope'].pop('password');secret=None
  receipt=dict(claim,root=str(root),command=command,birth=birth,cpu_runners_started=1,controllers_restarted=0,
   detached=True,checked_unix=time.time())
  publish(root/'launch-receipt.json',receipt)
  print(json.dumps(receipt,sort_keys=True),flush=True)
 except BaseException as error:
  if child is not None and child.poll() is None:
   if child.stdin is not None and not child.stdin.closed:child.stdin.close()
   child.terminate()
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:child.kill();child.wait(timeout=15)
  detail=str(error).replace(secret,'[REDACTED]') if secret else str(error)
  publish(root/'launch-failure.json',dict(error_type=type(error).__name__,error=detail[-2000:],source_job='745371',
   credentials_persisted=False,training_modified=False,automatic_retry=False,checked_unix=time.time()))
  raise RuntimeError('POST_CALLBACK5_LAUNCH_FAILED_INSPECT_PRESERVED_FAILURE') from None
'''


def need(condition, message):
    if not condition: raise ValueError('POST_CALLBACK5_LOCAL_' + message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded)
    return loaded


def published_sources(repo, git=None):
    if git is None:
        def git(*args): return subprocess.check_output(['git','-C',str(repo),*args])
    need(not git('status','--porcelain').strip(), 'CLEAN_PUBLISHED_WORKTREE_REQUIRED')
    commit = git('rev-parse','HEAD').decode().strip()
    branch = git('symbolic-ref','--quiet','--short','HEAD').decode().strip()
    need(re.fullmatch('[0-9a-f]{40}',commit) is not None and branch == BRANCH, 'EXPECTED_BRANCH_HEAD_REQUIRED')
    remote = git('ls-remote','origin','refs/heads/'+branch).decode().splitlines()
    need(len(remote)==1 and remote[0].split()==[commit,'refs/heads/'+branch], 'REMOTE_HEAD_MUST_EQUAL_LOCAL_HEAD')
    raw = git('show',MODULE_COMMIT+':reproduction/easy2560_joint4/post_callback_archive.py')
    need(hashlib.sha256(raw).hexdigest()==MODULE_SHA and len(raw)==20774, 'FROZEN_MODULE_BYTES_DIFFER')
    files = {}
    for name in (RUNNER,OBSERVER):
        data = git('show',commit+':'+PREFIX+name)
        need(0<len(data)<=65536, 'SMALL_PUBLIC_CODE_REQUIRED')
        files[name] = dict(source=data.decode('utf-8'),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    return commit, raw, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('launch','observe'),required=True)
    parser.add_argument('--verify-files',action='store_true',help='Optional eight-file readback in observe mode')
    args = parser.parse_args()
    need(args.mode=='observe' or not args.verify_files, 'VERIFY_FILES_IS_READ_ONLY_OBSERVATION')
    directory = Path(__file__).resolve().parent; repo = directory.parents[2]; workspace = repo.parents[1]
    commit, raw_module, files = published_sources(repo)
    transport = module('existing_bound_bridge', workspace/'work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py')
    password = None
    if args.mode=='launch':
        secret = Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text()
        line = next(line for line in secret.splitlines() if line.startswith('账号 jxu957'))
        password = re.split(r'密码\s*[:：]?\s*',line,1)[1].strip().strip('`');secret=line=None
        payload = json.dumps(dict(launcher_commit=commit,remote_head=commit,origin_ref='refs/heads/'+BRANCH,files=files,
            runner_envelope=dict(password=password,module_source=raw_module.decode('utf-8'),
                                 module_sha256=MODULE_SHA,published_commit=MODULE_COMMIT)))
        need(len(payload.encode())<=MAX_PAYLOAD, 'SMALL_RAM_ENVELOPE_REQUIRED')
        code = REMOTE_LAUNCH; argv=['python3','-B','-c',code]
    else:
        payload='';argv=['python3','-B','-c',files[OBSERVER]['source']]
        if args.verify_files:argv.append('--verify-files')
    result = transport._campus_long(shlex.join(argv),payload,55);payload=None
    stdout = result.stdout.replace(password,'[REDACTED]') if password else result.stdout
    stderr = result.stderr.replace(password,'[REDACTED]') if password else result.stderr
    password=None
    receipt = workspace/'outputs/easy2560-qwen7b-20261021'/('post-callback5-'+args.mode+'-'+str(int(time.time()))+'.json')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x') as output:
        json.dump(dict(returncode=result.returncode,stdout=stdout,stderr=stderr,launcher_commit=commit,
            module_commit=MODULE_COMMIT,module_sha256=MODULE_SHA,public_code_sha256={n:r['sha256'] for n,r in files.items()},
            credentials_persisted=False),output,indent=2);output.write('\n')
    print(str(receipt),'returncode',result.returncode)
    print(stdout if result.returncode==0 else stderr[-2000:])
    raise SystemExit(bool(result.returncode))


if __name__=='__main__':main()
