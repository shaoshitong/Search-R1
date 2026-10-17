"""Deploy only three published CPU source files, then independently rehash them."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6')
REPO = ROOT/'work/easy2560-co-rollout-hpc3-4gpu-20261017'
OUT = ROOT/'outputs/easy2560-co-rollout-hpc3-4gpu-20261017'
BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018'
COMMIT = '0f93e5fd4502478e1a6ab3e7a0e728229adba5ac'
TRAINING = '8a5bac94b8230a99b81977f5290589e3d3708dfe'
NAMES = ('reproduction/easy2560_joint4/__init__.py',
         'reproduction/easy2560_joint4/checkpoint_evidence.py',
         'reproduction/easy2560_joint4/final_cpu_acceptance.py')
DEPENDENCIES = ('reproduction/common8/__init__.py',
                'reproduction/common8/validation_identity.py',
                'reproduction/common8/verify_data.py',
                'reproduction/prefix20_joint4/__init__.py',
                'reproduction/prefix20_joint4/cpu_audit/__init__.py',
                'reproduction/prefix20_joint4/cpu_audit/safe_pt.py')
IDENTITY = dict(hostname='ACD-Login-1', boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',
                pid_namespace='pid:[4026531836]')

REMOTE = r'''
import base64,hashlib,json,os,shutil,sys,time,uuid
from pathlib import Path
p=json.load(sys.stdin);base=Path(p['base']);root=base/'preparation/cpu-acceptance-code'/p['operator_commit']
assert str(base)=='/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018'
assert base.resolve()==base and os.getuid()==14090
assert (base/'job.id').read_text().strip()=='730226'
binding=json.loads((base/'preparation/run-source.json').read_text())
assert binding['source_commit']==p['training_commit'] and binding['runtime_repo']==str(base/'repo')
def rec(path):
 assert path.is_file() and path.resolve()==path and not path.is_symlink()
 raw=path.read_bytes();return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def fsyncdir(path):
 fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
parent=root.parent
if p['mode']=='deploy':
 parent.mkdir(parents=True,exist_ok=True)
 if root.exists():
  existing=json.loads((root/'code-publication.json').read_text())
  assert existing['operator_commit']==p['operator_commit'] and existing['training_commit']==p['training_commit']
  assert existing['files']==p['files'] and existing['immutable_training_dependencies']==p['dependencies']
 else:
  stage=parent/('.partial-'+uuid.uuid4().hex);stage.mkdir()
  try:
   for name,expected in p['files'].items():
    assert name.startswith('reproduction/easy2560_joint4/') and '..' not in Path(name).parts
    raw=base64.b64decode(p['payload'][name]);assert dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==expected
    target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    assert rec(target)==expected
   publication=dict(status='EASY2560_INDEPENDENT_CPU_SOURCE_PUBLISHED',base=str(base),
    operator_commit=p['operator_commit'],training_commit=p['training_commit'],source_job='730226',
    files=p['files'],immutable_training_dependencies=p['dependencies'],training_repo_modified=False,
    checkpoint_seal_or_acceptance_executed=False,checked_unix=time.time())
   with (stage/'code-publication.json').open('x') as f:json.dump(publication,f,indent=2);f.flush();os.fsync(f.fileno())
   for directory in sorted((x for x in stage.rglob('*') if x.is_dir()),key=lambda x:len(x.parts),reverse=True):fsyncdir(directory)
   fsyncdir(stage);stage.rename(root);fsyncdir(parent)
  except Exception:
   if stage.exists():shutil.rmtree(stage)
   raise
files={name:rec(root/name) for name in p['files']};assert files==p['files']
dependencies={}
for name,expected in p['dependencies'].items():
 source=rec(base/'source'/p['training_commit']/name);runtime=rec(base/'repo'/name)
 assert source==runtime==expected and binding['files'][name]==expected
 dependencies[name]=dict(immutable_published_source=source,immutable_runtime=runtime)
runpath=base/'logs/run-manifest.json';run=json.loads(runpath.read_text()) if runpath.is_file() else None
if run is not None:assert run['source_commit']==p['training_commit']
print(json.dumps(dict(status='EASY2560_INDEPENDENT_CPU_SOURCE_SHA_PASS',mode=p['mode'],base=str(base),
 directory=str(root),operator_commit=p['operator_commit'],training_commit=p['training_commit'],source_job='730226',
 run_id=run.get('run_id') if run else None,files=files,dependencies=dependencies,
 code_publication=rec(root/'code-publication.json'),checked_unix=time.time(),
 training_repo_modified=False,checkpoint_seal_or_acceptance_executed=False,tensor_contents_read=False)))
'''


def git_bytes(commit, name):
    return subprocess.check_output(['git', '-C', str(REPO), 'show', commit+':'+name])


def records(commit, names):
    raw = {name:git_bytes(commit, name) for name in names}
    manifest = {name:dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
                for name,value in raw.items()}
    return raw, manifest


def main():
    raw, files = records(COMMIT, NAMES)
    _, dependencies = records(TRAINING, DEPENDENCIES)
    spec = importlib.util.spec_from_file_location('cpu_deployment_account_bridge',
        ROOT/'work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py')
    bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
    payload = dict(base=BASE, operator_commit=COMMIT, training_commit=TRAINING,
                   files=files, dependencies=dependencies,
                   payload={name:base64.b64encode(value).decode() for name,value in raw.items()})
    observations = []
    for mode in ('deploy', 'verify'):
        payload['mode'] = mode
        result = bridge.run('jxu957', 'python3 -B -c '+shlex.quote(REMOTE), json.dumps(payload),
            str(ROOT/'work'), login_host='10.120.48.31', expected_host_identity=IDENTITY,
            observation_timeout=55)
        proof = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                     operator_commit=COMMIT, training_commit=TRAINING, checked_unix=time.time())
        path = OUT/('cpu-acceptance-micro16-'+mode+'-'+str(time.time_ns())+'.json')
        path.write_text(json.dumps(proof, indent=2)+'\n');observations.append(str(path))
        if result.returncode:
            print(json.dumps(dict(status='DEPLOYMENT_NOT_CONFIRMED',observations=observations,stderr=result.stderr)))
            raise SystemExit(result.returncode)
        remote = json.loads(result.stdout)
        assert remote['files']==files
    assert records(COMMIT, NAMES)[1]==files and records(TRAINING, DEPENDENCIES)[1]==dependencies
    summary=dict(status='EASY2560_CPU_SOURCE_DEPLOYED_TWO_FRESH_SHA_CHECKS',
                 operator_commit=COMMIT,training_commit=TRAINING,base=BASE,files=files,
                 dependencies=dependencies,observations=observations,run_id=remote['run_id'],
                 source_job='730226',directory=remote['directory'],checked_unix=time.time(),
                 local_git_source_rechecked=True,training_repo_modified=False,
                 checkpoint_seal_or_acceptance_executed=False)
    target=OUT/('cpu-acceptance-micro16-deployment-summary-'+str(time.time_ns())+'.json')
    target.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'],summary_path=str(target),run_id=summary['run_id'],
                         directory=summary['directory'],observations=observations)))


if __name__=='__main__':main()
