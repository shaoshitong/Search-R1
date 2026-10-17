"""Publish, stage code only, prepare assets, and submit one HPC3 Prefix5 job."""
import argparse,hashlib,io,json,pathlib,shlex,subprocess,tarfile,time,uuid
from .local_control import run
REPO=pathlib.Path(__file__).resolve().parents[2]
OUTPUT=REPO.parents[1]/'outputs/hpc3-prefix5-switch-20261017'
BASE='/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017'
PYTHON='/data/user/jxu957/experiments/topd-20261008/environment/topd-py310/bin/python'
BRANCH='codex/prefix5-hpc3-20261017'
BOOTSTRAP=r'''
import sys,json,hashlib,tarfile,io,pathlib,os,time
r=json.loads(sys.stdin.buffer.readline());data=sys.stdin.buffer.read();b=pathlib.Path(r['base']);pub=r['publication']
assert str(b)=='/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017' and b.resolve()==b and b.stat().st_uid==os.getuid()
assert hashlib.sha256(data).hexdigest()==r['archive_sha256'] and pub['remote_sha']==pub['commit']
for n in ['source','preparation','logs','env/bin','env/lib/python3.10','cache','tmp','assets','retrieval-cache']:(b/n).mkdir(parents=True,exist_ok=True)
if (b/'preparation/submit-intent.json').exists():raise ValueError('A claimed runtime is immutable')
source=b/'source'/pub['commit']
if not source.exists():
 source.mkdir()
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
  for m in tar.getmembers():
   p=pathlib.Path(m.name)
   if not m.isfile() or p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe code archive')
   target=source/p;target.parent.mkdir(exist_ok=True,parents=True)
   with tar.extractfile(m) as f:target.write_bytes(f.read())
for name,digest in pub['files'].items():
 if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise ValueError('Deployed source SHA differs: '+name)
link=b/'repo'
if link.is_symlink():link.unlink()
link.symlink_to(source)
reused=pathlib.Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')
r17=pathlib.Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-periodic25-thread5-20261016-r17')
def symlink(target,origin):
 if target.is_symlink():assert target.resolve()==origin.resolve()
 else:target.symlink_to(origin.resolve())
site=reused/'source/env/lib/python3.10/site-packages'
symlink(b/'env/lib/python3.10/site-packages',site)
symlink(b/'env/overlay',r17/'env/overlay')
symlink(b/'retriever-gpu-lib',reused/'runtime/gpu-faiss-alignment-candidate')
wrapper='#!/usr/bin/env bash\nset -euo pipefail\nexport PYTHONPATH="'+str(b/'repo')+':'+str(b/'env/overlay')+':'+str(b/'retriever-gpu-lib')+':'+str(site)+'${PYTHONPATH:+:$PYTHONPATH}"\nexport LD_LIBRARY_PATH="'+str(reused/'runtime/retriever-python310-candidate/lib')+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\nunset PYTHONHOME\nexec '+str(reused/'runtime/retriever-python310-candidate/bin/python3.10')+' "$@"\n'
(b/'env/bin/python').write_text(wrapper);(b/'env/bin/python').chmod(0o755)
(b/'preparation/source-publication.json').write_text(json.dumps(pub))
print(json.dumps(dict(status='PREFIX5_HPC3_CODE_FULL_SHA_VERIFIED',files=len(pub['files']),commit=pub['commit'],epoch=time.time())))
'''
SBATCH='''#!/usr/bin/env bash
#SBATCH --job-name=prefix5-thread5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
set -euo pipefail
base=BASE_VALUE
export QPROBE_SOURCE="$base/repo"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CPATH=/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/bootstrap-tools/python-dev/usr/include/python3.10
export HF_HOME="$base/cache/hf"
export XDG_CACHE_HOME="$base/cache"
export TMPDIR="$base/tmp"
cd "$base/repo"
exec 9>"$base/preparation/build-run.lock"
flock -n 9
exec "$base/env/bin/python" -u -m reproduction.hpc3_prefix5.runtime run
'''.replace('BASE_VALUE',shlex.quote(BASE))


def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args])


def deploy():
 if git('status','--porcelain').strip():raise ValueError('Publish the committed exact implementation')
 commit=git('rev-parse','HEAD').decode().strip()
 subprocess.run(['git','-C',str(REPO),'push','origin','HEAD:refs/heads/'+BRANCH],check=True)
 subprocess.run(['git','-C',str(REPO),'fetch','origin',BRANCH],check=True)
 actual=git('rev-parse','FETCH_HEAD').decode().strip();assert actual==commit
 previous=OUTPUT/'publication.json'
 old=json.loads(previous.read_text()) if previous.exists() else {}
 pub=dict(commit=commit,remote_sha=actual,run_id=old.get('run_id',uuid.uuid4().hex),execution_id=old.get('execution_id',uuid.uuid4().hex),files={},created_unix=time.time(),reference='79c645a4a51bd0bcb0fab4ee36b8fcd634290b2f',branch=BRANCH)
 stream=io.BytesIO()
 with tarfile.open(fileobj=stream,mode='w:gz') as tar:
  for name in git('ls-tree','-r','--name-only','HEAD').decode().splitlines():
   p=pathlib.Path(name)
   if p.suffix not in ('.py','.sh','.json','.yaml','.yml','.md','.toml','.txt') and not name.endswith('/version'):continue
   raw=git('show','HEAD:'+name);pub['files'][name]=hashlib.sha256(raw).hexdigest()
   member=tarfile.TarInfo(name);member.size=len(raw);member.mode=0o644;tar.addfile(member,io.BytesIO(raw))
 archive=stream.getvalue();OUTPUT.mkdir(exist_ok=True,parents=True)
 previous.write_text(json.dumps(pub,indent=2));(OUTPUT/'code-only.tar.gz').write_bytes(archive)
 r=run(PYTHON+' -c '+shlex.quote(BOOTSTRAP),json.dumps(dict(base=BASE,publication=pub,archive_sha256=hashlib.sha256(archive).hexdigest())).encode()+b'\n'+archive)
 save('deployment',r)
 if r.returncode:raise RuntimeError('Code deployment failed')
 return pub


def save(name,r):
 v=dict(returncode=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode(),epoch=time.time());(OUTPUT/(name+'.json')).write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)


def prepare():
 r=run(BASE+'/env/bin/python -m reproduction.hpc3_prefix5.runtime prepare');save('preparation',r)
 if r.returncode:raise RuntimeError('CPU asset preparation failed')


def submit(partition):
 command=r'''
import json,pathlib,subprocess,os,fcntl,sys,time
p=json.load(sys.stdin);b=pathlib.Path(p['base']);receipt=b/'preparation/slurm-submission.json'
with (b/'preparation/slurm-submit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if receipt.exists():print(receipt.read_text());raise SystemExit(0)
 if not (b/'preparation/assets-ready.json').exists():raise ValueError('Assets not ready')
 active=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'--name=prefix5-thread5','-o','%i|%T|%R'],universal_newlines=True).strip()
 if active:raise ValueError('Existing owned Prefix5 job; reconcile before submission')
 path=b/'preparation/train.sbatch';path.write_text(p['script'])
 (b/'preparation/slurm-submit-intent.json').write_text(json.dumps({'epoch':time.time(),'partition':p['partition'],'dependency':None}))
 r=subprocess.run(['sbatch','--parsable','--partition='+p['partition'],'--output='+str(b/'logs/slurm-%j.log'),str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 v=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,epoch=time.time(),partition=p['partition'],dependency=None)
 if r.returncode==0:v['job']=r.stdout.strip().split(';')[0]
 receipt.write_text(json.dumps(v));print(json.dumps(v))
'''
 r=run(PYTHON+' -c '+shlex.quote(command),json.dumps(dict(base=BASE,script=SBATCH,partition=partition)).encode());save('submission',r)
 if r.returncode:raise RuntimeError('Submission transport failed: reconcile actual remote intent')


def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','prepare','submit']);p.add_argument('--partition',default='acd_u',choices=['acd_u','acd_ue','emergency_acd']);a=p.parse_args()
 {'deploy':deploy,'prepare':prepare,'submit':lambda:submit(a.partition)}[a.action]()

if __name__=='__main__':main()
