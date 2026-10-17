"""Publish a four-total-GPU correction to owned pending728815, never resubmit."""
import base64
import gzip
import hashlib
import json
import shlex
import subprocess
import time

from .control import BASE,OUTPUT,PYTHON,REPO,BRANCH,SBATCH,git
from .direct_control import run


def remote(code,timeout=45):
    packed=base64.b64encode(gzip.compress(code.encode())).decode()
    loader='import base64,gzip;exec(compile(gzip.decompress(base64.b64decode('+repr(packed)+')),"four-shared-control","exec"))'
    address=subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip()
    result=run(PYTHON+' -c '+shlex.quote(loader),account='mou176',source_address=address,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stdout.decode()+result.stderr.decode())
    lines=[line for line in result.stdout.decode().splitlines() if line.strip().startswith('{')]
    if not lines:raise RuntimeError('Missing control receipt')
    return json.loads(lines[-1])


GUARD='''import json,pathlib,subprocess,time,os,hashlib
b=pathlib.Path(BASE_VALUE)
assert b.stat().st_uid==os.getuid()==21930 and b.resolve()==b
job=subprocess.check_output(['scontrol','show','job','728815','-o'],text=True)
assert 'UserId=mou176(21930)' in job and 'JobName=prefix5-thread5' in job
assert 'JobState=PENDING' in job and 'Reason=JobHeldUser' in job,job
assert not (b/'preparation/submit-intent.json').exists()
'''.replace('BASE_VALUE',repr(BASE))


def main():
    assert not git('status','--porcelain').strip(),'Commit latest source first'
    commit=git('rev-parse','HEAD').decode().strip()
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=REPO,check=True)
    subprocess.run(['git','fetch','origin',BRANCH],cwd=REPO,check=True)
    assert git('rev-parse','FETCH_HEAD').decode().strip()==commit
    old=json.loads((OUTPUT/'publication.json').read_text());pub=dict(old,commit=commit,remote_sha=commit,files={},created_unix=time.time())
    for name in git('ls-tree','-r','--name-only','HEAD').decode().splitlines():
        if __import__('pathlib').Path(name).suffix not in ('.py','.sh','.json','.yaml','.yml','.md','.toml','.txt') and not name.endswith('/version'):continue
        pub['files'][name]=hashlib.sha256(git('show','HEAD:'+name)).hexdigest()
    remote(GUARD+'''import shutil
old=json.loads((b/'preparation/source-publication.json').read_text())
assert old['commit']==OLD
target=b/'source'/NEW
assert not target.exists()
shutil.copytree(b/'source'/OLD,target)
print(json.dumps(dict(status='UNREFERENCED_SOURCE_CLONE_CREATED',commit=NEW)))
'''.replace('OLD',repr(old['commit'])).replace('NEW',repr(commit)))
    for name,digest in pub['files'].items():
        if old['files'].get(name)==digest:continue
        payload=base64.b64encode(gzip.compress(git('show','HEAD:'+name))).decode()
        remote(GUARD+'''import base64,gzip
p=b/'source'/NEW/NAME;p.parent.mkdir(parents=True,exist_ok=True)
raw=gzip.decompress(base64.b64decode(PAYLOAD));assert hashlib.sha256(raw).hexdigest()==DIGEST
p.write_bytes(raw)
print(json.dumps(dict(status='SOURCE_PATCH_SHA_VERIFIED',file=NAME)))
'''.replace('NEW',repr(commit)).replace('NAME',repr(name)).replace('PAYLOAD',repr(payload)).replace('DIGEST',repr(digest)))
    # Bounded command chunks; no model bytes, credentials or stdin payload.
    rows=list(pub['files'].items());parts=[]
    for index in range(0,len(rows),120):
        subset=dict(rows[index:index+120]);part='four-shared-manifest-%03d.json'%(index//120);parts.append(part)
        remote(GUARD+'''files=FILES
source=b/'source'/NEW
for name,digest in files.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
(b/'preparation'/PART).write_text(json.dumps(files))
print(json.dumps(dict(status='SOURCE_CHUNK_SHA_VERIFIED',files=len(files))))
'''.replace('FILES',repr(subset)).replace('NEW',repr(commit)).replace('PART',repr(part)))
    tests=remote(GUARD+'''import sys,unittest,io
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TMPDIR']=str(b/'tmp');os.environ['PYTHONDONTWRITEBYTECODE']='1'
source=b/'source'/NEW;sys.path.insert(0,str(source));sys.path.insert(1,str(b/'env/lib/python3.10/site-packages'));sys.path.insert(1,str(b/'env/overlay'))
import torch;torch.set_num_threads(2)
names=['reproduction.hpc3_prefix5.test_shared_gpu','reproduction.prefix_selection200.test_actor_reference_contract','reproduction.prefix_selection200.test_grounding_protocol','reproduction.prefix_selection200.test_prefix5_capture_plan']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
(b/'preparation/four-shared-cpu-tests.log').write_text(log.getvalue())
value=dict(success=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),epoch=time.time(),scope='CPU placement/admission and unchanged loss-gradient/grounding/prefix regressions; GPU sharing not yet tested')
(b/'preparation/four-shared-cpu-tests.json').write_text(json.dumps(value));print(json.dumps(value))
assert result.wasSuccessful(),log.getvalue()
'''.replace('NEW',repr(commit)),timeout=60)
    (OUTPUT/'four-shared-cpu-tests.json').write_text(json.dumps(tests,indent=2))
    metadata=dict(pub);metadata.pop('files')
    receipt=remote(GUARD+'''import fcntl
source=b/'source'/NEW;files={}
for part in PARTS:files.update(json.loads((b/'preparation'/part).read_text()))
actual={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
assert actual==set(files),(actual-set(files),set(files)-actual)
for name,digest in files.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
pub=METADATA;pub['files']=files
with (b/'preparation/build-run.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 alias=b/'repo';temp=b/'repo-four-shared-next';assert not temp.exists();temp.symlink_to(source);os.replace(temp,alias)
 (b/'preparation/source-publication.json').write_text(json.dumps(pub))
 sys=__import__('sys');sys.dont_write_bytecode=True;sys.path.insert(0,str(source))
 from reproduction.hpc3_prefix5.runtime import prepare
 result=prepare();assert result['total_physical_gpus']==4
 (b/'preparation/train.sbatch').write_text(SBATCH)
 updated=subprocess.run(['scontrol','update','JobId=728815','TresPerNode=gres/gpu:4'],capture_output=True,text=True)
 assert updated.returncode==0,updated.stderr
 actual_job=subprocess.check_output(['scontrol','show','job','728815','-o'],text=True)
 assert 'TresPerNode=gres/gpu:4 ' in actual_job and 'gres/gpu=8' not in actual_job and 'Partition=emergency_acd ' in actual_job,actual_job
 result=dict(status='FOUR_TOTAL_SHARED_GPUS_FULL_SOURCE_SHA_VERIFIED',job='728815',epoch=time.time(),commit=NEW,files=len(files),job_before_release=actual_job,total_physical_gpus=4)
 (b/'preparation/four-shared-publication.json').write_text(json.dumps(result))
 released=subprocess.run(['scontrol','release','728815'],capture_output=True,text=True);assert released.returncode==0,released.stderr
 result['after_release']=subprocess.check_output(['scontrol','show','job','728815','-o'],text=True)
 (b/'preparation/four-shared-release.json').write_text(json.dumps(result));print(json.dumps(result))
'''.replace('NEW',repr(commit)).replace('PARTS',repr(parts)).replace('METADATA',repr(metadata)).replace('SBATCH',repr(SBATCH)),timeout=60)
    (OUTPUT/'publication.json').write_text(json.dumps(pub,indent=2))
    (OUTPUT/'four-shared-deployment.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
