"""Test controller repairs in an isolated HPC3 CPU overlay, never hot-edit runtime."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile
import time

from .local_control import run
from .control import BASE,OUTPUT

FILES=['reproduction/prefix_selection200/'+name for name in (
    'i1_i6_pending_probe.py','test_i1_i6_pending_probe.py','i1_i6_joint_resume.py','launch.py')]
REMOTE=r'''
import sys,json,io,tarfile,hashlib,pathlib,os,time,unittest
m=json.loads(sys.stdin.buffer.readline());raw=sys.stdin.buffer.read()
assert hashlib.sha256(raw).hexdigest()==m['archive_sha']
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017')
root=b/'preparation'/('pending-recovery-cpu-'+str(time.time_ns()));root.mkdir()
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 for member in tar.getmembers():
  name=pathlib.PurePosixPath(member.name)
  assert member.isfile() and '..' not in name.parts and not name.is_absolute()
  dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
  dest.write_bytes(tar.extractfile(member).read())
  assert hashlib.sha256(dest.read_bytes()).hexdigest()==m['files'][member.name]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TMPDIR']=str(b/'tmp')
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.path.insert(0,str(b/'repo'))
import reproduction.prefix_selection200 as package
package.__path__.insert(0,str(root/'reproduction/prefix_selection200'))
import torch;torch.set_num_threads(2)
names=['reproduction.prefix_selection200.test_'+x for x in (
 'i1_i6_pending_probe','actor_reference_contract','adaptive_candidates','candidate_schedule',
 'grounding_protocol','interaction_protocol','period10_halfpool','prefix5_capture_plan',
 'actual_em_selection','storage_guard_boundary','zero_anchor_advantages','eval')]
names+=['reproduction.hpc3_prefix5.test_cpu_integration']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
with (root/'tests.log').open('w') as log:
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
value=dict(epoch=time.time(),tests=result.testsRun,
 failures=[dict(test=str(t),trace=e) for t,e in result.failures],
 errors=[dict(test=str(t),trace=e) for t,e in result.errors],skipped=result.skipped,
 success=result.wasSuccessful(),scope='CPU regressions and constructed pending/import/commit/Adam-RNG faults; native GPU and genuine full-state qualification remain unverified',
 runtime_modified=False,snapshot=str(root),files=m['files'])
(root/'test-result.json').write_text(json.dumps(value,sort_keys=True))
print(json.dumps(value));raise SystemExit(0 if result.wasSuccessful() else 1)
'''


def main():
    root=Path(__file__).resolve().parents[2];stream=io.BytesIO();hashes={}
    with tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for name in FILES:
            raw=(root/name).read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
            member=tarfile.TarInfo(name);member.size=len(raw);archive.addfile(member,io.BytesIO(raw))
    data=stream.getvalue();metadata=dict(archive_sha=hashlib.sha256(data).hexdigest(),files=hashes)
    r=run(BASE+'/env/bin/python -c '+shlex.quote(REMOTE),json.dumps(metadata).encode()+b'\n'+data)
    value=dict(returncode=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode(),epoch=time.time())
    (OUTPUT/'pending-recovery-cpu-tests-final.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(value));raise SystemExit(r.returncode)


if __name__=='__main__':main()
