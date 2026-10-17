"""Recheck only the CPU launcher fixture under a live same-UID Python parent."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

COMMIT='d91575dbca221f66c467dccafa8508d9fb703688'
NEW=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730226-20261018')
SOURCE=NEW/'source'/COMMIT
AUDIT=NEW/'preparation/d915-independent-cpu-reader-check-20261018'
TEST='reproduction.common8.test_configuration.ConfigurationTests.test_claimed_launcher_records_exact_exec_arguments_without_starting_trainer'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.getuid()==14090
parent=Path('/proc')/str(os.getpid())
assert parent.exists() and parent.stat().st_uid==os.getuid()
env=dict(os.environ,PYTHONPATH=str(SOURCE),PYTHONDONTWRITEBYTECODE='1')
for key in ('SEARCH_G1_BASE','SEARCH_G1_TRAINING_WORLD_SIZE','SLURM_JOB_ID','SLURM_JOBID'):
    env.pop(key,None)
code="""import os,sys,unittest
from pathlib import Path
source=Path(SOURCE_VALUE);sys.path[:0]=[str(source)]
parent=Path('/proc')/str(os.getppid())
assert parent.exists() and parent.stat().st_uid==os.getuid()
import reproduction.common8.test_configuration as module
assert Path(module.__file__).is_relative_to(source)
suite=unittest.defaultTestLoader.loadTestsFromName(TEST_VALUE)
result=unittest.TextTestRunner(verbosity=2).run(suite)
import torch
assert not torch.cuda.is_initialized()
sys.exit(0 if result.wasSuccessful() else 1)
""".replace('SOURCE_VALUE',repr(str(SOURCE))).replace('TEST_VALUE',repr(TEST))
# The environment wrapper supplies the published package roots; its underlying
# sys.executable is a shared bare interpreter and omits those roots in a child.
python='/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018/env/bin/python'
child=subprocess.run([python,'-B','-c',code],env=env,cwd=SOURCE,capture_output=True,text=True,timeout=90)
row=dict(status='VISIBLE_PARENT_FIXTURE_PASS' if child.returncode==0 else 'VISIBLE_PARENT_FIXTURE_FAILED',
    returncode=child.returncode,stdout=child.stdout,stderr=child.stderr,test=TEST,source_root=str(SOURCE),
    operator_commit=COMMIT,same_uid_parent_pid=os.getpid(),same_uid_parent_visible=True,
    cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],actual_cuda_restore_accepted=False,
    original_39_test_result_preserved=True,training_modified=False,checked_unix=time.time())
path=AUDIT/'visible-parent-fixture-wrapper-retest.json'
with path.open('x') as output:
    json.dump(row,output,sort_keys=True,indent=2);output.write('\n');output.flush();os.fsync(output.fileno())
print(json.dumps(dict(receipt=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**row),sort_keys=True))
sys.exit(child.returncode)
