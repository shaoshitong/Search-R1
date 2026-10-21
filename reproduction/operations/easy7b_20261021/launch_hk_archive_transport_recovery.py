"""Single CPU-controller recovery after the preserved terminal SSH failure."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

p = json.load(sys.stdin)
password = p.pop('password')
assert socket.gethostname() == 'xielab3' and os.getuid() == 1002
training_commit = 'e082a23b5a65b302e2016aa0e5ab43b10b543d4f'
job = '745371'
base = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021'
run_id = '232f20e112ac4a978c5aba52a0367195'
suffix = 'transport-r1-20261021'
hk = Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017')
assert re.fullmatch('[0-9a-f]{40}', p['controller_commit'])
source = hk/'source'/p['controller_commit']
publication = json.loads((source/'source-publication.json').read_text())
assert publication['source_commit'] == p['controller_commit']
for name in ('reproduction/easy2560_joint4/archive_controller.py',
             'reproduction/easy2560_joint4/checkpoint_archive.py',
             'reproduction/prefix20_joint4/archive_supervisor.py'):
    assert hashlib.sha256((source/name).read_bytes()).hexdigest() == publication['files'][name]['sha256']
sys.path.insert(0, str(source))
from reproduction.easy2560_joint4.archive_controller import controller_root
old = hk/'archive-controller'/('job-'+job)
root = controller_root(Path(base), training_commit, job, suffix)
assert root == old/('recovery-'+suffix)
failure = json.loads((old/'failure.json').read_text())
assert failure['completed_steps'] == [] and 'SOURCE_OBSERVATION_UNKNOWN' in failure['error']
preserved = {}
for name in ('launch-claim.json', 'claim.json', 'launch-receipt.json',
             'progress.json', 'failure.json', 'controller.stdout.log'):
    raw = (old/name).read_bytes()
    preserved[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
root.mkdir(parents=True, exist_ok=True)
assert not any((root/name).exists() for name in ('launch-claim.json', 'claim.json',
    'launch-receipt.json', 'controller.stdout.log', 'failure.json', 'result.json'))
claim = dict(source_job=job, base=base, run_id=run_id, training_commit=training_commit,
             controller_commit=p['controller_commit'], parent_failure_preserved=preserved,
             cpu_only=True, gpu_allocations=0, credentials_persisted=False,
             requested_steps=[5, 50, 100, 150, 200], created_unix=time.time())
with (root/'launch-claim.json').open('x') as stream:
    json.dump(claim, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
env = dict(os.environ, PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
command = [sys.executable, '-B', '-m', 'reproduction.easy2560_joint4.archive_controller',
           '--base', base, '--commit', training_commit, '--job', job,
           '--first-step', '5', '--controller-suffix', suffix]
with (root/'controller.stdout.log').open('xb') as output:
    child = subprocess.Popen(command, cwd=str(source), env=env, stdin=subprocess.PIPE,
                             stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    child.stdin.write(json.dumps(dict(password=password)).encode()); child.stdin.close()
password = None
fields = Path('/proc/%d/stat' % child.pid).read_text().rsplit(')', 1)[1].split()
birth = dict(pid=child.pid, start_ticks=int(fields[19]), state=fields[0], uid=os.getuid(),
             boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
receipt = dict(claim, root=str(root), source_root=str(source), command=command,
               birth=birth, controllers_started=1, training_jobs_submitted=0,
               training_runtime_edited=False, checked_unix=time.time())
with (root/'launch-receipt.json').open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
for name, record in preserved.items():
    assert hashlib.sha256((old/name).read_bytes()).hexdigest() == record['sha256']
print(json.dumps(receipt))
