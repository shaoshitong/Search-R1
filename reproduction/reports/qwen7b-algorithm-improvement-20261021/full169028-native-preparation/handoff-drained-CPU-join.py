"""Move this exact pending metadata job off drained debug; never load tensors."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

b = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint169028-native-training-bank-v1')
def read(p): return json.loads(p.read_bytes())
def proof(p):
    raw = p.read_bytes()
    return dict(path=str(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
def atomic(p, value):
    tmp = p.with_name(p.name + '.tmp-' + str(os.getpid()))
    with tmp.open('x') as f:
        json.dump(value, f, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(str(tmp), str(p))
guard = (b / 'cpu-submit.lock').open('a+')
fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (b / 'complete.json').exists():
    print(json.dumps(dict(status='ALREADY_COMPLETE', complete=proof(b / 'complete.json')))); raise SystemExit(0)
controller = b / 'login-CPU-controller.json'
if controller.exists():
    old = read(controller); proc = Path('/proc') / str(old['pid'])
    assert proc.exists() and proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] == old['birth']
    assert b'joint169028_bank' in proc.joinpath('cmdline').read_bytes()
    print(json.dumps(dict(status='ADOPT_EXISTING_BOUNDED_METADATA_JOIN', controller=old))); raise SystemExit(0)
actual = subprocess.check_output(['scontrol', 'show', 'job', '-o', '746061'], universal_newlines=True)
for token in ('JobName=q7b-full169028-bank-CPU', 'UserId=mou176(21930)', 'Account=mou176', 'Partition=debug', 'Command=' + str(b / 'join-cpu.sbatch')):
    assert token in actual, token
if 'JobState=RUNNING' in actual:
    print(json.dumps(dict(status='ADOPT_ALREADY_RUNNING_CPU_SLURM_JOIN', job_id='746061'))); raise SystemExit(0)
assert 'JobState=PENDING' in actual
node = subprocess.check_output(['sinfo', '-h', '-N', '-p', 'debug', '-o', '%N|%T'], universal_newlines=True)
assert node.strip().startswith('ACD1-63|') and 'drain' in node.lower()
pub = read(b / 'source-publication.json'); code = Path(pub['code_root'])
assert pub['commit'] == 'ba685344da51f9aaac0bd0d991195ce578fe1758'
for name, ref in pub['files'].items():
    p = proof(code / name); assert p['sha256'] == ref['sha256'] and p['bytes'] == ref['bytes']
config = proof(b / 'config.json')
lock = (b / 'join.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); lock.close()
atomic(b / 'drained-CPU-handoff-intent.json', dict(job_id='746061', actual_job=actual, actual_debug_node=node, config=config, reason='sole_debug_node_drained', only_small_metadata_no_models_or_tensors=True, CPU_limit=2, address_space_GiB_limit=6, epoch=time.time()))
subprocess.check_call(['scontrol', 'hold', '746061'])
held = subprocess.check_output(['scontrol', 'show', 'job', '-o', '746061'], universal_newlines=True)
assert 'JobState=PENDING' in held and 'JobName=q7b-full169028-bank-CPU' in held
subprocess.check_call(['scancel', '746061'])
for _ in range(20):
    state = subprocess.check_output(['squeue', '-h', '-j', '746061', '-o', '%T'], universal_newlines=True)
    if not state.strip(): break
    time.sleep(.2)
assert not state.strip(), 'Do not create a second writer until pending Slurm job is gone'
def bounded():
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 ** 3, 6 * 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    os.nice(10)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
for key in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_JOBID'): env.pop(key, None)
args = ['python3', '-u', '-m', 'reproduction.operations.qwen7b_warmup.joint169028_bank', '--config', config['path'], '--config-sha256', config['sha256']]
with (b / 'login-CPU-join.log').open('ab') as log:
    worker = subprocess.Popen(args, cwd=str(code), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=bounded)
birth = Path('/proc/%d/stat' % worker.pid).read_text().rsplit(')', 1)[1].split()[19]
out = dict(status='BOUNDED_REMOTE_CPU_METADATA_JOIN_STARTED', pid=worker.pid, birth=birth,
    canceled_pending_job='746061', config=config, CPU=2, GPU=0, address_space_limit=6*1024**3,
    CPU_time_limit_seconds=600, model_or_tensor_loaded=False, source_publication=proof(b / 'source-publication.json'), epoch=time.time())
atomic(controller, out); print(json.dumps(out, sort_keys=True))
