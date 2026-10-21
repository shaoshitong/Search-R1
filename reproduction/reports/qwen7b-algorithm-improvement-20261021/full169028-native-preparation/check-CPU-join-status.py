"""Read only actual metadata join and job state; never submits or loads tensors."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

b = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint169028-native-training-bank-v1')
def proof(p):
    raw = p.read_bytes()
    return dict(path=str(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
out = dict(epoch=time.time(), job_id='746061', GPU=0)
out['squeue'] = subprocess.check_output(['squeue', '-h', '-j', '746061', '-o', '%i|%j|%T|%R'], universal_newlines=True)
out['sacct'] = subprocess.check_output(['sacct', '-n', '-P', '-j', '746061', '--format=JobIDRaw,State,ExitCode,Elapsed,AllocCPUS,ReqMem,AllocTRES'], universal_newlines=True)
if 'PENDING' in out['squeue']:
    out['estimated_start'] = subprocess.check_output(['squeue', '--start', '-h', '-j', '746061', '-o', '%i|%T|%S|%R'], universal_newlines=True)
    out['job_configuration'] = subprocess.check_output(['scontrol', 'show', 'job', '-o', '746061'], universal_newlines=True)
for name in ('worker.json', 'progress.json', 'failure.json', 'complete.json'):
    p = b / name
    out[name] = dict(reference=proof(p), value=json.loads(p.read_bytes())) if p.exists() else None
controller = b / 'login-CPU-controller.json'
if controller.exists():
    c = json.loads(controller.read_bytes()); out['bounded_remote_CPU_controller'] = c
    proc = Path('/proc') / str(c['pid'])
    same = proc.exists() and proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] == c['birth']
    out['bounded_remote_CPU_same_process_birth'] = same
    if same:
        out['bounded_remote_CPU_memory'] = [s for s in proc.joinpath('status').read_text().splitlines() if s.startswith(('VmPeak:', 'VmSize:', 'VmRSS:', 'VmHWM:', 'State:', 'Cpus_allowed_list:'))]
    p = b / 'login-CPU-join.log'
    out['bounded_remote_CPU_log_tail'] = p.read_text()[-2200:] if p.exists() else None
if out['complete.json']:
    done = out['complete.json']['value']
    assert done['status'] == 'FROZEN_FIVE_BANK_REAL_G10_169028_NATIVE_JOINT_INPUTS_ACCEPTED'
    for ref in done['outputs'].values():
        p = Path(ref['path'])
        assert proof(p) == ref
    out['small_outputs_fresh_SHA_verified'] = True
else:
    p = b / 'join-cpu-746061.log'
    out['log_tail'] = p.read_text()[-2200:] if p.exists() else None
print(json.dumps(out, sort_keys=True))
