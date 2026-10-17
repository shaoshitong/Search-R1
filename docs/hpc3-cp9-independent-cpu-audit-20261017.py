"""Submit one CPU-only prequalification without modifying the GPU continuation.

Use the published R2 auditor and a separate proof/intent/log. Never overwrite
the continuation's proof, alter its dependency, or request a GPU for this audit.
Credentials stay in the existing RAM/PTY adapter. Run from this checkout.
"""
import json
from pathlib import Path
import shlex
import subprocess

from reproduction.hpc3_prefix5.control import BASE, OUTPUT, PYTHON
from reproduction.hpc3_prefix5.direct_control import run


def main():
    target = BASE + '-r2'
    audit = BASE + '/preparation/cp9-independent-cpu-prequalification.json'
    program = '''import json,pathlib
from reproduction.prefix_selection200.i1_i6_checkpoint_cpu import qualify
b=pathlib.Path(TARGET)
s=json.loads((b/'preparation/resume-config-template.json').read_text())['resume_joint']
v=qualify(pathlib.Path(SOURCE),9,pathlib.Path(s['terminal_proof']),s['terminal_proof_sha256'],pathlib.Path(OUTPUT))
keys=['status','completed_outer','actual_adam_steps_per_rank','probe_adam_steps_per_rank','actor_rank_optimizer_rng_scheduler_verified','payload_full_sha_read_once','actual_cpu_deserialization','cuda_restore_verified','completed_unix']
print(json.dumps({k:v.get(k) for k in keys}),flush=True)
'''.replace('TARGET', repr(target)).replace('SOURCE', repr(BASE)).replace('OUTPUT', repr(audit))
    batch = '''#!/bin/bash
#SBATCH --job-name=prefix5-thread5-cp9-cpu
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
set -euo pipefail
base=TARGET
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export HF_HOME="$base/cache/hf"
export XDG_CACHE_HOME="$base/cache"
export TMPDIR="$base/tmp"
cd "$base/repo"
exec "$base/env/bin/python" -c PROGRAM
'''.replace('TARGET', shlex.quote(target)).replace('PROGRAM', shlex.quote(program))
    remote = '''import pathlib,json,subprocess,time,os,hashlib
b=pathlib.Path(TARGET);p=b/'preparation'
receipt=p/'cp9-independent-cpu-submission.json';intent=p/'cp9-independent-cpu-submit-intent.json'
if receipt.exists():
 print(json.dumps(json.loads(receipt.read_text())));raise SystemExit(0)
if intent.exists():raise RuntimeError('Reconcile the prior CPU submission before another attempt')
queue=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%j|%i|%T'],text=True)
assert not any(line.split('|')[0]=='prefix5-thread5-cp9-cpu' for line in queue.splitlines())
template=json.loads((p/'resume-config-template.json').read_text())
assert template['resume_joint']['completed_outer']==9
gpu_job=str(json.loads((p/'slurm-submission.json').read_text())['job'])
live=subprocess.check_output(['squeue','-h','-j',gpu_job,'-o','%i|%T'],text=True).strip()
if live!=gpu_job+'|PENDING':
 print(json.dumps(dict(status='SKIPPED_GPU_NO_LONGER_PENDING',actual_queue=live)));raise SystemExit(0)
script=p/'cp9-independent-cpu.sbatch';script.write_text(BATCH);script.chmod(0o700)
with intent.open('x') as f:json.dump(dict(created_unix=time.time(),script=str(script),gpu_count=0,source_outer=9),f)
r=subprocess.run(['sbatch','--parsable','--output',str(b/'logs/cp9-independent-cpu-%j.log'),str(script)],capture_output=True,text=True,timeout=20)
assert r.returncode==0,(r.stdout,r.stderr)
job=r.stdout.strip().split(';')[0];assert job.isdigit(),r.stdout
v=dict(job=job,base=str(b),partition='debug',physical_gpus=0,cpus=2,memory='32G',proof=AUDIT,submitted_unix=time.time(),gpu_continuation_job=gpu_job,gpu_continuation_unchanged=True,script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
with receipt.open('x') as f:json.dump(v,f)
print(json.dumps(v))
'''.replace('TARGET', repr(target)).replace('BATCH', repr(batch)).replace('AUDIT', repr(audit))
    address = subprocess.check_output(['ipconfig', 'getifaddr', 'en0'], text=True).strip()
    result = run(PYTHON + ' -c ' + shlex.quote(remote), account='mou176', source_address=address, timeout=40)
    if result.returncode:
        raise RuntimeError(result.stdout.decode()[-4000:] + result.stderr.decode()[-1000:])
    value = next(json.loads(line) for line in result.stdout.decode().splitlines() if line.strip().startswith('{'))
    (Path(OUTPUT) / 'cp9-independent-cpu-submission.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
