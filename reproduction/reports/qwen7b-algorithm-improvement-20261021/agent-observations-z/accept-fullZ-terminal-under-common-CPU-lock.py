"""Run only the already-staged verifier, with a mutex for its entire lifetime.

No checkpoint is opened by this control wrapper and no GPU/Slurm is changed.
"""
from pathlib import Path
import json
import subprocess
import time

A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
base=A/'online-packed-full169028-M3to10-P20-20261022-v1/z_mse_anchor'
out=A/'packed-full169028-variance-CPU-acceptance-20261022-v1'
common=A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1/acceptance.lock'
entry=out/'validator-799e75f35347/acceptance.sh'
def read(p):return json.loads(p.read_bytes())

assert common.parent.is_dir() and entry.is_file()
complete=read(base/'complete.json')
assert (complete['epochs'],complete['cursor'],complete['training_questions'])==(10,13210,169028)
assert str(read(base/'root-submission.json')['job_id'])==str(read(base/'controller.json')['job_id'])=='749577'
accounting=subprocess.check_output(['sacct','-X','-n','-P','-j','749577','--format=JobIDRaw,State,ExitCode'],universal_newlines=True)
assert any(row.split('|')[:3]==['749577','COMPLETED','0:0'] for row in accounting.splitlines())
started=time.time(); log=out/'common-lock-final-Z-acceptance.log'
with log.open('ab') as stream:
    result=subprocess.run(['flock','-n',str(common),'bash',str(entry)],stdout=stream,stderr=stream,timeout=120)
if result.returncode:
    print(json.dumps(dict(status='CPU_VERIFIER_NOT_ACCEPTED_CHECK_SHARED_LOCK_OR_LOG',returncode=result.returncode,log=str(log),shared_global_CPU_lock=str(common),checkpoint_opened_by_control=False,time=time.time())))
else:
    accepted=read(out/'z_mse_anchor/accepted.json')
    value=dict(status='FULL_Z_TERMINAL_CPU_VERIFIER_FINISHED_UNDER_COMMON_MUTEX',accepted=accepted,shared_global_CPU_lock=str(common),lock_held_for_entire_foreground_verifier=True,seconds=time.time()-started,Slurm_mutated=False,GPU_used=False,time=time.time())
    (out/'common-lock-final-Z-receipt.json').write_text(json.dumps(value,sort_keys=True)+'\n')
    print(json.dumps(value))
