"""Launch only the staged verifier, holding a mutex for its entire lifetime.

No checkpoint is opened by this control wrapper and no GPU/Slurm is changed.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
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
guard=(out/'launch.lock').open('a+')
fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
def live(value):
    p=Path('/proc/%d'%value['pid'])
    if not p.exists():return False
    stat=(p/'stat').read_text().rsplit(')',1)[1].split()
    return stat[0]!='Z' and stat[19]==value['start_ticks'] and any(token in(p/'cmdline').read_bytes()for token in(b'accept_full169028_packed_variance.py',b'acceptance.sh',b'runtime.sh'))

previous=[]
for name in('controller.json','common-lock-launch-receipt.json'):
    if(out/name).exists():previous.append(read(out/name))
active=next((value for value in previous if live(value)),None)
if active:
    print(json.dumps(dict(status='EXISTING_ACTUAL_CPU_VALIDATOR_REUSED',controller=active,new_CPU_process_started=False,time=time.time())))
else:
    # Inherit this same open-file-description lock through bash/runtime exec.
    # The verifier retains the descriptor, so disconnecting the short SSH
    # control call cannot release the mutex or terminate a long plan audit.
    shared=common.open('a+')
    try:fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps(dict(status='WAIT_SHARED_CPU_MUTEX_NO_READER_STARTED',shared_global_CPU_lock=str(common),checkpoint_opened=False,time=time.time())))
    else:
        log=out/'common-lock-final-Z-acceptance.log'
        if(out/'run-complete.json').exists():
            old=(out/'run-complete.json').read_bytes(); archive=out/('prior-run-complete-'+hashlib.sha256(old).hexdigest()[:12]+'.json')
            if not archive.exists():archive.write_bytes(old)
        with log.open('ab')as stream:
            child=subprocess.Popen(['bash',str(entry)],stdout=stream,stderr=stream,stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=(shared.fileno(),))
        value=dict(status='COMMON_MUTEX_DURABLE_CPU_VERIFIER_STARTED_NOT_YET_ACCEPTED',pid=child.pid,start_ticks=(Path('/proc/%d/stat'%child.pid)).read_text().rsplit(')',1)[1].split()[19],entry=str(entry),shared_global_CPU_lock=str(common),inherited_lock_fd=shared.fileno(),lock_lifetime='entire_verifier_process',Slurm_mutated=False,GPU_used=False,time=time.time())
        (out/'common-lock-launch-receipt.json').write_text(json.dumps(value,sort_keys=True)+'\n')
        print(json.dumps(value))
    finally:shared.close()
