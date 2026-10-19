"""Own durable submission reconciliation; fitting requires accepted full bank.

No Slurm dependency on another experiment. Never blindly resubmit an unknown
submission. Failed samples preserve sealed batches and are reported for repair.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from .download_model import atomic


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    base=Path(args.base);control=base/'control';control.mkdir(exist_ok=True)
    lock=(control/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    script=Path(__file__).resolve().parent
    atomic(control/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],epoch=time.time(),code_root=str(script.parents[2])))
    for name in ('logs','probe'): (base/name).mkdir(exist_ok=True)
    while True:
        if not (base/'assets/qwen25-7b-instruct-a09a35458c70/complete.json').exists():
            atomic(control/'progress.json',dict(stage='WAITING_FOR_TARGET_FRESH_SHA',epoch=time.time()));time.sleep(30);continue
        assert read(base/'assets/qwen25-7b-instruct-a09a35458c70/complete.json')['status']=='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED'
        break
    if not (base/'config.json').exists():
        subprocess.run(['bash',str(script/'runtime.sh'),'-u','-m','reproduction.operations.qwen7b_warmup.prepare','--base',str(base)],check=True)
    assert read(base/'preparation/cpu-qualification.json')['status']=='CPU_INVARIANTS_PASSED'
    submission=control/'sampling-submission.json'
    if not submission.exists():
        if (control/'sampling-submit-intent.json').exists():
            raise RuntimeError('Previous sampling submission outcome unknown: reconcile squeue/sacct before retry')
        atomic(control/'sampling-submit-intent.json',dict(epoch=time.time(),script=str(script/'sampling.sbatch')))
        partition=read(base/'config.json').get('sampling_partition','acd_u')
        result=subprocess.run(['sbatch','--parsable','--partition='+partition,str(script/'sampling.sbatch')],text=True,capture_output=True)
        assert result.returncode==0,result.stderr
        job=result.stdout.strip().split(';')[0];assert job.isdigit()
        atomic(submission,dict(job_id=job,epoch=time.time(),shards=8,gpus_per_shard=2,partition=partition,qos='normal'))
        print('SAMPLING_ARRAY_ACCEPTED='+job,flush=True)
    while not all((base/'sampling'/('shard-%02d'%i)/'complete.json').exists() for i in range(8)):
        result=subprocess.run(['squeue','-h','-j',read(submission)['job_id'],'-o','%i %T %R'],text=True,capture_output=True)
        if result.returncode==0 and not result.stdout.strip():
            # Accounting can lag: report terminal evidence without resetting work.
            accounting=subprocess.run(['sacct','-n','-X','-j',read(submission)['job_id'],'--format=JobID,State,ExitCode'],text=True,capture_output=True)
            atomic(control/'needs-reconciliation.json',dict(sacct=accounting.stdout,epoch=time.time()))
        atomic(control/'progress.json',dict(stage='WAITING_FOR_EIGHT_G5_SHARDS',squeue=result.stdout,epoch=time.time()))
        time.sleep(60)
    subprocess.run(['bash',str(script/'runtime.sh'),'-u','-m','reproduction.operations.qwen7b_warmup.merge','--base',str(base)],check=True)
    fit=control/'fit-submission.json'
    if not fit.exists():
        if (control/'fit-submit-intent.json').exists():raise RuntimeError('Fit submit outcome unknown: reconcile before retry')
        atomic(control/'fit-submit-intent.json',dict(epoch=time.time()))
        result=subprocess.run(['sbatch','--parsable','--export=ALL,QWEN7B_CODE_ROOT='+str(script.parents[2]),str(script/'fit.sbatch')],text=True,capture_output=True)
        assert result.returncode==0,result.stderr
        job=result.stdout.strip().split(';')[0];assert job.isdigit()
        atomic(fit,dict(job_id=job,gpus=4,epoch=time.time(),bank_accepted=True,code_root=str(script.parents[2])))
        print('FULL_BANK_FIT_JOB_ACCEPTED='+job,flush=True)
    atomic(control/'handoff.json',dict(stage='FIT_SUBMITTED_NOT_YET_VALIDATED',fit=read(fit),epoch=time.time()))


if __name__=='__main__':main()
