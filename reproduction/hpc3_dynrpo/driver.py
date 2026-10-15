"""One preparation allocation, then one independent four-GPU formal job."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import BASE, PIN, own, read, sha, publish, progress, verify_source


def submit(phase, commit):
    own(BASE); verify_source(commit)
    record=BASE/'preparation'/(phase+'-submission.json')
    claim=BASE/'preparation'/(phase+'-submit-claim.json')
    if record.exists():
        previous=read(record)
        if previous['commit']!=commit or previous['returncode']!=0:
            raise ValueError('Existing submission requires reconciliation, not another submit')
        return previous
    if claim.exists():
        raise ValueError('Submission outcome unknown: query the existing job name/base before retry')
    if phase not in ('preparation','formal'): raise ValueError('Unknown phase')
    if phase=='formal':
        ready=read(BASE/'warmup/ready.json')
        if ready['head_adam_steps']!=200 or sha(BASE/'warmup/fresh-head.pt')!=ready['head_sha256']:
            raise ValueError('Cannot submit formal training before the own accepted warmup')
    gpus,cpus,memory=(1,12,'128G') if phase=='preparation' else (4,48,'512G')
    command=['sbatch','--parsable','--partition=emergency_acd','--nodes=1','--ntasks=1',
        '--gres=gpu:'+str(gpus),'--cpus-per-task='+str(cpus),'--mem='+memory,'--time=7-00:00:00',
        '--job-name=dynrpo-native5-'+phase,'--export=NONE','--chdir='+str(BASE),
        '--output='+str(BASE/'logs'/(phase+'-%j.log')),'--error='+str(BASE/'logs'/(phase+'-%j.log')),
        str(BASE/'source'/commit/'reproduction/hpc3_dynrpo/run.sh'),phase,str(BASE),commit]
    # The native runtime binds fixed loopback service ports. Use another node
    # than the observed other QProbe experiment, rather than touching its ports.
    excluded=read(BASE/'preparation/deployment-plan.json').get('excluded_service_nodes',[])
    if phase=='formal' and excluded:
        command.insert(2,'--exclude='+','.join(excluded))
    publish(claim,dict(phase=phase,commit=commit,base=str(BASE),command=command,created_unix=time.time(),
                      dependency_on_other_experiment=None),once=True)
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    value=dict(phase=phase,commit=commit,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,checked_unix=time.time())
    if result.returncode==0:
        job=result.stdout.strip().split(';')[0]
        if not job.isdecimal(): raise ValueError('Ambiguous submission response: reconcile the existing claim')
        value['job']=job
    publish(record,value,once=True)
    if result.returncode: raise RuntimeError(result.stderr)
    progress(phase.upper()+'_SUBMITTED',job_id=value['job'],gpus=gpus,formal_actor_outers=0)
    return value


def run_preparation(commit):
    own(BASE); verify_source(commit)
    lock=(BASE/'preparation/preparation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    controller=dict(pid=os.getpid(),uid=os.getuid(),job=os.environ['SLURM_JOB_ID'],commit=commit,started_unix=time.time())
    publish(BASE/'preparation'/('preparation-controller-'+os.environ['SLURM_JOB_ID']+'.json'),controller,once=True)
    publish(BASE/'preparation/preparation-controller.json',controller)
    # These tests instantiate the actual native HF layer, but cannot see CUDA.
    test_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    modules=['search_g1.test_qprobe_qwen_block_model','search_g1.test_qprobe_qwen_block_online',
             'search_g1.test_qprobe_qwen_runtime','reproduction.hpc3_dynrpo.test_contract']
    subprocess.run([sys.executable,'-m','unittest',*modules],env=test_env,check=True)
    publish(BASE/'preparation/native-cpu-tests.json',dict(passed=True,modules=modules,commit=commit,
        cuda_visible_devices='',checked_unix=time.time()))
    for module in ('features','warmup'):
        progress(module.upper()+'_PROCESS_STARTING',formal_actor_outers=0)
        process=subprocess.Popen([sys.executable,'-u','-m','reproduction.hpc3_dynrpo.'+module])
        code=process.wait()
        exit_record=dict(pid=process.pid,returncode=code,job=os.environ['SLURM_JOB_ID'],checked_unix=time.time())
        publish(BASE/'preparation'/(module+'-process-exit-'+os.environ['SLURM_JOB_ID']+'.json'),exit_record,once=True)
        publish(BASE/'preparation'/(module+'-process-exit.json'),exit_record)
        if code: raise RuntimeError(module+' exited '+str(code)+'; partial state preserved')
    submit('formal',commit)
    progress('PREPARATION_COMPLETED_FORMAL_QUEUED',warmup_adam_steps=200,formal_actor_outers=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('submit-preparation','prepare','bind'))
    p.add_argument('--commit',required=True);a=p.parse_args()
    if a.action=='submit-preparation': print(submit('preparation',a.commit))
    elif a.action=='prepare':run_preparation(a.commit)
    else:
        verify_source(a.commit)
        from .bind import main as bind
        bind()


if __name__=='__main__':
    main()
