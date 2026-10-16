"""Native4 prefix20 startup resources; scientific configuration is unchanged.

The shell entry execs this process, which raises only its own soft NPROC and
then execs the original pipeline. Its descendants inherit the actual limit.
Existing preparation/restore control remains responsible for Slurm admission.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

RAY_CPUS=16
OBJECT_STORE_BYTES=8*1024**3
SOFT_NPROC=8192
RESTORE_ROLE='prefix20_final200_restore_only'


def options(run):
    if (run.get('profile')!='common8' or run.get('execution_site')!='hpc3' or
            run.get('execution_backend','slurm')!='slurm' or run.get('training_world_size')!=4 or
            run.get('data_selection_mode')!='prefix20_probe'):
        raise ValueError('PREFIX20_NATIVE4_RAY_RESOURCE_SCOPE_DIFFERS')
    # One CPU driver plus four colocated GPU bundles and spare register-center
    # scheduling capacity. Slurm still reserves48CPU and4GPU for the job.
    return dict(num_cpus=RAY_CPUS,object_store_memory=OBJECT_STORE_BYTES)


def raise_self_soft(*,getlimit=resource.getrlimit,setlimit=resource.setrlimit):
    """Only this process; never lower soft or alter the actual hard value."""
    before=tuple(getlimit(resource.RLIMIT_NPROC));soft,hard=before
    if hard!=resource.RLIM_INFINITY and hard<SOFT_NPROC:
        raise ValueError('PREFIX20_SELF_NPROC_HARD_LIMIT_BELOW8192: '+repr(before))
    changed=soft!=resource.RLIM_INFINITY and soft<SOFT_NPROC
    if changed:setlimit(resource.RLIMIT_NPROC,(SOFT_NPROC,hard))
    after=tuple(getlimit(resource.RLIMIT_NPROC))
    expected=(SOFT_NPROC,hard) if changed else before
    if after!=expected:
        raise RuntimeError('PREFIX20_SELF_NPROC_ACTUAL_READBACK_DIFFERS: '+repr((before,after)))
    return dict(before=list(before),after=list(after),soft_raised=changed,hard_unchanged=after[1]==before[1],
                scope='current process only; children inherit on fork/exec')


def scope(base,run,role,*,environ=None,uid=None):
    environ=os.environ if environ is None else environ;uid=os.getuid() if uid is None else uid
    from reproduction.prefix20_joint4.control import own_base
    base=own_base(base);kwargs=options(run)
    if (uid!=14090 or environ.get('SEARCH_G1_BASE')!=str(base) or
            environ.get('SEARCH_G1_TRAINING_WORLD_SIZE')!='4' or
            not environ.get('SLURM_JOB_ID','').isdecimal()):
        raise ValueError('PREFIX20_SELF_NPROC_NATIVE4_SLURM_OWNER_REQUIRED')
    if role not in ('train','restore') or (run.get('restore_only_role')==RESTORE_ROLE)!=(role=='restore'):
        raise ValueError('PREFIX20_RAY_RESOURCE_TRAIN_RESTORE_ROLE_DIFFERS')
    # Reuse the existing actual preparation/submission identity, no new GPU or
    # Ray preflight, source scan or scheduler query is introduced here.
    if role=='train':
        prepared=json.loads((base/'logs/prefix20-configuration-prepared.json').read_text())
        job=prepared['actual_slurm_job_binding']['JobId']
        if prepared['run_id']!=run['run_id'] or prepared['source_commit']!=run['source_commit']:
            raise ValueError('PREFIX20_RAY_RESOURCE_PREPARED_RUN_DIFFERS')
    else:
        submitted=json.loads((base/'preparation/restore-submission.json').read_text());job=submitted['job']
        if submitted.get('outcome')!='SUBMITTED' or submitted.get('returncode')!=0:
            raise ValueError('PREFIX20_RAY_RESOURCE_RESTORE_SUBMISSION_DIFFERS')
    if job!=environ['SLURM_JOB_ID']:
        raise ValueError('PREFIX20_RAY_RESOURCE_EXISTING_SLURM_BINDING_DIFFERS')
    return base,kwargs


def configure(base,run,*,role,phase,getlimit=resource.getrlimit,setlimit=resource.setrlimit):
    base,kwargs=scope(base,run,role)
    if phase not in ('entry','ray'):
        raise ValueError('PREFIX20_RAY_RESOURCE_RECORD_PHASE_DIFFERS')
    from reproduction.prefix20_joint4.control import publish
    before=list(getlimit(resource.RLIMIT_NPROC))
    proof=dict(run_id=run['run_id'],source_commit=run['source_commit'],slurm_job_id=os.environ['SLURM_JOB_ID'],
        role=role,phase=phase,pid=os.getpid(),uid=os.getuid(),public_ray_init_kwargs=kwargs,
        requested_soft_nproc=SOFT_NPROC,actual_before=before,checked_unix=time.time(),
        scientific_configuration_changed=False)
    try:proof.update(raise_self_soft(getlimit=getlimit,setlimit=setlimit),accepted=True)
    except BaseException as error:
        proof.update(accepted=False,actual_after=list(getlimit(resource.RLIMIT_NPROC)),
                     error_type=type(error).__name__,error=str(error))
        publish(base/'logs'/('native-ray-resources-'+role+'-'+phase+'-'+str(os.getpid())+'.json'),proof)
        raise
    proof['actual_after']=proof['after']
    publish(base/'logs'/('native-ray-resources-'+role+'-'+phase+'-'+str(os.getpid())+'.json'),proof)
    print('PREFIX20_NATIVE_RAY_RESOURCES='+json.dumps(proof,allow_nan=False),flush=True)
    return kwargs


def command(base,role,python):
    if role=='train':
        return [python,'-m','reproduction.common8.run_pipeline','--base',str(base),'--export-port','18016']
    if role=='restore':
        return [python,'-m','reproduction.prefix20_joint4.restore_only','execute','--base',str(base)]
    raise ValueError('PREFIX20_RAY_RESOURCE_EXEC_ROLE_DIFFERS')


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--role',required=True,choices=('train','restore'));a=p.parse_args()
    base=Path(a.base);run=json.loads((base/'logs/run-manifest.json').read_text())
    configure(base,run,role=a.role,phase='entry')
    argv=command(base,a.role,sys.executable)
    os.execv(sys.executable,argv)


if __name__=='__main__':main()
