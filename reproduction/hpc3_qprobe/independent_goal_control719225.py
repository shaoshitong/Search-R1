"""Hold one final pure CPU consumer across SSH observation timeouts.

The final native GPU audit remains owned by the existing EA follower. This
wrapper can only run the independent evidence reader, after all200 and both
existing followers have actually finished. It never submits or alters jobs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from reproduction.hpc3_qprobe import final_control as ctl
from reproduction.hpc3_qprobe import independent_goal_acceptance719225 as reader

DIRECTORY=ctl.ROOT/'observations/independent-goal-final719225-20261015'
MODULE='reproduction.hpc3_qprobe.independent_goal_control719225'
CONSUMER='reproduction.hpc3_qprobe.independent_goal_acceptance719225'
RAW_STATUS='ACTUAL_RAW_ALL200_AND_ALL_NORMAL_RETURN_ATTEMPTS_PASS'


def caps():
    ctl.owner()
    for name in reader.CAPS:os.environ[name]='1'
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    sys.dont_write_bytecode=True


def ready():
    # A missing200 refuses before any directory, claim or process is created.
    _,_,history=reader.early_history()
    producer=reader.completed('719225')
    ea_dir=ctl.ROOT/'followthrough/719225'
    ea=ctl.read(ea_dir/'result.json');process=ctl.read(ea_dir/'process.json')
    ctl.need(type(ea['exit_code']) is int and ea['exit_code']==0 and ea['error'] is None and
        reader.birth_ok(process['actual_process']) and ea['actual_process']==process['actual_process'] and
        not ctl.live(process['actual_process']) and
        ea['result']['phase']=='ACTUAL_FULL_CPU_GOAL_CONSUMER_PASS' and
        ea['result']['actual_final_native_restore_accepted'] is True and
        ea['result']['full_objective_accepted'] is True,'existing EA final CPU consumer is not terminal PASS')
    raw_dir=ctl.ROOT/'observations/raw-all200-719225-20261015'
    raw=ctl.read(raw_dir/'result.json');raw_process=ctl.read(raw_dir/'process.json')
    ctl.need(raw['status']==RAW_STATUS and reader.birth_ok(raw_process) and raw['process']==raw_process and
        raw['successful_formal_outers']==list(range(1,201)) and raw['normal_return_attempts']>=200 and not ctl.live(raw_process),
        'existing all200 raw consumer is not terminal PASS')
    binding=ctl.read(ea_dir/'final-consumer-binding.json')
    audit_base=Path(binding['audit_base'])
    ctl.need(audit_base.is_absolute() and audit_base.resolve()==audit_base and
        audit_base.parent==ctl.ROOT.parent,'unsafe existing final native audit base')
    native=ctl.read(audit_base/'logs/final-restore-readback.json')
    native_accounting=reader.completed(native['slurm_job_id'])
    return dict(history=history,producer_accounting=producer,native_accounting=native_accounting,
        ea_result_sha256=ctl.digest(ea_dir/'result.json'),raw_result_sha256=ctl.digest(raw_dir/'result.json'),
        ea_consumer_binding_sha256=ctl.digest(ea_dir/'final-consumer-binding.json'),
        checked_unix=time.time())


def accept_output(proof,commit):
    ctl.need(proof.get('status')==reader.STATUS and proof.get('run_id')==reader.RUN and
        proof.get('source',{}).get('commit')==commit and proof.get('completed_outer')==200 and
        proof.get('world_size')==4 and proof.get('actual_native_two_readbacks_reconsumed') is True and
        proof.get('update201_performed') is False and proof.get('goal_complete') is False and
        proof.get('actor_model_loaded') is False and all(type(proof.get(key)) is int and proof[key]==0
            for key in ('receipt_or_cache_or_claim_writes','training_or_helper_or_native_job_submissions',
                        'generated_trajectories','new_actor_head_or_scheduler_updates')),
        'actual final independent consumer stdout is not the full pure200 conjunction')


def consume(commit):
    _,publication=ctl.source(commit)
    claim=ctl.read(DIRECTORY/'claim.json');process=ctl.read(DIRECTORY/'process.json')
    result=ctl.read(DIRECTORY/'result.json');child=ctl.read(DIRECTORY/'child.json')
    ctl.need(claim['source']==publication and claim['source']['commit']==commit and result['source']==claim['source'] and
        result['claim_sha256']==ctl.digest(DIRECTORY/'claim.json') and result['process']==process and
        result['child']==child and reader.birth_ok(process) and reader.birth_ok(child) and
        process['uid']==child['uid']==ctl.OWNER and
        process['pid']==process['sid']==child['sid'] and child['pid']!=process['pid'] and
        result['completed_unix']>=result['started_unix']>=claim['created_unix'] and
        type(result['actual_child_exit_code']) is int and result['actual_child_exit_code']==0 and
        type(result['supervisor_exit_code']) is int and result['supervisor_exit_code']==0 and
        result['error'] is None and not ctl.live(child),'actual held final consumer wait/source/birth differs')
    for name in ('stdout','stderr'):
        path=DIRECTORY/('consumer.'+name);row=result[name]
        ctl.need(row['path']==str(path) and row['bytes']==path.stat().st_size and
            row['sha256']==ctl.digest(path),'actual final consumer output changed')
    ctl.need(result['stderr']['bytes']==0 and result['stdout']['bytes']<=32*2**20,
        'final consumer errors or unbounded stdout')
    proof=ctl.read(DIRECTORY/'consumer.stdout');accept_output(proof,commit)
    return dict(status='ACTUAL_HELD_INDEPENDENT_FINAL200_CONSUMER_COMPLETED0',
        result_path=str(DIRECTORY/'result.json'),result_sha256=ctl.digest(DIRECTORY/'result.json'),
        stdout_path=str(DIRECTORY/'consumer.stdout'),stdout_sha256=result['stdout']['sha256'],
        actual_child_exit_code=0,consumer_status=proof['status'],completed_outer=200,world_size=4,
        goal_complete=False,scope='root must review the full stdout before marking its goal complete')


def inspect(commit):
    ctl.owner()
    if not DIRECTORY.exists():return dict(status='NOT_STARTED',directory=str(DIRECTORY),goal_complete=False)
    ctl.need(DIRECTORY.resolve()==DIRECTORY and DIRECTORY.stat().st_uid==ctl.OWNER,
        'noncanonical final observer directory')
    claim=ctl.read(DIRECTORY/'claim.json')
    ctl.need(claim['source']['commit']==commit,'existing final consumer source differs; inspect original source, never replay')
    process=ctl.read(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else None
    child=ctl.read(DIRECTORY/'child.json') if (DIRECTORY/'child.json').exists() else None
    live=process is not None and ctl.live(process)
    terminal=ctl.read(DIRECTORY/'result.json') if (DIRECTORY/'result.json').exists() else None
    sizes={}
    for name in ('stdout','stderr'):
        path=DIRECTORY/('consumer.'+name)
        if path.exists():sizes[name]=path.stat().st_size
    if terminal is not None and terminal['supervisor_exit_code']==0 and not live:
        return dict(consume(commit),actual_supervisor_live=False,process=process,child=child,
            output_bytes=sizes,checked_unix=time.time())
    return dict(status=('ACTUAL_INDEPENDENT_FINAL_CPU_CONSUMER_LIVE' if live else
        'ACTUAL_FINAL_CONSUMER_FAILED_INSPECT' if terminal is not None else 'UNKNOWN_FINAL_CONSUMER_NO_REPLAY'),
        actual_supervisor_live=live,actual_child_live=ctl.live(child) if child else False,
        process=process,child=child,result=terminal,output_bytes=sizes,goal_complete=False,
        checked_unix=time.time())


def start(commit):
    caps()
    if DIRECTORY.exists():return inspect(commit)
    admission=ready()
    repository,publication=ctl.source(commit)
    os.environ['QPROBE_SOURCE']=str(repository)
    # Exclusive once scope is created only after actual terminal readiness.
    try:DIRECTORY.mkdir(mode=0o770)
    except FileExistsError:return inspect(commit)
    command=[sys.executable,'-m',MODULE,'supervise','--commit',commit]
    ctl.publish(DIRECTORY/'claim.json',dict(source=publication,admission=admission,
        cwd=str(repository),command=command,created_unix=time.time(),training_control_permitted=False))
    with (DIRECTORY/'supervisor.stdout').open('xb') as out,(DIRECTORY/'supervisor.stderr').open('xb') as err:
        process=subprocess.Popen(command,cwd=str(repository),stdout=out,stderr=err,start_new_session=True)
    birth=ctl.birth(ctl.process(process.pid))
    ctl.need(birth['pid']==birth['sid'] and birth['uid']==ctl.OWNER,'actual detached observer identity differs')
    ctl.publish(DIRECTORY/'process.json',birth)
    return dict(status='ACTUAL_FINAL_CPU_OBSERVER_STARTED_NOT_ACCEPTED',process=birth,
        directory=str(DIRECTORY),goal_complete=False)


def supervise(commit):
    caps();repository,publication=ctl.source(commit)
    os.environ['QPROBE_SOURCE']=str(repository)
    own=ctl.birth(ctl.process(os.getpid()));started=time.time();child=None;child_code=None;error=None;code=1;process=None
    try:
        deadline=time.monotonic()+15
        while not (DIRECTORY/'process.json').exists():
            ctl.need(time.monotonic()<deadline,'launcher process binding missing; inspect without replay')
            time.sleep(.05)
        claim=ctl.read(DIRECTORY/'claim.json')
        ctl.need(ctl.read(DIRECTORY/'process.json')==own and claim['source']==publication and
            claim['cwd']==str(repository),'actual supervisor claim/source/birth differs')
        command=[sys.executable,'-m',CONSUMER,'--commit',commit]
        with (DIRECTORY/'consumer.stdout').open('xb') as out,(DIRECTORY/'consumer.stderr').open('xb') as err:
            process=subprocess.Popen(command,cwd=str(repository),stdout=out,stderr=err)
            child=ctl.birth(ctl.process(process.pid))
            ctl.need(child['uid']==ctl.OWNER and child['sid']==own['sid'] and child['pid']!=own['pid'],
                'actual pure CPU child identity differs')
            ctl.publish(DIRECTORY/'child.json',child)
            child_code=process.wait()
            for stream in (out,err):stream.flush();os.fsync(stream.fileno())
        ctl.need(child_code==0 and not ctl.live(child),'actual final consumer did not return0')
        ctl.need((DIRECTORY/'consumer.stderr').stat().st_size==0,'actual final consumer stderr is nonempty')
        ctl.need((DIRECTORY/'consumer.stdout').stat().st_size<=32*2**20,'unbounded final consumer stdout')
        proof=ctl.read(DIRECTORY/'consumer.stdout');accept_output(proof,commit);code=0
    except Exception as exc:
        error=repr(exc)
        import traceback
        traceback.print_exc()
        # A binding/output error after spawn does not abandon an actual child
        # or turn an observation failure into permission to launch another.
        if process is not None and child_code is None:child_code=process.wait()
    outputs={}
    for name in ('stdout','stderr'):
        path=DIRECTORY/('consumer.'+name)
        if path.exists():
            with path.open('rb') as stream:os.fsync(stream.fileno())
            outputs[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=ctl.digest(path))
    claim=ctl.read(DIRECTORY/'claim.json')
    ctl.publish(DIRECTORY/'result.json',dict(source=publication,claim_sha256=ctl.digest(DIRECTORY/'claim.json'),
        process=own,child=child,actual_child_exit_code=child_code,supervisor_exit_code=code,error=error,
        started_unix=started,completed_unix=time.time(),goal_complete=False,**outputs))
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('start','supervise','inspect'))
    parser.add_argument('--commit',required=True);args=parser.parse_args();caps()
    if args.mode=='supervise':sys.exit(supervise(args.commit))
    print(json.dumps(start(args.commit) if args.mode=='start' else inspect(args.commit),allow_nan=False),flush=True)
