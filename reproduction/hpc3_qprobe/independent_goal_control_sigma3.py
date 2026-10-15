"""Once held CPU consumer of terminal sigma3 evidence, never a job controller.

Missing200 refuses before creating any scope. Existing scopes are inspected,
never replayed. Heavy hashes run in the retained child across SSH timeouts.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from reproduction.hpc3_qprobe import final_control as ctl
from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as reader

DIRECTORY=ctl.ROOT/'observations/independent-goal-final-sigma3-721485-20261015'
MODULE='reproduction.hpc3_qprobe.independent_goal_control_sigma3'
CONSUMER='reproduction.hpc3_qprobe.independent_goal_acceptance_sigma3'
HELD_STATUS='ACTUAL_HELD_INDEPENDENT_SIGMA3_FINAL200_CONSUMER_COMPLETED0'
RAW_STATUS='ACTUAL_RAW_ALL200_AND_ALL_NORMAL_RETURN_ATTEMPTS_PASS'


def scope(recovery_commit=None):
    return DIRECTORY if recovery_commit is None else DIRECTORY.with_name(DIRECTORY.name+'-archive5-recovery')


def recovery_args(recovery_commit):
    return [] if recovery_commit is None else ['--recovery-commit',recovery_commit]


def caps():
    ctl.owner()
    for name in reader.CAPS:os.environ[name]='1'
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    sys.dont_write_bytecode=True


def source_header(commit):
    """Small admission anchor; the detached consumer verifies every source byte."""
    path=ctl.ROOT/'preparation'/('source-'+commit+'-verification.json')
    proof,record=reader.document(path)
    ctl.need(proof.get('commit')==commit and proof.get('status')=='PUBLISHED_SOURCE_FULL_SHA_PASS' and
        isinstance(proof.get('files'),dict) and proof['files'],'source publication header is absent')
    return dict(commit=commit,publication_path=record['path'],publication_sha256=record['sha256'])


def ready(followthrough_commit, recovery_commit=None):
    _,history=reader.early_history()  # No source traversal or scope before200.
    filename,folder=reader.followthrough_route(followthrough_commit,recovery_commit)
    accounting,binding=reader.producer_acceptance()
    source=source_header(followthrough_commit);raw_source=source_header(reader.RAW)
    follow,follow_file=reader.document(folder/'result.json');follow_process,_=reader.document(folder/'process.json')
    follow_claim,_=reader.document(folder/'claim.json');consumer,consumer_file=reader.document(folder/'final-consumer-binding.json')
    goal,goal_file=reader.document(reader.BASE/'logs/qprobe-final-goal-acceptance.json')
    actual=follow_process.get('actual_process');returned=follow.get('result') or {}
    _,settings_file=reader.document(reader.BASE/'preparation/sigma3-experiment.json')
    expected_binding=dict(binding,sigma3_settings_sha256=settings_file['sha256'])
    ctl.need(type(follow.get('exit_code')) is int and follow['exit_code']==0 and follow.get('error') is None and
        reader.birth_ok(actual) and actual['pid']==actual['sid'] and not ctl.live(actual) and
        follow.get('actual_process')==actual and follow_process.get('commit')==followthrough_commit and
        follow_claim.get('source')==source and follow_claim.get('binding')==expected_binding and
        consumer.get('actual_process')==actual and consumer.get('helper_commit')==followthrough_commit and
        consumer.get('full_cpu_consumer_returned') is True and consumer.get('goal_sha256')==goal_file['sha256'] and
        returned.get('phase')=='ACTUAL_FULL_CPU_GOAL_CONSUMER_PASS' and returned.get('full_objective_accepted') is True and
        returned.get('actual_final_native_restore_accepted') is True and
        returned.get('final_goal_path')==goal_file['path'] and returned.get('final_goal_sha256')==goal_file['sha256'] and
        goal.get('status')==reader.GOAL_STATUS and goal.get('run_id')==reader.RUN and goal.get('completed_outer')==200 and
        goal.get('update201_performed') is False,'sigma3 followthrough lacks its actual terminal CPU return')
    recovery=None
    if recovery_commit is not None:
        recovered,_=reader.load_fixed(recovery_commit,filename,'control')
        recovery=recovered.consume(recovery_commit)
    ctl.need(follow_claim.get('recovery')==recovery,'sigma3 archive5 recovery intent differs')
    audit_base,native,_,_,_=reader.native_evidence(goal,consumer)
    result,result_file=reader.document(reader.RAW_DIRECTORY/'result.json')
    process,process_file=reader.document(reader.RAW_DIRECTORY/'process.json')
    claim,claim_file=reader.document(reader.RAW_DIRECTORY/'claim.json')
    ctl.need(result['status']==RAW_STATUS and result['source']==claim['source']==raw_source and
        reader.birth_ok(process) and process['pid']==process['sid'] and result['process']==process and
        result['claim_sha256']==claim_file['sha256'] and not ctl.live(process) and
        result['successful_formal_outers']==list(range(1,201)) and result['normal_return_attempts']>=200,
        'all200 raw consumer is not actual terminal PASS')
    return dict(history=history,producer_binding=binding,producer_accounting=accounting,recovery_commit=recovery_commit,archive5_recovery=recovery,
        followthrough_source=source,followthrough_result_sha256=follow_file['sha256'],
        goal_sha256=goal_file['sha256'],followthrough_binding_sha256=consumer_file['sha256'],
        raw_source=raw_source,raw_result_sha256=result_file['sha256'],raw_claim_sha256=claim_file['sha256'],
        raw_process=process_file,audit_base=str(audit_base),native_job=native['slurm_job_id'],checked_unix=time.time())


def accept_output(proof,commit,followthrough_commit,admission,recovery_commit=None):
    reader.followthrough_route(followthrough_commit,recovery_commit)
    ctl.need(proof.get('status')==reader.STATUS and proof.get('run_id')==reader.RUN and
        proof.get('producer_job')==reader.JOB and proof.get('producer_commit')==reader.PRODUCER and
        proof.get('source',{}).get('commit')==commit and proof.get('followthrough_commit')==followthrough_commit and
        proof.get('recovery_commit')==recovery_commit and proof.get('completed_outer')==200 and proof.get('world_size')==4 and
        proof.get('current_history')==dict(first=1,last=200,fresh_official_actor=True) and
        proof.get('actual_native_two_readbacks_reconsumed') is True and proof.get('update201_performed') is False and
        proof.get('goal_complete') is False and proof.get('actor_model_loaded') is False and
        all(type(proof.get(key)) is int and proof[key]==0 for key in (
            'receipt_or_cache_or_claim_writes','training_or_helper_or_native_job_submissions',
            'generated_trajectories','new_actor_head_or_scheduler_updates')),
        'independent stdout is not the full pure sigma3 conjunction')
    follow=proof.get('actual_followthrough_full_CPU_consumer') or {}
    raw=proof.get('independent_all_normal_return_raw') or {}
    ctl.need(isinstance(admission,dict) and proof.get('producer_binding')==admission['producer_binding'] and
        follow.get('source')==admission['followthrough_source'] and
        follow.get('result',{}).get('sha256')==admission['followthrough_result_sha256'] and
        follow.get('goal',{}).get('sha256')==admission['goal_sha256'] and
        follow.get('binding',{}).get('sha256')==admission['followthrough_binding_sha256'] and
        raw.get('source')==admission['raw_source'] and
        raw.get('result',{}).get('sha256')==admission['raw_result_sha256'] and
        raw.get('claim',{}).get('sha256')==admission['raw_claim_sha256'] and
        follow.get('archive5_recovery')==admission.get('archive5_recovery') and
        admission.get('recovery_commit')==recovery_commit,
        'independent output changed its admitted followthrough/raw/source boundary')


def consume(commit,*,followthrough_commit,recovery_commit=None):
    directory=scope(recovery_commit)
    _,publication=ctl.source(commit)
    claim=ctl.read(directory/'claim.json');process=ctl.read(directory/'process.json')
    child=ctl.read(directory/'child.json');result=ctl.read(directory/'result.json')
    ctl.need(claim.get('followthrough_commit')==followthrough_commit and claim.get('recovery_commit')==recovery_commit and
        claim['source']==publication==result['source'] and claim['source']['commit']==commit and
        result['claim_sha256']==ctl.digest(directory/'claim.json') and result['process']==process and
        result['child']==child and reader.birth_ok(process) and reader.birth_ok(child) and
        process['pid']==process['sid']==child['sid'] and child['pid']!=process['pid'] and
        result['completed_unix']>=result['started_unix']>=claim['created_unix'] and
        type(result['actual_child_exit_code']) is int and result['actual_child_exit_code']==0 and
        type(result['supervisor_exit_code']) is int and result['supervisor_exit_code']==0 and
        result['error'] is None and not ctl.live(process) and not ctl.live(child),
        'actual held sigma3 wait/source/birth differs')
    for name in ('stdout','stderr'):
        path=directory/('consumer.'+name);row=result[name]
        ctl.need(row['path']==str(path) and row['bytes']==path.stat().st_size and
                 row['sha256']==ctl.digest(path),'held sigma3 output bytes changed')
    ctl.need(result['stderr']['bytes']==0 and result['stdout']['bytes']<=32*2**20,
             'held sigma3 child errors or unbounded output')
    proof=ctl.read(directory/'consumer.stdout')
    accept_output(proof,commit,followthrough_commit,claim['admission'],recovery_commit)
    return dict(status=HELD_STATUS,result_path=str(directory/'result.json'),result_sha256=ctl.digest(directory/'result.json'),
        stdout_path=str(directory/'consumer.stdout'),stdout_sha256=result['stdout']['sha256'],
        actual_child_exit_code=0,completed_outer=200,world_size=4,goal_complete=False,
        scope='root reviews complete stdout before marking the goal complete')


def inspect(commit,*,followthrough_commit,recovery_commit=None):
    directory=scope(recovery_commit)
    ctl.owner()
    if not directory.exists():return dict(status='NOT_STARTED',directory=str(directory),goal_complete=False)
    ctl.need(directory.resolve()==directory and directory.stat().st_uid==ctl.OWNER,'unsafe final sigma3 observer directory')
    claim=ctl.read(directory/'claim.json')
    ctl.need(claim['source']['commit']==commit and claim.get('followthrough_commit')==followthrough_commit and claim.get('recovery_commit')==recovery_commit,
             'existing final sigma3 scope has another source; inspect original, never replay')
    process=ctl.read(directory/'process.json') if (directory/'process.json').exists() else None
    child=ctl.read(directory/'child.json') if (directory/'child.json').exists() else None
    live=process is not None and ctl.live(process)
    result=ctl.read(directory/'result.json') if (directory/'result.json').exists() else None
    if result is not None and result['supervisor_exit_code']==0 and not live:
        return consume(commit,followthrough_commit=followthrough_commit,recovery_commit=recovery_commit)
    return dict(status='ACTUAL_SIGMA3_FINAL_CPU_LIVE' if live else
        'ACTUAL_SIGMA3_FINAL_CPU_FAILED_INSPECT' if result is not None else 'UNKNOWN_NO_REPLAY',
        process=process,child=child,actual_supervisor_live=live,actual_child_live=bool(child and ctl.live(child)),
        result=result,goal_complete=False,checked_unix=time.time())


def start(commit,*,followthrough_commit,recovery_commit=None):
    directory=scope(recovery_commit)
    caps()
    if directory.exists() or directory.is_symlink():return inspect(commit,followthrough_commit=followthrough_commit,recovery_commit=recovery_commit)
    admission=ready(followthrough_commit,recovery_commit)
    repository,publication=ctl.source(commit);os.environ['QPROBE_SOURCE']=str(repository)
    ctl.need(directory.parent.is_dir() and directory.parent.resolve()==directory.parent and
        directory.parent.stat().st_uid==ctl.OWNER,'owned observation parent must already exist')
    try:directory.mkdir(mode=0o770)
    except FileExistsError:return inspect(commit,followthrough_commit=followthrough_commit,recovery_commit=recovery_commit)
    command=[sys.executable,'-m',MODULE,'supervise','--commit',commit,'--followthrough-commit',followthrough_commit]+recovery_args(recovery_commit)
    ctl.publish(directory/'claim.json',dict(source=publication,admission=admission,command=command,
        cwd=str(repository),created_unix=time.time(),followthrough_commit=followthrough_commit,recovery_commit=recovery_commit,training_control_permitted=False))
    with (directory/'supervisor.stdout').open('xb') as out,(directory/'supervisor.stderr').open('xb') as err:
        child=subprocess.Popen(command,cwd=str(repository),stdout=out,stderr=err,start_new_session=True)
    birth=ctl.birth(ctl.process(child.pid))
    ctl.need(birth['pid']==birth['sid'] and birth['uid']==ctl.OWNER,'detached sigma3 observer identity differs')
    ctl.publish(directory/'process.json',birth)
    return dict(status='ACTUAL_SIGMA3_FINAL_CPU_STARTED_NOT_ACCEPTED',process=birth,goal_complete=False)


def supervise(commit,*,followthrough_commit,recovery_commit=None):
    directory=scope(recovery_commit)
    caps();repository,publication=ctl.source(commit);os.environ['QPROBE_SOURCE']=str(repository)
    own=ctl.birth(ctl.process(os.getpid()));started=time.time();process=None;child=None;child_code=None;error=None;code=1
    try:
        deadline=time.monotonic()+15
        while not (directory/'process.json').exists():
            ctl.need(time.monotonic()<deadline,'launcher sigma3 binding missing; no replay')
            time.sleep(.05)
        claim=ctl.read(directory/'claim.json')
        ctl.need(ctl.read(directory/'process.json')==own and claim['source']==publication and
            claim['cwd']==str(repository) and claim.get('followthrough_commit')==followthrough_commit and claim.get('recovery_commit')==recovery_commit,
            'actual sigma3 supervisor source/birth differs')
        command=[sys.executable,'-m',CONSUMER,'--commit',commit,'--followthrough-commit',followthrough_commit]+recovery_args(recovery_commit)
        with (directory/'consumer.stdout').open('xb') as out,(directory/'consumer.stderr').open('xb') as err:
            process=subprocess.Popen(command,cwd=str(repository),stdout=out,stderr=err)
            child=ctl.birth(ctl.process(process.pid))
            ctl.need(reader.birth_ok(child) and child['sid']==own['sid'] and child['pid']!=own['pid'],
                     'held sigma3 CPU child identity differs')
            ctl.publish(directory/'child.json',child)
            child_code=process.wait()  # No SSH-sized timeout on fresh all200 I/O.
            for stream in (out,err):stream.flush();os.fsync(stream.fileno())
        ctl.need(child_code==0 and not ctl.live(child),'actual sigma3 consumer did not return0')
        ctl.need((directory/'consumer.stderr').stat().st_size==0,'actual sigma3 stderr is nonempty')
        ctl.need((directory/'consumer.stdout').stat().st_size<=32*2**20,'unbounded sigma3 consumer stdout')
        accept_output(ctl.read(directory/'consumer.stdout'),commit,followthrough_commit,claim['admission'],recovery_commit);code=0
    except BaseException as exc:
        error=repr(exc)
        import traceback
        traceback.print_exc()
        if process is not None and child_code is None:child_code=process.wait()
    outputs={}
    for name in ('stdout','stderr'):
        path=directory/('consumer.'+name)
        if path.exists():
            with path.open('rb') as stream:os.fsync(stream.fileno())
            outputs[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=ctl.digest(path))
    ctl.publish(directory/'result.json',dict(source=publication,claim_sha256=ctl.digest(directory/'claim.json'),
        process=own,child=child,actual_child_exit_code=child_code,supervisor_exit_code=code,error=error,
        started_unix=started,completed_unix=time.time(),goal_complete=False,**outputs))
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('start','supervise','inspect'));parser.add_argument('--commit',required=True)
    parser.add_argument('--followthrough-commit',required=True);parser.add_argument('--recovery-commit');args=parser.parse_args();caps()
    if args.mode=='supervise':sys.exit(supervise(args.commit,followthrough_commit=args.followthrough_commit,recovery_commit=args.recovery_commit))
    print(json.dumps(start(args.commit,followthrough_commit=args.followthrough_commit,recovery_commit=args.recovery_commit) if args.mode=='start' else
        inspect(args.commit,followthrough_commit=args.followthrough_commit,recovery_commit=args.recovery_commit),allow_nan=False),flush=True)
