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


def route_arguments(recovery_commit, recovery_followthrough_commit):
    reader.recovery_options(recovery_commit, recovery_followthrough_commit)
    return [] if recovery_commit is None else ['--recovery-commit', recovery_commit,
        '--recovery-followthrough-commit', recovery_followthrough_commit]


def ready(*, recovery_commit=None, recovery_followthrough_commit=None):
    # A missing200 refuses before any directory, claim or process is created.
    _,_,history=reader.early_history()
    route_arguments(recovery_commit,recovery_followthrough_commit)
    if recovery_commit is not None:
        return recovered_ready(history,recovery_commit,recovery_followthrough_commit)
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


def recovered_ready(history,recovery_commit,followthrough_commit):
    producer,reference=reader.producer_acceptance(recovery_commit)
    audit_base,native,follow=reader.recovery_evidence(recovery_commit,followthrough_commit,reference)
    raw,raw_source=reader.load_fixed(reader.RAW,'raw_audit_followthrough719225','ctl')
    original,original_file=reader.document(raw.DIRECTORY/'result.json')
    process,_=reader.document(raw.DIRECTORY/'process.json')
    claim,claim_file=reader.document(raw.DIRECTORY/'claim.json')
    ctl.need(reader.birth_ok(process) and process['pid']==process['sid'] and not ctl.live(process) and
        original.get('process')==process and original.get('source')==claim.get('source')==raw_source and
        original.get('claim_sha256')==claim_file['sha256'],'original RAW actual terminal source/birth differs')
    failure_anchors={};failure_fingerprint=None
    if original.get('status')==raw.STATUS:
        tasks=raw.actual_tasks(claim['parts'])
        ctl.need(original.get('successful_formal_outers')==list(range(1,201)) and
            original.get('normal_return_attempts')==len(tasks) and
            [row['task'] for row in original['records']]==tasks and
            [row['step'] for row in tasks if row['complete_outer']]==list(range(1,201)),
            'original successful RAW lacks all normal-return tasks')
        raw_mode='ORIGINAL_RAW_PASS'
    else:
        from reproduction.hpc3_qprobe import raw_terminal_reaggregation719225 as recon
        ctl.need(original.get('status')==recon.FAILED_STATUS,'RAW live/unknown/different failure cannot enter explicit recovery')
        _,tasks,failure,_=recon.original_failure(recovery_commit)
        failure_anchors=failure['original_file_anchors'];failure_fingerprint=failure['fingerprint']
        raw_mode='FRESH_AFTER_ORIGINAL_RAW_READ_FAILURE'
    stable=dict(recovery_commit=recovery_commit,recovery_followthrough_commit=followthrough_commit,
        producer_final_cpu_recovery=reference,followthrough_source=follow['source'],
        followthrough_result_sha256=follow['result']['sha256'],goal_sha256=follow['goal']['sha256'],
        consumer_binding_sha256=follow['binding']['sha256'],original_ea_fingerprint=follow['original_EA_failure']['fingerprint'],
        audit_base=str(audit_base),native_job=native['slurm_job_id'],raw_mode=raw_mode,
        raw_source=raw_source,raw_result_sha256=original_file['sha256'],raw_claim_sha256=claim_file['sha256'],
        raw_failure_fingerprint=failure_fingerprint,normal_return_attempts=len(tasks))
    # Header readiness only. Full teacher/raw/transport reads run in held child.
    return dict(history=history,producer_accounting=producer,native_accounting=follow['accounting'],
        recovery_binding=stable,raw_original_failure_anchors=failure_anchors,checked_unix=time.time())


def preserved_recovery(claim,recovery_commit,followthrough_commit):
    if recovery_commit is None:
        return
    actual=ready(recovery_commit=recovery_commit,recovery_followthrough_commit=followthrough_commit)
    ctl.need(actual['recovery_binding']==claim['admission']['recovery_binding'] and
        actual['raw_original_failure_anchors']==claim['admission']['raw_original_failure_anchors'],
        'recovery readiness/RAW failure changed after final once claim')
    raw_directory=ctl.ROOT/'observations/raw-all200-719225-20261015'
    for row in actual['raw_original_failure_anchors'].values():
        original=Path(row['path'])
        ctl.need(original.parent==raw_directory,'RAW failure copy escapes original scope')
        reader.read_file(DIRECTORY/'original-raw795-failure'/original.name,
            expected_sha=row['sha256'],expected_bytes=row['bytes'])


def accept_output(proof,commit,*,recovery_commit=None,recovery_followthrough_commit=None,recovery_binding=None):
    ctl.need(proof.get('status')==reader.STATUS and proof.get('run_id')==reader.RUN and
        proof.get('source',{}).get('commit')==commit and proof.get('completed_outer')==200 and
        proof.get('world_size')==4 and proof.get('actual_native_two_readbacks_reconsumed') is True and
        proof.get('update201_performed') is False and proof.get('goal_complete') is False and
        proof.get('actor_model_loaded') is False and all(type(proof.get(key)) is int and proof[key]==0
            for key in ('receipt_or_cache_or_claim_writes','training_or_helper_or_native_job_submissions',
                        'generated_trajectories','new_actor_head_or_scheduler_updates')),
        'actual final independent consumer stdout is not the full pure200 conjunction')
    route_arguments(recovery_commit,recovery_followthrough_commit)
    if recovery_commit is None:
        ctl.need(proof.get('recovery_commit') is None and proof.get('recovery_followthrough_commit') is None and
            proof.get('producer_final_cpu_recovery') is None and recovery_binding is None,
            'normal final consumer cannot carry an explicit recovery result')
        return
    follow=proof.get('actual_recovery_full_CPU_consumer') or {}
    raw=proof.get('independent_all_normal_return_raw') or {}
    ctl.need(isinstance(recovery_binding,dict) and proof.get('recovery_commit')==recovery_commit and
        proof.get('recovery_followthrough_commit')==recovery_followthrough_commit and
        proof.get('producer_final_cpu_recovery')==recovery_binding['producer_final_cpu_recovery'] and
        proof.get('original_producer_accounting_remains_failed') is True and
        proof.get('original_EA_status_remains_failed') is True and 'actual_EA_full_CPU_consumer' not in proof and
        proof.get('producer_accounting',{}).get('state')=='FAILED' and
        proof['producer_accounting'].get('exit_code')=='1:0' and
        follow.get('source')==recovery_binding['followthrough_source'] and
        follow.get('result',{}).get('sha256')==recovery_binding['followthrough_result_sha256'] and
        follow.get('goal',{}).get('sha256')==recovery_binding['goal_sha256'] and
        follow.get('binding',{}).get('sha256')==recovery_binding['consumer_binding_sha256'] and
        raw.get('result',{}).get('sha256')==recovery_binding['raw_result_sha256'] and
        raw.get('claim',{}).get('sha256')==recovery_binding['raw_claim_sha256'],
        'actual recovered final stdout differs from its claimed typed producer/held/RAW route')


def consume(commit,*,recovery_commit=None,recovery_followthrough_commit=None):
    _,publication=ctl.source(commit)
    claim=ctl.read(DIRECTORY/'claim.json');process=ctl.read(DIRECTORY/'process.json')
    result=ctl.read(DIRECTORY/'result.json');child=ctl.read(DIRECTORY/'child.json')
    route_arguments(recovery_commit,recovery_followthrough_commit)
    ctl.need(claim.get('recovery_commit')==recovery_commit and
        claim.get('recovery_followthrough_commit')==recovery_followthrough_commit,
        'existing final once scope belongs to another route; never replay')
    ctl.need(claim['source']==publication and claim['source']['commit']==commit and result['source']==claim['source'] and
        result['claim_sha256']==ctl.digest(DIRECTORY/'claim.json') and result['process']==process and
        result['child']==child and reader.birth_ok(process) and reader.birth_ok(child) and
        process['uid']==child['uid']==ctl.OWNER and
        process['pid']==process['sid']==child['sid'] and child['pid']!=process['pid'] and
        result['completed_unix']>=result['started_unix']>=claim['created_unix'] and
        type(result['actual_child_exit_code']) is int and result['actual_child_exit_code']==0 and
        type(result['supervisor_exit_code']) is int and result['supervisor_exit_code']==0 and
        result['error'] is None and not ctl.live(process) and not ctl.live(child),'actual held final consumer wait/source/birth differs')
    for name in ('stdout','stderr'):
        path=DIRECTORY/('consumer.'+name);row=result[name]
        ctl.need(row['path']==str(path) and row['bytes']==path.stat().st_size and
            row['sha256']==ctl.digest(path),'actual final consumer output changed')
    ctl.need(result['stderr']['bytes']==0 and result['stdout']['bytes']<=32*2**20,
        'final consumer errors or unbounded stdout')
    proof=ctl.read(DIRECTORY/'consumer.stdout');accept_output(proof,commit,recovery_commit=recovery_commit,
        recovery_followthrough_commit=recovery_followthrough_commit,
        recovery_binding=(claim.get('admission') or {}).get('recovery_binding'))
    preserved_recovery(claim,recovery_commit,recovery_followthrough_commit)
    return dict(status='ACTUAL_HELD_INDEPENDENT_FINAL200_CONSUMER_COMPLETED0',
        result_path=str(DIRECTORY/'result.json'),result_sha256=ctl.digest(DIRECTORY/'result.json'),
        stdout_path=str(DIRECTORY/'consumer.stdout'),stdout_sha256=result['stdout']['sha256'],
        actual_child_exit_code=0,consumer_status=proof['status'],completed_outer=200,world_size=4,
        goal_complete=False,scope='root must review the full stdout before marking its goal complete')


def inspect(commit,*,recovery_commit=None,recovery_followthrough_commit=None):
    ctl.owner()
    if not DIRECTORY.exists():return dict(status='NOT_STARTED',directory=str(DIRECTORY),goal_complete=False)
    ctl.need(DIRECTORY.resolve()==DIRECTORY and DIRECTORY.stat().st_uid==ctl.OWNER,
        'noncanonical final observer directory')
    claim=ctl.read(DIRECTORY/'claim.json')
    route_arguments(recovery_commit,recovery_followthrough_commit)
    ctl.need(claim.get('recovery_commit')==recovery_commit and
        claim.get('recovery_followthrough_commit')==recovery_followthrough_commit,
        'existing final once scope has another normal/recovery intent; never replay')
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
        return dict(consume(commit,recovery_commit=recovery_commit,recovery_followthrough_commit=recovery_followthrough_commit),actual_supervisor_live=False,process=process,child=child,
            output_bytes=sizes,checked_unix=time.time())
    return dict(status=('ACTUAL_INDEPENDENT_FINAL_CPU_CONSUMER_LIVE' if live else
        'ACTUAL_FINAL_CONSUMER_FAILED_INSPECT' if terminal is not None else 'UNKNOWN_FINAL_CONSUMER_NO_REPLAY'),
        actual_supervisor_live=live,actual_child_live=ctl.live(child) if child else False,
        process=process,child=child,result=terminal,output_bytes=sizes,goal_complete=False,
        checked_unix=time.time())


def start(commit,*,recovery_commit=None,recovery_followthrough_commit=None):
    caps()
    extra={} if recovery_commit is None and recovery_followthrough_commit is None else dict(
        recovery_commit=recovery_commit,recovery_followthrough_commit=recovery_followthrough_commit)
    if DIRECTORY.exists() or DIRECTORY.is_symlink():return inspect(commit,**extra)
    route=route_arguments(recovery_commit,recovery_followthrough_commit)
    admission=ready(**extra)
    repository,publication=ctl.source(commit)
    os.environ['QPROBE_SOURCE']=str(repository)
    # Exclusive once scope is created only after actual terminal readiness.
    try:DIRECTORY.mkdir(mode=0o770)
    except FileExistsError:return inspect(commit,**extra)
    command=[sys.executable,'-m',MODULE,'supervise','--commit',commit]+route
    ctl.publish(DIRECTORY/'claim.json',dict(source=publication,admission=admission,
        cwd=str(repository),command=command,created_unix=time.time(),training_control_permitted=False,**extra))
    for row in admission.get('raw_original_failure_anchors',{}).values():
        original=Path(row['path'])
        ctl.need(original.parent==ctl.ROOT/'observations/raw-all200-719225-20261015','RAW failure copy scope differs')
        raw,_=reader.read_file(original,capture=True,expected_sha=row['sha256'],expected_bytes=row['bytes'])
        target=DIRECTORY/'original-raw795-failure'/original.name;target.parent.mkdir(mode=0o770,exist_ok=True)
        with target.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    preserved_recovery(ctl.read(DIRECTORY/'claim.json'),recovery_commit,recovery_followthrough_commit)
    with (DIRECTORY/'supervisor.stdout').open('xb') as out,(DIRECTORY/'supervisor.stderr').open('xb') as err:
        process=subprocess.Popen(command,cwd=str(repository),stdout=out,stderr=err,start_new_session=True)
    birth=ctl.birth(ctl.process(process.pid))
    ctl.need(birth['pid']==birth['sid'] and birth['uid']==ctl.OWNER,'actual detached observer identity differs')
    ctl.publish(DIRECTORY/'process.json',birth)
    return dict(status='ACTUAL_FINAL_CPU_OBSERVER_STARTED_NOT_ACCEPTED',process=birth,
        directory=str(DIRECTORY),goal_complete=False)


def supervise(commit,*,recovery_commit=None,recovery_followthrough_commit=None):
    caps();repository,publication=ctl.source(commit)
    os.environ['QPROBE_SOURCE']=str(repository)
    own=ctl.birth(ctl.process(os.getpid()));started=time.time();child=None;child_code=None;error=None;code=1;process=None
    try:
        deadline=time.monotonic()+15
        while not (DIRECTORY/'process.json').exists():
            ctl.need(time.monotonic()<deadline,'launcher process binding missing; inspect without replay')
            time.sleep(.05)
        claim=ctl.read(DIRECTORY/'claim.json')
        route=route_arguments(recovery_commit,recovery_followthrough_commit)
        ctl.need(ctl.read(DIRECTORY/'process.json')==own and claim['source']==publication and
            claim['cwd']==str(repository) and claim.get('recovery_commit')==recovery_commit and
            claim.get('recovery_followthrough_commit')==recovery_followthrough_commit,
            'actual supervisor claim/source/birth/route differs')
        preserved_recovery(claim,recovery_commit,recovery_followthrough_commit)
        command=[sys.executable,'-m',CONSUMER,'--commit',commit]+route
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
        proof=ctl.read(DIRECTORY/'consumer.stdout');accept_output(proof,commit,recovery_commit=recovery_commit,
            recovery_followthrough_commit=recovery_followthrough_commit,
            recovery_binding=claim['admission'].get('recovery_binding'))
        preserved_recovery(claim,recovery_commit,recovery_followthrough_commit);code=0
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
    parser.add_argument('--commit',required=True);parser.add_argument('--recovery-commit')
    parser.add_argument('--recovery-followthrough-commit');args=parser.parse_args();caps()
    options=dict(recovery_commit=args.recovery_commit,recovery_followthrough_commit=args.recovery_followthrough_commit)
    if args.mode=='supervise':sys.exit(supervise(args.commit,**options))
    print(json.dumps(start(args.commit,**options) if args.mode=='start' else inspect(args.commit,**options),allow_nan=False),flush=True)
