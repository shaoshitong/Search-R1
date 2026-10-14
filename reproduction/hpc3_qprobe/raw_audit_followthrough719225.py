"""Independent once-only CPU raw/canonical/token audit of the exact200 history.

Serially follows actual normal-return attempts, including fully skipped guards.
The live producer and EA archive/final-native4 controller are never modified.
A launched child is retained to its actual wait result; unknown is never replay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import uuid

from reproduction.hpc3_qprobe import final_control as ctl

BASE = ctl.ROOT.parent/'qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014'
DIRECTORY = ctl.ROOT/'observations/raw-all200-719225-20261015'
MODULE = 'reproduction.hpc3_qprobe.raw_audit_followthrough719225'
ENTRY = 'reproduction/hpc3_qprobe/raw_actor_readonly719225.py'
STATUS = 'ACTUAL_RAW_ALL200_AND_ALL_NORMAL_RETURN_ATTEMPTS_PASS'


class ChangingObservation(RuntimeError):
    pass


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def safe_bytes(path):
    path = Path(path)
    ctl.need(path.is_absolute() and path.resolve() == path and not path.is_symlink(), 'unsafe raw audit input')
    with os.fdopen(os.open(str(path), os.O_RDONLY|os.O_NOFOLLOW), 'rb') as f:
        a = os.fstat(f.fileno())
        ctl.need(stat.S_ISREG(a.st_mode) and a.st_uid == ctl.OWNER, 'raw audit input owner/type differs')
        data = f.read()
        b = os.fstat(f.fileno())
    signature = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if signature(a) != signature(b) or signature(b) != signature(path.lstat()):
        raise ChangingObservation('Active scalar ledger changed during read')
    return data


def rows(path):
    data = safe_bytes(path)
    if data and not data.endswith(b'\n'):
        raise ChangingObservation('Active scalar ledger final row is incomplete')
    return [json.loads(line) for line in data.splitlines()]


def validate(commit):
    ctl.owner(); ctl.caps()
    ctl.need(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'bytecode writes must be disabled')
    return ctl.source(commit)


def plan():
    from reproduction.hpc3_qprobe.verify_goal import history_parts
    run = ctl.read(BASE/'logs/run-manifest.json')
    completed=rows(BASE/'logs/actual-updates.jsonl')
    ctl.need(completed and 6<=completed[-1]['step']<=200,'actual resumed successful prefix missing')
    # Authenticate only the already completed prefix now. Future6..200 is a
    # declared scope, never passed to a verifier that requires actual receipts.
    parts = history_parts(BASE,run,last=completed[-1]['step'])
    result = []
    for part in parts:
        original, binding, _ = ctl.producer(part['base'])
        ctl.need(original == part['run'], 'authenticated historical producer differs')
        result.append(dict(first=part['first'],last=200 if part['base']==BASE else part['last'],producer=binding))
    ctl.need([(p['first'],p['last'],p['producer']['producer_job'],p['producer']['producer_commit'])
              for p in result] == [
              (1,5,'718650','5816a20f331e10aaff9bbb95c313611ead289fd7'),
              (6,200,'719225','1843babdd96b387dd10d91e3207c9ea9489b6941')],
             'original saved5 history partition differs')
    return result


def tasks_from_rows(parts, ledger_rows):
    ctl.need(len(parts)==len(ledger_rows),'actual source partition ledgers missing')
    tasks=[]; seen=set(); completed=0
    for part, source_rows in zip(parts,ledger_rows):
        producer=part['producer']
        for row in source_rows:
            outer=row['logical_update']
            ctl.need(type(outer) is int and part['first']<=outer<=200, 'logical201/copied prefix attempt')
            if outer > part['last']:
                continue  # preserved old execution outside authenticated saved5 prefix
            aid=row['attempt_id'];count=row['optimizer_writes_this_attempt'];complete=row['complete_outer']
            ctl.need(isinstance(aid,str) and len(aid)==32 and all(c in '0123456789abcdef' for c in aid) and
                     aid not in seen and row['attempt_index']==len(tasks)+1 and row['run_id']==producer['run_id'] and
                     row['slurm_job_id']==producer['producer_job'] and outer==completed+1 and
                     type(count) is int and 0<=count<=20 and type(complete) is bool and complete is (count>0) and
                     row['opportunities_processed']==20 and row['scheduler_advances']==int(complete) and
                     row['completed_outer']==completed+int(complete) and
                     row['status']==('complete' if complete else 'recoverable_guard'),
                     'actual normal-return attempt sequence/guard transaction differs')
            seen.add(aid);completed+=int(complete)
            tasks.append(dict(attempt_id=aid,attempt_index=row['attempt_index'],step=outer,
                              complete_outer=complete,actual_optimizer_writes=count,
                              record_sha256=canonical_sha(row),producer=producer))
    return tasks


def actual_tasks(parts):
    for part in parts:
        p=part['producer'];b=Path(p['base'])
        ctl.need(ctl.digest(b/'logs/run-manifest.json')==p['source_manifest_sha256'] and
                 ctl.digest(b/'logs/native-allocation-admission.json')==p['native_allocation_sha256'],
                 'actual historical manifest/allocation changed')
    return tasks_from_rows(parts,[rows(Path(p['producer']['base'])/'logs/actual-attempts.jsonl') for p in parts])


def task_ready(task):
    base=Path(task['producer']['base']);aid=task['attempt_id'];root=base/'artifacts/question-probe'
    folder=root/'attempts'/aid
    terminal=root/('completed' if task['complete_outer'] else 'aborted')/(aid+'.json')
    required=[folder/'real-rollout.pt',folder/'selection.json',folder/'rollout-audit.json',terminal,
              base/'logs'/('behavior-probability-attempt-'+aid+'.json'),
              base/'artifacts/behavior-traces'/('attempt-'+aid+'.json')]
    if not all(p.is_file() and not p.is_symlink() for p in required):
        return False
    try:
        terminal_row=json.loads(safe_bytes(terminal))
        updates=rows(base/'logs/actual-updates.jsonl')
    except (ChangingObservation,json.JSONDecodeError):
        return False
    linked=[r for r in updates if r['attempt_id']==aid]
    return (terminal_row.get('attempt_id')==aid and terminal_row.get('outer')==task['step'] and
            (len(linked)==1 if task['complete_outer'] else not linked))


def birth_ok(row):
    return (isinstance(row,dict) and all(type(row.get(k)) is int and row[k]>0
            for k in ('pid','start_ticks','sid','uid')) and row['uid']==ctl.OWNER)


def accept_result(task, result, output, supervisor, child_live):
    ctl.need(birth_ok(supervisor) and birth_ok(result.get('child')) and
             supervisor['sid']==supervisor['pid'] and result['process']==supervisor and
             result['child']['sid']==supervisor['sid'] and result['child']['pid']!=supervisor['pid'] and
             result['actual_child_exit_code']==0 and not child_live and not result.get('worker_error') and
             result['stderr']['bytes']==0 and result['task']==task, 'actual audit child wait/birth/task differs')
    producer=task['producer'];transaction=output['actual_attempt_transaction'];canonical=output['canonical_proof']
    ctl.need(output['status']=='ACTUAL_RAW_CANONICAL_TRACE_ALLTOKENS_AND_ACTOR_ROWS_EQUAL' and
             output['step']==task['step'] and output['job']==producer['producer_job'] and
             output['source_commit']==producer['producer_commit'] and
             output['source_manifest']['sha256']==producer['source_manifest_sha256'] and
             output['allocation_file']['sha256']==producer['native_allocation_sha256'] and
             transaction['attempt_id']==task['attempt_id'] and transaction['attempt_index']==task['attempt_index'] and
             transaction['complete_outer'] is task['complete_outer'] and
             transaction['actual_optimizer_writes']==task['actual_optimizer_writes'] and
             transaction['attempt_record_sha256']==task['record_sha256'] and
             canonical['actual_head_committed'] is task['complete_outer'] and
             canonical['actual_raw_decode_strict_terminal_em_equal'] is True and
             canonical['actual_512_qid_text_gold_equal'] is True and
             canonical['actual_selection_G5_E64_top448_prequential_head_equal'] is True and
             canonical['questions']==512 and canonical['trajectories']==2560 and
             all(output[k] is True for k in ('actual_private_trace_byte_equal','actual_all2560_ordered_ids_equal',
                 'actual_512_parent_digests_equal','actual_2560_actor_tensor_rows_equal',
                 'actual_all_retained_token_contexts_equal','actual_selected_logprob_projection_equal')) and
             output['hf_forward_performed'] is False and output['model_weights_loaded'] is False and
             output['generated_trajectories']==output['actor_or_head_updates_performed']==output['evidence_files_written']==0,
             'actual same-fd canonical/raw/context/actor proof differs')


def consume_task(task, directory, supervisor):
    claim=ctl.read(directory/'claim.json');result=ctl.read(directory/'result.json')
    ctl.need(claim['task']==task and result['task']==task and result['claim_sha256']==ctl.digest(directory/'claim.json') and
             result['source']==claim['source']==ctl.read(DIRECTORY/'claim.json')['source'] and
             result['completed_unix']>=result['started_unix']>=claim['created_unix'],
             'task claim/result/source differs')
    child=ctl.read(directory/'child.json')
    ctl.need(result['child']==child,'task child birth receipt differs')
    for name in ('stdout','stderr'):
        path=directory/('audit.'+name)
        ctl.need(result[name]['path']==str(path) and ctl.digest(path)==result[name]['sha256'] and
                 path.stat().st_size==result[name]['bytes'],'actual task output SHA/bytes differs')
    output=ctl.read(directory/'audit.stdout')
    accept_result(task,result,output,supervisor,ctl.live(child))
    return dict(task=task,result_path=str(directory/'result.json'),result_sha256=ctl.digest(directory/'result.json'),
                output_sha256=result['stdout']['sha256'],raw_sha256=output['raw_file']['sha256'],
                private_trace_sha256=output['private_trace_file']['sha256'],
                included_actor_token_count=output['included_actor_token_count'])


def state(value):
    temporary=DIRECTORY/('state-'+uuid.uuid4().hex+'.tmp')
    ctl.publish(temporary,dict(value,checked_unix=time.time()))
    os.replace(temporary,DIRECTORY/'state.json')
    fd=os.open(DIRECTORY,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def execute_task(repository, publication, task, supervisor):
    directory=DIRECTORY/'attempts'/task['attempt_id']
    ctl.need(not directory.exists(),'task already attempted; inspect original claim, never replay')
    directory.mkdir(mode=0o770)
    command=[sys.executable,str(repository/ENTRY),'--step',str(task['step']),'--attempt-id',task['attempt_id']]
    claim=dict(task=task,source=publication,command=command,process=supervisor,created_unix=time.time())
    ctl.publish(directory/'claim.json',claim)
    child=None;actual=None;error=None;started=time.time()
    with (directory/'audit.stdout').open('x') as stdout,(directory/'audit.stderr').open('x') as stderr:
        try:
            env=dict(os.environ,QPROBE_SOURCE=str(ctl.ROOT/'source'/task['producer']['producer_commit']))
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                   cwd=str(repository),close_fds=True,env=env)
            actual=ctl.birth(ctl.process(child.pid))
            ctl.need(birth_ok(actual) and actual['sid']==supervisor['sid'],'direct audit CPU birth differs')
            ctl.publish(directory/'child.json',actual)
            state(dict(phase='ACTUAL_RAW_CPU_CHILD_RUNNING',current_task=task,child=actual,process=supervisor))
        except Exception as exc:
            error=dict(error_type=type(exc).__name__,message=str(exc))
        finally:
            code=child.wait() if child is not None else None
        stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
    outputs={name:dict(path=str(directory/('audit.'+name)),sha256=ctl.digest(directory/('audit.'+name)),
                       bytes=(directory/('audit.'+name)).stat().st_size) for name in ('stdout','stderr')}
    result=dict(task=task,source=publication,claim_sha256=ctl.digest(directory/'claim.json'),process=supervisor,
                child=actual,actual_child_exit_code=code,started_unix=started,completed_unix=time.time(),**outputs)
    if error is not None:result['worker_error']=error
    ctl.publish(directory/'result.json',result)
    return consume_task(task,directory,supervisor)


def worker(commit):
    repository,publication=validate(commit);claim=ctl.read(DIRECTORY/'claim.json')
    supervisor=ctl.birth(ctl.process(os.getpid()))
    ctl.need(birth_ok(supervisor) and supervisor['sid']==supervisor['pid'],'detached audit birth differs')
    ctl.publish(DIRECTORY/'process.json',supervisor)
    records=[];started=time.time()
    try:
        ctl.need(claim['source']==publication and claim['parts']==plan(),'actual once audit source/plan differs')
        while True:
            try:tasks=actual_tasks(claim['parts'])
            except ChangingObservation:
                state(dict(phase='ACTIVE_LEDGER_READ_WAIT',process=supervisor,completed_attempt_audits=len(records)))
                time.sleep(20);continue
            ctl.need(len(tasks)>=len(records) and [r['task'] for r in records]==tasks[:len(records)],
                     'previous actual attempt prefix changed')
            if len(records)<len(tasks) and task_ready(tasks[len(records)]):
                record=execute_task(repository,publication,tasks[len(records)],supervisor)
                ctl.publish(DIRECTORY/'accepted'/('%06d.json'%record['task']['attempt_index']),record)
                records.append(record)
                state(dict(phase='ACTUAL_RAW_ATTEMPT_COMPLETED0',process=supervisor,
                           completed_attempt_audits=len(records),latest=record))
                continue
            successes=[r['task']['step'] for r in records if r['task']['complete_outer']]
            if successes==list(range(1,201)):
                ctl.need(len(tasks)==len(records),'unread normal-return attempt after successful200')
                from reproduction.hpc3_qprobe.verify_goal import history_parts,verify_attempt_parts
                run=ctl.read(BASE/'logs/run-manifest.json')
                attempts,hashes,formal=verify_attempt_parts(history_parts(BASE,run,last=200))
                ctl.need([r['attempt_id'] for r in attempts]==[r['task']['attempt_id'] for r in records],
                         'raw audits omitted or duplicated an actual guarded attempt')
                result=dict(status=STATUS,source=publication,claim_sha256=ctl.digest(DIRECTORY/'claim.json'),
                            process=supervisor,started_unix=started,completed_unix=time.time(),
                            normal_return_attempts=len(records),successful_formal_outers=successes,
                            records=records,record_list_sha256=canonical_sha(records),
                            formal_attempt_audits=formal,goal_complete=False,
                            scope='all200 canonical raw and every normal-return actor trace; final native4/archive/score acceptance separate')
                ctl.publish(DIRECTORY/'result.json',result);return result
            _,_,accounting=ctl.producer(BASE)
            if '|RUNNING|' not in accounting and '|COMPLETING|' not in accounting:
                raise ValueError('Producer terminal before exact200 raw/normal-return attempt audit')
            state(dict(phase='ACTUAL_PRODUCER_OR_HEAD_PUBLICATION_WAIT',process=supervisor,
                       completed_attempt_audits=len(records),successful_outers=len(successes),
                       accounting=accounting,latest=records[-1] if records else None))
            time.sleep(20)
    except Exception as exc:
        failure=dict(status='ACTUAL_READONLY_AUDIT_STOPPED_EVIDENCE_RETAINED',source=publication,
                     claim_sha256=ctl.digest(DIRECTORY/'claim.json'),process=supervisor,
                     started_unix=started,completed_unix=time.time(),completed_attempt_audits=len(records),
                     error_type=type(exc).__name__,error=str(exc),goal_complete=False)
        ctl.publish(DIRECTORY/'result.json',failure)
        return failure


def inspect(commit):
    _,publication=validate(commit)
    if not DIRECTORY.exists():return dict(status='NOT_STARTED',source=publication,directory=str(DIRECTORY))
    ctl.need(DIRECTORY.resolve()==DIRECTORY and DIRECTORY.stat().st_uid==ctl.OWNER,'unsafe audit directory')
    claim=ctl.read(DIRECTORY/'claim.json');ctl.need(claim['source']==publication,'once audit source differs')
    process=ctl.read(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else None
    live=process is not None and ctl.live(process)
    progress=ctl.read(DIRECTORY/'state.json') if (DIRECTORY/'state.json').exists() else None
    final=ctl.read(DIRECTORY/'result.json') if (DIRECTORY/'result.json').exists() else None
    latest=None
    accepted=sorted((DIRECTORY/'accepted').glob('*.json'))
    if progress and progress.get('latest'):
        task=progress['latest']['task']
        latest=consume_task(task,DIRECTORY/'attempts'/task['attempt_id'],process)
    elif accepted:
        record=ctl.read(accepted[-1]);task=record['task']
        latest=consume_task(task,DIRECTORY/'attempts'/task['attempt_id'],process)
        ctl.need(latest==record,'immutable accepted receipt differs from actual wait/output')
    child=progress.get('child') if progress else None
    if final is not None:
        ctl.need(final['source']==publication and final['claim_sha256']==ctl.digest(DIRECTORY/'claim.json') and
                 final['process']==process,'terminal audit source/claim/process differs')
        if final['status']==STATUS:
            records=[consume_task(r['task'],DIRECTORY/'attempts'/r['task']['attempt_id'],process)
                     for r in final['records']]
            ctl.need(birth_ok(process) and [r['task']['step'] for r in records if r['task']['complete_outer']]==list(range(1,201))
                     and records==final['records'] and canonical_sha(records)==final['record_list_sha256'] and
                     len(records)==final['normal_return_attempts'] and
                     [r['task'] for r in records]==actual_tasks(claim['parts']),
                     'all200 exact raw/guarded attempt consumer differs')
    return dict(status=final['status'] if final else ('ACTUAL_READONLY_AUDIT_LIVE' if live else 'UNKNOWN_NO_REPLAY'),
                process=process,actual_live_birth=live,actual_child_live_birth=ctl.live(child) if child else False,
                accepted_attempt_receipts=len(accepted),
                progress=progress,latest_completed_audit=latest,
                result={k:v for k,v in final.items() if k not in ('records','formal_attempt_audits')} if final else None,
                source=publication,checked_unix=time.time())


def start(commit):
    repository,publication=validate(commit)
    if DIRECTORY.exists():return inspect(commit)
    parts=plan()
    DIRECTORY.parent.mkdir(mode=0o770,exist_ok=True)
    ctl.need(DIRECTORY.parent.resolve()==DIRECTORY.parent and DIRECTORY.parent.stat().st_uid==ctl.OWNER,
             'actual JXU observation namespace required')
    DIRECTORY.mkdir(mode=0o770);(DIRECTORY/'attempts').mkdir();(DIRECTORY/'accepted').mkdir()
    command=[sys.executable,'-m',MODULE,'--commit',commit,'--worker']
    ctl.publish(DIRECTORY/'claim.json',dict(source=publication,parts=parts,command=command,
                created_unix=time.time(),training_control_permitted=False,goal_complete=False))
    with (DIRECTORY/'supervisor.log').open('x') as output:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                               cwd=str(repository),close_fds=True,start_new_session=True)
    for _ in range(50):
        if (DIRECTORY/'process.json').exists() or child.poll() is not None:break
        time.sleep(0.2)
    return inspect(commit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--start',action='store_true');g.add_argument('--worker',action='store_true');g.add_argument('--inspect',action='store_true')
    a=p.parse_args();os.umask(0o007)
    print(json.dumps(start(a.commit) if a.start else worker(a.commit) if a.worker else inspect(a.commit)),flush=True)
