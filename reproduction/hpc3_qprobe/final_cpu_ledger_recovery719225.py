"""Terminal-only CPU revalidation of a precisely bound final ledger-read failure.

This does not recover training or turn FAILED accounting into COMPLETED. It
preserves the original failure and holds one CPU child that revalidates the
completed scientific evidence into a separate structural receipt. Archives,
actual native4 restore and the independent final conjunction remain required.
No command here submits a Slurm job, reconstructs an actor or changes a producer.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import time

from reproduction.hpc3_qprobe import final_control as ctl
from reproduction.hpc3_qprobe import independent_goal_acceptance719225 as reader

BASE = reader.BASE
ROOT = reader.ROOT
DIRECTORY = ROOT/'observations/final-cpu-ledger-recovery719225-20261015'
MODULE = 'reproduction.hpc3_qprobe.final_cpu_ledger_recovery719225'
FAILURE_KIND = 'ACTUAL_FAILED719225_FINAL_CPU_PROCESS_LEDGER_ENOENT_ONLY'
STATUS = 'ACTUAL_FAILED719225_LEDGER_ONLY_FULL200_CPU_REVALIDATION_PASS'
HELD_STATUS = 'ACTUAL_HELD_FAILED719225_FINAL_CPU_REVALIDATION_COMPLETED0'
FILES = (
    'logs/run-manifest.json', 'logs/process-ledger.json', 'logs/pipeline-status.json',
    'logs/cleanup-status.json', 'logs/final-acceptance.log', 'logs/final-identity.json',
    'logs/actual-updates.jsonl', 'logs/actual-attempts.jsonl',
    'logs/native-allocation-admission.json', 'preparation/training-submission.json',
    'preparation/training-submit-claim.json', 'preparation/configuration-preflight.json',
    'preparation/checkpoint-restoration.json',
)


def need(ok, message):
    if not ok:
        raise ValueError('FINAL_CPU_LEDGER_RECOVERY719225: '+message)


def caps():
    ctl.owner()
    for name in reader.CAPS:
        os.environ[name] = '1'
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    sys.dont_write_bytecode = True


def accounting():
    raw = subprocess.check_output(['sacct', '-nP', '-X', '-j', '719225',
        '--format=JobID,State,ExitCode,User,UID,Account,QOS,Partition,End'],
        text=True, timeout=30)
    values = [line.split('|') for line in raw.splitlines() if line.strip()]
    expected = ['719225', 'FAILED', '1:0', 'jxu957', '14090', 'jxu957',
                'formal-user', 'emergency_acd']
    need(len(values) == 1 and values[0][:8] == expected and len(values[0]) == 9 and
         re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', values[0][8]) is not None,
         'actual producer must be terminal FAILED1:0 with an observed end time')
    return dict(job='719225', state='FAILED', exit_code='1:0', end=values[0][8], raw=raw)


def ledger_read_traceback(text):
    """Accept only the original CPU verifier's precise missing-ledger error."""
    need(isinstance(text, str) and text.count('Traceback (most recent call last):') == 1,
         'one original final CPU traceback required')
    lines = [line for line in text.splitlines() if line.strip()]
    bare = 'FileNotFoundError: [Errno 2] No such file or directory'
    expected = bare+": '"+str(BASE/'logs/process-ledger.json')+"'"
    need(lines and lines[-1] in (bare, expected) and
         'During handling of the above exception' not in text and
         'The above exception was the direct cause' not in text,
         'failure is not the precise process-ledger ENOENT exception')
    frames = re.findall(r'^\s*File "([^"]+)", line (\d+), in ([^\n]+)$', text, re.M)
    source = ROOT/'source'/reader.PRODUCER
    allowed_runtime = {str(source/'reproduction/common8/runtime.py'),
                       str(BASE/'repo/reproduction/common8/runtime.py')}
    allowed_full = {str(source/'reproduction/common8/verify_full.py'),
                    str(BASE/'repo/reproduction/common8/verify_full.py')}
    selected = [(index, int(line)) for index, (path, line, function) in enumerate(frames)
                if path in allowed_runtime and function == 'verify_scheduler_receipts']
    need(len(selected) == 1 and
         any(path in allowed_full and function == 'verify_full' for path, _, function in frames),
         'traceback is not the immutable original scheduler/full verifier')
    # Actual NFS read failures can omit filename even though open succeeded.
    # The pinned original source has exactly two process-ledger read sites.
    # Bind a bare ENOENT to their exact line + complete terminal call chain;
    # a different JSON, scheduler, model or data-file read is not recoverable.
    index, line = selected[0]
    tail = frames[index+1:]
    data_paths = {str(source/'reproduction/common8/verify_data.py'),
                  str(BASE/'repo/reproduction/common8/verify_data.py')}
    pathlib_path = str(Path(sysconfig.get_path('stdlib'))/'pathlib.py')
    digest_read = (line == 315 and len(tail) == 2 and
        all(path in data_paths and number == '17' for path, number, _ in tail) and
        [function for _, _, function in tail] == ['digest', '<lambda>'])
    json_read = (line == 181 and len(tail) == 1 and
        tail[0][0] == pathlib_path and tail[0][2] == 'read_text')
    direct_open = (lines[-1] == expected and line in (181, 315) and
        ((line == 315 and len(tail) in (1, 2) and tail[0][0] in data_paths and
          tail[0][1] in ('16', '17') and tail[0][2] == 'digest' and
          (len(tail) == 1 or (tail[1][0] == pathlib_path and tail[1][2] == 'open'))) or
         (line == 181 and all(path == pathlib_path and function in ('read_text', 'open')
                             for path, _, function in tail))))
    need(digest_read or json_read or direct_open, 'ENOENT is not an original process-ledger read chain')
    need(not re.search(r'CUDA out of memory|OutOfMemoryError|NCCL|AssertionError:|RuntimeError:', text, re.I),
         'another failure is present in final CPU output')
    return dict(exception='FileNotFoundError', errno=2,
        path=str(BASE/'logs/process-ledger.json'), original_scheduler_frame=True,
        original_full_verifier_frame=True, original_runtime_line=line,
        filename_present=lines[-1] == expected,
        read_site='digest-read' if digest_read else 'json-read' if json_read else 'open')


def terminal_documents(run, ledger, pipeline, cleanup, final, output, *, alive):
    need(run.get('run_id') == reader.RUN and run.get('source_commit') == reader.PRODUCER and
         run.get('training_world_size') == 4, 'immutable actual producer/run/topology differs')
    need(ledger.get('run_id') == reader.RUN and isinstance(ledger.get('jobs'), dict),
         'original process ledger run/schema differs')
    jobs = ledger['jobs']
    need('trainer' in jobs and 'final-acceptance' in jobs,
         'actual trainer and final CPU child are absent')
    for name, expected_code in (('trainer', 0), ('final-acceptance', 1)):
        job = jobs[name]
        need(type(job.get('exit_code')) is int and job['exit_code'] == expected_code and
             job.get('state') == 'STOPPED' and reader.birth_ok(job.get('leader')) and
             job['leader']['uid'] == ctl.OWNER and job['leader']['pid'] == job['leader']['sid'],
             'actual '+name+' birth/exit/cleanup differs')
    need(jobs['final-acceptance'].get('log') == str(BASE/'logs/final-acceptance.log') and
         jobs['final-acceptance'].get('arguments') == [str(BASE/'env/bin/python'), '-m',
             'reproduction.common8.verify_full', '--base', str(BASE)],
         'original CPU verifier arguments/log differ')
    for name, job in jobs.items():
        need(job.get('state') == 'STOPPED' and reader.birth_ok(job.get('leader')) and
             job['leader']['uid'] == ctl.OWNER and isinstance(job.get('ledger'), dict) and
             str(job['leader']['pid']) in job['ledger'] and
             reader.birth_ok(job['ledger'][str(job['leader']['pid'])]) and
             ctl.birth(job['ledger'][str(job['leader']['pid'])]) == ctl.birth(job['leader']),
             'original owned component is not cleanly stopped: '+name)
        for key, process in job['ledger'].items():
            need(reader.birth_ok(process) and key == str(process['pid']) and
                 process['uid'] == ctl.OWNER and process['sid'] == job['leader']['sid'] and
                 not alive(process), 'original owned process is live/unbound: '+name)
    need(pipeline.get('run_id') == reader.RUN and pipeline.get('stage') == 'FAILED' and
         pipeline.get('error') == repr(AssertionError('final-acceptance failed; see project log')) and
         type(pipeline.get('unix')) in (int, float) and math.isfinite(pipeline['unix']) and
         cleanup.get('run_id') == reader.RUN and cleanup.get('cleaned') is True and
         type(cleanup.get('checked_unix')) in (int, float) and math.isfinite(cleanup['checked_unix']) and
         cleanup['checked_unix'] >= pipeline['unix'], 'original failure/owned cleanup differs')
    need(final.get('run_id') == reader.RUN and type(final.get('step')) is int and final['step'] == 200 and
         final.get('final') is True and all(isinstance(final.get(key), list) and len(final[key]) == 51713
            for key in ('question_ids', 'data_sources', 'em', 'f1')) and
         isinstance(final.get('counts'), dict) and len(final['counts']) == 7 and
         all(type(x) is int and x > 0 for x in final['counts'].values()) and
         sum(final['counts'].values()) == 51713, 'actual final200/native51713 absent')
    return ledger_read_traceback(output)


def failure_evidence():
    """Actual terminal and missing200 gates precede source scans or side effects."""
    ctl.owner()
    actual = accounting()
    run, _, history = reader.early_history()
    current, binding, raw_accounting = ctl.producer(BASE, complete=False)
    need(current == run and binding['producer_job'] == '719225' and
         binding['producer_commit'] == reader.PRODUCER and binding['run_id'] == reader.RUN and
         [line.split('|')[1:3] for line in raw_accounting.splitlines() if line.strip()] == [['FAILED', '1:0']],
         'original producer source/submission/allocation/accounting differs')
    from reproduction.hpc3_qprobe.followthrough719225 import fixed_binding
    fixed = fixed_binding()
    snapshots = {}; proofs = {}; documents = {}
    for relative in FILES:
        raw, proof = reader.read_file(BASE/relative, capture=True, capture_limit=128*2**20)
        snapshots[relative] = raw; proofs[relative] = proof
        if relative.endswith('.json'):
            documents[relative] = json.loads(raw)
    trace = terminal_documents(run, documents['logs/process-ledger.json'],
        documents['logs/pipeline-status.json'], documents['logs/cleanup-status.json'],
        documents['logs/final-identity.json'], snapshots['logs/final-acceptance.log'].decode(),
        alive=ctl.live)
    need(not (BASE/'logs/full-run-acceptance.json').exists(),
         'original full structural receipt already exists; inspect a different failure boundary')
    source, publication = reader.publication(reader.PRODUCER)
    publication_bytes, _ = reader.read_file(Path(publication['publication_path']), capture=True,
        expected_sha=publication['publication_sha256'])
    published_files = json.loads(publication_bytes)['files']
    for name in ('runtime.py', 'verify_full.py', 'verify_data.py'):
        key = 'source/reproduction/common8/'+name
        raw, proof = reader.read_file(source/'reproduction/common8'/name, capture=True,
            expected_sha=published_files['reproduction/common8/'+name])
        snapshots[key] = raw; proofs[key] = proof
    # Keep identity metadata as evidence, while the typed content fingerprint
    # binds every original byte and accounting/source/claim. No timestamps are
    # rewritten or relabeled as producer success.
    stable = dict(kind=FAILURE_KIND, accounting=actual, producer=binding,
        fixed_submission=fixed, history=history, source=publication, traceback=trace,
        files={name:{key:row[key] for key in ('path', 'sha256', 'bytes')}
               for name, row in proofs.items()}, original_accounting_remains_failed=True,
        trained_outers=200, final_native_questions=51713, training_world_size=4)
    return dict(stable, fingerprint=reader.canonical_sha(stable),
        original_file_identities=proofs, checked_unix=time.time()), snapshots


def preserved_failure(claim):
    now, _ = failure_evidence()
    need(now['fingerprint'] == claim['failure']['fingerprint'],
         'original terminal failure/evidence changed after once claim')
    for name, record in claim['failure']['files'].items():
        reader.read_file(DIRECTORY/'failure'/name, expected_sha=record['sha256'],
                         expected_bytes=record['bytes'])
    return now


def accept_output(value, commit, fingerprint):
    need(value.get('status') == STATUS and value.get('source', {}).get('commit') == commit and
         value.get('failure_fingerprint') == fingerprint and value.get('run_id') == reader.RUN and
         value.get('completed_outer') == 200 and value.get('world_size') == 4 and
         value.get('final_native_questions') == 51713 and value.get('goal_complete') is False and
         value.get('original_accounting_remains_failed') is True and
         value.get('native4_restore_and_all_archives_and_independent_final_still_required') is True and
         all(value.get(key) is False for key in ('actor_model_loaded', 'training_performed',
             'gpu_allocation_requested', 'original_receipts_overwritten', 'update201_performed')),
         'CPU output is not the precise separate structural revalidation')


def task(commit):
    caps(); repository, publication = ctl.source(commit)
    need(os.environ.get('QPROBE_SOURCE') == str(repository), 'CPU task source differs')
    claim = ctl.read(DIRECTORY/'claim.json'); own = ctl.birth(ctl.process(os.getpid()))
    deadline = time.monotonic()+15
    while not (DIRECTORY/'child.json').exists():
        need(time.monotonic() < deadline, 'actual held child binding absent; never replay')
        time.sleep(.05)
    need(ctl.read(DIRECTORY/'child.json') == own and claim['source'] == publication,
         'actual held CPU task birth/source differs')
    before = preserved_failure(claim)
    from reproduction.common8.verify_full import verify_full
    full = verify_full(BASE, publish_receipt=False)
    from reproduction.hpc3_qprobe.verify_goal import (history_parts, part_updates,
        verify_evaluations, verify_attempt_parts)
    run = ctl.read(BASE/'logs/run-manifest.json'); parts = history_parts(BASE, run)
    _, history = part_updates(parts)
    evaluations, final, source_em, source_f1 = verify_evaluations(BASE, run, parts=parts)
    attempts, evidence, formal = verify_attempt_parts(parts)
    need(full.get('structural_accepted') is True and full.get('run_id') == reader.RUN and
         full.get('actual_outer_updates') == 200 and full['final_source_em'] == source_em and
         full['final_source_f1'] == source_f1 and full['final_weighted_em'] == final['weighted_em'] and
         full['final_weighted_f1'] == final['weighted_f1'], 'revalidated full200/canonical metrics differ')
    pairs = {}
    for step in ctl.STEPS:
        path, proof = ctl.pair(BASE, run, step)
        need(proof['path'] == str(path), 'paired checkpoint metadata path differs')
        pairs[str(step)] = proof
    after = preserved_failure(claim)
    need(before['fingerprint'] == after['fingerprint'], 'original failure changed during CPU child')
    structural = DIRECTORY/'structural-acceptance.json'
    ctl.publish(structural, full)
    return dict(status=STATUS, source=publication, run_id=reader.RUN, world_size=4,
        completed_outer=200, final_native_questions=51713,
        failure_fingerprint=after['fingerprint'], actual_process=own,
        structural_acceptance=dict(path=str(structural), sha256=ctl.digest(structural)),
        original_training_parts=history, actual_formal_part_audits=formal,
        actual_normal_return_attempts=len(attempts), attempt_evidence=evidence,
        fixed_validation=evaluations, completed_pair_metadata=pairs,
        actor_model_loaded=False, training_performed=False, gpu_allocation_requested=False,
        original_receipts_overwritten=False, update201_performed=False,
        original_accounting_remains_failed=True, goal_complete=False,
        native4_restore_and_all_archives_and_independent_final_still_required=True,
        checked_unix=time.time())


def consume(commit):
    caps(); _, publication = ctl.source(commit)
    claim = ctl.read(DIRECTORY/'claim.json'); parent = ctl.read(DIRECTORY/'process.json')
    child = ctl.read(DIRECTORY/'child.json'); result = ctl.read(DIRECTORY/'result.json')
    need(claim['source'] == publication and result['source'] == publication and
         result['claim_sha256'] == ctl.digest(DIRECTORY/'claim.json') and
         result['process'] == parent and result['child'] == child and
         reader.birth_ok(parent) and reader.birth_ok(child) and
         parent['uid'] == child['uid'] == ctl.OWNER and
         parent['pid'] == parent['sid'] == child['sid'] and child['pid'] != parent['pid'] and
         type(result.get('actual_child_exit_code')) is int and result['actual_child_exit_code'] == 0 and
         type(result.get('supervisor_exit_code')) is int and result['supervisor_exit_code'] == 0 and
         result.get('error') is None and result['completed_unix'] >= result['started_unix'] >= claim['created_unix'] and
         not ctl.live(parent) and not ctl.live(child), 'actual held CPU wait/birth/source/terminal differs')
    failure = preserved_failure(claim)
    for name in ('stdout', 'stderr'):
        row = result[name]; path = DIRECTORY/('consumer.'+name)
        need(row['path'] == str(path), 'held CPU output path differs')
        reader.read_file(path, expected_sha=row['sha256'], expected_bytes=row['bytes'])
    need(result['stderr']['bytes'] == 0, 'held CPU child stderr is not empty')
    value, output = reader.document(DIRECTORY/'consumer.stdout')
    accept_output(value, commit, failure['fingerprint'])
    need(value['actual_process'] == child and value['structural_acceptance']['path'] ==
         str(DIRECTORY/'structural-acceptance.json'), 'structural child/output path differs')
    full_bytes, full_file = reader.read_file(DIRECTORY/'structural-acceptance.json', capture=True,
        capture_limit=128*2**20, expected_sha=value['structural_acceptance']['sha256'])
    full = json.loads(full_bytes)
    need(full.get('structural_accepted') is True and full.get('actual_outer_updates') == 200 and
         full.get('run_id') == reader.RUN, 'structural receipt is not actual full200')
    return dict(status=HELD_STATUS, source=publication, failure=failure,
        result_path=str(DIRECTORY/'result.json'), result_sha256=ctl.digest(DIRECTORY/'result.json'),
        output=output, structural_acceptance=full_file, actual_child_exit_code=0,
        original_accounting_remains_failed=True, goal_complete=False,
        native4_restore_and_all_archives_and_independent_final_still_required=True)


def inspect(commit):
    caps()
    if not DIRECTORY.exists():
        return dict(status='NOT_STARTED', directory=str(DIRECTORY), goal_complete=False)
    need(DIRECTORY.resolve() == DIRECTORY and DIRECTORY.stat().st_uid == ctl.OWNER,
         'unsafe CPU recovery once directory')
    claim = ctl.read(DIRECTORY/'claim.json')
    need(claim['source']['commit'] == commit, 'existing scope belongs to another source; never replay')
    parent = ctl.read(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else None
    child = ctl.read(DIRECTORY/'child.json') if (DIRECTORY/'child.json').exists() else None
    result = ctl.read(DIRECTORY/'result.json') if (DIRECTORY/'result.json').exists() else None
    if result is not None and result.get('supervisor_exit_code') == 0 and parent and not ctl.live(parent):
        return consume(commit)
    return dict(status=('ACTUAL_FINAL_CPU_REVALIDATION_LIVE' if parent and ctl.live(parent) else
        'ACTUAL_FINAL_CPU_REVALIDATION_FAILED_INSPECT' if result else 'UNKNOWN_CPU_RECOVERY_NO_REPLAY'),
        process=parent, child=child, actual_supervisor_live=bool(parent and ctl.live(parent)),
        actual_child_live=bool(child and ctl.live(child)), result=result, goal_complete=False,
        checked_unix=time.time())


def start(commit):
    caps()
    if DIRECTORY.exists():
        return inspect(commit)
    failure, snapshots = failure_evidence()
    repository, publication = ctl.source(commit)
    os.environ['QPROBE_SOURCE'] = str(repository)
    try:
        DIRECTORY.mkdir(mode=0o770)
    except FileExistsError:
        return inspect(commit)
    command = [sys.executable, '-m', MODULE, 'supervise', '--commit', commit]
    ctl.publish(DIRECTORY/'claim.json', dict(source=publication, failure=failure,
        command=command, cwd=str(repository), created_unix=time.time(),
        training_or_gpu_allocation_permitted=False, original_accounting_remains_failed=True))
    for name, raw in snapshots.items():
        path = DIRECTORY/'failure'/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    # Revalidate the original and the copied bytes before any CPU child launch.
    preserved_failure(ctl.read(DIRECTORY/'claim.json'))
    with (DIRECTORY/'supervisor.stdout').open('xb') as out, (DIRECTORY/'supervisor.stderr').open('xb') as err:
        process = subprocess.Popen(command, cwd=str(repository), stdout=out, stderr=err, start_new_session=True)
    birth = ctl.birth(ctl.process(process.pid))
    need(birth['pid'] == birth['sid'] and birth['uid'] == ctl.OWNER, 'actual detached CPU supervisor differs')
    ctl.publish(DIRECTORY/'process.json', birth)
    return dict(status='ACTUAL_FAILED_FINAL_CPU_REVALIDATION_STARTED_NOT_ACCEPTED',
        process=birth, directory=str(DIRECTORY), goal_complete=False)


def supervise(commit):
    caps(); repository, publication = ctl.source(commit)
    os.environ['QPROBE_SOURCE'] = str(repository)
    own = ctl.birth(ctl.process(os.getpid())); started = time.time()
    process = None; child = None; child_code = None; error = None; code = 1
    try:
        deadline = time.monotonic()+15
        while not (DIRECTORY/'process.json').exists():
            need(time.monotonic() < deadline, 'launcher birth absent; never replay')
            time.sleep(.05)
        claim = ctl.read(DIRECTORY/'claim.json')
        need(ctl.read(DIRECTORY/'process.json') == own and claim['source'] == publication and
             claim['cwd'] == str(repository), 'actual CPU supervisor claim/source/birth differs')
        preserved_failure(claim)
        command = [sys.executable, '-m', MODULE, 'task', '--commit', commit]
        with (DIRECTORY/'consumer.stdout').open('xb') as out, (DIRECTORY/'consumer.stderr').open('xb') as err:
            process = subprocess.Popen(command, cwd=str(repository), stdout=out, stderr=err)
            child = ctl.birth(ctl.process(process.pid))
            need(child['uid'] == ctl.OWNER and child['sid'] == own['sid'] and child['pid'] != own['pid'],
                 'actual held CPU child identity differs')
            ctl.publish(DIRECTORY/'child.json', child)
            child_code = process.wait()
            for stream in (out, err):
                stream.flush(); os.fsync(stream.fileno())
        need(child_code == 0 and not ctl.live(child), 'actual structural CPU child did not return0')
        need((DIRECTORY/'consumer.stderr').stat().st_size == 0, 'structural CPU child stderr is nonempty')
        value, _ = reader.document(DIRECTORY/'consumer.stdout')
        accept_output(value, commit, claim['failure']['fingerprint'])
        preserved_failure(claim); code = 0
    except Exception as exc:
        error = repr(exc)
        import traceback
        traceback.print_exc()
        if process is not None and child_code is None:
            child_code = process.wait()
    outputs = {}
    for name in ('stdout', 'stderr'):
        path = DIRECTORY/('consumer.'+name)
        if path.exists():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
            outputs[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=ctl.digest(path))
    ctl.publish(DIRECTORY/'result.json', dict(source=publication,
        claim_sha256=ctl.digest(DIRECTORY/'claim.json'), process=own, child=child,
        actual_child_exit_code=child_code, supervisor_exit_code=code, error=error,
        started_unix=started, completed_unix=time.time(), goal_complete=False, **outputs))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('start', 'supervise', 'task', 'inspect'))
    parser.add_argument('--commit', required=True); args = parser.parse_args()
    if args.mode == 'supervise':
        sys.exit(supervise(args.commit))
    value = task(args.commit) if args.mode == 'task' else start(args.commit) if args.mode == 'start' else inspect(args.commit)
    print(json.dumps(value, allow_nan=False), flush=True)
