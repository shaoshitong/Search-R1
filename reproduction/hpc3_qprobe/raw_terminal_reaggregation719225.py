"""Pure fresh conjunction after original RAW795 retained every actual child.

The original failed reader did not persist an exception call site. This reader
does not invent one or rewrite that failure as PASS. It requires the separately
typed failed producer and all original normal-return child wait0 receipts, then
freshly reads original raw/private/scientific bytes and rechecks every attempt.
It creates no scope, cache, receipt, process or job and never decodes again. Its
output still requires the independent final holder's actual wait0 consumption.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from reproduction.hpc3_qprobe import final_control as ctl
from reproduction.hpc3_qprobe import independent_goal_acceptance719225 as reader

BASE = reader.BASE
ROOT = reader.ROOT
RAW = reader.RAW
ORIGINAL = ROOT/'observations/raw-all200-719225-20261015'
STATUS = 'ACTUAL_FRESH_RAW_ALL200_RECONJUNCTION_AFTER_ORIGINAL_CHILDREN_HELD0_PASS'
FAILED_STATUS = 'ACTUAL_READONLY_AUDIT_STOPPED_EVIDENCE_RETAINED'
MODULE = 'reproduction.hpc3_qprobe.raw_terminal_reaggregation719225'
# Fresh original claim read on 2026-10-15: sys.executable is the prepared
# runtime's real executable, rather than its BASE/env/bin/python alias.
ORIGINAL_RAW_EXECUTABLE = ROOT.parent/'cosearch-continue300-x203-20261008/runtime/retriever-python310-candidate/bin/python3.10'


def need(ok, message):
    if not ok:
        raise ValueError('RAW_TERMINAL_REAGGREGATION719225: '+message)


def caps():
    ctl.owner()
    for name in reader.CAPS:
        os.environ[name] = '1'
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    sys.dont_write_bytecode = True


def stopped_read_error(error_type, message):
    """Read-error class only; RAW795 did not save a traceback/call-site phase."""
    need(isinstance(message, str) and len(message) < 4096, 'bounded original error required')
    bare = {'FileNotFoundError':'[Errno 2] No such file or directory',
            'OSError':'[Errno 116] Stale file handle'}
    need(error_type in bare, 'original RAW error is not a missing/stale read')
    prefix = bare[error_type]
    allowed = {str(base/'logs'/name) for base in (BASE, reader.OLD_BASE)
        for name in ('process-ledger.json', 'actual-attempts.jsonl', 'run-manifest.json',
                     'native-allocation-admission.json')}
    need(message == prefix or any(message == prefix+': '+repr(path) for path in allowed),
         'original RAW read error names another input or contains another failure')
    return dict(error_type=error_type, original_error=message,
        original_exception_call_site_persisted=False,
        precise_original_aggregation_call_failure_proven=False,
        scope='all original child audits retained; original internal stopped read location is unproven')


def validate_terminal(claim, process, result, state, output, publication, parts, tasks, *, alive):
    need(claim.get('source') == publication and claim.get('parts') == parts and
         claim.get('command') == [str(ORIGINAL_RAW_EXECUTABLE), '-m',
             'reproduction.hpc3_qprobe.raw_audit_followthrough719225', '--commit', RAW, '--worker'] and
         claim.get('training_control_permitted') is False and claim.get('goal_complete') is False and
         type(claim.get('created_unix')) in (int, float) and math.isfinite(claim['created_unix']),
         'original RAW claim/source/parts/entry/scope differs')
    need(reader.birth_ok(process) and process['sid'] == process['pid'] and not alive(process) and
         result.get('process') == process and result.get('source') == publication and
         result.get('status') == FAILED_STATUS and result.get('goal_complete') is False and
         type(result.get('completed_attempt_audits')) is int and result['completed_attempt_audits'] == len(tasks) and
         all(type(result.get(key)) in (int, float) and math.isfinite(result[key])
             for key in ('started_unix', 'completed_unix')) and
         result['completed_unix'] >= result['started_unix'] >= claim['created_unix'],
         'original RAW is live, unknown, successful or stopped before all child receipts')
    need(len(tasks) >= 200 and [t['attempt_index'] for t in tasks] == list(range(1, len(tasks)+1)) and
         [t['step'] for t in tasks if t['complete_outer']] == list(range(1, 201)) and
         all(type(t['step']) is int and 1 <= t['step'] <= 200 and type(t['complete_outer']) is bool for t in tasks),
         'all200/all-normal-return original task set missing, copied or logical201')
    need(state.get('process') == process, 'original auxiliary RAW state belongs to another process')
    if state.get('child') is not None:
        child = state['child']
        need(reader.birth_ok(child) and child['sid'] == process['sid'] and not alive(child),
             'original auxiliary RAW child is live or differs')
    lines = [line for line in output.splitlines() if line.strip()]
    need(lines and json.loads(lines[-1]) == result,
         'original RAW full supervisor output does not retain the same failure result')
    return stopped_read_error(result.get('error_type'), result.get('error'))


def original_failure(recovery_commit):
    # No source traversal or full raw read until typed FAILED/full200 admission.
    run, binding, accounting, recovery = ctl.final_producer(BASE, recovery_commit=recovery_commit)
    need(recovery is not None and recovery['commit'] == recovery_commit and
         binding['producer_job'] == '719225' and run['run_id'] == reader.RUN,
         'actual typed failed full200 producer required')
    raw, publication = reader.load_fixed(RAW, 'raw_audit_followthrough719225', 'ctl')
    claim, claim_file = reader.document(ORIGINAL/'claim.json')
    process, process_file = reader.document(ORIGINAL/'process.json')
    result, result_file = reader.document(ORIGINAL/'result.json')
    state, state_file = reader.document(ORIGINAL/'state.json')
    log_bytes, log_file = reader.read_file(ORIGINAL/'supervisor.log', capture=True)
    need(result.get('claim_sha256') == claim_file['sha256'], 'original RAW failure claim SHA differs')
    parts = raw.plan(); tasks = raw.actual_tasks(parts)
    reason = validate_terminal(claim, process, result, state, log_bytes.decode(), publication, parts, tasks, alive=ctl.live)
    # accepted and attempts are exact original sets, not only a latest prefix.
    accepted = ORIGINAL/'accepted'; attempts = ORIGINAL/'attempts'
    for folder in (accepted, attempts):
        need(folder.resolve() == folder and folder.is_dir() and folder.stat().st_uid == ctl.OWNER,
             'original RAW retained child namespace differs')
    need({p.name for p in accepted.iterdir()} == {'%06d.json'%t['attempt_index'] for t in tasks} and
         {p.name for p in attempts.iterdir()} == {t['attempt_id'] for t in tasks},
         'missing or extra accepted/normal-return original child scope')
    anchors = dict(claim=claim_file, process=process_file, result=result_file, state=state_file, log=log_file)
    stable = dict(source=publication, producer=binding, producer_final_cpu_recovery=recovery,
        original_producer_accounting=accounting, original_process=process, original_failure=result,
        original_error_boundary=reason, original_file_anchors={key:{k:row[k] for k in ('path', 'sha256', 'bytes')}
            for key, row in anchors.items()}, normal_return_attempts=len(tasks),
        successful_formal_outers=list(range(1, 201)), original_raw_status_remains_failed=True,
        original_raw_external_parent_wait_receipt_available=False)
    return raw, tasks, dict(stable, fingerprint=reader.canonical_sha(stable)), anchors


def reconsume_children(raw, tasks, process, shared):
    records = []; originals = []
    for task in tasks:
        directory = ORIGINAL/'attempts'/task['attempt_id']
        record = raw.consume_task(task, directory, process)
        # Original795 used equality with0. Tighten bool/float ambiguity in this
        # new consumer without rewriting any original child receipt.
        result, result_file = reader.document(directory/'result.json')
        need(type(result.get('actual_child_exit_code')) is int and result['actual_child_exit_code'] == 0 and
             result.get('worker_error') is None and result_file['sha256'] == record['result_sha256'],
             'original child did not have an actual integer wait0')
        accepted, accepted_file = reader.document(ORIGINAL/'accepted'/('%06d.json'%task['attempt_index']))
        need(accepted == record, 'original accepted child differs from actual reconsumed wait/output')
        output, output_file = reader.document(directory/'audit.stdout')
        need(output_file['sha256'] == record['output_sha256'], 'original child stdout changed')
        child_claim, child_claim_file = reader.document(directory/'claim.json')
        child, child_file = reader.document(directory/'child.json')
        stderr_file = reader.read_file(directory/'audit.stderr', expected_sha=result['stderr']['sha256'],
            expected_bytes=result['stderr']['bytes'])[1]
        need(child_claim_file['sha256'] == result['claim_sha256'] and child_claim.get('task') == task and
             child_claim.get('process') == process and child_claim.get('source') == result['source'] and
             child == result['child'] and reader.birth_ok(child) and child['sid'] == process['sid'] and
             not ctl.live(child) and stderr_file['bytes'] == 0, 'original child claim/birth/stderr changed')
        for proof in (accepted_file, result_file, output_file, child_claim_file, child_file, stderr_file):
            shared.remember(proof)
        records.append(record); originals.append(reader.fresh_attempt(task, output, shared))
    return records, originals


def recheck_anchors(anchors):
    for row in anchors.values():
        reader.read_file(row['path'], expected_sha=row['sha256'], expected_bytes=row['bytes'])


def verify(commit, recovery_commit):
    caps()
    raw, tasks, failure, anchors = original_failure(recovery_commit)
    repository, publication = reader.publication(commit)
    need(Path(__file__).resolve() == repository/'reproduction/hpc3_qprobe/raw_terminal_reaggregation719225.py' and
         os.environ.get('QPROBE_SOURCE') == str(repository), 'actual immutable fresh reader source differs')
    shared = reader.SharedReads()
    records, originals = reconsume_children(raw, tasks, failure['original_process'], shared)
    from reproduction.hpc3_qprobe.verify_goal import history_parts, verify_attempt_parts
    run, _ = reader.document(BASE/'logs/run-manifest.json')
    attempts, hashes, formal = verify_attempt_parts(history_parts(BASE, run, last=200))
    need([row['attempt_id'] for row in attempts] == [task['attempt_id'] for task in tasks] and
         [row['attempt_index'] for row in attempts] == list(range(1, len(tasks)+1)),
         'fresh formal aggregation omitted or duplicated an original guarded attempt')
    shared.finish(); recheck_anchors(anchors)
    _, final_tasks, final_failure, _ = original_failure(recovery_commit)
    need(final_tasks == tasks and final_failure['fingerprint'] == failure['fingerprint'],
         'typed producer/original RAW failure/task set changed during fresh full reads')
    return dict(status=STATUS, source=publication, original_raw_source=failure['source'],
        producer_final_cpu_recovery=failure['producer_final_cpu_recovery'], original_failure=failure,
        result=anchors['result'], claim=anchors['claim'], normal_return_attempts=len(tasks),
        successful_formal_outers=list(range(1, 201)), fully_skipped_attempts=sum(not t['complete_outer'] for t in tasks),
        actual_record_list_sha256=reader.canonical_sha(records), fresh_original_proofs_sha256=reader.canonical_sha(originals),
        records=records, tasks=tasks, formal_attempt_audits=formal, attempt_evidence_sha256=reader.canonical_sha(hashes),
        original_raw_status_remains_failed=True, original_exception_call_site_persisted=False,
        precise_original_aggregation_call_failure_proven=False, decode_reexecuted=0,
        actor_model_loaded=False, generated_trajectories=0, actor_head_or_scheduler_updates=0,
        receipt_or_cache_or_claim_writes=0, process_or_job_submissions=0,
        independent_final_actual_held_wait0_still_required=True,
        full_objective_accepted=False, goal_complete=False, checked_unix=time.time())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True); parser.add_argument('--recovery-commit', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.commit, args.recovery_commit), allow_nan=False), flush=True)
