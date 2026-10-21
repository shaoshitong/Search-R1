"""Read-only small-file observer for the CP12-or-later FA3 transition.

The production progress reader is privately loaded with only its BASE rebound.
No tensor, credential, model, submission or training mutation is performed.
Missing receipts and a missing new job are observations, not acceptance.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import time


BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp12-20261022')
JOBS = dict(source='745371', retrieval_service='745198', sdpa_diagnostic='746114',
            source_state_read='746117')
DOCUMENTS = (
    'preparation/fa3-transition-prepared.json',
    'preparation/fa3-source-stop-result.json',
    'preparation/fa3-transition-state-read-r2-submission.json',
    'preparation/fa3-transition-source-ready.json',
    'preparation/fa3-transition-submitted.json',
    'preparation/training-submission.json',
    'logs/run-manifest.json',
    'logs/pipeline-status.json',
    'logs/full-state-restored-workers.json',
    'logs/full-state-restored-driver.json',
    'logs/easy2560-restored-stream-before-next-outer.json',
    'diagnostics/fa3-sdpa-math/submit-receipt.json',
    'diagnostics/fa3-sdpa-math/result.json',
)


def private_progress_reader():
    path = Path(__file__).with_name('observe_training_progress.py')
    spec = importlib.util.spec_from_file_location('_easy7b_fa3_private_progress', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BASE = BASE
    return module


def document(reader, path):
    if not path.is_file():
        return dict(path=str(path), present=False)
    return dict(path=str(path), present=True, value=reader.read_json(path))


def new_job():
    path = BASE/'job.id'
    try:
        value = path.read_text().strip()
    except OSError as error:
        return None, dict(path=str(path), present=False, observation_error=str(error))
    if re.fullmatch(r'[0-9]+', value) is None:
        return None, dict(path=str(path), present=True, observation_error='Invalid numeric job.id')
    return value, dict(path=str(path), present=True, job=value)


def execution_audits(reader):
    result = {}
    for rank in range(4):
        path = BASE/('logs/actual-flash3-execution-rank-%d.jsonl' % rank)
        rows, errors = reader.complete_jsonl(path)
        value = dict(path=str(path), present=path.is_file(), complete_records=len(rows),
                     observation_errors=errors)
        if rows:
            # The actual synchronized producer record is retained verbatim;
            # these cumulative returns are not inferred from the FA3 profile.
            value['latest_actual_record'] = rows[-1]
        result[str(rank)] = value
    return result


def observe():
    reader = private_progress_reader()
    job, job_record = new_job()
    jobs = dict(JOBS)
    if job is not None:
        jobs['new_training'] = job
    identifiers = ','.join(dict.fromkeys(jobs.values()))
    result = dict(epoch=time.time(), base=str(BASE), read_only=True, tensor_files_read=False,
                  credentials_read=False, jobs=jobs, new_job_record=job_record,
                  training_runtime_edited=False, jobs_submitted=False)
    result['queue'] = reader.command(['squeue', '-j', identifiers, '-h', '-o', '%i|%T|%N|%R|%S|%L'])
    result['accounting'] = reader.command(['sacct', '-j', identifiers, '-n', '-P', '-o',
                                            'JobIDRaw,State,ExitCode,Elapsed,NodeList'])
    result['documents'] = {name: document(reader, BASE/name) for name in DOCUMENTS}
    result['actual_restore_ranks'] = {
        str(rank): document(reader, BASE/('logs/full-state-restored-rank-%d.json' % rank))
        for rank in range(4)}
    result['actual_flash3_execution'] = execution_audits(reader)
    result['actual_flash3_execution_scope'] = (
        'Complete persisted producer audit records only; counters are cumulative successful forward/backward '
        'returns, not a claim that a new outer or full-state recovery completed.')
    audit_jobs = {JOBS['source_state_read']}
    submission = result['documents']['preparation/fa3-transition-state-read-r2-submission.json'].get('value', {})
    recorded_audit_job = submission.get('job') if isinstance(submission, dict) else None
    if isinstance(recorded_audit_job, str) and re.fullmatch(r'[0-9]+', recorded_audit_job):
        audit_jobs.add(recorded_audit_job)
    result['source_cpu_audit_log_tails'] = {
        audit_job: reader.tail(BASE/('logs/fa3-transition-source-audit-%s.log' % audit_job), 14000)
        for audit_job in sorted(audit_jobs)}
    result['sdpa_diagnostic_log_tail'] = reader.tail(
        BASE/('diagnostics/fa3-sdpa-math/slurm-%s.log' % JOBS['sdpa_diagnostic']), 14000)
    if job is None:
        result['new_training_progress'] = dict(status='NEW_JOB_NOT_YET_OBSERVED',
            job_record=job_record,
            training_log=reader.tail(BASE/'logs/search-g1-common8-training.log', 5000))
    else:
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                reader.main()
            result['new_training_progress'] = json.loads(captured.getvalue())
        except Exception as error:
            result['new_training_progress'] = dict(
                status='PROGRESS_READER_OBSERVATION_ERROR', observation_error=str(error),
                captured_stdout_tail=captured.getvalue()[-5000:])
    return result


if __name__ == '__main__':
    print(json.dumps(observe(), sort_keys=True, allow_nan=False), flush=True)
