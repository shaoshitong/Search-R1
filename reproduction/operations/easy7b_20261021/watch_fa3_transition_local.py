"""Bounded read-only FA3 followup watcher; never stops, submits or restores.

Exit0 requires actual five completed post-CP12 updates plus saved next-outer
generation. Exit2 requests inspection of observed errors; exit3 means the
bounded window ended without enough evidence. Neither case mutates training.
Credentials are handled only by the existing RAM/stdin bridge.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import time


OPERATOR_COMMIT = '0110264b81133a61515e89af52854d7c0d8cefae'
BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp12-20261022'
RUN = '232f20e112ac4a978c5aba52a0367195'
TERMINAL = {'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY',
            'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'COMPLETED'}
ERROR = re.compile(r'Traceback \(most recent call last\)|CUDA out of memory|RayTaskError|'
    r'OutOfMemoryError|NCCLError|FloatingPointError|(?:ModuleNotFound|Import|Runtime|Value|'
    r'FileNotFound|Permission|Assertion|Key)Error\b|undefined symbol|Segmentation fault|'
    r'slurmstepd: error:|Error executing job with overrides|CUBLAS_STATUS')


def load_bridge(workspace):
    path = workspace/'work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py'
    spec = importlib.util.spec_from_file_location('_fa3_existing_ram_bridge', path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    return bridge


def add_runtime_evidence(value, reader, base):
    """Project small ledgers only; do not repeatedly persist the large run manifest."""
    entry = value.get('documents', {}).get('logs/run-manifest.json', {})
    run = entry.get('value', {})
    if isinstance(run, dict):
        compact = {key:run.get(key) for key in ('run_id', 'source_commit', 'training_world_size',
            'actor_micro_batch_per_rank', 'actor_logical_micro_batch_per_rank', 'infra_profile')}
        binding = run.get('full_state_restore', {})
        compact['full_state_restore'] = {key:binding.get(key) for key in (
            'kind', 'completed_outer', 'source_job', 'source_commit', 'source_base',
            'source_receipt', 'source_receipt_sha256', 'actual_optimizer_writes_per_rank',
            'candidate_source_cursor', 'reward_probe_round', 'reward_probe_version')}
        entry['value'] = compact
        entry['projection'] = 'Identity and restore boundary only; no repeated full run manifest'
    keys = ('step', 'run_id', 'slurm_job_id', 'attempt_id', 'logical_update', 'rank', 'kind',
            'update_status', 'status', 'completed_outer', 'complete_outer', 'all_ranks_agreed',
            'real_rank_count', 'full_budget_complete', 'opportunities_processed', 'opportunities_unattempted',
            'actual_optimizer_writes', 'optimizer_writes_before', 'optimizer_writes_after',
            'optimizer_calls_this_attempt', 'optimizer_writes_this_attempt', 'optimizer_rollbacks_this_attempt',
            'scheduler_advances', 'writes_before', 'writes_after', 'scheduler_last_epoch_after')
    value['post_restore_ledger_observation_errors'] = []
    for name, field in (('actual-updates.jsonl', 'post_restore_update_records'),
                        ('actual-attempts.jsonl', 'post_restore_attempt_records')):
        rows, errors = reader.complete_jsonl(base/'logs'/name)
        value['post_restore_ledger_observation_errors'].extend(errors)
        value[field] = [{key:row[key] for key in keys if key in row} for row in rows
                        if type(row.get('logical_update', row.get('step'))) is int and
                        row.get('logical_update', row.get('step')) >= 13]
    value['post_restore_scheduler_records'] = {}
    for rank in range(4):
        rows, errors = reader.complete_jsonl(base/('logs/scheduler-rank-%d.jsonl' % rank))
        value['post_restore_ledger_observation_errors'].extend(errors)
        value['post_restore_scheduler_records'][str(rank)] = [
            {key:row[key] for key in keys if key in row} for row in rows
            if type(row.get('logical_update')) is int and row['logical_update'] >= 13]


def command(commit):
    # Older published observers have no top-level Slurm tail. This fallback
    # reads the same small log in the observer process, not the producer tree.
    code = (
        'import json,re,sys\nfrom pathlib import Path\n'
        'sys.dont_write_bytecode=True\n'
        'sys.path.insert(0,'+repr(BASE+'/source/'+commit)+')\n'
        'from reproduction.operations.easy7b_20261021 import observe_fa3_transition_hpc as o\n'
        'value=o.observe()\n'+inspect.getsource(add_runtime_evidence)+'\n'
        'add_runtime_evidence(value,o.private_progress_reader(),o.BASE)\n'
        "job=value.get('jobs',{}).get('new_training')\n"
        "if 'new_training_slurm_log' not in value and isinstance(job,str) and re.fullmatch(r'[0-9]+',job):\n"
        " value['new_training_slurm_log']=o.private_progress_reader().tail(o.BASE/('logs/training-%s.log'%job),18000)\n"
        'print(json.dumps(value,sort_keys=True,allow_nan=False),flush=True)\n')
    return shlex.join(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python', '-B', '-c', code])


def document(value, name):
    row = value.get('documents', {}).get(name, {})
    return row.get('value', {}) if isinstance(row, dict) and row.get('present') is True else {}


def job_state(value, job):
    if not job:
        return None
    queue = value.get('queue', {})
    if queue.get('rc') == 0:
        for line in queue.get('stdout', '').splitlines():
            fields = line.strip().split('|')
            if len(fields) > 1 and fields[0] == job:
                return fields[1].split()[0].split('+')[0]
    accounting = value.get('accounting', {})
    if accounting.get('rc') == 0:
        for line in accounting.get('stdout', '').splitlines():
            fields = line.strip().split('|')
            if len(fields) > 1 and fields[0] == job:
                return fields[1].split()[0].split('+')[0]
    return None


def restoration(value, job):
    workers = document(value, 'logs/full-state-restored-workers.json')
    driver = document(value, 'logs/full-state-restored-driver.json')
    stream = document(value, 'logs/easy2560-restored-stream-before-next-outer.json')
    run = document(value, 'logs/run-manifest.json')
    binding = run.get('full_state_restore', {})
    source = binding.get('completed_outer')
    rows = workers.get('ranks', [])
    if source != 12 or len(rows) != 4 or sorted(row.get('rank', -1) for row in rows) != list(range(4)):
        return False
    if workers.get('run_id') != RUN or workers.get('slurm_job_id') != job or workers.get('source_step') != 12 or \
            workers.get('actual_world_size') != 4 or workers.get('actual_optimizer_writes_per_rank') != 240 or \
            run.get('run_id') != RUN or binding.get('actual_optimizer_writes_per_rank') != 240 or \
            binding.get('source_job') != '745371':
        return False
    for row in rows:
        individual = value.get('actual_restore_ranks', {}).get(str(row['rank']), {})
        if individual.get('present') is not True or individual.get('value') != row or \
                row.get('slurm_job_id') != job or row.get('run_id') != RUN or row.get('source_step') != 12 or \
                row.get('actual_optimizer_writes') != 240 or row.get('scheduler_last_epoch') != 12 or \
                any(row.get(key) is not True for key in ('actual_model_collective_readback_equal',
                    'actual_adam_scheduler_readback_equal', 'actual_rng_readback_equal')):
            return False
    return bool(driver.get('slurm_job_id') == job and driver.get('run_id') == RUN and
        driver.get('source_step') == driver.get('completed_outer') == 12 and
        driver.get('source_job') == '745371' and driver.get('actual_driver_dataequal') is True and
        driver.get('actual_candidate_source_cursor') == 30720 and
        stream.get('completed_outer') == 12 and stream.get('actual_candidate_stream_readback_equal') is True and
        stream.get('candidate_source_cursor') == 30720 and stream.get('actual_driver_rng_readback_equal') is True)


def summarize(value, completed, next_outer):
    progress = value.get('new_training_progress', {})
    job = value.get('jobs', {}).get('new_training')
    state = job_state(value, job)
    service_state = job_state(value, value.get('jobs', {}).get('retrieval_service'))
    metrics = progress.get('completed_outer_metrics', [])
    wanted = list(range(13, completed+1))
    rows = [row for row in metrics if row.get('step') in wanted]
    complete = ([row.get('step') for row in rows] == wanted and
                all(row.get('optimizer_writes_this_outer') == 20 and
                    row.get('optimizer_writes') == row['step']*20 for row in rows))
    actual_rows = [row for row in value.get('post_restore_update_records', []) if row.get('step') in wanted]
    actual_complete = ([row.get('step') for row in actual_rows] == wanted and all(
        row.get('run_id') == RUN and row.get('update_status') == 'complete' and
        row.get('actual_optimizer_writes') == 20 and row.get('full_budget_complete') is True and
        row.get('opportunities_processed') == 20 and row.get('opportunities_unattempted') == 0 and
        row.get('optimizer_writes_before') == (row['step']-1)*20 and
        row.get('optimizer_writes_after') == row['step']*20 for row in actual_rows))
    attempts = [row for row in value.get('post_restore_attempt_records', []) if row.get('logical_update') in wanted]
    attempts_complete = ([row.get('logical_update') for row in attempts] == wanted and all(
        row.get('run_id') == RUN and row.get('slurm_job_id') == job and row.get('status') == 'complete' and
        row.get('complete_outer') is True and row.get('all_ranks_agreed') is True and
        row.get('real_rank_count') == 4 and row.get('full_budget_complete') is True and
        row.get('optimizer_writes_this_attempt') == 20 and row.get('scheduler_advances') == 1 and
        row.get('optimizer_writes_after') == row['logical_update']*20 for row in attempts))
    latest = progress.get('preparation/rolling-checkpoints/latest.json', {})
    sealed = (latest.get('status') == 'EASY2560_ROLLING_CHECKPOINT_READY' and latest.get('run_id') == RUN and
              type(latest.get('step')) is int and latest['step'] >= completed)
    pools = progress.get('candidate_pools', [])
    saved = any(type(pool.get('step')) is int and pool['step'] >= next_outer and
        type(pool.get('completed_questions')) is int and pool['completed_questions'] > 0 and
        type(pool.get('saved_chunk_files')) is int and pool['saved_chunk_files'] > 0 for pool in pools)
    audits, counters, audit_complete = value.get('actual_flash3_execution', {}), {}, True
    for rank in range(4):
        row = audits.get(str(rank), {}).get('latest_actual_record', {})
        forwards = row.get('forward_success_returns', {})
        count = sum(v for v in forwards.values() if type(v) is int)
        backward = row.get('kernel_backward_success_returns', 0)
        counters[str(rank)] = dict(forward_returns=count, backward_returns=backward,
            outer=row.get('logical_update'), writes=row.get('actual_optimizer_writes'))
        audit_complete = audit_complete and (row.get('rank') == rank and row.get('run_id') == RUN and
            row.get('actual_flash3_training_executed') is True and row.get('cuda_synchronized') is True and
            type(row.get('logical_update')) is int and row['logical_update'] >= completed and
            row.get('actual_optimizer_writes') == row['logical_update']*20 and
            row['actual_optimizer_writes'] >= completed*20 and
            count > 0 and type(backward) is int and backward > 0)
    logs = {'slurm': value.get('new_training_slurm_log', {}),
            'training': progress.get('training_log', {}), 'trainer': progress.get('trainer_log', {})}
    errors = {name: [line[-1800:] for line in record.get('tail', '').splitlines() if ERROR.search(line)][-8:]
              for name, record in logs.items()}
    issues = []
    scheduler_ok, scheduler_epochs = True, {}
    for rank in range(4):
        records = value.get('post_restore_scheduler_records', {}).get(str(rank), [])
        ends = [row for row in records if row.get('kind') == 'COMPLETE' and row.get('logical_update') in wanted]
        scheduler_epochs[str(rank)] = ends[-1].get('scheduler_last_epoch_after') if ends else None
        scheduler_ok = scheduler_ok and [row.get('logical_update') for row in ends] == wanted and all(
            row.get('run_id') == RUN and row.get('slurm_job_id') == job and row.get('rank') == rank and
            row.get('actual_optimizer_writes') == 20 and row.get('full_budget_complete') is True and
            row.get('scheduler_advances') == 1 and row.get('scheduler_last_epoch_after') == row['logical_update'] and
            row.get('writes_before') == (row['logical_update']-1)*20 and
            row.get('writes_after') == row['logical_update']*20 for row in ends)
        for row in ends:
            if row.get('actual_optimizer_writes') != 20 or row.get('full_budget_complete') is not True:
                issues.append('Observed partial PPO writes on rank%d outer%s: %s' % (rank, row.get('logical_update'), row))
    for row in actual_rows:
        if row.get('actual_optimizer_writes') != 20 or row.get('full_budget_complete') is not True:
            issues.append('Observed partial PPO update: '+str(row))
    for row in value.get('post_restore_attempt_records', []):
        if row.get('logical_update') in wanted and (row.get('status') != 'complete' or
                row.get('optimizer_rollbacks_this_attempt', 0) != 0 or row.get('optimizer_writes_this_attempt') != 20):
            issues.append('Observed guarded/partial/aborted PPO attempt: '+str(row))
    if value.get('post_restore_ledger_observation_errors'):
        issues.append('Actual ledger parsing errors: '+str(value['post_restore_ledger_observation_errors']))
    if any(errors.values()): issues.append('Actual startup/training error observed in log tail')
    if state in TERMINAL: issues.append('New training job is terminal: '+state)
    if service_state in TERMINAL: issues.append('Retrieval service is terminal: '+service_state)
    if progress.get('status') == 'PROGRESS_READER_OBSERVATION_ERROR':
        issues.append('Remote progress reader failed: '+str(progress.get('observation_error')))
    log_present = any(type(record.get('bytes')) is int and record['bytes'] > 0 for record in logs.values())
    restore_ok = restoration(value, job)
    success = bool(job and state == 'RUNNING' and service_state == 'RUNNING' and log_present and
                   complete and actual_complete and attempts_complete and scheduler_ok and sealed and saved and
                   restore_ok and audit_complete and not issues)
    last = metrics[-1] if metrics else {}
    summary = dict(epoch=value.get('epoch'), job=job, job_state=state, retrieval_service_state=service_state,
        phase=document(value, 'logs/pipeline-status.json').get('stage'), checkpoint_outer=latest.get('step'),
        completed_outer=last.get('step'), reward=last.get('reward'),
        required_completed_outers=wanted, five_updates_observed=bool(complete and actual_complete and attempts_complete),
        selected_update_writes=[dict(step=row.get('step'), writes=row.get('optimizer_writes_this_outer')) for row in rows],
        candidate_pools=[{key:pool.get(key) for key in ('name', 'step', 'completed_questions', 'saved_chunk_files')}
                         for pool in pools], next_outer_saved=saved, actual_all4_restore_observed=restore_ok,
        actual_flash3_counters=counters, all4_flash3_through_target_observed=bool(audit_complete),
        all4_scheduler_epochs=scheduler_epochs, all4_scheduler_writes_observed=bool(scheduler_ok),
        log_present=log_present, log_bytes={name:record.get('bytes') for name, record in logs.items()},
        error_lines=errors, inspection_issues=issues, requested_followup_observed=success,
        overall_goal_complete=False, production_mutated=False)
    return summary


def store(path, value):
    temporary = path.with_name(path.name+'.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-commit', default=OPERATOR_COMMIT)
    parser.add_argument('--completed-outer', type=int, default=17)
    parser.add_argument('--next-outer', type=int, default=18)
    parser.add_argument('--observations', type=int, default=180)
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args(argv)
    if re.fullmatch('[0-9a-f]{40}', args.operator_commit) is None or \
            not (17 <= args.completed_outer < args.next_outer <= 200) or \
            not (1 <= args.observations <= 180 and 10 <= args.interval <= 60):
        parser.error('Require full operatorSHA, target17..199, later next<=200, observations1..180, interval10..60')
    repo = Path(__file__).resolve().parents[3]
    workspace = repo.parents[1]
    bridge = load_bridge(workspace)
    output = workspace/'outputs/easy2560-qwen7b-20261021'/('fa3-transition-watch-'+str(time.time_ns()))
    output.mkdir(parents=True)
    remote = command(args.operator_commit)
    previous = None
    for index in range(args.observations):
        wrapper = {}
        try:
            result = bridge.run('jxu957', remote, '', str(workspace/'work'), login_host='10.120.48.31',
                expected_host_identity=dict(hostname='ACD-Login-1', boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',
                                            pid_namespace='pid:[4026531836]'), observation_timeout=55)
            wrapper = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            if result.returncode:
                raise ValueError('Read-only transport returned '+str(result.returncode))
            value = json.loads(result.stdout)
            summary = summarize(value, args.completed_outer, args.next_outer)
        except Exception as error:
            summary = dict(epoch=time.time(), observation_unknown=True, observation_error=str(error),
                inspection_issues=['Read-only observation failed; inspect before another observation'],
                requested_followup_observed=False, overall_goal_complete=False, production_mutated=False,
                training_failure_inferred=False)
        summary.update(observation=index+1, operator_commit=args.operator_commit)
        # Epoch and ever-growing log bytes do not themselves constitute a
        # phase/pool/writes state change deserving another full wrapper.
        stable = {key:value for key,value in summary.items() if key not in ('epoch', 'observation', 'log_bytes')}
        fingerprint = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
        changed = fingerprint != previous
        if changed or summary.get('inspection_issues'):
            path = output/('observation-%03d.json' % (index+1))
            store(path, wrapper)
            summary['full_observation_path'] = str(path)
        with (output/'summaries.jsonl').open('a') as stream:
            stream.write(json.dumps(summary, sort_keys=True, allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        store(output/'latest-summary.json', summary)
        if changed or summary.get('inspection_issues'):
            print(json.dumps(summary, sort_keys=True), flush=True)
        previous = fingerprint
        if summary.get('inspection_issues'):
            print('Inspection required; no job was restarted, stopped, submitted or changed.', flush=True)
            return 2
        if summary.get('requested_followup_observed'):
            print('Actual requested recovery followup observed; the outer200 goal remains active.', flush=True)
            return 0
        if index+1 < args.observations:
            time.sleep(args.interval)
    print('Bounded observation window ended without sufficient evidence; production unchanged.', flush=True)
    return 3


if __name__ == '__main__':
    raise SystemExit(main())
