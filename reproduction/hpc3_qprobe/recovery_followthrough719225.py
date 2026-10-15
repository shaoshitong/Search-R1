"""Held terminal-only continuation of the precisely recovered final CPU failure.

Training is never restarted. The original failed producer and stopped EA
follower remain immutable. Existing archive intents are consumed or waited on;
only absent intents may be created. A native4 restore-only audit and a separate
CPU goal receipt are intermediate evidence, never full objective acceptance.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from reproduction.hpc3_qprobe import final_control as ctl
from reproduction.hpc3_qprobe import independent_goal_acceptance719225 as reader
from reproduction.hpc3_qprobe import followthrough719225 as normal

ROOT = ctl.ROOT
BASE = reader.BASE
EA = ctl.RECOVERY_NATIVE_AUDIT_SOURCE
ORIGINAL = ROOT/'followthrough/719225'
DIRECTORY = ROOT/'followthrough/recovery719225-finalcpu-20261015'
MODULE = 'reproduction.hpc3_qprobe.recovery_followthrough719225'
ERROR = 'QPROBE_FOLLOWTHROUGH719225_ACTUAL_PRODUCER_FAILED_DIAGNOSE_NO_RESTART'
STATUS = 'ACTUAL_RECOVERY_NATIVE4_AND_GOAL_REVALIDATION_PASS'
HELD_STATUS = 'ACTUAL_HELD_RECOVERY_NATIVE4_AND_GOAL_REVALIDATION_COMPLETED0'
EA_FILES = ('claim.json', 'process.json', 'started.json', 'result.json',
            'state.json', 'producer-terminal-failure.json', 'output.log')


def need(ok, message):
    if not ok:
        raise ValueError('RECOVERY_FOLLOWTHROUGH719225: '+message)


def caps():
    ctl.owner()
    for name in reader.CAPS:
        os.environ[name] = '1'
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    sys.dont_write_bytecode = True


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_ea(documents, output, publication, fixed, actual, *, alive):
    """Pinned EA has self-reported exit1 and a dead birth, no held OS receipt."""
    claim = documents['claim.json']; process = documents['process.json']
    started = documents['started.json']; result = documents['result.json']
    state = documents['state.json']; marker = documents['producer-terminal-failure.json']
    repository = ROOT/'source'/EA
    need(claim.get('schema_version') == 1 and claim.get('source') == publication and
         claim.get('binding') == fixed and claim.get('command') == [str(BASE/'env/bin/python'),
             '-m', 'reproduction.hpc3_qprobe.followthrough719225', 'supervise', '--commit', EA] and
         claim.get('cwd') == str(repository) and finite(claim.get('created_unix')) and
         all(claim.get(key) is False for key in ('training_submission_permitted',
             'actor_model_reload_permitted', 'direct_gpu_access_permitted')) and
         claim.get('native_final_restore_only_submission_permitted') is True,
         'original EA claim/source/entry/scope differs')
    birth = process.get('actual_process')
    need(reader.birth_ok(birth) and birth['pid'] == birth['sid'] and process.get('commit') == EA and
         process.get('control') == str(ORIGINAL) and started.get('actual_process') == birth and
         result.get('actual_process') == birth and not alive(birth),
         'original EA birth is live, unknown or differs')
    expected = repr(ValueError(ERROR))
    need(type(result.get('exit_code')) is int and result['exit_code'] == 1 and
         result.get('error') == expected and state.get('phase') ==
         'ACTUAL_CPU_FOLLOWTHROUGH_STOPPED_FOR_DIAGNOSIS' and state.get('error') == expected and
         state.get('full_objective_accepted') is False and
         all(result.get(key) is False for key in ('training_submission_performed',
             'actor_model_loaded', 'direct_gpu_device_access_performed')) and
         result.get('native_final_restore_only_submission_permitted') is True and
         finite(result.get('checked_unix')) and result['checked_unix'] >= claim['created_unix'],
         'original EA did not stop at the precise producer-failure boundary')
    expected_row = ['719225', 'FAILED', '1:0', 'jxu957', '14090', 'jxu957', 'formal-user', 'emergency_acd']
    need(actual.get('job') == marker.get('job') == '719225' and
         actual.get('state') == marker.get('state') == 'FAILED' and
         actual.get('exit_code') == marker.get('exit_code') == '1:0' and
         [s.split('|') for s in marker.get('raw', '').splitlines() if s.strip()] == [expected_row] and
         [s.split('|') for s in actual.get('raw', '').splitlines() if s.strip()] == [expected_row],
         'original EA failure marker differs from actual owner accounting')
    need(output.count('Traceback (most recent call last):') == 1 and
         output.rstrip().splitlines()[-1] == 'ValueError: '+ERROR and
         'During handling of the above exception' not in output and
         'The above exception was the direct cause' not in output,
         'original EA output is not its single producer-failure traceback')
    frames = re.findall(r'^\s*File "([^"]+)", line (\d+), in ([^\n]+)$', output, re.M)
    path = str(repository/'reproduction/hpc3_qprobe/followthrough719225.py')
    need(len(frames) >= 2 and frames[-1] == (path, '147', 'one_tick') and
         frames[-2][0] == path and frames[-2][2] == 'supervise',
         'original EA traceback does not end at the pinned failure branch')
    # previous_actual_state may be stale or None: one_tick raised before its
    # assignment. It is deliberately not evidence of producer completion.
    return dict(actual_process=birth, self_reported_exit_code=1,
        external_parent_wait_receipt_available=False, actual_birth_dead=True,
        original_ea_accounting_not_invented=True)


def ea_failure(binding, raw_accounting):
    ea, publication = reader.load_fixed(EA, 'followthrough719225', 'control')
    fixed = ea.fixed_binding()
    documents = {}; snapshots = {}; files = {}
    for name in EA_FILES:
        raw, proof = reader.read_file(ORIGINAL/name, capture=True, capture_limit=128*2**20)
        snapshots[name] = raw
        files[name] = {key:proof[key] for key in ('path', 'sha256', 'bytes')}
        if name.endswith('.json'):
            documents[name] = json.loads(raw)
    actual = dict(job='719225', state='FAILED', exit_code='1:0', raw=raw_accounting)
    birth = validate_ea(documents, snapshots['output.log'].decode(), publication, fixed, actual, alive=ctl.live)
    need(not (ORIGINAL/'final-consumer-binding.json').exists() and
         not (BASE/'logs/qprobe-final-goal-acceptance.json').exists(),
         'original normal goal/consumer already exists; inspect, never replace')
    stable = dict(source=publication, fixed_submission=fixed, producer=binding,
        accounting=actual, terminal=birth, files=files)
    return dict(stable, fingerprint=reader.canonical_sha(stable)), snapshots


def admission(recovery_commit):
    # Actual FAILED, held structural recovery and original all200 gates precede
    # our source scan and every new directory/process/submission.
    run, binding, raw, recovery = ctl.final_producer(BASE, recovery_commit=recovery_commit)
    need(run['run_id'] == reader.RUN and binding['producer_job'] == '719225' and
         recovery is not None and recovery['commit'] == recovery_commit,
         'precisely typed failed full200 producer required')
    original, snapshots = ea_failure(binding, raw)
    return dict(producer=binding, producer_final_cpu_recovery=recovery,
        original_ea=original, original_accounting_remains_failed=True), snapshots


def preserved(claim):
    now, _ = admission(claim['recovery_commit'])
    need(now == claim['admission'], 'original failure/held structural/EA evidence changed')
    for name, row in now['original_ea']['files'].items():
        reader.read_file(DIRECTORY/'original-ea'/name, expected_sha=row['sha256'], expected_bytes=row['bytes'])
    return now


def validate_archive_documents(claim, runner, task_binding, result, task, *,
                               step, binding, publication, allowed_controllers, alive,
                               final_submit=False):
    role = 'detached_cpu_final_audit_submission' if final_submit else 'detached_cpu_pair_archive'
    need(claim.get('schema_version') == 1 and claim.get('role') == role and
         re.fullmatch('[0-9a-f]{32}', claim.get('token', '')) is not None and
         (final_submit or (type(claim.get('step')) is int and claim['step'] == step)) and claim.get('producer') == binding and
         claim.get('source') == publication and claim.get('controller') in allowed_controllers and
         reader.birth_ok(claim.get('controller')) and finite(claim.get('created_unix')) and
         all(claim.get(key) is False for key in ('training_permitted', 'direct_gpu_device_access_permitted')) and
         claim.get('native_gpu_audit_submission_permitted') is final_submit,
         'archive claim role/owner/source/controller/scope differs')
    directory = ROOT/'final-control'/reader.RUN/('audit-controller' if final_submit else 'archive-'+str(step))
    repository = ROOT/'source'/publication['commit']
    need(claim.get('command') == ctl.command(repository, 'audit-worker' if final_submit else 'archive-worker',
         directory/'submit-claim.json', claim['token']), 'archive command differs')
    sha = ctl.digest(directory/'submit-claim.json')
    for value in (runner, task_binding, result, task):
        if value is not None:
            need(value.get('token') == claim['token'] and value.get('claim_sha256') == sha,
                 'archive child token/claim SHA differs')
    if runner is None:
        need(result is None and task_binding is None and task is None,
             'archive child evidence exists without original runner binding')
        raise ValueError('RECOVERY_FOLLOWTHROUGH719225: archive runner missing; unknown, never replay')
    parent = runner.get('actual_process')
    need(reader.birth_ok(parent) and parent['pid'] == parent['sid'] and
         runner.get('controller') == claim['controller'], 'archive runner birth/controller differs')
    parent_live = alive(parent)
    child = task_binding.get('actual_process') if task_binding is not None else None
    if child is not None:
        need(reader.birth_ok(child) and child['sid'] == parent['sid'] and child['pid'] != parent['pid'] and
             task_binding.get('supervisor') == parent, 'archive held task/runner birth differs')
    if result is None:
        need(parent_live, 'archive runner dead without held outcome; unknown, never replay')
        return dict(ready=False, runner_alive=True, task_alive=bool(child and alive(child)))
    need(child is not None and result.get('actual_task_birth') == child and
         result.get('actual_supervisor') == parent and result.get('role') == role and
         type(result.get('actual_task_exit_code')) is int and result['actual_task_exit_code'] == 0 and
         result.get('status') == ('ACTUAL_CPU_FINAL_SUBMIT_CHILD_COMPLETED0' if final_submit else
             'ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0') and not alive(child) and
         task is not None and result.get('task_result_sha256') == ctl.digest(directory/'task-result.json') and
         task.get('status') == ('CPU_FINAL_SUBMIT_CONTROLLER_RETURNED' if final_submit else
             'CPU_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS') and
         task.get('actual_process') == child and
         all(task.get(key) is False for key in ('training_performed', 'direct_gpu_device_access_performed',
             'actor_model_loaded')) and task.get('native_gpu_audit_submission_attempted') is final_submit and
         finite(task.get('completed_unix')) and finite(result.get('finished_unix')) and
         result['finished_unix'] >= task['completed_unix'] >= claim['created_unix'],
         'archive is not the actual held original child0 with unchanged task result')
    # A valid wait receipt can precede the runner's own exit by milliseconds.
    # Wait for that same birth, never accept it early or launch a replacement.
    return dict(ready=not parent_live, runner_alive=parent_live, task_alive=False,
        claim_sha256=sha, worker_result_sha256=ctl.digest(directory/'worker-result.json'),
        task_result_sha256=result['task_result_sha256'], actual_runner=parent, actual_child=child)


def archive_state(step, claim, own, publication_cache):
    directory = ROOT/'final-control'/reader.RUN/('archive-'+str(step))
    intent = ctl.read(directory/'submit-claim.json')
    commit = intent['source']['commit']
    need(commit in (EA, claim['source']['commit']), 'archive belongs to another helper; never adopt')
    if commit not in publication_cache:
        publication_cache[commit] = reader.publication(commit)[1]
    old_birth = claim['admission']['original_ea']['terminal']['actual_process']
    expected_controller = old_birth if commit == EA else own
    run, binding, _ = ctl.producer(BASE)
    _, pair = ctl.pair(BASE, run, step)
    need(intent.get('pair') == pair, 'archive original pair bytes changed')
    def optional(name):
        path = directory/name
        return ctl.read(path) if path.exists() or path.is_symlink() else None
    return validate_archive_documents(intent, optional('runner-binding.json'), optional('task-binding.json'),
        optional('worker-result.json'), optional('task-result.json'), step=step, binding=binding,
        publication=publication_cache[commit], allowed_controllers=[expected_controller], alive=ctl.live)


def submit_state(claim, own):
    directory = ROOT/'final-control'/reader.RUN/'audit-controller'
    intent = ctl.read(directory/'submit-claim.json')
    need(intent.get('recovery_commit') == claim['recovery_commit'] and
         intent.get('producer_final_cpu_recovery') == claim['admission']['producer_final_cpu_recovery'],
         'final CPU submit recovery intent differs')
    def optional(name):
        path = directory/name
        return ctl.read(path) if path.exists() or path.is_symlink() else None
    return validate_archive_documents(intent, optional('runner-binding.json'), optional('task-binding.json'),
        optional('worker-result.json'), optional('task-result.json'), step=None,
        binding=claim['admission']['producer'], publication=claim['source'],
        allowed_controllers=[own], alive=ctl.live, final_submit=True)


def publish_state(value):
    path = DIRECTORY/'state.json'; temporary = DIRECTORY/('state.'+uuid.uuid4().hex+'.partial')
    with temporary.open('x') as stream:
        json.dump(dict(value, checked_unix=time.time()), stream, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(DIRECTORY, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def one_tick(claim, own, verified, publication_cache):
    preserved(claim)
    components = ROOT/'final-control'/reader.RUN
    archives = {}
    for step in ctl.STEPS:
        directory = components/('archive-'+str(step))
        payload = ROOT/'archives'/reader.RUN/('global_step_'+str(step))
        if not directory.exists() and not directory.is_symlink():
            need(not payload.exists() and not payload.is_symlink(), 'orphan archive payload; never replay')
            normal.pair_ready(step)
            ctl.launch_archive(BASE, claim['source']['commit'], step)
        state = archive_state(step, claim, own, publication_cache)
        if not state['ready']:
            archives[str(step)] = state
            return dict(phase='ACTUAL_RECOVERY_EXISTING_ARCHIVE_WAIT', archives=archives,
                goal_complete=False, full_objective_accepted=False)
        observed = ctl.inspect_archive(BASE, step, full=False)
        need(observed['status'] == 'ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0' and
             observed['actual_task_exit_code'] == 0, 'actual held archive outcome differs')
        transport_path = payload/'complete.json'
        need(observed['transport_path'] == str(transport_path), 'archive transport path differs')
        reader.read_file(transport_path, expected_sha=observed['transport_sha256'])
        key = (state['worker_result_sha256'], observed['transport_sha256'])
        if verified.get(step) != key:
            full = ctl.inspect_archive(BASE, step, full=True)
            need(full.get('fresh_full_sha') is not None, 'fresh source/archive/target SHA absent')
            verified[step] = key
        archives[str(step)] = dict(state, transport_path=observed['transport_path'],
            transport_sha256=observed['transport_sha256'], fresh_full_sha_verified=True)
    recovery_commit = claim['recovery_commit']
    if not (components/'audit-controller').exists() and not (components/'audit-controller').is_symlink() and \
            not (components/'final-audit').exists() and not (components/'final-audit').is_symlink():
        ctl.start_audit(BASE, claim['source']['commit'], recovery_commit=recovery_commit)
    audit = ctl.inspect_audit(BASE, recovery_commit=recovery_commit)
    cpu_births = submit_state(claim, own)
    cpu = audit.get('cpu_controller') or {}
    need(cpu.get('actual_task_exit_code') in (None, 0), 'actual final CPU preflight failed; never replay')
    if not audit.get('job'):
        need(audit['status'] == 'FINAL_SUBMIT_CPU_PREFLIGHT_RUNNING_OR_FAILED_NO_SLURM_INTENT' and
             (cpu.get('runner_alive') is True or cpu.get('task_alive') is True),
             'final native submission unknown or failed; inspect original intent, never replay')
        return dict(phase='ACTUAL_RECOVERY_FINAL_SUBMIT_CPU_WAIT', archives=archives, audit=audit,
            goal_complete=False, full_objective_accepted=False)
    need(audit['status'] in ('ACTUAL_SLURM_SUBMITTED', 'ACTUAL_SLURM_RECONCILED'),
         'known final audit has no actual submission/reconciliation')
    actual = normal.accounting(audit['job'], allow_missing=True)
    if actual is None:
        return dict(normal.final_registration_wait(audit), goal_complete=False, full_objective_accepted=False)
    normal.REGISTRATION_WAIT.pop(audit['job'], None)
    need(actual['state'] in ('PENDING', 'RUNNING', 'COMPLETING', 'COMPLETED') and
         (actual['state'] != 'COMPLETED' or actual['exit_code'] == '0:0'),
         'actual final native restore failed; diagnose, never replay')
    if actual['state'] != 'COMPLETED' or not cpu_births['ready']:
        return dict(phase='ACTUAL_RECOVERY_FINAL_NATIVE_RESTORE_WAIT', archives=archives,
            native_accounting=actual, cpu_controller=cpu, goal_complete=False, full_objective_accepted=False)
    need(cpu.get('status') == 'ACTUAL_CPU_FINAL_SUBMIT_CHILD_COMPLETED0' and
         type(cpu.get('actual_task_exit_code')) is int and cpu['actual_task_exit_code'] == 0,
         'native audit lacks actual held CPU submit0')
    # A same-worker existing receipt can only be consumed, never overwritten.
    path = DIRECTORY/'goal-acceptance.json'; consumer = DIRECTORY/'final-consumer-binding.json'
    if not path.exists():
        from reproduction.hpc3_qprobe.verify_goal import verify
        verify(BASE, audit['audit_base'], recovery_commit=recovery_commit)
        ctl.publish(consumer, dict(actual_process=own, helper_commit=claim['source']['commit'],
            recovery_commit=recovery_commit, producer_final_cpu_recovery=claim['admission']['producer_final_cpu_recovery'],
            audit_base=audit['audit_base'], goal_sha256=ctl.digest(path),
            actual_native_audit_accounting=actual, cpu_submit=cpu, full_cpu_consumer_returned=True))
    bound = ctl.read(consumer); goal = ctl.read(path)
    need(bound['actual_process'] == own and bound['helper_commit'] == claim['source']['commit'] and
         bound['recovery_commit'] == recovery_commit and bound['audit_base'] == audit['audit_base'] and
         bound['goal_sha256'] == ctl.digest(path) and bound['full_cpu_consumer_returned'] is True and
         bound['producer_final_cpu_recovery'] == claim['admission']['producer_final_cpu_recovery'] and
         goal['status'] == 'QPROBE_FULL200_TEACHER_HEAD_FIXED_VALIDATION_ARCHIVE_NATIVE4_GOAL_PASS' and
         goal['run_id'] == reader.RUN and goal['completed_outer'] == 200 and goal['update201_performed'] is False and
         goal.get('producer_final_cpu_recovery') == claim['admission']['producer_final_cpu_recovery'] and
         goal.get('original_producer_accounting_remains_failed') is True,
         'separate recovery goal is not this actual held worker and typed original failure')
    preserved(claim)
    return dict(phase=STATUS, native4_goal_reader_pass=True, archives=archives,
        actual_final_native_restore_accepted=True, native_accounting=actual,
        audit_base=audit['audit_base'], final_goal_path=str(path), final_goal_sha256=ctl.digest(path),
        consumer_binding_sha256=ctl.digest(consumer), original_accounting_remains_failed=True,
        independent_teacher_raw_final_consumer_still_required=True,
        full_objective_accepted=False, goal_complete=False)


def accept_output(value, claim, child):
    need(value.get('status') == STATUS and value.get('source') == claim['source'] and
         value.get('actual_process') == child and value.get('producer_final_cpu_recovery') ==
         claim['admission']['producer_final_cpu_recovery'] and value.get('original_ea_fingerprint') ==
         claim['admission']['original_ea']['fingerprint'] and value.get('goal_complete') is False and
         value.get('full_objective_accepted') is False and
         value.get('independent_teacher_raw_final_consumer_still_required') is True and
         value.get('result', {}).get('phase') == STATUS and value['result'].get('native4_goal_reader_pass') is True and
         value['result'].get('full_objective_accepted') is False and value['result'].get('goal_complete') is False and
         all(value.get(key) is False for key in ('training_submission_performed',
             'actor_model_loaded', 'direct_gpu_device_access_performed', 'update201_performed')),
         'held recovery worker output claims another source/scope or premature goal acceptance')


def native_goal_binding(goal, readback, accepted, readback_file, accepted_file, native):
    need(readback.get('slurm_job_id') == native['job'] and readback.get('run_id') == reader.RUN and
         readback.get('producer_commit') == reader.PRODUCER and readback.get('audit_commit') == EA and
         readback.get('world_size') == 4 and readback.get('actual_completed_outer') == 200 and
         readback.get('update201_performed') is False and readback.get('fit_called') is False and
         all(type(readback.get(key)) is int and readback[key] == 0 for key in
             ('actual_actor_optimizer_write_delta', 'actual_actor_scheduler_advance_delta',
              'actual_head_update_delta', 'actual_head_optimizer_write_delta', 'generated_trajectories')) and
         accepted.get('accepted') is True and accepted.get('actual_cuda_restore_accepted') is True and
         accepted.get('owned_cleanup') is True and accepted.get('readback_sha256') == readback_file['sha256'] and
         accepted.get('audit_id') == readback.get('audit_id') and
         goal.get('actual_native_restore_accepted_sha256') == accepted_file['sha256'],
         'actual terminal native audit/readback/goal bytes differ')


def task(commit, recovery_commit):
    caps(); repository, publication = ctl.source(commit)
    need(os.environ.get('QPROBE_SOURCE') == str(repository), 'worker source differs')
    claim = ctl.read(DIRECTORY/'claim.json'); own = ctl.birth(ctl.process(os.getpid()))
    deadline = time.monotonic()+15
    while not (DIRECTORY/'child.json').exists():
        need(time.monotonic() < deadline, 'actual held worker binding absent; never replay')
        time.sleep(.05)
    need(ctl.read(DIRECTORY/'child.json') == own and claim['source'] == publication and
         claim['recovery_commit'] == recovery_commit, 'actual held worker birth/source/recovery differs')
    verified = {}; publications = {}
    while True:
        result = one_tick(claim, own, verified, publications); publish_state(result)
        if result.get('native4_goal_reader_pass') is True:
            break
        time.sleep(60)
    return dict(status=STATUS, source=publication, actual_process=own, result=result,
        producer_final_cpu_recovery=claim['admission']['producer_final_cpu_recovery'],
        original_ea_fingerprint=claim['admission']['original_ea']['fingerprint'],
        training_submission_performed=False, actor_model_loaded=False,
        direct_gpu_device_access_performed=False, update201_performed=False,
        independent_teacher_raw_final_consumer_still_required=True,
        full_objective_accepted=False, goal_complete=False)


def consume(commit, recovery_commit):
    caps(); _, publication = ctl.source(commit)
    claim = ctl.read(DIRECTORY/'claim.json'); parent = ctl.read(DIRECTORY/'process.json')
    child = ctl.read(DIRECTORY/'child.json'); result = ctl.read(DIRECTORY/'result.json')
    need(claim['source'] == result['source'] == publication and claim['recovery_commit'] == recovery_commit and
         result['claim_sha256'] == ctl.digest(DIRECTORY/'claim.json') and result['process'] == parent and
         result['child'] == child and reader.birth_ok(parent) and reader.birth_ok(child) and
         parent['pid'] == parent['sid'] == child['sid'] and child['pid'] != parent['pid'] and
         type(result.get('actual_child_exit_code')) is int and result['actual_child_exit_code'] == 0 and
         type(result.get('supervisor_exit_code')) is int and result['supervisor_exit_code'] == 0 and
         result.get('error') is None and result['completed_unix'] >= result['started_unix'] >= claim['created_unix'] and
         not ctl.live(parent) and not ctl.live(child), 'actual held recovery wait/birth/terminal/source differs')
    preserved(claim)
    for name in ('stdout', 'stderr'):
        row = result[name]; path = DIRECTORY/('consumer.'+name)
        need(row['path'] == str(path), 'held recovery output path differs')
        reader.read_file(path, expected_sha=row['sha256'], expected_bytes=row['bytes'])
    need(result['stderr']['bytes'] == 0, 'held recovery worker stderr is nonempty')
    value, output = reader.document(DIRECTORY/'consumer.stdout'); accept_output(value, claim, child)
    state = value['result']; path = DIRECTORY/'goal-acceptance.json'
    need(state['final_goal_path'] == str(path), 'separate goal path differs')
    goal, goal_file = reader.document(path)
    binding, binding_file = reader.document(DIRECTORY/'final-consumer-binding.json')
    need(goal_file['sha256'] == state['final_goal_sha256'] and
         binding_file['sha256'] == state['consumer_binding_sha256'] and binding['actual_process'] == child and
         binding['goal_sha256'] == goal_file['sha256'] and binding['helper_commit'] == commit and
         binding['recovery_commit'] == recovery_commit and binding['audit_base'] == state['audit_base'] and
         binding['producer_final_cpu_recovery'] == goal.get('producer_final_cpu_recovery') ==
         claim['admission']['producer_final_cpu_recovery'] and binding['full_cpu_consumer_returned'] is True and
         goal.get('original_producer_accounting_remains_failed') is True and
         goal.get('run_id') == reader.RUN and goal.get('completed_outer') == 200 and
         goal.get('update201_performed') is False, 'held separate goal/consumer bytes or scope differ')
    native = reader.completed(state['native_accounting']['job'])
    audit_base = Path(state['audit_base'])
    need(audit_base.is_absolute() and audit_base.resolve() == audit_base and audit_base.parent == ROOT.parent,
         'native audit base differs')
    readback, readback_file = reader.document(audit_base/'logs/final-restore-readback.json')
    accepted, accepted_file = reader.document(audit_base/'logs/final-restore-audit-accepted.json')
    native_goal_binding(goal, readback, accepted, readback_file, accepted_file, native)
    return dict(status=HELD_STATUS, source=publication, result_path=str(DIRECTORY/'result.json'),
        result_sha256=ctl.digest(DIRECTORY/'result.json'), output=output, goal_acceptance=goal_file,
        consumer_binding=binding_file, audit_base=state['audit_base'], native_accounting=native,
        producer_final_cpu_recovery=claim['admission']['producer_final_cpu_recovery'],
        original_ea=claim['admission']['original_ea'], actual_child_exit_code=0,
        original_accounting_remains_failed=True, independent_teacher_raw_final_consumer_still_required=True,
        full_objective_accepted=False, goal_complete=False)


def inspect(commit, recovery_commit):
    caps()
    if not DIRECTORY.exists() and not DIRECTORY.is_symlink():
        return dict(status='NOT_STARTED', directory=str(DIRECTORY), goal_complete=False)
    need(DIRECTORY.resolve() == DIRECTORY and DIRECTORY.stat().st_uid == ctl.OWNER,
         'unsafe recovery once directory')
    claim = ctl.read(DIRECTORY/'claim.json')
    need(claim['source']['commit'] == commit and claim['recovery_commit'] == recovery_commit,
         'existing recovery scope belongs to another intent; never replay')
    def optional(name):
        return ctl.read(DIRECTORY/name) if (DIRECTORY/name).exists() else None
    parent = optional('process.json'); child = optional('child.json'); result = optional('result.json')
    parent_live = bool(parent and ctl.live(parent)); child_live = bool(child and ctl.live(child))
    if result is not None and result.get('supervisor_exit_code') == 0 and not parent_live and not child_live:
        return consume(commit, recovery_commit)
    return dict(status=('ACTUAL_RECOVERY_FOLLOWTHROUGH_LIVE' if parent_live else
        'ACTUAL_RECOVERY_FOLLOWTHROUGH_FAILED_INSPECT' if result else 'UNKNOWN_RECOVERY_NO_REPLAY'),
        actual_supervisor_live=parent_live, actual_child_live=child_live,
        process=parent, child=child, result=result, state=optional('state.json'),
        full_objective_accepted=False, goal_complete=False, checked_unix=time.time())


def start(commit, recovery_commit):
    caps()
    if DIRECTORY.exists() or DIRECTORY.is_symlink():
        return inspect(commit, recovery_commit)
    proof, snapshots = admission(recovery_commit)
    repository, publication = ctl.source(commit); os.environ['QPROBE_SOURCE'] = str(repository)
    try:
        DIRECTORY.mkdir(mode=0o770)
    except FileExistsError:
        return inspect(commit, recovery_commit)
    command = [str(BASE/'env/bin/python'), '-m', MODULE, 'supervise', '--commit', commit,
               '--recovery-commit', recovery_commit]
    ctl.publish(DIRECTORY/'claim.json', dict(source=publication, recovery_commit=recovery_commit,
        admission=proof, command=command, cwd=str(repository), created_unix=time.time(),
        training_submission_permitted=False, actor_model_reload_permitted=False,
        direct_gpu_access_permitted=False, native_final_restore_only_submission_permitted=True))
    for name, raw in snapshots.items():
        path = DIRECTORY/'original-ea'/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    preserved(ctl.read(DIRECTORY/'claim.json'))
    with (DIRECTORY/'supervisor.stdout').open('xb') as out, (DIRECTORY/'supervisor.stderr').open('xb') as err:
        process = subprocess.Popen(command, cwd=str(repository), stdout=out, stderr=err, start_new_session=True)
    birth = ctl.birth(ctl.process(process.pid))
    need(birth['pid'] == birth['sid'] and birth['uid'] == ctl.OWNER, 'actual detached recovery supervisor differs')
    ctl.publish(DIRECTORY/'process.json', birth)
    return dict(status='ACTUAL_RECOVERY_FOLLOWTHROUGH_STARTED_NOT_ACCEPTED', process=birth,
        directory=str(DIRECTORY), full_objective_accepted=False, goal_complete=False)


def supervise(commit, recovery_commit):
    caps(); repository, publication = ctl.source(commit); os.environ['QPROBE_SOURCE'] = str(repository)
    own = ctl.birth(ctl.process(os.getpid())); started = time.time()
    process = None; child = None; child_code = None; error = None; code = 1
    try:
        deadline = time.monotonic()+15
        while not (DIRECTORY/'process.json').exists():
            need(time.monotonic() < deadline, 'launcher birth absent; never replay')
            time.sleep(.05)
        claim = ctl.read(DIRECTORY/'claim.json')
        need(ctl.read(DIRECTORY/'process.json') == own and claim['source'] == publication and
             claim['recovery_commit'] == recovery_commit and claim['cwd'] == str(repository),
             'actual recovery supervisor source/birth/intent differs')
        preserved(claim)
        command = [str(BASE/'env/bin/python'), '-m', MODULE, 'task', '--commit', commit,
                   '--recovery-commit', recovery_commit]
        with (DIRECTORY/'consumer.stdout').open('xb') as out, (DIRECTORY/'consumer.stderr').open('xb') as err:
            process = subprocess.Popen(command, cwd=str(repository), stdout=out, stderr=err)
            child = ctl.birth(ctl.process(process.pid))
            need(child['uid'] == ctl.OWNER and child['sid'] == own['sid'] and child['pid'] != own['pid'],
                 'actual held recovery worker birth differs')
            ctl.publish(DIRECTORY/'child.json', child); child_code = process.wait()
            for stream in (out, err):
                stream.flush(); os.fsync(stream.fileno())
        need(child_code == 0 and not ctl.live(child), 'actual held recovery worker did not return0')
        need((DIRECTORY/'consumer.stderr').stat().st_size == 0, 'actual held recovery worker stderr is nonempty')
        value, _ = reader.document(DIRECTORY/'consumer.stdout'); accept_output(value, claim, child)
        preserved(claim); code = 0
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
    ctl.publish(DIRECTORY/'result.json', dict(source=publication, recovery_commit=recovery_commit,
        claim_sha256=ctl.digest(DIRECTORY/'claim.json'), process=own, child=child,
        actual_child_exit_code=child_code, supervisor_exit_code=code, error=error,
        started_unix=started, completed_unix=time.time(), full_objective_accepted=False,
        goal_complete=False, **outputs))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('start', 'supervise', 'task', 'inspect'))
    parser.add_argument('--commit', required=True); parser.add_argument('--recovery-commit', required=True)
    args = parser.parse_args()
    if args.mode == 'supervise':
        sys.exit(supervise(args.commit, args.recovery_commit))
    method = {'start':start, 'task':task, 'inspect':inspect}[args.mode]
    print(json.dumps(method(args.commit, args.recovery_commit), allow_nan=False), flush=True)
