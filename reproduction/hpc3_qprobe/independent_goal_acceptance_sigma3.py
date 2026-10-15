"""Pure CPU final conjunction for the isolated sigma3 experiment.

No writes, helper launches, actor loads, generation or job submissions.
The only subprocess queries read actual Slurm accounting. All200 is checked
before heavy imports or source/large evidence traversal. Historical teacher
identity is preserved; it is never relabeled as the new actor.
"""
import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

from reproduction.hpc3_qprobe import independent_goal_acceptance719225 as shared
from reproduction.hpc3_qprobe import final_control as ctl

ROOT = shared.ROOT
BASE = ROOT.parent/'qprobe-teacher-warmup200-sigma3-20261015'
OWNER = shared.OWNER
RUN = '70827e85fe584f0c84453aae907918a4'
JOB = '721485'
PRODUCER = 'aca0e779e94ddc3fbd50e49c3d59e2e889fbcf12'
RAW = '0cca743462da769667262417f8064084f93fc912'
FOLLOWTHROUGH = '9c34720ecd564dfe902127d0167474d27b115cfe'
RAW_DIRECTORY = ROOT/'observations/raw-all200-sigma3-721485-20261015'
FOLLOWTHROUGH_DIRECTORY = ROOT/'followthrough'/JOB
TEACHER_ORIGIN = ROOT.parent/'qprobe-teacher-warmup200-4gpu-jxu957-recovery718509-20261014/teacher-warmup-cache'
TEACHER_RUNS = ('0742d8e346284d6da9656a826ce14395',)*2 + ('c3856315d3f345d7920ebf5725a083b4',)*8
LOSS = dict(loss_mode='sigma3', mean_loss_weight=0., std_loss_weight=1., auxiliary_count_loss_weight=.1)
GOAL_STATUS = shared.GOAL_STATUS
STATUS = 'ACTUAL_INDEPENDENT_SIGMA3_TEACHER_RAW_ALL200_ARCHIVES_NATIVE4_GOAL_PASS'
BASELINE = shared.BASELINE
BASELINE_FIELDS = shared.BASELINE_FIELDS
BASELINE_PAYLOAD_SHA = shared.BASELINE_PAYLOAD_SHA
ORIGINAL_TEACHER_PROOF_SHA = shared.ORIGINAL_TEACHER_PROOF_SHA
OFFICIAL_MODEL_REVISION = shared.OFFICIAL_MODEL_REVISION
TOKENIZERS = shared.TOKENIZERS
CAPS = shared.CAPS
# Reuse only source-independent pure file/accounting primitives. No old route
# or old run-bound acceptance function is invoked or patched.
read_file = shared.read_file
document = shared.document
rows = shared.rows
canonical_sha = shared.canonical_sha
signature = shared.signature
bound_record = shared.bound_record
SharedReads = shared.SharedReads
completed = shared.completed
publication = shared.publication
load_fixed = shared.load_fixed


def need(value, message):
    if not value:
        raise ValueError('INDEPENDENT_SIGMA3: '+message)


def birth_ok(value):
    return isinstance(value,dict) and all(type(value.get(key)) is int and value[key]>0
        for key in ('pid','start_ticks','sid','uid')) and value['uid']==OWNER


def early_history():
    current, ledger = rows(BASE/'logs/actual-updates.jsonl')
    need([row.get('step') for row in current] == list(range(1,201)) and
         all(type(row.get('step')) is int and row.get('run_id') == RUN for row in current),
         'actual fresh1..200 absent or logical201/copied prefix present')
    run, manifest = document(BASE/'logs/run-manifest.json')
    settings = run.get('question_probe') or {}
    need(run.get('run_id') == RUN and run.get('source_commit') == PRODUCER and
         run.get('training_world_size') == 4 and run.get('actor_micro_batch_per_rank') == 4 and
         not run.get('full_state_restore') and run.get('model_revision') == OFFICIAL_MODEL_REVISION and
         settings.get('initial_head_checkpoint') is None and settings.get('snapshot_path') is None and
         settings.get('head_hidden_size') == 64 and all(settings.get(k) == v for k,v in LOSS.items()) and
         all(settings.get(k) == v for k,v in dict(candidate_questions=5120,selected_questions=512,
             responses_per_question=5,exploration_questions=64,seed=20261013).items()),
         'actual fresh sigma3 producer/loss/capacity differs')
    return run, dict(current_updates=ledger,current_manifest=manifest)


def producer_acceptance():
    run, binding, raw = ctl.producer(BASE, complete=True)
    need(run['run_id'] == RUN and binding['producer_job'] == JOB and
         binding['producer_commit'] == PRODUCER and binding.get('original_failed_job') == '721449',
         'actual sigma3 producer/failure chain differs')
    return completed(JOB), binding


def followthrough_evidence(follow, publication_proof, followthrough_commit):
    need(follow.BASE == BASE and follow.RUN == RUN and follow.JOB == JOB and follow.PRODUCER == PRODUCER and
         follow.DIRECTORY == FOLLOWTHROUGH_DIRECTORY, 'followthrough scope differs')
    observed = follow.inspect()
    need(observed.get('actual_live_birth') is False, 'followthrough actual supervisor has not finished')
    result = observed.get('result')
    process = observed['process']['actual_process']
    need(isinstance(result, dict) and type(result.get('exit_code')) is int and result['exit_code'] == 0 and result.get('error') is None and
         birth_ok(process) and process['sid'] == process['pid'] and result.get('actual_process') == process,
         'followthrough state alone or nonreturned CPU supervisor cannot pass')
    actual_result, result_file = document(follow.DIRECTORY/'result.json')
    actual_process, process_file = document(follow.DIRECTORY/'process.json')
    need(actual_result == result and actual_process == observed['process'], 'followthrough result/process changed during consumption')
    need(result.get('training_submission_performed') is False and result.get('actor_model_loaded') is False and
         result.get('direct_gpu_device_access_performed') is False and
         result.get('native_final_restore_only_submission_permitted') is True,
         'followthrough returned CPU scope differs')
    claim, claim_file = document(follow.DIRECTORY/'claim.json')
    need(claim['source'] == publication_proof and claim['source']['commit'] == followthrough_commit and
         claim['binding'] == follow.fixed_binding() and observed['process']['commit'] == followthrough_commit and
         claim.get('training_submission_permitted') is False and claim.get('actor_model_reload_permitted') is False and
         claim.get('direct_gpu_access_permitted') is False and claim.get('native_final_restore_only_submission_permitted') is True,
         'followthrough original once source/birth/intent differs')
    returned = result.get('result') or {}
    goal_path = BASE/'logs/qprobe-final-goal-acceptance.json'
    goal, goal_file = document(goal_path)
    binding, binding_file = document(follow.DIRECTORY/'final-consumer-binding.json')
    need(binding.get('actual_process') == process and binding.get('helper_commit') == followthrough_commit and
         binding.get('full_cpu_consumer_returned') is True and
         binding.get('goal_sha256') == goal_file['sha256'] and returned.get('full_objective_accepted') is True and
         returned.get('actual_final_native_restore_accepted') is True and
         returned.get('phase') == 'ACTUAL_FULL_CPU_GOAL_CONSUMER_PASS' and
         returned.get('final_goal_path') == str(goal_path) and returned.get('final_goal_sha256') == goal_file['sha256'] and
         goal.get('status') == GOAL_STATUS and goal.get('run_id') == RUN and
         goal.get('completed_outer') == 200 and goal.get('update201_performed') is False,
         'followthrough actual full CPU consumer/goal binding differs')
    audit_base, native, native_file, accepted_file, actual = native_evidence(goal, binding)
    return audit_base, native, dict(source=publication_proof, claim=claim_file, binding=binding_file, goal=goal_file,
        accepted=accepted_file, native=native_file, accounting=actual,
        result=result_file, process_file=process_file, process=process)

def native_evidence(goal, binding):
    """The same actual native4/no-new-work gates for either CPU final route."""
    audit_base = Path(binding['audit_base'])
    need(audit_base.is_absolute() and audit_base.resolve() == audit_base and
         audit_base.parent == ROOT.parent and audit_base != BASE, 'actual final audit base unsafe')
    native, native_file = document(audit_base/'logs/final-restore-readback.json')
    accepted, accepted_file = document(audit_base/'logs/final-restore-audit-accepted.json')
    need(accepted.get('accepted') is True and accepted.get('actual_cuda_restore_accepted') is True and
         accepted.get('owned_cleanup') is True and accepted.get('readback_sha256') == native_file['sha256'] and
         goal.get('actual_native_restore_accepted_sha256') == accepted_file['sha256'] and
         native.get('run_id') == RUN and native.get('producer_commit') == PRODUCER and
         native.get('world_size') == 4 and native.get('actual_completed_outer') == 200 and
         native.get('actual_cuda_restore_accepted') is True and native.get('update201_performed') is False and
         native.get('fit_called') is False and accepted.get('audit_id') == native.get('audit_id'),
         'authentic final native4 restore receipt missing')
    for key in ('actual_actor_optimizer_write_delta', 'actual_actor_scheduler_advance_delta',
                'actual_head_update_delta', 'actual_head_optimizer_write_delta', 'generated_trajectories'):
        need(type(native.get(key)) is int and native[key] == 0, 'native restore performed new work')
    actual = completed(native['slurm_job_id'])
    prior = binding.get('actual_native_audit_accounting') or {}
    need(prior.get('job') == actual['job'] and prior.get('state') == 'COMPLETED' and prior.get('exit_code') == '0:0' and
         prior.get('raw', '').strip() == actual['raw'].strip(), 'followthrough native accounting differs from actual job')
    return audit_base, native, native_file, accepted_file, actual

def fresh_attempt(task, output, shared_reads=None):
    """Hashes original inputs of a real, already returned same-fd CPU decode."""
    p = task['producer']; base = Path(p['base']); aid = task['attempt_id']
    expected_part = (BASE, PRODUCER, JOB)
    need((base, p['producer_commit'], p['producer_job']) == expected_part and p['run_id'] == RUN,
         'raw historical source/job/base differs')
    folder = base/'artifacts/question-probe/attempts'/aid
    terminal = base/'artifacts/question-probe'/('completed' if task['complete_outer'] else 'aborted')/(aid+'.json')
    private = base/'artifacts/behavior-traces'
    private_stat = private.lstat()
    need(private.resolve() == private and stat.S_ISDIR(private_stat.st_mode) and
         private_stat.st_uid == OWNER and stat.S_IMODE(private_stat.st_mode) == 0o700,
         'authentic private trace directory ownership/mode differs')
    canonical = output['canonical_proof']
    standalone = shared_reads is None
    shared_reads = shared_reads if shared_reads is not None else SharedReads()
    expected = (
        (output['raw_file'], folder/'real-rollout.pt', False),
        (output['private_trace_file'], private/('attempt-'+aid+'.json'), True),
        (output['gate_file'], base/'logs'/('behavior-probability-attempt-'+aid+'.json'), True),
        (output['source_manifest'], base/'logs/run-manifest.json', False),
        (output['allocation_file'], base/'logs/native-allocation-admission.json', False),
        (canonical['selection_file'], folder/'selection.json', False),
        (canonical['rollout_file'], folder/'rollout-audit.json', False),
        (canonical['head_file'], terminal, False))
    proofs = [bound_record(row, path, private=private) for row, path, private in expected]
    run, _ = document(base/'logs/run-manifest.json')
    need(run['run_id'] == RUN and run['source_commit'] == p['producer_commit'] and
         output['source_manifest']['sha256'] == p['source_manifest_sha256'] and
         output['allocation_file']['sha256'] == p['native_allocation_sha256'], 'decoded original producer binding differs')
    train = base/'data/train.parquet'
    need(canonical['train_sha256'] == run['input_sha256']['train.parquet'], 'decoded canonical train differs')
    proofs.append(shared_reads.read(train, canonical['train_sha256'], alias=True, shared=True))
    for name in TOKENIZERS:
        sha = canonical['tokenizer_sha256'][name]
        need(sha == run['initial_model_sha256'][name], 'decoded tokenizer/source model differs')
        proofs.append(shared_reads.read(Path(run['initial_model'])/name, sha, shared=True))
    # These are the actual strict-decode and pure acquisition/trace sources,
    # not a relabeling of any historical actor as the fresh sigma3 producer.
    source = ROOT/'source'/p['producer_commit']
    for name, sha in run['source_sha256'].items():
        proofs.append(shared_reads.read(source/name, sha))
    # actual-attempts.jsonl legitimately appended after the child measurement:
    # compare the original row/document SHA, never its obsolete whole-file SHA.
    ledger, _ = rows(base/'logs/actual-attempts.jsonl')
    matching = [r for r in ledger if r.get('attempt_id') == aid]
    need(len(matching) == 1 and canonical_sha(matching[0]) == task['record_sha256'] ==
         output['actual_attempt_transaction']['attempt_record_sha256'], 'original normal-return attempt changed')
    for proof in proofs:
        shared_reads.remember(proof)
    if standalone:
        shared_reads.finish()
    return dict(attempt_id=aid, step=task['step'], complete_outer=task['complete_outer'],
                fresh_original_files=len(proofs), original_file_proofs_sha256=canonical_sha(proofs),
                raw_sha256=output['raw_file']['sha256'], private_trace_sha256=output['private_trace_file']['sha256'])

def raw_evidence(raw):
    need(raw.BASE == BASE and raw.DIRECTORY == RAW_DIRECTORY, 'raw scope differs')
    observed = raw.inspect(RAW)
    need(observed.get('status') == raw.STATUS and observed.get('actual_live_birth') is False and
         observed.get('actual_child_live_birth') is False, 'actual all200 raw CPU audit not terminal/pass')
    final, final_file = document(raw.DIRECTORY/'result.json')
    claim, claim_file = document(raw.DIRECTORY/'claim.json')
    process, _ = document(raw.DIRECTORY/'process.json')
    tasks = raw.actual_tasks(claim['parts'])
    need(final['status'] == raw.STATUS and final['process'] == process and
         final['claim_sha256'] == claim_file['sha256'] and final['source'] == claim['source'] and
         final['source']['commit'] == RAW and birth_ok(process) and process['sid'] == process['pid'] and
         [t['step'] for t in tasks if t['complete_outer']] == list(range(1, 201)) and
         [r['task'] for r in final['records']] == tasks and final['normal_return_attempts'] == len(tasks) and
         final['successful_formal_outers'] == list(range(1, 201)), 'all normal-return raw attempt set differs')
    consumed = []; originals = []; shared_reads = SharedReads()
    for task in tasks:
        directory = raw.DIRECTORY/'attempts'/task['attempt_id']
        record = raw.consume_task(task, directory, process)
        consumed.append(record)
        output, output_file = document(directory/'audit.stdout')
        need(output_file['sha256'] == record['output_sha256'], 'actual consumed stdout changed before original readback')
        shared_reads.remember(output_file)
        shared_reads.remember(read_file(record['result_path'], expected_sha=record['result_sha256'])[1])
        originals.append(fresh_attempt(task, output, shared_reads))
    need(consumed == final['records'] and canonical_sha(consumed) == final['record_list_sha256'],
         'actual raw result/output/held wait receipts changed')
    shared_reads.finish()
    return dict(result=final_file, claim=claim_file, source=final['source'], normal_return_attempts=len(tasks),
        successful_formal_outers=list(range(1, 201)), fully_skipped_attempts=sum(not t['complete_outer'] for t in tasks),
        actual_record_list_sha256=final['record_list_sha256'], fresh_original_proofs_sha256=canonical_sha(originals)), tasks

def teacher_evidence(repository, run):
    baseline, baseline_file = document(repository/BASELINE)
    retained = SharedReads(); retained.remember(baseline_file)
    payload = {k: baseline[k] for k in BASELINE_FIELDS}
    need(canonical_sha(payload) == BASELINE_PAYLOAD_SHA, 'authentic teacher decode baseline changed')
    measured = payload[BASELINE_FIELDS[0]]; cross = payload[BASELINE_FIELDS[1]]
    need(len(measured) == 10 and [r['chunk'] for r in measured] == list(range(10)) and
         cross['questions'] == 5120 and cross['trajectories'] == 25600 and
         cross['actual_all_teacher_global_qid_trajectory_uniqueness'] is True,
         'authentic teacher decode scope incomplete')
    root = Path(run['question_probe']['teacher_warmup_cache']).resolve(strict=True)
    need(root.is_relative_to(ROOT.parent) and root != ROOT.parent, 'teacher cache escaped experiments')
    identity, identity_file = document(root/'identity.json'); complete, complete_file = document(root/'complete.json')
    retained.remember(identity_file); retained.remember(complete_file)
    need(identity['training_world_size'] == 4 and identity['seed'] == run['question_probe']['seed'] and
         len(identity['questions']) == 5120 and len({q['qid'] for q in identity['questions']}) == 5120 and
         identity['model_revision'] == run['model_revision'] == OFFICIAL_MODEL_REVISION and
         identity['model_files_sha256'] == run['initial_model_sha256'] and
         identity['train_parquet_sha256'] == run['input_sha256']['train.parquet'] and
         complete['identity_sha256'] == identity_file['sha256'] and complete['questions'] == 5120 and
         complete['trajectories'] == 25600 and len(complete['chunks']) == 10, 'teacher namespace/full completion differs')
    train_sha = retained.read(BASE/'data/train.parquet', run['input_sha256']['train.parquet'],
                             alias=True, shared=True)['sha256']
    from search_g1.qprobe_warmup import verify_frozen, moments
    groups = []; trajectory_ids = []; proof_rows = []; counts = {str(k): 0 for k in range(6)}
    for index, previous in enumerate(measured):
        need(previous['uid'] == OWNER and previous['actual_cpu_only'] is True and
             previous['cache_writer_constructed'] is False and previous['cache_lock_taken'] is False and
             previous['model_weights_loaded'] is False and previous['generated_trajectories'] == 0 and
             previous['actor_or_head_updates_performed'] == 0 and previous['questions'] == 512 and
             previous['trajectories'] == 2560 and previous['identity_sha256'] == identity_file['sha256'] and
             previous['canonical_train_sha256'] == train_sha and
             all(previous[k] is True for k in ('canonical_all5120_seed_order_qid_text_rebuilt_equal',
                 'actual_all512_raw_strict_terminal_EM_K_mean_std_ddof1_equal',
                 'actual_frozen_four_rank_and_raw_provenance_equal')), 'authentic teacher decode flags differ')
        chunk_path = root/('chunk-%02d.json' % index)
        chunk, chunk_file = document(chunk_path)
        retained.remember(chunk_file)
        need(chunk_file['sha256'] == previous['chunk_sha256'] == complete['chunks'][index]['sha256'] and
             complete['chunks'][index]['relative_path'] == chunk_path.name and chunk['index'] == index and
             len(chunk['groups']) == 512, 'teacher label/chunk bytes changed since actual decode')
        raw = chunk['raw_rollout']; relative = Path(raw['relative_path'])
        need(relative.name == str(relative) and previous['raw_sha256'] == raw['sha256'] and
             previous['raw_bytes'] == raw['bytes'], 'authentic teacher raw identity differs')
        raw_file = read_file(root/relative, expected_sha=previous['raw_sha256'], expected_bytes=previous['raw_bytes'])[1]
        retained.remember(raw_file)
        verify_frozen(chunk['frozen_workers']['before'], chunk['frozen_workers']['after'], 4)
        expected_run = TEACHER_RUNS[index]
        need(previous['teacher_run_id'] == expected_run, 'historical teacher producer was relabeled')
        chunk_counts = {str(k): 0 for k in range(6)}
        for expected, group in zip(identity['questions'][index*512:(index+1)*512], chunk['groups']):
            need(group['qid'] == expected['qid'] and group['question_hash'] == expected['question_hash'] and
                 group['run_id'] == expected_run and group['source_outer'] == 0,
                 'teacher question/provenance differs')
            count, mean, std = moments(group['strict_em'])
            need((group['correct_count'], group['mean'], group['std']) == (count, mean, std) and
                 math.isclose(std, math.sqrt(sum((x-mean)**2 for x in group['strict_em'])/4), abs_tol=1e-14, rel_tol=1e-14),
                 'teacher sample std ddof1/K/mean differs')
            need(len(group['trajectory_ids']) == 5 and len(set(group['trajectory_ids'])) == 5,
                 'teacher actual G5 identities missing')
            trajectory_ids.extend(group['trajectory_ids']); groups.append(group); chunk_counts[str(count)] += 1
        need(chunk_counts == previous['K_distribution'] and
             sum(k*chunk_counts[str(k)] for k in range(6)) == previous['strict_correct'], 'teacher decoded K distribution differs')
        for k, value in chunk_counts.items(): counts[k] += value
        proof_rows.append(dict(chunk=chunk_file, raw=raw_file, teacher_run_id=expected_run))
    need(len(groups) == 5120 and len(set(trajectory_ids)) == len(trajectory_ids) == 25600 and
         counts == cross['K_distribution'], 'teacher global identities/K0/K5 changed')
    for mapping in (identity['scorer_sources'], identity['rollout_sources']):
        for name, sha in mapping.items():
            need(run['source_sha256'].get(name) == sha, 'teacher original source namespace differs')
            retained.read(ROOT/'source'/run['source_commit']/name, sha)
    for name in TOKENIZERS:
        retained.read(Path(run['initial_model'])/name, identity['model_files_sha256'][name], shared=True)
    retained.finish()
    return dict(baseline=baseline_file, authentic_original_summary_sha256=ORIGINAL_TEACHER_PROOF_SHA,
        authentic_payload_sha256=BASELINE_PAYLOAD_SHA, identity=identity_file, complete=complete_file,
        fresh_chunk_raw_proofs_sha256=canonical_sha(proof_rows), questions=5120, trajectories=25600,
        K_distribution=counts, actual_original_decode_bytes_still_equal=True,
        historical_first1024_relabelled_as_current=False, regenerated_trajectories=0, historical_teacher_run_ids=list(TEACHER_RUNS))


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def validate_archive_documents(claim, runner, task_binding, result, task, *,
                               step, binding, publication, allowed_controllers, alive,
                               final_submit=False):
    role = 'detached_cpu_final_audit_submission' if final_submit else 'detached_cpu_pair_archive'
    need(claim.get('schema_version') == 1 and claim.get('role') == role and
         re.fullmatch('[0-9a-f]{32}', claim.get('token', '')) is not None and
         (final_submit or (type(claim.get('step')) is int and claim['step'] == step)) and claim.get('producer') == binding and
         claim.get('source') == publication and claim.get('controller') in allowed_controllers and
         birth_ok(claim.get('controller')) and finite(claim.get('created_unix')) and
         all(claim.get(key) is False for key in ('training_permitted', 'direct_gpu_device_access_permitted')) and
         claim.get('native_gpu_audit_submission_permitted') is final_submit,
         'archive claim role/owner/source/controller/scope differs')
    directory = ROOT/'final-control'/RUN/('audit-controller' if final_submit else 'archive-'+str(step))
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
        raise ValueError('INDEPENDENT_SIGMA3: archive runner missing; unknown, never replay')
    parent = runner.get('actual_process')
    need(birth_ok(parent) and parent['pid'] == parent['sid'] and
         runner.get('controller') == claim['controller'], 'archive runner birth/controller differs')
    parent_live = alive(parent)
    child = task_binding.get('actual_process') if task_binding is not None else None
    if child is not None:
        need(birth_ok(child) and child['sid'] == parent['sid'] and child['pid'] != parent['pid'] and
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


def held_archive_evidence(run, producer, followthrough, follow_source, follow_process):
    """Consume existing held children; never create or replay an intent."""
    retained = SharedReads(); results = {}
    components = ROOT/'final-control'/RUN
    for step in (5,50,100,150,200,None):
        folder = components/('audit-controller' if step is None else 'archive-'+str(step))
        values = []
        for name in ('submit-claim.json','runner-binding.json','task-binding.json','worker-result.json','task-result.json'):
            value, proof = document(folder/name)
            retained.remember(proof); values.append(value)
        claim, runner, child, result, task = values
        measured = validate_archive_documents(claim,runner,child,result,task,step=step,
            binding=producer,publication=follow_source,allowed_controllers=[follow_process],
            alive=ctl.live,final_submit=step is None)
        need(measured['ready'] is True,'held archive/submit supervisor has not actually finished')
        if step is not None:
            _, pair = followthrough.control.pair(BASE,run,step)
            need(claim.get('pair') == pair,'held archive pair differs')
            transport_path = ROOT/'archives'/RUN/('global_step_'+str(step))/'complete.json'
            transport, proof = document(transport_path);retained.remember(proof)
            need(task['archive']['path'] == str(transport_path) and
                 task['archive']['sha256'] == proof['sha256'] and transport['run_id'] == RUN and
                 transport['step'] == step and transport['world_size'] == 4,
                 'held archive transport bytes/identity differ')
            measured['transport'] = proof
        else:
            need((task.get('submission') or {}).get('status') != 'SUBMIT_FAILED_DO_NOT_REPLAY',
                 'held native submit failed; never infer another submission')
            # A genuine sbatch timeout remains preserved in task-result.json.
            # Only the existing unique actual-job reconciliation may resolve it.
            submission = followthrough.control.inspect_audit(BASE)
            need(submission.get('status') in ('ACTUAL_SLURM_SUBMITTED','ACTUAL_SLURM_RECONCILED') and
                 isinstance(submission.get('job'),str),'held native submit outcome is unknown or failed')
            measured['actual_native_submission'] = dict(job=submission['job'],audit_base=submission['audit_base'])
        results[str(step) if step is not None else 'native_submit'] = measured
    retained.finish()
    return results, [value[0] for value in retained.files.values()]


def teacher_copy_evidence(run):
    target = BASE/'teacher-warmup-cache'
    need(Path(run['question_probe']['teacher_warmup_cache']) == target and
         target.resolve() == target and target != TEACHER_ORIGIN and
         TEACHER_ORIGIN.resolve() == TEACHER_ORIGIN,'teacher copy root aliases or differs')
    copied, copied_file = document(BASE/'preparation/teacher-copy-verification.json')
    names = ['identity.json','complete.json']+['chunk-%02d.json'%i for i in range(10)]+['rollout-%02d.pt'%i for i in range(10)]
    need(copied.get('origin') == str(TEACHER_ORIGIN) and copied.get('target') == str(target) and
         copied.get('questions') == 5120 and copied.get('trajectories') == 25600 and
         len(copied.get('files',[])) == len(names) and
         [row.get('name') for row in copied['files']] == names,'teacher copy receipt fileset/source differs')
    retained = SharedReads();retained.remember(copied_file)
    for row in copied['files']:
        original, current = TEACHER_ORIGIN/row['name'], target/row['name']
        a,b = original.lstat(),current.lstat()
        need(stat.S_ISREG(a.st_mode) and stat.S_ISREG(b.st_mode) and b.st_nlink == 1 and
             (a.st_dev,a.st_ino) != (b.st_dev,b.st_ino),'teacher target is not an independent regular copy')
        for path in (original,current):
            _,proof = read_file(path,expected_sha=row['sha256'],expected_bytes=row['bytes'])
            retained.remember(proof)
    complete, _ = document(target/'complete.json')
    need(copied.get('receipt') == complete,'teacher copy completion receipt differs')
    retained.finish()
    return dict(receipt=copied_file,origin=str(TEACHER_ORIGIN),target=str(target),
                files=len(names),all_original_and_independent_copy_bytes_equal=True)


def initial_head_evidence(run):
    """Load only the small CPU head/replay, never an actor/HF model."""
    import torch
    from search_g1.qprobe_model import QProbeConfig
    ready, ready_file = document(BASE/'artifacts/question-probe/teacher-warmup-ready.json')
    path = BASE/'artifacts/question-probe/teacher-warmup-head/initial.pt'
    proof = bound_record(ready['head_checkpoint'],path)
    head = torch.load(path,map_location='cpu',weights_only=True)
    expected = QProbeConfig(seed=20261013,bootstrap_max_groups_per_outer=5120,**LOSS)
    need(QProbeConfig.from_mapping(run['question_probe']) == expected and head.get('config') == asdict(expected),
         'initial head loss/training configuration differs')
    need(head.get('feature_dim') == 2048 and head.get('actor_snapshot_id') == RUN+':0' and
         head.get('completed_outer') == head.get('update_count') == 0 and not head.get('completed_attempts') and
         len(head.get('replay_labels',[])) == len({r['qid'] for r in head['replay_labels']}) == 5120 and
         tuple(head['replay_features'].shape) == (5120,2048) and
         ready.get('fitted_questions') == ready.get('reused_questions') == 5120 and
         ready.get('completed_training_outers') == 0,'initial head is not a new full5120 bootstrap')
    parameters = sum(value.numel() for key,value in head['model'].items() if key.startswith('net.'))
    optimizer = head['optimizer']['state']
    need(parameters == 135622 and len(optimizer) == 6 and
         all(float(state['step']) == 40 for state in optimizer.values()) and
         ready['head_metrics']['head_optimizer_updates'] == 40,'initial head capacity/actual bootstrap Adam differs')
    metrics = ready['head_metrics']
    need(math.isclose(metrics['replay_weighted_loss'],metrics['sigma_ce']+metrics['std_mse']+.1*metrics['count_ce'],
                      rel_tol=1e-6,abs_tol=2e-6),'actual bootstrap sigma3 objective differs')
    return dict(ready=ready_file,head=proof,head_parameters=parameters,actual_bootstrap_adam_steps=40,
                fresh_actor_outer=0,loss=LOSS,actor_model_weights_loaded=False,head_checkpoint_loaded=True)


def native_source_gate(run, native, audit_run, source_document, followthrough_commit):
    need(followthrough_commit == FOLLOWTHROUGH and
         native['audit_commit'] == audit_run['source_commit'] == followthrough_commit and
         audit_run['run_id'] == RUN and audit_run['final_restore_audit']['audit_id'] == native['audit_id'],
         'native audit source is not the authenticated sigma3 followthrough source')
    for name, sha in run['source_sha256'].items():
        if name.startswith(('verl/','search_g1/','search_r1/','reproduction/common8/')) and not Path(name).name.startswith('test_'):
            need(source_document['files'].get(name) == sha and audit_run['source_sha256'].get(name) == sha,
                 'native sigma3 scientific source differs: '+name)


def fresh_goal_gate(fresh):
    need(fresh['status'] == GOAL_STATUS and fresh['run_id'] == RUN and fresh['completed_outer'] == 200 and
         fresh['update201_performed'] is False and fresh['actual_attempts'] >= 200 and
         fresh['bitwise_continuation_of_failed_execution'] is None and
         not fresh.get('producer_final_cpu_recovery') and
         len(fresh['actual_training_parts']) == 1 and fresh['actual_training_parts'][0]['first'] == 1 and
         fresh['actual_training_parts'][0]['last'] == 200 and
         fresh['actual_training_parts'][0]['base'] == str(BASE) and
         fresh['actual_training_parts'][0]['source_commit'] == PRODUCER,'fresh complete sigma3 history differs')


def verify(commit, *, followthrough_commit):
    need(os.getuid() == OWNER and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
         os.environ.get('PYTHONDONTWRITEBYTECODE') == '1' and sys.dont_write_bytecode and
         all(os.environ.get(k) == '1' for k in CAPS),'actual CPU owner/caps/no-bytecode required')
    run, early = early_history()  # Before Torch/NumPy, publication or large I/O.
    need(followthrough_commit == FOLLOWTHROUGH,'exact published sigma3 followthrough pin required')
    producer_accounting, producer = producer_acceptance()
    repository, own_source = publication(commit)
    need(Path(__file__).resolve() == repository/'reproduction/hpc3_qprobe/independent_goal_acceptance_sigma3.py' and
         os.environ.get('QPROBE_SOURCE') == str(repository),'loaded independent immutable source differs')
    follow, follow_source = load_fixed(followthrough_commit,'followthrough_sigma3','control')
    audit_base, native, follow_proof = followthrough_evidence(follow,follow_source,followthrough_commit)
    held, held_anchors = held_archive_evidence(run,producer,follow,follow_source,follow_proof['process'])
    submitted = held['native_submit']['actual_native_submission']
    need(submitted['job'] == native['slurm_job_id'] and submitted['audit_base'] == str(audit_base),
         'actual native submit/restore identity differs')
    from reproduction.hpc3_qprobe.verify_goal import verify as verify_goal
    fresh = verify_goal(BASE,audit_base,publish_receipt=False)
    fresh_goal_gate(fresh)
    from reproduction.hpc3_qprobe.final_restore_audit import verify_two_boundaries
    allocation, _ = document(audit_base/'logs/native-allocation-admission.json')
    audit_run, audit_manifest = document(audit_base/'logs/run-manifest.json')
    source_document, _ = document(Path(follow_source['publication_path']))
    native_source_gate(run,native,audit_run,source_document,followthrough_commit)
    own_document, _ = document(Path(own_source['publication_path']))
    for name, sha in run['source_sha256'].items():
        if name.startswith(('verl/','search_g1/','search_r1/','reproduction/common8/')) and not Path(name).name.startswith('test_'):
            need(own_document['files'].get(name) == sha,'independent scientific source differs: '+name)
    receipt, _ = document(Path(native['actual_loaded_target_receipt']))
    verify_two_boundaries(native['workers_first'],native['workers_second'],allocation,receipt,
                          native['driver_first'],native['driver_second'])
    need(native.get('head_scheduler') is None,'head acquired a synthetic scheduler')
    for driver, phase in ((native['driver_first'],'first'),(native['driver_second'],'second')):
        need(driver.get('phase') == phase and driver.get('head_scheduler') is None and
             driver.get('head_schedule') == 'constant_lr_Adam' and all(driver.get(key) is True for key in (
                 'actual_head_model_adam_private_rng_replay_readback_equal','actual_selector_rng_readback_equal',
                 'actual_candidate_stream_readback_equal','actual_driver_rng_readback_equal')),'actual native head/data readback missing')
    from search_g1.qprobe_checkpoint import verified_runtime_prefix
    prefix = verified_runtime_prefix(audit_base,audit_run)
    need(prefix == native['actual_worker_restore_prefix'] and prefix['completed_outer'] == 200,'actual all4 restored prefix differs')
    for name in ('training-launch-claim.json','actual-updates.jsonl','actual-attempts.jsonl','five-updates-acceptance.json'):
        need(not (audit_base/'logs'/name).exists(),'native audit contains forbidden training output')
    feature_root = Path(run['question_probe']['cache_dir']).resolve(strict=True)
    manifest, feature_file = document(feature_root/'manifest.json')
    measured = native['actual_feature_cache']
    need(manifest['spec_sha'] == measured['spec_sha'] and manifest['count'] == measured['count'] == 169028 and
         manifest['spec']['feature_dim'] == measured['feature_dim'] == 2048 and
         manifest['files']['features.npy']['sha256'] == measured['feature_file_sha256'],'native fixed feature namespace differs')
    features = [feature_file]
    for name in ('features.npy','index.jsonl'):
        item = manifest['files'][name]
        features.append(read_file(feature_root/name,expected_sha=item['sha256'],expected_bytes=item['bytes'])[1])
    raw, raw_source = load_fixed(RAW,'raw_audit_followthrough_sigma3','ctl')
    raw_proof, tasks = raw_evidence(raw)
    need(raw_proof['source'] == raw_source and len(tasks) == fresh['actual_attempts'],'raw source/all normal-return attempt count differs')
    copy = teacher_copy_evidence(run)
    teacher = teacher_evidence(repository,run)
    initial = initial_head_evidence(run)
    anchors = list(early.values())+[follow_proof[key] for key in ('claim','binding','goal','accepted','native','result','process_file')]
    anchors += [audit_manifest,raw_proof['result'],raw_proof['claim'],copy['receipt'],initial['ready'],initial['head']]+features+held_anchors
    for row in anchors:
        read_file(row['path'],expected_sha=row['sha256'],expected_bytes=row.get('bytes'))
    after_accounting, after_producer = producer_acceptance()
    need(after_producer == producer,'producer recovery chain changed during final conjunction')
    completed(native['slurm_job_id'])
    return dict(status=STATUS,run_id=RUN,producer_job=JOB,producer_commit=PRODUCER,
        current_history=dict(first=1,last=200,fresh_official_actor=True),completed_outer=200,world_size=4,
        source=own_source,followthrough_commit=followthrough_commit,producer_accounting=after_accounting,
        producer_binding=producer,actual_followthrough_full_CPU_consumer=follow_proof,
        independent_all_normal_return_raw=raw_proof,authentic_teacher=teacher,independent_teacher_copy=copy,
        fresh_head=initial,fresh_full_goal=fresh,actual_archive_and_submit_held_children=held,
        actual_native_two_readbacks_reconsumed=True,update201_performed=False,
        receipt_or_cache_or_claim_writes=0,training_or_helper_or_native_job_submissions=0,
        actor_model_loaded=False,generated_trajectories=0,new_actor_head_or_scheduler_updates=0,
        goal_complete=False,checked_unix=time.time(),scope='pure sigma3 final conjunction; root reviews stdout before goal completion')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True);parser.add_argument('--followthrough-commit',required=True)
    args=parser.parse_args()
    print(__import__('json').dumps(verify(args.commit,followthrough_commit=args.followthrough_commit),allow_nan=False),flush=True)


if __name__ == '__main__':main()
