"""Final, read-only CPU conjunction for the authentic719225 experiment.

This is a consumer, not a controller: it never creates a receipt/claim/cache,
launches a child/job, loads an actor, decodes again or regenerates trajectories.
The saved decode measurements are reusable only while every original byte is
still equal. Missing200 fails before imports of NumPy/Torch or large reads.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

ROOT = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
BASE = ROOT.parent/'qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014'
OLD_BASE = ROOT.parent/'qprobe-teacher-warmup200-4gpu-jxu957-recovery718509-20261014'
OWNER = 14090
SHARED_OWNER = 21930
SHARED_GROUP = 21919
RUN = 'c3856315d3f345d7920ebf5725a083b4'
PRODUCER = '1843babdd96b387dd10d91e3207c9ea9489b6941'
OLD_PRODUCER = '5816a20f331e10aaff9bbb95c313611ead289fd7'
EA = 'ea4601109969e358b65ab8251d64798926748b7d'
RAW = '7958980caabf9864e92db0ed991e8c8d2a2daa0b'
GOAL_STATUS = 'QPROBE_FULL200_TEACHER_HEAD_FIXED_VALIDATION_ARCHIVE_NATIVE4_GOAL_PASS'
STATUS = 'ACTUAL_INDEPENDENT_TEACHER_RAW_ALL200_ARCHIVES_NATIVE4_GOAL_PASS'
BASELINE = 'reproduction/hpc3_qprobe/teacher_raw_baseline719225.json'
# Canonical SHA of exactly these two fields of the actual durable15 audit.
# The baseline may be a small extracted JSON, never a newly fabricated success.
BASELINE_FIELDS = ('teacher_all10_actual_raw_redecode', 'teacher_cross_chunk_uniqueness')
BASELINE_PAYLOAD_SHA = '12bf48b2a0d1f5409df4e53d469d3c7c4a5754554f864f23658e628a12f9d2b9'
ORIGINAL_TEACHER_PROOF_SHA = 'd86e5bee7d25af134915c46143328c6789d4801c1ae83cdf8579264845b3669c'
OFFICIAL_MODEL_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
TOKENIZERS = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')
CAPS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
        'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def need(ok, message):
    if not ok:
        raise ValueError('INDEPENDENT_GOAL719225: '+message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def signature(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def read_file(path, *, expected_sha=None, expected_bytes=None, capture=False,
              shared=False, alias=False, mode=None, capture_limit=32*2**20):
    """One retained fd, complete hash, stable birth/metadata; never writes.

    Only manifest-bound canonical data/model assets may have a prepared symlink
    alias or the actual shared MOU owner. Private evidence/source cannot.
    """
    path = Path(path)
    need(path.is_absolute(), 'absolute file path required')
    resolved = path.resolve(strict=True)
    need(alias or (resolved == path and not path.is_symlink()), 'unsafe linked evidence: '+str(path))
    before = resolved.lstat()
    allowed_owner = (before.st_uid == OWNER or
        (shared and before.st_uid == SHARED_OWNER and before.st_gid == SHARED_GROUP and
         SHARED_GROUP in os.getgroups()))
    need(stat.S_ISREG(before.st_mode) and allowed_owner, 'file owner/type differs: '+str(path))
    if mode is not None:
        need(stat.S_IMODE(before.st_mode) == mode, 'private evidence mode differs')
    if capture:
        need(type(capture_limit) is int and 0 < capture_limit <= 128*2**20 and
             before.st_size <= capture_limit, 'bounded JSON/source read exceeded')
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    pieces = [] if capture else None
    with os.fdopen(os.open(str(resolved), flags), 'rb') as stream:
        opened = os.fstat(stream.fileno())
        need(signature(before) == signature(opened), 'file changed before open')
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8*2**20), b''):
            digest.update(block)
            if capture:
                pieces.append(block)
        after = os.fstat(stream.fileno())
    need(signature(before) == signature(after) == signature(resolved.lstat()) and
         path.resolve(strict=True) == resolved, 'file/alias changed during full read')
    actual_sha = digest.hexdigest()
    if expected_sha is not None:
        need(isinstance(expected_sha, str) and re.fullmatch('[0-9a-f]{64}', expected_sha) and
             actual_sha == expected_sha, 'fresh original full SHA differs: '+str(path))
    if expected_bytes is not None:
        need(type(expected_bytes) is int and before.st_size == expected_bytes, 'fresh original byte count differs')
    proof = dict(path=str(path), resolved_path=str(resolved), sha256=actual_sha, bytes=before.st_size,
                 stat_identity=list(signature(before)))
    return (b''.join(pieces) if capture else None), proof


def document(path):
    raw, proof = read_file(path, capture=True)
    value = json.loads(raw)
    need(isinstance(value, dict), 'JSON object required: '+str(path))
    return value, proof


def rows(path):
    raw, proof = read_file(path, capture=True)
    need(not raw or raw.endswith(b'\n'), 'terminal ledger is incomplete')
    result = [json.loads(line) for line in raw.splitlines()]
    need(all(isinstance(row, dict) for row in result), 'ledger object required')
    return result, proof


def completed(job):
    """Only read accounting; no allocation, reconciliation or submission."""
    need(isinstance(job, str) and re.fullmatch('[1-9][0-9]*', job), 'actual job required')
    raw = subprocess.check_output(['sacct', '-nP', '-X', '-j', job,
        '--format=JobID,State,ExitCode,User,UID,Account,QOS,Partition'], text=True, timeout=30)
    values = [line.split('|') for line in raw.splitlines() if line.strip()]
    need(values == [[job, 'COMPLETED', '0:0', 'jxu957', '14090', 'jxu957', 'formal-user', 'emergency_acd']],
         'actual job has not completed0 with the authenticated owner')
    return dict(job=job, state='COMPLETED', exit_code='0:0', raw=raw, checked_unix=time.time())


def early_history():
    """No heavy import, source traversal, process start or directory creation."""
    current, ledger = rows(BASE/'logs/actual-updates.jsonl')
    need([r.get('step') for r in current] == list(range(6, 201)) and
         all(type(r.get('step')) is int and r.get('run_id') == RUN for r in current),
         'actual current6..200 absent or logical201/copied prefix present')
    run, manifest = document(BASE/'logs/run-manifest.json')
    need(run.get('run_id') == RUN and run.get('source_commit') == PRODUCER and
         run.get('training_world_size') == 4 and run.get('actor_micro_batch_per_rank') == 4 and
         (run.get('full_state_restore') or {}).get('completed_outer') == 5 and
         run['full_state_restore'].get('kind') == 'question_probe_saved_checkpoint', 'actual restored5 producer differs')
    old, old_ledger = rows(OLD_BASE/'logs/actual-updates.jsonl')
    old_run, old_manifest = document(OLD_BASE/'logs/run-manifest.json')
    need([r.get('step') for r in old] == list(range(1, 6)) and
         all(type(r.get('step')) is int and r.get('run_id') == RUN for r in old) and
         old_run.get('run_id') == RUN and old_run.get('source_commit') == OLD_PRODUCER and
         old_run.get('training_world_size') == 4 and not old_run.get('full_state_restore'),
         'authentic original1..5 prefix differs')
    return run, old_run, dict(current_updates=ledger, original_updates=old_ledger,
                             current_manifest=manifest, original_manifest=old_manifest)


def publication(commit):
    need(isinstance(commit, str) and re.fullmatch('[0-9a-f]{40}', commit), 'exact source commit required')
    repository = ROOT/'source'/commit
    need(repository.resolve() == repository and repository.is_dir(), 'canonical published source root required')
    proof, record = document(ROOT/'preparation'/('source-'+commit+'-verification.json'))
    need(proof.get('commit') == commit and proof.get('status') == 'PUBLISHED_SOURCE_FULL_SHA_PASS' and
         isinstance(proof.get('files'), dict) and proof['files'], 'published full source verification missing')
    actual = {str(p.relative_to(repository)) for p in repository.rglob('*') if p.is_file() and
              (p.suffix in ('.py', '.sh', '.yaml', '.yml', '.json') or
               str(p.relative_to(repository)) in ('verl/version/version', 'verl/single_controller/version/version')) and
              not {'.git', '__pycache__', 'evidence'}.intersection(p.relative_to(repository).parts)}
    need(actual <= set(proof['files']), 'published runtime/JSON file omitted')
    for name, sha in proof['files'].items():
        path = repository/name
        need(path.is_relative_to(repository) and '..' not in Path(name).parts, 'source escapes publication')
        read_file(path, expected_sha=sha)
    return repository, dict(commit=commit, publication_path=record['path'], publication_sha256=record['sha256'])


def load_fixed(commit, filename, control_attribute):
    """Private imported objects; do not replace any active/shared module globals.

    Each pinned controller's source() checks its own __file__. Binding its
    actual pinned final_control avoids the future helper's module cache being
    mistaken for the EA/795 source. All imported bytes were just full-hashed.
    """
    repository, proof = publication(commit)
    def load(name):
        path = repository/'reproduction/hpc3_qprobe'/(name+'.py')
        spec = importlib.util.spec_from_file_location('_independent_'+commit+'_'+name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    controller = load('final_control')
    module = load(filename)
    setattr(module, control_attribute, controller)
    return module, proof


def birth_ok(value):
    return (isinstance(value, dict) and all(type(value.get(k)) is int and value[k] > 0
            for k in ('pid', 'start_ticks', 'sid', 'uid')) and value['uid'] == OWNER)


def ea_evidence(ea, publication_proof):
    observed = ea.inspect()
    need(observed.get('actual_live_birth') is False, 'EA actual supervisor has not finished')
    result = observed.get('result')
    process = observed['process']['actual_process']
    need(isinstance(result, dict) and type(result.get('exit_code')) is int and result['exit_code'] == 0 and result.get('error') is None and
         birth_ok(process) and process['sid'] == process['pid'] and result.get('actual_process') == process,
         'EA state alone or nonreturned CPU supervisor cannot pass')
    actual_result, result_file = document(ea.DIRECTORY/'result.json')
    actual_process, process_file = document(ea.DIRECTORY/'process.json')
    need(actual_result == result and actual_process == observed['process'], 'EA result/process changed during consumption')
    need(result.get('training_submission_performed') is False and result.get('actor_model_loaded') is False and
         result.get('direct_gpu_device_access_performed') is False and
         result.get('native_final_restore_only_submission_permitted') is True,
         'EA returned CPU scope differs')
    claim, claim_file = document(ea.DIRECTORY/'claim.json')
    need(claim['source'] == publication_proof and claim['source']['commit'] == EA and
         claim['binding'] == ea.fixed_binding() and observed['process']['commit'] == EA and
         claim.get('training_submission_permitted') is False and claim.get('actor_model_reload_permitted') is False and
         claim.get('direct_gpu_access_permitted') is False and claim.get('native_final_restore_only_submission_permitted') is True,
         'EA original once source/birth/intent differs')
    returned = result.get('result') or {}
    goal_path = BASE/'logs/qprobe-final-goal-acceptance.json'
    goal, goal_file = document(goal_path)
    binding, binding_file = document(ea.DIRECTORY/'final-consumer-binding.json')
    need(binding.get('actual_process') == process and binding.get('helper_commit') == EA and
         binding.get('full_cpu_consumer_returned') is True and
         binding.get('goal_sha256') == goal_file['sha256'] and returned.get('full_objective_accepted') is True and
         returned.get('actual_final_native_restore_accepted') is True and
         returned.get('phase') == 'ACTUAL_FULL_CPU_GOAL_CONSUMER_PASS' and
         returned.get('final_goal_path') == str(goal_path) and returned.get('final_goal_sha256') == goal_file['sha256'] and
         goal.get('status') == GOAL_STATUS and goal.get('run_id') == RUN and
         goal.get('completed_outer') == 200 and goal.get('update201_performed') is False,
         'EA actual full CPU consumer/goal binding differs')
    audit_base = Path(binding['audit_base'])
    need(audit_base.is_absolute() and audit_base.resolve() == audit_base and
         audit_base.parent == ROOT.parent and audit_base not in (BASE, OLD_BASE), 'actual final audit base unsafe')
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
         prior.get('raw', '').strip() == actual['raw'].strip(), 'EA native accounting differs from actual job')
    return audit_base, native, dict(claim=claim_file, binding=binding_file, goal=goal_file,
        accepted=accepted_file, native=native_file, accounting=actual,
        result=result_file, process_file=process_file, process=process)


def bound_record(record, expected, *, private=False):
    need(isinstance(record, dict) and record.get('path') == str(expected), 'decoded original path was relabeled')
    return read_file(expected, expected_sha=record['sha256'], expected_bytes=record['bytes'],
                     mode=0o600 if private else None)[1]


class SharedReads:
    """Full-hash identical canonical/source assets once, reject any later change.

    An attempt's raw and trace always get their own complete read. The same
    immutable train/tokenizer/science files would otherwise be reread hundreds
    of times. Reuse requires identical expected SHA and stable inode/ctime, and
    the final terminal pass checks every retained identity again.
    """
    def __init__(self):
        self.files = {}

    def read(self, path, sha, *, shared=False, alias=False):
        path = Path(path); key = str(path)
        if key in self.files:
            proof, resolved, identity = self.files[key]
            need(sha == proof['sha256'] and path.resolve(strict=True) == resolved and
                 signature(resolved.lstat()) == identity, 'shared original asset changed across attempt proofs')
            return proof
        proof = read_file(path, expected_sha=sha, shared=shared, alias=alias)[1]
        self.remember(proof)
        return proof

    def remember(self, proof):
        path = Path(proof['path']); resolved = path.resolve(strict=True)
        identity = tuple(proof['stat_identity'])
        need(str(resolved) == proof['resolved_path'] and signature(resolved.lstat()) == identity,
             'original asset changed after full read')
        old = self.files.get(str(path))
        need(old is None or (old[0]['sha256'] == proof['sha256'] and old[1:] == (resolved, identity)),
             'original asset changed across full reads')
        self.files[str(path)] = (proof, resolved, identity)

    def finish(self):
        for path, (_, resolved, identity) in self.files.items():
            need(Path(path).resolve(strict=True) == resolved and signature(resolved.lstat()) == identity,
                 'shared original asset changed before final acceptance')


def fresh_attempt(task, output, shared_reads=None):
    """Hashes original inputs of a real, already returned same-fd CPU decode."""
    p = task['producer']; base = Path(p['base']); aid = task['attempt_id']
    expected_part = ((OLD_BASE, OLD_PRODUCER, '718650') if task['step'] <= 5 else (BASE, PRODUCER, '719225'))
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
    # not a claim that old5816 used the current1843 identity.
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


def teacher_evidence(repository, old_run):
    baseline, baseline_file = document(repository/BASELINE)
    payload = {k: baseline[k] for k in BASELINE_FIELDS}
    need(canonical_sha(payload) == BASELINE_PAYLOAD_SHA, 'authentic teacher decode baseline changed')
    measured = payload[BASELINE_FIELDS[0]]; cross = payload[BASELINE_FIELDS[1]]
    need(len(measured) == 10 and [r['chunk'] for r in measured] == list(range(10)) and
         cross['questions'] == 5120 and cross['trajectories'] == 25600 and
         cross['actual_all_teacher_global_qid_trajectory_uniqueness'] is True,
         'authentic teacher decode scope incomplete')
    root = Path(old_run['question_probe']['teacher_warmup_cache']).resolve(strict=True)
    need(root.is_relative_to(ROOT.parent) and root != ROOT.parent, 'teacher cache escaped experiments')
    identity, identity_file = document(root/'identity.json'); complete, complete_file = document(root/'complete.json')
    need(identity['training_world_size'] == 4 and identity['seed'] == old_run['question_probe']['seed'] and
         len(identity['questions']) == 5120 and len({q['qid'] for q in identity['questions']}) == 5120 and
         identity['model_revision'] == old_run['model_revision'] == OFFICIAL_MODEL_REVISION and
         identity['model_files_sha256'] == old_run['initial_model_sha256'] and
         identity['train_parquet_sha256'] == old_run['input_sha256']['train.parquet'] and
         complete['identity_sha256'] == identity_file['sha256'] and complete['questions'] == 5120 and
         complete['trajectories'] == 25600 and len(complete['chunks']) == 10, 'teacher namespace/full completion differs')
    train_sha = read_file(OLD_BASE/'data/train.parquet', expected_sha=old_run['input_sha256']['train.parquet'],
                          alias=True, shared=True)[1]['sha256']
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
        need(chunk_file['sha256'] == previous['chunk_sha256'] == complete['chunks'][index]['sha256'] and
             complete['chunks'][index]['relative_path'] == chunk_path.name and chunk['index'] == index and
             len(chunk['groups']) == 512, 'teacher label/chunk bytes changed since actual decode')
        raw = chunk['raw_rollout']; relative = Path(raw['relative_path'])
        need(relative.name == str(relative) and previous['raw_sha256'] == raw['sha256'] and
             previous['raw_bytes'] == raw['bytes'], 'authentic teacher raw identity differs')
        raw_file = read_file(root/relative, expected_sha=previous['raw_sha256'], expected_bytes=previous['raw_bytes'])[1]
        verify_frozen(chunk['frozen_workers']['before'], chunk['frozen_workers']['after'], 4)
        expected_run = '0742d8e346284d6da9656a826ce14395' if index < 2 else RUN
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
            need(old_run['source_sha256'].get(name) == sha, 'teacher original source namespace differs')
            read_file(ROOT/'source'/old_run['source_commit']/name, expected_sha=sha)
    for name in TOKENIZERS:
        read_file(Path(old_run['initial_model'])/name, expected_sha=identity['model_files_sha256'][name], shared=True)
    return dict(baseline=baseline_file, authentic_original_summary_sha256=ORIGINAL_TEACHER_PROOF_SHA,
        authentic_payload_sha256=BASELINE_PAYLOAD_SHA, identity=identity_file, complete=complete_file,
        fresh_chunk_raw_proofs_sha256=canonical_sha(proof_rows), questions=5120, trajectories=25600,
        K_distribution=counts, actual_original_decode_bytes_still_equal=True,
        historical_first1024_relabelled_as_current=False, regenerated_trajectories=0)


def verify(commit):
    need(os.getuid() == OWNER and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
         os.environ.get('PYTHONDONTWRITEBYTECODE') == '1' and sys.dont_write_bytecode and
         all(os.environ.get(k) == '1' for k in CAPS), 'actual CPU-only owner/caps/no-bytecode required')
    # Fail before any heavy dependency/import/large source or evidence read.
    run, old_run, early = early_history()
    producer_accounting = completed('719225')
    repository, own_publication = publication(commit)
    need(Path(__file__).resolve() == repository/'reproduction/hpc3_qprobe/independent_goal_acceptance719225.py' and
         os.environ.get('QPROBE_SOURCE') == str(repository), 'loaded independent immutable source differs')
    ea, ea_publication = load_fixed(EA, 'followthrough719225', 'control')
    audit_base, native, ea_proof = ea_evidence(ea, ea_publication)
    from reproduction.hpc3_qprobe.verify_goal import verify as verify_goal
    fresh = verify_goal(BASE, audit_base, publish_receipt=False)
    need(fresh['status'] == GOAL_STATUS and fresh['run_id'] == RUN and fresh['completed_outer'] == 200 and
         fresh['update201_performed'] is False and fresh['actual_attempts'] >= 200 and
         fresh['bitwise_continuation_of_failed_execution'] is False, 'fresh full goal/failed-boundary scope differs')
    # Reconsume the native producer's actual two readbacks, never reload CUDA.
    from reproduction.hpc3_qprobe.final_restore_audit import verify_two_boundaries
    allocation, _ = document(audit_base/'logs/native-allocation-admission.json')
    audit_run, audit_manifest = document(audit_base/'logs/run-manifest.json')
    need(native['audit_commit'] == EA and audit_run['source_commit'] == EA and
         audit_run['run_id'] == RUN and (audit_run.get('final_restore_audit') or {}).get('audit_id') == native['audit_id'],
         'actual native audit source/manifest differs from EA submitted source')
    receipt, _ = document(Path(native['actual_loaded_target_receipt']))
    verify_two_boundaries(native['workers_first'], native['workers_second'], allocation, receipt,
                          native['driver_first'], native['driver_second'])
    need(native.get('head_scheduler') is None, 'head acquired a synthetic scheduler')
    for driver, phase in ((native['driver_first'], 'first'), (native['driver_second'], 'second')):
        need(driver.get('phase') == phase and driver.get('head_scheduler') is None and
             driver.get('head_schedule') == 'constant_lr_Adam' and all(driver.get(key) is True for key in (
                'actual_head_model_adam_private_rng_replay_readback_equal', 'actual_selector_rng_readback_equal',
                'actual_candidate_stream_readback_equal', 'actual_driver_rng_readback_equal')),
             'actual head/private RNG/replay/selector/stream/driver readback missing')
    from search_g1.qprobe_checkpoint import verified_runtime_prefix
    native_prefix = verified_runtime_prefix(audit_base, audit_run)
    need(native_prefix == native['actual_worker_restore_prefix'] and native_prefix['completed_outer'] == 200,
         'actual all4 HF/Adam/scheduler/RNG/driver restored prefix differs')
    for forbidden in ('training-launch-claim.json', 'actual-updates.jsonl', 'actual-attempts.jsonl', 'five-updates-acceptance.json'):
        need(not (audit_base/'logs'/forbidden).exists(), 'native restore audit contains forbidden training output')
    # Actual native restore verified the complete fixed feature cache. Preserve
    # that proof's binding to current bytes as well as the constructed stream.
    feature_root = Path(run['question_probe']['cache_dir']).resolve(strict=True)
    feature_manifest, feature_file = document(feature_root/'manifest.json')
    actual_features = native['actual_feature_cache']
    need(feature_manifest['spec_sha'] == actual_features['spec_sha'] and
         feature_manifest['count'] == actual_features['count'] and
         feature_manifest['spec']['feature_dim'] == actual_features['feature_dim'] and
         feature_manifest['files']['features.npy']['sha256'] == actual_features['feature_file_sha256'],
         'actual native feature-cache namespace differs')
    feature_proofs = [feature_file]
    for name in ('features.npy', 'index.jsonl'):
        expected = feature_manifest['files'][name]
        feature_proofs.append(read_file(feature_root/name, expected_sha=expected['sha256'], expected_bytes=expected['bytes'])[1])
    raw, raw_publication = load_fixed(RAW, 'raw_audit_followthrough719225', 'ctl')
    raw_proof, tasks = raw_evidence(raw)
    need(len(tasks) == fresh['actual_attempts'] and raw_proof['source'] == raw_publication,
         'independent raw omitted or added normal-return attempts')
    teacher = teacher_evidence(repository, old_run)
    # Prevent cross-gate time-of-check substitution; all terminal small anchors
    # are stable while the heavy full reads complete.
    for row in [early['current_updates'], early['original_updates'], early['current_manifest'],
                early['original_manifest'], ea_proof['claim'], ea_proof['binding'], ea_proof['goal'],
                ea_proof['accepted'], ea_proof['native'], ea_proof['result'], ea_proof['process_file'],
                audit_manifest, raw_proof['result'], raw_proof['claim']]+feature_proofs[:1]:
        read_file(row['path'], expected_sha=row['sha256'], expected_bytes=row['bytes'])
    for row in feature_proofs:
        need(signature(Path(row['resolved_path']).lstat()) == tuple(row['stat_identity']),
             'native fixed feature cache changed during final conjunction')
    completed('719225'); completed(native['slurm_job_id'])
    return dict(status=STATUS, run_id=RUN, producer_job='719225', producer_commit=PRODUCER,
        original_prefix=dict(job='718650', source_commit=OLD_PRODUCER, first=1, last=5),
        current_suffix=dict(first=6, last=200), completed_outer=200, world_size=4, source=own_publication,
        producer_accounting=producer_accounting, actual_EA_full_CPU_consumer=ea_proof,
        fresh_full_goal=fresh, independent_all_normal_return_raw=raw_proof, authentic_teacher=teacher,
        actual_native_two_readbacks_reconsumed=True, update201_performed=False,
        bitwise_continuation_of_failed_execution=False, receipt_or_cache_or_claim_writes=0,
        training_or_helper_or_native_job_submissions=0, actor_model_loaded=False, generated_trajectories=0,
        new_actor_head_or_scheduler_updates=0, goal_complete=False, checked_unix=time.time(),
        scope='pure final evidence conjunction; root alone marks the authorized goal complete after reviewing stdout')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.commit), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
