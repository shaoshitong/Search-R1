"""CPU metadata-only recovery of exactly twelve retained node10 attempts.

`plan` is local/static; `status` is read-only on physical NAS10. Only
`seal --execute` writes one immutable acceptance and missing old resolutions.
No service, HTTP, GPU, trajectory-body, model, sampler-state or audit execution.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time

BASE = Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
ALIAS = Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
RUN = '70188b8e50fd4b45bc420ce58014f46e'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
AUDITOR = '3def055adebfdb38e9326b9db385338c44800160c1b0a64e1ee78d8c3b68b906'
ROOT = BASE / 'preparation/node10-mixed-v2-v3-productive-closure-20261021'
ACCEPTANCE = ROOT / 'complete.json'
ACCEPTED = 'FIVE_MIXED_NODE10_SLOTS_EACH_TWO_AUDITED_NEW_FULL_G5_BATCHES'
RESOLVED = 'RECOVERED_FAILED_ATTEMPT_WITH_PRODUCTIVE_REPLACEMENT'
SLOTS = (('gpu1-tp1', 'v2', (1,), 1), ('gpu2-tp1', 'v3', (2,), 1),
         ('gpu3-tp1', 'v2', (3,), 1), ('gpu5-tp1', 'v3', (5,), 1), ('tp4-0-4-6-7', 'v3', (0, 4, 6, 7), 4))
TERMINALS = (
    ('node10-native-failures-20261021-v1', '2c3a7de77b48c08435bed4df6e045be5a44b39bfc276249925962e5848b6613d',
     'OWN_NODE10_NATIVE_FAILURES_ORIGINAL_SHA_RELEASE_PASS', (2371322, 2452973, 2830315, 2830845), False),
    ('node10-primary-v1-pre-model-terminal-20261021', '1397fbde7efee668f13dc31581001a05b15652aa172dc7fb567e0b284bd64751',
     'OWN_NODE10_PRIMARY_V1_PRE_MODEL_FAILURES_ORIGINAL_SHA_RELEASE_PASS', (2886544, 2887149, 2888798, 2890177, 2891150), False),
    ('node10-primary-v2-readiness-repair-20261021', '714aadbcaae9140a9c5cba7c0ae731677c12cc57699738ef5171ea06d31b8368',
     'OWN_NODE10_PRIMARY_V2_READINESS_CONTROLLED_STOP_ORIGINAL_SHA_RELEASE_PASS', (2905668, 2909621, 2910951), True))


def physical(path):
    path = Path(os.path.normpath(path))
    if path.is_relative_to(ALIAS):path = BASE / path.relative_to(ALIAS)
    assert path.is_relative_to(BASE), 'Path leaves the same-run NAS root'
    return path


def stat(path):
    s = path.stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, inode=s.st_ino, device=s.st_dev)


def record(path):
    path = Path(path);assert path.suffix == '.json', 'Only small JSON metadata is hashed'
    before = stat(path);assert before['bytes'] <= 2 * 1024 * 1024, 'Metadata exceeds bounded JSON size'
    raw = path.read_bytes();assert before == stat(path), 'Metadata changed during read'
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), value=json.loads(raw))


def reference(item):return dict(path=item['path'], sha256=item['sha256'])


def immutable(path, value):
    raw = (json.dumps(value, indent=2) + '\n').encode();temporary = path.with_name(path.name + '.tmp-%d-%d' % (os.getpid(), time.time_ns()))
    try:
        with temporary.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary, path)
        except FileExistsError:assert path.read_bytes() == raw, 'Existing immutable metadata differs'
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temporary.unlink(missing_ok=True)


def old_attempts():
    result, proofs = [], []
    for directory, digest, status, pids, controlled in TERMINALS:
        proof = record(BASE / 'preparation' / directory / 'complete.json');value = proof['value']
        assert proof['sha256'] == digest and value['status'] == status and value['run_id'] == RUN and value['node'] == 'h100-10'
        assert value['controlled_stop'] is controlled
        assert len(value['release_observations']) == 2
        first, second = value['release_observations'];assert second['epoch'] - first['epoch'] >= 3
        assert {row['pid'] for row in value['units']} == set(pids) and len(value['units']) == len(pids)
        originals = {row['path']: row for row in value['original_files']}
        for observation in (first, second):
            assert observation['remaining_known_pids'] == observation['own_gpu_contexts'] == [] and observation['own_cgroups_empty']
            assert all(row['observed']['MainPID'] == '0' and not row['remaining_cgroup_pids'] for row in observation['units'])
        for row in value['units']:
            pid, unit = row['pid'], row['unit'];job = physical(row['job_directory'])
            assert job.parent.parent == BASE / 'pool/jobs' and job.name.startswith('attempt-')
            prefix = 'relay10-primary' if pid >= 2886544 else 'relay10-tp1' if pid in (2830315, 2830845) else 'relay10'
            assert job.parent.name == 'job-%s-%d' % (prefix, pid)
            allocation, failure = record(job / 'allocation.json'), record(job / 'failure.json')
            assert allocation['value']['controller_pid'] == pid and allocation['value']['native_unit'] == unit
            for item in (allocation, failure):
                original = originals[item['path']];assert original['sha256'] == item['sha256']
                assert stat(Path(item['path'])) == {key: original[key] for key in ('bytes', 'mtime_ns', 'ctime_ns', 'inode', 'device')}
            if controlled:
                assert failure['value']['error_type'] == 'SystemExit' and failure['value']['error'] == '143'
                reason = 'AUTHORIZED_READINESS_REPAIR_CONTROLLED_SYSTEMEXIT143_NOT_FAILURE_OR_OOM'
            elif pid >= 2886544:reason = 'ACTUAL_PREMODEL_DEVICE_BINDING_KEYERROR_ORIGINAL_CUDA_VISIBLE_DEVICES'
            elif pid == 2830845:reason = 'ACTUAL_BORROWED_LOCAL_RELAY_DEPENDENCY_EXIT'
            else:reason = 'ACTUAL_ASYNC_RETRIEVAL_ATTEMPTS_EXHAUSTED_HTTPERROR'
            result.append(dict(pid=pid, unit=unit, directory=str(job), failure=reference(failure), terminal=reference(proof),
                original_attempt_reason=reason, original_attempt_controlled_stop=controlled))
        proofs.append(reference(proof))
    assert len(result) == len({row['pid'] for row in result}) == 12
    return result, proofs


def first_two(job, job_id):
    events = []
    with (job / 'sampler.log').open(errors='replace') as stream:
        for line in stream:
            try:event = json.loads(line)
            except ValueError:continue
            if not isinstance(event, dict) or event.get('status') != 'POOL_OWN_SEALED_BATCH_ADVANCED':continue
            assert event['run_id'] == RUN and event['job_id'] == job_id
            assert event['completed_own_batches'] == len(events) + 1
            events.append(event)
            if len(events) == 2:return events
    raise ValueError('Replacement has fewer than two actual own whole-batch events: ' + job_id)


def batch_paths(inp):
    dest = BASE / 'sampling' / inp.parent.name / 'batches' / inp.stem;outer = dest / 'outers/outer-000001'
    return dict(batch_seal=dest / 'complete.json', config=dest / 'config.json', questions=outer / 'questions.json',
        prefix_seal=outer / 'prefix/complete.json', prefix_scan=outer / 'prefix/scan.json', prefix_rank=outer / 'prefix/rank-0-complete.json',
        full_seal=outer / 'full/complete.json', full_rank=outer / 'full/rank-0-complete.json', selection=outer / 'probe-score/selection.json',
        prefix=outer / 'prefix/rank-0-prefix.jsonl.gz', full=outer / 'full/rank-0-full.jsonl.gz', own_em=outer / 'full/rank-0-own-em.jsonl')


def valid_cache(cache, audit, source_sha, config_sha, files, questions):
    assert cache and cache.get('status') == 'VERIFIED', 'Selected new batch has no VERIFIED independent audit cache'
    binding = cache['binding'];assert cache['verified_epoch'] <= audit['epoch']
    assert binding['run_id'] == RUN and binding['seed'] == 20261021 and binding['auditor_sha256'] == AUDITOR
    assert binding['root_config_sha256'] == config_sha and binding['input_sha256'] == source_sha
    assert binding['files'] == {key: stat(path) for key, path in files.items()}, 'Audited file binding changed'
    proof = cache['proof'];assert proof['questions'] == questions and proof['trajectories'] == proof['labels'] == 5 * questions


def valid_full(full, allocation, unit, indices, tp):
    assert full['run_id'] == RUN and full['phase'] == 'continue' and full['actor_version'] == 'official7b-' + REVISION
    device = full['device_binding']
    assert device['native_unit'] == unit, 'New full receipt belongs to another unit'
    assert device['tensor_parallel_size'] == tp and device['physical_gpu_indices'] == list(indices)
    assert device['gpu_uuids'] == allocation['model_cuda_uuids']
    assert device['normalized_cuda_visible_devices'] == ','.join(map(str, indices))


def replacement(slot, instance, indices, tp, audit, root_config, seen):
    unit = 'sst-q7b-g5-primary10-' + slot + '-' + instance + '-' + RUN[:12] + '.service'
    control = BASE / 'preparation' / ('relay-primary10-' + slot + '-' + instance)
    submit, intent = record(control / 'native-submit-receipt.json'), record(control / 'native-submit-intent.json')
    assert submit['value']['returncode'] == 0
    pid = int(submit['value']['observed']['properties']['MainPID']);assert pid > 0
    jobs = list((BASE / 'pool/jobs' / ('job-relay10-primary-' + str(pid))).glob('attempt-*'));assert len(jobs) == 1
    job = jobs[0];assert not any((job / name).exists() for name in ('failure.json', 'sampler-failure.json'))
    allocation = record(job / 'allocation.json');a = allocation['value']
    assert a['controller_pid'] == pid and a['run_id'] == RUN and a['node'] == 'h100-10' and a['instance'] == instance
    assert a['native_unit'] == unit and a['model_indices'] == list(indices) and a['tensor_parallel_size'] == tp
    assert a['source_owner']['pid'] == 2882823 and a['local_relay_process']['pid'] == 2882852
    evidence = record(physical(a['allocation_evidence']));e = evidence['value']
    assert evidence['sha256'] == a['allocation_evidence_sha256'] and e['main_pid'] == pid and e['unit'] == unit
    assert type(a['controller_birth']) is int and a['controller_birth'] == e['controller_birth'] and a['boot_id'] == e['boot_id']
    assert intent['value']['unit'] == unit and intent['value']['run_id'] == RUN
    context = record(physical(a['slot_context']));assert context['sha256'] == a['slot_context_sha256'] == intent['value']['dependency_context_sha256']
    bundle = intent['value']['source_bundle_sha256'];manifest = record(control / bundle / 'manifest.json')
    assert manifest['value']['source_bundle_sha256'] == bundle and manifest['value']['unit'] == unit and manifest['value']['run_id'] == RUN
    progress = record(job / 'progress.json');p = progress['value'];job_id = 'relay10-primary-' + str(pid)
    assert p['run_id'] == RUN and p['job_id'] == job_id
    assert p['completed_own_batches'] >= 2, slot + ' has fewer than two own complete batches: ' + str(p['completed_own_batches'])
    assert p['questions'] > 0 and p['trajectories'] == p['questions'] * 5
    initialized = [record(path) for path in sorted((job / 'sampler-launches').glob('launch-*/model-initialized.json'))];assert initialized
    for marker in initialized:
        m = marker['value'];assert m['actual_model_initialized'] is True and m['tensor_parallel_size'] == tp
        assert m['max_model_len'] == 4096 and m['actual_compute_dtype'] == 'torch.bfloat16'
    batches = []
    for event in first_two(job, job_id):
        inp = physical(event['latest_input']);assert inp.parent.parent == BASE / 'inputs' and inp.suffix == '.json'
        assert inp not in seen, 'Replacement slots claim a duplicate input batch';seen.add(inp)
        assert event['epoch'] <= audit['epoch'], 'Independent audit predates selected new batch'
        relative = str(inp.relative_to(BASE));assert relative not in audit['unsealed_inputs'] and relative not in audit['sealed_pending_inputs']
        source = record(inp);questions = len(source['value']['rows']);files = batch_paths(inp)
        cache = record(BASE / 'audit/batches' / inp.parent.name / (inp.stem + '.json'))
        valid_cache(cache['value'], audit, source['sha256'], root_config['sha256'], files, questions)
        metadata = {key: record(files[key]) for key in ('batch_seal', 'config', 'full_rank', 'full_seal')}
        seal, config, full, full_seal = (metadata[key]['value'] for key in ('batch_seal', 'config', 'full_rank', 'full_seal'))
        assert seal['run_id'] == RUN and seal['input_sha256'] == source['sha256'] and seal['questions'] == questions
        assert seal['trajectories'] == 5 * questions and seal['full'] == full
        assert config['run_id'] == RUN and config['seed'] == 20261021 and config['model_revision'] == REVISION
        assert config['actor_version'] == 'official7b-' + REVISION and config['warmup_only'] is True and config['actor_optimizer_updates'] == 0
        assert config['world_size'] == 1 and physical(config['actor_model_dir']).resolve() == physical(root_config['value']['model_dir']).resolve()
        valid_full(full, a, unit, indices, tp)
        assert physical(full['actor_model_dir']).resolve() == physical(root_config['value']['model_dir']).resolve()
        assert full['config_sha256'] == metadata['config']['sha256'] and full['questions'] == questions and full['trajectories'] == 5 * questions
        assert all(full_seal.get(key) == value for key, value in full.items()) and full_seal['labels'] == 5 * questions
        for key in ('full', 'own_em'):assert full[key]['sha256'] == cache['value']['proof']['independent_sha256'][key]
        valid_cache(cache['value'], audit, source['sha256'], root_config['sha256'], files, questions)
        batches.append(dict(event=event, input=reference(source), questions=questions, audit_cache=reference(cache),
            audit_proof=cache['value']['proof'], metadata={key: reference(value) for key, value in metadata.items()}))
    assert not any((job / name).exists() for name in ('failure.json', 'sampler-failure.json'))
    return dict(slot=slot, instance=instance, unit=unit, controller_pid=pid, directory=str(job),
        allocation=reference(allocation), evidence=reference(evidence), submit=reference(submit), intent=reference(intent),
        manifest=reference(manifest), model_initialized=[reference(row) for row in initialized], progress=p, first_two=batches)


def gather():
    olds, terminals = old_attempts();root = record(BASE / 'config.json');assert root['value']['run_id'] == RUN
    audit = record(BASE / 'audit/latest.json');a = audit['value']
    assert a['status'] in ('INCOMPLETE', 'COMPLETE') and a['run_id'] == RUN and a['seed'] == 20261021 and a['model_revision'] == REVISION
    assert a['base'] == str(BASE) and a['auditor_sha256'] == AUDITOR and a['failed_batches'] == 0 and a['errors'] == []
    assert a['strict_em_recomputed'] is False and a['states_shards_independently_audited'] is False
    seen = set();replacements = [replacement(*slot, a, root, seen) for slot in SLOTS]
    assert len(seen) == 10 and len({row['controller_pid'] for row in replacements}) == 5
    return dict(status=ACCEPTED, run_id=RUN, node='h100-10', terminals=terminals, old_attempts=olds,
        replacements=replacements, audit=audit, root_config=reference(root), replacement_new_complete_batches=10,
        replacement_new_complete_trajectories=sum(5 * batch['questions'] for row in replacements for batch in row['first_two']),
        audited_scope='First two own complete batches from each exact mixed v2/v3 slot; ten globally distinct inputs',
        retained_old_sealed_prefix_allowed=True, new_full_provenance_required=True, original_failures_retained=True,
        strict_em_recomputed=False, states_shards_independently_audited=False, whole_goal_complete=False,
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), epoch=time.time())


def resolution(old, acceptance):
    value = acceptance['value']
    return dict(status=RESOLVED, run_id=RUN, original_controller_pid=old['pid'], original_unit=old['unit'],
        original_attempt_reason=old['original_attempt_reason'], original_attempt_controlled_stop=old['original_attempt_controlled_stop'],
        resolution_kind='PRODUCTIVE_REPLACEMENT_AFTER_CONTROLLED_READINESS_REPAIR' if old['original_attempt_controlled_stop'] else 'PRODUCTIVE_REPLACEMENT_AFTER_ACTUAL_FAILURE',
        original_failure_sha256=old['failure']['sha256'], terminal_proof=old['terminal']['path'], terminal_proof_sha256=old['terminal']['sha256'],
        replacement_controller_pids=[row['controller_pid'] for row in value['replacements']],
        replacement_directories=[row['directory'] for row in value['replacements']], replacement_new_complete_batches=10,
        replacement_new_complete_trajectories=value['replacement_new_complete_trajectories'],
        immutable_acceptance=acceptance['path'], immutable_acceptance_sha256=acceptance['sha256'],
        original_failure_not_deleted_or_rewritten=True, strict_em_recomputed=False, states_shards_independently_audited=False,
        whole_goal_complete=False, epoch=value['epoch'])


def existing():
    acceptance = record(ACCEPTANCE);value = acceptance['value'];olds, terminals = old_attempts()
    assert value['status'] == ACCEPTED and value['run_id'] == RUN and value['terminals'] == terminals
    assert value['old_attempts'] == olds and value['replacement_new_complete_batches'] == 10
    assert value['source_code_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert value['strict_em_recomputed'] is False and value['states_shards_independently_audited'] is False
    assert [(r['slot'], r['instance']) for r in value['replacements']] == [(s[0], s[1]) for s in SLOTS]
    assert all(len(row['first_two']) == 2 for row in value['replacements']) and value['whole_goal_complete'] is False
    assert len({batch['input']['path'] for row in value['replacements'] for batch in row['first_two']}) == 10
    return acceptance


def report(acceptance, mutate=False):
    outputs = []
    for old in acceptance['value']['old_attempts']:
        path = Path(old['directory']) / 'resolution.json';expected = resolution(old, acceptance)
        if path.exists():assert record(path)['value'] == expected
        elif mutate:immutable(path, expected)
        outputs.append(dict(path=str(path), present=path.exists(), controlled_stop=old['original_attempt_controlled_stop']))
    return dict(status='EXISTING_ACCEPTANCE_READ_NO_PRODUCTIVITY_REPLAY', acceptance=reference(acceptance), resolutions=outputs,
        replacement_new_complete_batches=10, strict_em_recomputed=False, states_shards_independently_audited=False, whole_goal_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('action', choices=('plan', 'status', 'seal'), nargs='?', default='plan')
    parser.add_argument('--execute', action='store_true');args = parser.parse_args()
    if args.execute and args.action != 'seal':parser.error('--execute is only valid with seal')
    if args.action == 'plan':
        print(json.dumps(dict(status='LOCAL_STATIC_PLAN_NO_IO', slots=SLOTS, terminals=TERMINALS, old_attempts=12,
            required_distinct_audited_batches=10, acceptance=str(ACCEPTANCE), mutation='seal --execute only', whole_goal_complete=False)));return
    assert os.geteuid() == 0 and socket.gethostname().split('.')[0] == 'h100-10'
    if args.action == 'status':
        try:
            if ACCEPTANCE.exists():value = report(existing())
            else:
                candidate = gather();value = dict(status='READY_TO_SEAL_NO_MUTATION', replacement_new_complete_batches=10,
                    replacement_new_complete_trajectories=candidate['replacement_new_complete_trajectories'], old_attempts=12, whole_goal_complete=False)
        except (AssertionError, OSError, ValueError, KeyError) as error:
            value = dict(status='NOT_READY_NO_MUTATION', error_type=type(error).__name__, error=str(error), whole_goal_complete=False)
        print(json.dumps(value));return
    assert args.execute, 'seal requires explicit --execute'
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'seal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ACCEPTANCE.exists():acceptance = existing()
        else:
            candidate = gather()
            assert not any((Path(row['directory']) / 'resolution.json').exists() for row in candidate['old_attempts']), 'Resolution exists without this immutable acceptance'
            immutable(ACCEPTANCE, candidate);acceptance = record(ACCEPTANCE)
        print(json.dumps(report(acceptance, mutate=True)))


if __name__ == '__main__':main()
