"""Close only retired TP4/2021698 after three TP1s each seal two audited full G5s.

Run only after publication, as root on physical NAS10. Reads small metadata,
canonical inputs and existing audit proofs; never reads trajectory bodies or
loads a model. The old failure and partial files remain intact. A full receipt
must belong to its replacement unit; a retained old sealed prefix is allowed.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import time

BASE = Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
RECORDED_ROOT = Path('/mnt/shared/workspace/training_data_node_10')
RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
AUDITOR_SHA = '3def055adebfdb38e9326b9db385338c44800160c1b0a64e1ee78d8c3b68b906'
OLD_PID, OLD_BIRTH = 2021698, '7430669'
OLD_UNIT = 'sst-q7b-g5-source5-extra-model-v1-' + RUN_ID[:12] + '.service'
ROOT = BASE / 'preparation/source5-model-tp4-to-tp1-v1'
PROOF = ROOT / 'complete.json'
PROOF_SHA = '67b096a2d6852b3791184e05c8c44cf9be45cd826f367124cec27bec90a8024c'
ACCEPTANCE = ROOT / 'tp1-productive-closure/complete.json'
ACCEPTED = 'THREE_TP1_REPLACEMENTS_EACH_TWO_AUDITED_NEW_FULL_G5_BATCHES'
RESOLVED = 'RECOVERED_FAILED_ATTEMPT_WITH_PRODUCTIVE_REPLACEMENT'
GROUPS = (
    (4, 2271442, 'b6acea02fc0b6f9dbab4765581804919cf6f8273532ba0d40ecec097326302e6'),
    (6, 2272406, '7021b497adcd05437ddd5f56a6cc55f6d5e6186ce42a9d2b3f5a03e22c3a9237'),
    (7, 2272887, '6dccf4c69b514728462f59b54fc706872bcb92dacd5dc1541babf4c4590af723'),
)
RUNTIME_SHA = {
    'source_sublease_worker.py': 'cb8dcb34a25fd0235d8094ff1bdfa0f2e13d4babe9832f8cba80e82205321447',
    'pool_worker.py': 'a7d30f57ab4995fece1a236317060f2fc0d5f39f7c0031899e1bd65a663f3750',
    'shared_vram_worker.py': '3a601748de704c4baee82bcb53927fe8e1f511f3f7af0d05953afabc4c6ef172',
}


def physical(path):
    path = Path(os.path.normpath(path))
    return Path('/mnt/data') / path.relative_to(RECORDED_ROOT) if path.is_relative_to(RECORDED_ROOT) else path


def record(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), value=json.loads(raw))


def reference(path):
    item = record(path)
    return dict(path=item['path'], sha256=item['sha256'])


def checked(item):
    path = Path(item['path'])
    assert path.is_relative_to(BASE) or path.is_relative_to(Path('/mnt/data/shaoshitong'))
    current = record(path)
    assert current['sha256'] == item['sha256']
    return current['value']


def immutable(path, value):
    payload = (json.dumps(value, indent=2) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-%d-%d' % (os.getpid(), time.time_ns()))
    try:
        with temporary.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            assert path.read_bytes() == payload, 'Conflicting immutable closure already exists'
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def attempt(pid):
    paths = list((BASE / ('pool/jobs/job-source5-model-' + str(pid))).glob('attempt-*'))
    assert len(paths) == 1
    return paths[0]


def stat(path):
    value = path.stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                inode=value.st_ino, device=value.st_dev)


def batch_paths(inp):
    dest = BASE / 'sampling' / inp.parent.name / 'batches' / inp.stem
    outer = dest / 'outers/outer-000001'
    return dict(batch_seal=dest / 'complete.json', config=dest / 'config.json', questions=outer / 'questions.json',
        prefix_seal=outer / 'prefix/complete.json', prefix_scan=outer / 'prefix/scan.json',
        prefix_rank=outer / 'prefix/rank-0-complete.json', full_seal=outer / 'full/complete.json',
        full_rank=outer / 'full/rank-0-complete.json', selection=outer / 'probe-score/selection.json',
        prefix=outer / 'prefix/rank-0-prefix.jsonl.gz', full=outer / 'full/rank-0-full.jsonl.gz',
        own_em=outer / 'full/rank-0-own-em.jsonl')


def first_two(job, pid):
    events = []
    with (job / 'sampler.log').open(errors='replace') as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get('status') != 'POOL_OWN_SEALED_BATCH_ADVANCED':
                continue
            assert event['run_id'] == RUN_ID and event['job_id'] == 'source5-model-' + str(pid)
            assert event['completed_own_batches'] == len(events) + 1
            inp = physical(event['latest_input'])
            assert inp.is_relative_to(BASE / 'inputs')
            files = batch_paths(inp)
            events.append(dict(progress=event, input=reference(inp),
                audit_cache=reference(BASE / 'audit/batches' / inp.parent.name / (inp.stem + '.json')),
                metadata={key: reference(files[key]) for key in ('batch_seal', 'config', 'full_rank', 'full_seal')}))
            if len(events) == 2:
                return events
    raise ValueError('Two actual own whole-batch events are not available')


def replacement(index, pid, bundle):
    job = attempt(pid)
    assert not any((job / name).exists() for name in ('failure.json', 'sampler-failure.json'))
    root = BASE / ('preparation/source5-tp1-replacement-gpu%d-v1' % index)
    return dict(index=index, controller_pid=pid, bundle=bundle, directory=str(job),
        allocation=reference(job / 'allocation.json'), submit=reference(root / 'submit-receipt.json'),
        intent=reference(root / 'submit-intent.json'), manifest=reference(root / bundle / 'manifest.json'),
        progress=record(job / 'progress.json')['value'], first_two=first_two(job, pid),
        model_initialized=[record(path) for path in sorted((job / 'sampler-launches').glob('launch-*/model-initialized.json'))])


def validate(value, old_allocation, root_config):
    assert value['status'] == ACCEPTED and value['run_id'] == RUN_ID and value['terminal_proof_sha256'] == PROOF_SHA
    assert checked(value['root_config']) == root_config
    audit = value['audit']['value']
    assert audit['status'] in ('INCOMPLETE', 'COMPLETE') and audit['run_id'] == RUN_ID and audit['seed'] == 20261021
    assert audit['model_revision'] == REVISION and audit['base'] == str(BASE) and audit['auditor_sha256'] == AUDITOR_SHA
    assert audit['failed_batches'] == 0 and audit['errors'] == [] and audit['sealed_pending_batches'] == 0
    assert audit['strict_em_recomputed'] is False and audit['states_shards_independently_audited'] is False
    seen, trajectories = set(), 0
    assert len(value['replacements']) == 3
    for item, (index, pid, bundle) in zip(value['replacements'], GROUPS):
        job = attempt(pid)
        unit = 'sst-q7b-g5-source5-tp1-replacement-gpu%d-v1-%s.service' % (index, RUN_ID[:12])
        assert (item['index'], item['controller_pid'], item['bundle'], item['directory']) == (index, pid, bundle, str(job))
        root = BASE / ('preparation/source5-tp1-replacement-gpu%d-v1' % index)
        assert item['allocation']['path'] == str(job / 'allocation.json')
        assert item['submit']['path'] == str(root / 'submit-receipt.json') and item['intent']['path'] == str(root / 'submit-intent.json')
        assert item['manifest']['path'] == str(root / bundle / 'manifest.json')
        allocation, manifest, intent, submit = (checked(item[key]) for key in ('allocation', 'manifest', 'intent', 'submit'))
        assert allocation['run_id'] == RUN_ID and allocation['controller_pid'] == pid and allocation['node'] == 'h100-5'
        assert allocation['native_unit'] == unit and allocation['tensor_parallel_size'] == 1 and allocation['model_indices'] == [index]
        assert allocation['source_parent'] == old_allocation['source_parent']
        assert allocation['source_sublease_worker_sha256'] == RUNTIME_SHA['source_sublease_worker.py']
        assert manifest['source_bundle_sha256'] == intent['source_bundle_sha256'] == bundle
        assert manifest['unit'] == intent['unit'] == unit and manifest['run_id'] == intent['run_id'] == RUN_ID
        assert manifest['model_gpus'] == [index] and manifest['tensor_parallel_size'] == 1 and manifest['context_schema_version'] == 3
        assert manifest['retired_terminal']['sha256'] == intent['retired_terminal']['sha256'] == PROOF_SHA
        for name, digest in RUNTIME_SHA.items():
            assert manifest['source_files'][name]['sha256'] == digest
        assert int(submit['observed']['properties']['MainPID']) == pid and submit['returncode'] == 0
        progress = item['progress']
        assert progress['run_id'] == RUN_ID and progress['job_id'] == 'source5-model-' + str(pid)
        assert type(progress['completed_own_batches']) is int and progress['completed_own_batches'] >= 2
        assert progress['trajectories'] == 5 * progress['questions'] and len(item['first_two']) == 2
        assert item['model_initialized']
        for marker in item['model_initialized']:
            assert Path(marker['path']).is_relative_to(job / 'sampler-launches')
            initialized = checked(marker)
            assert initialized['actual_model_initialized'] is True and initialized['tensor_parallel_size'] == 1
            assert initialized['max_model_len'] == 4096 and initialized['actual_compute_dtype'] == 'torch.bfloat16'
        for ordinal, batch in enumerate(item['first_two'], 1):
            event = batch['progress']
            inp = physical(event['latest_input'])
            relative = str(inp.relative_to(BASE))
            assert event['run_id'] == RUN_ID and event['job_id'] == progress['job_id'] and event['completed_own_batches'] == ordinal
            assert inp not in seen and audit['epoch'] >= event['epoch']
            assert relative not in audit['unsealed_inputs'] and relative not in audit['sealed_pending_inputs']
            seen.add(inp)
            source, cache = checked(batch['input']), checked(batch['audit_cache'])
            assert batch['input']['path'] == str(inp)
            assert batch['audit_cache']['path'] == str(BASE / 'audit/batches' / inp.parent.name / (inp.stem + '.json'))
            files = batch_paths(inp)
            binding = cache['binding']
            assert cache['status'] == 'VERIFIED' and cache['verified_epoch'] <= audit['epoch']
            assert binding['run_id'] == RUN_ID and binding['seed'] == 20261021 and binding['auditor_sha256'] == AUDITOR_SHA
            assert binding['root_config_sha256'] == value['root_config']['sha256'] and binding['input_sha256'] == batch['input']['sha256']
            assert binding['files'] == {key: stat(path) for key, path in files.items()}
            count = len(source['rows'])
            assert cache['proof']['questions'] == count and cache['proof']['trajectories'] == cache['proof']['labels'] == 5 * count
            metadata = {}
            for key, entry in batch['metadata'].items():
                assert entry['path'] == str(files[key])
                metadata[key] = checked(entry)
            seal, config, full, full_seal = (metadata[key] for key in ('batch_seal', 'config', 'full_rank', 'full_seal'))
            assert seal['run_id'] == RUN_ID and seal['input_sha256'] == batch['input']['sha256']
            assert seal['questions'] == count and seal['trajectories'] == 5 * count and seal['full'] == full
            assert config['run_id'] == RUN_ID and config['seed'] == 20261021 and config['model_revision'] == REVISION
            assert config['actor_version'] == 'official7b-' + REVISION and config['warmup_only'] is True
            assert config['actor_optimizer_updates'] == 0 and config['world_size'] == 1
            assert physical(config['actor_model_dir']).resolve() == physical(root_config['model_dir']).resolve()
            assert full['phase'] == 'continue' and full['run_id'] == RUN_ID and full['actor_version'] == 'official7b-' + REVISION
            assert full['config_sha256'] == batch['metadata']['config']['sha256']
            assert physical(full['actor_model_dir']).resolve() == physical(root_config['model_dir']).resolve()
            assert full['questions'] == count and full['trajectories'] == 5 * count
            assert all(full_seal.get(key) == field for key, field in full.items()) and full_seal['labels'] == 5 * count
            device = full['device_binding']
            assert device['native_unit'] == unit and device['tensor_parallel_size'] == 1
            assert device['physical_gpu_indices'] == [index] and device['gpu_uuids'] == allocation['model_cuda_uuids']
            assert device['normalized_cuda_visible_devices'] == str(index)
            for key in ('full', 'own_em'):
                assert full[key]['sha256'] == cache['proof']['independent_sha256'][key]
            trajectories += 5 * count
    assert len(seen) == value['replacement_new_complete_batches'] == 6
    assert trajectories == value['replacement_new_complete_trajectories']
    assert value['strict_em_recomputed'] is False and value['states_shards_independently_audited'] is False
    assert value['whole_goal_complete'] is False


def main():
    assert os.geteuid() == 0 and socket.gethostname().split('.')[0] == 'h100-10'
    config = record(BASE / 'config.json')
    assert config['value']['run_id'] == RUN_ID and config['value']['seed'] == 20261021 and config['value']['model_revision'] == REVISION
    model_receipt = record(physical(config['value']['model_dir']) / 'complete.json')
    assert model_receipt['value']['status'] == 'OFFICIAL_SNAPSHOT_ALL_FILES_FRESH_SHA_VERIFIED'
    assert model_receipt['value']['revision'] == REVISION and model_receipt['value']['model'] == 'Qwen/Qwen2.5-7B-Instruct'
    old = attempt(OLD_PID)
    proof, failure, allocation = record(PROOF), record(old / 'failure.json'), record(old / 'allocation.json')
    assert proof['sha256'] == PROOF_SHA
    terminal = proof['value']
    assert terminal['status'] == 'OWN_SOURCE5_MODEL_TP4_CONTROLLED_THROUGHPUT_STOP_FULL_SHA_RELEASE_PASS' and terminal['run_id'] == RUN_ID
    assert terminal['retired']['pid'] == OLD_PID and terminal['retired']['birth'] == OLD_BIRTH and terminal['retired']['unit'] == OLD_UNIT
    assert physical(terminal['retired']['allocation_path']) == old / 'allocation.json'
    assert terminal['retired']['allocation_sha256'] == allocation['sha256']
    originals = {str(physical(row['path'])): row['sha256'] for row in terminal['original_files']}
    assert originals[failure['path']] == failure['sha256'] and originals[allocation['path']] == allocation['sha256']
    assert failure['value']['error_type'] == 'SystemExit' and str(failure['value']['error']) == '143'
    parent = allocation['value']['source_parent']
    assert parent['pid'] == 1490214 and str(parent['birth']) == '6807965'
    assert parent['unit'] == 'sst-q7b-g5-sharedvram-v6-' + RUN_ID[:12] + '.service'
    target = old / 'resolution.json'
    if not ACCEPTANCE.exists():
        assert not target.exists(), 'Resolution exists without its immutable closure proof'
        replacements = [replacement(*group) for group in GROUPS]
        value = dict(status=ACCEPTED, run_id=RUN_ID, terminal_proof_sha256=PROOF_SHA,
            original_failure_sha256=failure['sha256'], root_config=dict(path=config['path'], sha256=config['sha256']),
            official_model_receipt=dict(path=model_receipt['path'], sha256=model_receipt['sha256']),
            replacements=replacements, audit=record(BASE / 'audit/latest.json'),
            replacement_new_complete_batches=6,
            replacement_new_complete_trajectories=sum(checked(batch['audit_cache'])['proof']['trajectories']
                for item in replacements for batch in item['first_two']),
            audited_scope='Exactly the first two own whole batches from each of the three replacement controllers',
            retained_old_sealed_prefix_allowed=True, new_full_provenance_required=True,
            original_partial_outputs_retained=True, no_zero_partial_loss_claim=True,
            strict_em_recomputed=False, states_shards_independently_audited=False, whole_goal_complete=False, epoch=time.time())
        validate(value, allocation['value'], config['value'])
        for item in replacements:
            assert not any((Path(item['directory']) / name).exists() for name in ('failure.json', 'sampler-failure.json'))
        immutable(ACCEPTANCE, value)
    acceptance = record(ACCEPTANCE)
    value = acceptance['value']
    validate(value, allocation['value'], config['value'])
    assert value['original_failure_sha256'] == failure['sha256']
    assert checked(value['official_model_receipt']) == model_receipt['value']
    existing = target.exists()
    if not existing:
        immutable(target, dict(status=RESOLVED, run_id=RUN_ID, original_controller_pid=OLD_PID,
            original_attempt_reason='CONTROLLED_TP4_THROUGHPUT_REPLACEMENT_SYSTEMEXIT143_WITH_RETAINED_PARTIAL_OUTPUTS',
            original_failure_sha256=failure['sha256'], terminal_proof=str(PROOF), terminal_proof_sha256=PROOF_SHA,
            replacement_controller_pids=[pid for _, pid, _ in GROUPS],
            replacement_directories=[item['directory'] for item in value['replacements']],
            replacement_new_complete_batches=6, replacement_new_complete_trajectories=value['replacement_new_complete_trajectories'],
            immutable_acceptance=str(ACCEPTANCE), immutable_acceptance_sha256=acceptance['sha256'],
            original_failure_not_deleted_or_rewritten=True, original_partial_outputs_retained=True,
            strict_em_recomputed=False, states_shards_independently_audited=False, whole_goal_complete=False, epoch=time.time()))
    resolution = record(target)
    result = resolution['value']
    assert result['status'] == RESOLVED and result['run_id'] == RUN_ID and result['original_controller_pid'] == OLD_PID
    assert result['original_failure_sha256'] == failure['sha256'] and result['terminal_proof_sha256'] == PROOF_SHA
    assert result['terminal_proof'] == str(PROOF) and result['immutable_acceptance'] == str(ACCEPTANCE)
    assert result['immutable_acceptance_sha256'] == acceptance['sha256'] and result['replacement_new_complete_batches'] == 6
    assert result['replacement_controller_pids'] == [pid for _, pid, _ in GROUPS]
    assert result['replacement_directories'] == [item['directory'] for item in value['replacements']]
    assert result['replacement_new_complete_trajectories'] == value['replacement_new_complete_trajectories']
    assert result['original_failure_not_deleted_or_rewritten'] is True and result['whole_goal_complete'] is False
    print(json.dumps(dict(status=RESOLVED, resolution=str(target), sha256=resolution['sha256'],
        existing_resolution_read_no_replay=existing, closure=str(ACCEPTANCE), closure_sha256=acceptance['sha256'],
        replacement_controller_pids=result['replacement_controller_pids'], audited_new_complete_batches=6,
        audited_new_complete_trajectories=value['replacement_new_complete_trajectories'],
        strict_em_recomputed=False, states_shards_independently_audited=False, whole_goal_complete=False)))


if __name__ == '__main__':
    main()
