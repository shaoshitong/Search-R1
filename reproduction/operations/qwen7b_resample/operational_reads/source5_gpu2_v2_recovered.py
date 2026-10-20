"""Close only GPU2 V2's failed import after productive, audited GPU2 V3 work.

Run on physical NAS10 as root after publication. CPU metadata only: no model,
process control, audit replay, sampler edits or deletion of the original failure.
Existing immutable acceptance/resolution is checked and read without replay.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import time

BASE = Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
RECORDED_BASE = '/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020'
RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
OLD_PID = 2175836
OLD_UNIT = 'sst-q7b-g5-source5-extra-gpu2-v2-70188b8e50fd.service'
NEW_UNIT = 'sst-q7b-g5-source5-extra-gpu2-v3-70188b8e50fd.service'
PROOF = BASE / 'preparation/source5-model-sublease-gpu2-v2/v2-terminal/complete.json'
NEW_ROOT = BASE / 'preparation/source5-model-sublease-gpu2-v3'
ACCEPTANCE = NEW_ROOT / 'v3-productive-closure/accepted-two-whole-batches.json'
RESOLVED = 'RECOVERED_FAILED_ATTEMPT_WITH_PRODUCTIVE_REPLACEMENT'
ACCEPTED = 'ACTUAL_TWO_NEW_WHOLE_G5_BATCHES_AND_NEWER_BANK_AUDIT'


def record(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), value=json.loads(raw))


def original_path(path):
    return BASE / Path(path).relative_to(RECORDED_BASE) if str(path).startswith(RECORDED_BASE + '/') else Path(path)


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
            assert path.read_bytes() == payload, 'Conflicting immutable receipt already exists'
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def unique_attempt(pid):
    paths = list((BASE / ('pool/jobs/job-source5-model-' + str(pid))).glob('attempt-*'))
    assert len(paths) == 1
    return paths[0]


def two_own_batches(replacement, pid):
    events = []
    with (replacement / 'sampler.log').open(errors='replace') as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get('status') != 'POOL_OWN_SEALED_BATCH_ADVANCED':
                continue
            assert event['run_id'] == RUN_ID and event['job_id'] == 'source5-model-' + str(pid)
            assert event['completed_own_batches'] == len(events) + 1
            inp = original_path(event['latest_input'])
            assert inp.is_relative_to(BASE / 'inputs')
            cache = BASE / 'audit/batches' / inp.parent.name / (inp.stem + '.json')
            source = record(inp)
            source['questions'] = len(source.pop('value')['rows'])
            events.append(dict(progress=event, input=source, audited_batch=record(cache)))
            if len(events) == 2:
                return events
    raise ValueError('Two actual own batch progress records were not found')


def accepted(value, old_allocation):
    assert value['status'] == ACCEPTED and value['run_id'] == RUN_ID
    pid = value['replacement_controller_pid']
    assert type(pid) is int and pid > 0 and pid != OLD_PID
    replacement = unique_attempt(pid)
    assert value['replacement_directory'] == str(replacement)
    submit = value['submit_receipt']
    assert submit['path'] == str(NEW_ROOT / 'submit-receipt.json')
    assert int(submit['value']['observed']['properties']['MainPID']) == pid
    assert record(Path(submit['path']))['sha256'] == submit['sha256']
    allocation = value['allocation']['value']
    assert value['allocation']['path'] == str(replacement / 'allocation.json')
    assert allocation['controller_pid'] == pid and allocation['native_unit'] == NEW_UNIT
    assert allocation['run_id'] == RUN_ID and allocation['node'] == 'h100-5'
    assert allocation['tensor_parallel_size'] == 1 and allocation['model_indices'] == [2]
    assert allocation['source_parent'] == old_allocation['source_parent']
    assert record(Path(value['allocation']['path']))['sha256'] == value['allocation']['sha256']
    progress, audit = value['progress'], value['audit']
    assert progress['run_id'] == RUN_ID and progress['job_id'] == 'source5-model-' + str(pid)
    assert type(progress['completed_own_batches']) is int and progress['completed_own_batches'] >= 2
    assert type(progress['questions']) is int and type(progress['trajectories']) is int
    assert progress['questions'] > 0 and progress['trajectories'] == 5 * progress['questions']
    assert audit['status'] in ('INCOMPLETE', 'COMPLETE') and audit['run_id'] == RUN_ID and audit['seed'] == 20261021
    assert audit['base'] == str(BASE) and audit['failed_batches'] == 0 and audit['errors'] == []
    assert audit['sealed_pending_batches'] == 0 and audit['epoch'] >= progress['epoch']
    assert audit['verified_unique_trajectories'] >= progress['trajectories']
    assert audit['strict_em_recomputed'] is False and audit['states_shards_independently_audited'] is False
    assert len(value['two_own_audited_batches']) == 2
    inputs = set()
    for ordinal, batch in enumerate(value['two_own_audited_batches'], 1):
        event, source, cache = batch['progress'], batch['input'], batch['audited_batch']
        inp = original_path(event['latest_input'])
        relative = str(inp.relative_to(BASE))
        assert relative not in audit['unsealed_inputs'] and relative not in audit['sealed_pending_inputs']
        assert event['run_id'] == RUN_ID and event['job_id'] == 'source5-model-' + str(pid)
        assert event['completed_own_batches'] == ordinal and inp not in inputs
        inputs.add(inp)
        current = record(inp)
        assert source['path'] == str(inp) and current['sha256'] == source['sha256']
        assert cache['path'] == str(BASE / 'audit/batches' / inp.parent.name / (inp.stem + '.json'))
        assert record(Path(cache['path']))['sha256'] == cache['sha256']
        verified = cache['value']
        assert verified['status'] == 'VERIFIED' and verified['verified_epoch'] <= audit['epoch']
        assert verified['binding']['run_id'] == RUN_ID and verified['binding']['seed'] == 20261021
        assert verified['binding']['auditor_sha256'] == audit['auditor_sha256']
        assert verified['binding']['input_sha256'] == source['sha256']
        count = len(current['value']['rows'])
        assert source['questions'] == count
        assert verified['proof']['questions'] == count and verified['proof']['trajectories'] == verified['proof']['labels'] == 5 * count
    assert value['model_initialized']
    for entry in value['model_initialized']:
        marker_path = Path(entry['path'])
        assert marker_path.is_relative_to(replacement / 'sampler-launches') and marker_path.name == 'model-initialized.json'
        assert record(marker_path)['sha256'] == entry['sha256']
        marker = entry['value']
        assert marker['actual_model_initialized'] is True and marker['tensor_parallel_size'] == 1
        assert marker['max_model_len'] == 4096
    assert value['whole_goal_complete'] is False


def main():
    assert os.geteuid() == 0 and socket.gethostname().split('.')[0] == 'h100-10'
    assert record(BASE / 'config.json')['value']['run_id'] == RUN_ID
    old = unique_attempt(OLD_PID)
    proof, failure, old_allocation = record(PROOF), record(old / 'failure.json'), record(old / 'allocation.json')
    terminal = proof['value']
    assert terminal['status'] == 'OWN_SOURCE5_GPU2_V2_TERMINAL_FULL_SHA_RELEASE_PASS'
    assert terminal['controller_pid'] == OLD_PID and terminal['unit'] == OLD_UNIT and terminal['run_id'] == RUN_ID
    assert terminal['job_directory'] == str(old)
    assert terminal['new_complete_batches'] == terminal['new_trajectories'] == 0
    assert terminal['real_gpu_model_initialized'] is False and terminal['new_sampling_request_started'] is False
    originals = {row['path']: row['sha256'] for row in terminal['original_files']}
    for entry in (failure, old_allocation):
        assert originals[entry['path']] == entry['sha256']
    parent = old_allocation['value']['source_parent']
    assert parent['pid'] == 1490214 and str(parent['birth']) == '6807965'
    assert parent['unit'] == 'sst-q7b-g5-sharedvram-v6-70188b8e50fd.service'
    assert original_path(parent['allocation_path']).is_relative_to(BASE / 'pool/jobs/job-shared5-1490214')
    assert record(original_path(parent['allocation_path']))['sha256'] == parent['allocation_sha256']
    target = old / 'resolution.json'
    if not ACCEPTANCE.exists():
        assert not target.exists(), 'Resolution exists without its immutable acceptance'
        submit = record(NEW_ROOT / 'submit-receipt.json')
        pid = int(submit['value']['observed']['properties']['MainPID'])
        replacement = unique_attempt(pid)
        assert not any((replacement / name).exists() for name in ('failure.json', 'sampler-failure.json'))
        allocation = record(replacement / 'allocation.json')
        progress = record(replacement / 'progress.json')
        initialized = [record(p) for p in sorted((replacement / 'sampler-launches').glob('launch-*/model-initialized.json'))]
        audit = record(BASE / 'audit/latest.json')
        value = dict(status=ACCEPTED, run_id=RUN_ID, replacement_controller_pid=pid,
            replacement_directory=str(replacement), allocation=allocation, submit_receipt=submit,
            progress=progress['value'], progress_sha256=progress['sha256'], audit=audit['value'], audit_sha256=audit['sha256'],
            two_own_audited_batches=two_own_batches(replacement, pid),
            model_initialized=initialized, original_failure_sha256=failure['sha256'], terminal_proof_sha256=proof['sha256'],
            strict_em_recomputed=False, states_shards_independently_audited=False, whole_goal_complete=False, epoch=time.time())
        accepted(value, old_allocation['value'])
        assert not any((replacement / name).exists() for name in ('failure.json', 'sampler-failure.json'))
        immutable(ACCEPTANCE, value)
    acceptance = record(ACCEPTANCE)
    value = acceptance['value']
    accepted(value, old_allocation['value'])
    assert value['original_failure_sha256'] == failure['sha256'] and value['terminal_proof_sha256'] == proof['sha256']
    existing = target.exists()
    if not existing:
        immutable(target, dict(status=RESOLVED, run_id=RUN_ID,
            original_attempt_reason='ACTUAL_PREMODEL_PHYSICAL_WORLD_SIZE_CONTRACT_IMPORT_FAILURE',
            original_controller_pid=OLD_PID, original_failure_sha256=failure['sha256'],
            terminal_proof=str(PROOF), terminal_proof_sha256=proof['sha256'],
            replacement_directory=value['replacement_directory'], replacement_controller_pid=value['replacement_controller_pid'],
            replacement_new_complete_batches=value['progress']['completed_own_batches'],
            replacement_new_complete_trajectories=value['progress']['trajectories'],
            immutable_acceptance=str(ACCEPTANCE), immutable_acceptance_sha256=acceptance['sha256'],
            original_failure_not_deleted_or_rewritten=True, whole_goal_complete=False, epoch=time.time()))
    resolution = record(target)
    result = resolution['value']
    assert result['status'] == RESOLVED and result['run_id'] == RUN_ID and result['original_controller_pid'] == OLD_PID
    assert result['original_failure_sha256'] == failure['sha256'] and result['terminal_proof_sha256'] == proof['sha256']
    assert result['terminal_proof'] == str(PROOF) and result['immutable_acceptance'] == str(ACCEPTANCE)
    assert result['immutable_acceptance_sha256'] == acceptance['sha256']
    assert result['replacement_directory'] == value['replacement_directory']
    assert result['replacement_controller_pid'] == value['replacement_controller_pid']
    assert result['replacement_new_complete_batches'] == value['progress']['completed_own_batches']
    assert result['replacement_new_complete_trajectories'] == value['progress']['trajectories']
    assert result['original_failure_not_deleted_or_rewritten'] is True and result['whole_goal_complete'] is False
    print(json.dumps(dict(status=RESOLVED, resolution=str(target), sha256=resolution['sha256'],
        existing_resolution_read_no_replay=existing, acceptance=str(ACCEPTANCE), acceptance_sha256=acceptance['sha256'],
        replacement_controller_pid=value['replacement_controller_pid'],
        replacement_new_complete_batches=result['replacement_new_complete_batches'],
        replacement_new_complete_trajectories=result['replacement_new_complete_trajectories'],
        strict_em_recomputed=False, states_shards_independently_audited=False, whole_goal_complete=False)))


if __name__ == '__main__':
    main()
