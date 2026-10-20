"""Close historical attempt alerts only after two real, independently audited v6 batches."""
import hashlib
import json
import os
import time
from pathlib import Path

b = Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


replacement = list((b / 'pool/jobs/job-shared5-1490214').glob('attempt-*'))
assert len(replacement) == 1
progress_path = replacement[0] / 'progress.json'
progress = read(progress_path)
assert progress['completed_own_batches'] >= 2
assert progress['run_id'] == '70188b8e50fd4b45bc420ce58014f46e'
assert not (replacement[0] / 'failure.json').exists()
audit_path = b / 'audit/latest.json'
audit = read(audit_path)
assert audit['failed_batches'] == 0 and audit['verified_batches'] >= 246
assert audit['epoch'] >= progress['epoch']
output = []
for job, proof_name, proof_sha, reason in [
    ('job-shared5-1259750', 'shared-vram-terminal-v3-controlled-throughput',
     '169e0892d5084658736cd4c097f55d98b041bf7a1d0951a29fe7badaea543e57',
     'CONTROLLED_THROUGHPUT_STOP_AFTER_TWO_REAL_BATCHES_NOT_OOM'),
    ('job-shared5-1437188', 'shared-vram-terminal-v5',
     '2ae5c349e7980503292de3c28ecbf0a8b0d9bfe4ac83bff69b264eca25b47cd2',
     'ACTUAL_SECOND_TP2_ENGINE_WEIGHT_LOADING_CUDA_OOM'),
    ('job-shared5-1437188-engine-00', 'shared-vram-terminal-v5',
     '2ae5c349e7980503292de3c28ecbf0a8b0d9bfe4ac83bff69b264eca25b47cd2',
     'OWN_FIRST_ENGINE_STOPPED_AFTER_OTHER_ENGINE_WEIGHT_OOM'),
    ('job-shared5-1437188-engine-01', 'shared-vram-terminal-v5',
     '2ae5c349e7980503292de3c28ecbf0a8b0d9bfe4ac83bff69b264eca25b47cd2',
     'ACTUAL_WEIGHT_LOADING_CUDA_OOM_THREE_CONSTRUCTOR_ATTEMPTS'),
]:
    attempts = list((b / 'pool/jobs' / job).glob('attempt-*'))
    assert len(attempts) == 1
    old = attempts[0]
    failure = old / 'failure.json'
    assert failure.is_file()
    proof = b / 'preparation' / proof_name / 'complete.json'
    assert sha(proof) == proof_sha
    target = old / 'resolution.json'
    if target.exists():
        value = read(target)
        assert value['original_failure_sha256'] == sha(failure)
        assert value['terminal_proof_sha256'] == proof_sha
        assert value['replacement_new_complete_batches'] >= 2
        output.append(dict(path=str(target), status='EXISTING_RESOLUTION_READ_NO_REPLAY'))
        continue
    value = dict(
        status='RECOVERED_FAILED_ATTEMPT_WITH_PRODUCTIVE_REPLACEMENT',
        run_id=progress['run_id'], original_attempt_reason=reason,
        original_failure_sha256=sha(failure), terminal_proof=str(proof),
        terminal_proof_sha256=proof_sha, replacement_directory=str(replacement[0]),
        replacement_new_complete_batches=progress['completed_own_batches'],
        replacement_new_complete_trajectories=progress['trajectories'],
        replacement_progress_sha256=sha(progress_path),
        independent_audit=str(audit_path), independent_audit_sha256=sha(audit_path),
        original_failure_not_deleted_or_rewritten=True,
        whole_goal_complete=False, epoch=time.time(),
    )
    with target.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    output.append(dict(path=str(target), sha256=sha(target),
                       replacement_batches=progress['completed_own_batches'],
                       status=value['status']))
print(json.dumps(dict(resolutions=output, whole_goal_complete=False)))
