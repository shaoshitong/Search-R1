"""Close only evidenced old consumer alerts after two audited new whole batches.

CPU metadata only. Never launches, stops or edits any sampler, model or relay.
Original failures and terminal originals remain intact. Run only once the
actual V5 sampler has completed two own batches and a newer bank audit passed.
"""
import hashlib
import json
import os
from pathlib import Path
import time

BASE = Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
ROOT = BASE / 'preparation/relay-consumer10-control-v1'
RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


replacement = list((BASE / 'pool/jobs/job-relay10-2371322').glob('attempt-*'))
assert len(replacement) == 1
replacement = replacement[0]
progress = read(replacement / 'progress.json')
assert progress['run_id'] == RUN_ID and progress['completed_own_batches'] >= 2
assert not (replacement / 'failure.json').exists()
audit_path = BASE / 'audit/latest.json'
audit = read(audit_path)
assert audit['failed_batches'] == 0 and audit['verified_batches'] >= 253
assert audit['epoch'] >= progress['epoch'] and audit['sealed_pending_batches'] == 0
acceptance_root = ROOT / 'v5-productive-closure'
acceptance_root.mkdir(exist_ok=True)
acceptance_path = acceptance_root / 'accepted-two-whole-batches.json'
if acceptance_path.exists():
    acceptance = read(acceptance_path)
    assert acceptance['replacement_directory'] == str(replacement)
    assert acceptance['progress']['completed_own_batches'] >= 2
else:
    acceptance = dict(status='ACTUAL_TWO_NEW_WHOLE_G5_BATCHES_AND_NEWER_BANK_AUDIT',
        replacement_directory=str(replacement), progress=progress, audit=audit,
        strict_em_recomputed=False, states_shards_independently_audited=False,
        whole_goal_complete=False, epoch=time.time())
    create(acceptance_path, acceptance)

rows = [
    (1, 2316020, 'd191b88b93e4ea01b25932cd121d8acbfba75e16728d1340b392722e055adcb1',
     'ACTUAL_PREMODEL_ENTRY_DIRECTORY_FAILURE'),
    (2, 2327023, '1946c1e9d4ac5bfe9f776fa48a9c2e27d95550e0c901e5dd24eb42cfd1e5c5e8',
     'ACTUAL_BF16_WEIGHT_LOADING_CUDA_OOM'),
    (3, 2351003, '8de5fbb140d0ed8fb248a4f1d5b7a3d3b1533ad661873958bde0e3bc52e2ab33',
     'ACTUAL_PROFILE_TORCH_AND_NCCL_MEMORY_FAILURE'),
    (4, 2362735, '0ca236ddb49fc9890cfcad34d800cdfae189940e3e0c5216717cb36880952dea',
     'ACTUAL_FULL4096_CONTEXT_CACHE_MINIMUM_REJECTED'),
]
output = []
for version, pid, proof_sha, reason in rows:
    attempts = list((BASE / ('pool/jobs/job-relay10-' + str(pid))).glob('attempt-*'))
    assert len(attempts) == 1
    old = attempts[0]
    failure = old / 'failure.json'
    proof = ROOT / ('v%d-terminal/complete.json' % version)
    assert failure.is_file() and sha(proof) == proof_sha
    terminal = read(proof)
    assert terminal['controller_pid'] == pid and terminal['new_complete_batches'] == 0
    assert terminal['status'] == 'OWN_CONSUMER_V%d_TERMINAL_FULL_SHA_RELEASE_PASS' % version
    target = old / 'resolution.json'
    if target.exists():
        existing = read(target)
        assert existing['original_failure_sha256'] == sha(failure)
        assert existing['terminal_proof_sha256'] == proof_sha
        assert existing['replacement_new_complete_batches'] >= 2
        output.append(dict(path=str(target), status='EXISTING_RESOLUTION_READ_NO_REPLAY'))
        continue
    value = dict(status='RECOVERED_FAILED_ATTEMPT_WITH_PRODUCTIVE_REPLACEMENT',
        run_id=RUN_ID, original_attempt_reason=reason,
        original_failure_sha256=sha(failure), terminal_proof=str(proof), terminal_proof_sha256=proof_sha,
        replacement_directory=str(replacement),
        replacement_new_complete_batches=acceptance['progress']['completed_own_batches'],
        replacement_new_complete_trajectories=acceptance['progress']['trajectories'],
        immutable_acceptance=str(acceptance_path), immutable_acceptance_sha256=sha(acceptance_path),
        original_failure_not_deleted_or_rewritten=True, whole_goal_complete=False, epoch=time.time())
    create(target, value)
    output.append(dict(path=str(target), sha256=sha(target), status=value['status']))
print(json.dumps(dict(resolutions=output, acceptance=str(acceptance_path),
    acceptance_sha256=sha(acceptance_path), whole_goal_complete=False)))
