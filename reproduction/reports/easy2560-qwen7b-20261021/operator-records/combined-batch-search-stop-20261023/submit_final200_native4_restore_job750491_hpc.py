"""Future once-only HPC CPU operator; do not run before the final receipt exists.

First use the existing allocated-CPU easy2560_final_checkpoint create command
AFTER actual200/native51713/seven-source scores/all five ancestral archives.
Deploy published operator code into TARGET/source/OPERATOR_COMMIT with the
existing source deployer. This operator then prepares and submits one separate
four-GPU restore-only job. It never creates the receipt, trains, or monitors.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-batch-searchstop-r1-20261023')
SOURCE_JOB = '750491'
TRAINING_COMMIT = 'dde1452d2ec65fb3f4a0464d4beacf3b8136414c'
RETRIEVER_JOB = '750471'
TARGET = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-finalrestore750491-20261023'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-receipt', required=True, type=Path)
    parser.add_argument('--target-base', default=TARGET, type=Path)
    parser.add_argument('--operator-commit', default=TRAINING_COMMIT)
    args = parser.parse_args()
    if sys.platform == 'darwin':
        raise SystemExit('Remote CPU operator only. No final-state read or submission on Mac.')
    target = args.target_base
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(target / 'source' / args.operator_commit))
    from search_g1.easy2560_final_checkpoint import source
    from reproduction.easy2560_joint4.final_restore_only import prepare
    from reproduction.easy2560_joint4.control import own_base, publish, sha
    target = own_base(target)
    claim = target / 'preparation/final200-native4-submit-claim.json'
    submission = target / 'preparation/final200-native4-submission.json'
    if claim.exists() or (target / 'job.id').exists():
        print(json.dumps(dict(status='EXISTING_ATTEMPT_INSPECT_DO_NOT_REPLAY',
            claim=str(claim), submission=str(submission), automatic_retry=False)), flush=True)
        return
    path, receipt, checkpoint, origin, run = source(args.source_receipt)
    assert origin == SOURCE and receipt['source_job'] == SOURCE_JOB
    assert receipt['source_commit'] == run['source_commit'] == TRAINING_COMMIT
    external = run['external_retriever']
    assert external['service_job_id'] == RETRIEVER_JOB
    ready = external['service_ready']
    assert ready['request_rebatching_enabled'] is True and ready['singleton_e5'] is True
    assert ready['faiss_batch_size'] == 16 and ready['faiss_max_wait_ms'] == 2
    progress = receipt['final_cpu_acceptance']['progress']
    assert receipt['world_size'] == progress['rank_count'] == 4
    assert receipt['attempt_progress']['cumulative_writes'] == 4000
    assert receipt['actual_optimizer_writes_per_rank'] == progress['cumulative_actual_optimizer_writes_per_rank'] == 4000
    assert progress['actual_completed_updates'] == 200
    assert progress['no_update201'] is True and progress['no_post200_driver_scheduler_worker_attempts'] is True
    assert sorted(row['rank'] for row in receipt['ranks']) == list(range(4))
    for row in receipt['ranks']:
        assert row['actual_optimizer_writes'] == 4000 and row['completed_outer'] == 200
        assert row['scheduler_last_epoch'] == 200
        assert row['scheduler_step_count'] == 201  # One initialization call; not outer201 or write4001.
    publish(claim, dict(source_base=str(SOURCE), source_job=SOURCE_JOB, training_commit=TRAINING_COMMIT,
        retriever_job=RETRIEVER_JOB, source_receipt=str(path), source_receipt_sha256=sha(path),
        source_checkpoint=str(checkpoint), target_base=str(target), operator_commit=args.operator_commit,
        expected_world_size=4, expected_adam_writes_per_rank=4000, expected_scheduler_last_epoch=200,
        scheduler_step_count_including_initialization=201, update201_allowed=False,
        automatic_retry=False, training_modified=False, created_unix=time.time()))
    prepared = prepare(target, path, args.operator_commit)
    command = ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user',
        '--account=jxu957', '--nodes=1', '--ntasks=1', '--gres=gpu:4', '--cpus-per-task=48',
        '--mem=512G', '--time=01:00:00', '--exclude=ACD1-13', '--export=NONE',
        '--job-name=easy7b-final200-native4-restore', '--chdir=' + str(target / 'repo'),
        '--output=' + str(target / 'logs/restore-%j.log'), '--error=' + str(target / 'logs/restore-%j.log'),
        str(target / 'repo/reproduction/easy2560_joint4/final_restore_only.sh'), str(target)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    accepted_output = SOURCE / 'preparation/final200-native4-actual-restore-accepted.json'
    consume = ['env', 'QPROBE_SOURCE=' + str(target / 'repo'), 'CUDA_VISIBLE_DEVICES=',
        str(target / 'env/bin/python'), '-B', '-m', 'reproduction.easy2560_joint4.final_cpu_acceptance',
        '--base', str(SOURCE), '--training-commit', TRAINING_COMMIT,
        '--final-restore-proof', str(target / 'logs/final-restore-only-complete.json'),
        '--output', str(accepted_output)]
    record = dict(source_job=SOURCE_JOB, target_base=str(target), source_receipt=str(path), prepared=prepared,
        command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        acceptance_after_actual_restore_command=consume, automatic_retry=False,
        actual_restore_accepted=False, training_modified=False, checked_unix=time.time())
    if result.returncode == 0:
        job = result.stdout.strip().split(';')[0]
        if not job.isdecimal():
            publish(submission, dict(record, status='UNKNOWN_SUBMISSION_INSPECT_CLAIM'))
            raise SystemExit('Unknown submission outcome; do not repeat.')
        record['job'] = job
        with (target / 'job.id').open('x') as stream:
            stream.write(job + '\n'); stream.flush(); os.fsync(stream.fileno())
    publish(submission, record)
    print(json.dumps(record, sort_keys=True), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
