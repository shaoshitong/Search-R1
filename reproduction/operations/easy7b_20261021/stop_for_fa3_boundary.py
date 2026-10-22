"""One explicit authorized producer stop after the next full boundary is sealed.

Run from the published operator tree on the login host. This never edits live
training code, never stops retrieval, and never submits a replacement job.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def stop(target_base, authorization_path=None):
    from reproduction.easy2560_joint4.control import own_base, publish, sha
    target = own_base(target_base)
    authorization_path = (Path(authorization_path) if authorization_path is not None else
                          Path(__file__).with_name('human-fa3-sdpa-transition-authorization.json'))
    authorization = json.loads(authorization_path.read_text())
    source = own_base(authorization['source_base'])
    job = authorization['source_job']
    if os.getuid() != 14090 or (source/'job.id').read_text().strip() != job:
        raise ValueError('Owned producer identity differs')
    result_path = target/'preparation/fa3-source-stop-result.json'
    claim_path = target/'preparation/fa3-source-stop-claim.json'
    if result_path.exists() or claim_path.exists():
        return dict(status='EXISTING_STOP_ATTEMPT_INSPECT_ACTUAL',
                    claim=str(claim_path), result=str(result_path))
    ready = json.loads((target/'preparation/fa3-transition-prepared.json').read_text())
    if ready['status'] != 'READY_FOR_AUTHORIZED_BOUNDARY_TRANSITION':
        raise ValueError('Replacement environment is not prepared')
    run = json.loads((source/'logs/run-manifest.json').read_text())
    if run['run_id'] != authorization['source_run_id'] or run['source_commit'] != authorization['source_commit']:
        raise ValueError('Actual producer differs from human authorization')
    latest = json.loads((source/'preparation/rolling-checkpoints/latest.json').read_text())
    if latest['step'] < authorization['transition_boundary_minimum_outer']:
        return dict(status='WAITING_NEXT_SEALED_FULL_BOUNDARY', latest=latest)
    if latest['status'] != 'EASY2560_ROLLING_CHECKPOINT_READY' or latest['run_id'] != run['run_id']:
        raise ValueError('Actual source boundary differs')
    manifest = Path(latest['manifest_path'])
    if sha(manifest) != latest['manifest_sha256']:
        raise ValueError('Small source boundary seal changed')
    if not Path(latest['checkpoint']).is_dir():
        raise ValueError('Source checkpoint is absent')
    queue = subprocess.run(['squeue', '-j', job, '-h', '-o', '%i|%u|%T'],
                           capture_output=True, text=True, timeout=12)
    if queue.returncode or queue.stdout.strip() != job+'|jxu957|RUNNING':
        return dict(status='SOURCE_NOT_RUNNING_INSPECT_TERMINAL', queue=queue.stdout,
                    stderr=queue.stderr, latest=latest)
    claim = dict(source_job=job, source_base=str(source), target_base=str(target),
        source_boundary_observed=latest, target_prepared=ready,
        authorization=authorization, authorization_sha256=sha(authorization_path),
        checked_unix=time.time(), command=['scancel', job],
        reason='Explicit user-authorized FA3 transition at a complete checkpoint',
        goal_remains_active=True, external_retrieval_job_untouched='745198',
        final_restore_source_must_be_selected_again_after_actual_terminal=True)
    publish(claim_path, claim)
    result = subprocess.run(claim['command'], capture_output=True, text=True, timeout=20)
    value = dict(claim, returncode=result.returncode, stdout=result.stdout,
                 stderr=result.stderr, returned_unix=time.time(),
                 actual_source_terminal_not_yet_verified=True)
    publish(result_path, value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-base', required=True)
    parser.add_argument('--authorization-json')
    args = parser.parse_args()
    print(json.dumps(stop(args.target_base, args.authorization_json)), flush=True)
