"""Read-only observation of the independent Easy7B future callback companion."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import time


HK = Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017')
ROOT = HK/'post-callback-boundary-control/job-745371/future-50-200'
RUN = '232f20e112ac4a978c5aba52a0367195'
JOB = '745371'
TRAINING = 'e082a23b5a65b302e2016aa0e5ab43b10b543d4f'
BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021'
STEPS = (50, 100, 150, 200)


def need(condition, message):
    if not condition: raise ValueError('POST_CALLBACK_FUTURE_OBSERVE_' + message)


def regular(path, maximum=16*1024**2):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink() and path.is_file() and
         path.stat().st_size <= maximum, 'SMALL_CANONICAL_FILE_REQUIRED')
    return path


def read_json(path):
    return json.loads(regular(path).read_bytes())


def records(root, names):
    return {name: read_json(root/name) for name in names if (root/name).exists()}


def load_runner(launch):
    name = 'archive_post_callback_future_hk.py'
    proof = launch.get('public_code', {}).get(name, {})
    commit = launch.get('launcher_commit', '')
    need(isinstance(commit, str) and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit),
         'PUBLISHED_LAUNCHER_COMMIT_REQUIRED')
    tools = HK/'post-callback-future-launch-tools'/commit
    need(launch.get('tools') == str(tools) and proof.get('path') == str(tools/name), 'PUBLISHED_RUNNER_PATH_DIFFERS')
    path = regular(tools/name, 128*1024)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    need(proof.get('sha256') == actual and proof.get('bytes') == path.stat().st_size and
         proof.get('path') == str(path), 'PUBLISHED_RUNNER_SOURCE_DIFFERS')
    spec = importlib.util.spec_from_file_location('_future_companion_read_only', str(path))
    loaded = importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded)
    return loaded


def observe(verify_files=False):
    need(socket.gethostname() == 'xielab3' and os.getuid() == 1002, 'HKUST3_OWNER_REQUIRED')
    control = records(ROOT, ('launch-claim.json', 'launch-receipt.json', 'launch-failure.json',
        'claim.json', 'progress.json', 'result.json', 'failure.json', 'completed-five-read-only.json'))
    for name in ('claim.json', 'launch-claim.json', 'launch-receipt.json'):
        value = control.get(name)
        if value is not None:
            need(value.get('base') == BASE and value.get('source_job') == JOB and value.get('run_id') == RUN and
                 value.get('training_commit') == TRAINING, 'CONTROL_IDENTITY_DIFFERS')
    launch = control.get('launch-receipt.json', {})
    runner = load_runner(launch) if launch else None
    birth = control.get('claim.json', {}).get('birth') or launch.get('birth')
    actual = runner.process_observation(birth) if runner is not None and birth else dict(state='UNKNOWN')
    dependency_module = helper = None
    if verify_files:
        need(runner is not None and 'claim.json' in control, 'PUBLISHED_CLAIM_REQUIRED_FOR_READBACK')
        helper = runner.frozen_helper(); helper.bootstrap(); helper.load_module()
        claim = control['claim.json']
        data = dict(commit=claim['dependencies_commit'], sha256=claim['dependencies_sha256'],
            hk_tool=HK/'post-callback-dependency-tools'/claim['dependencies_commit']/'post_callback_dependencies.py')
        dependency_module = runner.load_dependencies(helper, data)
    milestones = {}
    for step in (5,) + STEPS:
        own = ROOT/('qualification-global_step_5' if step == 5 else 'global_step_' + str(step))
        values = records(own, ('claim.json', 'boundary-result.json', 'dependencies-result.json',
            'result.json', 'boundary-transfer.json', 'dependencies-transfer.json'))
        core = HK/'checkpoints'/RUN/('global_step_' + str(step))
        accepted = records(core, ('post-callback-boundary/boundary-archive-receipt.json',
            'post-callback-dependencies/dependency-archive-receipt.json'))
        boundary = accepted.get('post-callback-boundary/boundary-archive-receipt.json')
        dependencies = accepted.get('post-callback-dependencies/dependency-archive-receipt.json')
        for receipt in (boundary, dependencies):
            if receipt is not None:
                need(all(receipt.get('binding', {}).get(key) == value for key, value in dict(
                    base=BASE, run_id=RUN, source_commit=TRAINING, step=step, core_archive=str(core)).items()),
                    'MILESTONE_ARCHIVE_IDENTITY_DIFFERS')
        checks = None
        if verify_files and dependencies is not None:
            helper.STEP = step; helper.CORE = core
            checks = runner.inspect_closure(helper, dependency_module)
        boundary_pass = (boundary is not None and boundary.get('status') ==
            'EASY2560_POST_CALLBACK_BOUNDARY_ARCHIVE_SHA_FSYNC_PASS' and boundary.get('file_count') == 8 and
            boundary.get('fresh_destination_sha_verified') is True and boundary.get('destination_fsynced') is True)
        dependencies_pass = (dependencies is not None and dependencies.get('status') ==
            'EASY2560_POST_CALLBACK_DEPENDENCIES_SHA_FSYNC_PASS' and dependencies.get('all_manifest_dependencies_covered') is True)
        milestones[str(step)] = dict(control=values, boundary_accepted=boundary_pass,
            dependencies_accepted=dependencies_pass, receipts=accepted, full_dependency_readback=checks,
            boundary_writer_replayed=False if step == 5 else None)
    result = control.get('result.json', {})
    done = (result.get('status') == 'EASY7B_FUTURE_POST_CALLBACK_COMPANION_COMPLETE' and result.get('completed_steps') == list(STEPS) and
        all(milestones[str(step)]['boundary_accepted'] and milestones[str(step)]['dependencies_accepted'] for step in (5,) + STEPS))
    failed = 'failure.json' in control or 'launch-failure.json' in control
    state = 'COMPLETE' if done else 'FAILED' if failed else 'RUNNING' if actual['state'] == 'LIVE' else (
        'NOT_LAUNCHED' if not control else 'TERMINAL_OR_UNKNOWN')
    log = ROOT/'runner.stdout.log'; tail = ''; size = 0
    if log.exists():
        regular(log, 32*1024**2); size = log.stat().st_size
        with log.open('rb') as stream: stream.seek(max(0, size-2400)); tail = stream.read().decode(errors='replace')
    return dict(status='POST_CALLBACK_FUTURE_READ_ONLY_OBSERVATION', state=state, root=str(ROOT),
        records=control, actual_process=actual, milestones=milestones, requested_steps=list(STEPS),
        full_dependency_sha_requested=verify_files, actual_cuda_restore_accepted=False,
        log_bytes=size, log_tail=tail, training_modified=False, controllers_started=0,
        jobs_submitted=0, deletion_performed=False, checked_unix=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-files', action='store_true', help='Fresh transferred dependency SHA and accepted-core mapping proofs; no encoder payload rehash')
    args = parser.parse_args()
    print(json.dumps(observe(args.verify_files), sort_keys=True), flush=True)


if __name__ == '__main__': main()
