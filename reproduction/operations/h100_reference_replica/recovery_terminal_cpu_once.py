"""Seal a genuinely failed native execution and qualify its actual latest joint.

Run remotely, CPU only, in preparation rather than the frozen trainer repo.
The caller reconciles its unique start intent; this worker never submits,
signals, retries training, or copies original logs/model payloads to the Mac.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def inspect(base, expected_outer, expected_run_id):
    base = Path(base)
    sys.path.insert(0, str(base / 'repo'))
    from reproduction.prefix20_online.launch import read, sha, once
    from reproduction.prefix_only_correctness_ab.launch import birth_states
    from reproduction.a800.owned_processes import same

    prep = base / 'preparation'
    result_path = prep / ('recovery-%04d-terminal-and-cpu-result.json' % expected_outer)
    if result_path.exists():
        return dict(status='EXISTING_RESULT_NO_REPLAY', result=str(result_path), sha256=sha(result_path))
    joint = read(base / 'latest-joint.json')
    config = read(prep / 'launch-config.json')
    terminal_path = prep / 'native-actual-waitpid-exit.json'
    terminal = read(terminal_path)
    submit = read(prep / 'submit-receipt.json')
    if (config['run_id'] != expected_run_id or joint['run_id'] != expected_run_id
            or joint['outer'] != expected_outer
            or terminal['status'] != 'ACTUAL_OWN_CHILD_WAITPID_EXIT_CAPTURED'
            or terminal['actual_returncode'] == 0
            or terminal['native_leader'] != submit['leader']
            or terminal['native_receipt']['config_sha256'] != sha(prep / 'launch-config.json')):
        raise ValueError('Require this failed native birth and its actual latest complete joint')
    once(prep / ('recovery-%04d-cpu-worker-intent.json' % expected_outer),
         dict(run_id=expected_run_id, outer=expected_outer, actual_native_returncode=terminal['actual_returncode']))
    ledger = read(base / 'logs/process-ledger.json')
    records = [v for job in ledger['jobs'].values() for v in job.get('ledger', {}).values()]
    owned_pids = {row['pid'] for row in records}
    checks = []
    for _ in range(2):
        remaining = birth_states(records)
        leader_live = same(submit['leader'])
        output = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
            '--format=csv,noheader,nounits'], text=True)
        contexts = [line for line in output.splitlines()
                    if line.split(',', 1)[0].strip().isdigit()
                    and int(line.split(',', 1)[0].strip()) in owned_pids]
        checks.append(dict(checked_unix=time.time(), native_same_birth=leader_live,
                           remaining_owned_births=remaining, owned_gpu_pid_rows=contexts))
        if remaining or leader_live:
            raise ValueError('Failed source still has live owned writers; never restore alongside them')
        time.sleep(.25)
    if not read(base / 'logs/cleanup-status.json').get('cleaned'):
        raise ValueError('Actual source cleanup has not completed')
    del ledger, records
    gc.collect()
    seal_path = prep / ('recovery-%04d-terminal-original-full-sha.json' % expected_outer)
    if not seal_path.exists():
        paths = set((base / 'logs').rglob('*.log'))
        paths.update((base / 'outers' / ('outer-%06d' % (expected_outer + 1))).rglob('*.log'))
        paths.update((terminal_path, prep / 'launch-config.json', prep / 'submit-receipt.json',
                      base / 'logs/process-ledger.json', base / 'logs/cleanup-status.json'))
        files = []
        for path in sorted(paths):
            before = path.stat()
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 ** 2), b''):
                    digest.update(chunk)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError('Original failed-execution inode changed while sealing: ' + str(path))
            files.append(dict(path=str(path), inode=before.st_ino, bytes=before.st_size,
                              mtime_ns=before.st_mtime_ns, sha256=digest.hexdigest()))
        once(seal_path, dict(status='FAILED_NATIVE_ORIGINAL_INODES_FULL_SHA_SEALED',
                            run_id=expected_run_id, outer=expected_outer,
                            actual_native_returncode=terminal['actual_returncode'],
                            native_terminal_sha256=sha(terminal_path), owned_release_checks=checks,
                            all_owned_gpu_rows_absent=all(not x['owned_gpu_pid_rows'] for x in checks),
                            files=files, models_or_logs_transferred_to_mac=False))
    from reproduction.prefix_selection200.ab_checkpoint_once import inspect as inspect_joint
    qualified = inspect_joint(base, expected_outer)
    proof_path = prep / ('ab-joint-%04d-independent-full-sha-cpu.json' % expected_outer)
    proof = read(proof_path)
    if proof['status'] != 'AB_JOINT_FULL_SHA_CPU_PASS' or proof['outer'] != expected_outer:
        raise ValueError('Latest complete state did not qualify; never fall back to an older point')
    result = dict(status='FAILED_NATIVE_SEALED_LATEST_JOINT_CPU_QUALIFIED',
                  run_id=expected_run_id, completed_outer=expected_outer,
                  terminal_seal=str(seal_path), terminal_seal_sha256=sha(seal_path),
                  checkpoint_proof=str(proof_path), checkpoint_proof_sha256=sha(proof_path),
                  actor_actual_adam_steps_per_rank=proof['actor_actual_adam_steps_per_rank'],
                  probe_actual_adam_steps_per_rank=proof['probe_actual_adam_steps_per_rank'],
                  gpu_forward_performed=False, training_launched=False)
    once(result_path, result)
    return dict(result, result=str(result_path), result_sha256=sha(result_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--outer', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This source readback must be CPU only')
    print(json.dumps(inspect(args.base, args.outer, args.run_id)), flush=True)
