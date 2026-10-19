"""Bind one prepared continuation to a fresh allowed H100, then launch once."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    base = Path(args.base)
    node = socket.gethostname()
    if node not in ('h100-5', 'h100-8', 'h100-10'):
        raise ValueError('Only authorized same-cluster targets; H10011 excluded')
    from reproduction.prefix20_online.launch import read, once, publish, sha
    from reproduction.prefix_selection200.launch import validate_config
    if ((base / 'preparation/submit-intent.json').exists()
            or (base / 'logs/process-ledger.json').exists()
            or (base / 'preparation/shared-activation.json').exists()):
        raise ValueError('Reconcile an existing activation/submission; never replay it')
    ready = read(base / 'preparation/native-recovery-complete.json')
    config_path = base / 'preparation/launch-config.json'
    config = read(config_path)
    if (ready['status'] != 'LATEST_NATIVE_JOINT_PREPARED_AWAITING_TARGET'
            or ready['training_started'] or ready['optimizer_reset']
            or not config.get('preparation_only')
            or config.get('training_authorized') is not False
            or config.get('expected_execution_node') != 'UNASSIGNED'):
        raise ValueError('Require an unlaunched complete native recovery')
    raw = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    cards = [[item.strip() for item in line.split(',')] for line in raw.splitlines()]
    cards.sort(key=lambda row: int(row[0]))
    if [int(row[0]) for row in cards] != list(range(8)):
        raise ValueError('Require the actual eight physical GPU indices')
    before = sha(config_path)
    uuids = [row[1] for row in cards]
    config.update(expected_execution_node=node, physical_gpu_uuids=uuids,
                  auxiliary_gpu_uuids=uuids, preparation_only=False,
                  training_authorized=True, experiment_assignment_pending=False)
    validate_config(config, base)
    once(base / 'preparation/shared-activation.json', dict(node=node,
        before_config_sha256=before, inventory=cards, observed_unix=time.time(),
        source_outer=ready['outer'], actor_actual_adam=ready['actor'],
        probe_actual_adam=ready['probe'], prefix_tokens=ready['prefix_tokens'],
        shared_existing_foreign_jobs_authorized=True, foreign_processes_stopped=False,
        optimizer_reset=False))
    publish(config_path, config)
    os.execv(sys.executable, [sys.executable, '-u', '-m',
        'reproduction.operations.b_replica_retained18.shared_native_start', '--base', str(base)])


if __name__ == '__main__':
    main()
