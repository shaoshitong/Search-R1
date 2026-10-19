"""Wait without GPU allocation, then activate the qualified native joint once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

# Actual owned-process peaks from completed actor171, not an idle-node rule.
OBSERVED_MIB = [38546, 36690, 37428, 41628, 52502, 32836, 39296, 44202]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--activation', required=True)
    parser.add_argument('--activation-sha256', required=True)
    args = parser.parse_args()
    base, activation = Path(args.base), Path(args.activation)
    node = socket.gethostname()
    if node not in ('h100-5', 'h100-8', 'h100-10'):
        raise ValueError('Allowed same-cluster H100 only; H10011 excluded')
    if hashlib.sha256(activation.read_bytes()).hexdigest() != args.activation_sha256:
        raise ValueError('Published activation bootstrap identity changed')
    from reproduction.prefix20_online.launch import once, read, publish
    forbidden = ('preparation/shared-activation.json', 'preparation/submit-intent.json',
                 'logs/process-ledger.json')
    if any((base / name).exists() for name in forbidden):
        raise ValueError('Existing activation/submission must be reconciled, never repeated')
    config = read(base / 'preparation/launch-config.json')
    ready = read(base / 'preparation/native-recovery-complete.json')
    if (not config.get('preparation_only') or config.get('training_authorized') is not False
            or config.get('expected_execution_node') != 'UNASSIGNED'
            or ready['status'] != 'LATEST_NATIVE_JOINT_PREPARED_AWAITING_TARGET'
            or ready['training_started'] or ready['optimizer_reset']):
        raise ValueError('Require an unlaunched qualified native joint')
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    once(base / 'preparation/capacity-wait-receipt.json', dict(node=node, pid=os.getpid(),
        start_ticks=int(stat[19]), observed_unix=time.time(), source_outer=ready['outer'],
        source_commit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        activation_sha256=args.activation_sha256, interval_seconds=5,
        observed_owned_process_MiB=OBSERVED_MIB, training_started=False))
    with (base / 'logs/capacity-wait-samples.jsonl').open('x', buffering=1) as log:
        while True:
            if any((base / name).exists() for name in forbidden):
                raise ValueError('Activation appeared while waiting; reconcile instead of duplicating')
            raw = subprocess.check_output(['nvidia-smi',
                '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True, timeout=15)
            cards = [[item.strip() for item in line.split(',')] for line in raw.splitlines()]
            cards.sort(key=lambda card: int(card[0]))
            if [int(card[0]) for card in cards] != list(range(8)):
                raise ValueError('Require eight actual physical GPUs')
            fits = all(int(card[3]) - int(card[2]) >= OBSERVED_MIB[int(card[0])]
                       for card in cards)
            sample = dict(observed_unix=time.time(), node=node, inventory=cards,
                          observed_footprint_fits=fits, training_started=False)
            log.write(json.dumps(sample) + '\n')
            publish(base / 'logs/capacity-wait-status.json', sample)
            if fits:
                once(base / 'preparation/capacity-wait-dispatch.json', sample)
                os.execv(sys.executable, [sys.executable, '-u', str(activation), '--base', str(base)])
            time.sleep(5)


if __name__ == '__main__':
    main()
