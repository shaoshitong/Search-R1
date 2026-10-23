"""Publish the existing five-field external receipt from actual service READY.

CPU JSON only: no HTTP request, Slurm action, asset staging or model load.
The normal external_binding consumer retains its stage64/asset contract.
"""
import argparse
import json
from pathlib import Path

from reproduction.easy2560_joint4.control import publish, sha


def publish_receipt(service_base, output):
    ready_path = Path(service_base) / 'logs/service-ready.json'
    ready = json.loads(ready_path.read_text())
    receipt = dict(service_ready_path=str(ready_path), service_ready_sha256=sha(ready_path),
        source_commit=ready['source_commit'], service_job_id=ready['service_job_id'],
        endpoint=ready['endpoint'])
    output = Path(output)
    if output.exists():
        if json.loads(output.read_text()) != receipt:
            raise ValueError('Existing external receipt differs; retain it and use the intended new target')
    else:
        publish(output, receipt)
    return dict(receipt_path=str(output), receipt_sha256=sha(output), **receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service-base', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(publish_receipt(args.service_base, args.output), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
