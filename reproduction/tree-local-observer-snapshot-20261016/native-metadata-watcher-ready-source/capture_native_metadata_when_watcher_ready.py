#!/usr/bin/env python3
"""Gate the published JSON-only native readback on actual watcher readiness.

This changes only the local observation entry. It never changes training,
starts a watcher, reads tensors, or retries a remote command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

W = Path(__file__).resolve().parents[2]
B = W / 'outputs/tree-nested-branch-handoff-20261013'
SOURCE = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
ORIGINAL = W / 'work/tree-speed-repair-20261008/nested-local-observer-snapshot-source/reproduction/tree-local-observer-snapshot-20261016/05_tree_generic_native_history_readback_cmd_778.command.txt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', required=True, type=int, choices=range(6, 201))
    parser.add_argument('--poll', required=True, type=int)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError('Readiness guards cannot be disabled with -O')
    if args.poll <= 0:
        parser.error('An actual positive poll ID is required')
    held = {}
    for kind in ('POLL', 'RESOURCE'):
        path = B / f'NATIVE5_CONTINUE_{kind}_{args.poll}.json'
        raw = path.read_bytes()
        if not 0 < len(raw) < 4000000:
            raise ValueError('Small original observation budget differs')
        held[kind] = (path, raw, json.loads(raw))
    poll, resource = held['POLL'][2], held['RESOURCE'][2]
    assert poll['source_commit'] == SOURCE
    assert poll['progress']['actual_completed_outer_from_attempt_results'] == args.outer
    assert poll['progress']['errors'] == [] and poll['run_result'] is None
    unit = resource['training_systemd']
    assert 'MainPID=503255\n' in unit and 'InvocationID=a313bda919834d24b4c5d491698d39ce\n' in unit
    assert 'ActiveState=active\n' in unit
    assert args.outer in resource['state']['indexed_completed_outers'], 'Wait for the original watcher receipt; native writer completion alone is insufficient'
    assert resource['state']['audit_error_count'] == 0
    receipt = f'source-{SOURCE}-outer-{args.outer:06d}.receipt.json'
    assert receipt in resource['archive_receipts'], 'Original source-bound receipt must actually be listed'
    mirrors = [W / 'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch', Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
    for root in (B, *mirrors):
        assert not (root / f'NATIVE_HISTORY_{args.outer}_METADATA_READBACK.json').exists(), 'Retain existing success; do not execute another native readback'
    original = ORIGINAL.read_bytes()
    assert len(original) == 3587 and hashlib.sha256(original).hexdigest() == '4a0cf7b0b5cba4b34670642c7ff86774b11ff9c62ae20ce109909c1675f21fa6'
    for path, raw, unused in held.values():
        assert path.read_bytes() == raw, 'Original local observation changed before invocation'
    command = original.decode().replace('__OUTER__', str(args.outer)).replace('__POLL__', str(args.poll))
    print(json.dumps(dict(status='ACTUAL_WATCHER_READY_ORIGINAL_READBACK_ALLOWED', outer=args.outer, poll=args.poll, original_receipt_name=receipt, original_source_sha256=hashlib.sha256(original).hexdigest(), automatic_retry=False)), flush=True)
    raise SystemExit(subprocess.run(['bash'], input=command, text=True, cwd=W, check=False).returncode)


if __name__ == '__main__':
    main()
