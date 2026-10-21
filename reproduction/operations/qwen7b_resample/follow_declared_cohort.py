"""Read-only follow of an explicitly pinned active cohort and retained history.

The registry is an operator artifact, not a claim that its models started.
The existing follower still checks actual unit, allocation and stage identity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from . import follow_guarded_generation as follower


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--registry-sha256', required=True)
    args, remainder = parser.parse_known_args()
    raw = Path(args.registry).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.registry_sha256
    plan = json.loads(raw)
    assert plan['run_id'] == follower.control.RUN_ID
    rows = plan['registry']
    assert rows and len(rows) == len({row['unit'] for row in rows}) == len({row['key'] for row in rows})
    for row in rows:
        assert row['node'] in ('h100-5', 'h100-10')
        assert row['role'] in ('owner', 'model', 'upstream', 'downstream')
        assert row['unit'].startswith('sst-q7b-') and row['unit'].endswith(follower.control.RUN_ID[:12] + '.service')
        assert re.fullmatch('[a-z0-9-]+', row['directory'])
        assert re.fullmatch('[0-9a-f]{64}', row['source_bundle_sha256'])
    previous, argv = follower.registry, sys.argv
    try:
        follower.registry = lambda *_: rows
        sys.argv = [sys.argv[0], *(remainder or ['plan'])]
        return follower.main()
    finally:
        follower.registry, sys.argv = previous, argv


if __name__ == '__main__':
    raise SystemExit(main())
