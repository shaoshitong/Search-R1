"""Bounded extension to the seven channels in the inspected public manifest."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.root / 'raw/openchannelstats/theartharrison/channels-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    handles = [entry['handle'].lstrip('@') for entry in manifest['channels']]
    assert len(handles) <= 10 and all(re.fullmatch(r'[A-Za-z0-9._-]+', h) for h in handles)
    collector = Path(__file__).with_name('collect_openchannelstats_pilot.py')
    audits = []
    for handle in handles:
        if handle == 'theartharrison':
            root = args.root
        else:
            root = args.root / 'public_pool' / handle
        audit_path = root / 'audit/openchannelstats_pilot.json'
        if audit_path.exists():
            exit_code = 0
        else:
            result = subprocess.run([sys.executable, str(collector), '--root', str(root),
                                     '--handle', handle])
            exit_code = result.returncode
        item = {'public_handle': handle, 'root': str(root), 'exit_code': exit_code}
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            item.update(audit_path=str(audit_path), csvs=audit['csvs'],
                        downloaded_raw_bytes=audit['downloaded_raw_bytes'])
        audits.append(item)
        print(json.dumps({'channel_completed': handle, 'exit_code': exit_code,
                          'csv_count': len(item.get('csvs', {}))}), flush=True)
    ranges = {item['public_handle']: item['csvs']['channel_daily.csv']['date_like_field_ranges']['date']
              for item in audits if item.get('csvs', {}).get('channel_daily.csv', {}).get('date_like_field_ranges', {}).get('date')}
    overlaps = []
    for (left, a), (right, b) in itertools.combinations(ranges.items(), 2):
        start, end = max(a[0], b[0]), min(a[1], b[1])
        if start <= end:
            overlaps.append({'left': left, 'right': right, 'common_calendar_range': [start, end]})
    report = {'completed_utc': datetime.now(timezone.utc).isoformat(),
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'collector_sha256': hashlib.sha256(collector.read_bytes()).hexdigest(),
              'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'public_channels_in_manifest': len(handles), 'channels': audits,
              'overlapping_channel_calendar_ranges': overlaps,
              'scope': 'Opt-in channels; public identity is channel identity, not economic owner.'}
    target = args.root / 'audit/openchannelstats_public_pool.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'pool_complete': True, 'audit': str(target), 'channels': len(audits),
                      'overlapping_pairs': len(overlaps)}), flush=True)


if __name__ == '__main__':
    main()
