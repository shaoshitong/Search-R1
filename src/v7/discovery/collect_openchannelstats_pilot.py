"""Download one explicitly public, metadata-only Open Channel Stats record."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, target):
    partial = target.with_suffix(target.suffix + '.part')
    process = subprocess.run(
        ['curl', '-L', '--max-time', '35', '--max-filesize', '5000000',
         '-sS', '-o', str(partial), '-w', '%{http_code}', url],
        capture_output=True, text=True)
    receipt = {'url': url, 'http_status': process.stdout.strip(),
               'curl_exit': process.returncode, 'stderr': process.stderr.strip()}
    if partial.exists():
        receipt['received_bytes'] = partial.stat().st_size
    if process.returncode == 0 and process.stdout.strip() == '200':
        partial.replace(target)
        receipt.update(path=str(target), sha256=digest(target))
    return receipt


def describe_csv(path):
    text = path.read_text(encoding='utf-8-sig')
    # Public CSVs carry source/rights notes before the header.
    reader = csv.DictReader(io.StringIO('\n'.join(
        line for line in text.splitlines() if not line.startswith('#'))))
    rows = list(reader)
    fields = reader.fieldnames or []
    summary = {'rows': len(rows), 'columns': fields,
               'missing_counts': {key: sum(row.get(key) in ('', None) for row in rows)
                                  for key in fields}}
    summary['identifier_counts'] = {
        key: len({row[key] for row in rows if row.get(key)})
        for key in ('video_id', 'channel_id') if key in fields}
    summary['date_like_field_ranges'] = {}
    for key in fields:
        if any(word in key.lower() for word in ('date', 'published', 'snapshot')):
            values = [row[key] for row in rows if row.get(key)]
            if values:
                summary['date_like_field_ranges'][key] = [min(values), max(values)]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--handle', default='theartharrison')
    args = parser.parse_args()
    raw = args.root / 'raw' / 'openchannelstats' / args.handle
    audit = args.root / 'audit'
    raw.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    report = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'One public opt-in channel, no authentication or media.',
              'source_code_sha256': digest(Path(__file__)), 'requests': [], 'csvs': {}}
    manifest = raw / 'channels-manifest.json'
    receipt = fetch('https://openchannelstats.com/channels-manifest.json', manifest)
    report['requests'].append(receipt)
    if 'path' not in receipt:
        raise RuntimeError('Public manifest unavailable; no channel probing attempted.')
    manifest_data = json.loads(manifest.read_text())
    # Require the already advertised public handle to be in the live manifest.
    if args.handle not in json.dumps(manifest_data):
        raise RuntimeError('Requested handle is not in the public manifest.')
    report['manifest_root_type'] = type(manifest_data).__name__
    if isinstance(manifest_data, dict):
        report['manifest_keys'] = list(manifest_data)
    report['public_handle_confirmed'] = True
    base = 'https://openchannelstats.com/@' + args.handle + '/data/'
    for name in ('README.md', 'SCHEMA.md', 'videos.csv', 'video_daily.csv', 'channel_daily.csv'):
        target = raw / name
        item = fetch(base + name, target)
        report['requests'].append(item)
        if 'path' not in item:
            continue
        checksum_file = raw / (name + '.sha256')
        sidecar = fetch(base + name + '.sha256', checksum_file)
        report['requests'].append(sidecar)
        if 'path' in sidecar:
            match = re.search(r'\b[0-9a-fA-F]{64}\b', checksum_file.read_text())
            item['source_sha256'] = match.group().lower() if match else None
            item['source_sha256_matches'] = item['source_sha256'] == item['sha256']
            if not item['source_sha256_matches']:
                raise RuntimeError('Source SHA mismatch for ' + name)
        if name.endswith('.csv'):
            report['csvs'][name] = describe_csv(target)
        print(json.dumps({'file_complete': name, 'bytes': item['received_bytes'],
                          'source_sha256_matches': item.get('source_sha256_matches')}), flush=True)
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    report['downloaded_raw_bytes'] = sum(path.stat().st_size for path in raw.iterdir()
                                        if path.is_file() and not path.name.endswith('.part'))
    report['completed_csv_count'] = len(report['csvs'])
    result = audit / 'openchannelstats_pilot.json'
    result.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'audit': str(result), 'completed_csv_count': report['completed_csv_count'],
                      'raw_bytes': report['downloaded_raw_bytes']}), flush=True)


if __name__ == '__main__':
    main()
