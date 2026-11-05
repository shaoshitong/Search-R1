"""Independent file acceptance and exact calendar overlap; no model fitting."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re


def rows(path):
    return list(csv.DictReader(line for line in path.read_text(encoding='utf-8-sig').splitlines()
                               if not line.startswith('#')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    pool_path = args.root / 'audit/openchannelstats_public_pool.json'
    pool = json.loads(pool_path.read_text())
    channels, dates = [], {}
    for item in pool['channels']:
        handle = item['public_handle']
        root = Path(item['root'])
        raw = root / 'raw/openchannelstats' / handle
        audit = json.loads(Path(item['audit_path']).read_text())
        checked = []
        for name in ('README.md','SCHEMA.md','videos.csv','video_daily.csv','channel_daily.csv'):
            actual = hashlib.sha256((raw / name).read_bytes()).hexdigest()
            expected = re.search(r'\b[0-9a-fA-F]{64}\b', (raw / (name+'.sha256')).read_text()).group().lower()
            assert actual == expected
            checked.append({'file': name, 'bytes': (raw/name).stat().st_size, 'sha256': actual})
        video_rows = rows(raw/'video_daily.csv')
        channel_rows = rows(raw/'channel_daily.csv')
        dates[handle] = {row['date'] for row in channel_rows}
        channels.append({'public_handle': handle, 'raw_path': str(raw),
                         'source_files_verified': checked,
                         'video_daily_rows': len(video_rows),
                         'video_daily_unique_videos': len({row['video_id'] for row in video_rows}),
                         'video_daily_missing_impressions': sum(not row['impressions'] for row in video_rows),
                         'video_daily_zero_impressions': sum(row['impressions']=='0' for row in video_rows),
                         'pre_publish_rows': sum(float(row['days_since_publish'])<0 for row in video_rows),
                         'channel_daily_rows': len(channel_rows),
                         'channel_daily_date_range': [min(dates[handle]),max(dates[handle])] if dates[handle] else None,
                         'collected_raw_bytes': audit['downloaded_raw_bytes']})
    pairwise=[]
    for left,right in itertools.combinations(dates,2):
        common=dates[left].intersection(dates[right])
        pairwise.append({'left':left,'right':right,'common_observed_dates':len(common),
                         'common_date_range':[min(common),max(common)] if common else None})
    common_all=set.intersection(*dates.values())
    nonempty_dates=[value for value in dates.values() if value]
    common_nonempty=set.intersection(*nonempty_dates)
    report={'verified_utc':datetime.now(timezone.utc).isoformat(),'accepted':True,
            'verifier_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'public_channels':len(channels),'channels':channels,
            'total_video_daily_rows':sum(row['video_daily_rows'] for row in channels),
            'total_channel_daily_rows':sum(row['channel_daily_rows'] for row in channels),
            'total_raw_bytes':sum(row['collected_raw_bytes'] for row in channels),
            'pairwise_observed_calendar_overlap':pairwise,
            'all_channels_common_observed_dates':len(common_all),
            'all_channels_common_date_range':[min(common_all),max(common_all)] if common_all else None,
            'nonempty_public_channels':len(nonempty_dates),
            'nonempty_channels_common_observed_dates':len(common_nonempty),
            'nonempty_channels_common_date_range':[min(common_nonempty),max(common_nonempty)] if common_nonempty else None,
            'scope':'File and observed-date audit only. Public channels are not verified economic owners.'}
    target=args.root/'audit/openchannelstats_public_pool_acceptance.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['accepted','public_channels','total_video_daily_rows',
                     'total_channel_daily_rows','total_raw_bytes','all_channels_common_observed_dates',
                     'all_channels_common_date_range']}),flush=True)


if __name__ == '__main__':
    main()
