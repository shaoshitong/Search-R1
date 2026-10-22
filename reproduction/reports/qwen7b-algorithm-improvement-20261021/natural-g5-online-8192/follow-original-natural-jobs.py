"""Bounded read-only observer for the exact two already submitted jobs."""
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
CONTROL = REPOSITORY.parent.parent / 'outputs/qwen7b-warmup-20261019/balanced-control.py'
previous = None
for _ in range(40):
    result = subprocess.run(['python3', str(CONTROL), str(HERE / 'read-actual-natural-progress.py'),
                             '--timeout', '30'], cwd=REPOSITORY, capture_output=True, text=True)
    if result.returncode:
        print(json.dumps({'observer_error': result.stderr[-2000:], 'returncode': result.returncode}), flush=True)
        break
    snapshot = json.loads(result.stdout)
    (HERE / 'actual-natural-progress-latest.json').write_text(json.dumps(snapshot, sort_keys=True) + '\n')
    with (HERE / 'natural-actual-progress-observations.jsonl').open('a') as stream:
        stream.write(json.dumps(snapshot, sort_keys=True) + '\n')
    summary = []
    terminal = 0
    failed = False
    for row in snapshot['variants']:
        files = row['files']
        allocation = next((s.split('|') for s in row['sacct'] if s.split('|')[0] == row['job_id']), [])
        state = allocation[1] if allocation else 'UNKNOWN'
        terminal += state == 'COMPLETED'
        errors = {k: v for k, v in files.items() if k.startswith('failure-')}
        failed |= bool(errors) or state in {'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL', 'OUT_OF_MEMORY'}
        summary.append({'layer': row['layer'], 'job': row['job_id'], 'state': state,
                        'ranks': [files.get('progress-rank-%d.json' % rank) for rank in (0, 1)],
                        'evaluated_epochs': row['epochs_with_evaluations'],
                        'complete': files.get('complete.json'), 'errors': errors})
    signature = json.dumps(summary, sort_keys=True)
    if signature != previous:
        print(json.dumps({'epoch': snapshot['epoch'], 'variants': summary}), flush=True)
        previous = signature
    if failed or terminal == 2:
        print(json.dumps({'stop_reason': 'ACTUAL_FAILURE' if failed else 'BOTH_ORIGINAL_JOBS_COMPLETED'}), flush=True)
        break
    time.sleep(30)
