"""Bounded read-only follow-up; never submits or changes a job or runtime."""
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[4]
CONTROL = REPOSITORY.parent.parent / 'outputs/qwen7b-warmup-20261019/balanced-control.py'
last_signature = None
actual_steps = []
for _ in range(240):
    result = subprocess.run(['python3', str(CONTROL), str(HERE / 'read-live-mae.py'), '--timeout', '40'],
                            cwd=REPOSITORY, capture_output=True, text=True)
    if result.returncode:
        print(json.dumps({'control_read_failed': True, 'code': result.returncode,
                          'stderr': result.stderr[-1000:], 'stdout_tail': result.stdout[-2000:]}), flush=True)
        break
    snapshot = json.loads(result.stdout)
    (HERE / 'actual-mae-latest.json').write_text(json.dumps(snapshot, sort_keys=True) + '\n')
    with (HERE / 'observations-mae.jsonl').open('a') as stream:
        stream.write(json.dumps(snapshot, sort_keys=True) + '\n')
    files = snapshot['files']
    ranks = [files.get('progress-rank-%d.json' % rank) for rank in (0, 1)]
    evaluations = [name for name in files if name.startswith('evaluation-epoch-')]
    errors = {name: value for name, value in files.items() if name.startswith('failure-rank-')}
    state = snapshot['accounting']['stdout'].split('|')[1] if snapshot['accounting']['stdout'] else 'UNKNOWN'
    summary = {'time': snapshot['time'], 'job_id': snapshot['job_id'], 'state': state,
               'ranks': ranks, 'evaluations': evaluations, 'errors': errors,
               'coverage': {name: value for name, value in files.items() if name.startswith('coverage-epoch-')},
               'latest_checkpoint': files.get('latest.json'), 'complete': files.get('complete.json')}
    signature = json.dumps({k: v for k, v in summary.items() if k != 'time'}, sort_keys=True)
    if signature != last_signature:
        print(json.dumps(summary), flush=True)
        last_signature = signature
    if ranks[0] and ranks[1] and ranks[0]['cursor'] == ranks[1]['cursor']:
        cursor = ranks[0]['cursor']
        if not actual_steps or cursor != actual_steps[-1]:
            actual_steps.append(cursor)
        if len(actual_steps) >= 2 and max(actual_steps) - min(actual_steps) >= 5:
            proof = {'job_id': '749569', 'actual_cursors': actual_steps, 'both_ranks_agree': True,
                     'two_real_samples': True, 'at_least_five_actual_updates_between_samples': True,
                     'final4096_read': False, 'time': snapshot['time']}
            (HERE / 'actual-first-five-update-proof.json').write_text(json.dumps(proof, sort_keys=True) + '\n')
    if errors or state in {'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL'}:
        print(json.dumps({'stop_reason': 'ACTUAL_OWNED_JOB_FAILURE'}), flush=True)
        break
    if 'evaluation-epoch-01.json' in files and all(row and row['epoch'] >= 2 and row['cursor'] >= 1326 for row in ranks):
        print(json.dumps({'stop_reason': 'REAL_EPOCH1_FINISHED_AND_EPOCH2_CONTINUES', 'CPU_checkpoint_acceptance_not_claimed': True}), flush=True)
        break
    if state == 'COMPLETED':
        print(json.dumps({'stop_reason': 'ORIGINAL_JOB_TERMINAL'}), flush=True)
        break
    time.sleep(60 if state == 'PENDING' else 10)
