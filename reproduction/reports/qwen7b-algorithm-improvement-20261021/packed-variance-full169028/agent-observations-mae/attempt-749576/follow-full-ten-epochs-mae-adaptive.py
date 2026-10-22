"""Serial read-only follow-up through original replacement job terminal state."""
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[5]
CONTROL = REPOSITORY.parent.parent / 'outputs/qwen7b-warmup-20261019/metadata-control.py'

def fetch(script):
    result = subprocess.run(['python3', str(CONTROL), str(script), '--read-only', '--timeout', '25'],
                            cwd=REPOSITORY, capture_output=True, text=True)
    if result.returncode:
        return None, {'code': result.returncode, 'stderr': result.stderr[-1000:], 'stdout_tail': result.stdout[-2000:]}
    try:
        return json.loads(result.stdout), None
    except ValueError:
        return None, {'invalid_JSON': result.stdout[-2000:]}

def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True) + '\n')
    temporary.replace(path)

last_notice = None
comparison_last_try = {}
deadline = time.time() + 9*3600
while time.time() < deadline:
    snapshot, error = fetch(HERE / 'read-live-mae.py')
    if error:
        event = {'time': time.time(), 'control_read_failed': error, 'job_state_not_inferred': True}
        with (HERE / 'control-read-errors.jsonl').open('a') as stream:
            stream.write(json.dumps(event, sort_keys=True) + '\n')
        if last_notice != 'CONTROL_READ_FAILED':
            print(json.dumps(event), flush=True)
            last_notice = 'CONTROL_READ_FAILED'
        time.sleep(45)
        continue
    save(HERE / 'actual-mae-latest.json', snapshot)
    files = snapshot['files']
    ranks = [files.get('progress-rank-%d.json' % rank) for rank in (0, 1)]
    accounting = snapshot['accounting']['stdout']
    state = accounting.split('|')[1] if accounting else 'UNKNOWN'
    epochs = [int(name.rsplit('-', 1)[1].split('.')[0]) for name in files if name.startswith('evaluation-epoch-')]
    errors = {name: value for name, value in files.items() if name.startswith('failure-rank-')}
    summary = {'time': snapshot['time'], 'job_id': '749576', 'state': state,
               'ranks': ranks, 'evaluated_epochs': sorted(epochs), 'errors': errors,
               'latest_checkpoint': files.get('latest.json'), 'complete': files.get('complete.json')}
    with (HERE / 'compact-full-observations.jsonl').open('a') as stream:
        stream.write(json.dumps(summary, sort_keys=True) + '\n')
    notice = (state, tuple(epochs), bool(errors))
    if notice != last_notice:
        print(json.dumps(summary), flush=True)
        last_notice = notice
    for epoch in sorted(epochs):
        score_path = HERE / ('actual-score-spread-baselines-epoch-%02d.json' % epoch)
        if not score_path.exists():
            source = (HERE / 'read-score-spread-and-baselines-mae.py').read_text()
            script = HERE / ('read-score-spread-baselines-epoch-%02d.py' % epoch)
            script.write_text(source.replace('EPOCH = 0', 'EPOCH = %d' % epoch))
            score, failure = fetch(script)
            if not failure:
                save(score_path, score)
                print(json.dumps({'actual_score_spread_saved': str(score_path), 'epoch': epoch}), flush=True)
        comparison_path = HERE / ('actual-same-epoch-three-arm-%02d.json' % epoch)
        existing = json.loads(comparison_path.read_text()) if comparison_path.exists() and comparison_path.stat().st_size > 10 else None
        if (not existing or existing.get('status') != 'SAME_EPOCH_FIXED_INPUT_THREE_ARM_DIAGNOSTICS_COMPLETE') and time.time()-comparison_last_try.get(epoch, 0) >= 300:
            comparison_last_try[epoch] = time.time()
            source = (HERE / 'read-same-epoch-three-arm-comparison.py').read_text()
            script = HERE / ('read-same-epoch-three-arm-%02d.py' % epoch)
            script.write_text(source.replace('EPOCH = 0', 'EPOCH = %d' % epoch))
            comparison, failure = fetch(script)
            if not failure:
                save(comparison_path, comparison)
                print(json.dumps({'actual_same_epoch_comparison_saved': str(comparison_path), 'status': comparison['status'], 'epoch': epoch}), flush=True)
    if errors or state in {'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL'}:
        print(json.dumps({'stop_reason': 'ACTUAL_OWNED_JOB_FAILURE', 'job_id': '749576'}), flush=True)
        break
    if state == 'COMPLETED':
        print(json.dumps({'stop_reason': 'ORIGINAL_REPLACEMENT_TERMINAL', 'job_id': '749576',
                          'CPU_checkpoint_acceptance_still_required': True}), flush=True)
        break
    time.sleep(60)
