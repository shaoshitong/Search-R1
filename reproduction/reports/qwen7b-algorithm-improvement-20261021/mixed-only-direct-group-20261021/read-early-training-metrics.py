"""Small read-only real-update/throughput and initial Mixed-test summary."""
import json
from pathlib import Path
import time
b = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-v1')
def read(p): return json.loads(p.read_bytes())
assert read(b / 'submission-result.json')['job_id'] == '746083'
fields = ('cursor', 'time', 'seconds', 'binary_Mixed_BCE', 'loss', 'questions',
          'real_native_trajectories', 'views', 'mixed_views', 'nonmixed_views',
          'native_steps', 'head_steps', 'confusion', 'native_gradient_L2',
          'group_head_gradient_L2', 'trajectory_loss_computed', 'trajectory_accuracy_computed')
rows = []
curve = b / 'curve.jsonl'
if curve.exists():
    with curve.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows.append({key: row.get(key) for key in fields})
out = dict(epoch=time.time(), job_id='746083', accepted_rows=len(rows),
           first_updates=rows[:12], last_updates=rows[-12:], metrics_are_actual_not_estimated=True)
out['training_curve_blocks'] = []
for start in range(0, len(rows), 32):
    block = rows[start:start + 32]
    confusion = {key: sum(v['confusion'][key] for v in block) for key in ('tp', 'tn', 'fp', 'fn')}
    positive = confusion['tp'] + confusion['fn']
    negative = confusion['tn'] + confusion['fp']
    out['training_curve_blocks'].append(dict(
        first_cursor=block[0]['cursor'], last_cursor=block[-1]['cursor'], updates=len(block),
        mean_binary_Mixed_BCE=sum(v['binary_Mixed_BCE'] for v in block) / len(block),
        mixed_balanced_accuracy=.5 * (confusion['tp'] / positive + confusion['tn'] / negative),
        mixed_recall=confusion['tp'] / positive,
        nonmixed_recall=confusion['tn'] / negative,
        confusion=confusion, training_views_only=True, independent_test_result=False))
window = rows[-min(32, len(rows)):]
if window:
    seconds = sum(v['seconds'] for v in window)
    out['measured_compute_window'] = dict(
        first_cursor=window[0]['cursor'], last_cursor=window[-1]['cursor'], updates=len(window),
        mean_step_seconds=seconds / len(window),
        questions_per_minute=60 * sum(v['questions'] for v in window) / seconds,
        genuine_native_trajectory_rows_per_minute=60 * sum(v['real_native_trajectories'] for v in window) / seconds,
        G5_views_per_minute=60 * sum(v['views'] for v in window) / seconds,
        excludes_checkpoint_and_evaluation_overhead=True, startup_window_not_yet_steady=True)
    if len(window) >= 2:
        elapsed = window[-1]['time'] - window[0]['time']
        out['measured_accepted_wall_window'] = dict(
            first_cursor=window[0]['cursor'], last_cursor=window[-1]['cursor'], elapsed_seconds=elapsed,
            questions_per_minute=60 * sum(v['questions'] for v in window[1:]) / elapsed,
            includes_startup_replay_checkpoint_audit_overhead=True, steady_state_claim=False)
for name in ('progress.json', 'latest.json', 'first-update-replay.json', 'first-cp-independent-audit.json', 'durable-next-update-replay.json', 'failure.json', 'complete.json'):
    p = b / name
    out[name] = read(p) if p.exists() else None
eval_path = b / 'evaluations/epoch-00/complete.json'
if eval_path.exists():
    result = read(eval_path)
    out['initial_evaluation'] = dict(status=result['status'], cursor=result['cursor'],
        metrics=result['metrics'], models_two_Adam_RNG_native_buffers_unchanged=result['models_two_Adam_RNG_native_buffers_unchanged'],
        trajectory_probability_or_accuracy_produced=result['trajectory_probability_or_accuracy_produced'])
print(json.dumps(out, sort_keys=True))
