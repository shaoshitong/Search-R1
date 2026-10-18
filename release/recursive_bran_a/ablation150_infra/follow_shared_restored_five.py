"""Read-only follow-through of restored111 -> complete112..116 -> real117."""
import json
import math
from pathlib import Path
import time


def main():
    base = Path('/dataset/shaoshitong/recursive-ablation150-20261014')
    out = base / 'shared-h1005-20261018'
    expected = '9418facb200649a9b2f9001bbfe7ec39'
    seen = {}; bound = None; restored = None; started = time.time()
    while time.time() - started < 14400:
        entry = json.loads((base / 'budget-5/receipts/current-entry.json').read_text())
        if entry.get('execution_backend') != 'h100_shared':
            time.sleep(10)
            continue
        assert entry['run_id'] == expected
        if bound is None:
            bound = {k: entry[k] for k in ('pid', 'uid', 'tick')}
        assert all(entry[k] == v for k, v in bound.items())
        proc = Path('/proc') / str(bound['pid'])
        output = Path(entry['output'])
        if not proc.exists() or (output / 'entry-exit.json').exists():
            raise RuntimeError('Actual resumed entry ended; inspect real exit before another recovery')
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        assert int(fields[19]) == bound['tick'] and proc.stat().st_uid == bound['uid'] and fields[0] != 'Z'
        path = output / 'native-resumed.json'
        if path.exists():
            restored = json.loads(path.read_text())
            assert restored['completed_outer'] == 111 and len(restored['workers']) == 8
            assert all(x['actual_state_equal'] and x['actual_total_optimizer_calls'] == 2220
                       for x in restored['workers'])
        journal = output / 'completed-outers.jsonl'
        if journal.exists():
            lines = journal.read_text().splitlines()
            for line in lines:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Concurrent trailing append is not a model error.
                n = value['completed_outer']
                if 112 <= n <= 116:
                    assert restored and value['run_id'] == expected
                    assert all(math.isfinite(float(x)) for x in value['metrics'].values())
                    assert value['metrics']['actor/scheduler_step_after_outer'] == n
                    seen[n] = value
        state = dict(time=time.time(), entry=bound, restored111=bool(restored),
            completed_restored_outers=sorted(seen), goal_complete=False,
            scope='Native producer scalar and restored-state receipts; no extra evaluation')
        (out / 'follow-progress.json').write_text(json.dumps(state, sort_keys=True))
        if set(seen) == set(range(112, 117)):
            study = json.loads((base / 'study-progress.json').read_text())
            log = Path(study['current_training_log'])
            with log.open('rb') as f:
                f.seek(max(0, log.stat().st_size - 300000))
                tail = f.read().decode(errors='replace')
            for line in tail.splitlines():
                if 'TREE_TIMING=' not in line:
                    continue
                try:
                    e = json.JSONDecoder().raw_decode(line.split('TREE_TIMING=', 1)[1])[0]
                except ValueError:
                    continue
                if (e.get('outer') == 117 and e.get('run_id') == expected
                        and e.get('event') == 'trajectory_complete'
                        and e.get('new_generated_tokens', e.get('policy_tokens', 0)) > 0):
                    state.update(accepted=True, boundaries=seen, native_restore=restored,
                                 next_actual_generation=e, training_continues=True)
                    (out / 'restored-five-accepted.json').write_text(json.dumps(state, sort_keys=True))
                    print(json.dumps(state), flush=True)
                    return
        time.sleep(15)
    raise TimeoutError('Follower window ended; inspect current training, do not infer training exit')


if __name__ == '__main__':
    main()
