"""Read saved E32/E33 selection metadata; never change training or pool state."""
from pathlib import Path
import datetime
import hashlib
import json
import sys

W = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work')
R = Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016')
s = json.loads((W / 'prefix20-online1000-state.json').read_text())
assert s['current_job_id'] == 3409
assert s['run_id'] == '8a6b24dc19664cb3a3fb97101f21c54e'
sys.path.insert(0, str(R))
from reproduction.h1008_qprobe.management_transport import remote

script = 'B = ' + repr(s['base']) + '\n' + r'''
import hashlib, json, random
from pathlib import Path
B = Path(B)
before_path = B/'epochs/epoch-0032/pool-state-at-head-completion.json'
selection_path = B/'epochs/epoch-0033/selection.json'
after_path = B/'epochs/epoch-0033/pool-state-at-head-completion.json'
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()
def read_stable(path):
    before = path.stat()
    assert before.st_size < 4 * 1024 * 1024
    raw = path.read_bytes()
    after = path.stat()
    sig = lambda v: (v.st_dev, v.st_ino, v.st_size, v.st_mtime_ns, v.st_ctime_ns)
    assert sig(before) == sig(after)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()
missing = [p for p in (before_path, selection_path) if not p.is_file()]
if missing:
    print(json.dumps(dict(status='PENDING_ACTUAL_E33_SELECTION', missing_files=len(missing))))
    raise SystemExit
before, before_sha = read_stable(before_path)
selection, selection_sha = read_stable(selection_path)
order = before['order']; size = len(order)
assert size == 168510 and set(order) == set(range(size))
assert before['cycle'] == 0
assert before['cursor'] == before['consumed_questions'] == 163840
assert before['next_epoch'] == 33 and before['epoch_questions'] == 5120
assert before['seed'] == 1729
old_tail = order[before['cursor']:]; old_set = set(old_tail)
assert len(old_tail) == 4670
shuffled = list(range(size))
random.Random(before['seed'] + before['cycle'] + 1).shuffle(shuffled)
new_order = [i for i in shuffled if i not in old_set] + [i for i in shuffled if i in old_set]
new_count = before['epoch_questions'] - len(old_tail)
new_prefix = new_order[:new_count]
assert new_count == 450 and not old_set.intersection(new_prefix)
rows = selection['rows']
assert selection['epoch'] == 33 and selection['questions'] == len(rows) == 5120
assert selection['trajectories'] == 25600 and selection['trajectory_group_size'] == 5
assert selection['fresh_rollout_required'] is True
assert selection['pool_cycle_start'] == 0 and selection['pool_cycle_end'] == 1
assert [r['pool_index'] for r in rows] == old_tail + new_prefix
assert [r['pool_cycle'] for r in rows] == [0] * 4670 + [1] * 450
assert len({r['qid'] for r in rows}) == 5120
assert len({r['pool_index'] for r in rows}) == 5120
assert len({r['normalized_question_hash'] for r in rows}) == 5120
assert selection['question_order_sha256'] == hashlib.sha256(canonical(rows)).hexdigest()
expected_after = dict(before, order=new_order, cycle=1, cursor=new_count,
                      next_epoch=34, consumed_questions=168960)
after_present = after_path.is_file(); after_sha = None
if after_present:
    after, after_sha = read_stable(after_path)
    assert after == expected_after
print(json.dumps(dict(
    status='ACTUAL_FIRST_POOL_CYCLE_SELECTION_PASS',
    before_snapshot_path=str(before_path), before_snapshot_sha256=before_sha,
    selection_path=str(selection_path), selection_sha256=selection_sha,
    selection_order_sha256=selection['question_order_sha256'],
    old_cycle_questions=len(old_tail), new_cycle_questions=new_count,
    unique_qids=5120, unique_pool_indices=5120, unique_normalized_hashes=5120,
    old_tail_matches=True, new_seed1730_order_matches=True,
    old_tail_disjoint_from_consumed_order_prefix=not old_set.intersection(order[:before['cursor']]),
    within_epoch_no_repeat=True, expected_next_epoch=34, expected_cycle=1,
    expected_cursor=new_count, expected_consumed_questions=168960,
    after_snapshot_present=after_present, after_snapshot_sha256=after_sha,
    after_saved_state_pending=not after_present, after_saved_state_matches=after_present,
    historical_selection_files_rechecked=False, fixed_diagnostic_exclusion_rechecked=False,
    question_text_transferred=False, gpu_used=False, training_modified=False
), sort_keys=True))
'''
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
transport = W / ('prefix20-first-pool-cycle-transport-' + stamp + '.json')
result = json.loads(remote(script, transport, control_path=None, target='h100-8'))
result.update(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              run_id=s['run_id'], job_id=s['current_job_id'], runtime_commit=s['runtime_source_commit'],
              audit_scope='saved E32 order to actual E33 selection; optional E33 saved boundary only',
              transport_receipt=str(transport),
              local_controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
p = W / ('prefix20-first-pool-cycle-summary-' + stamp + '.json')
p.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), summary=result)))
