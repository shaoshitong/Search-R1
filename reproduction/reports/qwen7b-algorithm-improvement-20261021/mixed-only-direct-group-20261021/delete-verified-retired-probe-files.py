"""Remove only four SHA-bound terminal owned probe binaries under human pivot."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time

A = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
main = A / 'joint-balanced-native-E4-124194-v1'
snapshot = A / 'joint124194-current-snapshot-eval-v1'
base = A / 'retired-probe-cleanup-mixed-only-20261021'
plan_path = base / 'plan.json'
PLAN_SHA = '0a04a41833b8f3795e7c3482306fa4830e58ab4aa78c891140f87908ae614fdf'
def proof(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 ** 2), b''):
            h.update(block)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())
def atomic(p, value):
    tmp = p.with_name(p.name + '.tmp-' + str(os.getpid()))
    with tmp.open('x') as f:
        json.dump(value, f, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))

guard = (base / 'cleanup.lock').open('a+')
fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert proof(plan_path)['sha256'] == PLAN_SHA
plan = json.loads(plan_path.read_bytes())
assert plan['old_jobs'] == ['745501', '745981'] and len(plan['files']) == 4
assert plan['official_model_and_all_rollout_feature_banks_excluded'] is True
sacct = subprocess.check_output(['sacct', '-n', '-P', '-j', '745501,745981', '--format=JobIDRaw,State,ExitCode'], universal_newlines=True)
for job in plan['old_jobs']:
    assert job + '|COMPLETED|0:0' in sacct
locks = []
for p in (main / 'training.lock', snapshot / 'evaluation.lock'):
    f = p.open('a+')
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(f)
complete_path = base / 'complete.json'
if complete_path.exists():
    done = json.loads(complete_path.read_bytes())
    assert done['plan']['sha256'] == PLAN_SHA
    assert all(not Path(v['path']).exists() for v in plan['files'])
    print(json.dumps(dict(status='RETIRED_FILES_ALREADY_REMOVED', complete=proof(complete_path), value=done)))
    raise SystemExit(0)

# Each unlink has its own durable intent and receipt. After an interrupted unlink,
# the missing exact path is accepted only if this run's SHA-bound intent exists.
removed = []
for i, ref in enumerate(plan['files']):
    p = Path(ref['path'])
    assert p.parent in (main / 'checkpoints', snapshot / 'inputs') and p.resolve() == p and p.suffix == '.pt'
    intent_path = base / ('unlink-%d-intent.json' % i)
    receipt_path = base / ('unlink-%d-removed.json' % i)
    if p.exists():
        st = p.stat()
        assert stat.S_ISREG(st.st_mode) and st.st_uid == 21930 and not p.is_symlink()
        assert (st.st_ino, st.st_nlink, st.st_mtime_ns, st.st_size) == (ref['inode'], ref['nlink'], ref['mtime_ns'], ref['bytes'])
        assert proof(p) == {k: ref[k] for k in ('path', 'bytes', 'sha256')}
        intent = dict(plan_sha256=PLAN_SHA, file=ref, epoch=time.time())
        if intent_path.exists():
            prior = json.loads(intent_path.read_bytes())
            assert prior['plan_sha256'] == PLAN_SHA and prior['file'] == ref
        else:
            atomic(intent_path, intent)
        p.unlink()
    else:
        prior = json.loads(intent_path.read_bytes())
        assert prior['plan_sha256'] == PLAN_SHA and prior['file'] == ref
    assert not p.exists()
    receipt = dict(path=ref['path'], bytes=ref['bytes'], previous_sha256=ref['sha256'],
                   prior_nlink=ref['nlink'], removed=True, epoch=time.time())
    if not receipt_path.exists():
        atomic(receipt_path, receipt)
    removed.append(json.loads(receipt_path.read_bytes()))

done = dict(status='FOUR_RETIRED_OWNED_PROBE_BINARY_PATHS_REMOVED_BY_HUMAN_AUTHORIZATION',
    plan=proof(plan_path), files=removed, logical_bytes_removed=sum(v['bytes'] for v in removed),
    physical_freed_bytes_not_claimed=True, other_hardlinks_not_followed=True,
    official_models_rollouts_features_code_statistics_preserved=True,
    new_source='official_Qwen2.5_7B_a09a_block27_no_old_probe_state', epoch=time.time())
atomic(complete_path, done)
print(json.dumps(dict(complete=proof(complete_path), value=done), sort_keys=True))
