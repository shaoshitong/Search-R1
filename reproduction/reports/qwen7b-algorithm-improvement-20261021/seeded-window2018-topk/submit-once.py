"""Root-only independent second training seed; never repeat an unknown submission."""
import fcntl
import hashlib
import json
import pathlib
import re
import subprocess
import time

suite = pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-window2018-seed19137-plan2026102302-L15P80-8192-20261023-v1')
base = suite / 'window2018'

def read(path):
    return json.loads(path.read_bytes())

def proof(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

def run(args):
    process = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, timeout=20)
    if process.returncode:
        raise RuntimeError(process.stdout[-800:] + process.stderr[-1000:])
    return process.stdout.strip()

lock = (base / 'root-submit.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
entry = base / 'training.sbatch'
assert proof(entry)['sha256'] == 'fcd836189c9e198f29d0a3b7366821a8b1eef4a691769337e4cd5d64e298c591'
publication = read(suite / 'source-publication.json')
assert publication['commit'] == 'e1afe3b1616bcee34a8944366859cb57b57ed44a'
assert proof(suite / 'source-publication.json')['sha256'] == 'b86b0b9758f5ee4158ae19ccb43d8c2053e1d53ffa9fb6d669570c5038786826'
assert proof(base / 'ready.json')['sha256'] == 'c393610e23a37cb13a688d876e5723e32d5b8d5f7439de4005490400c3fe372d'
assert read(base / 'ready.json')['entry'] == proof(entry)
receipt = base / 'root-submission.json'
intent = base / 'root-submit-intent.json'
if receipt.exists():
    print(json.dumps(read(receipt)))
else:
    name = re.search(r'^#SBATCH --job-name=(.+)$', entry.read_text(), re.M).group(1)
    assert name == 'q7b-L15P80-w2018-s19137'
    rows = run(['squeue', '-h', '-u', 'mou176', '-n', name, '-o', '%i']).splitlines()
    if rows:
        assert len(rows) == 1
        job = rows[0]
        details = run(['scontrol', 'show', 'job', job, '-o'])
        assert 'UserId=mou176(' in details and 'Command=' + str(entry) in details
    else:
        if intent.exists():
            raise RuntimeError('Prior submission outcome unknown; inspect same-name accounting before retrying.')
        assert not (base / 'latest.json').exists() and not (base / 'controller.json').exists()
        intent.write_text(json.dumps(dict(time=time.time(), entry=proof(entry), name=name), sort_keys=True) + '\n')
        job = run(['sbatch', '--parsable', '--exclude=ACD1-13', str(entry)]).split(';')[0]
        assert job.isdigit()
    result = dict(time=time.time(), job_id=job, name=name, entry=proof(entry),
                  ready=proof(base / 'ready.json'), source_commit=publication['commit'],
                  init_seed=19137, train_plan_seed=2026102302, GPUs=2, partition='acd_u',
                  epochs=10, global_batch=64, batch_attempt_budget=1280,
                  cache_window_total=2018, historical_detached_max=1954, current_differentiable=64,
                  fresh=True, resume_checkpoint=None, primary_metric='Top512_real_Mixed',
                  replication_epoch=10, replication_state='raw', final4096_read=False,
                  no_dependencies=True)
    receipt.write_text(json.dumps(result, sort_keys=True) + '\n')
    print(json.dumps(result))
