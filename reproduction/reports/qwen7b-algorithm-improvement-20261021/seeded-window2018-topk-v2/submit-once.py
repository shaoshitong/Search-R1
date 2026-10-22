"""Root-only independent second training seed; never repeat an unknown submission."""
import fcntl
import hashlib
import json
import pathlib
import re
import subprocess
import time

suite = pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-window2018-seed19137-plan2026102302-L15P80-8192-20261023-v2')
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
assert proof(entry)['sha256'] == 'd15b7ab72b9b5fed7233fe605c96b5430a2b989777c3bc6b94eba50c2fea89f7'
publication = read(suite / 'source-publication.json')
assert publication['commit'] == '895051de98c6b2a4f5e13468623034a625b97eb8'
assert proof(suite / 'source-publication.json')['sha256'] == 'cbe9ebc96b24197ed3b7df511232541f32fa1b7bedee9ea787cd000b12a4b24d'
assert proof(base / 'ready.json')['sha256'] == '877c97add231fd6c90df4245460536a04d731617f180521097400cf89dc76b47'
ready = read(base / 'ready.json')
assert ready['entry'] == proof(entry)
checks = pathlib.Path(ready['actual_seed_checks']['path'])
assert proof(checks) == ready['actual_seed_checks']
assert read(checks)['actual_entry_CLI_2018_accepted'] is True
old_accounting = run(['sacct', '-n', '-X', '-j', '750023', '--format=JobIDRaw,State,ExitCode', '-P'])
assert '750023|FAILED|1:0' in old_accounting
receipt = base / 'root-submission.json'
intent = base / 'root-submit-intent.json'
if receipt.exists():
    print(json.dumps(read(receipt)))
else:
    name = re.search(r'^#SBATCH --job-name=(.+)$', entry.read_text(), re.M).group(1)
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
                  no_dependencies=True, superseded_failed_job='750023', previous_attempt_training_updates=0, actual_entry_CLI_2018_accepted=True)
    receipt.write_text(json.dumps(result, sort_keys=True) + '\n')
    print(json.dumps(result))
