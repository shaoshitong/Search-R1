"""Publish this observer's completed-epoch small metadata through an isolated Git index."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[5]
BRANCH = 'refs/heads/codex/q7b-full169028-mae-observer-20261022'
parser = argparse.ArgumentParser()
parser.add_argument('epoch', type=int)
epoch = parser.parse_args().epoch
assert 1 <= epoch <= 10
subprocess.check_call(['python3', str(HERE/'summarize-completed-epoch-diagnostics.py')], cwd=REPOSITORY)
snapshot = json.loads((HERE/'actual-mae-latest.json').read_text())
files = snapshot['files']
assert snapshot['accounting']['stdout'].split('|')[0] == '749576'
assert files['controller.json']['job_id'] == '749576'
key = 'evaluation-epoch-%02d.json' % epoch
assert key in files
checkpoint = files['latest.json']
assert checkpoint['path'].endswith('/checkpoint-epoch-%02d.pt' % epoch)
result = {'observation_time': snapshot['time'], 'job_id': '749576',
          'accounting_at_observation': snapshot['accounting'],
          'coverage': files['coverage-epoch-%02d.json' % epoch], 'evaluation': files[key],
          'checkpoint': checkpoint,
          'subsequent_two_rank_progress': [files.get('progress-rank-%d.json'%rank) for rank in (0,1)],
          'checkpoint_CPU_acceptance': 'WAIT_UNTIL_FINAL_TERMINAL',
          'no_training_or_Slurm_changes': True, 'final4096_read': False}
proof = HERE/('actual-epoch-%02d-proof.json'%epoch)
proof.write_text(json.dumps(result, sort_keys=True)+'\n')
owned = [proof, HERE/'actual-all-completed-epoch-metrics.json', Path(__file__)]
for name in ('follow-full-ten-epochs-mae-campus.py','actual-observer-campus-handoff.json','control-read-errors.jsonl','control-terminal-CPU-acceptance-campus.py','actual-terminal-CPU-campus-status.json'):
    if (HERE/name).exists(): owned.append(HERE/name)
for template in ('actual-score-spread-baselines-epoch-%02d.json', 'actual-same-epoch-three-arm-%02d.json',
                 'read-score-spread-baselines-epoch-%02d.py', 'read-same-epoch-three-arm-%02d.py'):
    path = HERE/(template%epoch)
    assert path.exists()
    owned.append(path)
fd, index = tempfile.mkstemp(prefix='q7b-mae-epoch-index-')
os.close(fd)
os.unlink(index)
env = dict(os.environ, GIT_INDEX_FILE=index)
def run(args, **kwargs):
    return subprocess.check_output(args, cwd=REPOSITORY, env=env, text=True, **kwargs).strip()
try:
    base = run(['git','rev-parse',BRANCH])
    run(['git','read-tree',base])
    run(['git','add','--']+[str(path.relative_to(REPOSITORY)) for path in owned])
    tree = run(['git','write-tree'])
    if tree == run(['git','rev-parse',base+'^{tree}']):
        commit = base
    else:
        commit = run(['git','commit-tree',tree,'-p',base],
                     input='Record full169028 MAE completed epoch %d and untouched fixed-input diagnostics\n'%epoch)
        run(['git','update-ref',BRANCH,commit,base])
    run(['git','push','origin',BRANCH])
    remote = run(['git','ls-remote','origin',BRANCH]).split()[0]
    assert remote == commit
    print(json.dumps({'epoch':epoch,'published_commit':commit,'remote_SHA_verified':remote,'shared_HEAD_index_untouched':True}))
finally:
    if os.path.exists(index): os.unlink(index)
