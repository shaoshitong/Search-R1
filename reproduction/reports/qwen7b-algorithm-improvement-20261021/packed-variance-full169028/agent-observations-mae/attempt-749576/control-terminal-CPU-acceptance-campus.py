"""Use the original published CPU acceptor via the verified campus control route."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import types

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[5]
COMMIT = '799e75f35347055a281d290f8f22777ac97738f5'
PACKAGE = 'reproduction/operations/qwen7b_warmup/'
BRANCH = 'refs/heads/codex/q7b-full169028-packed-cpu-acceptance-20261022'
parser = argparse.ArgumentParser()
parser.add_argument('--action', choices=('status','start'), required=True)
a = parser.parse_args()
remote = subprocess.check_output(['git','ls-remote','origin',BRANCH],cwd=REPOSITORY,text=True).split()[0]
subprocess.check_call(['git','merge-base','--is-ancestor',COMMIT,remote],cwd=REPOSITORY)
deploy = subprocess.check_output(['git','show',COMMIT+':'+PACKAGE+'deploy_full169028_packed_CPU_acceptance.py'],cwd=REPOSITORY)
original = types.ModuleType('published_full169028_CPU_acceptance_transport')
original.__file__ = str(REPOSITORY/PACKAGE/'deploy_full169028_packed_CPU_acceptance.py')
exec(compile(deploy,original.__file__,'exec'),original.__dict__)
source = subprocess.check_output(['git','show',COMMIT+':'+PACKAGE+'accept_full169028_packed_variance.py'],cwd=REPOSITORY)
values = {'base': original.BASE, 'commit':COMMIT, 'source':base64.b64encode(source).decode(),
          'sha256':hashlib.sha256(source).hexdigest()}
body = original.STATUS if a.action=='status' else original.STAGE+original.START
script = 'v='+repr(values)+'\n'+original.COMMON+body
compile(script,'original-published-full169028-CPU-acceptance-via-campus','exec')
control = REPOSITORY.parent.parent/'outputs/qwen7b-warmup-20261019/campus-control.py'
with tempfile.TemporaryDirectory(prefix='q7b-CPU-campus-') as folder:
    entry = Path(folder)/'control.py'
    entry.write_text(script)
    result = subprocess.run(['python3',str(control),str(entry)],cwd=REPOSITORY,capture_output=True,text=True,timeout=65)
if result.returncode:
    raise RuntimeError('Campus CPU acceptance control failed: '+result.stderr[-1000:]+result.stdout[-1600:])
report = json.loads(result.stdout)
(HERE/('actual-terminal-CPU-campus-'+a.action+'.json')).write_text(json.dumps(report,sort_keys=True)+'\n')
print(json.dumps(report,sort_keys=True))
