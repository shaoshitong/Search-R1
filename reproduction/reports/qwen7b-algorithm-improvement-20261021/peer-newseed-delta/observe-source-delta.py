"""Published source-only metadata worker via the existing strict H100 route.

The source program/input binding travel through stdin RAM. No remote source,
claim, lease, service or payload file is written. Only a small local metadata
plan/aggregate is persisted; raw question/prompt/label rows never leave NAS10.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CONTROL = Path('/Users/shaoshitong/Documents/remote_codex_control/control.py')
MODULE = 'reproduction/operations/qwen7b_warmup/peer_newseed_delta.py'
BRANCH = 'refs/heads/codex/qwen7b-probe-warmup-20261019'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--observation', type=int, choices=(1,2), required=True)
args = parser.parse_args()
out = HERE / ('source-observation-%d.json' % args.observation)
if out.exists():
    raise ValueError('Existing finite observation preserved; inspect its actual result')
commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
remote = subprocess.check_output(['git','ls-remote','origin',BRANCH],cwd=REPO,text=True).split()[0]
if commit != remote:
    raise ValueError('Publish and verify source before remote CPU invocation')
source = (REPO/MODULE).read_bytes()
if source != subprocess.check_output(['git','show',commit+':'+MODULE],cwd=REPO):
    raise ValueError('Unpublished metadata worker refused')
operator = Path(__file__).relative_to(REPO)
if Path(__file__).read_bytes() != subprocess.check_output(['git','show',commit+':'+str(operator)],cwd=REPO):
    raise ValueError('Unpublished observer refused')
binding_path = HERE/'imported-snapshot-fresh-binding.json'
binding_bytes = binding_path.read_bytes(); binding = json.loads(binding_bytes)
packet = dict(source=source.decode(),source_sha256=hashlib.sha256(source).hexdigest(),binding=binding)
program = r'''
import hashlib,json,os,resource,sys,time
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120))
p=json.loads(sys.stdin.buffer.read());assert hashlib.sha256(p['source'].encode()).hexdigest()==p['source_sha256']
ns={'__name__':'read_only_peer_delta'};exec(compile(p['source'],'<published_peer_newseed_delta>','exec'),ns)
started=time.time();v=ns['inventory'](p['binding']);v.update(elapsed_seconds=time.time()-started,worker_source_sha256=p['source_sha256'])
print(json.dumps(v,sort_keys=True))
'''
command = 'python3 -c ' + shlex.quote(program)
if len(command.encode()) >= 16000:
    raise ValueError('Control command exceeds16KB')
spec = importlib.util.spec_from_file_location('peer_delta_safe_route',CONTROL)
ctl = importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
started = time.time()
r = subprocess.run(ctl.route('h100-10',command),input=json.dumps(packet).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=150)
if r.returncode:
    raise RuntimeError('Read-only metadata observation failed, no payload action occurred: '+r.stderr.decode(errors='replace')[-2000:])
result = json.loads(r.stdout.decode().splitlines()[-1])
if len(r.stdout)>1024**2:
    raise ValueError('Metadata plan unexpectedly exceeds1MiB; do not save large payload')
result.update(source_commit=commit,remote_sha=remote,operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              imported_binding_sha256=hashlib.sha256(binding_bytes).hexdigest(),operator_elapsed_seconds=time.time()-started)
out.write_text(json.dumps(result,indent=2)+'\n')
p = result['plan']; keys=('current_sealed_batches','current_sealed_questions','current_audit_fstat_bound_batches','imported_batches',
    'delta_batches','delta_questions','delta_trajectories','audited_delta_batches','audited_delta_questions',
    'sealed_pending_audit_batches','sealed_pending_audit_questions','domain_counts','imported_overlap')
print(json.dumps(dict(status=result['status'],epoch=result['epoch'],plan_sha256=result['plan_sha256'],
      source_commit=commit,summary={k:p[k] for k in keys},output=str(out)),sort_keys=True))
