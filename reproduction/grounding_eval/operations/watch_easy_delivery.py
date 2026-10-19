"""Read-only compact task progress; no raw trajectories/model bytes leave A800."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--phase', choices=['pilot96','normal3610','dual-cf'], required=True)
parser.add_argument('--audit-dir', required=True)
args = parser.parse_args()
code = r'''
import pathlib,json,time,subprocess,sys
phase=json.loads(INPUT)['phase'];base=pathlib.Path('/data/shaoshitong/grounding-eval-20261019')
name={'pilot96':'easy-selection-normal-pilot96','normal3610':'easy-selection-normal3610','dual-cf':'easy-selection-dual-cf-control'}[phase]
root=base/name;out={'at':time.time(),'phase':phase,'root':str(root),'exists':root.exists()}
control=root if phase=='dual-cf'else root/'control'
for name in ['launch.json','worker.json','exit.json','progress.json','failed.json','complete.json']:
 p=control/name
 if p.exists():out['control_'+name]=json.loads(p.read_text())
if 'control_launch.json'in out:
 pid=out['control_launch.json']['pid'];out['process']=subprocess.run(['ps','-p',str(pid),'-o','pid,ppid,etimes,args'],capture_output=True,text=True,timeout=5).stdout
if phase!='dual-cf':
 for name in ['progress.json','normal-error.json','complete.json']:
  p=root/name
  if p.exists():out[name]=json.loads(p.read_text())
 out['raw_shards']=sum(1 for _ in(root/'raw').glob('*.json'));out['raw_bytes']=sum(p.stat().st_size for p in(root/'raw').glob('*.json'))
else:
 for version,name in [('v4','easy-selection-cf-full-v4'),('v5','easy-selection-cf-strict-v5')]:
  cf=base/name;counts={'results':sum(1 for _ in(cf/'results').glob('*.json'))}
  for filename in ['summary.json','verified-summary.json','independent-acceptance-20261019.json','official-changed-rule-replay.json']:
   p=cf/filename
   if p.exists():counts[filename]=json.loads(p.read_text())
  out[version]=counts
print(json.dumps(out))
'''.replace('INPUT', repr(json.dumps(dict(phase=args.phase))))
spec = importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py')
ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
command = ctl.route('baai-a800','python3 -B -c '+shlex.quote(code))
command[1:1] = ['-o','ServerAliveInterval=5','-o','ServerAliveCountMax=6']
result = subprocess.run(command,capture_output=True,text=True,timeout=25)
if result.returncode:
    raise RuntimeError('Read-only task observation failed; no task restarted')
out = json.loads(result.stdout); directory=Path(args.audit_dir); directory.mkdir(exist_ok=True)
(directory/('easy-'+args.phase+'-observation-'+str(time.time_ns())+'.json')).write_text(json.dumps(out,indent=2)+'\n')
(directory/('easy-'+args.phase+'-latest.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
