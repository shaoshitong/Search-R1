"""仅部署 HKUST4 独立入口和持久源码副本；不启动训练。"""
import base64
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
CONTROLLER = '/Users/shaoshitong/Documents/remote_codex_control_cc/control.py'
runid = datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S')
name = 'resnet50-noise-20261026-hkust4-' + runid
local_remote = '/home/shaoshitong/' + name
shared = '/weka_3/shaoshitong/lse-sparse-repro-20261026/shared/resnet50-' + runid
files = {n: base64.b64encode((ROOT / n).read_bytes()).decode()
         for n in ['train.py', 'launch_hkust4.py', 'deploy_hkust4.py', 'HKUST4.md']}
sha = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in files}
meta = dict(unit=name, directory=local_remote, source_backup=shared + '-source', output=shared,
            python='/home/shaoshitong/miniconda3/envs/EditVerse/bin/python',
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), sha256=sha)
code = '''from pathlib import Path
import json,base64,hashlib
meta=META
for target in [meta['directory'],meta['source_backup']]:
 root=Path(target);root.mkdir(exist_ok=False)
 for name,data in FILES.items():
  p=root/name;p.write_bytes(base64.b64decode(data));p.chmod(0o644)
  assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'][name]
 (root/'deployment-hkust4.json').write_text(json.dumps(meta,indent=2)+'\\n')
print('VERIFIED_SHA256:'+json.dumps(sorted(meta['sha256'].values())))
'''.replace('META', repr(meta)).replace('FILES', repr(files))
r = subprocess.run(['python3', CONTROLLER, 'run', '--timeout', '25', 'lab_4090_4', '--', 'python3', '-c', code], capture_output=True, text=True, timeout=35)
assert r.returncode == 0, r.stderr
checks = [json.loads(s.split(':', 1)[1]) for s in r.stdout.splitlines() if s.startswith('VERIFIED_SHA256:')]
assert checks == [sorted(sha.values())], r.stdout
(ROOT / 'deployment-hkust4.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta, indent=2))
