"""Fetch only the small completed audit, not checkpoint/prediction tensors."""
from pathlib import Path
import json,subprocess,sys
C=Path(__file__).resolve().parents[2];O=C/'outputs/qwen7b-warmup-20261019';P=O/'reports'
program="""import pathlib,json,hashlib
p=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-within1-matched256-v1/final-independent-audit.json');b=p.read_bytes();assert len(b)<65536;print(json.dumps(dict(audit=json.loads(b),ref=dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))))
"""
script=O/'read-peer-within1-final-audit-remote.py';script.write_text(program)
r=subprocess.run([sys.executable,str(O/'balanced-control.py'),str(script),'--timeout','55'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
if r.returncode:raise RuntimeError(r.stderr[-1500:]+' '+r.stdout[-1500:])
rows=[json.loads(x) for x in r.stdout.splitlines() if x.startswith('{')];v=rows[-1]
(P/'peer-within1-final-independent-audit.json').write_text(json.dumps(v['audit'],sort_keys=True,indent=2)+'\n');(P/'peer-within1-final-independent-audit-ref.json').write_text(json.dumps(v['ref'],sort_keys=True,indent=2)+'\n');print(json.dumps(v['ref']))
