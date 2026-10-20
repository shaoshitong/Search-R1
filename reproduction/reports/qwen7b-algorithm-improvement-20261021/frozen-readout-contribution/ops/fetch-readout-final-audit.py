"""Transfer only small aggregate audit/config/provenance; no per-row logits."""
from pathlib import Path
import json,subprocess,sys
C=Path(__file__).resolve().parents[2];O=C/'outputs/qwen7b-warmup-20261019';P=O/'reports'
program="""from pathlib import Path
import json,hashlib
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-readout-question-prefix-diagnostic-v1')
def fp(p):
 v=p.read_bytes();return dict(path=str(p),bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
p=b/'independent-audit.json';v=json.loads(p.read_text());assert p.stat().st_size<131072
print(json.dumps(dict(audit=v,audit_ref=fp(p),bound_config=json.loads((b/'config.json').read_text()),source_publication=fp(b/'source-publication.json'),actual_CPU_tests=json.loads((b/'actual-cpu-tests.json').read_text()),all_per_row_logits_remain_remote=True)))
"""
script=O/'read-readout-final-audit-remote.py';script.write_text(program)
r=subprocess.run([sys.executable,str(O/'balanced-control.py'),str(script),'--timeout','55'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
if r.returncode:raise RuntimeError(r.stderr[-1500:]+' '+r.stdout[-1500:])
v=[json.loads(line) for line in r.stdout.splitlines() if line.startswith('{')][-1];(P/'readout-contribution-final-delivery.json').write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');print(json.dumps(dict(audit_ref=v['audit_ref'],complete=v['audit']['complete'],job_id=v['audit']['job_id'])))
