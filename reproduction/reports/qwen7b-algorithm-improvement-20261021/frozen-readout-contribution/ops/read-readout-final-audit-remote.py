from pathlib import Path
import json,hashlib
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-readout-question-prefix-diagnostic-v1')
def fp(p):
 v=p.read_bytes();return dict(path=str(p),bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
p=b/'independent-audit.json';v=json.loads(p.read_text());assert p.stat().st_size<131072
print(json.dumps(dict(audit=v,audit_ref=fp(p),bound_config=json.loads((b/'config.json').read_text()),source_publication=fp(b/'source-publication.json'),actual_CPU_tests=json.loads((b/'actual-cpu-tests.json').read_text()),all_per_row_logits_remain_remote=True)))
