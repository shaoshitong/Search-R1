"""Read published native Prefix80 evaluations and candidate metadata only."""
from pathlib import Path
import hashlib
import json
import time

A = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
suite = A / 'online-mixed-layer-prefix-20261022-2gpu-v1'
out = dict(time=time.time(), experiments={}, model_or_checkpoint_opened=False, final4096_read=False)
def proof(p):
    raw = p.read_bytes()
    return dict(path=str(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
for layer in (15, 18, 21):
    base = suite / ('layer%d-prefix80' % layer)
    v = dict(base=str(base), evaluations={}, candidate_paths=[], prediction_paths=[])
    for name in ('prefix80-ready.json', 'factorial-ready.json', 'root-submission.json', 'controller.json', 'latest.json', 'complete.json'):
        p = base / name
        if p.is_file():
            v[name] = dict(proof=proof(p), value=json.loads(p.read_bytes()))
    for epoch in range(4):
        p = base / ('evaluation-epoch-%02d.json' % epoch)
        assert p.is_file(), str(p)
        v['evaluations'][str(epoch)] = dict(proof=proof(p), value=json.loads(p.read_bytes()))
        for role in ('fixed_train2048', 'fixed_validation2048'):
            p = base / ('epoch-%02d-%s-predictions.jsonl.gz' % (epoch, role))
            if p.is_file():
                v['prediction_paths'].append(dict(path=str(p), bytes=p.stat().st_size))
        if epoch:
            p = base / ('checkpoint-epoch-%02d.pt' % epoch)
            assert p.is_file()
            v['candidate_paths'].append(dict(path=str(p), bytes=p.stat().st_size, states=['raw', 'EMA'], model_contents_opened=False))
    out['experiments'][str(layer)] = v
for name in ('fixture-complete.json', 'inputs/fixed-train2048.json', 'inputs/validation-groups.json'):
    p = A / 'online-mixed-layer-prefix-20261022-v1' / name
    out[name] = proof(p)
out['fixture_metadata'] = json.loads((A / 'online-mixed-layer-prefix-20261022-v1/fixture-complete.json').read_bytes())
print(json.dumps(out))
