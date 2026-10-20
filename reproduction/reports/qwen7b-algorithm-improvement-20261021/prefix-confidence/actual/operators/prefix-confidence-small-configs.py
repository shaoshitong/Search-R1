from pathlib import Path
import json,hashlib
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/prefix-confidence-incremental-v1')
def ref(p):
 x=p.read_bytes();return dict(path=str(p),bytes=len(x),sha256=hashlib.sha256(x).hexdigest())
out={}
for name in ('config.json','fit-config-v2.json','fit-deployment-v2/source-publication.json','fit-deployment-v2/preparation.json','fit-deployment-v2/exit.json'):
 p=b/name;x=json.loads(p.read_text());assert not any(k in x for k in ('raw','normalizer','state_dict','optimizer_state_dict','rng','p_j','features_matrix'));out[name]=dict(reference=ref(p),configuration_or_metadata=x)
print(json.dumps(dict(status='SMALL_CONFIG_AND_PROVENANCE_ONLY_NO_LEARNED_SCALAR_OR_ROWS',files=out),sort_keys=True))
