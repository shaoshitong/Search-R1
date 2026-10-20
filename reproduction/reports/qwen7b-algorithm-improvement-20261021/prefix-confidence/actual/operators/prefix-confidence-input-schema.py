from pathlib import Path
import json,hashlib
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';S=A/'peer8192-frozen-calibration-scores-v2'
def read(p):return json.loads(Path(p).read_text())
v=read(S/'complete.json');c=read(S/'config.json');e=read(A/'peer-newseed8192-g10-train-v1/E4-same-HF-validation/complete.json');d=read(A/'validation-g10-20261021/materialized/bank/complete.json');print(json.dumps(dict(scoring_complete_keys=list(v),scoring_complete=v,config_keys=list(c),checkpoint=[x for x in c['checkpoints'] if x['name']=='E4'],external_descriptor=e['checkpoints']['E4']['predictions'],validation_consumer=d['consumer_inputs'],validation_audit=d['outputs']['source-audit.json']),sort_keys=True))
