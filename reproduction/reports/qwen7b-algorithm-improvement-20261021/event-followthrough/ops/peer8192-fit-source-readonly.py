import json, pathlib, hashlib
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
paths=[B/'replay-matched9088-bank-v2-20261021/prepared/bank-complete.json',B/'algorithm-improvement-20261021/matched-loss128-v2/preparation.json']
r={}
for p in paths:
 v=json.loads(p.read_text());r[str(p)]={'keys':list(v),'status':v.get('status'),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 if 'feature_manifests_by_capture_layer' in v:r[str(p)]['feature_manifests_by_capture_layer']=v['feature_manifests_by_capture_layer']
 if 'features_by_capture_layer' in v:r[str(p)]['features_by_capture_layer']=v['features_by_capture_layer']
 if 'variants' in v:
  c=json.loads(pathlib.Path(v['variants']['event']['config']['path']).read_text());r[str(p)]['reference']={k:c[k] for k in ['source_checkpoint','source_config','source_bank','model_config','model_dir','bank']}
print(json.dumps(r))
