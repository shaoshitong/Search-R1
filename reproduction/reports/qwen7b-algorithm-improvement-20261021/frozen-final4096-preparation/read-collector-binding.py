import hashlib,json
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');D=B/'confirmation-materialization-deploy-20261021'
p=D/'source-publication.json';pub=json.loads(p.read_bytes());out=dict(publication=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()),publication_keys=list(pub),publication_metadata={k:v for k,v in pub.items()if k in('commit','code_root','source_manifest_sha256')},events=[])
m=json.loads((D/'materialized/materialization.json').read_bytes())
for pool,v in m['pools'].items():
 for e in v['event_configs']:
  c=json.loads(Path(e['path']).read_bytes());out['events'].append(dict(pool=pool,config=e,fields={k:c[k]for k in('base','code_root','model_dir','model_revision','gpus_per_sampling_job','sampling_shards','capture_prefix_tokens','question_max_tokens','fit_epochs','fitting_authorized','evaluation_only','run_id','sampling_attempt_id','actor_complete')if k in c},code_files=len(c['collector_code_files_sha256'])))
print(json.dumps(out))
