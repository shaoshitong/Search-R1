import hashlib,json
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
D=B/'confirmation-materialization-deploy-20261021'
def ref(p):
 raw=Path(p).read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
result=dict(final_labels_or_results_read=False)
for p in (D/'result.json',D/'failure.json',D/'materialized/materialization.json'):
 if p.exists():
  c=json.loads(p.read_bytes());result[p.name]=dict(reference=ref(p),metadata={k:c[k]for k in ('status','actual_responses_generated','jobs_submitted','runtime_qualified','remaining','source_freeze','pools')if k in c})
M=D/'materialized/materialization.json'
if M.exists():
 m=json.loads(M.read_bytes());paths=[]
 for name,pool in m['pools'].items():
  own=dict(name=name,question_plan=pool['question_plan'],planned_G10_and_fixed_G5=pool['planned_G10_and_fixed_G5'],event_configs=pool['event_configs'],event_states=[])
  for config in pool['event_configs']:
   base=Path(config['path']).parent;found=[]
   for root in (base,base/'bank',base/'merged',base/'capture'):
    if root.exists():found.extend(dict(path=str(p),bytes=p.stat().st_size)for p in root.glob('*.json')if p.name in('complete.json','progress.json','failure.json','features-layer26.json','groups.json','submission-result.json','sampling-submission.json'))
   own['event_states'].append(dict(base=str(base),files=found))
  paths.append(own)
 result['materialized_identity_and_event_locators']=paths
print(json.dumps(result))
