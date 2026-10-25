"""Read-only small evidence extraction; no GPU forward or training mutation."""
from reproduction.dynrpo7b.local_control import remote

SCRIPT = r'''
import pathlib,json,re,time,hashlib,collections
root=pathlib.Path('/data/user/jxu957/experiments')
names=[
 ('dynrpo','dynrpo7b-formal-20261023-recoveryfix-r6',50,100),
 ('easy','easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022',50,52),
 ('easy','easy2560-corollout-joint4-qwen7b-fa3-lean-r1-20261022',53,54),
 ('easy','easy2560-corollout-joint4-qwen7b-fa3-nativeprefix-r2-20261023',55,60),
 ('easy','easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023',61,70),
 ('easy','easy2560-corollout-joint4-qwen7b-fa3-batch-searchstop-r1-20261023',71,100)]
files=['verl/workers/actor/dp_actor.py','verl/trainer/ppo/core_algos.py',
 'search_g1/logical_micro_loss.py','search_g1/signals.py','reproduction/common8/train.sh',
 'search_g1/async_rollout_runtime.py','search_g1/token_probability.py',
 'verl/trainer/main_ppo_eco.py','reproduction/common8/context_budget.py',
 'reproduction/common8/refit.py','search_g1/search_action_stop.py']
result=dict(observed_unix=time.time(),training_modified=False,gpu_forwards=0,runs=[])
for label,name,lo,hi in names:
 b=root/name;r=json.loads((b/'logs/run-manifest.json').read_text())
 cfg=json.loads((b/'logs/resolved-training-config.json').read_text())['resolved_config']
 path=pathlib.Path(r['train_log']);text=path.read_text();metrics=[];collections_out=[]
 for line in text.splitlines():
  if 'actor/completed_outer:' in line:
   values={k:float(v) for k,v in re.findall(r'([A-Za-z0-9_/]+):(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)',line)}
   step=int(values['actor/completed_outer'])
   if lo<=step<=hi:metrics.append(values)
  if 'DYNRPO_COLLECTION=' in line:
   v=json.loads(line.split('DYNRPO_COLLECTION=',1)[1])
   if lo<=v['step']<=hi:collections_out.append(v)
 if label=='easy':
  for p in (b/'artifacts/mixed-selection').glob('*.json'):
   v=json.loads(p.read_text())
   if not lo<=v['step']<=hi:continue
   groups=v.pop('groups');v['pool_counts']=dict(collections.Counter(g['category'] for g in groups))
   v['selected_counts']=dict(collections.Counter(g['category'] for g in groups if g['selected']))
   v['pool_correct_histogram']=dict(collections.Counter(sum(g['strict_correctness']) for g in groups))
   v['selected_correct_histogram']=dict(collections.Counter(sum(g['strict_correctness']) for g in groups if g['selected']))
   v.pop('candidate_generation_phases',None);collections_out.append(v)
 receipts=[]
 for step in range(lo,hi+1):
  workers=[]
  for rank in range(4):
   p=b/('logs/formal-inline-optimizer-step-%d-rank-%d.json'%(step,rank))
   if p.is_file():
    q=json.loads(p.read_text());workers.append({k:q.get(k) for k in ['rank','optimizer_writes','consumed_lr','all_parameters_and_adam_finite','run_id','logical_update']})
  receipts.append(dict(step=step,workers=workers))
 rewards=[]
 for p in (b/'artifacts').glob('round_*/version-manifest.json'):
  q=json.loads(p.read_text())
  if q['round'] in (0,50,100):rewards.append({k:q.get(k) for k in ['round','version','encoder','encoder_sha256','head_sha256','calibration_report','published_unix']})
 result['runs'].append(dict(label=label,base=str(b),steps=[lo,hi],source_commit=r['source_commit'],
  config=cfg,budget=r['budget'],source_sha={f:hashlib.sha256((b/'repo'/f).read_bytes()).hexdigest() for f in files if (b/'repo'/f).is_file()},
  infra_optimizations=r.get('infra_optimizations'),infra_rollout=r.get('infra_rollout'),
  eco_init=[l for l in text.splitlines() if '[ECO-INIT]' in l],
  metrics=metrics,collections=sorted(collections_out,key=lambda x:x['step']),optimizer_receipts=receipts,reward_heads=rewards))
print(json.dumps(result))
'''

if __name__ == '__main__':
    remote(SCRIPT)
