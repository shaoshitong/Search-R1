import gzip,json,time,hashlib
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');B=A/'full169028-policy80-ownG10-20261023-v1';D=A/'online-packed-full169028-M3to10-P20-20261022-v1/fixture'
def read(p):return json.loads(Path(p).read_bytes())
c=read(B/'input-contract.json');source=c['RAW_files'][573];events=read(A/'peer-newseed-ingest-audited138-after971-20261021-51057f37/raw-g10/source-events.json');
def walk(x):
 if isinstance(x,dict):
  if x.get('full',{}).get('path')==source['path']:yield x
  for v in x.values():yield from walk(v)
 elif isinstance(x,list):
  for v in x:yield from walk(v)
event=next(walk(events));out=dict(time=time.time(),event=event,raw=[],own_em=[],P20=[],final4096_read=False)
ids={'70188b8e50fd4b45bc420ce58014f46e-outer000001-c107521-g3','70188b8e50fd4b45bc420ce58014f46e-outer000001-c107529-g3','70188b8e50fd4b45bc420ce58014f46e-outer000001-c107543-g2'}
raw={}
with gzip.open(source['path'],'rt')as f:
 for line in f:
  r=json.loads(line)
  if r['trajectory_id']in ids:
   pol=[]
   for turn in r['behavior_turns']:pol.extend(turn['raw_generated_token_ids'][:turn['kept_token_count']])
   raw[r['trajectory_id']]=(r['initial_prompt_token_ids'],pol[:20]);out['raw'].append({k:r.get(k)for k in('qid','trajectory_id','strict_em','question_hash','actor_version')})
own=Path(event['own_em']['path']);op=gzip.open if own.suffix=='.gz'else open
with op(str(own),'rt')as f:
 for line in f:
  r=json.loads(line)
  if r.get('trajectory_id')in ids:out['own_em'].append(r)
for ref in read(D/'fixture-complete.json')['token_shards']:
 with gzip.open(ref['path'],'rt')as f:
  for line in f:
   r=json.loads(line);t=r['trajectory_id']
   if t in ids:out['P20'].append(dict(qid=r['qid'],trajectory_id=t,prompt_equal=r['prompt']==raw[t][0],prefix20_equal=r['policy20']==raw[t][1],row_keys=sorted(r)))
print(json.dumps(out))
