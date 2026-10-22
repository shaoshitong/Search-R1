"""Inspect small source-event scope/count fields, never final or feature data."""
from collections import Counter
import hashlib,json,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');F=A/'online-packed-full169028-M3to10-P20-20261022-v1/fixture'
def read(p):return json.loads(Path(p).read_bytes())
d=read(F/'full-training-descriptor.json');out=[]
for ref in d['source_metadata']:
 if Path(ref['path']).name!='source-events.json':continue
 x=read(ref['path']);events=x['events'];one=dict(source=ref,event_type=type(events).__name__,scopes={})
 for key,rows in (events.items()if isinstance(events,dict)else[(r.get('event','event-%d'%i),r.get('batches',[]))for i,r in enumerate(events)]):
  if not isinstance(rows,list):continue
  first=rows[0]if rows else{};sums={k:sum(r.get(k,0)for r in rows)for k in('questions','trajectories','rows')if all(type(r.get(k))is int for r in rows)}
  one['scopes'][key]=dict(events=len(rows),summed_scalars=sums,first_keys=list(first),first_scalars={k:v for k,v in first.items()if isinstance(v,(str,int,float,bool))or v is None},first_list_lengths={k:len(v)for k,v in first.items()if isinstance(v,list)})
 out.append(one)
f=read(A/'online-packed-matched-L15P80-8192-20261023-v1/fixture/fixture-complete.json')
print(json.dumps(dict(time=time.time(),scopes=out,accepted_P80_pilot_token_rows=92160,accepted_P80_pilot_token_compressed_bytes=sum(x['bytes']for x in f['token_shards']),accepted_P80_pilot_tokens=f['token_shards'],target_train_G10_rows=1690280,target_train_G10_plus_fixeddev_G5_rows=1700520,estimated_output_bytes_linear_same_encoding=sum(x['bytes']for x in f['token_shards'])*1700520/92160,estimate_not_actual_extraction=True,final4096_read=False)))
