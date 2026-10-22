import json,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def read(p):return json.loads(Path(p).read_bytes())
f=A/'joint169028-native-training-bank-v1/groups.json';x=read(f);groups=x['train'];wanted={'["hotpotqa", "hotpotqa:train:70604"]','["hotpotqa", "hotpotqa:train:2094"]','["hotpotqa", "hotpotqa:train:81793"]'};groups=list(groups.values())if isinstance(groups,dict)else groups
out=dict(time=time.time(),original_top_keys=sorted(x),group_keys=sorted(groups[0]),selected_groups=[g for g in groups if g['qid']in wanted],bank_metadata={})
for name in('complete.json','source-resolution-contract.json','sources.json','bank.json'):
 p=f.parent/name
 if p.exists():
  z=read(p);out['bank_metadata'][name]=z if p.stat().st_size<25000 else dict(keys=sorted(z),bytes=p.stat().st_size)
print(json.dumps(out))
