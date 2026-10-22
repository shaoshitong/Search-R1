import gzip,json,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-policy80-ownG10-20261023-v1')
def read(p):return json.loads(Path(p).read_bytes())
c=read(B/'input-contract.json');groups=read(B/'fixture/inputs/train-groups.json');p=read(B/'progress.json');start=p['source_files_completed'];out=dict(time=time.time(),source_index=start,source=c['RAW_files'][start],mismatches=[],final4096_read=False)
lookup={t:(q,y)for q,g in groups.items()for t,y in zip(g['trajectory_ids'],g['labels'])}
for si in range(start,min(start+8,len(c['RAW_files']))):
 with gzip.open(c['RAW_files'][si]['path'],'rt')as f:
  for i,line in enumerate(f):
   r=json.loads(line);tid=r['trajectory_id']
   if tid not in lookup:continue
   q,y=lookup[tid];exp=dict(qid=q,question_hash=groups[q]['question_hash'],actor_version='official7b-a09a35458c702b33eeacc393d103063234e8bc28',strict_em=y);bad={k:dict(actual=r.get(k),expected=v)for k,v in exp.items()if r.get(k)!=v}
   if bad:
    out['mismatches'].append(dict(source_index=si,source=c['RAW_files'][si],line=i,trajectory_id=tid,canonical_qid=q,differing_fields=bad))
    if len(out['mismatches'])>=3:break
 if len(out['mismatches'])>=3:break
print(json.dumps(out))
