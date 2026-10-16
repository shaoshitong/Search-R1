import json,hashlib,time
from pathlib import Path
p=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015/logs/validation-identities.jsonl')
out=[]
for l in p.open():
 d=json.loads(l);ids=d.get('question_ids',[]);sources=d.get('data_sources',[])
 if len(ids)==51713:out.append(dict(row_sha256=hashlib.sha256(l.encode()).hexdigest(),run_id=d['run_id'],step=d['step'],rows=len(ids),unique_source_scoped=len(set(zip(sources,ids))),validation_sha256=d.get('validation_sha256'),final=d.get('final'),split=d.get('split'),correct=d['correct'],em=d['weighted_em'],f1=d['weighted_f1']))
print(json.dumps(dict(epoch=time.time(),final=out)))
