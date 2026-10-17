from pathlib import Path
import json,hashlib,re,time
b=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015');p=b/'data/final.parquet'
if not p.exists():print(json.dumps(dict(path=str(p),exists=False)));raise SystemExit(0)
import pandas as pd
raw=p.read_bytes();m=json.loads((b/'logs/run-manifest.json').read_text());rows=pd.read_parquet(p,columns=['prompt','data_source']);ids=[]
for chat in rows['prompt']:
 if isinstance(chat,str):chat=json.loads(chat)
 text=chat[0]['content'];matches=list(re.finditer(r'Question:\s*',text));assert matches
 q=re.sub(r'\s+',' ',text[matches[-1].end():]).strip().casefold();ids.append(hashlib.sha256(q.encode()).hexdigest())
ledger=next(json.loads(l) for l in (b/'logs/validation-identities.jsonl').open() if len(json.loads(l).get('question_ids',[]))==51713)
assert ids==ledger['question_ids'] and rows['data_source'].tolist()==ledger['data_sources']
assert hashlib.sha256(raw).hexdigest()==m['input_sha256']['final.parquet']==ledger['validation_sha256']
print(json.dumps(dict(epoch=time.time(),ordered_ids_and_sources_equal=True,canonical_rows=len(ids),qid_duplicates_are_canonical=True,unique_source_scoped=len(set(zip(rows['data_source'],ids))),data_sha256=hashlib.sha256(raw).hexdigest())))
