import hashlib,json,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021'
def ref(p):
 raw=Path(p).read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
p=A/'training-extraG5-matched6528-20261021/final-exclusion-plan.json';v=json.loads(p.read_bytes());rows=v.get('questions',v.get('groups'))
out=dict(final_identity_reference=ref(p),top_level_keys=sorted(v),questions=len(rows),first_row_schema=sorted(rows[0]),final_identity_source_refs={k:x for k,x in v.items()if k not in('questions','groups')and(isinstance(x,(str,int,bool))or k.endswith('ref'))},final_qids_sha256=hashlib.sha256(json.dumps([r['qid']for r in rows],separators=(',',':')).encode()).hexdigest(),final_labels_read=False)
out['B_root_names']=[x.name for x in B.iterdir()if x.is_dir()]
out['extraG5_top_files']=[dict(path=str(x),bytes=x.stat().st_size)for x in p.parent.iterdir()if x.is_file()and x.suffix=='.json']
out['confirmation_top_files']=[ref(x)for x in (A/'confirmation').iterdir()if x.is_file()and x.suffix=='.json']
print(json.dumps(out))
