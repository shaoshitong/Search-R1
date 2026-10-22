import hashlib,json,os,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021'
def ref(p):
 p=Path(p);s=p.stat();return dict(path=str(p),bytes=s.st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
roots=[p for p in A.iterdir() if p.is_dir() and any(word in p.name.lower()for word in ('confirmation','final','extraG5','validation'))]
listing=[]
for p in roots:
 rows=[]
 for current,dirs,files in os.walk(p):
  depth=len(Path(current).relative_to(p).parts)
  if depth>=3:dirs[:]=[]
  for name in files:
   q=Path(current)/name
   if q.suffix=='.json' or name.endswith('.jsonl.gz'):
    rows.append(dict(path=str(q),bytes=q.stat().st_size))
 listing.append(dict(root=str(p),files=rows[:250],truncated=len(rows)>250,total_metadata_files=len(rows)))
f=A/'training-extraG5-matched6528-20261021/final-exclusion-plan.json'
out=dict(status='FINAL4096_LOCATORS_ONLY_NO_LABEL_RESULTS',directories=listing,final_identity_plan=ref(f)if f.exists()else None,final_labels_or_predictions_read=False,model_tensor_checkpoint_read=False,epoch=time.time())
m=A/'confirmation/manifest.json'
if m.exists():
 c=json.loads(m.read_bytes());out['confirmation_manifest_reference']=ref(m);out['confirmation_manifest_metadata']={k:c[k]for k in ('status','source_sha256','source_revision','pools','score_or_generated_EM_observed','fit_or_tuning_allowed')if k in c}
print(json.dumps(out))
