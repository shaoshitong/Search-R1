from pathlib import Path
import json,time
b=Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020');a=sorted((b/'pool/jobs/job-shared5-1259750').glob('attempt-*'))[-1]
v={'epoch':time.time(),'profiles':{},'progress':[]}
for p in (a/'sampler-launches').glob('launch-*/kv-profile/rank-*.json'):v['profiles'][str(p)]=json.loads(p.read_text())
for p in (b/'sampling').glob('shard-*/batches/batch-*/outers/outer-000001/full/rank-0-progress.json'):
 if p.stat().st_mtime>1792519824:v['progress'].append({'path':str(p),'value':json.loads(p.read_text())})
print(json.dumps(v))
