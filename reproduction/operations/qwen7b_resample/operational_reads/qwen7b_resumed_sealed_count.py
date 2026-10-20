from pathlib import Path
import json,time
b=Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020');out=[]
for i in range(2):
 rows=[json.loads(p.read_text())for p in sorted((b/'sampling'/('shard-%02d'%i)/'batches').glob('batch-*/complete.json'))]
 out.append(dict(shard=i,batches=len(rows),questions=sum(r['questions']for r in rows),trajectories=sum(r['trajectories']for r in rows),latest_complete_epoch=max(r['epoch']for r in rows)))
print(json.dumps(dict(epoch=time.time(),run_id=json.loads((b/'config.json').read_text())['run_id'],shards=out,questions=sum(r['questions']for r in out),trajectories=sum(r['trajectories']for r in out))))
