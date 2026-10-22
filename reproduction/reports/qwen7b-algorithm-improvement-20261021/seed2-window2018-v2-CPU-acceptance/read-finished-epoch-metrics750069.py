from pathlib import Path
import json,time,hashlib
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');b=A/'online-packed-window2018-seed19137-plan2026102302-L15P80-8192-20261023-v2/window2018';out=dict(time=time.time(),epochs={},model_or_checkpoint_read=False,final4096_read=False)
for epoch in range(11):
 p=b/('evaluation-epoch-%02d.json'%epoch)
 if not p.is_file():continue
 raw=p.read_bytes();ev=json.loads(raw);assert not ev['final4096_scored'];out['epochs'][str(epoch)]=dict(epoch=epoch,cursor=ev['cursor'],actual_Adam_updates=ev['actual_Adam_updates'],EMA_updates=ev['EMA_updates'],metrics=ev['metrics'],evaluation=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
print(json.dumps(out))
