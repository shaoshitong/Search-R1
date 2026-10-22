from pathlib import Path
import json,time,hashlib
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');C=A/'selected-seed2-checkpoint-CPU-coordinator-20261023-v1';O=A/'seed2-matched-window-terminal-CPU-acceptance-20261023-v1';out=dict(time=time.time(),receipts={})
for suite,cell,e in [('online-packed-current64-seed19137-plan2026102302-L15P80-8192-20261023-v1','current64',5),('online-packed-matched-MSE-M5-seed19137-plan2026102302-L15P80-8192-20261023-v1','mse-M5',7),('online-packed-window2018-seed19137-plan2026102302-L15P80-8192-20261023-v1','window2018',10)]:
 p=O/suite/cell/('epoch-%02d'%e)/'accepted.json'
 if p.exists():
  raw=p.read_bytes();out['receipts'][cell]=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),accepted=json.loads(raw))
for n in('controller.json','progress.json','results-so-far.json','complete.json','failure.json'):
 if(C/n).exists():out[n]=json.loads((C/n).read_bytes())
ctrl=out.get('controller.json');p=Path('/proc/%d/stat'%ctrl['pid']);out['coordinator_identity_live']=p.exists()and p.read_text().rsplit(')',1)[1].split()[19]==ctrl['start_ticks'];out['failure_exists']=(C/'failure.json').exists();print(json.dumps(out))
