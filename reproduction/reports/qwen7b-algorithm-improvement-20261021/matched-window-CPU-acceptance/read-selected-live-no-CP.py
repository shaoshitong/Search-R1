"""Resolve unknown SSH timeout outcome using only processes and CPU receipts."""
from pathlib import Path
import os,json,time
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
out=dict(time=time.time(),processes=[],cells={},checkpoint_opened=False,Slurm_mutated=False)
for p in Path('/proc').glob('[0-9]*'):
 try:
  if p.stat().st_uid!=os.getuid():continue
  raw=(p/'cmdline').read_bytes()
  if b'accept_matched_window_checkpoint.py' not in raw or b' -c 'in raw:continue
  stat=(p/'stat').read_text().rsplit(')',1)[1].split();out['processes'].append(dict(pid=int(p.name),state=stat[0],start_ticks=stat[19],command=raw.replace(b'\0',b' ').decode()))
 except(OSError,UnicodeError):pass
b=A/'matched-window-terminal-CPU-acceptance-20261023-v1'
for cell,epoch in(('current64',5),('mse-M5',7),('window2018',10)):
 root=b/cell/('epoch-%02d'%epoch);v={}
 for name in('controller.json','progress.json','accepted.json','run-complete.json','failure.json'):
  if(root/name).is_file():v[name]=json.loads((root/name).read_bytes())
 out['cells'][cell+'-E%d'%epoch]=v
out['entry_files']=[str(p)for p in(b/'code-76637b36e33a').glob('*.sh')]
print(json.dumps(out))
