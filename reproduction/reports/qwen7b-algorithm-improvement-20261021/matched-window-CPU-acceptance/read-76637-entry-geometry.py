from pathlib import Path
import json,time
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
b=A/'matched-window-terminal-CPU-acceptance-20261023-v1/code-76637b36e33a'
out=dict(time=time.time(),entries={p.name:p.read_text()for p in b.glob('*.sh')})
p=b/'accept_matched_window_checkpoint.py'
if p.is_file():out['validator_tail']='\n'.join(p.read_text().splitlines()[-24:])
out['accepted_metadata_files']=[str(p)for p in(b.parent).rglob('accepted.json')]
print(json.dumps(out))
