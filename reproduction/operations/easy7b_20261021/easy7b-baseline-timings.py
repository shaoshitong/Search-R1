import re,json,time
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-micro8-after731063-20261018');d=dict(epoch=time.time(),base=str(b),role='historical stopped3B baseline; not7B performance')
p=b/'logs/search-g1-common8-training.log'
if p.is_file():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-1500000));s=f.read().decode(errors='replace')
 s=re.sub(r'\x1b\[[0-9;]*m','',s)
 lines=[line for line in s.splitlines()if 'timing_s/' in line or 'timing_per_token_ms/' in line]
 d['metric_lines']=lines[-3:]
root=b/'artifacts/mixed-selection'
if root.is_dir():
 paths=sorted(root.glob('*.json'),key=lambda p:p.stat().st_mtime)[-3:]; d['selection_timing']=[]
 for p in paths:
  x=json.loads(p.read_text()); z={k:v for k,v in x.items() if type(v)in(int,float,bool,str)and ('time' in k or 'seconds' in k or k in ['step','selected_questions','selected_mixed','mixed_count'])};d['selection_timing'].append(dict(name=p.name,values=z))
print(json.dumps(d))
