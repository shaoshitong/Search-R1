import json,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-policy80-physical-ownG10-20261023-v2')
def read(p):return json.loads(Path(p).read_bytes())
out=dict(time=time.time(),files={},actual_owned_process_live=False,no_GPU=True,Slurm_changed=False,final4096_read=False)
for name in('P20-crosscheck-process.json','P20-crosscheck-controller.json','P20-crosscheck-progress.json','P20-crosscheck-failure.json','P20-crosscheck-complete.json','P20-crosscheck-mismatch.json','fixture/accepted.json'):
 if(B/name).is_file():out['files'][name]=read(B/name)
r=out['files'].get('P20-crosscheck-process.json')
if r:
 p=Path('/proc/%d'%r['pid'])
 if p.exists():
  live=(p/'stat').read_text().rsplit(')',1)[1].split()[19]==r['start_ticks']and b'crosscheck_full169028_policy80_P20.py'in(p/'cmdline').read_bytes();out['actual_owned_process_live']=live
  if live:out['process_resources']={line.split(':',1)[0]:line.split(':',1)[1].strip()for line in(p/'status').read_text().splitlines()if line.startswith(('VmRSS:','VmPeak:','Threads:'))}
print(json.dumps(out))
