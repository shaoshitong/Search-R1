import json,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-L15P80-MSE-metadata-observer-20261023-v1')
def read(p):return json.loads(Path(p).read_bytes())
out=dict(time=time.time(),files={},actual_observer_identity_live=False,Slurm_mutations=False,checkpoint_tensors_loaded=False,final4096_read=False)
for n in('observer-process.json','controller.json','latest.json','phase-changes.json','startup-proof-seed1.json','startup-proof-seed2.json','failure-seed1.json','failure-seed2.json','metadata-read-error-seed1.json','metadata-read-error-seed2.json','observer-failure.json','complete.json','deadline.json'):
 if(B/n).is_file():out['files'][n]=read(B/n)
r=out['files'].get('observer-process.json')
if r:
 p=Path('/proc/%d'%r['pid'])
 if p.exists():out['actual_observer_identity_live']=(p/'stat').read_text().rsplit(')',1)[1].split()[19]==r['start_ticks']and b'observe_full169028_MSE_metadata.py'in(p/'cmdline').read_bytes()
print(json.dumps(out))
