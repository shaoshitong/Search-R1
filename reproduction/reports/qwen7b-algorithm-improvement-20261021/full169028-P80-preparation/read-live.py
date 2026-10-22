"""Read-only actual process/lock/progress metadata of owned CPU preparation."""
import hashlib,json,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-policy80-ownG10-20261023-v1')
def read(p):return json.loads(Path(p).read_bytes())
out=dict(time=time.time(),base=str(B),final4096_read=False,Slurm_changed=False,files={})
for name in('prepare-process.json','controller.json','progress.json','failure.json','complete.json','fixture/accepted.json'):
 if(B/name).is_file():out['files'][name]=read(B/name)
ref=out['files'].get('prepare-process.json');live=False
if ref:
 p=Path('/proc/%d'%ref['pid'])
 if p.exists():live=(p/'stat').read_text().rsplit(')',1)[1].split()[19]==ref['start_ticks']and b'prepare_full169028_policy80.py'in(p/'cmdline').read_bytes()
out['actual_owned_process_live']=live
if'fixture/accepted.json'in out['files']:
 a=out['files']['fixture/accepted.json'];f=B/'fixture/fixture-complete.json';raw=f.read_bytes();assert a['fixture']==dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
if(B/'prepare.log').exists():
 with(B/'prepare.log').open('rb')as f:f.seek(max(0,(B/'prepare.log').stat().st_size-1800));out['log_tail']=f.read().decode(errors='replace')
print(json.dumps(out))
