from pathlib import Path
import json,time
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');out=A/'fullZ-seed2-749935-terminal-CPU-coordinator-20261023-v1';r=dict(time=time.time(),Slurm_mutated=False,CP_or_tensor_read=False,final4096_read=False)
for name in('launch.json','controller.json','progress.json','active-child.json','complete.json','failure.json'):
 if(out/name).is_file():r[name]=json.loads((out/name).read_bytes())
c=r['controller.json'];p=Path('/proc/%d'%c['pid']);r['actual_identity_live']=p.exists()and(p/'stat').read_text().rsplit(')',1)[1].split()[19]==c['start_ticks']
if r['actual_identity_live']:
 r['actual_argv']=[x.decode()for x in(p/'cmdline').read_bytes().split(bytes([0]))if x];r['open_fds']=[]
 for f in(p/'fd').iterdir():
  try:r['open_fds'].append(str(f.resolve()))
  except(FileNotFoundError,PermissionError):pass
 common=str(A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1/acceptance.lock');r['shared_CPU_lock_FD_open_by_coordinator']=common in r['open_fds'];assert not r['shared_CPU_lock_FD_open_by_coordinator']
print(json.dumps(r))
