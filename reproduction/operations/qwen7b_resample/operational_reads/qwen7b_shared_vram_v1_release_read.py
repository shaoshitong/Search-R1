import json,time,subprocess
from pathlib import Path
unit='sst-q7b-g5-sharedvram-70188b8e50fd.service';pids=[1160466,1160508,1167933,1168251]
rows=[]
for i in range(2):
 show=subprocess.check_output(['systemctl','show',unit,'-p','MainPID','-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus','-p','ControlGroup'],text=True)
 cg={}
 paths=[Path('/sys/fs/cgroup/system.slice')/unit/'cgroup.procs']+list(Path('/sys/fs/cgroup').glob('*/system.slice/'+unit+'/cgroup.procs'))
 for p in paths:
  if p.exists():cg[str(p)]=p.read_text().strip().splitlines()
 r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=15)
 rows.append(dict(epoch=time.time(),unit=unit,unit_show=show,known_own_pids_existing={str(p):Path('/proc/'+str(p)).exists()for p in pids},own_cgroup_members=cg,gpu_query=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
 if not i:time.sleep(5)
print(json.dumps(dict(observations=rows,observed_known_own_pids=pids,foreign_tasks_signalled=False)))
