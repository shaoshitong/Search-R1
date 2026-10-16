from pathlib import Path
import os,sys,json,shlex,time
sys.path.insert(0,'work/tree-speed-repair-20261008');import ssh_retained18 as ssh
os.environ['TREE_MGMT_CONTROL_PATH']='/tmp/tree-mgmt-3fw9a1ab/control'
b=Path('outputs/tree-nested-branch-handoff-20261013');attempt=b/'FINAL200_COMPLETED_CPU_AUDIT_CLEANUP_ATTEMPT.json';assert not attempt.exists();attempt.write_text(json.dumps(dict(epoch=time.time(),scope='Only two completed own CPU audit units; no evaluation/training stop',authorized_by='Active user goal requires own resource cleanup; CPU and trace actual exit0 receipts retained',units=['sst-tree-nativehf200-cpu-caac340-778b2b43.service','sst-tree-trace200-durable-repair001.service']))+'\n')
src=r'''import json,time,subprocess,os
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:778b2b43dad90fc0c1c5d95d77c2326286011b73/output/final-eval-fast')
p=r/'completed-cpu-audit-cleanup-20261016.json'
assert not p.exists()
units={'sst-tree-nativehf200-cpu-caac340-778b2b43.service':'510e864008f548fcbefcc7143d7f5976','sst-tree-trace200-durable-repair001.service':'94fd3b56080743aa884d19e9fefb40e6'}
def read(u):
 a=subprocess.run(['systemctl','show',u,'--property=LoadState,ActiveState,SubState,MainPID,InvocationID,Result,ExecMainStatus,ControlGroup,CPUUsageNSec,IOReadBytes,IOWriteBytes'],capture_output=True,text=True)
 values=dict(line.split('=',1) for line in a.stdout.splitlines() if '=' in line);cg=values.get('ControlGroup');q=Path('/sys/fs/cgroup')/cg.lstrip('/')/'cgroup.procs' if cg else None
 return dict(epoch=time.time(),returncode=a.returncode,raw=a.stdout,properties=values,own_cgroup_pids=q.read_text().split() if q and q.is_file() else [])
before={u:read(u) for u in units}
for u,v in before.items():
 d=v['properties'];assert d.get('MainPID') in ('0',None) and not v['own_cgroup_pids']
 if d.get('ActiveState')=='active':assert d.get('SubState')=='exited' and d['InvocationID']==units[u] and d['ExecMainStatus']=='0' and d['Result']=='success'
 else:assert d.get('ActiveState')=='inactive' and d.get('LoadState') in ('loaded','not-found')
x=dict(schema='tree_completed_cpu_audit_cleanup_v1',started_epoch=time.time(),status='INTENT',before=before,stop_results={},post_stop_readbacks=[],training_or_evaluation_signals_sent=False,foreign_process_signals_sent=False)
with p.open('x') as f:json.dump(x,f);f.flush();os.fsync(f.fileno())
for u,v in before.items():
 if v['properties'].get('ActiveState')=='active':
  a=subprocess.run(['systemctl','stop',u],capture_output=True,text=True,timeout=15);x['stop_results'][u]=dict(returncode=a.returncode,stderr=a.stderr);assert a.returncode==0
 else:x['stop_results'][u]=dict(not_needed_already_inactive=True)
for _ in range(2):
 frame={u:read(u) for u in units}
 for v in frame.values():assert v['properties'].get('MainPID') in ('0',None) and v['properties'].get('ActiveState')=='inactive' and not v['own_cgroup_pids']
 x['post_stop_readbacks'].append(frame);time.sleep(.5)
x.update(status='COMPLETE',end_epoch=time.time())
tmp=p.with_suffix('.complete.tmp');tmp.write_text(json.dumps(x));os.replace(tmp,p)
print(json.dumps(x))'''
a=ssh.run('python3 -B -c '+shlex.quote(src),timeout=60);x=dict(returncode=a.returncode,stderr=a.stderr,local_epoch=time.time())
if a.returncode==0:x['remote']=json.loads(a.stdout)
else:x['stdout']=a.stdout
p=b/'FINAL200_COMPLETED_CPU_AUDIT_CLEANUP_RECEIPT.json';assert not p.exists();p.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(dict(returncode=a.returncode,status=x.get('remote',{}).get('status'),stop_results=x.get('remote',{}).get('stop_results'))))
