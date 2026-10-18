"""Run the bound full CPU audit inside the existing four-GPU allocation.

Own CPU-only Slurm step, no new job/GPU allocation and no training control.
The detached srun PID, script SHA, lock and durable proof prevent blind retry
after a control timeout. A stale intent requires explicit reconciliation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from reproduction.hpc3_prefix5.control import OUTPUT, PYTHON
from reproduction.hpc3_prefix5.direct_control import run

MEMORY_PROBE = r'''
import os,json,socket
from pathlib import Path
rows=[]
for line in Path('/proc/self/cgroup').read_text().splitlines():
 _,controllers,group=line.split(':',2)
 if 'memory' in controllers.split(','):root=Path('/sys/fs/cgroup/memory');leaf=root/group.lstrip('/');a='memory.limit_in_bytes';b='memory.usage_in_bytes'
 elif not controllers:root=Path('/sys/fs/cgroup');leaf=root/group.lstrip('/');a='memory.max';b='memory.current'
 else:continue
 p=leaf
 while p==root or root in p.parents:
  if (p/a).exists() and (p/b).exists():
   limit=(p/a).read_text().strip();used=int((p/b).read_text())
   if limit!='max' and int(limit)<2**60:rows.append(dict(path=str(p),limit_bytes=int(limit),used_bytes=used,headroom_bytes=int(limit)-used))
  if p==root:break
  p=p.parent
assert rows,'Require a finite enclosing cgroup memory bound'
assert os.environ.get('SLURM_JOB_ID')=='733898' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
print(json.dumps(dict(status='OWN_ALLOCATION_CPU_PREFLIGHT_PASS',host=socket.gethostname(),job='733898',step=os.environ.get('SLURM_STEP_ID'),
 cuda_visible_devices='',memory_ancestors=rows,minimum_headroom_bytes=min(r['headroom_bytes'] for r in rows))))
'''

QUERY = r'''
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
b=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4');n=TARGET_OUTER;mode=MODE_VALUE
def rd(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert rd(b/'preparation/source-publication.json')['commit']=='2000d46a530c64d78be7802bf11175a68b65effe'
assert rd(b/'preparation/slurm-submission.json')['job']=='733898'
script=Path(OP_PATH);assert b/'preparation/operators' in script.parents and sha(script)==OP_SHA
proof=b/'preparation'/('live-milestone-%06d-v1-full-sha-cpu.json'%n)
controller=proof.with_name(proof.name+'.launcher.json');log=proof.with_name(proof.name+'.launcher.log')
def ticks(pid):
 try:
  p=Path('/proc')/str(pid);stat=(p/'stat').read_text();return dict(start_ticks=stat[stat.rfind(')')+2:].split()[19],uid=p.stat().st_uid,cmd=(p/'cmdline').read_bytes().split(b'\0'))
 except FileNotFoundError:return None
def live(c):
 actual=ticks(c['launcher_pid'])
 return actual is not None and actual['start_ticks']==c['start_ticks'] and actual['uid']==c['uid'] and [a.decode() for a in actual['cmd'] if a]==c['command']
def output(v):print(json.dumps(v));raise SystemExit(0)
def status():
 c=rd(controller) if controller.exists() else None
 v=dict(outer=n,checked_unix=time.time(),controller=c,launcher_live=live(c) if c else False,
  progress=rd(proof.with_name(proof.name+'.progress.json')) if proof.with_name(proof.name+'.progress.json').exists() else None)
 if proof.exists():
  p=rd(proof);assert p['status']=='LIVE_IMMUTABLE_MILESTONE_FULL_SHA_CPU_READBACK_PASS'
  assert p['operator_script_sha256']==OP_SHA and p['source_joint_sha256']==sha(b/'outers'/('outer-%06d'%n)/'complete.json')
  for seal in p['verified_files']:
   s=Path(seal['path']).stat();assert (s.st_ino,s.st_size,s.st_mtime_ns,int(os.statvfs(seal['path']).f_fsid))==(seal['inode'],seal['bytes'],seal['mtime_ns'],seal['shared_filesystem_fsid'])
  v.update(status='EXISTING_SEALED_FULL_CPU_PROOF',proof=str(proof),sha256=sha(proof),actual_adam=p['actor_actual_adam_steps_per_rank'])
 else:
  v['status']='CPU_AUDIT_LAUNCHER_RUNNING' if v['launcher_live'] else 'CPU_AUDIT_NOT_RUNNING_PROOF_ABSENT'
  if log.exists():
   with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2000));v['log_tail']=f.read().decode(errors='replace')
 return v
if mode=='status' or proof.exists() or controller.exists():output(status())
prefix=['srun','--jobid=733898','--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=12G','--gres=gpu:0','--kill-on-bad-exit=1','/usr/bin/env','CUDA_VISIBLE_DEVICES=',AUDIT_PYTHON]
if mode=='launch' and not (b/'outers'/('outer-%06d'%n)/'complete.json').exists():output(dict(status='PENDING_ACTUAL_MILESTONE_JOINT',outer=n))
running=subprocess.check_output(['squeue','-h','-j','733898','-o','%T'],text=True).strip()
assert running=='RUNNING','Use only the existing live own allocation'
with proof.with_name(proof.name+'.launcher.lock').open('a+') as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:output(dict(status='EXISTING_CPU_LAUNCH_REQUEST',outer=n))
 if proof.exists() or controller.exists():output(status())
 if mode=='launch' and (proof.with_name(proof.name+'.intent.json').exists() or log.exists()):output(dict(status='RECONCILE_EXISTING_AUDIT_INTENT_NO_DUPLICATE_LAUNCH',outer=n))
 p=subprocess.run(prefix+['-c',MEMORY_CODE],text=True,capture_output=True,timeout=30)
 assert p.returncode==0,p.stderr[-1500:]
 memory=next(json.loads(line) for line in reversed(p.stdout.splitlines()) if line.lstrip().startswith('{'))
 if mode=='preflight':output(dict(status='OWN_ALLOCATION_CPU_PREFLIGHT_PASS',outer=n,preflight=memory,new_GPU_allocations=0))
 if memory['minimum_headroom_bytes']<8*1024**3:output(dict(status='WAITING_OWN_CPU_MEMORY_HEADROOM',outer=n,preflight=memory))
 command=prefix+[str(script),'--outer',str(n)]
 with log.open('xb') as handle:
  child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
  identity=ticks(child.pid)
  c=dict(outer=n,job='733898',created_unix=time.time(),launcher_pid=child.pid,start_ticks=identity['start_ticks'] if identity else None,
   uid=os.getuid(),command=command,operator_sha256=OP_SHA,source_joint_sha256=sha(b/'outers'/('outer-%06d'%n)/'complete.json'),
   proof=str(proof),log=str(log),preflight=memory,extra_GPU_allocations=0)
  with controller.open('x') as f:json.dump(c,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 output(dict(status='OWN_CPU_READBACK_DETACHED_LAUNCH_NOT_ACCEPTANCE',outer=n,controller=c,launcher_live=live(c)))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', type=int, choices=(100, 150, 200), required=True)
    parser.add_argument('--mode', choices=('preflight', 'launch', 'status'), required=True)
    args = parser.parse_args()
    registry = json.loads((OUTPUT/'two-stage-r4-independent-audit-operators.json').read_text())
    operator = next(x for x in registry['files'] if x['name']=='hpc3-two-stage-milestone-readback-20261018.py')
    assert operator['sha256']==hashlib.sha256(Path(__file__).with_name(operator['name']).read_bytes()).hexdigest()
    code = QUERY.replace('TARGET_OUTER',str(args.outer)).replace('MODE_VALUE',repr(args.mode)).replace('OP_PATH',repr(operator['path'])).replace('OP_SHA',repr(operator['sha256'])).replace('AUDIT_PYTHON',repr(PYTHON)).replace('MEMORY_CODE',repr(MEMORY_PROBE))
    address = subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip()
    result = run(PYTHON+' -c '+shlex.quote(code),account='mou176',source_address=address,timeout=45)
    if result.returncode:
        raise RuntimeError('Own CPU audit control failed; inspect exact existing PID/progress before retry: '+result.stdout.decode()[-1000:]+result.stderr.decode()[-1000:])
    value = next(json.loads(line) for line in reversed(result.stdout.decode().splitlines()) if line.lstrip().startswith('{'))
    (OUTPUT/('two-stage-milestone-%06d-cpu-%s.json'%(args.outer,args.mode))).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':
    main()
