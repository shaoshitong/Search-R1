import json,os,socket,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);password=p.pop('password');source=Path(p['source_root']);root=Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/archive-controller/job-730226')
assert socket.gethostname()=='xielab3' and os.getuid()==1002
assert p['job']=='730226' and p['source_commit']=='8a5bac94b8230a99b81977f5290589e3d3708dfe'
assert p['base']=='/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018'
assert source==Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/source')/p['source_commit']
assert json.loads((source/'source-publication.json').read_text())['source_commit']==p['source_commit']
assert len(p['run_id'])==32 and all(c in '0123456789abcdef' for c in p['run_id'])
root.mkdir(parents=True,exist_ok=True)
assert not any((root/n).exists() for n in ('launch-claim.json','claim.json','controller.stdout.log','result.json','failure.json')),'Inspect existing730226 archive claim; duplicate launch forbidden'
claim=dict(base=p['base'],source_commit=p['source_commit'],source_job=p['job'],run_id=p['run_id'],credentials_persisted=False,created_unix=time.time())
with (root/'launch-claim.json').open('x') as f:json.dump(claim,f,indent=2);f.flush();os.fsync(f.fileno())
env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
command=[sys.executable,'-B','-m','reproduction.easy2560_joint4.archive_controller','--base',p['base'],'--commit',p['source_commit'],'--job',p['job']]
with (root/'controller.stdout.log').open('xb') as output:
 child=subprocess.Popen(command,cwd=str(source),env=env,stdin=subprocess.PIPE,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
 child.stdin.write(json.dumps(dict(password=password)).encode());child.stdin.close()
password=None
fields=Path('/proc/%d/stat'%child.pid).read_text().rsplit(')',1)[1].split()
birth=dict(pid=child.pid,start_ticks=int(fields[19]),state=fields[0],uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
value=dict(status='NEW730226_ARCHIVE_CONTROLLER_LAUNCHED_ONCE',root=str(root),source_root=str(source),command=command,birth=birth,base=p['base'],run_id=p['run_id'],source_commit=p['source_commit'],source_job=p['job'],credentials_persisted=False,checked_unix=time.time())
with (root/'launch-receipt.json').open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
fd=os.open(root,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(value))
