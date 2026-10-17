import json,os,socket,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);password=p.pop('password');source=Path(p['source_root']);root=Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/archive-controller/job-731063')
assert socket.gethostname()=='xielab3' and os.getuid()==1002
assert p['job']=='731063' and p['source_commit']=='a61b48139386b346e4fef12511ece1ac0d09a8e3'
assert p['base']=='/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730961-20261018'
assert source==Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/source')/p['source_commit']
assert json.loads((source/'source-publication.json').read_text())['source_commit']==p['source_commit']
assert p['run_id']=='dfe9c94084bb48b2a9bb7b60f9d882da'
root.mkdir(parents=True,exist_ok=True)
assert not any((root/n).exists() for n in ('launch-claim.json','claim.json','controller.stdout.log','result.json','failure.json')),'Inspect existing731063 archive claim; duplicate launch forbidden'
claim=dict(base=p['base'],source_commit=p['source_commit'],source_job=p['job'],run_id=p['run_id'],credentials_persisted=False,requested_steps=[50,100,150,200],preserve_existing_checkpoint5=True,created_unix=time.time())
with (root/'launch-claim.json').open('x') as f:json.dump(claim,f,indent=2);f.flush();os.fsync(f.fileno())
env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
command=[sys.executable,'-B','-m','reproduction.easy2560_joint4.archive_controller','--base',p['base'],'--commit',p['source_commit'],'--job',p['job'],'--first-step','50']
with (root/'controller.stdout.log').open('xb') as output:
 child=subprocess.Popen(command,cwd=str(source),env=env,stdin=subprocess.PIPE,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
 child.stdin.write(json.dumps(dict(password=password)).encode());child.stdin.close()
password=None
fields=Path('/proc/%d/stat'%child.pid).read_text().rsplit(')',1)[1].split()
birth=dict(pid=child.pid,start_ticks=int(fields[19]),state=fields[0],uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
value=dict(status='NEW731063_ARCHIVE_CONTROLLER_LAUNCHED_ONCE',root=str(root),source_root=str(source),command=command,birth=birth,base=p['base'],run_id=p['run_id'],source_commit=p['source_commit'],source_job=p['job'],credentials_persisted=False,requested_steps=[50,100,150,200],preserve_existing_checkpoint5=True,checked_unix=time.time())
with (root/'launch-receipt.json').open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
fd=os.open(root,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(value))
