# ONE-SHOT CPU COMMAND TEMPLATE, NOT EXECUTED. Password arrives only in JSON stdin.
import hashlib,json,os,shlex,socket,subprocess,sys,tarfile,tempfile,time
from pathlib import Path
BASE='/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018'; TRAINING='8a5bac94b8230a99b81977f5290589e3d3708dfe'; OPERATOR='5a16192f8c8de43b6c1f122b77ae49c7a4144603'; RUN='dfe9c94084bb48b2a9bb7b60f9d882da'; JOB='730226'
HK=Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017'); CODE='/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730226-20261018/source/5a16192f8c8de43b6c1f122b77ae49c7a4144603'; OPERATOR_SHA='a503352c5113834be34735e3b8e7f8d0d071483d8edd6ebd7ae71ebd40166db9'
REMOTE_VERIFY='import sys,json,hashlib\nfrom pathlib import Path\ncode=\'/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730226-20261018/source/5a16192f8c8de43b6c1f122b77ae49c7a4144603\'\nbase=\'/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018\'\nstep=int(sys.argv[1])\noperator=Path(code)/"reproduction/easy2560_joint4/checkpoint_evidence.py"\nassert hashlib.sha256(operator.read_bytes()).hexdigest()==\'a503352c5113834be34735e3b8e7f8d0d071483d8edd6ebd7ae71ebd40166db9\'\nsys.path[:0]=[code,base+"/repo"]\nfrom reproduction.easy2560_joint4 import checkpoint_evidence as m\nassert Path(m.__file__)==operator\nresult=m.verify(base+"/preparation/checkpoint-evidence/global_step_"+str(step))\nprint(json.dumps(dict(operator_file=str(operator),operator_sha256=\'a503352c5113834be34735e3b8e7f8d0d071483d8edd6ebd7ae71ebd40166db9\',verification=result),sort_keys=True))'
step=int(sys.argv[1]); assert step in (5,50,100,150,200)
assert socket.gethostname()=='xielab3' and os.getuid()==1002
sys.path[:0]=[str(HK/'source'/TRAINING)]
from reproduction.prefix20_joint4.archive_supervisor import ssh_argv
password=json.load(sys.stdin).pop('password')
env=dict(os.environ,SSHPASS=password); password=None
source=BASE+'/preparation/checkpoint-evidence/global_step_'+str(step)
destination=HK/'checkpoint-evidence'/RUN/('global_step_'+str(step))
core=HK/'checkpoints'/RUN/('global_step_'+str(step))
def remote_verify():
 command=shlex.join([BASE+'/env/bin/python','-B','-c',REMOTE_VERIFY,str(step)])
 child=subprocess.run(ssh_argv(command),env=env,capture_output=True,text=True,timeout=180)
 if child.returncode: raise RuntimeError('SOURCE_SIDECAR_VERIFY_FAILED: '+child.stderr[-2000:])
 result=json.loads(child.stdout)
 assert result['operator_file']==CODE+'/reproduction/easy2560_joint4/checkpoint_evidence.py'
 assert result['operator_sha256']==OPERATOR_SHA
 v=result['verification']
 assert (v['source_base'],v['source_commit'],v['run_id'],v['source_job'],v['step'])==(BASE,TRAINING,RUN,JOB,step)
 assert v['status']=='EASY2560_CHECKPOINT_EVIDENCE_SEALED' and v['actual_cuda_restore_accepted'] is False
 assert v['post_callback_rng_captured'] is False and v['single_original_job_prefix'] is True
 rows=v['files_to_archive']; files={r['relative_path']:r for r in rows}
 assert len(files)==len(rows) and {'supplement.json','complete.json'}<=set(files)
 for name,row in files.items():
  relative=Path(name)
  assert not relative.is_absolute() and '..' not in relative.parts and str(relative)==name
  assert row['path']==source+'/'+name and type(row['bytes']) is int and 0<=row['bytes']<=128*1024**2
 return result,files
before,files=remote_verify()
accepted=json.loads((core/'source-acceptance.json').read_text())
assert (accepted['base'],accepted['source_commit'],accepted['run_id'],accepted['step'])==(BASE,TRAINING,RUN,step)
assert accepted['status']=='EASY2560_FULL_CHECKPOINT_SOURCE_ARCHIVE_SHA_PASS'
assert all(accepted[x] is True for x in ('fresh_source_after_archive_sha_verified','fresh_destination_sha_verified','destination_fsynced'))
assert accepted['actual_cuda_restore_accepted'] is False
assert not destination.exists() and not destination.is_symlink(), 'Independent sidecar destination exists; inspect, do not overwrite'
destination.parent.mkdir(parents=True,exist_ok=True)
assert destination.parent.resolve()==destination.parent
stage=Path(tempfile.mkdtemp(prefix='.'+destination.name+'-partial-',dir=destination.parent))
command=shlex.join(['tar','-C',source,'-cf','-','--',*sorted(files)])
child=subprocess.Popen(ssh_argv(command),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
try:
 seen=set()
 with tarfile.open(fileobj=child.stdout,mode='r|') as archive:
  for member in archive:
   name=member.name
   assert name in files and name not in seen and member.isfile() and member.size==files[name]['bytes']
   path=stage/name; path.parent.mkdir(parents=True,exist_ok=True)
   stream=archive.extractfile(member); h=hashlib.sha256(); count=0
   with path.open('xb') as output:
    for block in iter(lambda:stream.read(1024**2),b''):
     count+=len(block); h.update(block); output.write(block)
    output.flush(); os.fsync(output.fileno())
   assert count==files[name]['bytes'] and h.hexdigest()==files[name]['sha256']
   seen.add(name)
 child.stdout.close(); error=child.stderr.read(); rc=child.wait(timeout=30)
 assert rc==0 and seen==set(files), 'Exact source transfer failed: '+error.decode(errors='replace')[-1000:]
 after,second_files=remote_verify()
 assert {n:(r['bytes'],r['sha256']) for n,r in second_files.items()}=={n:(r['bytes'],r['sha256']) for n,r in files.items()}
 assert json.loads((stage/'identity/accepted-archive.json').read_text())==accepted
 for name,row in files.items():
  path=stage/name
  assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
 receipt=dict(status='EASY2560_INDEPENDENT_SMALL_SIDECAR_SHA_FSYNC_PASS',base=BASE,source_commit=TRAINING,
  operator_commit=OPERATOR,operator_source_sha256=OPERATOR_SHA,source_job=JOB,run_id=RUN,step=step,
  source_directory=source,destination=str(destination),file_count=len(files),total_bytes=sum(r['bytes'] for r in files.values()),
  files={n:dict(bytes=r['bytes'],sha256=r['sha256']) for n,r in sorted(files.items())},
  source_verify_before=before,source_verify_after=after,fresh_source_SHA_verified_twice=True,
  fresh_destination_SHA_verified=True,destination_fsynced=True,core_archive_unmodified=True,
  actual_cuda_restore_accepted=False,post_callback_rng_captured=False,credentials_persisted=False,checked_unix=time.time())
 with (stage/'sidecar-archive-receipt.json').open('x') as out:
  json.dump(receipt,out,sort_keys=True,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
 for folder,_,_ in os.walk(stage,topdown=False):
  fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 assert not destination.exists()
 os.rename(stage,destination)
 fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 print(json.dumps(dict(status=receipt['status'],directory=str(destination),file_count=receipt['file_count'],total_bytes=receipt['total_bytes'],
  receipt=str(destination/'sidecar-archive-receipt.json'),actual_cuda_restore_accepted=False,post_callback_rng_captured=False)))
except BaseException:
 if child.poll() is None:
  child.terminate(); child.wait(timeout=15)
 # Keep failed CPU staging for inspection; do not touch the accepted core archive or training.
 print('SIDECAR_FAILED_STAGE_RETAINED='+str(stage),file=sys.stderr)
 raise
