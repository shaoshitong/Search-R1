"""Unlink only the exact human-authorized, identity-bound old checkpoint plan."""
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import stat
import subprocess
import sys
import time

ROOT=Path('/data/user/jxu957/experiments')
ALLOWED_ROOTS={
 'dynrpo-prefix5-thread5-20261017','dynrpo-prefix5-thread5-20261017-r2',
 'dynrpo-prefix5-thread5-20261017-r3','dynrpo-prefix5-thread5-20261017-r4',
 'topd-20261008','cosearch-continue300-x203-20261008',
 'easy2560-corollout-joint4-micro8-after731063-20261018',
 'easy2560-corollout-joint4-micro16-after730080-20261018',
 'easy2560-corollout-joint4-recovery5-after730961-20261018',
 'easy2560-corollout-joint4-20261017',
 'easy2560-corollout-joint4-recovery5-after730912-20261018',
 'easy2560-corollout-joint4-recovery5-after730226-20261018',
}
TENSOR={'.pt','.pth','.safetensors','.bin','.ckpt'}


def allowed(path):
 rel=path.relative_to(ROOT)
 if rel.parts[0]not in ALLOWED_ROOTS or path.suffix.lower()not in TENSOR:return False
 # A step number alone is not a checkpoint: final trajectories may be grouped by step.
 if any(x in {'trajectory','trajectories','rollouts','sampling','features','feature','candidate-pools','mixed-candidate-pools','evaluations','eval','final-eval'}for x in rel.parts[1:-1]):return False
 if any(x in path.name.lower()for x in ('trajectory','rollout','feature','candidate','prediction')):return False
 parts=rel.parts[1:-1]
 explicit=any(x in {'checkpoint','checkpoints','training_state','full-state','resume-state','checkpoint-archives','rolling-checkpoints'}for x in parts)
 dynrpo=(rel.parts[0].startswith('dynrpo-prefix5-thread5-20261017')and len(parts)>=3 and parts[:2]==('actor','outers')and re.fullmatch(r'outer-[0-9]+(?:-attempt-[a-f0-9]+)?',parts[2])is not None)
 dynrpo_probe=(rel.parts[0].startswith('dynrpo-prefix5-thread5-20261017')and path.suffix=='.pt'and (
  (parts==('probe',)and (path.name=='latest.pt'or re.fullmatch(r'head-outer-[0-9]+\.pt',path.name)))or
  (len(parts)==2 and parts[0]=='outers'and re.fullmatch(r'outer-[0-9]+',parts[1])and path.name=='probe-state-at-completion.pt')or
  (len(parts)==3 and parts[0]=='outers'and re.fullmatch(r'outer-[0-9]+',parts[1])and parts[2]=='probe-update'and re.fullmatch(r'epoch-[0-9]+\.pt',path.name))or
  (parts==('preparation','probe-audit-pins')and re.fullmatch(r'outer-[0-9]+-[a-f0-9]+\.pt',path.name))or
  (parts==('reference-assets',)and path.name=='warmup-epoch5-ema.pt')))
 cosearch_final=(rel.parts[0]=='cosearch-continue300-x203-20261008'and len(parts)==4 and parts[:3]==('runtime','cosearch-final200-four-gpu-20261015','models')and parts[3]in('main','reranker')and re.fullmatch(r'model-[0-9]+-of-[0-9]+\.safetensors',path.name)is not None)
 return explicit or dynrpo or dynrpo_probe or cosearch_final


def identity(path):
 s=path.lstat()
 assert stat.S_ISREG(s.st_mode)and not path.is_symlink()and path.resolve()==path,str(path)
 return dict(bytes=s.st_size,allocated_bytes=s.st_blocks*512,device=s.st_dev,inode=s.st_ino,
             nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,uid=s.st_uid)


def check(row):
 path=Path(row['path']);assert allowed(path),str(path)
 seen=identity(path)
 for key in ('bytes','device','inode','mtime_ns','uid'):
  assert seen[key]==row[key],(str(path),key)
 return seen


def preserved(plan):
 checked=0
 for row in plan['preserved_text']+plan['preserved_binary']:
  path=Path(row['path'])
  if not os.access(path,os.R_OK):continue
  seen=identity(path)
  assert seen['device']==row['device']and seen['inode']==row['inode']and seen['bytes']>=row['bytes'],str(path)
  checked+=1
 return checked


def main():
 assert os.getuid()in(14090,21930)
 envelope=json.load(sys.stdin);raw=envelope['plan'];digest=hashlib.sha256(raw.encode()).hexdigest()
 assert digest==envelope['plan_sha256'];plan=json.loads(raw)
 assert plan['authorization']=='2026-10-21-human-delete-all-checkpoints-exact-old-experiments-preserve-text-and-eval-trajectories'
 assert set(plan['scope'])==ALLOWED_ROOTS
 root=Path('/tmp/authorized-old-checkpoints-20261021-'+str(os.getuid()));assert root.stat().st_uid==os.getuid()
 lock=(root/'apply.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not(root/'apply-claim.json').exists(),'Existing attempt must be audited, never replayed'
 rows=[row for row in plan['delete_files']if row['apply_uid']==os.getuid()]
 assert rows
 # Every target is frozen in the plan; check all identities before the first unlink.
 for row in rows:
  check(row);assert os.access(Path(row['path']).parent,os.W_OK),row['path']
 protected=preserved(plan)
 q=subprocess.run(['squeue','-h','-u','jxu957,mou176','-o','%i|%u|%T|%j'],capture_output=True,text=True,timeout=15)
 assert q.returncode==0,q.stderr
 old_jobs={line.split('|')[0]for line in plan['queue'].splitlines()}
 new_jobs={line.split('|')[0]for line in q.stdout.splitlines()}
 # Refresh every live job; only a reference to a concrete target blocks deletion.
 references=[]
 for job in sorted(new_jobs):
  paths=set()
  for command in (['scontrol','show','job','-o',job],['scontrol','write','batch_script',job,'-']):
   observed=subprocess.run(command,capture_output=True,text=True,timeout=15)
   assert observed.returncode==0,('live job reference query failed',job)
   paths.update(x.rstrip(';,)]}') for x in re.findall(r'/data/user/[^\s\"\x27<>$]+',observed.stdout))
  for name in list(paths):
   path=Path(name)
   if path.suffix in {'.json','.yaml','.yml','.sh'} and path.is_file() and path.stat().st_size<=1048576:
    paths.update(x.rstrip(';,)]}') for x in re.findall(r'/data/user/[^\s\"\x27<>$]+',path.read_text(errors='replace')))
  resolved=set(paths)
  for name in paths:
   path=Path(name)
   if path.exists() or path.is_symlink():resolved.add(str(path.resolve()))
  for row in rows:
   target=Path(row['path'])
   for name in resolved:
    ref=Path(name)
    if ref==ROOT:continue
    if target==ref or ref in target.parents:references.append(dict(job=job,path=str(target),reference=str(ref)))
 assert not references,('active target references',references)
 assert not plan['active_retired_checkpoint_references']
 with(root/'plan.json').open('x')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 with(root/'apply-claim.json').open('x')as stream:
  json.dump(dict(epoch=time.time(),pid=os.getpid(),uid=os.getuid(),plan_sha256=digest,queue=q.stdout),stream);stream.flush();os.fsync(stream.fileno())
 done=[];released=0
 try:
  with(root/'unlinked.jsonl').open('x')as ledger:
   for row in rows:
    seen=check(row);path=Path(row['path']);path.unlink()
    if seen['nlink']==1:released+=seen['allocated_bytes']
    entry=dict(path=str(path),before=seen,unlinked_unix=time.time());done.append(entry)
    ledger.write(json.dumps(entry)+'\n');ledger.flush();os.fsync(ledger.fileno())
  for parent in sorted({str(Path(row['path']).parent)for row in rows}):
   fd=os.open(parent,os.O_RDONLY)
   try:os.fsync(fd)
   finally:os.close(fd)
  assert all(not Path(row['path']).exists()for row in rows)
  after=preserved(plan)
  result=dict(status='AUTHORIZED_OLD_CHECKPOINT_PAYLOADS_REMOVED_TEXT_AND_TRAJECTORIES_RETAINED',epoch=time.time(),
              uid=os.getuid(),plan_sha256=digest,removed_files=len(done),confirmed_lastlink_released_bytes=released,
              preserved_text_verified_before=protected,preserved_text_verified_after=after,all_targets_absent=True,
              feature7b_modified=False,current7b_modified=False,qprobe_modified=False,remote_archives_modified=False,
              training_stopped_or_restarted=False)
  with(root/'result.json').open('x')as stream:json.dump(result,stream);stream.flush();os.fsync(stream.fileno())
  print(json.dumps(result))
 except BaseException as error:
  with(root/'failure.json').open('x')as stream:json.dump(dict(epoch=time.time(),error_type=type(error).__name__,error=str(error),removed_files=len(done),confirmed_lastlink_released_bytes=released),stream);stream.flush();os.fsync(stream.fileno())
  raise

if __name__=='__main__':main()
