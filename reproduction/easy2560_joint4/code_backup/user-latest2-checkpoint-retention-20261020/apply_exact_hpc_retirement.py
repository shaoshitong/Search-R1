import pathlib,json,hashlib,os,stat,time
root=pathlib.Path("/tmp/easy-authorized-latest2-eebf738cb45d434cb9cebb167145cf6f");raw=(root/'plan.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=="2282bd9a2c8d6be8368fa1042c9d24450ba55346da7b39d21be30f649e9114e4";plan=json.loads(raw)
assert os.getuid()==21930 and plan['execution_performed']==False and len(plan['delete_files'])==52
def check(x):
 p=pathlib.Path(x['path']);assert p.resolve()==p;s=p.lstat();assert stat.S_ISREG(s.st_mode)
 assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(x['device'],x['inode'],x['bytes'],x['mtime_ns']),str(p)
 return s
for x in plan['preserved_payload_members']:check(x)
allowed=[pathlib.Path(p) for g in plan['groups'] for p in g['delete_checkpoint_dirs']]
for x in plan['delete_files']:
 p=pathlib.Path(x['path']);assert p.suffix in ('.pt','.safetensors','.bin') and any(a in p.parents for a in allowed);s=check(x);assert s.st_nlink==x['nlink'] and os.access(str(p.parent),os.W_OK)
assert not (root/'apply-claim.json').exists() and not (root/'result.json').exists()
with (root/'apply-claim.json').open('x') as f:json.dump(dict(started_unix=time.time(),plan_sha256="2282bd9a2c8d6be8368fa1042c9d24450ba55346da7b39d21be30f649e9114e4",pid=os.getpid(),uid=os.getuid()),f);f.flush();os.fsync(f.fileno())
deleted=[];lastlink_bytes=0
try:
 with (root/'unlinked.jsonl').open('x') as log:
  for x in plan['delete_files']:
   p=pathlib.Path(x['path']);s=check(x);p.unlink();rec=dict(path=str(p),unlinked_unix=time.time(),before_nlink=s.st_nlink,allocated_bytes=s.st_blocks*512,inode=s.st_ino,device=s.st_dev);deleted.append(rec)
   if s.st_nlink==1:lastlink_bytes+=s.st_blocks*512
   log.write(json.dumps(rec,sort_keys=True)+'\n');log.flush();os.fsync(log.fileno())
 for parent in sorted({str(pathlib.Path(x['path']).parent) for x in plan['delete_files']}):
  fd=os.open(parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 assert all(not pathlib.Path(x['path']).exists() for x in plan['delete_files'])
 for x in plan['preserved_payload_members']:check(x)
 result=dict(status='AUTHORIZED_OLD_CHECKPOINT_TENSORS_RETIRED_LATEST_TWO_PRESERVED',finished_unix=time.time(),plan_sha256="2282bd9a2c8d6be8368fa1042c9d24450ba55346da7b39d21be30f649e9114e4",deleted_file_count=len(deleted),confirmed_last_link_unlinked_allocated_bytes=lastlink_bytes,expected_reclaimable_allocated_bytes=plan['expected_reclaimable_allocated_bytes'],preserved_payload_member_count=len(plan['preserved_payload_members']),all_preserved_payload_metadata_unchanged=True,all_target_paths_absent=True,groups=plan['groups'],source_code_logs_data_cache_modified=False,Easy_or_q7b_modified=False,remote_archives_modified=False,personal_quota_free_bytes_measured=False)
 assert lastlink_bytes==plan['expected_reclaimable_allocated_bytes']
 with (root/'result.json').open('x') as f:json.dump(result,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 print(json.dumps(result))
except BaseException as e:
 with (root/'failure.json').open('x') as f:json.dump(dict(error_type=type(e).__name__,error=str(e),checked_unix=time.time(),deleted_file_count=len(deleted),confirmed_last_link_unlinked_allocated_bytes=lastlink_bytes),f);f.flush();os.fsync(f.fileno())
 raise
