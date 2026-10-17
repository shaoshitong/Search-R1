import hashlib,json,os,socket,stat,sys,time
from pathlib import Path
p=json.load(sys.stdin);root=Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017');dest=root/'checkpoint-evidence/dfe9c94084bb48b2a9bb7b60f9d882da/global_step_5';core=root/'checkpoints/dfe9c94084bb48b2a9bb7b60f9d882da/global_step_5'
assert socket.gethostname()=='xielab3' and os.getuid()==1002 and dest.resolve()==dest and not dest.is_symlink()
started=time.time();rpath=dest/'sidecar-archive-receipt.json';rraw=rpath.read_bytes();receipt=json.loads(rraw)
assert receipt['status']=='EASY2560_INDEPENDENT_SMALL_SIDECAR_SHA_FSYNC_PASS' and receipt['files']==p['files']
assert receipt['operator_commit']=='5a16192f8c8de43b6c1f122b77ae49c7a4144603' and receipt['operator_source_sha256']=='a503352c5113834be34735e3b8e7f8d0d071483d8edd6ebd7ae71ebd40166db9'
assert receipt['run_id']=='dfe9c94084bb48b2a9bb7b60f9d882da' and receipt['source_commit']=='8a5bac94b8230a99b81977f5290589e3d3708dfe' and receipt['source_job']=='730226' and receipt['step']==5
assert receipt['source_verify_before']==receipt['source_verify_after'] and receipt['fresh_source_SHA_verified_twice'] is True and receipt['fresh_destination_SHA_verified'] is True and receipt['destination_fsynced'] is True and receipt['core_archive_unmodified'] is True
assert receipt['actual_cuda_restore_accepted'] is False and receipt['post_callback_rng_captured'] is False
actual={}
for name,row in p['files'].items():
 part=Path(name);assert not part.is_absolute() and '..' not in part.parts
 path=dest/part;s=path.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==1002 and s.st_nlink==1 and path.resolve()==path
 h=hashlib.sha256();count=0
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):count+=len(block);h.update(block)
 after=path.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 actual[name]={'bytes':count,'sha256':h.hexdigest()};assert actual[name]==row
assert len(actual)==receipt['file_count']==42 and sum(x['bytes'] for x in actual.values())==receipt['total_bytes']==72795748
allfiles={str(x.relative_to(dest)) for x in dest.rglob('*') if x.is_file()};assert allfiles==set(actual)|{'sidecar-archive-receipt.json'}
core_shas={name:hashlib.sha256((core/name).read_bytes()).hexdigest() for name in p['core_small_shas']};assert core_shas==p['core_small_shas']
print(json.dumps({'status':'INDEPENDENT_HK_SMALL_SIDECAR_FULL_SHA_READBACK_PASS','started_unix':started,'finished_unix':time.time(),'uid':os.getuid(),'host':socket.gethostname(),'destination':str(dest),'file_count':len(actual),'total_bytes':sum(x['bytes'] for x in actual.values()),'fresh_SHA_all42':True,'receipt_sha256':hashlib.sha256(rraw).hexdigest(),'fsync_receipt_verified':receipt['destination_fsynced'],'source_before_after_verify_equal':True,'core_small_receipt_SHAs_unchanged':core_shas,'core_large_payload_not_rehashed':True,'actual_cuda_restore_accepted':False,'mutation':False,'files':actual},sort_keys=True))
