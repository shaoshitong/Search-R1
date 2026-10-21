"""Stdlib-only immutable CP128 hardlink; never reads checkpoint payload bytes."""
import argparse,fcntl,hashlib,json,os,stat,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');MAIN=A/'joint-mixed-direct-official-169028-v1';SNAP=A/'mixed-direct-official-cp128-eval-v1';CONFIG=dict(path=str(MAIN/'config.json'),bytes=9615,sha256='2a7f039f4172f5284e3b8282e8ca6200e566bd92bb3013e9802306eca84e7393');JOB='746083'
KEYS={'raw','ema','native_adam','head_adam','raw_buffers','ema_buffers','rng','sampler','official_source'}
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def small(p):
 p=Path(p);assert p.stat().st_size<2*1024**2,'Metadata only';raw=p.read_bytes();return raw,dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):return json.loads(small(p)[0])
def sync(p):
 fd=os.open(str(p),os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
 try:os.fsync(fd)
 finally:os.close(fd)
def once(p,raw):
 if p.exists():assert p.read_bytes()==raw,'Existing immutable snapshot metadata differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return small(p)[1]
def file_stat(s):return dict(device=s.st_dev,inode=s.st_ino,uid=s.st_uid,gid=s.st_gid,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,links=s.st_nlink)
def same_inode(a,b):return all(a[k]==b[k]for k in('device','inode','uid','bytes','mtime_ns'))
def pin(main=MAIN,snap=SNAP,config=CONFIG,job=JOB):
 main=Path(main);snap=Path(snap);snap.mkdir(parents=True,exist_ok=True)
 with(snap/'pin.lock').open('a+')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);raw,actual=small(main/'config.json');assert actual==config;cfg=json.loads(raw);assert cfg['base']==str(main)and cfg['initialization_source']=='official_initial'and cfg['fit_questions']==169028 and cfg['update_budget']==21132 and cfg['loss']=={'binary_mixed_weight':1.};assert not any(k in cfg for k in('source_checkpoint','source_receipt','source_config'))
  proof_path=snap/'snapshot-pin.json'
  if proof_path.exists():
   prior=read(proof_path);assert prior['source_main_config']==config;dest=Path(prior['checkpoint']['path']);now=file_stat(dest.stat());assert same_inode(now,prior['pinned_stat'])and now['uid']==os.getuid()and now['bytes']==prior['checkpoint']['bytes'];assert small(snap/'source-config.json')[1]==prior['source_config'];return dict(status='CP128_IMMUTABLE_HARDLINK_ALREADY_ACCEPTED',pin=small(proof_path)[1],checkpoint=prior['checkpoint'],source_config=prior['source_config'])
  latest=main/'latest.json'
  if not latest.exists():return dict(status='WAITING_FOR_ACCEPTED_MAIN_CP128',latest_cursor=None,GPU_submitted=False)
  receipt_bytes,receipt_ref=small(latest);receipt=json.loads(receipt_bytes);cursor=receipt['cursor']
  if cursor<128:return dict(status='WAITING_FOR_ACCEPTED_MAIN_CP128',latest_cursor=cursor,GPU_submitted=False)
  if cursor>128:raise ValueError('CP128 pin window missed; never substitute a later or rolled-back checkpoint')
  assert receipt['native_steps']==receipt['head_steps']==128 and receipt['config_sha256']==config['sha256']and receipt['allocation_job_id']==str(job)and receipt['full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed']is True;assert set(receipt['accepted_hashes'])==KEYS and all(len(h)==64 for h in receipt['accepted_hashes'].values())
  journal=read(main/'mixed-checkpoint-transaction.json')
  if journal['phase']!='published'or journal['new_checkpoint']!=receipt:return dict(status='WAITING_FOR_CP128_ATOMIC_PUBLICATION',latest_cursor=128,GPU_submitted=False)
  source=Path(receipt['path']);assert source.parent==main/'checkpoints'and source.name.startswith('.rolling-accepted-cursor-000128-')and not source.is_symlink();inputs=snap/'inputs';inputs.mkdir(exist_ok=True);dest=inputs/'checkpoint-cp128.pt';fd=os.open(str(source),os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
  try:
   before=os.fstat(fd);assert stat.S_ISREG(before.st_mode)and before.st_uid==os.getuid()and before.st_size==receipt['bytes']and before.st_dev==inputs.stat().st_dev
   if dest.exists():assert os.path.samefile(source,dest),'Prior unknown snapshot inode preserved'
   else:os.link(str(source),str(dest));sync(inputs)
   after=os.fstat(fd);pinned=dest.stat();assert same_inode(file_stat(before),file_stat(after))and same_inode(file_stat(after),file_stat(pinned))
  finally:os.close(fd)
  copied=once(snap/'source-config.json',raw);source_receipt=once(snap/'source-checkpoint-receipt.json',receipt_bytes);pin_receipt=dict(receipt,path=str(dest));pin_ref=once(snap/'pinned-checkpoint.json',canonical(pin_receipt)+bytes([10]))
  result=dict(status='MAIN_CP128_FULL_ACCEPTED_STATE_IMMUTABLY_HARDLINKED_NO_PAYLOAD_READ',source_main_config=config,source_config=copied,source_latest_receipt_observed=receipt_ref,source_receipt_copy=source_receipt,source_checkpoint_original=receipt,checkpoint=pin_receipt,pinned_checkpoint_receipt=pin_ref,source_stat_before=file_stat(before),source_stat_after=file_stat(after),pinned_stat=file_stat(pinned),source_config_bytes_unchanged=True,source_config_base_preserved=str(main),producer_full_CPU_reload_acceptance_inherited=True,checkpoint_payload_read_or_rehashed_by_pin=False,snapshot_hardlink_immutable=True,GPU_fresh_full_checkpoint_SHA_required_before_load=True,main_training_paused_or_modified=False,actor_updates=0,final4096_read=False,epoch=time.time());ref=once(proof_path,canonical(result)+bytes([10]));return dict(status=result['status'],pin=ref,checkpoint=pin_receipt,source_config=copied)
if __name__=='__main__':print(json.dumps(pin(),sort_keys=True))
