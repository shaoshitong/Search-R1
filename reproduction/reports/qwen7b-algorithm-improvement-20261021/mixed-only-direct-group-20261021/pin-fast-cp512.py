"""Metadata-only stable hardlink of accepted512; no main process mutation."""
import pathlib,json,hashlib,os,time,fcntl,stat
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');main=A/'joint-mixed-direct-official-169028-fast-v1';b=A/'mixed-direct-official-cp512-eval-v1';b.mkdir(exist_ok=True);(b/'inputs').mkdir(exist_ok=True)
def read(p):return json.loads(p.read_bytes())
def immutable(p,value):
 raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
 if p.exists():assert p.read_bytes()==raw,'Immutable pin metadata differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
with(b/'pin.lock').open('a+')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if(b/'pin-complete.json').exists():print(json.dumps(read(b/'pin-complete.json')));raise SystemExit(0)
 raw=(main/'config.json').read_bytes();assert len(raw)==14625 and hashlib.sha256(raw).hexdigest()=='22098d5a4678db2f9c56d56cb5709829c3a4619c626e0b52d16ce726844402ac';c=json.loads(raw)
 latest=read(main/'latest.json');p=pathlib.Path(latest['path']);assert latest['cursor']==latest['native_steps']==latest['head_steps']==512 and latest['full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed'] and latest['config_sha256']==hashlib.sha256(raw).hexdigest()
 assert p.parent==main/'checkpoints' and p.name.startswith('.rolling-accepted-cursor-000512-') and len(latest['accepted_hashes'])==9
 before=p.lstat();assert stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_size==latest['bytes']
 target=b/'inputs/checkpoint-cp512.pt'
 if target.exists():assert (target.stat().st_dev,target.stat().st_ino)==(before.st_dev,before.st_ino)
 else:os.link(str(p),str(target))
 after=target.stat();assert(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 config=b/'source-config.json'
 if config.exists():assert config.read_bytes()==raw
 else:config.write_bytes(raw)
 pinned=dict(latest,path=str(target));immutable(b/'source-checkpoint-original.json',latest);immutable(b/'pinned-checkpoint.json',pinned)
 result=dict(status='FAST_OFFICIAL_CP512_METADATA_PIN_COMPLETE',epoch=time.time(),source_cursor=512,source_main=str(main),source_config=proof(config),source_main_config=dict(proof(config),path=str(main/'config.json')),source_checkpoint_original=latest,checkpoint=pinned,pinned_checkpoint_receipt=proof(b/'pinned-checkpoint.json'),source_checkpoint_original_receipt=proof(b/'source-checkpoint-original.json'),pinned_stat=dict(device=after.st_dev,inode=after.st_ino,bytes=after.st_size,uid=after.st_uid,mtime_ns=after.st_mtime_ns),source_full_CPU_reload_inherited=True,pin_checkpoint_payload_read_or_rehashed=False,GPU_fresh_full_checkpoint_SHA_required_before_load=True,main_source_or_training_mutated=False,final4096_read=False)
 immutable(b/'pin-complete.json',result);print(json.dumps(result))
