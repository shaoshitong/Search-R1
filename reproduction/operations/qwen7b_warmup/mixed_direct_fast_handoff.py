"""Metadata-only exact-state official Mixed continuation handoff.

No checkpoint payload reads, tensor loads, job submissions or implicit signals.
Only the separate explicit signal controller may stop the verified own process.
"""
import fcntl,hashlib,json,os,re,stat,subprocess,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');MAIN=A/'joint-mixed-direct-official-169028-v1';BASE=A/'joint-mixed-direct-official-169028-fast-v1';SOURCE_JOB='746083';SOURCE_CONFIG=dict(path=str(MAIN/'config.json'),bytes=9615,sha256='2a7f039f4172f5284e3b8282e8ca6200e566bd92bb3013e9802306eca84e7393');SOURCE_PUBLICATION=dict(path=str(MAIN/'source-publication.json'),bytes=6458,sha256='cdae3137aec68c9521afe2bb11e60fb67fc5922b7db00bb076b0e2f787bfce80');KEYS={'raw','ema','native_adam','head_adam','raw_buffers','ema_buffers','rng','sampler','official_source'}
def canonical(x):return(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n').encode()
def proof(p):
 p=Path(p);assert p.stat().st_size<4*1024**2 and p.suffix!='.pt','Small metadata/code only';raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(ref):assert proof(ref['path'])==ref;return read(ref['path'])
def read(p):return json.loads(Path(p).read_bytes())
def syncdir(p):
 fd=os.open(str(p),os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
 try:os.fsync(fd)
 finally:os.close(fd)
def once(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==data,'Immutable handoff metadata differs'
 else:
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  syncdir(p.parent)
 return proof(p)
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert r.returncode==0,r.stderr[-1200:];return r.stdout.strip()
def source_metadata(main=MAIN):
 main=Path(main);cfg=checked(dict(SOURCE_CONFIG,path=str(main/'config.json')));pub=checked(dict(SOURCE_PUBLICATION,path=str(main/'source-publication.json')))
 assert cfg['base']==str(main)and cfg['initialization_source']=='official_initial'and cfg['fit_questions']==169028 and cfg['update_budget']==21132 and cfg['loss']=={'binary_mixed_weight':1.}and cfg['architecture']['micro_questions']==1
 assert cfg['code_hashes']=={name:item['sha256']for name,item in pub['files'].items()}and len(pub['files'])==21
 for name,item in pub['files'].items():assert proof(Path(pub['code_root'])/name)==dict(path=str(Path(pub['code_root'])/name),**item)
 assert read(main/'submission-result.json')['job_id']==SOURCE_JOB
 return cfg,pub

def final_state_metadata(main):
 main=Path(main);cp=read(main/'latest.json');p=read(main/'progress.json');intent=read(main/'update-intent.json');stop=read(main/'interrupted.json');journal=read(main/'mixed-checkpoint-transaction.json');cursor=cp['cursor']
 assert 0<cursor<21132 and cp['native_steps']==cp['head_steps']==cursor and p['cursor']==p['native_steps']==p['head_steps']==intent['cursor']==stop['cursor']==stop['native_steps']==stop['head_steps']==cursor,'Highest accepted/current transaction must be fully saved; never roll back'
 assert intent['phase']=='accepted_update_complete'and stop['status']=='INTERRUPTED_AFTER_COMPLETE_ACCEPTED_UPDATE'and stop['checkpoint']==cp and journal['phase']=='published'and journal['new_checkpoint']==cp
 assert cp['config_sha256']==SOURCE_CONFIG['sha256']and cp['allocation_job_id']==SOURCE_JOB and cp['full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed']is True and set(cp['accepted_hashes'])==KEYS and all(re.fullmatch('[0-9a-f]{64}',v)for v in cp['accepted_hashes'].values())
 assert cp['sampler_state_sha256']==cp['accepted_hashes']['sampler']==cp['data_cursor']['sampler_state_sha256']and cp['data_cursor']['cursor']==cursor
 return cp,{name:proof(main/name)for name in('latest.json','progress.json','update-intent.json','interrupted.json','mixed-checkpoint-transaction.json')}

def inode(s):return dict(device=s.st_dev,inode=s.st_ino,uid=s.st_uid,bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def pin_terminal_source(base=BASE,main=MAIN,*,terminal=None):
 base=Path(base);main=Path(main);base.mkdir(parents=True,exist_ok=True);source_metadata(main)
 with(base/'handoff-finalize.lock').open('a+')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(base/'source-handoff.json').exists():
   prior=read(base/'source-handoff.json');assert inode(Path(prior['pinned_checkpoint']['path']).stat())==prior['pinned_stat'];assert checked(prior['source_config'])==read(main/'config.json');return prior
  if terminal is None:
   rows=command(['sacct','-X','-n','-P','-j',SOURCE_JOB,'--format=JobIDRaw,State,ExitCode,Elapsed']).splitlines();rows=[r for r in rows if r.split('|')[0]==SOURCE_JOB]
   if len(rows)!=1 or rows[0].split('|')[1]!='COMPLETED':return dict(status='WAITING_FOR_OLD_MAIN_GRACEFUL_TERMINAL',main_sacct=rows)
   terminal=rows[0]
  assert terminal.split('|')[:3]==[SOURCE_JOB,'COMPLETED','0:0'],'Exact original main successful terminal required'
  assert not command(['squeue','-h','-u','mou176','-n','q7b-mixed-direct169028','-o','%i|%T']),'An old writer identity remains active'
  lock=(main/'training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   assert not(main/'failure.json').exists(),'Preserve actual source failure; no automatic rollback';cp,refs=final_state_metadata(main);source=Path(cp['path']);assert source.parent==main/'checkpoints'and not source.is_symlink();inputs=base/'inputs';inputs.mkdir(exist_ok=True);target=inputs/'source-checkpoint.pt';fd=os.open(str(source),os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
   try:
    before=os.fstat(fd);assert stat.S_ISREG(before.st_mode)and before.st_uid==os.getuid()and before.st_size==cp['bytes']and before.st_dev==inputs.stat().st_dev
    if target.exists():assert os.path.samefile(source,target)
    else:os.link(str(source),str(target));syncdir(inputs)
    after=os.fstat(fd);pinned=target.stat();assert inode(before)==inode(after)==inode(pinned)
   finally:os.close(fd)
   source_cfg=once(base/'source-config.json',(main/'config.json').read_bytes());original_receipt=once(base/'source-checkpoint-receipt.json',canonical(cp));pinned_cp=dict(cp,path=str(target));pinned_ref=once(base/'pinned-source-checkpoint.json',canonical(pinned_cp));hist=[]
   for epoch in range(cp['cursor']//5283+1):
    path=main/'evaluations'/('epoch-%02d'%epoch)/'complete.json'
    if path.exists():
     item=read(path);assert item['cursor']==epoch*5283 and item['config_sha256']==SOURCE_CONFIG['sha256'];hist.append(proof(path))
    else:assert epoch==cp['cursor']//5283 and cp['cursor']%5283==0 and read(main/'interrupted.json')['pending_evaluation']is True,'Missing required past fixed evaluation'
   result=dict(status='OFFICIAL_DIRECT_MIXED_HIGHEST_ACCEPTED_SOURCE_GRACEFULLY_RETIRED_AND_PINNED',source_main=str(main),source_job_id=SOURCE_JOB,job_id=SOURCE_JOB,terminal_exit_0_0=True,highest_accepted_complete_cursor_equal=True,no_unresolved_tail=True,training_lock_free=True,source_main_terminal=terminal,source_config=source_cfg,source_main_config=dict(SOURCE_CONFIG,path=str(main/'config.json')),source_publication=dict(SOURCE_PUBLICATION,path=str(main/'source-publication.json')),source_checkpoint_original=cp,source_original_receipt_copy=original_receipt,checkpoint=cp,pinned_checkpoint=pinned_cp,pinned_checkpoint_receipt=pinned_ref,source_terminal_state_refs=refs,source_fixed_evaluation_history=hist,source_stat=inode(before),pinned_stat=inode(pinned),cursor=cp['cursor'],pending_evaluation=read(main/'interrupted.json')['pending_evaluation'],accepted_hashes=cp['accepted_hashes'],source_current_world_size=1,current_world_size=1,source_9state_unchanged=True,checkpoint_payload_read_or_rehashed_on_CPU=False,actual_complete_tensor_restore_required_on_new_GPU=True,no_silent_accepted_update_rollback=True,old_training_lock_released_before_new_submit=True,epoch=time.time());once(base/'source-handoff.json',canonical(result));return result
  finally:lock.close()
