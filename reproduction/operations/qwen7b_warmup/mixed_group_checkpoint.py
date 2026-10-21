"""Accepted raw/EMA + two-Adam/RNG/sampler checkpoints for direct Mixed.

New phase from official block27: both native and group-head Adam start0.
Every component is reloaded and checked BEFORE publishing latest. One rolling
body plus immutable CP0/1/epoch pins; no early legacy-saver latest publication.
"""
import copy,fcntl,gc,hashlib,json,math,os,shutil,time,uuid
from pathlib import Path
import numpy as np
import torch
from . import probe_ablation_train as u
ROLE='OFFICIAL_NATIVE_BLOCK27_DIRECT_MIXED_NEW_PHASE'
SOURCE_STEPS=0

def tree_hash(value):
 h=hashlib.sha256()
 def visit(v):
  if torch.is_tensor(v):
   x=v.detach().cpu().contiguous();h.update(str(x.dtype).encode());h.update(repr(tuple(x.shape)).encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(v,np.ndarray):h.update(str(v.dtype).encode());h.update(repr(v.shape).encode());h.update(v.tobytes())
  elif isinstance(v,dict):
   h.update(b'dict')
   for k in sorted(v,key=repr):visit(k);visit(v[k])
  elif isinstance(v,(list,tuple)):
   h.update(type(v).__name__.encode());h.update(str(len(v)).encode())
   for x in v:visit(x)
  else:h.update(type(v).__name__.encode());h.update(repr(v).encode())
 visit(value);return h.hexdigest()

def cpu_tree(v):
 if torch.is_tensor(v):return v.detach().cpu().clone()
 if isinstance(v,dict):return {k:cpu_tree(x)for k,x in v.items()}
 if isinstance(v,list):return [cpu_tree(x)for x in v]
 if isinstance(v,tuple):return tuple(cpu_tree(x)for x in v)
 return copy.deepcopy(v)

def optimizer_steps(saved,empty_zero=False):
 values={int(v['step'].item()if torch.is_tensor(v['step'])else v['step'])for v in saved['state'].values()if v}
 if not values and empty_zero:return 0
 if len(values)!=1:raise ValueError('Adam moments/clock absent or inconsistent')
 return values.pop()

def components(blob):
 return dict(raw=u.model_mapping_hash(blob['raw']),ema=u.model_mapping_hash(blob['ema']),native_adam=tree_hash(blob['native_optimizer']),head_adam=tree_hash(blob['head_optimizer']),raw_buffers=u.tensor_hash(blob['raw_buffers']),ema_buffers=u.tensor_hash(blob['ema_buffers']),rng=u.rng_hash(blob['rng']),sampler=u.digest(blob['sampler_state']),official_source=tree_hash(blob['official_initialization']))

def validate(blob,binding=None):
 c=blob.get('cursor');sampler=blob.get('sampler_state',{})
 if (blob.get('role')!=ROLE or type(c)is not int or c<0 or blob.get('source_steps')!=SOURCE_STEPS or blob.get('native_steps')!=SOURCE_STEPS+c or blob.get('head_steps')!=c or blob.get('phase')!='accepted_update_complete'
     or optimizer_steps(blob['native_optimizer'],True)!=SOURCE_STEPS+c or optimizer_steps(blob['head_optimizer'],True)!=c or sampler.get('accepted_updates')!=c or sampler.get('binding')!=blob.get('data_cursor',{}).get('sampler_binding')or blob['data_cursor'].get('cursor')!=c or u.digest(sampler)!=blob.get('sampler_state_sha256')):raise ValueError('Direct Mixed native/head/sampler accepted clocks differ')
 if blob['data_cursor'].get('sampler_state_sha256')!=blob['sampler_state_sha256']or blob.get('official_initialization',{}).get('initialization_source')!='official_initial_Qwen2.5-7B-Instruct_block27':raise ValueError('Data cursor or official initialization provenance absent')
 if binding and any(blob.get(k)!=v for k,v in binding.items()):raise ValueError('Different immutable new-phase model/data/config binding')
 if any(k.startswith('output.')for k in blob['raw'])or not any(k.startswith('group_head.')for k in blob['raw']):raise ValueError('Retired scalar output must not enter new trainable state')
 if blob['accepted_hashes']!=components(blob):raise ValueError('Full direct Mixed checkpoint component SHA differs')
 for mapping in(blob['raw'],blob['ema'],blob['native_optimizer']['state'],blob['head_optimizer']['state']):
  values=mapping.values()
  for v in values:
   for t in(v.values()if isinstance(v,dict)else(v,)):
    if torch.is_tensor(t)and not bool(torch.isfinite(t).all()):raise ValueError('Nonfinite saved parameter/Adam tensor')
 return c

def snapshot(model,ema,native_opt,head_opt,device,sampler,cursor,binding,provenance,seen,stage):
 value=dict(role=ROLE,phase='accepted_update_complete',stage=stage,cursor=cursor,source_steps=SOURCE_STEPS,native_steps=optimizer_steps(native_opt.state_dict(),True),head_steps=optimizer_steps(head_opt.state_dict(),True),raw=cpu_tree(model.state_dict()),ema=cpu_tree(ema.state_dict()),native_optimizer=cpu_tree(native_opt.state_dict()),head_optimizer=cpu_tree(head_opt.state_dict()),raw_buffers=u.buffers(model),ema_buffers=u.buffers(ema),rng=cpu_tree(u.rng_state(device)),sampler_state=sampler.state_dict(),official_initialization=copy.deepcopy(provenance),training_seen_qids=sorted(seen),new_architecture_phase_cursor=True,original_124194_sampler_resume_claim=False,source_optimizer_steps=0,current_world_size=1,prior_probe_checkpoint_loaded=False,actor_updates=0,**binding)
 value['sampler_state_sha256']=u.digest(value['sampler_state']);value['data_cursor']=dict(cursor=cursor,sampler_binding=sampler.binding,sampler_state_sha256=value['sampler_state_sha256']);value['accepted_hashes']=components(value);validate(value,binding);return value

def restore(blob,model,ema,native_opt,head_opt,device,sampler,binding):
 cursor=validate(blob,binding);model.load_state_dict(blob['raw'],strict=True);ema.load_state_dict(blob['ema'],strict=True);u.restore_buffers(model,blob['raw_buffers']);u.restore_buffers(ema,blob['ema_buffers'])
 for opt,key in((native_opt,'native_optimizer'),(head_opt,'head_optimizer')):
  opt.load_state_dict(copy.deepcopy(blob[key]))
  for state in opt.state.values():
   for k,x in state.items():
    if torch.is_tensor(x)and k!='step':state[k]=x.to(device)
 sampler.load_state_dict(blob['sampler_state']);u.restore_rng(blob['rng'],device)
 actual=dict(blob,raw=model.state_dict(),ema=ema.state_dict(),native_optimizer=native_opt.state_dict(),head_optimizer=head_opt.state_dict(),raw_buffers=u.buffers(model),ema_buffers=u.buffers(ema),rng=u.rng_state(device),sampler_state=sampler.state_dict())
 if components(actual)!=blob['accepted_hashes']:raise ValueError('Actual native/head/Adam/RNG/sampler restoration mismatch')
 return cursor,set(blob['training_seen_qids'])

def _sync(path):
 fd=os.open(str(path),os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
 try:os.fsync(fd)
 finally:os.close(fd)

def save(base,blob,pin=False,owned_allocation_required=True):
 if owned_allocation_required and not os.environ.get('SLURM_JOB_ID'):raise ValueError('Production full-state I/O requires owned allocation')
 validate(blob);base=Path(base);root=base/'checkpoints';root.mkdir(exist_ok=True,parents=True)
 with(base/'mixed-checkpoint.lock').open('a+')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);journal=base/'mixed-checkpoint-transaction.json';previous=u.read(base/'latest.json')if(base/'latest.json').exists()else None
  if journal.exists()and u.read(journal)['phase']!='published':raise ValueError('Unresolved accepted checkpoint transaction; preserve without rollback')
  if previous:
   if previous['cursor']>blob['cursor']:raise ValueError('Never roll back accepted direct Mixed state')
   if previous['cursor']==blob['cursor']:
    if previous['accepted_hashes']!=blob['accepted_hashes']:raise ValueError('Same cursor has different accepted state')
    u.checked(previous);return previous
  stage=root/('.mixed-stage-'+uuid.uuid4().hex);stage.mkdir();temp=stage/'state.pt';published=False
  try:
   with temp.open('wb')as f:torch.save(blob,f);f.flush();os.fsync(f.fileno())
   digest=u.sha(temp);check=torch.load(temp,map_location='cpu',mmap=True,weights_only=False);validate(check,{k:blob[k]for k in('config_sha256','matched_binding_sha256')})
   if check['accepted_hashes']!=blob['accepted_hashes']or check['stage']!=blob['stage']:raise ValueError('Reloaded full native/head state differs before latest publication')
   del check;gc.collect();body=root/(('cursor-%06d.pt'%blob['cursor'])if pin else('.rolling-accepted-cursor-%06d-%s.pt'%(blob['cursor'],uuid.uuid4().hex)))
   if body.exists():raise ValueError('Unknown immutable checkpoint preserved')
   os.replace(temp,body);_sync(root);receipt=dict(path=str(body),bytes=body.stat().st_size,sha256=digest,cursor=blob['cursor'],native_steps=blob['native_steps'],head_steps=blob['head_steps'],stage=blob['stage'],config_sha256=blob['config_sha256'],matched_binding_sha256=blob['matched_binding_sha256'],accepted_hashes=blob['accepted_hashes'],data_cursor=blob['data_cursor'],sampler_state_sha256=blob['sampler_state_sha256'],full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed=True,pinned=pin,allocation_job_id=os.environ.get('SLURM_JOB_ID','syntheticCPU'),epoch_time=time.time())
   if pin:u.atomic(body.with_suffix('.json'),receipt)
   u.atomic(journal,dict(phase='verified_new_body_before_publish',new_checkpoint=receipt,previous_checkpoint=previous));alias=root/('.rolling-link-'+uuid.uuid4().hex);os.link(body,alias);os.replace(alias,root/'rolling-latest.pt');_sync(root);u.atomic(base/'latest.json',receipt);_sync(base);u.atomic(journal,dict(phase='published',new_checkpoint=receipt,previous_checkpoint=previous));_sync(base);published=True
   if previous:
    old=Path(previous['path'])
    if old!=body and old.parent==root and old.name.startswith('.rolling-accepted-cursor-'):old.unlink();_sync(root)
   shutil.rmtree(stage);return receipt
  except BaseException as e:
   if not published:u.atomic(stage/'failure.json',dict(cursor=blob['cursor'],error=repr(e),old_latest_preserved=True))
   raise
