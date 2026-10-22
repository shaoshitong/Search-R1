"""Full-state unique-question continuation, raw gradients then independent EMA.

The source is the accepted first-epoch checkpoint. Native Adam stops at 3898;
head Adam and the global sampler continue. The frozen raw block is bitwise
bound to the phase boundary; EMA remains passive tracking of the full model.
"""
import copy,fcntl,gc,os,shutil,time,uuid
from pathlib import Path
import torch
from . import mixed_group_checkpoint as previous
from . import probe_ablation_train as u

ROLE='DIRECT_MIXED_E1_UNIQUE_G5_LOW_LR_THEN_HEAD_ONLY'
PHASE2='phase2_joint_low_lr'
PHASE3='phase3_head_only'
FREEZE_CURSOR=3898
SOURCE_CURSOR=2577
FINAL_CURSOR=5219
JOINT_LR=5e-7
HEAD_LR=5e-7
tree_hash=previous.tree_hash
cpu_tree=previous.cpu_tree
optimizer_steps=previous.optimizer_steps
_sync=previous._sync


def native_hash(value):
 mapping=value.state_dict() if hasattr(value,'state_dict') else value
 native={k:v for k,v in mapping.items() if k.startswith('block.')}
 if not native:raise ValueError('Raw block parameters absent from freeze proof')
 return u.tensor_hash(native)


def components(blob):
 result=previous.components(blob)
 result['phase_policy']=tree_hash({k:blob[k] for k in ('role','stage','source_steps','training_phase','native_freeze_cursor','native_frozen','native_frozen_boundary_sha256','source_epoch1_checkpoint','native_learning_rate','head_learning_rate','EMA_full_model_passive_tracking','phase_state')})
 return result


def validate(blob,binding=None):
 c=blob.get('cursor');sampler=blob.get('sampler_state',{});phase=blob.get('training_phase')
 joint=phase==PHASE2
 if type(c)is not int or not SOURCE_CURSOR<=c<=FINAL_CURSOR or phase not in (PHASE2,PHASE3):raise ValueError('Unknown phased continuation cursor or stage')
 if (joint and c>FREEZE_CURSOR)or(not joint and c<FREEZE_CURSOR):raise ValueError('Phased update cursor outside its stage')
 native_expected=c if joint else FREEZE_CURSOR
 if (blob.get('role')!=ROLE or blob.get('source_steps')!=0 or blob.get('phase')!='accepted_update_complete'
     or blob.get('native_steps')!=native_expected or blob.get('head_steps')!=c
     or optimizer_steps(blob['native_optimizer'])!=native_expected or optimizer_steps(blob['head_optimizer'])!=c
     or sampler.get('accepted_updates')!=c or sampler.get('binding')!=blob.get('data_cursor',{}).get('sampler_binding')
     or blob['data_cursor'].get('cursor')!=c or u.digest(sampler)!=blob.get('sampler_state_sha256')):raise ValueError('Actual native/head/global clocks differ from phase policy')
 if blob.get('phase_state')!={k:blob[k]for k in('training_phase','native_freeze_cursor','native_frozen_boundary_sha256','source_epoch1_checkpoint')}:raise ValueError('Phase state aliases differ')
 if blob.get('native_freeze_cursor')!=FREEZE_CURSOR or blob.get('native_frozen')!=(not joint) or blob.get('EMA_full_model_passive_tracking')is not True:raise ValueError('Frozen native/EMA phase policy differs')
 head_lr=JOINT_LR if joint else HEAD_LR
 if blob.get('native_learning_rate')!=JOINT_LR or blob.get('head_learning_rate')!=head_lr:raise ValueError('Saved phase learning rates differ')
 for key,lr in (('native_optimizer',JOINT_LR),('head_optimizer',head_lr)):
  if any(group['lr']!=lr for group in blob[key]['param_groups']):raise ValueError('Actual Adam learning rate differs from phase metadata')
 if not joint and native_hash(blob['raw'])!=blob.get('native_frozen_boundary_sha256'):raise ValueError('Raw native block changed during frozen head-only phase')
 if blob.get('source_epoch1_checkpoint',{}).get('cursor')!=SOURCE_CURSOR or not blob['source_epoch1_checkpoint'].get('sha256'):raise ValueError('Accepted epoch1 source receipt missing')
 if blob['data_cursor'].get('sampler_state_sha256')!=blob['sampler_state_sha256']or blob.get('official_initialization',{}).get('initialization_source')!='official_initial_Qwen2.5-7B-Instruct_block27':raise ValueError('Sampler or official block provenance differs')
 if binding and any(blob.get(k)!=v for k,v in binding.items()):raise ValueError('Different immutable phased model/data/config binding')
 if any(k.startswith('output.')for k in blob['raw'])or not any(k.startswith('group_head.')for k in blob['raw']):raise ValueError('Wrong direct Mixed architecture')
 if blob.get('accepted_hashes')!=components(blob):raise ValueError('Full phased checkpoint component SHA differs')
 for mapping in(blob['raw'],blob['ema'],blob['native_optimizer']['state'],blob['head_optimizer']['state']):
  for v in mapping.values():
   for t in(v.values()if isinstance(v,dict)else(v,)):
    if torch.is_tensor(t)and not bool(torch.isfinite(t).all()):raise ValueError('Nonfinite phased parameter/Adam tensor')
 return c


def snapshot(model,ema,native_opt,head_opt,device,sampler,cursor,binding,provenance,seen,stage,phase_state=None):
 state=dict(phase_state or {})
 phase=state['training_phase'];joint=phase==PHASE2
 value=dict(role=ROLE,phase='accepted_update_complete',stage=stage,cursor=cursor,source_steps=0,native_steps=optimizer_steps(native_opt.state_dict()),head_steps=optimizer_steps(head_opt.state_dict()),raw=cpu_tree(model.state_dict()),ema=cpu_tree(ema.state_dict()),native_optimizer=cpu_tree(native_opt.state_dict()),head_optimizer=cpu_tree(head_opt.state_dict()),raw_buffers=u.buffers(model),ema_buffers=u.buffers(ema),rng=cpu_tree(u.rng_state(device)),sampler_state=sampler.state_dict(),official_initialization=copy.deepcopy(provenance),training_seen_qids=sorted(seen),source_optimizer_steps=SOURCE_CURSOR,current_world_size=1,prior_probe_checkpoint_loaded=True,actor_updates=0,**binding)
 value.update(training_phase=phase,native_freeze_cursor=FREEZE_CURSOR,native_frozen=not joint,native_frozen_boundary_sha256=state.get('native_frozen_boundary_sha256'),source_epoch1_checkpoint=copy.deepcopy(state['source_epoch1_checkpoint']),native_learning_rate=JOINT_LR,head_learning_rate=JOINT_LR if joint else HEAD_LR,EMA_full_model_passive_tracking=True)
 value['phase_state']={k:copy.deepcopy(value[k])for k in('training_phase','native_freeze_cursor','native_frozen_boundary_sha256','source_epoch1_checkpoint')}
 value['sampler_state_sha256']=u.digest(value['sampler_state']);value['data_cursor']=dict(cursor=cursor,sampler_binding=sampler.binding,sampler_state_sha256=value['sampler_state_sha256']);value['accepted_hashes']=components(value);validate(value,binding);return value


def restore(blob,model,ema,native_opt,head_opt,device,sampler,binding):
 cursor=validate(blob,binding);model.load_state_dict(blob['raw'],strict=True);ema.load_state_dict(blob['ema'],strict=True);u.restore_buffers(model,blob['raw_buffers']);u.restore_buffers(ema,blob['ema_buffers'])
 for opt,key in((native_opt,'native_optimizer'),(head_opt,'head_optimizer')):
  opt.load_state_dict(copy.deepcopy(blob[key]))
  for state in opt.state.values():
   for k,x in state.items():
    if torch.is_tensor(x)and k!='step':state[k]=x.to(device)
 for p in model.block.parameters():p.requires_grad_(not blob['native_frozen'])
 sampler.load_state_dict(blob['sampler_state']);u.restore_rng(blob['rng'],device)
 actual=dict(blob,raw=model.state_dict(),ema=ema.state_dict(),native_optimizer=native_opt.state_dict(),head_optimizer=head_opt.state_dict(),raw_buffers=u.buffers(model),ema_buffers=u.buffers(ema),rng=u.rng_state(device),sampler_state=sampler.state_dict())
 if components(actual)!=blob['accepted_hashes']:raise ValueError('Actual phased state restoration differs')
 return cursor,set(blob['training_seen_qids'])


def save(base,blob,pin=False,owned_allocation_required=True):
 if owned_allocation_required and not os.environ.get('SLURM_JOB_ID'):raise ValueError('Production full-state I/O requires owned allocation')
 validate(blob);base=Path(base);root=base/'checkpoints';root.mkdir(exist_ok=True,parents=True)
 with(base/'mixed-checkpoint.lock').open('a+')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);journal=base/'mixed-checkpoint-transaction.json';old=u.read(base/'latest.json')if(base/'latest.json').exists()else None
  if journal.exists()and u.read(journal)['phase']!='published':raise ValueError('Unresolved full-state transaction; preserve without rollback')
  if old:
   if old['cursor']>blob['cursor']:raise ValueError('Never roll back accepted phased state')
   if old['cursor']==blob['cursor']:
    if old['accepted_hashes']!=blob['accepted_hashes']:raise ValueError('Same cursor has different accepted state; save after next real update')
    u.checked(old);return old
  stage=root/('.mixed-stage-'+uuid.uuid4().hex);stage.mkdir();temp=stage/'state.pt';published=False
  try:
   with temp.open('wb')as f:torch.save(blob,f);f.flush();os.fsync(f.fileno())
   digest=u.sha(temp);check=torch.load(temp,map_location='cpu',mmap=True,weights_only=False);validate(check,{k:blob[k]for k in('config_sha256','matched_binding_sha256')})
   if check['accepted_hashes']!=blob['accepted_hashes']:raise ValueError('Reloaded full phased state differs before publication')
   del check;gc.collect();body=root/(('cursor-%06d.pt'%blob['cursor'])if pin else('.rolling-accepted-cursor-%06d-%s.pt'%(blob['cursor'],uuid.uuid4().hex)))
   if body.exists():raise ValueError('Unknown immutable checkpoint preserved')
   os.replace(temp,body);_sync(root)
   receipt=dict(path=str(body),bytes=body.stat().st_size,sha256=digest,cursor=blob['cursor'],native_steps=blob['native_steps'],head_steps=blob['head_steps'],stage=blob['stage'],training_phase=blob['training_phase'],phase_state=blob['phase_state'],native_frozen=blob['native_frozen'],native_frozen_boundary_sha256=blob['native_frozen_boundary_sha256'],native_learning_rate=blob['native_learning_rate'],head_learning_rate=blob['head_learning_rate'],config_sha256=blob['config_sha256'],matched_binding_sha256=blob['matched_binding_sha256'],accepted_hashes=blob['accepted_hashes'],data_cursor=blob['data_cursor'],sampler_state_sha256=blob['sampler_state_sha256'],full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed=True,pinned=pin,allocation_job_id=os.environ.get('SLURM_JOB_ID','syntheticCPU'),epoch_time=time.time())
   if pin:u.atomic(body.with_suffix('.json'),receipt)
   u.atomic(journal,dict(phase='verified_new_body_before_publish',new_checkpoint=receipt,previous_checkpoint=old));alias=root/('.rolling-link-'+uuid.uuid4().hex);os.link(body,alias);os.replace(alias,root/'rolling-latest.pt');_sync(root);u.atomic(base/'latest.json',receipt);_sync(base);u.atomic(journal,dict(phase='published',new_checkpoint=receipt,previous_checkpoint=old));_sync(base);published=True
   if old:
    path=Path(old['path'])
    if path!=body and path.parent==root and path.name.startswith('.rolling-accepted-cursor-'):path.unlink();_sync(root)
   shutil.rmtree(stage);return receipt
  except BaseException as e:
   if not published:u.atomic(stage/'failure.json',dict(cursor=blob['cursor'],error=repr(e),old_latest_preserved=True))
   raise
