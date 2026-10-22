"""Bounded sequential CPU acceptance of published epoch-boundary two-GPU checkpoints.

No GPU allocation, final4096 input, training mutation or recovery is performed.
One mmap-backed3.77GB checkpoint at a time; actual one-block/head geometry is
constructed on meta and strict-loaded by assignment without a duplicate copy.
"""
import argparse,fcntl,gc,hashlib,inspect,json,os,resource,subprocess,time,traceback
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021'
SUITE=A/'online-mixed-layer-prefix-20261022-2gpu-v1'
OUT=A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1'
PINNED_L27P20_SHA='f28778eff8d5ad0bab4f0d82db7506b779f21439ee2b1606408708e6e32e236c'


def read(p):return json.loads(Path(p).read_bytes())
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.%d.tmp'%os.getpid())
 with tmp.open('w')as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def sha(p,progress=None,**binding):
 p=Path(p);h=hashlib.sha256();n=0
 with p.open('rb')as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):
   h.update(chunk);n+=len(chunk)
   if progress and(n%(256*1024**2)==0 or n==p.stat().st_size):
    atomic(progress,dict(stage='ACTUAL_CHECKPOINT_SHA_READING',bytes_read=n,total_bytes=p.stat().st_size,time=time.time(),**binding))
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def stamp(p):
 s=Path(p).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def finite_tensor(t,torch):
 if not torch.is_tensor(t):raise ValueError('Actual tensor required')
 if t.device.type!='cpu' or t.layout!=torch.strided:raise ValueError('CPU dense checkpoint tensor required')
 if t.is_floating_point() or t.is_complex():
  flat=t.detach().reshape(-1)
  for offset in range(0,flat.numel(),1048576):
   if not bool(torch.isfinite(flat[offset:offset+1048576]).all()):raise ValueError('Nonfinite tensor')
 return t.numel()
def mapping_sha(mapping,torch):
 h=hashlib.sha256()
 for name,t in mapping.items():
  h.update(name.encode());h.update(str(t.dtype).encode());h.update(repr(tuple(t.shape)).encode())
  raw=t.detach().contiguous().reshape(-1).view(torch.uint8)
  for offset in range(0,raw.numel(),1048576):h.update(raw[offset:offset+1048576].numpy().tobytes())
 return h.hexdigest()


def validate_one(layer,prefix,expected_epoch):
 expected_cursor=64*expected_epoch
 import numpy as np
 import random
 import torch
 from torch import nn
 from transformers import Qwen2Config
 from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RotaryEmbedding
 from reproduction.operations.qwen7b_warmup.native_mixed_group_classifier import MixedTokenHead,NativeMixedGroupProbe
 variant='layer%d-prefix%d'%(layer,prefix);base=SUITE/variant;dest=OUT/variant/('epoch-%02d'%expected_epoch);dest.mkdir(exist_ok=True,parents=True)
 latest=read(base/'latest.json');job=str(read(base/'root-submission.json')['job_id'])
 training_pub=SUITE/'depth-expansion-source-publication.json'
 if sha(training_pub)['sha256']!='cc76d46721f9416b7c32a5c38b2e88a68b7e4eb2acbc608ec3bd71bb7e39a6cf'or read(training_pub)['commit']!='c366ad4bf11c32f78b2622775c2a7bf83e1a5299':raise ValueError('Actual original P80 runtime publication differs')
 path=Path(latest['path']);expected=base/('checkpoint-epoch-%02d.pt'%expected_epoch)
 if path!=expected or path.is_symlink()or path.stat().st_size!=latest['bytes']:raise ValueError('Exact published immutable epoch checkpoint required')
 accepted=dest/'accepted.json'
 if accepted.exists():
  old=read(accepted)
  current=stamp(path)
  # The shared mount has node-local st_dev48/50 across login hosts. Its file
  # inode, bytes, mtime and ctime remain the actual content-change evidence.
  stable=('inode','bytes','mtime_ns','ctime_ns')
  if old['checkpoint']!=latest or any(old['file_stat'][k]!=current[k]for k in stable):raise ValueError('Previously accepted checkpoint changed')
  return dict(status='EXISTING_ACTUAL_CPU_CHECKPOINT_ACCEPTANCE_REUSED',variant=variant,accepted=str(accepted),checkpoint=latest,node_local_device_number_ignored=True)
 started=time.time();before=stamp(path);progress=OUT/'progress.json';binding=dict(variant=variant,job_id=job)
 actualproof=sha(path,progress,**binding)
 if actualproof!=latest:raise ValueError('Fresh entire completed checkpoint SHA differs from latest receipt')
 atomic(progress,dict(stage='CPU_MMAP_TORCH_LOAD',time=time.time(),checkpoint=actualproof,**binding))
 blob=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
 if (blob['cursor']!=expected_cursor or blob['epoch']!=expected_epoch or blob['layer']!=layer or blob['prefix']!=prefix
     or blob['actor_updates']!=0 or blob['raw_updates_EMA']is not True or blob['learning_rate']!=5e-6
     or len(blob['RNG_by_rank'])!=2):raise ValueError('Actual epoch and64*epoch updates, official actor or two-rank source differs')
 plan=read(A/'online-mixed-layer-prefix-20261022-v1/inputs'/('epoch-%02d-plan.json'%expected_epoch))
 plan_sha=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
 if blob['plan_sha256']!=plan_sha:raise ValueError('Final checkpoint source epoch plan SHA differs')
 atomic(progress,dict(stage='FINITE_RAW_EMA_AND_META_STRICT_LOAD',time=time.time(),**binding))
 config=Qwen2Config.from_pretrained(str(B/'assets/qwen25-7b-instruct-a09a35458c70'),local_files_only=True)
 config._attn_implementation='flash_attention_2'
 with torch.device('meta'):
  block=Qwen2DecoderLayer(config,layer_idx=layer).float();rotary=Qwen2RotaryEmbedding(config=config).float();head=MixedTokenHead()
  if prefix!=20:head.policy_slots=nn.Embedding(prefix,256);head.prefix=prefix
  probe=NativeMixedGroupProbe(block,rotary,'cache_position'in inspect.signature(block.forward).parameters,head)
 required={n:tuple(t.shape)for n,t in probe.state_dict().items()};parameter_names=[n for n,_ in probe.named_parameters()]
 geometry={};totals={}
 for key in('raw','EMA'):
  mapping=blob[key]
  if set(mapping)!=set(required)or any(tuple(mapping[n].shape)!=shape for n,shape in required.items()):raise ValueError('Actual checkpoint is incompatible with exact native block plus head geometry')
  totals[key]=sum(finite_tensor(t,torch)for t in mapping.values())
  probe.load_state_dict(mapping,strict=True,assign=True)
  if any(not torch.equal(t,mapping[n])for n,t in probe.state_dict().items()):raise ValueError('Actual strict-loaded state differs')
  geometry[key]=dict(strict_load_state_dict_passed=True,tensors=len(mapping),parameter_elements=totals[key],independent_dtype_shape_tensor_SHA256=mapping_sha(mapping,torch))
 if geometry['raw']['independent_dtype_shape_tensor_SHA256']==geometry['EMA']['independent_dtype_shape_tensor_SHA256']:
  raise ValueError('Completed pilot raw/EMA unexpectedly identical; preserve and inspect')
 # Assign RAW again before loading the saved Adam against the real parameter order.
 probe.load_state_dict(blob['raw'],strict=True,assign=True)
 opt=torch.optim.Adam(probe.parameters(),lr=5e-6);opt.load_state_dict(blob['Adam'])
 source=blob['Adam'];ids=[i for group in source['param_groups']for i in group['params']]
 if len(ids)!=len(parameter_names)or len(set(ids))!=len(ids)or set(ids)!=set(source['state']):raise ValueError('Complete one-to-one native/head Adam states required')
 moments=0
 for name,source_id,param in zip(parameter_names,ids,probe.parameters()):
  saved=source['state'][source_id];loaded=opt.state[param]
  if (not {'step','exp_avg','exp_avg_sq'}<=set(saved)or int(saved['step'])!=expected_cursor or int(loaded['step'])!=expected_cursor
      or tuple(saved['exp_avg'].shape)!=tuple(param.shape)or tuple(saved['exp_avg_sq'].shape)!=tuple(param.shape)):
   raise ValueError('Missing expected64*epoch Adam clock or incompatible parameter moments: '+name)
  for key in('exp_avg','exp_avg_sq'):
   moments+=finite_tensor(saved[key],torch)
   if not torch.equal(loaded[key],saved[key]):raise ValueError('Restored Adam moment differs: '+name)
  if not bool(saved['exp_avg_sq'].ge(0).all()):raise ValueError('Negative Adam second moment')
 rngproof=[]
 for rank,rng in enumerate(blob['RNG_by_rank']):
  if set(rng)!={'python','numpy','torch','cuda'}:raise ValueError('Complete Python/Numpy/CPU/CUDA source RNG required')
  random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy']);torch.Generator(device='cpu').set_state(rng['torch'])
  cuda=rng['cuda']
  if not torch.is_tensor(cuda)or cuda.dtype!=torch.uint8 or cuda.ndim!=1 or cuda.numel()<16 or cuda.device.type!='cpu':raise ValueError('Serialized actual CUDA RNG byte-state required')
  rngproof.append(dict(rank=rank,python_state_accepted=True,numpy_state_accepted=True,CPU_torch_generator_state_accepted=True,CUDA_RNG_serialized_bytes=int(cuda.numel()),CUDA_runtime_restore_tested=False))
 for group in source['param_groups']:
  if group['lr']!=5e-6:raise ValueError('Saved actual Adam LR differs')
 if read(base/'latest.json')!=latest or before!=stamp(path):raise ValueError('Completed checkpoint changed during CPU verification')
 result=dict(status='PUBLISHED_EPOCH_BOUNDARY_CPU_SHA_RAW_EMA_ADAM_TWO_RANK_RNG_ACCEPTED',variant=variant,job_id=job,
  checkpoint=actualproof,file_stat=before,source_published_latest=sha(base/'latest.json'),source_root_receipt=sha(base/'root-submission.json'),source_plan_sha256=plan_sha,
  source_training_publication=sha(training_pub),
  cursor=expected_cursor,completed_epoch=expected_epoch,layer=layer,prefix=prefix,geometry=geometry,
  Adam=dict(actual_steps=expected_cursor,parameter_states=len(ids),finite_first_and_second_moment_elements=moments,strict_actual_optimizer_state_load_passed=True),
  source_RNG_by_rank=rngproof,raw_and_EMA_separate=True,actor_updates=0,probe_optimizer_updates=0,
  actual_GPU_forward_or_next_update_restore_tested=False,full_trajectory_sampler_RNG_resume_not_claimed=True,
  raw_or_EMA_finite=True,CPU_mmap=True,one_checkpoint_at_a_time=True,
  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.time()-started,time=time.time())
 atomic(accepted,result);del blob,probe,opt;gc.collect();return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--layer',type=int,choices=(15,18),required=True);p.add_argument('--prefix',type=int,choices=(80,),required=True);p.add_argument('--expected-epoch',type=int,choices=(1,2),required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
 if os.uname().sysname!='Linux'or os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise RuntimeError('CPU-only remote epoch boundary acceptance required')
 if sha(Path(__file__))['sha256']!=a.source_sha:raise ValueError('Validator source SHA differs')
 OUT.mkdir(exist_ok=True);guard=(OUT/'acceptance.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 atomic(OUT/'controller.json',dict(pid=os.getpid(),layer=a.layer,prefix=a.prefix,expected_epoch=a.expected_epoch,expected_cursor=64*a.expected_epoch,source_sha=a.source_sha,time=time.time()))
 try:print(json.dumps(validate_one(a.layer,a.prefix,a.expected_epoch)))
 except BaseException as e:atomic(OUT/'failure.json',dict(layer=a.layer,prefix=a.prefix,epoch=a.expected_epoch,error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
if __name__=='__main__':main()
