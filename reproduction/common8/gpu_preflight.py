"""Native-Slurm eight-rank mechanics test; not evidence of GRPO/model acceptance."""
import datetime,json,os,time
from pathlib import Path
from search_g1.process_jit_cache import configure
process_jit_cache=configure()
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision
from search_g1.numerics import require_finite
from search_g1.gradient_clipping import clip_global_l2
from reproduction.common8.scheduler import constant_scheduler

from reproduction.common8.topology import world_size
world=world_size()
rank=int(os.environ['LOCAL_RANK']);assert rank in range(world)
assert os.environ['SLURM_JOB_ID'] and torch.cuda.device_count()==world
root=Path(os.environ['SEARCH_G1_GPU_PREFLIGHT_DIR'])
def collective_check(ok, message):
 value=torch.tensor([int(bool(ok))],device='cuda',dtype=torch.int32)
 dist.all_reduce(value,op=dist.ReduceOp.MIN)
 if not value.item():raise RuntimeError(message)
torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180))
try:
 assert dist.get_world_size()==world
 value=torch.tensor([float(rank)],device='cuda');dist.all_reduce(value);assert value.item()==world*(world-1)//2
 a=torch.randn(128,128,device='cuda',dtype=torch.bfloat16);assert torch.isfinite(a@a).all()
 def computation(z):return z.sin()+z.cos()
 z=torch.arange(1024,device='cuda',dtype=torch.float32)/1000
 compiled=torch.compile(computation,fullgraph=True)(z);torch.cuda.synchronize()
 assert torch.allclose(compiled,computation(z),rtol=1e-5,atol=1e-5)
 torch.manual_seed(42)
 model=FSDP(torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.ReLU(),torch.nn.Linear(32,16)),
  device_id=rank,cpu_offload=CPUOffload(offload_params=True),
  mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32))
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6);scheduler=constant_scheduler(optimizer)
 model(torch.ones(4,16,device='cuda')).float().sum().backward()
 params=list(model.parameters());original_grads=[p.grad.clone() for p in params]
 upstream=model.clip_grad_norm_(1.0)
 upstream_grads=[p.grad.clone() for p in params]
 for p,g in zip(params,original_grads):p.grad.copy_(g)
 norm=clip_global_l2(model,1.0,diagnostic=False)
 collective_check(torch.allclose(norm.float().cpu(),upstream.float().cpu(),rtol=1e-5,atol=1e-5), 'global L2 norm differs from upstream')
 collective_check(all(torch.allclose(p.grad,g,rtol=1e-5,atol=1e-6) for p,g in zip(params,upstream_grads)), 'clipped gradients differ from upstream')
 before=[p.detach().clone() for p in params]
 assert optimizer.param_groups[0]['lr']==1e-6
 optimizer.step()
 fields={f'param{i}':p.detach() for i,p in enumerate(params)}
 for i,state in enumerate(optimizer.state.values()):
  fields.update({f'adam{i}_{name}':v for name,v in state.items() if torch.is_tensor(v)})
 require_finite(fields,stage='eight-rank-after-real-adam-write',process_group=model.process_group)
 collective_check(any(not torch.equal(p,b) for p,b in zip(params,before)), 'AdamW did not change local parameters')
 scheduler.step();assert optimizer.param_groups[0]['lr']==1e-6
 # Finite FP32 gradients can overflow the ordinary norm without being NaN.
 for p in params:p.grad.fill_(1e20)
 norm=clip_global_l2(model,1.,diagnostic=False)
 collective_check(torch.isfinite(norm) and norm.item()>1e20, 'large finite norm rejected or miscomputed')
 clipped_norm=clip_global_l2(model,1.,diagnostic=False);collective_check(clipped_norm.item()<=1.00001, 'clipping exceeded norm limit')
 # One-rank NaN must reject on every rank before any clipping/write mutation.
 for p in params:p.grad.fill_(1.)
 if rank==2:params[0].grad.reshape(-1)[0]=float('nan')
 saved=[p.grad.clone() for p in params];rejected=False
 try:clip_global_l2(model,1.,diagnostic=False)
 except RuntimeError:rejected=True
 collective_check(rejected and all(torch.allclose(p.grad,g,equal_nan=True) for p,g in zip(params,saved)), 'NaN rejection mutated gradients')
 rejected=False
 try:require_finite({} if rank==1 else {'value':torch.ones(1,device='cuda')},stage='eight-rank-empty-rejection')
 except RuntimeError:rejected=True
 collective_check(rejected, 'empty rank was not rejected')
 dist.barrier()
 proof=dict(accepted=True,rank=rank,world_size=world,slurm_job_id=os.environ['SLURM_JOB_ID'],
  visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),nccl=True,bf16=True,cuda_jit=True,
  fsdp_cpu_offload=True,global_l2_upstream_equivalence=True,large_finite_l2=True,
  one_rank_nan_rejected_without_gradient_mutation=True,empty_rank_collective_rejection=True,
  real_adam_lr=1e-6,scheduler_scope='synthetic mechanics with legacy constant LR; not fresh200 warmup evidence',
  parameters_and_adam_finite=True,actual_grpo_updates=0,process_jit_cache=process_jit_cache,checked_unix=time.time())
 with (root/f'rank-{rank}.json').open('x') as f:
  json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
 dist.barrier()
 if rank==0:
  proofs=[json.loads((root/f'rank-{r}.json').read_text()) for r in range(world)]
  assert all(p['accepted'] is True and p['rank']==r and p['slurm_job_id']==os.environ['SLURM_JOB_ID'] for r,p in enumerate(proofs))
  print('NATIVE_RANK_MECHANICS_ACCEPTED',flush=True)
finally:dist.destroy_process_group()
