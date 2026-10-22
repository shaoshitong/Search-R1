"""Pinned-FSDP L2 gradient clipping with FP64 accumulation and diagnostics.

Same global sharded plus once-counted replicated L2 formula and epsilon as
PyTorch2.4 FSDP; preserves clipping threshold. Never sanitizes invalid gradients.
"""
import json
import math
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from search_g1.numerics import require_finite


def _partition(module):
    if not isinstance(module,FSDP):return [],list(module.parameters()),None,None
    assert module._is_root and hasattr(module,'_all_handles'), 'Initialized root FSDP required'
    sharded=set();replicated=set()
    for handle in module._all_handles:
        target=sharded if handle.uses_sharded_strategy else replicated
        target.update(handle.flat_param._params if handle._use_orig_params else [handle.flat_param])
    replicated.update(p for p in module.parameters() if p not in sharded and p not in replicated)
    return list(sharded),list(replicated),module.process_group,module.compute_device


def _sum_squares(params,device,chunk_size):
    total=torch.zeros((),device=device,dtype=torch.float64)
    maximum=torch.zeros_like(total)
    for p in params:
        if p.grad is None:continue
        flat=p.grad.detach().reshape(-1)
        for start in range(0,flat.numel(),chunk_size):
            x=flat[start:start+chunk_size].to(dtype=torch.float64)
            total.add_(torch.sum(x*x).to(device))
            if x.numel():maximum=torch.maximum(maximum,x.abs().max().to(device))
    return total,maximum


@torch.no_grad()
def clip_global_l2(module,max_norm,*,chunk_size=1048576,diagnostic=True,context=None,recoverable=False,check_finite=True):
    assert math.isfinite(max_norm) and max_norm>0 and chunk_size>0
    sharded,replicated,group,compute_device=_partition(module)
    params=list(dict.fromkeys(sharded+replicated))
    grads=[p.grad for p in params if p.grad is not None]
    communication_device=(compute_device if isinstance(module,FSDP) else (grads[0].device if grads else 'cpu'))
    supported=all(g.dtype in (torch.float32,torch.bfloat16,torch.float16) and g.is_contiguous() for g in grads)
    if check_finite:
        require_finite({'supported_dtype_layout':torch.tensor(int(supported),device=communication_device)},stage='before_gradient_layout_mutation',process_group=group,positive=('supported_dtype_layout',))
        require_finite({str(i):g for i,g in enumerate(grads)},stage='before_fp64_gradient_clipping',process_group=group,recoverable=recoverable)
    elif not supported or not grads:
        # Keep the structural contract without launching a numerical scan or a
        # scalar device-to-host check. This does not alter the clipping formula.
        raise RuntimeError('Gradient clipping requires nonempty contiguous floating gradients')
    shard_sq,shard_max=_sum_squares(sharded,communication_device,chunk_size)
    replica_sq,replica_max=_sum_squares(replicated,communication_device,chunk_size)
    local_sharded_l2=torch.sqrt(shard_sq.clone())
    local_sharded_max=shard_max.clone()
    if sharded:dist.all_reduce(shard_sq,group=group)
    norm=torch.sqrt(shard_sq+replica_sq)
    maximum=torch.maximum(shard_max,replica_max)
    if dist.is_initialized():dist.all_reduce(maximum,op=dist.ReduceOp.MAX,group=group)
    if check_finite:
        require_finite({'grad_norm':norm,'max_abs':maximum},stage='before_fp64_clip_write',process_group=group,recoverable=recoverable)
    coefficient=torch.clamp(float(max_norm)/(norm+1e-6),max=1.)
    if diagnostic and (not dist.is_initialized() or dist.get_rank()==0):
        print('SEARCH_G1_GRADIENT_NORM_DIAGNOSTIC='+json.dumps(dict(global_l2_fp64=norm.item(),max_abs=maximum.item(),local_rank0_sharded_l2_fp64=local_sharded_l2.item(),local_rank0_sharded_max_abs=local_sharded_max.item(),clip_coefficient_fp64=coefficient.item(),gradient_dtypes=sorted({str(g.dtype) for g in grads}),sharded_parameter_count=len(sharded),replicated_parameter_count=len(replicated),context=context,semantics='finite magnitude diagnostic; overflow-safe accumulation does not prove stable training')),flush=True)
    for grad in grads:
        flat=grad.view(-1)
        for start in range(0,flat.numel(),chunk_size):
            x=flat[start:start+chunk_size]
            x.copy_((x.to(dtype=torch.float64)*coefficient.to(x.device)).to(x.dtype))
    if check_finite:
        require_finite({str(i):g for i,g in enumerate(grads)},stage='after_fp64_gradient_clipping',process_group=group,recoverable=recoverable)
    post_shard,_=_sum_squares(sharded,communication_device,chunk_size)
    post_replica,_=_sum_squares(replicated,communication_device,chunk_size)
    if sharded:dist.all_reduce(post_shard,group=group)
    module._last_clipped_global_l2=torch.sqrt(post_shard+post_replica)
    if check_finite:
        require_finite({'actual_post_clip_norm':module._last_clipped_global_l2},stage='measured_after_gradient_clipping',process_group=group,recoverable=recoverable)
    return norm
