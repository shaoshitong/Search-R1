"""Isolated KV4 versus repeated-KV28 FA3 correctness and warmed CUDA timing.

No policy/checkpoint/optimizer/rollout is loaded. Both FA3 paths receive the
same BF16 Q28/K4/V4 leaves; the expanded path repeats K/V before the kernel and
autograd reduces the seven head gradients back to those shared leaves. The
independent reference uses explicit FP32 SDPBackend.MATH. Timings include the
head repeat and its backward, excluding input generation and cold warmup.
"""
import argparse
import gc
import itertools
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import time

sys.dont_write_bytecode=True

LIMITS=dict(output_relative_l2=.01,gradient_relative_l2=.02,
            max_abs_relative_to_reference_peak=.02,max_abs_floor=.001)


def _need(value,message):
    if not value:raise ValueError('FA3_COMPACT_GQA_'+message)


def _comparisons(reference,actual,names):
    from reproduction.operations.easy7b_20261021.qualify_fa3_sdpa_cuda import _metrics
    values={name:_metrics(left,right) for name,left,right in zip(names,reference,actual)}
    checks={name:bool(row['finite'] and row['relative_l2']<=LIMITS[
                'output_relative_l2' if name=='output' else 'gradient_relative_l2'] and
                row['max_abs']<=LIMITS['max_abs_floor']+
                LIMITS['max_abs_relative_to_reference_peak']*row['reference_max_abs'])
            for name,row in values.items()}
    return dict(metrics=values,checks=checks,passed=all(checks.values()))


def expanded_kv(value):
    # Exactly the head-major layout made by verl repeat_kv then its transpose:
    # [T,4,D] -> [4,T,D] -> [28,T,D] -> [T,28,D].
    return value.transpose(0,1).repeat_interleave(7,dim=0).transpose(0,1)


def _forward(kernel,values,cu,maximum,expanded):
    q,k,v=values
    if expanded:k,v=expanded_kv(k),expanded_kv(v)
    return kernel(q,k,v,cu,cu,maximum,maximum,dropout_p=0.,softmax_scale=None,
                  causal=True,window_size=(-1,-1),softcap=0.,deterministic=False)


def candidate(kernel,values,cu,maximum,upstream,expanded):
    tensors=[value.detach().clone().requires_grad_(True) for value in values]
    out=_forward(kernel,tensors,cu,maximum,expanded)
    out.backward(upstream)
    return [out.detach()]+[value.grad.detach() for value in tensors]


def warmed_timing(kernel,values,cu,maximum,upstream,expanded,*,warmup=5,repeats=15):
    import torch
    _need(type(warmup) is int and warmup>=2 and type(repeats) is int and repeats>=5,
          'WARMED_REPEATED_TIMING_REQUIRED')
    def iteration():
        # Fresh leaf objects reuse the immutable input storage. This prevents
        # the adapter's per-forward diagnostic hooks accumulating on reused
        # leaves, and matches production's fresh projected Q/K/V graph.
        tensors=[value.detach().requires_grad_(True) for value in values]
        out=_forward(kernel,tensors,cu,maximum,expanded)
        out.backward(upstream)
        del out,tensors
    for _ in range(warmup):iteration()
    torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    samples=[]
    for _ in range(repeats):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();iteration();end.record();end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    result=dict(mode='expanded_KV28' if expanded else 'compact_KV4',warmup=warmup,repeats=repeats,
                milliseconds=samples,median_ms=statistics.median(samples),mean_ms=statistics.mean(samples),
                min_ms=min(samples),max_ms=max(samples),baseline_allocated_bytes=baseline,
                peak_incremental_allocated_bytes=torch.cuda.max_memory_allocated()-baseline,
                scope='CUDA-event attention forward+backward; includes expanded repeat and shared-KV gradient reduction; '
                      'no input generation, SDPA reference, projection, optimizer, FSDP, or cold warmup')
    return result


def _math_output(values,lengths):
    import torch
    from torch.nn.attention import SDPBackend,sdpa_kernel
    q,k,v=values;parts=[];start=0
    for length in lengths:
        end=start+length
        qs=q[start:end].float().transpose(0,1).unsqueeze(0)
        ks=k[start:end].float().repeat_interleave(7,dim=1).transpose(0,1).unsqueeze(0)
        vs=v[start:end].float().repeat_interleave(7,dim=1).transpose(0,1).unsqueeze(0)
        with torch.autocast(device_type=q.device.type,enabled=False),sdpa_kernel(SDPBackend.MATH):
            result=torch.nn.functional.scaled_dot_product_attention(qs,ks,vs,
                dropout_p=0.,is_causal=True,scale=q.shape[-1]**-.5)
        parts.append(result.squeeze(0).transpose(0,1));start=end
    _need(start==q.shape[0],'MATH_SEQUENCE_TOTAL_DIFFERS')
    return torch.cat(parts)


def projected_case(kernel,generator):
    """Random one-block production-size Q/K/V/O projections and Qwen2 RoPE.

    This measures the chain-rule into real 3584x3584 / 512x3584 parameter
    shapes. It is not a pretrained full-actor, loss, FSDP or optimizer test.
    The math path keeps the same BF16 projections/RoPE/output projection,
    replacing only attention with FP32 SDPA math and casting its output back.
    """
    import torch
    from torch.nn import functional as F
    lengths=[31,65];total=sum(lengths);width=3584;head=128
    cu=torch.tensor([0]+list(itertools.accumulate(lengths)),device='cuda',dtype=torch.int32)
    x=torch.randn(total,width,device='cuda',generator=generator).to(torch.bfloat16)
    weights=[(torch.randn(out,width,device='cuda',generator=generator)/math.sqrt(width)).to(torch.bfloat16)
             for out in (3584,512,512,3584)]
    upstream=torch.randn(total,width,device='cuda',generator=generator).to(torch.bfloat16)
    positions=torch.cat([torch.arange(length,device='cuda') for length in lengths])
    inv=1./(1000000.**(torch.arange(0,head,2,device='cuda').float()/head))
    angles=positions.float()[:,None]*inv[None,:]
    cos=torch.cat((angles,angles),dim=-1).cos().to(torch.bfloat16)[:,None,:]
    sin=torch.cat((angles,angles),dim=-1).sin().to(torch.bfloat16)[:,None,:]
    def calculate(mode):
        variables=[value.detach().clone().requires_grad_(True) for value in [x]+weights]
        hidden,wq,wk,wv,wo=variables
        q=F.linear(hidden,wq).view(total,28,head)
        k=F.linear(hidden,wk).view(total,4,head);v=F.linear(hidden,wv).view(total,4,head)
        def rope(value):
            a,b=value.chunk(2,dim=-1)
            return value*cos+torch.cat((-b,a),dim=-1)*sin
        q,k=rope(q),rope(k)
        if mode=='math':out=_math_output([q,k,v],lengths).to(torch.bfloat16)
        else:out=_forward(kernel,[q,k,v],cu,max(lengths),mode=='expanded')
        output=F.linear(out.reshape(total,width),wo)
        output.backward(upstream)
        return [output.detach()]+[value.grad.detach() for value in variables]
    compact=calculate('compact');expanded=calculate('expanded');reference=calculate('math')
    names=('output','input_gradient','q_proj_gradient','k_proj_gradient','v_proj_gradient','o_proj_gradient')
    ce=_comparisons(expanded,compact,names);cm=_comparisons(reference,compact,names)
    return dict(lengths=lengths,hidden_size=width,q_heads=28,kv_heads=4,head_dim=head,
                rope_theta=1000000.,projection_dtype='bfloat16',sdpa_attention_dtype='float32',
                compact_vs_expanded=ce,compact_vs_math=cm,passed=ce['passed'] and cm['passed'],
                scope='one random attention block Q/K/V/O projections+RoPE+parameter/input gradients; '
                      'not pretrained full-actor or optimizer equivalence')


def run(base,*,warmup=5,repeats=15):
    import torch
    from flash_attn import flash_attn_varlen_func
    from search_g1 import flash3_attention as adapter
    from reproduction.operations.easy7b_20261021.qualify_fa3_sdpa_cuda import _reference
    base=Path(base).resolve(strict=True)
    manifest=json.loads((base/'logs/run-manifest.json').read_text())
    _need(torch.cuda.is_available() and torch.cuda.device_count()==1 and
          'H100' in torch.cuda.get_device_name(0),'ISOLATED_ONE_H100_REQUIRED')
    _need(manifest.get('training_world_size')==4,'NATIVE4_SOURCE_BINDING_REQUIRED')
    torch.cuda.set_device(0)
    backend=adapter.load_backend(manifest);audit=adapter._Audit(backend)
    kernel=adapter._Kernel(flash_attn_varlen_func,backend,audit,'varlen')
    generator=torch.Generator(device='cuda').manual_seed(747219431)
    before_tf32=torch.backends.cuda.matmul.allow_tf32;before_cudnn=torch.backends.cudnn.allow_tf32
    report=dict(format='easy7b_compact_gqa_qualification_v1',started_unix=time.time(),
                source_commit=manifest['source_commit'],diagnostic_job=os.environ.get('SLURM_JOB_ID'),
                hostname=socket.gethostname(),torch=torch.__version__,device=torch.cuda.get_device_name(0),
                provenance=backend.provenance,model_loaded=False,checkpoint_loaded=False,
                optimizer_calls=0,scheduler_calls=0,production_modified=False,automatic_switch=False,
                tolerances=LIMITS,cases=[])
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        for label,lengths,qscale,kscale,vscale in (
                ('production_micro8',[31,127,257,513,1024,1500,2343,4096],1.,1.,1.),
                ('high_magnitude',[31,127,257,513,1024,1500,2343,4096],3.5,26.7,.25)):
            total=sum(lengths);cu=torch.tensor([0]+list(itertools.accumulate(lengths)),device='cuda',dtype=torch.int32)
            values=[(torch.randn(total,heads,128,device='cuda',generator=generator)*scale).to(torch.bfloat16)
                    for heads,scale in ((28,qscale),(4,kscale),(4,vscale))]
            upstream=torch.randn(total,28,128,device='cuda',generator=generator,dtype=torch.bfloat16)
            compact=candidate(kernel,values,cu,max(lengths),upstream,False)
            expanded=candidate(kernel,values,cu,max(lengths),upstream,True)
            reference=_reference(values,lengths,upstream)
            names=('output','dq','shared_dk','shared_dv')
            ce=_comparisons(expanded,compact,names);cm=_comparisons(reference,compact,names)
            em=_comparisons(reference,expanded,names)
            changed=[value.clone() for value in values]
            for value in changed:value[:lengths[0]].add_(3)
            with torch.no_grad():isolated=_forward(kernel,changed,cu,max(lengths),False)
            isolation=bool(torch.equal(isolated[lengths[0]:],compact[0][lengths[0]:]))
            changed=[value.clone() for value in values]
            boundary=total-lengths[-1]+lengths[-1]//2
            for value in changed[1:]:value[boundary:].add_(3)
            with torch.no_grad():future=_forward(kernel,changed,cu,max(lengths),False)
            causal=bool(torch.equal(future[total-lengths[-1]:boundary],compact[0][total-lengths[-1]:boundary]))
            del compact,expanded,reference,changed,isolated,future
            gc.collect();torch.cuda.empty_cache()
            # ABBA order limits a simple warm-clock/order bias. Cold calls are
            # excluded separately for each mode, with all raw samples retained.
            timings=[warmed_timing(kernel,values,cu,max(lengths),upstream,mode,
                                  warmup=warmup,repeats=repeats) for mode in (True,False,False,True)]
            compact_ms=statistics.median([t for row in timings if row['mode']=='compact_KV4' for t in row['milliseconds']])
            expanded_ms=statistics.median([t for row in timings if row['mode']=='expanded_KV28' for t in row['milliseconds']])
            report['cases'].append(dict(label=label,lengths=lengths,q_heads=28,native_kv_heads=4,
                expanded_kv_heads=28,head_dim=128,scales=dict(q=qscale,k=kscale,v=vscale),
                compact_input_strides=[list(value.stride()) for value in values],
                expanded_kv_strides=list(expanded_kv(values[1]).stride()),
                compact_vs_expanded=ce,compact_vs_fp32_math=cm,expanded_vs_fp32_math=em,
                other_sequences_bitwise_unchanged=isolation,causal_prefix_bitwise_unchanged=causal,
                timings=timings,warmed_compact_median_ms=compact_ms,warmed_expanded_median_ms=expanded_ms,
                expanded_over_compact_speed_ratio=expanded_ms/compact_ms,
                passed=ce['passed'] and cm['passed'] and em['passed'] and isolation and causal))
            del values,upstream
            gc.collect();torch.cuda.empty_cache()
        report['projected_parameter_gradients']=projected_case(kernel,generator)
        report['execution']=audit.report(synchronize=True)
        report['passed']=all(row['passed'] for row in report['cases']) and report['projected_parameter_gradients']['passed']
        report['status']='COMPACT_GQA_FORWARD_BACKWARD_PASS' if report['passed'] else 'COMPACT_GQA_NUMERICAL_REJECTED'
        report['scope']=('independent one-H100 packed attention and random one-block parameter-gradient qualification; '
                         'not full-model numerical/quality or whole-training speedup acceptance')
        report['finished_unix']=time.time()
        return report
    finally:
        torch.backends.cuda.matmul.allow_tf32=before_tf32;torch.backends.cudnn.allow_tf32=before_cudnn


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True)
    parser.add_argument('--warmup',type=int,default=5);parser.add_argument('--repeats',type=int,default=15)
    args=parser.parse_args(argv)
    try:
        report=run(args.base,warmup=args.warmup,repeats=args.repeats);code=0 if report['passed'] else 2
    except Exception as error:
        report=dict(status='COMPACT_GQA_DIAGNOSTIC_FAILED',passed=False,error_type=type(error).__name__,
                    error=str(error),production_modified=False,automatic_switch=False);code=1
    print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
