"""Owned-GPU synthetic compute/synchronization profile; never trains the main.

run(config, output, device) loads ONLY the official native block via Probe,
not worker.initialize (which validates the entire training feature bank).
No checkpoint, real feature payload, sampler, optimizer.step or actor call.
"""
import argparse,contextlib,gc,hashlib,json,math,os,statistics,time
from pathlib import Path
import torch
import torch.nn.functional as F


def view_indices(questions,device):
    local=torch.tensor([[0,1,2,3,4],[5,6,7,8,9],[0,2,4,6,8],[1,3,5,7,9]],device=device)
    return (local[None]+10*torch.arange(questions,device=device)[:,None,None]).reshape(questions*4,5)


def work_facts(model,questions=32):
    d=3584;m=18944;length=293;groups=questions*4;s=374;h=256;ff=1024;layers=len(model.group_head.layers)
    linear_weights=sum(module.weight.numel()for module in model.block.modules()if isinstance(module,torch.nn.Linear))
    native_linear=2*questions*10*length*linear_weights
    native_attention_dense=4*questions*10*length*length*d
    head_projection=2*groups*373*d*h
    head_attention_FFN=layers*(8*groups*s*h*h+4*groups*s*s*h+4*groups*s*h*ff)
    return dict(questions=questions,native_trajectories=questions*10,complete_G5_views=groups,real_shape=[questions*10,length,d],native_trainable_parameters=sum(p.numel()for p in model.block.parameters()),head_trainable_parameters=sum(p.numel()for p in model.group_head.parameters()),all_trainable_parameters=sum(p.numel()for p in model.parameters()),native_linear_weight_elements=linear_weights,head_layers=layers,head_dimension=h,head_heads=8,head_sequence_length=374,head_is_single_layer=layers==1,analytical_forward_FLOPs=dict(native_linear=native_linear,native_attention_dense_upper_bound=native_attention_dense,native_attention_causal_approx=native_attention_dense/2,head_projection=head_projection,head_attention_and_FFN=head_attention_FFN,total_using_dense_attention_bound=native_linear+native_attention_dense+head_projection+head_attention_FFN),training_forward_backward_FLOPs_approx=3*(native_linear+native_attention_dense/2+head_projection+head_attention_FFN),FLOPs_scope='Dense matrix multiplication model; normalization/activation/masks/validation not included; backward3x is approximate, not a measured total')


def _event():return torch.cuda.Event(enable_timing=True)


def _checks(model,vectorized=False):
    params=[p for p in model.parameters()if p.requires_grad]
    if any(p.grad is None for p in params):raise ValueError('Missing gradient')
    if vectorized:
        if not bool(torch.stack([torch.isfinite(p.grad).all()for p in params]).all()):raise ValueError('Nonfinite gradient')
        norms={name:float(torch.stack([p.grad.detach().float().square().sum()for p in module.parameters()]).sum().sqrt())for name,module in(('native',model.block),('head',model.group_head))}
    else:
        if any(not bool(torch.isfinite(p.grad).all())for p in params):raise ValueError('Nonfinite gradient')
        norms={name:math.sqrt(sum(float(p.grad.detach().float().square().sum())for p in module.parameters()))for name,module in(('native',model.block),('head',model.group_head))}
    if any(not math.isfinite(x)or x<=0 for x in norms.values()):raise ValueError('Zero/nonfinite gradient norm')
    clip=float(torch.nn.utils.clip_grad_norm_(params,1.))
    return dict(norms=norms,clip=clip,trainable_parameter_tensors=len(params))


def one_update(model,data,micro_questions,*,questions=32,defer_metrics=False,vectorized_checks=False):
    """Exactly mean32(mean4 MixedBCE); no weight update, source unchanged."""
    model.train()
    for p in model.parameters():
        if p.grad is None:p.grad=torch.zeros_like(p)
        else:p.grad.zero_()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=_event();end=_event();start.record();wall=time.perf_counter()
    spans=[];records=[];host_metrics=[];loss_value=0.
    for qstart in range(0,questions,micro_questions):
        qcount=min(micro_questions,questions-qstart);begin=qstart*10;stop=(qstart+qcount)*10
        batch={key:value[begin:stop]for key,value in data.items()};indices=view_indices(qcount,data['features'].device)
        target=torch.tensor([0.,0.,1.,1.],device=data['features'].device).repeat(qcount)
        nf0,nf1,hf0,hf1,bw0,boundary,bw1=[_event()for _ in range(7)]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            nf0.record();encoded=model.encode_trajectories(batch);nf1.record();hf0.record()
            logits=model.group_logits({k:v[indices]for k,v in encoded.items()}).float();hf1.record()
        per=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
        if not bool(torch.isfinite(per).all()):raise ValueError('Nonfinite same-objective loss')
        fired=[]
        def mark(gradient,event=boundary,seen=fired):event.record();seen.append(True)
        hook=encoded['hidden'].register_hook(mark);bw0.record();(per.sum()/(4*questions)).backward();bw1.record();hook.remove()
        if not fired:raise ValueError('Native backward boundary did not execute')
        records.append((nf0,nf1,hf0,hf1,bw0,boundary,bw1))
        if defer_metrics:host_metrics.append((per.detach(),logits.detach().ge(0)))
        else:
            t=time.perf_counter();numbers=per.detach().cpu().tolist();pred=logits.detach().ge(0).cpu().tolist();spans.append(time.perf_counter()-t);loss_value+=sum(numbers)/(4*questions)
        del encoded,logits,per,target,indices,batch
    if defer_metrics:
        t=time.perf_counter();loss_value=float(torch.cat([x[0]for x in host_metrics]).sum().cpu())/(4*questions);pred=torch.cat([x[1]for x in host_metrics]).cpu().tolist();spans.append(time.perf_counter()-t);del host_metrics
    checks0,checks1=_event(),_event();checks0.record();before=time.perf_counter();checks=_checks(model,vectorized_checks);checks_wall=time.perf_counter()-before;checks1.record();end.record();end.synchronize();elapsed=time.perf_counter()-wall
    phases=dict(native_forward=sum(a.elapsed_time(b)for a,b,_,_,_,_,_ in records),head_forward_and_G5_gather=sum(a.elapsed_time(b)for _,_,a,b,_,_,_ in records),head_backward_and_gather=sum(a.elapsed_time(b)for _,_,_,_,a,b,_ in records),native_backward=sum(a.elapsed_time(b)for _,_,_,_,_,a,b in records),gradient_checks_and_clipping=checks0.elapsed_time(checks1))
    return dict(wall_seconds=elapsed,cuda_event_stream_elapsed_seconds=start.elapsed_time(end)/1000,phase_cuda_event_ms=phases,host_metrics_wall_seconds=sum(spans),gradient_checks_wall_seconds=checks_wall,loss=loss_value,native_forward_calls=len(records),head_forward_calls=len(records),native_rows_per_call=10*micro_questions,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),**checks)


def compare_gradients(model,reference):
    error=reference_norm=0.;maximum=0.
    for name,p in model.named_parameters():
        r=reference[name].to(p.device);diff=p.grad.detach().float()-r;error+=float(diff.square().sum());reference_norm+=float(r.square().sum());maximum=max(maximum,float(diff.abs().max()));del r,diff
    return dict(relative_L2=math.sqrt(error/max(reference_norm,1e-30)),max_abs=maximum,comparison='Same objective and weights; BF16 batch/reduction differences allowed, not bit-exact')


def profiler_flops(model,data):
    """One question only; supported operators, NOT a complete FlashAttn count."""
    for p in model.parameters():p.grad=None
    batch={k:v[:10]for k,v in data.items()};idx=view_indices(1,data['features'].device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_flops=True)as profiler:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            encoded=model.encode_trajectories(batch);logits=model.group_logits({k:v[idx]for k,v in encoded.items()}).float()
        F.binary_cross_entropy_with_logits(logits,torch.tensor([0.,0.,1.,1.],device=logits.device)).backward();torch.cuda.synchronize()
    ops=[dict(operator=e.key,flops=int(e.flops or 0),calls=e.count)for e in profiler.key_averages()if e.flops]
    return dict(questions=1,native_trajectories=10,views=4,supported_operator_FLOPs=sum(x['flops']for x in ops),operators=ops,scope='PyTorch with_flops supported operators for forward+backward; fused FlashAttention/normalization may be uncounted; do not call this total model FLOPs')


def run(config,output,device=None):
    """Config is a dict or source-config path; no feature-bank traversal."""
    if not os.environ.get('SLURM_JOB_ID')or not torch.cuda.is_available():raise ValueError('Independent owned GPU allocation required')
    if isinstance(config,(str,Path)):
        source=Path(config);raw=source.read_bytes();config=json.loads(raw);source_ref=dict(path=str(source),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    else:config=dict(config);source_ref=None
    from .model import Probe
    from .native_mixed_group_classifier import NativeMixedGroupProbe,block_state_hash,OFFICIAL_REVISION
    device=torch.device(device or'cuda:0');torch.cuda.set_device(device)
    if device.type!='cuda'or 'H100'not in torch.cuda.get_device_name(device):raise ValueError('Actual H100 required for this profile')
    torch.manual_seed(9137);torch.backends.cuda.matmul.allow_tf32=False
    benchmark=config.get('compute_benchmark',{});micros=benchmark.get('micro_questions',[1,4,8,16,32]);warmup=benchmark.get('warmup',1);repeats=benchmark.get('repeats',3)
    if warmup<1 or repeats<3 or any(type(x)is not int or x not in(1,4,8,16,32)for x in micros)or 1 not in micros:raise ValueError('Warmup>=1, repeats>=3 and declared complete microquestion cases required')
    setup=time.perf_counter();legacy=Probe(config['model_dir'],273,20)
    model=NativeMixedGroupProbe.from_official_probe(legacy,official_inventory=config['official_inventory'],expected_block_hash=block_state_hash(legacy.block),head_seed=config.get('head_seed',9137),official_revision=OFFICIAL_REVISION).to(device);del legacy;gc.collect();setup_seconds=time.perf_counter()-setup
    features=torch.randn(32,10,293,3584,device=device,dtype=torch.bfloat16);features[:,:,:273]=features[:,:1,:273].clone().expand(-1,10,-1,-1);features=features.reshape(320,293,3584)
    data=dict(features=features,attention_mask=torch.ones(320,293,device=device,dtype=torch.bool),position_ids=torch.arange(293,device=device).expand(320,-1).contiguous())
    facts=work_facts(model);cases=[];reference=None
    settings=[dict(micro_questions=m,defer_metrics=False,vectorized_checks=False)for m in micros]
    if benchmark.get('synchronization_ablation',True):settings.append(dict(micro_questions=1,defer_metrics=True,vectorized_checks=True))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    for setting in settings:
        item=dict(setting,status='RUNNING');cases.append(item)
        try:
            for _ in range(warmup):one_update(model,data,**setting)
            item['samples']=[one_update(model,data,**setting)for _ in range(repeats)]
            item['median_wall_seconds']=statistics.median(x['wall_seconds']for x in item['samples']);item['median_cuda_event_seconds']=statistics.median(x['cuda_event_stream_elapsed_seconds']for x in item['samples'])
            if reference is None:reference={name:p.grad.detach().float().cpu().clone()for name,p in model.named_parameters()};item['gradient_comparison']=dict(reference=True)
            else:item['gradient_comparison']=compare_gradients(model,reference)
            item['status']='COMPLETED';item['speedup_vs_micro1']=cases[0]['median_wall_seconds']/item['median_wall_seconds']
        except torch.OutOfMemoryError as exc:
            item['status']='CUDA_OOM_NO_TRUNCATION_OR_DTYPE_FALLBACK';item['error']=str(exc)[:1000]
            for p in model.parameters():p.grad=None
            gc.collect();torch.cuda.empty_cache()
        result=dict(status='SYNTHETIC_REAL_SHAPE_COMPUTE_PROFILE_IN_PROGRESS',source_config=source_ref,job_id=os.environ['SLURM_JOB_ID'],device=torch.cuda.get_device_name(device),setup_seconds_excluded_from_timings=setup_seconds,facts=facts,cases=cases,real_training_payload_read=False,old_probe_checkpoint_read=False,optimizer_steps=0,main_process_or_source_modified=False,CUDA_Event_scope='stream elapsed including possible host enqueue starvation; not a summed kernel-utilization counter',epoch=time.time())
        temporary=output.with_suffix(output.suffix+'.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(output)
    if benchmark.get('profile_supported_flops',True):
        try:result['supported_operator_profile']=profiler_flops(model,data)
        except Exception as exc:result['supported_operator_profile']=dict(status='UNAVAILABLE',error=repr(exc))
    result['status']='SYNTHETIC_REAL_SHAPE_COMPUTE_PROFILE_COMPLETE';result['epoch']=time.time();output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-config',required=True);p.add_argument('--output',required=True);args=p.parse_args();value=run(args.source_config,args.output);print(json.dumps(dict(status=value['status'],output=args.output,cases=[dict(micro_questions=x['micro_questions'],status=x['status'],median_wall_seconds=x.get('median_wall_seconds'))for x in value['cases']])))
