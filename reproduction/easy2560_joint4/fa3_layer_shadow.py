"""One isolated FA2 actor trajectory, with same-QKV per-layer FA3 shadows.

Every attention result returned to the model is the original FA2 result. This
separates individual kernel differences from their propagation through28
decoder layers. No backwards, updates, RNG draws, tensor files or qualification
acceptance are performed here. The independent caller verifies model bytes.
"""
import copy
import functools
import math
import time
import types

import torch


def _need(condition,message):
    if not condition:raise ValueError('FA3_LAYER_SHADOW_'+message)


def _clone(function,replacements):
    _need(isinstance(function,types.FunctionType) and set(replacements)<=set(function.__code__.co_names),
          'ORIGINAL_PYTHON_BINDINGS_REQUIRED')
    namespace=dict(function.__globals__);namespace.update(replacements)
    result=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__;functools.update_wrapper(result,function)
    return result


def _error(reference,actual):
    _need(reference.shape==actual.shape,'COMPARISON_SHAPES_DIFFER')
    left=reference.detach().float();right=actual.detach().float();delta=right-left
    finite=bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    norm=float(torch.linalg.vector_norm(left)) if finite else None
    difference=float(torch.linalg.vector_norm(delta)) if finite else None
    return dict(elements=left.numel(),all_finite=finite,
        max_absolute_error=float(delta.abs().max()) if finite and left.numel() else None,
        root_mean_square_error=difference/math.sqrt(left.numel()) if finite and left.numel() else None,
        reference_l2=norm,difference_l2=difference,
        relative_l2=difference/norm if finite and norm else (0. if finite and difference==0 else None))


def _tensor_stats(value):
    view=value.detach().float();finite=bool(torch.isfinite(view).all())
    return dict(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),
        device=str(value.device),contiguous=value.is_contiguous(),all_finite=finite,
        absolute_max=float(view.abs().max()) if finite else None,
        root_mean_square=float(view.square().mean().sqrt()) if finite else None)


def _plain(value):
    if isinstance(value,torch.Tensor):
        _need(value.numel()==1,'NONSCALAR_OPTION_CANNOT_BE_RECORDED')
        value=value.item()
    if isinstance(value,(tuple,list)):return [_plain(item) for item in value]
    _need(value is None or type(value) in (bool,int,float,str),'UNRECOGNIZED_KERNEL_OPTION')
    return value


def _repeat_proof(value,original_kv_heads):
    heads=value.shape[-2]
    if heads%original_kv_heads:
        return dict(applicable=False,current_heads=heads,original_kv_heads=original_kv_heads)
    group=heads//original_kv_heads
    shaped=value.reshape(*value.shape[:-2],original_kv_heads,group,value.shape[-1])
    return dict(applicable=True,current_heads=heads,original_kv_heads=original_kv_heads,repeat_group_size=group,
                exact_repeats=bool(torch.equal(shaped,shaped[...,:1,:].expand_as(shaped))))


def _query_indices(length,count):
    if count==0:return []
    values=[0] if count==1 else ([0,length-1] if count==2 else [0,length//2,length-1])
    return sorted(set(values))


def _fp32_sample(q,k,v,query_indices,options):
    """Use the complete eligible K/V sequence, not a cropped prefix target."""
    query=q[query_indices].float();keys=k.float();values=v.float()
    group=q.shape[-2]//k.shape[-2]
    if group!=1:
        keys=keys.repeat_interleave(group,dim=-2);values=values.repeat_interleave(group,dim=-2)
    # The only GEMMs here have FP32 operands. Explicitly disable TF32 locally
    # so this reference does not inherit a diagnostic process's global setting.
    original_tf32=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        # This helper executes inside the actor's BF16 autocast scope. Merely
        # converting operands to float() does not stop autocast from lowering
        # einsum back to BF16; the reference needs its own disabled scope.
        with torch.autocast(device_type=q.device.type,enabled=False):
            scores=torch.einsum('qhd,khd->hqk',query,keys)
            scale=options.get('softmax_scale')
            scores.mul_(q.shape[-1]**-.5 if scale is None else scale)
            softcap=options.get('softcap',0.0)
            if softcap:scores=softcap*torch.tanh(scores/softcap)
            aligned=torch.tensor(query_indices,device=q.device)+k.shape[0]-q.shape[0]
            positions=torch.arange(k.shape[0],device=q.device)
            allowed=torch.ones((len(query_indices),k.shape[0]),dtype=torch.bool,device=q.device)
            if options.get('causal',False):allowed&=positions.unsqueeze(0)<=aligned.unsqueeze(1)
            left,right=options.get('window_size',(-1,-1))
            if left>=0:allowed&=positions.unsqueeze(0)>=aligned.unsqueeze(1)-left
            if right>=0:allowed&=positions.unsqueeze(0)<=aligned.unsqueeze(1)+right
            scores=scores.masked_fill(~allowed.unsqueeze(0),float('-inf'))
            probabilities=torch.nan_to_num(torch.softmax(scores,dim=-1),nan=0.)
            return torch.einsum('hqk,khd->qhd',probabilities,values)
    finally:torch.backends.cuda.matmul.allow_tf32=original_tf32


def _sequence_reference(values,fa2,repeat,fa3,kind,query_count):
    q,k,v=(values[name] for name in ('q','k','v'))
    if not query_count:return dict(enabled=False,query_count=0)
    sequences=[]
    if kind=='varlen':
        cuq=values['cu_seqlens_q'].tolist();cuk=values['cu_seqlens_k'].tolist()
        _need(len(cuq)==len(cuk) and cuq[0]==cuk[0]==0 and cuq[-1]==q.shape[0] and
              cuk[-1]==k.shape[0],'PACKED_SEQUENCE_ENDPOINTS_DIFFER')
        for index,(a,b,c,d) in enumerate(zip(cuq[:-1],cuq[1:],cuk[:-1],cuk[1:])):
            sequences.append((index,q[a:b],k[c:d],v[c:d],fa2[a:b],repeat[a:b],fa3[a:b]))
    else:
        for index in range(q.shape[0]):sequences.append((index,q[index],k[index],v[index],fa2[index],repeat[index],fa3[index]))
    rows=[];ref_all=[];old_all=[];repeat_all=[];new_all=[]
    for index,qs,ks,vs,old,repeated,new in sequences:
        _need(qs.shape[0]>0 and ks.shape[0]>0,'EMPTY_PACKED_SEQUENCE')
        indices=_query_indices(qs.shape[0],query_count);reference=_fp32_sample(qs,ks,vs,indices,values)
        rows.append(dict(sequence=index,query_length=qs.shape[0],key_length=ks.shape[0],query_indices=indices,
            complete_key_sequence_used=True,fa2=_error(reference,old[indices]),
            fa2_repeat=_error(reference,repeated[indices]),fa3=_error(reference,new[indices])))
        ref_all.append(reference);old_all.append(old[indices]);repeat_all.append(repeated[indices]);new_all.append(new[indices])
    reference=torch.cat(ref_all)
    return dict(enabled=True,queries_per_sequence_at_most=query_count,query_count=reference.shape[0],
        math='FP32 QK/softmax/V; TF32 disabled; original full sequence + causal/window/scale/softcap',
        no_rng_draws=True,sequences=rows,
        all_sampled_queries=dict(fa2=_error(reference,torch.cat(old_all)),
                                fa2_repeat=_error(reference,torch.cat(repeat_all)),
                                fa3=_error(reference,torch.cat(new_all))))


class _ShadowKernel:
    def __init__(self,layer_index,kind,original,candidate,rows,query_count,kv_heads):
        self.layer_index=layer_index;self.kind=kind;self.original=original;self.candidate=candidate
        self.rows=rows;self.query_count=query_count;self.kv_heads=kv_heads

    def __call__(self,*args,**kwargs):
        bound=self.candidate.signature.bind(*args,**kwargs);bound.apply_defaults();values=dict(bound.arguments)
        # The original result alone advances the model. Both shadows see the
        # same actual QKV objects, cu lengths and HF-supplied arguments.
        first=self.original(*args,**kwargs);repeated=self.original(*args,**kwargs)
        candidate=self.candidate(*args,**kwargs)
        _need(isinstance(first,torch.Tensor) and isinstance(repeated,torch.Tensor),'ORIGINAL_FA2_TENSOR_REQUIRED')
        q,k,v=(values[name] for name in ('q','k','v'))
        options={key:_plain(value) for key,value in values.items()
                 if key not in ('q','k','v','cu_seqlens_q','cu_seqlens_k')}
        row=dict(layer_index=self.layer_index,kernel=self.kind,
            q=_tensor_stats(q),k=_tensor_stats(k),v=_tensor_stats(v),actual_fa2_options=options,
            effective_softmax_scale=q.shape[-1]**-.5 if values.get('softmax_scale') is None else values['softmax_scale'],
            fa3_pinned_interface_defaults=dict(num_splits=1,pack_gqa=None,sink_token_length=0,
                                             q_descale=None,k_descale=None,v_descale=None,sm_margin=0),
            kv_repeat_proof=dict(k=_repeat_proof(k,self.kv_heads),v=_repeat_proof(v,self.kv_heads)),
            fa2_repeat_output_noise=_error(first,repeated),fa3_vs_same_input_fa2=_error(first,candidate),
            model_continues_with='original_FA2_output_only',forward_grad_enabled=torch.is_grad_enabled())
        if self.kind=='varlen':
            cuq=values['cu_seqlens_q'].tolist();cuk=values['cu_seqlens_k'].tolist()
            row['packed_sequences']=dict(cu_seqlens_q=cuq,cu_seqlens_k=cuk,
                query_lengths=[b-a for a,b in zip(cuq[:-1],cuq[1:])],key_lengths=[b-a for a,b in zip(cuk[:-1],cuk[1:])])
        row['fp32_sample_reference']=_sequence_reference(values,first,repeated,candidate,self.kind,self.query_count)
        self.rows.append(row)
        return first


def run_layer_shadow(actor,micro_batch,temperature,*,attention_module=None,adapter=None,fp32_queries_per_sequence=3):
    """Return one rank's JSON-only evidence; caller gathers ranks and hashes weights.

    Requires the already installed FA3 actor solely to obtain its private bound
    kernels. No training entry calls this helper. Existing failed full-model
    qualification remains rejected regardless of these diagnostic measurements.
    """
    from search_g1 import sparse_policy_qualification as numerical
    if adapter is None:from search_g1 import flash3_attention as adapter
    _need(type(fp32_queries_per_sequence) is int and 0<=fp32_queries_per_sequence<=3,'BOUNDED_QUERY_COUNT_REQUIRED')
    module=actor.actor_module;raw=module if attention_module is None else attention_module
    plans=getattr(raw,'_search_g1_flash3_forwards',())
    _need(getattr(raw,'_search_g1_flash3_installed',False) and len(plans)==28,'INSTALLED28_LAYER_ACTOR_REQUIRED')
    _need(all(p.grad is None for p in module.parameters()),'PENDING_GRADIENTS_FORBIDDEN')
    original_kv_heads=getattr(raw.config,'num_key_value_heads',4)
    _need(type(original_kv_heads) is int and original_kv_heads>0,'ORIGINAL_KV_HEAD_COUNT_REQUIRED')
    device=micro_batch['input_ids'].device;saved_rng=numerical._rng(device)
    attributes={name:copy.deepcopy(getattr(actor,name)) if hasattr(actor,name) else numerical._MISSING
                for name in numerical._LOG_ATTRIBUTES}
    modes=[(child,child.training) for child in module.modules()]
    forwards=[(layer,layer.forward) for layer,_,_ in plans];rows=[];patches=[]
    before=adapter.execution_report(raw,synchronize=device.type=='cuda');started=time.time()
    for index,(layer,original,private) in enumerate(plans):
        helper=original.__func__.__globals__['_flash_attention_forward']
        candidate_helper=private.__func__.__globals__['_flash_attention_forward']
        replacements={name:_ShadowKernel(index,kind,helper.__globals__[name],candidate_helper.__globals__[name],
                rows,fp32_queries_per_sequence,original_kv_heads)
            for name,kind in (('flash_attn_func','dense'),('flash_attn_varlen_func','varlen'))}
        shadow_helper=_clone(helper,replacements)
        shadow_forward=_clone(original.__func__,{'_flash_attention_forward':shadow_helper})
        patches.append((layer,types.MethodType(shadow_forward,layer)))
    output=None
    try:
        for layer,forward in patches:layer.forward=forward
        module.train(True)
        with torch.no_grad():output=actor._forward_micro_batch(micro_batch,temperature=temperature,compute_entropy=False)
        after=adapter.execution_report(raw,synchronize=device.type=='cuda')
        _need(len(rows)==28 and sorted(row['layer_index'] for row in rows)==list(range(28)),
              'EXACTLY_ONE_KERNEL_CALL_PER_EACH28_LAYERS_REQUIRED')
        _need(after['kernel_backward_success_returns']==before['kernel_backward_success_returns'] and
              all(p.grad is None for p in module.parameters()),'BACKWARD_OR_PENDING_GRADIENT_OBSERVED')
        _need(sum(after['forward_success_returns'].values())-sum(before['forward_success_returns'].values())==28,
              'ACTUAL28_SHADOW_FA3_FORWARD_RETURNS_REQUIRED')
    finally:
        output=None
        for layer,forward in forwards:layer.forward=forward
        for child,mode in modes:child.training=mode
        numerical._restore_attributes(actor,attributes);numerical._restore_rng(saved_rng,device)
    return dict(format='easy7b_same_input_layer_FA2_FA3_shadow_v1',
        acceptance_status='DIAGNOSTIC_ONLY_NOT_QUALIFICATION',numerical_qualification_accepted=False,
        existing_full_model_rejection_unchanged=True,real_actor_forward_calls=1,backward_calls=0,
        optimizer_calls=0,scheduler_calls=0,actual_attention_layers=28,layer_reports=rows,
        model_continues_with='original_FA2_output_only',same_exact_qkv_cu_options=True,
        all28_fa2_repeat_output_shadows=True,forwards_modes_and_rng_restored=True,
        caller_must_verify_all_parameter_bytes=True,execution_before=before,execution_after=after,
        started_unix=started,completed_unix=time.time())
