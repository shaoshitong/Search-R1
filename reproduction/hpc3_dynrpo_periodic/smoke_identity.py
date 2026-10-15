"""Exact parameter identity for the approved BF16 sampling-journal recipe.

Cross-backend BF16 log-probability differences remain visible diagnostics;
they are not the legacy FP32 backend-parity contract.
"""
import hashlib

import torch


def active(run):
    if not run or run.get('question_probe',{}).get('head_kind')!='qwen_native_periodic_lastblock_binary_v1':
        return False
    from reproduction.common8.precision_contract import sampling_journal_only,NATIVE4_BF16_PROFILE
    return run.get('precision_profile')==NATIVE4_BF16_PROFILE and sampling_journal_only(run)


def master_identity(module):
    digest=hashlib.sha256();count=0
    for name,value in module.named_parameters():
        value=value.detach().cpu().contiguous()
        if value.dtype!=torch.float32 or not torch.isfinite(value).all():
            raise ValueError('BF16 smoke requires finite actual FP32 master parameters')
        digest.update(name.encode());digest.update(str(tuple(value.shape)).encode())
        raw=memoryview(value.view(torch.uint8).numpy()).cast('B')
        for offset in range(0,len(raw),8*1024**2):digest.update(raw[offset:offset+8*1024**2])
        count+=value.numel()
    if count<=0:raise ValueError('No actual model parameters were observed')
    return dict(sha256=digest.hexdigest(),local_parameters=count)


def qwen_weight_readback(weights,model,config):
    """Check all actual vLLM Qwen tensors against the synced actor BF16 values.

    This reader does not invoke vLLM's weight_loader. It independently checks
    direct tensors, contiguous Q/K/V slices and gate/up slices with TP=1.
    DTensor full_tensor calls follow the same ordered keys on all four ranks.
    """
    params=dict(model.named_parameters(remove_duplicate=False))
    hidden=config.hidden_size
    kv=config.num_key_value_heads*(hidden//config.num_attention_heads)
    intermediate=config.intermediate_size
    errors=[];checked=set();elements=0
    for name,source in sorted(weights.items()):
        value=source.full_tensor() if hasattr(source,'full_tensor') else source
        target_name=name;part=None
        for old,start,length in (('q_proj',0,hidden),('k_proj',hidden,kv),('v_proj',hidden+kv,kv)):
            if '.self_attn.'+old+'.' in name:
                target_name=name.replace('.'+old+'.','.qkv_proj.');part=(start,length);break
        if '.mlp.gate_proj.' in name:
            target_name=name.replace('.gate_proj.','.gate_up_proj.');part=(0,intermediate)
        elif '.mlp.up_proj.' in name:
            target_name=name.replace('.up_proj.','.gate_up_proj.');part=(intermediate,intermediate)
        if name=='lm_head.weight' and target_name not in params and config.tie_word_embeddings:
            target_name='model.embed_tokens.weight'
        target=params.get(target_name)
        if target is None:
            errors.append('missing target '+target_name);continue
        checked.add(target_name)
        if part is not None:target=target.narrow(0,*part)
        expected=value.detach().to(device=target.device,dtype=torch.bfloat16)
        if target.dtype!=torch.bfloat16 or target.shape!=expected.shape or not torch.equal(target.detach(),expected):
            errors.append('wrong synchronized tensor '+name)
        elements+=expected.numel()
        del value,expected
    if config.tie_word_embeddings and 'lm_head.weight' in params and 'model.embed_tokens.weight' in checked:
        if torch.equal(params['lm_head.weight'],params['model.embed_tokens.weight']):checked.add('lm_head.weight')
        else:errors.append('tied output embedding differs')
    if set(params)!=checked:errors.append('unverified vLLM tensors '+str(sorted(set(params)-checked)))
    minimum=config.num_hidden_layers*12+2
    if len(weights)<minimum or elements<=0:errors.append('incomplete actor state dictionary')
    return dict(accepted=not errors,actor_tensor_count=len(weights),
        checked_vllm_tensor_count=len(checked),checked_elements=elements,errors=errors,
        source_cast='actual actor FP32 masters to BF16',comparison='bitwise exact',
        independent_reader=True,tensor_parallel_size=1)
