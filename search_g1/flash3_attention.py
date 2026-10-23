"""Opt-in, instance-local FA3 kernels with an optional native-KV4 fast path.

No Transformers class/module, sys.path, reference model, or vLLM backend is
patched. Each actor receives private clones of its existing Python forward and
HF helper functions. The pinned FA3 interface returns (out, lse); only out is
returned to HF, with its original autograd connection intact.
The separately opted-in compact path replaces only the private forward's
repeat_kv binding; RoPE, packing, masking, output projection and state stay put.
"""
import ctypes
from contextlib import contextmanager
import functools
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
import types
import uuid

import torch

INTERFACE_SHA = 'a50f564fa1fc4cdd85ffa61f79e913a463c2066e519f9c5e8bc2e5c74457fa38'
BINDING_FIELDS = frozenset({'interface_path','interface_sha256','extension_path','extension_sha256',
                            'cudart_path','cudart_sha256'})


def _need(value, message):
    if not value: raise ValueError('EASY7B_FLASH3_' + message)


def _digest(path):
    path=Path(path); before=path.stat(); h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    after=path.stat()
    _need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
          (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'SOURCE_CHANGED_DURING_SHA')
    return h.hexdigest()


def _binding(run):
    value=run.get('infra_flash_attention_3')
    _need(isinstance(value,dict) and set(value)==BINDING_FIELDS,'EXPLICIT_SIX_FIELD_SOURCE_BINDING_REQUIRED')
    result={}
    for kind in ('interface','extension','cudart'):
        path=Path(value[kind+'_path'])
        _need(path.is_absolute() and '..' not in path.parts,'ABSOLUTE_SOURCE_REQUIRED')
        path=path.resolve(); expected=value[kind+'_sha256']
        _need(isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected) is not None and
              path.is_file() and _digest(path)==expected,'PINNED_'+kind.upper()+'_SHA_DIFFERS')
        result[kind+'_path']=str(path);result[kind+'_sha256']=expected
    _need(result['interface_sha256']==INTERFACE_SHA,'UNQUALIFIED_INTERFACE_VERSION')
    _need(Path(result['extension_path']).name.startswith('flash_attn_3_cuda.') and
          Path(result['extension_path']).suffix=='.so','EXPECTED_FLASH3_EXTENSION_REQUIRED')
    return result


def load_backend(run):
    """Load explicit files only; never add the foreign site-packages to sys.path."""
    binding=_binding(run)
    preloaded={str(Path(p).resolve()) for p in os.environ.get('LD_PRELOAD','').split(':') if p}
    _need(binding['cudart_path'] in preloaded,'PINNED_CUDART_MUST_BE_PRELOADED_BY_PRIVATE_PROCESS')
    extension_name='flash_attn_3_cuda'; extension=sys.modules.get(extension_name)
    if extension is None:
        spec=importlib.util.spec_from_file_location(extension_name,binding['extension_path'])
        _need(spec is not None and spec.loader is not None,'EXTENSION_IMPORT_SPEC_REQUIRED')
        extension=importlib.util.module_from_spec(spec);sys.modules[extension_name]=extension
        try:spec.loader.exec_module(extension)
        except BaseException:
            sys.modules.pop(extension_name,None);raise
    _need(str(Path(extension.__file__).resolve())==binding['extension_path'],'FOREIGN_FLASH3_EXTENSION_ALREADY_LOADED')
    name='_search_g1_actor_flash3_'+uuid.uuid4().hex
    spec=importlib.util.spec_from_file_location(name,binding['interface_path'])
    interface=importlib.util.module_from_spec(spec);sys.modules[name]=interface
    try:spec.loader.exec_module(interface)
    except BaseException:
        sys.modules.pop(name,None);raise
    _need(interface.flash_attn_3_cuda is extension and callable(interface.flash_attn_varlen_func) and
          callable(interface.flash_attn_func) and callable(interface._flash_attn_backward),
          'PINNED_FORWARD_AND_BACKWARD_INTERFACE_REQUIRED')
    runtime=ctypes.c_int()
    function=ctypes.CDLL(binding['cudart_path']).cudaRuntimeGetVersion
    function.argtypes=[ctypes.POINTER(ctypes.c_int)];function.restype=ctypes.c_int
    _need(function(ctypes.byref(runtime))==0,'CUDART_VERSION_READ_FAILED')
    provenance=dict(binding,interface_module=name,interface_version='3.0.0b1',
        extension_module=extension_name,cudart_runtime_version=runtime.value,
        torch_version=torch.__version__,compiled_torch_cuda=torch.version.cuda,
        source_scope='private actor interface + explicit extension; no foreign site-packages path')
    return types.SimpleNamespace(module=interface,provenance=provenance)


class _Audit:
    def __init__(self,backend):
        self.provenance=dict(backend.provenance)
        self.forward_returns={'dense':0,'varlen':0};self.qkv_backward_returns=0;self.kernel_backward_returns=0
        self.shapes=[];self.attention_layers=0
        original=backend.module._flash_attn_backward
        @functools.wraps(original)
        def backward(*args,**kwargs):
            result=original(*args,**kwargs)
            self.kernel_backward_returns+=1
            return result
        # This module object was privately imported by load_backend, not the
        # globally installed flash_attn_interface or any reference/vLLM module.
        backend.module._flash_attn_backward=backward

    def called(self,kind,q,k,v):
        self.forward_returns[kind]+=1
        shape=dict(kind=kind,q=list(q.shape),k=list(k.shape),v=list(v.shape),dtype=str(q.dtype))
        if shape not in self.shapes and len(self.shapes)<6:self.shapes.append(shape)
        active=[x for x in (q,k,v) if x.requires_grad]
        if not torch.is_grad_enabled() or not active:return
        done=set();expected=len(active)
        def hook(index):
            def received(gradient):
                done.add(index)
                if len(done)==expected:self.qkv_backward_returns+=1
                return gradient
            return received
        for index,value in enumerate(active):value.register_hook(hook(index))

    def report(self,synchronize=False):
        synced=False
        if synchronize:
            _need(torch.cuda.is_initialized(),'INITIALIZED_ACTOR_CUDA_REQUIRED_FOR_SYNCHRONIZED_AUDIT')
            torch.cuda.synchronize();synced=True
        return dict(role='easy7b_instance_local_flash3_execution',provenance=self.provenance,
            attention_layers=self.attention_layers,forward_success_returns=dict(self.forward_returns),
            qkv_gradient_returns=self.qkv_backward_returns,kernel_backward_success_returns=self.kernel_backward_returns,
            cuda_synchronized=synced,actual_flash3_training_executed=bool(synced and
                sum(self.forward_returns.values()) and self.kernel_backward_returns),observed_shapes=list(self.shapes),
            state_dict_keys_changed=False,global_transformers_patch=False,checked_unix=time.time())


class _Kernel:
    def __init__(self,original,backend,audit,kind,require_cuda=True):
        self.signature=inspect.signature(original);self.backend=backend;self.audit=audit
        self.kind=kind;self.require_cuda=require_cuda

    def __call__(self,*args,**kwargs):
        bound=self.signature.bind(*args,**kwargs);bound.apply_defaults();values=dict(bound.arguments)
        allowed={'q','k','v','dropout_p','softmax_scale','causal','window_size','softcap',
                 'alibi_slopes','deterministic','return_attn_probs','block_table'}
        if self.kind=='varlen':allowed|={'cu_seqlens_q','cu_seqlens_k','max_seqlen_q','max_seqlen_k'}
        _need(set(values)<=allowed,'UNSUPPORTED_FA2_ARGUMENTS_'+','.join(sorted(set(values)-allowed)))
        _need(float(values.get('dropout_p',0))==0,'NONZERO_DROPOUT_UNSUPPORTED')
        _need(values.get('alibi_slopes') is None and values.get('block_table') is None and
              values.get('return_attn_probs',False) is False,'ALIBI_BLOCK_TABLE_OR_ATTN_PROBS_UNSUPPORTED')
        q,k,v=(values[name] for name in ('q','k','v'))
        _need(all(isinstance(x,torch.Tensor) for x in (q,k,v)) and q.dtype==k.dtype==v.dtype==torch.bfloat16,
              'BF16_QKV_REQUIRED_NO_FP8')
        _need(q.device==k.device==v.device and (not self.require_cuda or q.is_cuda),'SAME_CUDA_QKV_REQUIRED')
        _need(q.ndim==k.ndim==v.ndim==(3 if self.kind=='varlen' else 4) and
              q.shape[-1]==k.shape[-1]==v.shape[-1] and k.shape[-2]==v.shape[-2] and
              q.shape[-2]%k.shape[-2]==0,'QKV_GQA_SHAPE_DIFFERS')
        window=values.get('window_size',(-1,-1))
        _need(isinstance(window,(tuple,list)) and len(window)==2 and all(type(x) is int and x>=-1 for x in window),
              'WINDOW_SEMANTICS_UNSUPPORTED')
        options=dict(softmax_scale=values.get('softmax_scale'),causal=values.get('causal',False),
            window_size=tuple(window),softcap=values.get('softcap',0.0),deterministic=values.get('deterministic',False))
        _need(type(options['causal']) is bool and type(options['deterministic']) is bool,'BOOLEAN_KERNEL_POLICY_REQUIRED')
        if self.kind=='varlen':
            for key,total in (('cu_seqlens_q',q.shape[0]),('cu_seqlens_k',k.shape[0])):
                cu=values[key]
                _need(isinstance(cu,torch.Tensor) and cu.dtype==torch.int32 and cu.ndim==1 and
                      cu.device==q.device and cu.numel()>=2,'PACKED_CU_SEQLENS_REQUIRED')
            # No re-packing, tensor concatenation, head repeats, RNG draws, or
            # scale/mask reconstruction: original HF objects pass through.
            result=self.backend.module.flash_attn_varlen_func(q,k,v,values['cu_seqlens_q'],values['cu_seqlens_k'],
                None,None,values['max_seqlen_q'],values['max_seqlen_k'],**options)
        else:result=self.backend.module.flash_attn_func(q,k,v,**options)
        _need(isinstance(result,tuple) and len(result)==2 and isinstance(result[0],torch.Tensor),
              'PINNED_FA3_OUT_LSE_PAIR_REQUIRED')
        output=result[0]
        _need(output.shape==q.shape and output.dtype==q.dtype and output.device==q.device,'KERNEL_OUTPUT_SHAPE_OR_DTYPE_DIFFERS')
        self.audit.called(self.kind,q,k,v)
        return output


def _clone(function,replacements):
    _need(isinstance(function,types.FunctionType),'ORIGINAL_PYTHON_FUNCTION_REQUIRED')
    _need(set(replacements)<=set(function.__code__.co_names),'ORIGINAL_FUNCTION_BINDINGS_REQUIRED')
    namespace=dict(function.__globals__);namespace.update(replacements)
    cloned=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    cloned.__kwdefaults__=function.__kwdefaults__
    functools.update_wrapper(cloned,function)
    return cloned


def _compact_requested(explicit):
    if explicit is not None:
        _need(type(explicit) is bool,'BOOLEAN_COMPACT_GQA_REQUIRED')
        return explicit
    setting=os.environ.get('SEARCH_G1_FA3_COMPACT_GQA','0')
    _need(setting in ('0','1'),'COMPACT_GQA_ENV_MUST_BE_ZERO_OR_ONE')
    return setting=='1'


def _compact_repeat(layer,function):
    """Keep native KV heads only inside this actor's private forward namespace.

    FA3 itself performs grouped-query attention and sums the seven query-head
    gradient contributions into each KV head. All operations before/after the
    original repeat_kv calls, including RoPE and HF packing, remain unchanged.
    """
    _need('repeat_kv' in function.__code__.co_names,'COMPACT_GQA_REPEAT_BINDING_REQUIRED')
    groups=getattr(layer,'num_key_value_groups',None)
    _need(groups==7 and getattr(layer,'num_key_value_heads',None)==4 and
          getattr(layer,'num_heads',None)==28,'COMPACT_GQA_NATIVE28_QUERY4_KV_REQUIRED')
    sequence_world=function.__globals__.get('get_ulysses_sequence_parallel_world_size')
    if sequence_world is None:
        # Native SP1 uses the standard HF forward, whose namespace has no
        # Ulysses functions. The worker already requires TP1/SP1 before this
        # installation. Keep every byte of that HF forward and its RoPE path.
        _need(function.__module__=='transformers.models.qwen2.modeling_qwen2',
              'COMPACT_GQA_PINNED_HF_OR_VERL_FORWARD_REQUIRED')
        sequence_world=lambda:1
    _need(callable(sequence_world) and sequence_world()==1,'COMPACT_GQA_SEQUENCE_PARALLEL_ONE_REQUIRED')
    def keep_native_kv(hidden_states,n_rep):
        _need(sequence_world()==1,'COMPACT_GQA_SEQUENCE_PARALLEL_CHANGED')
        _need(isinstance(hidden_states,torch.Tensor) and hidden_states.ndim==4 and
              hidden_states.shape[1]==4 and hidden_states.shape[-1]==layer.head_dim and
              type(n_rep) is int and n_rep==groups,'COMPACT_GQA_REPEAT_INPUT_DIFFERS')
        return hidden_states
    return keep_native_kv


def install_flash3_actor(model,run=None,*,backend=None,require_cuda=True,compact_gqa=None):
    """Called only for the 7B native4 actor, before FSDP wraps that instance."""
    if run is None:run=json.loads((Path(os.environ['SEARCH_G1_BASE'])/'logs/run-manifest.json').read_text())
    from search_g1.easy7b_infra import flags
    from reproduction.common8.precision_contract import native4_7b
    _need(flags(run).get('flash_attention_3_actor') is True and native4_7b(run),'EXPLICIT_NATIVE4_7B_ACTOR_FLAG_REQUIRED')
    config=model.config
    _need(config.model_type=='qwen2' and config.hidden_size==3584 and config.num_hidden_layers==28 and
          config._attn_implementation=='flash_attention_2','PINNED_QWEN7B_HF_FA2_SHELL_REQUIRED')
    _need(not getattr(model,'_search_g1_flash3_installed',False),'ACTOR_ALREADY_INSTALLED')
    compact_gqa=_compact_requested(compact_gqa)
    layers=[m for m in model.modules() if type(m).__name__=='Qwen2FlashAttention2']
    _need(len(layers)==28,'ALL28_ACTOR_ATTENTION_INSTANCES_REQUIRED')
    before=tuple(model.state_dict().keys())
    backend=load_backend(run) if backend is None else backend
    audit=_Audit(backend);plans=[];helpers={}
    for layer in layers:
        function=layer.forward.__func__
        _need('_flash_attention_forward' in function.__globals__,'HF_ATTENTION_HELPER_BINDING_REQUIRED')
        helper=function.__globals__['_flash_attention_forward']
        if helper not in helpers:
            _need(isinstance(helper,types.FunctionType),'HF_PYTHON_ATTENTION_HELPER_REQUIRED')
            namespace=helper.__globals__
            kernels={name:_Kernel(namespace[name],backend,audit,kind,require_cuda) for name,kind in
                (('flash_attn_func','dense'),('flash_attn_varlen_func','varlen'))}
            helpers[helper]=_clone(helper,kernels)
        replacements={'_flash_attention_forward':helpers[helper]}
        if compact_gqa:replacements['repeat_kv']=_compact_repeat(layer,function)
        plans.append((layer,layer.forward,types.MethodType(_clone(function,replacements),layer)))
    for layer,original,private in plans:layer.forward=private
    audit.attention_layers=len(layers)
    model._search_g1_flash3_audit=audit;model._search_g1_flash3_installed=True
    # Plain Python objects only: neither a new Module/Parameter nor checkpoint
    # state. The original bound forwards remain available for isolated A/B
    # qualification without a second model allocation.
    model._search_g1_flash3_forwards=tuple(plans);model._search_g1_flash3_enabled=True
    model._search_g1_flash3_compact_gqa=compact_gqa
    _need(tuple(model.state_dict().keys())==before,'STATE_DICT_KEYS_CHANGED')
    return execution_report(model)


def set_flash3_enabled(model,enabled):
    """Switch this actor only; keep one mode through forward AND backward.

    In particular, gradient checkpointing must recompute with the same mode as
    its initial forward. This API is for the isolated qualification process;
    production leaves the installed actor enabled for its entire lifetime.
    Returns the previous mode and never reads/writes RNG or tensor state.
    """
    _need(type(enabled) is bool,'BOOLEAN_ENABLE_REQUIRED')
    _need(getattr(model,'_search_g1_flash3_installed',False),'INSTALLED_ACTOR_REQUIRED')
    previous=model._search_g1_flash3_enabled
    plans=model._search_g1_flash3_forwards
    for layer,original,private in plans:
        expected=private if previous else original
        _need(layer.forward==expected,'INSTANCE_FORWARD_CHANGED_OUTSIDE_SWITCH')
    for layer,original,private in plans:layer.forward=private if enabled else original
    model._search_g1_flash3_enabled=enabled
    return previous


@contextmanager
def flash3_mode(model,*,enabled):
    """Temporarily select original FA2 or private FA3, restoring on failure."""
    previous=set_flash3_enabled(model,enabled)
    try:yield model
    finally:set_flash3_enabled(model,previous)


def execution_report(model,synchronize=False):
    audit=getattr(model,'_search_g1_flash3_audit',None)
    _need(isinstance(audit,_Audit),'INSTALLED_ACTOR_AUDIT_REQUIRED')
    report=audit.report(synchronize=synchronize)
    report['flash3_enabled']=model._search_g1_flash3_enabled
    report['compact_gqa_enabled']=model._search_g1_flash3_compact_gqa
    return report


def write_execution_audit(model,*,rank,logical_update,optimizer_writes):
    report=execution_report(model,synchronize=True)
    base=Path(os.environ['SEARCH_G1_BASE'])
    run=json.loads((base/'logs/run-manifest.json').read_text())
    report.update(run_id=run['run_id'],source_commit=run['source_commit'],rank=rank,logical_update=logical_update,
        actual_optimizer_writes=optimizer_writes)
    path=base/'logs'/('actual-flash3-execution-rank-%d.jsonl'%rank)
    with path.open('a') as stream:
        stream.write(json.dumps(report,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    print('SEARCH_G1_ACTUAL_FLASH3_EXECUTION='+json.dumps(report,sort_keys=True),flush=True)
    return report
