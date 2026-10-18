"""Tree-only sampled-token prefixes plus separately bound legacy semantics.

No token/probability is repaired, no optimizer is called and no probability
tolerance is defined here. Unsupported boundaries reject the current batch.
"""
import codecs
import copy
import ast
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import textwrap
import types


PROTOCOL = 'tree_sampled_prefix_legacy_semantics_v1'
FIELD = 'tree_raw_semantics'
EMISSION = '__tree_native_emission'
UTF8_BOUNDARY_PROTOCOL = 'tree_original_bytelevel_native_replacement_prefix_v2'
BYTELEVEL_DECODER_STATE = dict(type='ByteLevel',add_prefix_space=False,trim_offsets=False,use_regex=False)
_VALIDATION_TOKENIZERS = {}
_TOKENIZER_FILE_SHA_CACHE = {}
_ENCODER_SEMANTICS_CACHE = None
_ACTIVE_VALIDATION_CONTEXT = ContextVar('tree_raw_validation_context',default=None)
_HF_DECODE_CODE_CHECKS = {}


def require(value,message):
    if not value:raise ValueError('Tree raw-token protocol: '+message)


def _native_require(value,message,diagnostics):
    if not value:
        # Fixed messages and only numeric/boolean facts may reach the log.
        # Never expose sampled IDs, text, contexts or selected probabilities.
        print('TREE_NATIVE_EMISSION_DIAGNOSTIC='+json.dumps(diagnostics,sort_keys=True,
            allow_nan=False),file=sys.stderr,flush=True)
        require(False,message)


def validate_native_emission(emission,pad,eos):
    """Validate the native token timeline; genuine PAD is an emitted token."""
    n=emission.get('true_emitted_length');ids=emission.get('raw_emitted_token_ids')
    values=emission.get('raw_selected_logprobs')
    diagnostic=dict(emitted_length_is_integer=type(n) is int,
        metadata_id_count=len(ids) if type(ids) is list else -1,
        metadata_logprob_count=len(values) if type(values) is list else -1)
    if type(n) is int:diagnostic['reported_emitted_length']=n
    _native_require(type(pad) is int and type(eos) is int and pad!=eos,
        'Native PAD/EOS identity is unsupported',diagnostic)
    _native_require(type(n) is int and 0<n<=500,
        'Native true emitted length is outside the supported boundary',diagnostic)
    _native_require(type(ids) is list and len(ids)==n and
        all(type(token) is int and token>=0 for token in ids),
        'Native emitted token IDs do not cover the true length',diagnostic)
    _native_require(type(values) is list and len(values)==n and
        all(type(value) in (int,float) and math.isfinite(value) for value in values),
        'Native selected scalars do not cover the true length',diagnostic)
    stopping=emission.get('native_stopping')
    _native_require(type(stopping) is dict and
        stopping.get('protocol')=='vllm_native_eos_and_stop_set_v1' and
        stopping.get('source')=='detached_sampling_params_clone_native_update_from_generation_config_v0_6_3' and
        stopping.get('primary_eos_source')=='native_input_preprocessor_get_eos_token_id_no_lora' and
        isinstance(stopping.get('generation_config_sha256'),str) and
        re.fullmatch('[0-9a-f]{64}',stopping['generation_config_sha256']) is not None,
        'Native effective EOS/stop metadata is missing',diagnostic)
    fields=('generation_config_eos_token_ids','sampling_stop_token_ids',
        'sampling_all_stop_token_ids','effective_stop_token_ids')
    _native_require(stopping.get('primary_eos_token_id')==eos and
        all(type(stopping.get(key)) is list and
            all(type(token) is int and token>=0 for token in stopping[key]) and
            stopping[key]==sorted(set(stopping[key])) for key in fields),
        'Native effective EOS/stop metadata is invalid',diagnostic)
    expected_stops={eos}
    for key in fields[:-1]:expected_stops.update(stopping[key])
    stop_ids=set(stopping['effective_stop_token_ids'])
    _native_require(stop_ids==expected_stops,
        'Native effective EOS/stop set binding changed',diagnostic)
    positions=[index for index,token in enumerate(ids) if token in stop_ids]
    strings=stopping.get('sampling_stop_strings',[])
    include=stopping.get('include_stop_str_in_output',False)
    settings=emission.get('sampling_params',{})
    _native_require(strings in ([],['</search>'],['</search>','</answer>']) and
        strings==list(settings.get('stop') or []) and
        include==settings.get('include_stop_str_in_output',False) and
        (not strings or include is True),'Native string-stop binding differs',diagnostic)
    string_stop=(type(emission.get('stop_reason')) is str and emission['stop_reason'] in strings)
    if string_stop:
        returned=emission.get('native_returned_text')
        _native_require(type(returned) is str and returned.endswith(emission['stop_reason']),
            'Actual native returned text does not retain its stopping tag',diagnostic)
    diagnostic.update(true_emitted_pad_count=sum(token==pad for token in ids),
        emitted_native_stop_count=len(positions),last_emitted_token_is_native_stop=ids[-1] in stop_ids,
        early_emitted_native_stop_present=any(index!=n-1 for index in positions),
        finish_is_stop=emission.get('finish_reason')=='stop',
        finish_is_length=emission.get('finish_reason')=='length',
        stop_reason_matches_native_stop_or_none=(emission.get('stop_reason') is None or
            type(emission.get('stop_reason')) is int and emission['stop_reason'] in stop_ids or string_stop))
    _native_require(emission.get('finished') is True and
        emission.get('finish_reason') in ('stop','length'),
        'Native finish status is unsupported',diagnostic)
    _native_require(diagnostic['stop_reason_matches_native_stop_or_none'],
        'Native stop reason differs from the bound native EOS/stop set',diagnostic)
    _native_require(not diagnostic['early_emitted_native_stop_present'],
        'Native emission continued after a native EOS/stop token',diagnostic)
    _native_require((emission['finish_reason']=='stop')==(ids[-1] in stop_ids or string_stop),
        'Native finish reason disagrees with the emitted native EOS/stop boundary',diagnostic)
    if emission['finish_reason']=='stop':
        _native_require((emission['stop_reason'] is None and ids[-1]==eos) or
            (type(emission['stop_reason']) is int and emission['stop_reason']==ids[-1]) or string_stop,
            'Native finish stop reason disagrees with the last emitted token',diagnostic)
    _native_require(emission['finish_reason']!='length' or emission.get('stop_reason') is None,
        'Native length finish has an EOS stop reason',diagnostic)
    return ids,n,values


def native_response_mask(emission,raw_padded,pad,eos,response_mask=None):
    """Bind exactly the emitted prefix and its separate right-padding suffix."""
    ids,n,_=validate_native_emission(emission,pad,eos)
    diagnostic=dict(reported_emitted_length=n,tensor_width=len(raw_padded),
        metadata_id_count=len(ids),true_emitted_pad_count=sum(token==pad for token in ids))
    _native_require(len(raw_padded)==500 and n<=len(raw_padded),
        'Native returned tensor width disagrees with the supported boundary',diagnostic)
    _native_require(all(type(token) is int for token in raw_padded),
        'Native returned tensor token IDs are invalid',diagnostic)
    diagnostic['prefix_ids_equal']=raw_padded[:n]==ids
    _native_require(diagnostic['prefix_ids_equal'],
        'Native raw token IDs differ from the returned tensor prefix',diagnostic)
    diagnostic['padding_suffix_all_pad']=all(token==pad for token in raw_padded[n:])
    _native_require(diagnostic['padding_suffix_all_pad'],
        'Native returned tensor padding suffix contains non-PAD',diagnostic)
    mask=[1]*n+[0]*(len(raw_padded)-n)
    if response_mask is not None:
        diagnostic['response_mask_matches_true_length']=(len(response_mask)==len(mask)
            and all(type(mark) in (int,bool) and mark in (0,1) for mark in response_mask)
            and response_mask==mask)
        _native_require(diagnostic['response_mask_matches_true_length'],
            'Native response mask disagrees with the true emitted length',diagnostic)
    return mask


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _same_binding(left,right):
    """Snapshot equality with JSON's type and signed-zero distinctions."""
    if left is right:return True
    if type(left) is not type(right):return False
    if type(left) is dict:
        return left.keys()==right.keys() and all(_same_binding(value,right[key]) for key,value in left.items())
    if type(left) in (list,tuple):
        return len(left)==len(right) and all(_same_binding(a,b) for a,b in zip(left,right))
    if type(left) is float:return left.hex()==right.hex()
    return left==right


def _bytelevel_alphabet():
    # The reversible ByteLevel alphabet uses visible Latin-1 first, then maps
    # the remaining bytes to consecutive codepoints starting at U+0100.
    visible=list(range(33,127))+list(range(161,173))+list(range(174,256))
    codepoints=visible.copy();extra=0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte);codepoints.append(256+extra);extra+=1
    return {chr(codepoint):byte for byte,codepoint in zip(visible,codepoints)}


BYTELEVEL_INVERSE = _bytelevel_alphabet()
BYTELEVEL_ALPHABET_SHA256 = fingerprint(sorted(BYTELEVEL_INVERSE.items(),key=lambda pair:pair[1]))


def _decoder_binding(tokenizer):
    require(tokenizer.clean_up_tokenization_spaces is False,'UTF8 proof requires cleanup=false')
    backend=getattr(tokenizer,'backend_tokenizer',None)
    decoder=getattr(backend,'decoder',None)
    require(decoder is not None and callable(getattr(decoder,'__getstate__',None)),
            'Explicit native ByteLevel decoder state required')
    try:
        state=decoder.__getstate__()
        if isinstance(state,(str,bytes)):state=json.loads(state)
    except (TypeError,ValueError,UnicodeError) as error:
        raise ValueError('Tree raw-token protocol: Decoder state cannot be proven') from error
    require(type(state) is dict and state==BYTELEVEL_DECODER_STATE and
            all(type(state[key]) is bool for key in ('add_prefix_space','trim_offsets','use_regex')),
            'Unsupported native decoder state; exact frozen ByteLevel required')
    special=getattr(tokenizer,'all_special_ids',None)
    require(isinstance(special,(list,tuple)) and all(type(x) is int for x in special) and
            len(set(special))==len(special),'Explicit native special-token IDs required')
    require(callable(getattr(tokenizer,'convert_ids_to_tokens',None)),
            'Native original token pieces required for UTF8 proof')
    return dict(protocol=UTF8_BOUNDARY_PROTOCOL,utf8_errors='replace',decoder_state=copy.deepcopy(state),
        decoder_state_sha256=fingerprint(state),bytelevel_alphabet_sha256=BYTELEVEL_ALPHABET_SHA256,
        skip_special_token_ids=sorted(special))


def _invalid_utf8_events(data,byte_start,token_prefix_length,phase):
    """Identify replacement-producing original bytes, distinct from literal U+FFFD."""
    events=[]
    while data:
        try:
            data.decode('utf-8',errors='strict')
            break
        except UnicodeDecodeError as error:
            events.append(dict(token_prefix_length=token_prefix_length,phase=phase,
                original_byte_start=byte_start+error.start,original_byte_end=byte_start+error.end,
                original_bytes_hex=data[error.start:error.end].hex(),reason=error.reason))
            byte_start+=error.end;data=data[error.end:]
    return events


def _same_hf_code(left,right,skip_doc=False):
    fields=('co_code','co_names','co_varnames','co_argcount','co_posonlyargcount','co_kwonlyargcount','co_flags')
    if not all(getattr(left,key)==getattr(right,key) for key in fields):return False
    left_consts=left.co_consts[1:] if skip_doc else left.co_consts
    right_consts=right.co_consts[1:] if skip_doc else right.co_consts
    return len(left_consts)==len(right_consts) and all(
        _same_hf_code(a,b) if isinstance(a,types.CodeType) and isinstance(b,types.CodeType) else _same_binding(a,b)
        for a,b in zip(left_consts,right_consts))


def _approved_hf_decode_method(method,body_sha,defaults=(False,None)):
    """Bind live code to the inspected HF wrapper, not just its class/name."""
    if not _same_binding(method.__defaults__,defaults) or method.__kwdefaults__ is not None:return False
    code=method.__code__;key=(code,body_sha)
    if key not in _HF_DECODE_CODE_CHECKS:
        try:
            source=textwrap.dedent(inspect.getsource(method));node=ast.parse(source).body[0]
            body=node.body
            if isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
                body=body[1:]
            digest=hashlib.sha256(ast.dump(ast.Module(body=body,type_ignores=[]),
                include_attributes=False).encode()).hexdigest()
            expected=next(value for value in compile(source,'<tree-hf-decode-binding>','exec',
                dont_inherit=True).co_consts if isinstance(value,types.CodeType))
            # Dedenting changes only the wrapper docstring's whitespace.
            valid=digest==body_sha and _same_hf_code(code,expected,skip_doc=True)
        except (OSError,TypeError,ValueError,SyntaxError,StopIteration):valid=False
        _HF_DECODE_CODE_CHECKS[key]=valid
    return _HF_DECODE_CODE_CHECKS[key]


def _bound_fast_prefix_decoder(tokenizer):
    """Only the inspected, unmodified Qwen2 HF wrappers authorize native decode."""
    try:
        from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
        from transformers.utils.generic import to_py_obj
        from tokenizers import Tokenizer
    except ImportError:return None
    if (type(tokenizer) is not Qwen2TokenizerFast or type(tokenizer.backend_tokenizer) is not Tokenizer or
        tokenizer.backend_tokenizer is not tokenizer._tokenizer or tokenizer.clean_up_tokenization_spaces is not False or
        PreTrainedTokenizerBase.decode.__globals__.get('to_py_obj') is not to_py_obj or
        getattr(tokenizer.decode,'__func__',None) is not PreTrainedTokenizerBase.decode or
        getattr(tokenizer._decode,'__func__',None) is not PreTrainedTokenizerFast._decode):return None
    # Both HF 4.44.2 (local CPU) and 4.46.3 (own training environment) have
    # these bodies: to_py_obj(list[int]) -> _decode -> native decode, cleanup=false.
    if not (_approved_hf_decode_method(PreTrainedTokenizerBase.decode,
        '849e5b7280369da401eb82a89c67692a7b424296001bab4ed7eaf8d379144144') and
        _approved_hf_decode_method(PreTrainedTokenizerFast._decode,
        'd651226d2aba095d2180450e48566362685e0a0deee5094e13fc3b23e99ca08d') and
        _approved_hf_decode_method(to_py_obj,
        '11c54124f7603ea1e7541d5b06e24e8ef7c5493c6dc2a5d74b31215ebb8ee152',defaults=None)):return None
    return tokenizer.backend_tokenizer.decode


def _decode_method_binding(tokenizer):
    public=tokenizer.decode;private=getattr(tokenizer,'_decode',None)
    functions=(getattr(public,'__func__',public),getattr(private,'__func__',private))
    converter=getattr(functions[0],'__globals__',{}).get('to_py_obj')
    return (public,private,*[(getattr(method,'__code__',None),
        copy.deepcopy(getattr(method,'__defaults__',None)),copy.deepcopy(getattr(method,'__kwdefaults__',None)))
        for method in functions],converter,getattr(converter,'__code__',None),
        copy.deepcopy(getattr(converter,'__defaults__',None)),copy.deepcopy(getattr(converter,'__kwdefaults__',None)))


def _scan_utf8_prefixes(tokenizer,ids,identity,check_native_prefixes=True,_context=None):
    _context=_context or _validation_context(tokenizer,identity)
    require(type(_context) is _TokenizerValidationContext and _context.tokenizer is tokenizer and
            _context.identity==identity,'Actual tokenizer validation context differs')
    _context.check_runtime()
    binding=_context.identity['utf8_decoder_binding']
    require(identity.get('utf8_decoder_binding')==binding and
            identity.get('tokenizer_class')==type(tokenizer).__module__+'.'+type(tokenizer).__qualname__ and
            identity.get('clean_up_tokenization_spaces') is False and
            identity.get('policy_decode_skip_special_tokens') is True and
            identity.get('pad_token_id')==tokenizer.pad_token_id and
            identity.get('eos_token_id')==tokenizer.eos_token_id and
            identity.get('bos_token_id')==tokenizer.bos_token_id,
            'Frozen decoder/tokenizer/special-token identity differs')
    special=set(binding['skip_special_token_ids']);decoder=codecs.getincrementaldecoder('utf-8')('replace')
    raw=bytearray();stable='';decoded=[];complete=[];pending_events=[];prefix_hashes=[];invalid_events=[]
    # Hash the same JSON string as fingerprint(expected), extending only the
    # stable UTF8 portion. Pending bytes may become a different character later.
    stable_hash=hashlib.sha256(b'"')
    for stop,token in enumerate(ids,1):
        piece=b''
        if token not in special:
            spelling=tokenizer.convert_ids_to_tokens(token)
            require(isinstance(spelling,str) and spelling and all(x in BYTELEVEL_INVERSE for x in spelling),
                    'Original token piece is not a provable ByteLevel byte sequence')
            piece=bytes(BYTELEVEL_INVERSE[x] for x in spelling)
        pending_before=decoder.getstate()[0];raw.extend(piece)
        addition=decoder.decode(piece,final=False)
        current=stable+addition
        # Native ByteLevel uses replacement decoding. Only the unfinished suffix
        # may change with later tokens; replacements already emitted are stable.
        require(current.startswith(stable),'Nonmonotonic stable native replacement prefix unsupported')
        stable=current;pending=decoder.getstate()[0]
        stable_hash.update(json.dumps(addition,ensure_ascii=False)[1:-1].encode())
        resolved=pending_before+piece
        if pending:resolved=resolved[:-len(pending)]
        invalid_events.extend(_invalid_utf8_events(resolved,len(raw)-len(piece)-len(pending_before),
            stop,'stable_incremental'))
        pending_text=pending.decode('utf-8',errors='replace')
        expected=stable+pending_text
        if check_native_prefixes:
            if _context.native_prefix_decoder is not None:
                # Preserve HF _decode's only side effect; IDs are already real
                # Python integers, so its generic framework conversion is inert.
                tokenizer._decode_use_source_tokenizer=False
                native=_context.native_prefix_decoder(ids[:stop],skip_special_tokens=True)
            else:native=tokenizer.decode(ids[:stop],skip_special_tokens=True)
            require(isinstance(native,str) and native==expected,
                    'Native prefix decode differs from proven ByteLevel replacement/cleanup')
        prefix_hash=stable_hash.copy()
        prefix_hash.update(json.dumps(pending_text,ensure_ascii=False)[1:-1].encode()+b'"')
        expected_sha=prefix_hash.hexdigest();prefix_hashes.append(expected_sha)
        decoded.append(stable);complete.append(not pending)
        if pending:
            pending_events.append(dict(token_prefix_length=stop,pending_utf8_hex=pending.hex(),
                native_prefix_sha256=expected_sha,stable_prefix_sha256=fingerprint(stable)))
    require(complete,'Nonempty original-token emission required')
    final_pending=decoder.getstate()[0]
    invalid_events.extend(_invalid_utf8_events(final_pending,len(raw)-len(final_pending),
        len(ids),'native_finalization'))
    final=stable+decoder.decode(b'',final=True)
    require(final==bytes(raw).decode('utf-8',errors='replace'),
            'Final native replacement decode differs from original bytes')
    # A finished emission cannot acquire more bytes. Its native final replacement
    # is stable even when the original byte suffix was incomplete UTF8.
    decoded[-1]=final;complete[-1]=True
    proof=dict(protocol=UTF8_BOUNDARY_PROTOCOL,decoder_binding_sha256=fingerprint(binding),
        original_emitted_byte_sha256=hashlib.sha256(raw).hexdigest(),
        stable_original_token_boundaries=[i+1 for i,valid in enumerate(complete) if valid],
        pending_prefix_events=pending_events,invalid_original_utf8_events=invalid_events,
        native_prefix_agreement_count=len(ids),
        native_prefix_agreement_sha256=fingerprint(prefix_hashes),
        final_original_utf8_complete=not final_pending,final_original_utf8_valid=not invalid_events,
        final_pending_utf8_hex=final_pending.hex(),final_native_decode_finalized=True,
        final_native_text_sha256=fingerprint(decoded[-1]))
    return decoded,complete,proof


def _tokenizer_file_identity(tokenizer):
    path=(Path(tokenizer.name_or_path)/'tokenizer.json').resolve(strict=True)
    stat=path.stat()
    signature=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_mode)
    cached=_TOKENIZER_FILE_SHA_CACHE.get(str(path))
    if cached is None or cached[0]!=signature:
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
        after=path.stat()
        require(signature==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_mode),
                'Tokenizer file changed while its actual SHA was checked')
        cached=(signature,digest.hexdigest());_TOKENIZER_FILE_SHA_CACHE[str(path)]=cached
    return str(path),cached[1]


def _encoder_binding(tokenizer):
    global _ENCODER_SEMANTICS_CACHE
    backend=getattr(tokenizer,'backend_tokenizer',None)
    require(callable(getattr(backend,'to_str',None)),'Actual full native encoder state required')
    try:
        serialized=backend.to_str()
        cached=_ENCODER_SEMANTICS_CACHE
        cache_hit=type(serialized) is str and cached is not None and serialized==cached[0]
        if not cache_hit:state=json.loads(serialized)
    except (TypeError,ValueError,UnicodeError) as error:
        raise ValueError('Tree raw-token protocol: Actual full native encoder state cannot be proven') from error
    if cache_hit:
        semantics_sha=cached[1]
    else:
        require(type(state) is dict and all(key in state for key in
            ('model','normalizer','pre_tokenizer','post_processor','decoder','added_tokens')),
            'Complete actual native encoder semantic state required')
        # HF batch tokenization legitimately changes these per-call options. Neither
        # affects encode(canonical, add_special_tokens=False), which resets them.
        state.pop('padding',None);state.pop('truncation',None)
        semantics_sha=fingerprint(state)
        # One immutable tuple; local reads cannot mix concurrent state/hash updates.
        if type(serialized) is str:_ENCODER_SEMANTICS_CACHE=(serialized,semantics_sha)
    return dict(backend_semantics_sha256=semantics_sha,hf_semantic_runtime_flags=_encoder_runtime_flags(tokenizer))


def _encoder_runtime_flags(tokenizer):
    return {key:dict(present=hasattr(tokenizer,key),value=getattr(tokenizer,key,None)) for key in
        ('clean_up_tokenization_spaces','split_special_tokens','add_prefix_space','do_lower_case',
         'bos_token','eos_token','pad_token','unk_token','additional_special_tokens','all_special_tokens')}


def _current_tokenizer_identity(tokenizer):
    path,sha=_tokenizer_file_identity(tokenizer)
    return dict(tokenizer_json_sha256=sha,tokenizer_path=path,
        tokenizer_class=type(tokenizer).__module__+'.'+type(tokenizer).__qualname__,
        clean_up_tokenization_spaces=tokenizer.clean_up_tokenization_spaces,
        policy_decode_skip_special_tokens=True,reward_decode='original_tokenizer.decode_default_flags',
        pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,utf8_decoder_binding=_decoder_binding(tokenizer),
        encoder_binding=_encoder_binding(tokenizer))


class _TokenizerValidationContext:
    """Live binding and immutable action snapshots for one explicit batch/outer."""
    def __init__(self,tokenizer,identity):
        started=time.process_time()
        require(_current_tokenizer_identity(tokenizer)==identity,
                'Actual registered tokenizer file/path/encoder/decode identity drifted')
        self.tokenizer=tokenizer;self.identity=copy.deepcopy(identity)
        self.identity_key=fingerprint(identity)
        self.cpu_seconds=time.process_time()-started
        self.actions={}
        self.native_methods=(tokenizer.decode,tokenizer.encode,tokenizer.convert_ids_to_tokens)
        self.backend=tokenizer.backend_tokenizer
        self.decode_method_binding=_decode_method_binding(tokenizer)
        self.native_prefix_decoder=_bound_fast_prefix_decoder(tokenizer)

    def check_runtime(self):
        tokenizer=self.tokenizer;identity=self.identity
        path,sha=_tokenizer_file_identity(tokenizer)
        require(tokenizer.backend_tokenizer is self.backend and
            (tokenizer.decode,tokenizer.encode,tokenizer.convert_ids_to_tokens)==self.native_methods and
            _same_binding(_decode_method_binding(tokenizer),self.decode_method_binding) and
            path==identity['tokenizer_path'] and sha==identity['tokenizer_json_sha256'] and
            type(tokenizer).__module__+'.'+type(tokenizer).__qualname__==identity['tokenizer_class'] and
            tokenizer.pad_token_id==identity['pad_token_id'] and tokenizer.eos_token_id==identity['eos_token_id'] and
            tokenizer.bos_token_id==identity['bos_token_id'] and _decoder_binding(tokenizer)==identity['utf8_decoder_binding'] and
            _encoder_runtime_flags(tokenizer)==identity['encoder_binding']['hf_semantic_runtime_flags'],
            'Actual registered tokenizer runtime identity drifted')

    def check_complete(self):
        self.check_runtime()
        require(_current_tokenizer_identity(self.tokenizer)==self.identity,
                'Actual registered tokenizer file/path/encoder/decode identity drifted')

    def action_is_validated(self,action):
        self.check_runtime()
        require(action['tokenizer_identity']==self.identity,'Actual tokenizer validation context differs')
        bound={k:v for k,v in action.items() if not k.startswith('prefix_scan_') and k!='action_binding_sha256'}
        key=action.get('action_binding_sha256');cached=self.actions.get(key)
        if cached is not None and _same_binding(cached,bound):return True
        require(fingerprint(bound)==key,'Inherited action ledger changed')
        return False

    def remember_action(self,action):
        bound={k:v for k,v in action.items() if not k.startswith('prefix_scan_') and k!='action_binding_sha256'}
        self.actions[action['action_binding_sha256']]=copy.deepcopy(bound)


def _validation_context(tokenizer,identity):
    context=_ACTIVE_VALIDATION_CONTEXT.get()
    if context is not None and context.tokenizer is tokenizer:
        require(context.identity==identity,'Actual tokenizer validation context differs')
        return context
    return _TokenizerValidationContext(tokenizer,identity)


@contextmanager
def tokenizer_validation_scope(tokenizer,identity,_context=None):
    """Share proofs only in an explicit scope, with full live checks at its boundary.

    Runtime flags/file/decoder are checked at use; the complete encoder is checked
    again before a successful scope can return. A mutated live backend therefore
    rejects this batch even when an inherited action used its existing proof.
    """
    active=_ACTIVE_VALIDATION_CONTEXT.get()
    if active is not None and active.tokenizer is tokenizer:
        require(active.identity==identity,'Actual tokenizer validation context differs')
        yield active
        return
    context=_context or _TokenizerValidationContext(tokenizer,identity)
    require(type(context) is _TokenizerValidationContext and context.tokenizer is tokenizer and
            context.identity==identity,'Actual tokenizer validation context differs')
    context.check_runtime()
    token=_ACTIVE_VALIDATION_CONTEXT.set(context)
    try:
        yield context
        context.check_complete()
    finally:
        _ACTIVE_VALIDATION_CONTEXT.reset(token)


def tokenizer_identity(tokenizer,expected_sha):
    require(isinstance(expected_sha,str) and re.fullmatch('[0-9a-f]{64}',expected_sha),
            'Explicit frozen tokenizer.json SHA required')
    context=_ACTIVE_VALIDATION_CONTEXT.get()
    if context is not None and context.tokenizer is tokenizer:
        context.check_runtime();identity=copy.deepcopy(context.identity)
    else:identity=_current_tokenizer_identity(tokenizer)
    require(identity['tokenizer_json_sha256']==expected_sha,'Tokenizer file identity differs')
    # Only a live object whose real file SHA and complete decode identity were
    # checked may authorize later inherited/actor validation in this process.
    # Colocated actor/ref instances may register the same frozen identity.
    # Preserve the first trusted object instead of replacing its authority.
    key=context.identity_key if context is not None and context.tokenizer is tokenizer else fingerprint(identity)
    _VALIDATION_TOKENIZERS.setdefault(key,tokenizer)
    return identity


def register_validation_tokenizer(tokenizer):
    """Register the worker's already loaded object, never a ledger-supplied path."""
    _,sha=_tokenizer_file_identity(tokenizer)
    return tokenizer_identity(tokenizer,sha)


def _validation_tokenizer(identity):
    tokenizer=_VALIDATION_TOKENIZERS.get(fingerprint(identity))
    require(tokenizer is not None,'Actual tokenizer is not registered in this process; no ledger-path loading')
    return tokenizer


def canonical_text(text):
    # Preserve the original SEARCH-first rule, not earliest-tag priority.
    if '</search>' in text:return text.split('</search>')[0]+'</search>'
    if '</answer>' in text:return text.split('</answer>')[0]+'</answer>'
    return text


def capture_native_sampling_settings(sampling_params):
    """Read original native attributes; usable before engine processor construction."""
    fields=('n','best_of','_real_n','temperature','top_p','top_k','min_p','presence_penalty','frequency_penalty',
        'repetition_penalty','skip_special_tokens','spaces_between_special_tokens','min_tokens','max_tokens','ignore_eos','stop','stop_token_ids',
        'include_stop_str_in_output','logprobs','prompt_logprobs','seed','logits_processors',
        'truncate_prompt_tokens','guided_decoding','logit_bias','allowed_token_ids')
    for key in fields:
        require(hasattr(sampling_params,key), 'Explicit actual '+key+' metadata is required')
    settings={key:copy.deepcopy(getattr(sampling_params,key)) for key in fields}
    for key in ('best_of','_real_n'):
        require(type(settings[key]) is int and settings[key]==1,
                'Native raw-token protocol requires actual '+key+'=1')
    for key in ('truncate_prompt_tokens','guided_decoding','logit_bias','allowed_token_ids'):
        require(settings[key] is None, 'Native raw-token protocol requires actual '+key+'=None')
    require(settings['logits_processors'] is None, 'Native raw-token protocol requires actual logits_processors=None')
    require(settings['n']==1 and settings['max_tokens']==500 and settings['ignore_eos'] is False and
            settings['min_tokens']==0 and (not settings['stop'] or
            settings['stop'] in (['</search>'],['</search>','</answer>']) and settings['include_stop_str_in_output'] is True),
            'Unsupported native stop/distribution protocol')
    if settings['stop']:
        require(getattr(sampling_params,'detokenize',False) is True,
            'Native string stopping requires actual native detokenization')
    return settings


def capture_native_stopping(sampling_params,primary_eos_token_id,generation_config):
    """Expand only a detached copy with the native engine's own update method."""
    require(type(primary_eos_token_id) is int and primary_eos_token_id>=0 and
        type(generation_config) is dict,'Actual native tokenizer/engine stopping inputs required')
    configured=generation_config.get('eos_token_id')
    generation_eos=[] if configured is None else [configured] if type(configured) is int else configured
    native_params=sampling_params.clone()
    require(native_params is not sampling_params,'Native stopping clone is not detached')
    native_params.update_from_generation_config(copy.deepcopy(generation_config),primary_eos_token_id)
    explicit=native_params.stop_token_ids
    native_all=getattr(native_params,'all_stop_token_ids',None)
    require(type(generation_eos) in (list,tuple) and
        type(explicit) in (list,tuple) and type(native_all) in (set,frozenset) and
        all(type(token) is int and token>=0 for tokens in (generation_eos,explicit,native_all) for token in tokens),
        'Actual native EOS/stop collections are invalid')
    effective={primary_eos_token_id,*explicit}
    require(set(native_all)=={primary_eos_token_id,*generation_eos,*explicit},
        'Native engine EOS/stop expansion differs from its recorded inputs')
    return dict(protocol='vllm_native_eos_and_stop_set_v1',primary_eos_token_id=primary_eos_token_id,
        source='detached_sampling_params_clone_native_update_from_generation_config_v0_6_3',
        primary_eos_source='native_input_preprocessor_get_eos_token_id_no_lora',
        generation_config_sha256=fingerprint(generation_config),
        generation_config_eos_token_ids=sorted(set(generation_eos)),
        sampling_stop_token_ids=sorted(set(explicit)),sampling_all_stop_token_ids=sorted(native_all),
        effective_stop_token_ids=sorted(effective),sampling_stop_strings=list(native_params.stop or []),
        include_stop_str_in_output=native_params.include_stop_str_in_output)


def capture_native_emissions(request_outputs,sampling_params,call_id,rank,*,native_stopping=None):
    """Called before facade padding, retaining real request IDs and scalar values."""
    settings=capture_native_sampling_settings(sampling_params)
    require(type(native_stopping) is dict,'Actual native effective EOS/stop capture required')
    provenance={key:os.environ.get(key) for key in ('SEARCH_R1_BASELINE_COMMIT','SEARCH_R1_BASELINE_CONFIG_SHA256',
        'SEARCH_R1_BASELINE_DATA_IDENTITY','SEARCH_R1_MODEL_IDENTITY','SEARCH_R1_BASELINE_RUN_ID',('RBA_H100_SHARED_EXECUTION_ID' if os.environ.get('RBA_H100_SHARED_EXECUTION_ID') else 'SLURM_JOB_ID'))}
    require(all(isinstance(value,str) and value for value in provenance.values()),
            'Own source/config/data/model/run/job provenance is required for opt-in emission capture')
    rows=[];seen=set()
    for row,request in enumerate(request_outputs):
        require(request.request_id not in seen and request.finished and len(request.outputs)==1,
                'Unique final n1 native requests required')
        seen.add(request.request_id);out=request.outputs[0]
        ids=[int(token) for token in out.token_ids]
        require(0<len(ids)<=500 and out.logprobs is not None and len(out.logprobs)==len(ids),
                'Empty/missing/over-cap native emission or scalar metadata')
        logps=[float(values[token].logprob) for values,token in zip(out.logprobs,ids)]
        require(all(math.isfinite(value) for value in logps),'Nonfinite sampled scalar')
        require(out.finish_reason in ('stop','length'), 'Unsupported native abort/finish status')
        identity=dict(call_id=call_id,rank=int(rank),request_id=request.request_id,physical_row=row,
                      runtime_provenance=provenance.copy())
        record=dict(schema_version=1,protocol=PROTOCOL,sampling_identity=identity,
            sample_id=fingerprint(identity),prompt_token_ids=[int(x) for x in request.prompt_token_ids],
            raw_emitted_token_ids=ids,raw_selected_logprobs=logps,true_emitted_length=len(ids),
            finish_reason=out.finish_reason,stop_reason=out.stop_reason,finished=True,
            sampling_params=copy.deepcopy(settings),native_stopping=copy.deepcopy(native_stopping))
        if type(out.stop_reason) is str:
            require(settings['stop'] and out.stop_reason in settings['stop'] and
                settings['include_stop_str_in_output'] is True and
                isinstance(out.text,str) and out.text.endswith(out.stop_reason),
                'Native returned output omitted or mismatched its stopping tag')
            record['native_returned_text']=out.text
        rows.append(record)
    return rows


def validate_decoded_string_stop(emission,decoded,tokenizer=None):
    """Bind actual native returned text to original IDs, never fabricate mu."""
    reason=emission.get('stop_reason')
    if type(reason) is not str:return
    strings=emission['native_stopping']['sampling_stop_strings']
    returned=emission.get('native_returned_text');full=decoded[-1]
    require(type(returned) is str and returned.endswith(reason),
        'Native returned stopping text omitted stopping tag')
    if not full.startswith(returned):
        # vLLM decodes against the prompt's token suffix, unlike an isolated
        # response decode. A leading byte fragment can consequently differ.
        # Replay its actual incremental decoder; retain original policy IDs,
        # probabilities and the legacy isolated-decode semantics unchanged.
        require(tokenizer is not None,'Native incremental stop replay requires tokenizer')
        from vllm.transformers_utils.detokenizer import convert_prompt_ids_to_tokens,detokenize_incrementally
        settings=emission['sampling_params']
        skip=settings.get('skip_special_tokens',True)
        spaces=settings.get('spaces_between_special_tokens',True)
        prompt=emission['prompt_token_ids'];ids=emission['raw_emitted_token_ids']
        previous,prefix,read=convert_prompt_ids_to_tokens(tokenizer,prompt,skip_special_tokens=skip)
        all_ids=prompt.copy();output='';stopped=False
        for index,token in enumerate(ids):
            all_ids.append(token)
            new,fragment,prefix,read=detokenize_incrementally(tokenizer,all_ids,previous,prefix,read,
                skip_special_tokens=skip,spaces_between_special_tokens=spaces)
            previous.extend(new);output+=fragment
            if fragment:
                for marker in strings:
                    position=output.find(marker,-len(fragment)-len(marker))
                    if position>=0:
                        require(index==len(ids)-1 and marker==reason,
                            'Native incremental stop occurred before final emitted token')
                        output=output[:position+len(marker)];stopped=True;break
            if stopped:break
        require(stopped and output==returned,
            'Native incremental returned text differs from replay of original prompt/emitted IDs')
    require(all(not any(marker in prefix for marker in strings) for prefix in decoded[:-1]),
        'Native emission continued past a completed string stop token boundary')
    # Native string stopping may cut within the last whole sampled token; that
    # token and its actual selected probability must remain indivisible.
    require(reason in full,'Actual emitted tokens do not cover native string stop')


LINEAR_PREFIX_PROTOCOL = 'bound_bytelevel_linear_with_native_full_cut_v1'


def _linear_boundary_proof(tokenizer,ids,cut,decoded,proof,context):
    # The bound decoder validates actual full/cut strings; intermediate prefixes
    # are derived from the immutable ByteLevel byte vocabulary, not native calls.
    require(context.native_prefix_decoder is not None,
            'Linear prefix scan requires the approved unchanged HF native decoder')
    context.check_runtime()
    positions=sorted(set((len(ids),cut)))
    checks=[]
    for position in positions:
        tokenizer._decode_use_source_tokenizer=False
        actual=context.native_prefix_decoder(ids[:position],skip_special_tokens=True)
        require(actual==decoded[position-1],
                'Linear ByteLevel proof differs from actual native full/cut decode')
        checks.append(dict(token_prefix_length=position,text_sha256=fingerprint(actual)))
    result=copy.deepcopy(proof)
    result['linear_prefix_sha256']=result.pop('native_prefix_agreement_sha256')
    result.pop('native_prefix_agreement_count')
    result.update(native_check_protocol=LINEAR_PREFIX_PROTOCOL,
        native_checked_prefixes=checks,native_decode_call_count=len(positions),
        intermediate_prefix_source='bound_incremental_bytelevel_utf8_replace',
        intermediate_prefix_count=len(ids))
    return result


def sampled_action(tokenizer,emission,raw_padded,legacy_ids,legacy_text,identity,response_mask=None,_context=None):
    """Scan stable native prefixes; never encode a policy token or split one."""
    started=time.perf_counter(),time.process_time()
    require(emission.get('protocol')==PROTOCOL and emission.get('schema_version')==1,
            'Native emission metadata missing')
    pad,eos=tokenizer.pad_token_id,tokenizer.eos_token_id
    require(type(pad) is int and type(eos) is int and pad!=eos,'PAD==EOS unsupported')
    ids,n,values=validate_native_emission(emission,pad,eos)
    native_response_mask(emission,raw_padded,pad,eos,response_mask)
    require(emission['sample_id']==fingerprint(emission['sampling_identity']), 'Sampling identity mutated')
    utf8_started=time.process_time()
    context=_context or _validation_context(tokenizer,identity)
    linear=os.environ.get('TREE_LINEAR_PREFIX_SCAN')=='1'
    decoded,complete,utf8_proof=_scan_utf8_prefixes(tokenizer,ids,identity,
        check_native_prefixes=not linear,_context=context)
    utf8_cpu_seconds=time.process_time()-utf8_started
    full=decoded[-1];text=canonical_text(full)
    validate_decoded_string_stop(emission,decoded,tokenizer)
    semantic_ids=tokenizer.encode(text,add_special_tokens=False)
    require(isinstance(semantic_ids,list) and all(type(token) is int for token in semantic_ids) and
            isinstance(legacy_ids,list) and all(type(token) is int for token in legacy_ids) and
            text==legacy_text and semantic_ids==legacy_ids,
            'Legacy parser text/canonical semantic IDs differ')
    reward_text=tokenizer.decode(semantic_ids)
    tagged='</search>' in full or '</answer>' in full
    if tagged:
        choices=[i+1 for i,prefix in enumerate(decoded) if complete[i] and prefix.startswith(text)]
        require(choices,'Canonical boundary not covered by a stable original-token prefix')
        cut=choices[0]
    else:cut=n
    if linear:
        utf8_proof=_linear_boundary_proof(tokenizer,ids,cut,decoded,utf8_proof,context)
    action=ids[:cut];projection=decoded[cut-1]
    require(projection.startswith(text),'Whole-token boundary does not cover canonical semantics')
    extra=projection[len(text):]
    ledger=dict(protocol=PROTOCOL,tokenizer_identity=copy.deepcopy(identity),utf8_boundary_proof=utf8_proof,
        native_emission=copy.deepcopy(emission),cut=cut,action_token_ids=action,
        selected_logprobs=values[:cut],canonical_text=text,canonical_text_sha256=fingerprint(text),
        canonical_semantic_token_ids=semantic_ids.copy(),legacy_reward_action_text=reward_text,
        whole_prefix_decode=projection,whole_prefix_decode_sha256=fingerprint(projection),
        indivisible_extra_text=extra,indivisible_extra_utf8_hex=extra.encode('utf8').hex(),
        boundary_reason='shortest_whole_original_prefix_covering_legacy_tag' if tagged else 'all_true_emitted_tokens_no_tag',
        raw_emitted_tokens=n,retained_sampled_tokens=cut,discarded_sampled_tail_tokens=n-cut,
        emitted_eos_positions=[i for i,x in enumerate(ids) if x==eos],
        eos_retained=any(x==eos for x in action),empty_canonical=text=='',
        prefix_scan_wall_seconds=time.perf_counter()-started[0],
        prefix_scan_cpu_seconds=time.process_time()-started[1],
        prefix_scan_utf8_cpu_seconds=utf8_cpu_seconds,
        prefix_scan_tokenizer_validation_cpu_seconds=context.cpu_seconds,
        probability_accepted=False)
    ledger['action_binding_sha256']=fingerprint({k:v for k,v in ledger.items() if not k.startswith('prefix_scan_')})
    context.remember_action(ledger)
    return action,ledger


def validate_action_utf8_proof(action,tokenizer,_context=None):
    """Reconstruct native replacement proof and actual full/cut decode, without resampling."""
    _context=_context or _validation_context(tokenizer,action['tokenizer_identity'])
    require(type(_context) is _TokenizerValidationContext and _context.tokenizer is tokenizer,
            'Actual tokenizer validation context differs')
    if _context.action_is_validated(action):return
    emission=action['native_emission'];cut=action['cut']
    ids,_,_=validate_native_emission(emission,tokenizer.pad_token_id,tokenizer.eos_token_id)
    decoded,complete,proof=_scan_utf8_prefixes(tokenizer,ids,action['tokenizer_identity'],False,_context)
    validate_decoded_string_stop(emission,decoded,tokenizer)
    stored=action.get('utf8_boundary_proof')
    validated_linear=isinstance(stored,dict) and stored.get('native_check_protocol')==LINEAR_PREFIX_PROTOCOL
    if validated_linear:
        require(type(cut) is int and 0<cut<=len(ids),'Invalid linear proof cut')
        proof=_linear_boundary_proof(tokenizer,ids,cut,decoded,proof,_context)
    require(proof==stored,'Bound original-token native replacement proof differs')
    require(type(cut) is int and 0<cut<=len(ids) and complete[cut-1],
            'Retained boundary has an unfinished native replacement prefix')
    # Linear proof already checked these exact native full/cut strings above.
    full=decoded[-1] if validated_linear else tokenizer.decode(ids,skip_special_tokens=True)
    projection=decoded[cut-1] if validated_linear else tokenizer.decode(ids[:cut],skip_special_tokens=True)
    text=canonical_text(full)
    semantic_ids=tokenizer.encode(text,add_special_tokens=False)
    require(isinstance(semantic_ids,list) and all(type(token) is int for token in semantic_ids) and
            isinstance(action['canonical_semantic_token_ids'],list) and
            all(type(token) is int for token in action['canonical_semantic_token_ids']) and
            action['canonical_semantic_token_ids']==semantic_ids and
            action['legacy_reward_action_text']==tokenizer.decode(semantic_ids),
            'Legacy IDs/reward text differ from actual original canonical tokenizer encode/decode')
    tagged='</search>' in full or '</answer>' in full
    choices=[i+1 for i,prefix in enumerate(decoded) if complete[i] and prefix.startswith(text)]
    require(not tagged or choices,'Canonical tag has no proven stable original-token boundary')
    require(full==decoded[-1] and projection==decoded[cut-1] and action['canonical_text']==text and
            action['canonical_text_sha256']==fingerprint(text) and
            action['whole_prefix_decode']==projection and cut==(choices[0] if tagged else len(ids)) and
            action['whole_prefix_decode_sha256']==fingerprint(projection) and
            action['action_token_ids']==ids[:cut] and action['selected_logprobs']==emission['raw_selected_logprobs'][:cut] and
            action['indivisible_extra_text']==projection[len(text):] and
            action['indivisible_extra_utf8_hex']==projection[len(text):].encode('utf-8').hex(),
            'Native/canonical stable original-token boundary differs')
    _context.remember_action(action)


def _trajectory_values(state,cap,_contexts=None):
    contexts_by_identity={} if _contexts is None else _contexts
    raw=[];mask=[];semantic=[];semantic_mask=[];contexts=[]
    for turn,ledger in enumerate(state.raw_ledger):
        require(ledger['turn']==turn and ledger['generation_context_token_ids']==state.prompt+raw,
                'Inherited generation context differs from complete scoring prefix')
        require(ledger['generation_context_positions']==list(range(len(state.prompt+raw))) and
                ledger['generation_context_attention']==[1]*len(state.prompt+raw), 'Generation position/mask history differs')
        action=ledger['action'];obs=ledger['observation_token_ids'];n=action['cut']
        require(action['action_token_ids']==action['native_emission']['raw_emitted_token_ids'][:n] and
                action['selected_logprobs']==action['native_emission']['raw_selected_logprobs'][:n],
                'Inherited sampled token/scalar prefix changed')
        identity=action['tokenizer_identity'];active=_ACTIVE_VALIDATION_CONTEXT.get()
        key=active.identity_key if active is not None and active.identity==identity else fingerprint(identity)
        if key not in contexts_by_identity:
            tokenizer=_VALIDATION_TOKENIZERS.get(key)
            require(tokenizer is not None,'Actual tokenizer is not registered in this process; no ledger-path loading')
            contexts_by_identity[key]=(active if active is not None and active.identity==identity else
                _validation_context(tokenizer,identity))
        context=contexts_by_identity[key]
        validate_action_utf8_proof(action,context.tokenizer,context)
        raw+=action['action_token_ids']+obs;mask+=[1]*n+[0]*len(obs)
        legacy=action['canonical_semantic_token_ids'];semantic+=legacy+obs
        semantic_mask+=[1]*len(legacy)+[0]*len(obs);contexts.append(ledger['generation_context_token_ids'])
    require(raw==state.response and mask==state.model_mask and len(raw)<=cap and len(semantic)<=cap,
            'Policy/semantic score cap or retained timeline mismatch')
    return dict(protocol=PROTOCOL,prompt_token_ids=state.prompt.copy(),policy_response_token_ids=raw,
        policy_model_mask=mask,semantic_response_token_ids=semantic,semantic_model_mask=semantic_mask,
        raw_ledger=state.raw_ledger,response_cap=cap)


def trajectory_projection(state,cap,_contexts=None):
    value=_trajectory_values(state,cap,_contexts)
    value['raw_ledger']=copy.deepcopy(value['raw_ledger'])
    value['binding_sha256']=fingerprint(value)
    return value


def semantic_spans(item,tokenizer,_contexts=None):
    """Rebuild legacy contiguous spans; canonical IDs never enter policy tensors."""
    if FIELD not in item.non_tensor_batch:
        require(getattr(item,'meta_info',{}).get('tree_raw_token_protocol') is None,
                'Opted-in Tree semantic projection was dropped')
        return None
    value=item.non_tensor_batch[FIELD]
    bound={k:v for k,v in value.items() if k!='binding_sha256'}
    require(fingerprint(bound)==value['binding_sha256'] and value['protocol']==PROTOCOL,
            'Semantic projection binding changed')
    require(value['response_cap']==4096,'Frozen score cap changed')
    width=item.batch['responses'].shape[-1]
    valid=item.batch['attention_mask'][-width:].bool();ids=item.batch['responses'][valid].tolist()
    mask=item.batch['info_mask'][-width:][valid].tolist()
    prompt_tensor=item.batch['prompts'];prompt_width=prompt_tensor.shape[-1]
    prompt_attention=item.batch['attention_mask'][:prompt_width]
    require(len(prompt_attention)==prompt_width and
            all(mark in (0,1) for mark in prompt_attention.tolist()),
            'Semantic prompt attention boundary is invalid')
    prompt=prompt_tensor[prompt_attention.bool()].tolist()
    require(ids==value['policy_response_token_ids'] and mask==value['policy_model_mask'] and
            prompt==value['prompt_token_ids'],'Semantic projection attached to different policy inputs/mask')
    state=type('ProjectionState',(),{})()
    state.prompt=prompt;state.response=ids;state.model_mask=mask;state.raw_ledger=value['raw_ledger']
    contexts={} if _contexts is None else _contexts
    require(_trajectory_values(state,value['response_cap'],contexts)==bound, 'Semantic ledger does not reconstruct bound timeline')
    for ledger in value['raw_ledger']:
        action=ledger['action'];identity=action['tokenizer_identity']
        require(identity['pad_token_id']==tokenizer.pad_token_id and identity['eos_token_id']==tokenizer.eos_token_id and
                identity['clean_up_tokenization_spaces']==tokenizer.clean_up_tokenization_spaces and
                tokenizer.decode(action['canonical_semantic_token_ids'])==action['legacy_reward_action_text'],
                'Frozen tokenizer flags or legacy reward projection changed')
        active=_ACTIVE_VALIDATION_CONTEXT.get()
        key=active.identity_key if active is not None and active.identity==identity else fingerprint(identity)
        context=contexts[key]
        if context.tokenizer is not tokenizer:
            validate_action_utf8_proof(action,tokenizer,_validation_context(tokenizer,identity))
    spans=[];start=None;tokens=value['semantic_response_token_ids'];marks=value['semantic_model_mask']
    for index,is_model in enumerate(marks):
        if is_model and start is None:start=index
        elif not is_model and start is not None:
            spans.append(tokenizer.decode(tokens[start:index]));start=None
    if start is not None:spans.append(tokenizer.decode(tokens[start:]))
    return spans
