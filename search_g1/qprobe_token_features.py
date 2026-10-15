"""Immutable packed, all-question-token features for a fresh binary probe.

No pooled-vector cache can satisfy this representation. Large arrays and token
metadata must live on remote storage; batch padding is created only in memory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import shutil

import numpy as np
import torch

from search_g1.qprobe_features import (
    COMMON_MANIFEST_SHA, OFFICIAL_ENCODER, OFFICIAL_REVISION,
    QUESTION_EXTRACTION_RULE, question_text_sha, sha_file, sha_json,
)

FILES=('packed_features.npy','offsets.npy','lengths.npy','index.jsonl')


def remote_path(path):
    path=Path(path)
    if not path.is_absolute() or path.resolve()!=path or not str(path).startswith(('/data/','/mnt/','/backup/')):
        raise ValueError('TOKEN_FEATURES_REQUIRE_REAL_REMOTE_STORAGE')
    return path


def validate_spec(spec):
    spec=dict(spec)
    if (spec.get('encoder_id')!=OFFICIAL_ENCODER or spec.get('encoder_revision')!=OFFICIAL_REVISION or
            spec.get('train_manifest_sha256')!=COMMON_MANIFEST_SHA or spec.get('split')!='train' or
            spec.get('question_extraction')!=QUESTION_EXTRACTION_RULE):
        raise ValueError('TOKEN_FEATURE_OFFICIAL_MODEL_OR_CANONICAL_DATA_CHANGED')
    if (spec.get('representation')!='all_nonpadding_question_tokens_packed_v1' or spec.get('feature_dim')!=2048 or
            spec.get('feature_layer')!=27 or spec.get('encoder_compute_dtype')!='bfloat16' or
            spec.get('feature_dtype') not in ('bfloat16','float32') or spec.get('truncation') is not False):
        raise ValueError('TOKEN_FEATURE_REPRESENTATION_CHANGED')
    if type(spec.get('global_max_tokens')) is not int or spec['global_max_tokens']<=0 or spec.get('global_train_rows')!=169028:
        raise ValueError('TOKEN_FEATURE_GLOBAL_BMAX_NOT_CANONICAL')
    for key in ('encoder_files_sha256','tokenizer_sha256','prompt_template_sha256','train_parquet_sha256','teacher_identity_sha256','teacher_complete_sha256','global_scan_sha256','token_feature_producer_sha256','original_label_feature_spec_sha256'):
        value=spec.get(key)
        if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('TOKEN_FEATURE_SPEC_SHA_MISSING: '+key)
    return spec


def scan_token_lengths(tokenizer, rows, *, batch_size=256, progress=None):
    """CPU tokenizer scan of every canonical row; never truncate or pool."""
    if len(rows)!=169028 or batch_size<=0:raise ValueError('TOKEN_SCAN_REQUIRES_ALL_CANONICAL_TRAIN')
    lengths=np.empty(len(rows),dtype=np.int32);token_hashes=[];ordered=hashlib.sha256()
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size]
        encoded=tokenizer([question for question,_ in batch],padding=False,truncation=False,add_special_tokens=True)['input_ids']
        if len(encoded)!=len(batch):raise ValueError('TOKEN_SCAN_ROW_COUNT_CHANGED')
        for local,((question,qid),tokens) in enumerate(zip(batch,encoded)):
            size=len(tokens)
            if size<=0:raise ValueError('TOKEN_SCAN_EMPTY_QUESTION')
            lengths[start+local]=size
            token_sha=hashlib.sha256(np.asarray(tokens,dtype='<i8').tobytes()).hexdigest();token_hashes.append(token_sha)
            ordered.update(json.dumps([qid,question_text_sha(question),size,token_sha],ensure_ascii=False,separators=(',',':')).encode()+b'\n')
        if progress:progress(dict(stage='token_global_length_scan',completed=min(start+len(batch),len(rows)),total=len(rows)))
    proof=dict(global_train_rows=len(rows),global_max_tokens=int(lengths.max()),min_tokens=int(lengths.min()),
        total_tokens=int(lengths.sum()),ordered_bindings_sha256=ordered.hexdigest(),truncation=False,tokenizer_add_special_tokens=True)
    return lengths,token_hashes,proof


def extract_question_token_features(model,tokenizer,questions,*,feature_layer=27,max_tokens,device='cuda:0'):
    """Same public decoder-layer hook as qprobe_features; retain every real token."""
    for question in questions:question_text_sha(question)
    layers=getattr(model,'layers',None)
    if layers is None or not 0<=feature_layer<len(layers):raise ValueError('TOKEN_FEATURE_ENCODER_LAYER_MISSING')
    encoded=tokenizer(list(questions),padding=True,truncation=False,add_special_tokens=True,return_tensors='pt')
    mask=encoded['attention_mask'].bool()
    if not len(questions) or bool((mask.sum(-1)<=0).any()) or int(mask.sum(-1).max())>max_tokens:
        raise ValueError('TOKEN_FEATURE_LENGTH_EXCEEDS_VERIFIED_BMAX')
    encoded={key:value.to(device) for key,value in encoded.items()};captured=[]
    def hook(_module,_inputs,output):
        hidden=output[0] if isinstance(output,tuple) else output
        if hidden.shape[:2]!=encoded['attention_mask'].shape or hidden.shape[-1]!=2048:
            raise ValueError('TOKEN_FEATURE_ENCODER_SHAPE_CHANGED')
        actual=encoded['attention_mask'].bool()
        rows=[hidden[row,actual[row]].detach().cpu().contiguous() for row in range(hidden.shape[0])]
        if any(not bool(torch.isfinite(value).all()) for value in rows):raise ValueError('TOKEN_FEATURE_NONFINITE')
        captured.append(rows)
    handle=layers[feature_layer].register_forward_hook(hook);training=model.training
    try:
        model.eval()
        with torch.inference_mode():model(**encoded,use_cache=False,output_hidden_states=False,return_dict=True)
    finally:handle.remove();model.train(training)
    if len(captured)!=1:raise ValueError('TOKEN_FEATURE_HOOK_NOT_EXACTLY_ONCE')
    return captured[0]


class TokenFeatureCacheWriter:
    def __init__(self,path,spec,records):
        self.path=remote_path(path);self.spec=validate_spec(spec);self.records=list(records);self.written=0
        if self.path.exists():raise FileExistsError('TOKEN_FEATURE_CACHE_ALREADY_EXISTS')
        if not self.records or len({row['qid'] for row in self.records})!=len(self.records):raise ValueError('TOKEN_FEATURE_DUPLICATE_IDS')
        lengths=np.asarray([row['length'] for row in records],dtype=np.int32)
        if int(lengths.min())<=0 or int(lengths.max())>self.spec['global_max_tokens']:raise ValueError('TOKEN_FEATURE_INVALID_LENGTH')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        item_bytes=2 if self.spec['feature_dtype']=='bfloat16' else 4
        required=int(lengths.sum(dtype=np.int64))*2048*item_bytes+128*1024**2
        if shutil.disk_usage(self.path.parent).free<required:raise OSError('TOKEN_FEATURE_REMOTE_STORAGE_INSUFFICIENT')
        self.partial=Path(tempfile.mkdtemp(prefix=self.path.name+'.partial-',dir=self.path.parent))
        self.offsets=np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(lengths,dtype=np.int64)))
        np.save(self.partial/'offsets.npy',self.offsets,allow_pickle=False);np.save(self.partial/'lengths.npy',lengths,allow_pickle=False)
        dtype=np.uint16 if self.spec['feature_dtype']=='bfloat16' else np.float32
        self.packed=np.lib.format.open_memmap(self.partial/'packed_features.npy',mode='w+',dtype=dtype,shape=(int(self.offsets[-1]),2048))
        self.index=(self.partial/'index.jsonl').open('x',encoding='utf-8')

    def append(self,ids,features):
        if len(ids)!=len(features) or self.written+len(ids)>len(self.records):raise ValueError('TOKEN_FEATURE_APPEND_COUNT')
        for qid,value in zip(ids,features):
            row=self.records[self.written]
            if qid!=row['qid'] or tuple(value.shape)!=(row['length'],2048) or not bool(torch.isfinite(value).all()):
                raise ValueError('TOKEN_FEATURE_APPEND_BINDING')
            if self.spec['feature_dtype']=='bfloat16':array=value.detach().cpu().to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)
            else:array=value.detach().cpu().float().contiguous().numpy()
            start,end=map(int,self.offsets[self.written:self.written+2]);self.packed[start:end]=array
            record=dict(row,row=self.written,offset=start,feature_sha256=hashlib.sha256(array.tobytes()).hexdigest())
            self.index.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n');self.written+=1

    def finish(self):
        if self.written!=len(self.records):raise ValueError('TOKEN_FEATURE_CACHE_INCOMPLETE')
        self.packed.flush();self.index.flush();os.fsync(self.index.fileno());self.index.close();del self.packed
        for name in FILES:
            with (self.partial/name).open('rb') as stream:os.fsync(stream.fileno())
        manifest=dict(schema_version=1,representation='all_nonpadding_question_tokens_packed_v1',spec=self.spec,spec_sha=sha_json(self.spec),
            count=self.written,feature_dim=2048,total_tokens=int(self.offsets[-1]),
            files={name:dict(bytes=(self.partial/name).stat().st_size,sha256=sha_file(self.partial/name)) for name in FILES})
        with (self.partial/'manifest.json').open('x') as stream:
            json.dump(manifest,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        for path in self.partial.iterdir():path.chmod(0o444)
        descriptor=os.open(self.partial,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        if self.path.exists():raise FileExistsError('TOKEN_FEATURE_CACHE_ALREADY_EXISTS')
        os.rename(self.partial,self.path)
        fd=os.open(self.path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return manifest


class TokenFeatureCache:
    def __init__(self,path,*,expected_spec_sha=None):
        self.path=remote_path(path);self.manifest=json.loads((self.path/'manifest.json').read_text());self.spec=validate_spec(self.manifest['spec'])
        self.spec_sha=sha_json(self.spec)
        if (self.manifest.get('schema_version')!=1 or self.manifest.get('representation')!='all_nonpadding_question_tokens_packed_v1' or
                self.manifest['spec_sha']!=self.spec_sha or expected_spec_sha is not None and expected_spec_sha!=self.spec_sha):
            raise ValueError('TOKEN_FEATURE_MANIFEST_IDENTITY_CHANGED')
        if set(self.manifest['files'])!=set(FILES):raise ValueError('TOKEN_FEATURE_PACKED_FILES_MISSING')
        for name in FILES:
            path=self.path/name
            if path.resolve()!=path or path.is_symlink() or path.stat().st_size!=self.manifest['files'][name]['bytes']:
                raise ValueError('TOKEN_FEATURE_FILE_BINDING_CHANGED')
        for name in ('offsets.npy','lengths.npy','index.jsonl'):
            if sha_file(self.path/name)!=self.manifest['files'][name]['sha256']:raise ValueError('TOKEN_FEATURE_METADATA_SHA_CHANGED')
        self.offsets=np.load(self.path/'offsets.npy',allow_pickle=False);self.lengths=np.load(self.path/'lengths.npy',allow_pickle=False)
        self.packed=np.load(self.path/'packed_features.npy',allow_pickle=False,mmap_mode='r');self.feature_dim=2048
        expected_dtype=np.dtype('uint16') if self.spec['feature_dtype']=='bfloat16' else np.dtype('float32')
        if (self.packed.dtype!=expected_dtype or self.packed.shape!=(self.manifest['total_tokens'],2048) or
                self.offsets.dtype!=np.int64 or self.lengths.dtype!=np.int32 or self.offsets.shape!=(self.manifest['count']+1,) or
                self.lengths.shape!=(self.manifest['count'],) or self.offsets[0]!=0 or self.offsets[-1]!=self.packed.shape[0] or
                not np.array_equal(np.diff(self.offsets),self.lengths) or int(self.lengths.min())<=0 or int(self.lengths.max())>self.spec['global_max_tokens']):
            raise ValueError('TOKEN_FEATURE_PACKED_SHAPE_CHANGED')
        self.ids=[];self.index={};self.verified=set()
        for ordinal,line in enumerate((self.path/'index.jsonl').read_text().splitlines()):
            row=json.loads(line)
            if row['row']!=ordinal or row['qid'] in self.index or row['offset']!=int(self.offsets[ordinal]) or row['length']!=int(self.lengths[ordinal]):
                raise ValueError('TOKEN_FEATURE_INDEX_ORDER_CHANGED')
            self.ids.append(row['qid']);self.index[row['qid']]=row
        if len(self.ids)!=self.manifest['count']:raise ValueError('TOKEN_FEATURE_INDEX_COUNT_CHANGED')

    def get_batch(self,qids,device='cpu'):
        if not qids:raise ValueError('TOKEN_FEATURE_EMPTY_BATCH')
        rows=[self.index[qid] for qid in qids];lengths=[row['length'] for row in rows]
        dtype=torch.bfloat16 if self.spec['feature_dtype']=='bfloat16' else torch.float32
        result=torch.zeros((len(rows),max(lengths),2048),dtype=dtype);mask=torch.zeros((len(rows),max(lengths)),dtype=torch.bool)
        for ordinal,row in enumerate(rows):
            start,end=map(int,self.offsets[row['row']:row['row']+2]);array=np.array(self.packed[start:end],copy=True)
            value=torch.from_numpy(array.view(np.int16) if dtype==torch.bfloat16 else array)
            if dtype==torch.bfloat16:value=value.view(torch.bfloat16)
            if row['qid'] not in self.verified:
                if hashlib.sha256(array.tobytes()).hexdigest()!=row['feature_sha256'] or not bool(torch.isfinite(value).all()):raise ValueError('TOKEN_FEATURE_ROW_SHA_OR_FINITE_CHANGED')
                self.verified.add(row['qid'])
            result[ordinal,:row['length']]=value;mask[ordinal,:row['length']]=True
        return dict(features=result.to(device),attention_mask=mask.to(device))

    def verify_all(self):
        for name in FILES:
            if sha_file(self.path/name)!=self.manifest['files'][name]['sha256']:raise ValueError('TOKEN_FEATURE_FULL_SHA_CHANGED')
        return dict(count=len(self.ids),total_tokens=self.manifest['total_tokens'],spec_sha=self.spec_sha)
