"""Ragged frozen token features; no pooled-vector substitute or gold fields."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from search_g1.qprobe_features import validate_spec, sha_json, sha_file, question_text_sha


class SequenceFeatures:
    def __init__(self, hidden, positions):
        self.hidden = list(hidden); self.positions = list(positions)
        if len(self.hidden) != len(self.positions): raise ValueError('QPROBE_SEQUENCE_COUNT_MISMATCH')
        self.feature_dim = self.hidden[0].shape[-1] if self.hidden else None
        for h, p in zip(self.hidden, self.positions):
            if h.ndim != 2 or h.shape[0] < 1 or h.shape[-1] != self.feature_dim or p.shape != h.shape[:1]:
                raise ValueError('QPROBE_RAGGED_FEATURE_SHAPE')
            if p.dtype != torch.long or bool((p < 0).any()) or not torch.isfinite(h).all():
                raise ValueError('QPROBE_RAGGED_FEATURE_VALUES')
    def __len__(self):return len(self.hidden)
    def __getitem__(self, indices):
        if isinstance(indices, torch.Tensor):indices=indices.tolist()
        if isinstance(indices, slice):indices=range(len(self))[indices]
        if isinstance(indices, int):indices=[indices]
        return SequenceFeatures([self.hidden[i] for i in indices], [self.positions[i] for i in indices])
    def detach(self):return SequenceFeatures([h.detach() for h in self.hidden], [p.detach() for p in self.positions])
    def cpu(self):return SequenceFeatures([h.cpu() for h in self.hidden], [p.cpu() for p in self.positions])
    def clone(self):return SequenceFeatures([h.clone() for h in self.hidden], [p.clone() for p in self.positions])
    def state_dict(self):return {'representation':'frozen_token_sequence_v2','hidden':self.hidden,'positions':self.positions}
    @classmethod
    def from_state(cls, state):
        if state.get('representation') != 'frozen_token_sequence_v2':raise ValueError('QPROBE_REQUIRES_SEQUENCE_STATE')
        return cls(state['hidden'],state['positions'])
    @classmethod
    def cat(cls, batches):return cls([h for b in batches for h in b.hidden],[p for b in batches for p in b.positions])
    def padded(self, device, dtype):
        if not len(self):raise ValueError('QPROBE_EMPTY_SEQUENCE_BATCH')
        width=max(len(h) for h in self.hidden);n=len(self)
        hidden=torch.zeros(n,width,self.feature_dim,device=device,dtype=dtype)
        mask=torch.zeros(n,width,device=device,dtype=torch.bool)
        positions=torch.zeros(n,width,device=device,dtype=torch.long)
        for i,(h,p) in enumerate(zip(self.hidden,self.positions)):
            hidden[i,:len(h)]=h.to(device=device,dtype=dtype);mask[i,:len(h)]=True;positions[i,:len(h)]=p.to(device)
        return hidden,mask,positions


def validate_sequence_spec(spec):
    value=dict(spec)
    if value.get('representation') != 'frozen_token_sequence_v2' or value.get('pooling') != 'all_nonpadding_prompt_tokens':
        raise ValueError('QPROBE_SEQUENCE_SPEC_REQUIRED')
    # Reuse the strict official encoder/data/feature-layer provenance contract.
    temporary=dict(value,pooling='last_nonpadding_prompt_token');validate_spec(temporary)
    return value


class SequenceCacheWriter:
    def __init__(self, path, spec, count):
        self.path=Path(path);self.spec=validate_sequence_spec(spec);self.count=count;self.written=0;self.tokens=0;self.ids=set()
        if type(count) is not int or count<1:raise ValueError('QPROBE_INVALID_SEQUENCE_COUNT')
        if self.path.exists():raise FileExistsError(str(self.path))
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.partial=Path(tempfile.mkdtemp(prefix=self.path.name+'.partial-',dir=self.path.parent))
        self.h=(self.partial/'hidden.bin').open('xb');self.p=(self.partial/'positions.bin').open('xb');self.index=(self.partial/'index.jsonl').open('x')
    def append(self,qids,questions,features):
        if not isinstance(features,SequenceFeatures) or len(qids)!=len(questions) or len(qids)!=len(features):raise ValueError('QPROBE_SEQUENCE_APPEND_SHAPE')
        if self.written+len(qids)>self.count:raise ValueError('QPROBE_SEQUENCE_CACHE_OVERFLOW')
        for qid,question,h,pos in zip(qids,questions,features.hidden,features.positions):
            if not isinstance(qid,str) or not qid or qid in self.ids:raise ValueError('QPROBE_DUPLICATE_SEQUENCE_QID')
            if h.shape[-1]!=self.spec['feature_dim'] or len(h)>self.spec['max_prompt_tokens']:raise ValueError('QPROBE_SEQUENCE_ENCODER_SHAPE')
            a=h.detach().cpu().float().numpy().astype('<f2');b=pos.detach().cpu().numpy().astype('<i8')
            if not np.isfinite(a).all():raise ValueError('QPROBE_SEQUENCE_FP16_OVERFLOW')
            raw=a.tobytes();positions=b.tobytes()
            row=dict(qid=qid,question_hash=question_text_sha(question),row=self.written,offset=self.tokens,length=len(h),hidden_sha256=hashlib.sha256(raw).hexdigest(),position_sha256=hashlib.sha256(positions).hexdigest())
            self.h.write(raw);self.p.write(positions);self.index.write(json.dumps(row)+'\n');self.ids.add(qid);self.written+=1;self.tokens+=len(h)
    def abort(self):
        # Keep incomplete remote staging for diagnosis; never publish it.
        for f in (self.h,self.p,self.index):
            if not f.closed:f.close()
    def finish(self):
        if self.written!=self.count:raise ValueError('QPROBE_INCOMPLETE_SEQUENCE_CACHE')
        for f in (self.h,self.p,self.index):f.flush();os.fsync(f.fileno());f.close()
        files={p.name:{'bytes':p.stat().st_size,'sha256':sha_file(p)} for p in self.partial.iterdir()}
        manifest=dict(schema_version=2,spec=self.spec,spec_sha=sha_json(self.spec),count=self.count,total_tokens=self.tokens,files=files)
        with (self.partial/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.flush();os.fsync(f.fileno())
        os.rename(self.partial,self.path)
        return manifest


class SequenceCache:
    def __init__(self,path):
        self.path=Path(path);self.manifest=json.loads((self.path/'manifest.json').read_text())
        if self.manifest.get('schema_version')!=2:raise ValueError('QPROBE_SEQUENCE_CACHE_SCHEMA')
        self.spec=validate_sequence_spec(self.manifest['spec']);self.spec_sha=sha_json(self.spec);self.feature_dim=self.spec['feature_dim']
        if self.spec_sha!=self.manifest['spec_sha']:raise ValueError('QPROBE_SEQUENCE_SPEC_HASH')
        for n,item in self.manifest['files'].items():
            if (self.path/n).stat().st_size!=item['bytes']:raise ValueError('QPROBE_SEQUENCE_CACHE_SIZE')
        if sha_file(self.path/'index.jsonl')!=self.manifest['files']['index.jsonl']['sha256']:raise ValueError('QPROBE_SEQUENCE_INDEX_HASH')
        self.index={};offset=0
        for i,line in enumerate((self.path/'index.jsonl').read_text().splitlines()):
            row=json.loads(line)
            if row['row']!=i or row['offset']!=offset or row['length']<1 or row['qid'] in self.index:raise ValueError('QPROBE_SEQUENCE_INDEX_ORDER')
            self.index[row['qid']]=row;offset+=row['length']
        if offset!=self.manifest['total_tokens'] or len(self.index)!=self.manifest['count']:raise ValueError('QPROBE_SEQUENCE_INDEX_COUNT')
        if (self.path/'hidden.bin').stat().st_size!=offset*self.feature_dim*2 or (self.path/'positions.bin').stat().st_size!=offset*8:raise ValueError('QPROBE_SEQUENCE_ARRAY_SIZE')
        self.hidden=np.memmap(self.path/'hidden.bin',dtype='<f2',mode='r',shape=(offset,self.feature_dim))
        self.positions=np.memmap(self.path/'positions.bin',dtype='<i8',mode='r',shape=(offset,))
    def get_features(self,qids,questions):
        if not qids or len(qids)!=len(questions):raise ValueError('QPROBE_SEQUENCE_REQUEST_SHAPE')
        hidden=[];positions=[]
        for qid,q in zip(qids,questions):
            row=self.index[qid]
            if row['question_hash']!=question_text_sha(q):raise ValueError('QPROBE_SEQUENCE_QUESTION_TEXT')
            a=self.hidden[row['offset']:row['offset']+row['length']];b=self.positions[row['offset']:row['offset']+row['length']]
            if hashlib.sha256(a.tobytes()).hexdigest()!=row['hidden_sha256'] or hashlib.sha256(b.tobytes()).hexdigest()!=row['position_sha256']:raise ValueError('QPROBE_CORRUPT_SEQUENCE')
            hidden.append(torch.from_numpy(a.copy()));positions.append(torch.from_numpy(b.copy()))
        return SequenceFeatures(hidden,positions)
    def verify_all(self):
        for n,item in self.manifest['files'].items():
            if sha_file(self.path/n)!=item['sha256']:raise ValueError('QPROBE_SEQUENCE_FULL_SHA:'+n)
        return dict(count=len(self.index),total_tokens=self.manifest['total_tokens'],feature_dim=self.feature_dim,spec_sha=self.spec_sha)


def extract_question_sequences(model, tokenizer, questions, *, feature_layer, max_prompt_tokens=4096, device='cpu'):
    if not questions:raise ValueError('QPROBE_EMPTY_SEQUENCE_EXTRACTION')
    for q in questions:question_text_sha(q)
    if feature_layer != len(model.layers)-1:raise ValueError('QPROBE_REQUIRES_FINAL_DECODER_FEATURES')
    encoded=tokenizer(list(questions),padding=True,truncation=False,return_tensors='pt')
    if int(encoded['attention_mask'].sum(-1).max())>max_prompt_tokens:raise ValueError('QPROBE_SEQUENCE_NO_SILENT_TRUNCATION')
    encoded={k:v.to(device) for k,v in encoded.items()}
    mask=encoded['attention_mask'].bool()
    # Bind the exact positions actually used by the frozen encoder, including
    # its tokenizer padding layout. Compressing valid tokens must retain these.
    positions=torch.arange(mask.shape[1],device=device).expand(mask.shape)
    captured=[]
    def hook(_module,_inputs,out):
        hidden=out[0] if isinstance(out,tuple) else out
        captured.append(SequenceFeatures([h[m].detach().cpu() for h,m in zip(hidden,mask)],
            [p[m].detach().cpu() for p,m in zip(positions,mask)]))
    handle=model.layers[feature_layer].register_forward_hook(hook);training=model.training
    try:
        model.eval();model.requires_grad_(False)
        with torch.inference_mode():model(**encoded,position_ids=positions,use_cache=False,output_hidden_states=False,return_dict=True)
    finally:handle.remove();model.train(training)
    if len(captured)!=1:raise ValueError('QPROBE_SEQUENCE_HOOK_COUNT')
    return captured[0]
