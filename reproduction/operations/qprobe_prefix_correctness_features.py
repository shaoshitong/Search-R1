"""Prepare immutable remote features from actual prompt plus five policy tokens.

Old 5120x5 and the already generated new512x5 are read and fully hashed. No
generation, decoding, re-tokenization, actor update or Head fitting occurs.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

sys.dont_write_bytecode=True
RUNTIME_SHA='fcb77204f48d5cb8ec21c68df7d4a2c937b3557a'
REVISION='aa8e72537993ba99e69dfaafa59ed015b17504d1'
MODEL='/data/search-g1/assets/qwen2.5-3b-instruct'
OLD_TEACHER='/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v4-sigma3-fullouter'
OLD_IDENTITY_SHA='88998de7a30455b9135b02cc3c82251fbc1ba1765708244538b6a66c1371e871'
OLD_COMPLETE_SHA='07f4e2e3937c814637b783468ff1062af7ba701e1ac1e136da2dd56f838b45b1'
TEST_BASE='/data/shaoshitong/Search-G1-a800-fresh200-16034e59031e4f21a323ebb2556bef7f'
TEST_RAW_SHA='a43b2884a44a4f01a0efbdb083aa4d3ff7e0209d02b6b27a336169d4ec4f4517'
REPRESENTATION='actual_prompt_policy_prefix5_alltokens_layer27_v1'
INPUT_FILES=('packed_input_ids.npy','packed_position_ids.npy','packed_policy_mask.npy','offsets.npy','lengths.npy')
FILES=('packed_features.npy',)+INPUT_FILES+('index.jsonl',)


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def remote(path):
    p=Path(path)
    assert p.is_absolute() and p.resolve()==p and str(p).startswith(('/data/','/mnt/','/backup/'))
    return p


def once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(canonical(value)+b'\n');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def progress(out,**value):
    event=dict(value,observed_unix=time.time())
    with (out/'progress.jsonl').open('ab') as f:f.write(canonical(event)+b'\n');f.flush();os.fsync(f.fileno())
    temp=out/'progress.partial.json';temp.write_bytes(canonical(event)+b'\n');os.replace(temp,out/'progress.json')
    print(json.dumps(event,allow_nan=False),flush=True)


def source_descriptors(args):
    from search_g1.qprobe_warmup import moments
    teacher=remote(args.teacher_cache);test=remote(args.test_base)
    assert str(teacher)==OLD_TEACHER and str(test)==TEST_BASE
    assert sha(teacher/'identity.json')==OLD_IDENTITY_SHA and sha(teacher/'complete.json')==OLD_COMPLETE_SHA
    identity=read(teacher/'identity.json');complete=read(teacher/'complete.json')
    assert complete['questions']==5120 and complete['trajectories']==25600 and complete['identity_sha256']==OLD_IDENTITY_SHA
    assert identity['model_revision']==REVISION
    descriptors=[];train=[]
    for entry in complete['chunks']:
        label_path=teacher/entry['relative_path'];assert label_path.parent==teacher and sha(label_path)==entry['sha256']
        labels=read(label_path);raw=labels['raw_rollout'];raw_path=teacher/raw['relative_path']
        assert raw_path.parent==teacher and raw_path.stat().st_size==raw['bytes']
        for g in labels['groups']:
            assert moments(g['strict_em'])==(g['correct_count'],g['mean'],g['std'])
        train.extend(labels['groups'])
        descriptors.append(dict(split='train',raw_path=str(raw_path),raw_sha256=raw['sha256'],raw_bytes=raw['bytes'],
            label_path=str(label_path),label_sha256=entry['sha256'],groups=labels['groups']))
    assert len(train)==5120 and len({g['qid'] for g in train})==5120
    assert [(g['qid'],g['question_hash']) for g in train]==[(g['qid'],g['question_hash']) for g in identity['questions']]
    label_path=test/'teacher-cache/labels.json';labels=read(label_path);run=read(test/'logs/run-manifest.json')
    new_identity_path=test/'teacher-cache/identity.json';new_identity=read(new_identity_path)
    assert labels['role']=='actual_frozen_official_new512_G5_labels' and labels['questions']==512 and labels['trajectories']==2560
    assert labels['raw_rollout_sha256']==TEST_RAW_SHA
    assert run['model_revision']==REVISION and run['initial_model']==MODEL
    assert identity['model_files_sha256']==run['initial_model_sha256']
    assert new_identity['model_files_sha256']==identity['model_files_sha256']
    for name in ('model_revision','train_parquet_sha256','data_manifest_sha256','retriever_version','agent_config','max_turns','scorer_sources','rollout_sources'):
        assert new_identity[name]==identity[name], 'Historical/new label protocol differs: '+name
    assert new_identity['sampling_config']==identity['sampling_config'], 'Historical/new complete sampling configuration differs'
    sampling_keys=('name','dtype','temperature','top_p','top_k','do_sample','n','n_agent','response_length')
    old_sampling={k:identity['sampling_config'].get(k) for k in sampling_keys}
    new_sampling={k:new_identity['sampling_config'].get(k) for k in sampling_keys}
    assert old_sampling==new_sampling, 'Historical/new teacher sampling semantics differ'
    assert old_sampling['temperature']==old_sampling['top_p']==1. and old_sampling['dtype']=='bfloat16'
    protocol=dict(old_identity_sha256=OLD_IDENTITY_SHA,new_identity_sha256=sha(new_identity_path),
        official_model_equal=True,canonical_train_equal=True,agent_config_equal=True,retriever_version_equal=True,
        strict_scorer_sources_equal=True,sampling_semantics_equal=True,full_sampling_config_equal=True,
        rollout_sources_equal=True,sampling_semantics=old_sampling,
        agent_config=identity['agent_config'],max_turns=identity['max_turns'],retriever_version=identity['retriever_version'],
        scorer_sources=identity['scorer_sources'],
        old_full_sampling_config_sha256=hashlib.sha256(canonical(identity['sampling_config'])).hexdigest(),
        new_full_sampling_config_sha256=hashlib.sha256(canonical(new_identity['sampling_config'])).hexdigest(),
        old_rollout_sources_sha256=hashlib.sha256(canonical(identity['rollout_sources'])).hexdigest(),
        new_rollout_sources_sha256=hashlib.sha256(canonical(new_identity['rollout_sources'])).hexdigest())
    assert labels['selection_sha256']==sha(test/'selection.json')
    selection=read(test/'selection.json');heldout=labels['groups']
    assert [(g['qid'],g['question_hash']) for g in heldout]==[(g['qid'],g['question_hash']) for g in selection['questions']]
    for g in heldout:assert moments(g['strict_em'])==(g['correct_count'],g['mean'],g['std'])
    assert len(heldout)==512 and len({g['qid'] for g in heldout})==512
    train_ids={g['qid'] for g in train};test_ids={g['qid'] for g in heldout}
    train_hash={g['question_hash'] for g in train};test_hash={g['question_hash'] for g in heldout}
    assert train_ids.isdisjoint(test_ids) and train_hash.isdisjoint(test_hash), 'Train/test question identity overlap'
    # Verify normalized question isolation against the unchanged canonical data.
    import pyarrow.parquet as pq
    from search_g1.qprobe_features import original_question_and_id
    from search_g1.mixed_selection import question_identity
    canonical_train=test/'data/train.parquet'
    assert sha(canonical_train)==identity['train_parquet_sha256']==run['input_sha256']['train.parquet']
    contents={'train':set(),'test':set()};found=set()
    for row in pq.read_table(canonical_train,columns=['prompt','data_source','extra_info']).to_pylist():
        question,qid=original_question_and_id(row)
        if qid in train_ids or qid in test_ids:
            split='train' if qid in train_ids else 'test';contents[split].add(question_identity(question));found.add(qid)
    assert found==train_ids|test_ids and contents['train'].isdisjoint(contents['test']), 'Normalized train/test question overlap'
    descriptors.append(dict(split='test',raw_path=str(test/'teacher-cache/rollout.pt'),raw_sha256=TEST_RAW_SHA,
        raw_bytes=labels['raw_rollout_bytes'],label_path=str(label_path),label_sha256=sha(label_path),groups=heldout))
    proof=dict(train=dict(question_count=5120,trajectory_count=25600,ordered_qids_sha256=hashlib.sha256(canonical([g['qid'] for g in train])).hexdigest()),
        test=dict(question_count=512,trajectory_count=2560,ordered_qids_sha256=hashlib.sha256(canonical([g['qid'] for g in heldout])).hexdigest()),
        disjoint_qid=True,disjoint_question_hash=True,disjoint_normalized_question_content=True,
        fixed_test_reused_each_epoch=True,test_g5_regenerated=False,old_teacher_identity_sha256=OLD_IDENTITY_SHA,
        old_teacher_complete_sha256=OLD_COMPLETE_SHA,test_selection_sha256=sha(test/'selection.json'),
        canonical_train_sha256=sha(canonical_train),historical_new_teacher_protocol_binding=protocol)
    return descriptors,proof,run


def load_verified_raw(descriptor):
    import torch
    path=remote(descriptor['raw_path'])
    with path.open('rb') as f:
        before=os.fstat(f.fileno());h=hashlib.sha256()
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
        assert before.st_size==descriptor['raw_bytes'] and h.hexdigest()==descriptor['raw_sha256'], 'Full raw rollout SHA changed'
        f.seek(0);raw=torch.load(f,map_location='cpu',weights_only=False)
        after=os.fstat(f.fileno())
        assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    return raw


def actual_prefix(raw,row_index):
    """Take actual prompt and every valid context token through policy token5."""
    import torch
    batch=raw['output'].batch;p=int(batch['prompts'].shape[1])
    tokens=batch['input_ids'][row_index];attention=batch['attention_mask'][row_index]
    positions=batch['position_ids'][row_index];policy=batch['sampling_mask'][row_index]
    assert tokens.ndim==positions.ndim==attention.ndim==policy.ndim==1
    assert tokens.dtype==positions.dtype==torch.int64 and policy.dtype==torch.bool
    assert len(tokens)==p+len(policy) and bool(((attention==0)|(attention==1)).all())
    assert torch.equal(tokens[:p],batch['prompts'][row_index]) and torch.equal(tokens[p:],batch['responses'][row_index])
    assert torch.equal(batch['info_mask'][row_index,p:].bool(),policy)
    valid=attention.bool();assert not bool((policy&~valid[p:]).any())
    loc=policy.nonzero().flatten();count=int(loc.numel())
    if count==0:raise ValueError('PREFIX_IDENTITY_UNKNOWN_ZERO_POLICY_TOKENS')
    k=min(5,count);response_end=int(loc[k-1])+1;full_end=p+response_end
    prefix_valid=valid[:full_end];assert bool(prefix_valid[:p].any())
    ids=tokens[:full_end][prefix_valid].detach().cpu().numpy().astype('<i8',copy=True)
    pos=positions[:full_end][prefix_valid].detach().cpu().numpy().astype('<i8',copy=True)
    full_policy=torch.cat((torch.zeros(p,dtype=torch.bool),policy[:response_end]))
    pmask=full_policy[prefix_valid].cpu().numpy().astype(np.bool_,copy=True)
    assert len(ids)==len(pos)==len(pmask) and int(pmask.sum())==k and bool((pos>=0).all())
    prefix_sha=hashlib.sha256(canonical(dict(version=1,length=len(ids),policy_tokens=k))+ids.tobytes()+pos.tobytes()+pmask.tobytes()).hexdigest()
    meta=dict(length=len(ids),actual_prompt_tokens=int(valid[:p].sum()),policy_prefix_tokens=k,
        total_actual_policy_tokens=count,short_policy_prefix=count<5,response_cut_exclusive=response_end,
        observation_or_correction_tokens_before_cut=int(valid[p:full_end].sum())-k,
        prefix_sha256=prefix_sha,input_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        position_ids_sha256=hashlib.sha256(pos.tobytes()).hexdigest(),policy_mask_sha256=hashlib.sha256(pmask.tobytes()).hexdigest())
    pad_values=tokens[~valid].unique();assert pad_values.numel()==1
    return ids,pos,pmask,meta,int(pad_values.item())


def scan(args,out):
    from search_g1.qprobe_warmup import verify_frozen
    descriptors,split_proof,run=source_descriptors(args)
    rows=[];inputs=[];positions=[];masks=[];seen=set();pad_id=None;sources=[];hashed_bytes=0
    for descriptor in descriptors:
        raw=load_verified_raw(descriptor);output=raw['output'];identity=raw['identity'];groups=descriptor['groups']
        frozen=raw['frozen_workers'];verify_frozen(frozen['before'],frozen['after'],4)
        tids=identity['trajectory_ids'];actual=output.non_tensor_batch['behavior_trajectory_id'].tolist()
        assert tids==actual and len(tids)==len(set(tids))==len(groups)*5
        assert raw['question_bindings']==[dict(qid=g['qid'],question_hash=g['question_hash']) for g in groups]
        label_tids=[t for g in groups for t in g['trajectory_ids']];assert tids==label_tids
        assert output.batch['input_ids'].shape[0]==len(tids) and output.batch['position_ids'].ndim==2
        assert output.meta_info['policy_snapshot_id']==identity['policy_snapshot_id']
        for local,tid in enumerate(tids):
            assert tid not in seen;seen.add(tid);g=groups[local//5]
            ids,pos,mask,meta,actual_pad=actual_prefix(raw,local)
            pad_id=actual_pad if pad_id is None else pad_id;assert pad_id==actual_pad
            label=g['strict_em'][local%5];assert type(label) is int and label in (0,1)
            rows.append(dict(row=len(rows),trajectory_id=tid,split=descriptor['split'],qid=g['qid'],question_hash=g['question_hash'],
                strict_em=label,source_raw_path=descriptor['raw_path'],source_raw_sha256=descriptor['raw_sha256'],source_raw_row=local,
                source_label_path=descriptor['label_path'],source_label_sha256=descriptor['label_sha256'],
                actor_run_id=identity['run_id'],policy_snapshot_id=identity['policy_snapshot_id'],
                source_commit=run['source_commit'] if descriptor['split']=='test' else None,
                source_map_sha256=hashlib.sha256(canonical(identity['source_sha256'])).hexdigest(),**meta))
            inputs.append(ids);positions.append(pos);masks.append(mask)
        hashed_bytes+=descriptor['raw_bytes'];sources.append({k:v for k,v in descriptor.items() if k!='groups'})
        progress(out,stage='FULL_SHA_RAW_PREFIX_SCAN',completed_trajectories=len(rows),raw_files_verified=len(sources),raw_bytes_full_sha_verified=hashed_bytes)
        del raw,output
    assert sum(r['split']=='train' for r in rows)==25600 and sum(r['split']=='test' for r in rows)==2560
    lengths=np.asarray([r['length'] for r in rows],dtype=np.int32)
    offsets=np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(lengths,dtype=np.int64)))
    prepared=out/'prepared-prefixes';prepared.mkdir()
    for name,array in (('packed_input_ids.npy',np.concatenate(inputs)),('packed_position_ids.npy',np.concatenate(positions)),
                       ('packed_policy_mask.npy',np.concatenate(masks)),('offsets.npy',offsets),('lengths.npy',lengths)):
        np.save(prepared/name,array,allow_pickle=False)
        with (prepared/name).open('rb') as f:os.fsync(f.fileno())
    with (prepared/'index.jsonl').open('x') as f:
        for r,offset in zip(rows,offsets):r['offset']=int(offset);f.write(canonical(r).decode()+'\n')
        f.flush();os.fsync(f.fileno())
    model=remote(MODEL);config_path=model/'config.json';assert sha(config_path)==run['initial_model_sha256']['config.json']
    spec=dict(representation=REPRESENTATION,encoder_model_dir=str(model),encoder_revision=REVISION,
        encoder_files_sha256=run['initial_model_sha256'],encoder_config_sha256=sha(config_path),
        feature_layer=27,feature_dim=2048,encoder_compute_dtype='bfloat16',feature_dtype='bfloat16',
        global_max_tokens=int(lengths.max()),prefix_policy_tokens=5,actual_position_ids=True,
        input_rule='raw input_ids+attention_mask through inclusive fifth true response sampling_mask token; no decode/tokenize',
        short_prefix_rule='keep actual1..4 policy tokens without adding/repeating EOS or tokens; zero is unknown and rejected',
        split_question_isolation=True,teacher_resampled=False,freeze_encoder=True,detach_features=True,pad_token_id=pad_id)
    proof=dict(schema_version=1,role='actual_full_raw_prompt_policy_prefix_scan',count=len(rows),
        split_counts=dict(train=25600,test=2560),minimum_tokens=int(lengths.min()),global_max_tokens=int(lengths.max()),
        total_tokens=int(offsets[-1]),short_prefix_count=sum(r['short_policy_prefix'] for r in rows),
        unknown_zero_policy_count=0,spec=spec,spec_sha=hashlib.sha256(canonical(spec)).hexdigest(),
        split_proof=split_proof,sources=sources,raw_bytes_full_sha_verified=hashed_bytes,
        files={name:dict(bytes=(prepared/name).stat().st_size,sha256=sha(prepared/name)) for name in INPUT_FILES+('index.jsonl',)},
        runtime_source_commit=RUNTIME_SHA,producer_sha256=sha(__file__),checked_unix=time.time())
    once(out/'scan-complete.json',proof)
    progress(out,stage='ACTUAL_PREFIX_SCAN_COMPLETE',trajectories=len(rows),global_max_tokens=spec['global_max_tokens'],total_tokens=proof['total_tokens'])
    return proof,rows


def frozen_encoder_state(model):
    """Hash real loaded parameters and buffers; no optimizer or model copy."""
    import torch
    h=hashlib.sha256();tensors=elements=0
    for kind,items in (('parameter',model.named_parameters()),('buffer',model.named_buffers())):
        for name,value in items:
            assert not value.requires_grad
            cpu=value.detach().cpu().contiguous()
            h.update(canonical(dict(kind=kind,name=name,dtype=str(cpu.dtype),shape=list(cpu.shape))))
            raw=memoryview(cpu.reshape(-1).view(torch.uint8).numpy()).cast('B')
            for offset in range(0,len(raw),8*1024**2):h.update(raw[offset:offset+8*1024**2])
            tensors+=1;elements+=cpu.numel()
    return dict(tensors=tensors,elements=elements,loaded_parameter_and_buffer_sha256=h.hexdigest(),
        requires_grad=False,model_eval=model.training is False)


def encode(args,out,scan_proof,rows,gpu):
    import torch
    from transformers import AutoModel
    spec=scan_proof['spec'];model_path=remote(spec['encoder_model_dir'])
    for name,expected in spec['encoder_files_sha256'].items():assert sha(model_path/name)==expected, 'Official encoder asset SHA changed: '+name
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    model=AutoModel.from_pretrained(str(model_path),torch_dtype=torch.bfloat16,attn_implementation='sdpa',
        local_files_only=True,trust_remote_code=False,low_cpu_mem_usage=True)
    model.requires_grad_(False);model.eval();model.to('cuda:0')
    assert len(model.layers)==36 and not any(p.requires_grad for p in model.parameters())
    before=frozen_encoder_state(model)
    progress(out,stage='OFFICIAL_FROZEN_BASE_DECODER_LOADED',layer=27,full_vocab_lm_head_called=False,
        loaded_state_sha256=before['loaded_parameter_and_buffer_sha256'])
    source=out/'prepared-prefixes';offsets=np.load(source/'offsets.npy',allow_pickle=False)
    input_ids=np.load(source/'packed_input_ids.npy',allow_pickle=False,mmap_mode='r')
    positions=np.load(source/'packed_position_ids.npy',allow_pickle=False,mmap_mode='r')
    required=int(offsets[-1])*2048*2+1024**3
    assert shutil.disk_usage(out).free>=required, 'Insufficient remote space for the actual BF16 packed cache'
    target=out/'cache';assert not target.exists()
    partial=Path(tempfile.mkdtemp(prefix='cache.partial-',dir=out))
    for name in INPUT_FILES:os.link(source/name,partial/name)
    packed=np.lib.format.open_memmap(partial/'packed_features.npy',mode='w+',dtype=np.uint16,shape=(int(offsets[-1]),2048))
    index_file=(partial/'index.jsonl').open('x');captured=[]
    def hook(_module,_inputs,output):
        value=output[0] if isinstance(output,tuple) else output
        assert value.ndim==3 and value.shape[-1]==2048 and value.dtype==torch.bfloat16
        captured.append(value.detach())
    handle=model.layers[27].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for start in range(0,len(rows),args.micro_batch_size):
                chosen=rows[start:start+args.micro_batch_size];width=max(r['length'] for r in chosen)
                tokens=torch.full((len(chosen),width),spec['pad_token_id'],dtype=torch.long,device='cuda:0')
                pos=torch.zeros_like(tokens);mask=torch.zeros_like(tokens)
                for ordinal,row in enumerate(chosen):
                    lo,hi=map(int,offsets[row['row']:row['row']+2]);length=hi-lo
                    tokens[ordinal,:length]=torch.from_numpy(np.array(input_ids[lo:hi],copy=True)).to('cuda:0')
                    pos[ordinal,:length]=torch.from_numpy(np.array(positions[lo:hi],copy=True)).to('cuda:0');mask[ordinal,:length]=1
                captured.clear();model(input_ids=tokens,attention_mask=mask,position_ids=pos,use_cache=False,output_hidden_states=False,return_dict=True)
                assert len(captured)==1 and captured[0].shape[:2]==tokens.shape
                hidden=captured.pop()
                for ordinal,row in enumerate(chosen):
                    value=hidden[ordinal,:row['length']].detach().cpu().contiguous()
                    assert bool(torch.isfinite(value).all()) and value.shape==(row['length'],2048)
                    array=value.view(torch.int16).numpy().view(np.uint16);lo,hi=map(int,offsets[row['row']:row['row']+2])
                    packed[lo:hi]=array
                    record=dict(row,feature_sha256=hashlib.sha256(array.tobytes()).hexdigest())
                    index_file.write(canonical(record).decode()+'\n')
                index_file.flush();packed.flush()
                progress(out,stage='FROZEN_OFFICIAL_LAYER27_PREFIX_ENCODING',completed_trajectories=start+len(chosen),total_trajectories=len(rows))
    finally:handle.remove();index_file.flush();os.fsync(index_file.fileno());index_file.close();packed.flush()
    after=frozen_encoder_state(model);assert before==after, 'Frozen encoder parameters/buffers changed during extraction'
    once(out/'frozen-encoder-proof.json',dict(before=before,after=after,exact_equal=True,
        encoder_model_dir=str(model_path),encoder_files_sha256=spec['encoder_files_sha256'],
        encoder_config_sha256=spec['encoder_config_sha256'],encoder_revision=REVISION,
        native_base_decoder_only=True,full_vocab_lm_head_called=False,only_layer27_hook_output_retained=True,
        labels_passed_to_encoder=False,encoder_optimizer_created=False,checked_unix=time.time()))
    del packed,model;torch.cuda.empty_cache()
    for name in FILES:
        with (partial/name).open('rb') as f:os.fsync(f.fileno())
    manifest=dict(schema_version=1,representation=REPRESENTATION,spec=spec,spec_sha=scan_proof['spec_sha'],count=len(rows),
        feature_dim=2048,global_max_tokens=spec['global_max_tokens'],total_tokens=int(offsets[-1]),
        split_counts=scan_proof['split_counts'],split_proof=scan_proof['split_proof'],sources=scan_proof['sources'],
        scan_complete_sha256=sha(out/'scan-complete.json'),frozen_encoder_proof_sha256=sha(out/'frozen-encoder-proof.json'),
        frozen_encoder_eval=True,encoder_optimizer_updates=0,
        actor_optimizer_updates=0,head_optimizer_updates=0,runtime_source_commit=RUNTIME_SHA,producer_sha256=sha(__file__),
        files={name:dict(bytes=(partial/name).stat().st_size,sha256=sha(partial/name)) for name in FILES},checked_unix=time.time())
    once(partial/'manifest.json',manifest)
    for p in partial.iterdir():p.chmod(0o444)
    os.rename(partial,target)
    fd=os.open(out,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    result=dict(role='ACTUAL_PROMPT_PREFIX5_LAYER27_CACHE_COMPLETE',cache_dir=str(target),manifest_sha256=sha(target/'manifest.json'),
        count=manifest['count'],split_counts=manifest['split_counts'],global_max_tokens=manifest['global_max_tokens'],
        total_tokens=manifest['total_tokens'],feature_bytes=manifest['files']['packed_features.npy']['bytes'],gpu=gpu,
        G5_regenerated=False,actor_trained=False,head_trained=False,checked_unix=time.time())
    once(out/'features-complete.json',result);progress(out,**result)


class PrefixFeatureCache:
    """NumPy-only validation until get_batch imports Torch after GPU admission."""
    def __init__(self,path,expected_manifest_sha=None):
        self.path=remote(path);manifest_path=self.path/'manifest.json'
        if expected_manifest_sha is not None:assert sha(manifest_path)==expected_manifest_sha
        self.manifest=read(manifest_path);self.spec=self.manifest['spec'];self.spec_sha=hashlib.sha256(canonical(self.spec)).hexdigest()
        assert self.manifest['schema_version']==1 and self.manifest['representation']==REPRESENTATION==self.spec['representation']
        assert self.spec_sha==self.manifest['spec_sha'] and self.spec['encoder_revision']==REVISION
        assert self.spec['feature_layer']==27 and self.spec['feature_dtype']==self.spec['encoder_compute_dtype']=='bfloat16'
        assert self.spec['actual_position_ids'] is True and self.spec['split_question_isolation'] is True
        self.feature_dim=2048;self.max_tokens=self.spec['global_max_tokens'];self.ids=[];self.index={};self.ids_by_split={'train':[],'test':[]}
        assert self.max_tokens==self.manifest['global_max_tokens']>0 and set(self.manifest['files'])==set(FILES)
        for name in FILES:
            p=self.path/name;assert p.resolve()==p and not p.is_symlink() and p.stat().st_size==self.manifest['files'][name]['bytes']
        for name in ('offsets.npy','lengths.npy','index.jsonl'):assert sha(self.path/name)==self.manifest['files'][name]['sha256']
        self.offsets=np.load(self.path/'offsets.npy',allow_pickle=False);self.lengths=np.load(self.path/'lengths.npy',allow_pickle=False)
        self.packed=np.load(self.path/'packed_features.npy',allow_pickle=False,mmap_mode='r')
        self.position_ids=np.load(self.path/'packed_position_ids.npy',allow_pickle=False,mmap_mode='r')
        self.input_ids=np.load(self.path/'packed_input_ids.npy',allow_pickle=False,mmap_mode='r')
        self.policy_mask=np.load(self.path/'packed_policy_mask.npy',allow_pickle=False,mmap_mode='r')
        n=self.manifest['count'];total=self.manifest['total_tokens']
        assert self.offsets.dtype==np.int64 and self.offsets.shape==(n+1,) and self.offsets[0]==0 and self.offsets[-1]==total
        assert self.lengths.dtype==np.int32 and self.lengths.shape==(n,) and np.array_equal(np.diff(self.offsets),self.lengths)
        assert int(self.lengths.min())>0 and int(self.lengths.max())==self.max_tokens
        assert self.packed.dtype==np.uint16 and self.packed.shape==(total,2048)
        assert self.position_ids.dtype==self.input_ids.dtype==np.int64 and self.position_ids.shape==self.input_ids.shape==(total,)
        assert self.policy_mask.dtype==np.bool_ and self.policy_mask.shape==(total,)
        question_ids={'train':set(),'test':set()};question_hashes={'train':set(),'test':set()}
        for ordinal,line in enumerate((self.path/'index.jsonl').read_text().splitlines()):
            row=json.loads(line);tid=row['trajectory_id'];split=row['split']
            assert row['row']==ordinal and tid not in self.index and split in self.ids_by_split
            assert row['offset']==int(self.offsets[ordinal]) and row['length']==int(self.lengths[ordinal])
            assert type(row['strict_em']) is int and row['strict_em'] in (0,1) and 1<=row['policy_prefix_tokens']<=5
            self.ids.append(tid);self.index[tid]=row;self.ids_by_split[split].append(tid)
            question_ids[split].add(row['qid']);question_hashes[split].add(row['question_hash'])
        assert len(self.ids)==n==28160 and len(self.ids_by_split['train'])==25600 and len(self.ids_by_split['test'])==2560
        assert len(question_ids['train'])==5120 and len(question_ids['test'])==512
        assert question_ids['train'].isdisjoint(question_ids['test']) and question_hashes['train'].isdisjoint(question_hashes['test'])
        self.verified=set();self.all_verified=False

    def verify_all(self):
        for name in FILES:assert sha(self.path/name)==self.manifest['files'][name]['sha256'], 'Full cache SHA changed: '+name
        self.all_verified=True
        return dict(role='actual_prefix_cache_full_sha_pass',manifest_sha256=sha(self.path/'manifest.json'),count=len(self.ids),
            spec_sha=self.spec_sha,split_counts={k:len(v) for k,v in self.ids_by_split.items()},
            all_files_sha_verified=True,global_max_tokens=self.max_tokens,checked_unix=time.time())

    def get_batch(self,trajectory_ids,device='cpu'):
        import torch
        assert trajectory_ids and not isinstance(trajectory_ids,(str,bytes))
        rows=[self.index[tid] for tid in trajectory_ids];width=max(r['length'] for r in rows)
        features=torch.zeros((len(rows),width,2048),dtype=torch.bfloat16)
        attention=torch.zeros((len(rows),width),dtype=torch.bool);positions=torch.zeros((len(rows),width),dtype=torch.long)
        tokens=torch.full((len(rows),width),self.spec['pad_token_id'],dtype=torch.long);policy=torch.zeros_like(attention)
        for ordinal,row in enumerate(rows):
            lo,hi=map(int,self.offsets[row['row']:row['row']+2]);array=np.array(self.packed[lo:hi],copy=True)
            value=torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
            if row['trajectory_id'] not in self.verified:
                if not self.all_verified:assert hashlib.sha256(array.tobytes()).hexdigest()==row['feature_sha256']
                assert bool(torch.isfinite(value).all());self.verified.add(row['trajectory_id'])
            length=hi-lo;features[ordinal,:length]=value;attention[ordinal,:length]=True
            positions[ordinal,:length]=torch.from_numpy(np.array(self.position_ids[lo:hi],copy=True))
            tokens[ordinal,:length]=torch.from_numpy(np.array(self.input_ids[lo:hi],copy=True))
            policy[ordinal,:length]=torch.from_numpy(np.array(self.policy_mask[lo:hi],copy=True))
        return dict(features=features.to(device),attention_mask=attention.to(device),position_ids=positions.to(device),
            input_ids=tokens.to(device),policy_mask=policy.to(device))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime-repo',required=True)
    p.add_argument('--teacher-cache',default=OLD_TEACHER);p.add_argument('--test-base',default=TEST_BASE)
    p.add_argument('--output-dir',required=True);p.add_argument('--physical-index',type=int,required=True)
    p.add_argument('--micro-batch-size',type=int,default=32);a=p.parse_args()
    repo=remote(a.runtime_repo);out=remote(a.output_dir)
    assert a.physical_index in (0,1,2,3,5,6) and a.micro_batch_size==32
    assert not out.is_relative_to(repo) and not out.is_relative_to(remote(a.teacher_cache)) and not out.is_relative_to(remote(a.test_base))
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==RUNTIME_SHA
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip()
    sys.path.insert(0,str(repo));out.mkdir(parents=True,exist_ok=False)
    for name in ('runtime-cache/hf','runtime-cache/triton','runtime-cache/inductor','tmp'):(out/name).mkdir(parents=True)
    os.environ.update(HF_HOME=str(out/'runtime-cache/hf'),XDG_CACHE_HOME=str(out/'runtime-cache'),TMPDIR=str(out/'tmp'),
        TRITON_CACHE_DIR=str(out/'runtime-cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(out/'runtime-cache/inductor'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    inventory=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=30))))
    expected=next(r[1].strip() for r in inventory if int(r[0])==a.physical_index)
    environ=dict(item.split(b'=',1) for item in Path('/proc/self/environ').read_bytes().split(b'\0') if b'=' in item)
    born=environ.get(b'CUDA_VISIBLE_DEVICES',b'').decode()
    assert born==os.environ.get('CUDA_VISIBLE_DEVICES')==expected, 'Feature process must be born with its exact admitted GPU UUID'
    from reproduction.a800_common.qprobe_token_warmup import gpu_admission
    lock,gpu=gpu_admission(a.physical_index);assert os.environ['CUDA_VISIBLE_DEVICES']==born
    gpu.update(birth_cuda_visible_devices=born,visibility_changed_after_birth=False);once(out/'gpu-admission.json',gpu)
    from reproduction.a800.owned_processes import identity
    once(out/'feature-claim.json',dict(producer=identity(os.getpid()),runtime_source_commit=RUNTIME_SHA,
        producer_sha256=sha(__file__),G5_regeneration_forbidden=True,actor_and_head_training_forbidden=True,claimed_unix=time.time()))
    try:
        proof,rows=scan(a,out);encode(a,out,proof,rows,gpu)
    except BaseException as error:
        once(out/'feature-failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,
            unknown_identity=str(error)=='PREFIX_IDENTITY_UNKNOWN_ZERO_POLICY_TOKENS',failed_unix=time.time()))
        raise


if __name__=='__main__':main()
