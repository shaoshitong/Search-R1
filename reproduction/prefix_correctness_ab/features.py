"""Remote-only immutable first-request raw-token features for prefix10/20.

Scan raw teacher data once on CPU. Four producer ranks then mmap those inputs,
encode disjoint trajectory occurrences, and publish independent BF16 shards.
No decoding, re-tokenizing, generation, complete-trajectory forward or fitting.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
REPRESENTATION = 'actual_first_request_raw_prefix_alltokens_layer27_v2'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def remote(path):
    p = Path(path)
    if not p.is_absolute() or p.resolve() != p or not str(p).startswith(('/mnt/', '/data/', '/dataset/', '/backup/')):
        raise ValueError('Explicit remote artifact path required: '+str(p))
    return p


def once(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(canonical(value)+b'\n'); f.flush(); os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def progress(out, rank, **value):
    value.update(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), rank=rank)
    p = out/('progress-rank-%d.json'%rank); tmp = p.with_suffix('.tmp')
    tmp.write_bytes(canonical(value)+b'\n'); os.replace(tmp, p)
    print(json.dumps(value, allow_nan=False), flush=True)


def checked_json(path, expected):
    p = remote(path)
    if not expected or p.is_symlink() or sha(p) != expected: raise ValueError('Source JSON SHA mismatch: '+str(p))
    return read(p)


def descriptors(sources):
    """sources.json pins original H100 teacher and an already-generated new512.

    Required: model_revision/model_files_sha256, train{teacher_cache,
    identity_sha256,complete_sha256}, test{identity_path,identity_sha256,
    labels_path,labels_sha256,raw_path,raw_sha256,raw_bytes,selection_path,
    selection_sha256}, canonical_train{path,sha256}. Model path is CLI-owned.
    """
    if sources['model_revision'] != REVISION: raise ValueError('Wrong official revision')
    t = sources['train']; root = remote(t['teacher_cache'])
    identity = checked_json(root/'identity.json', t['identity_sha256'])
    complete = checked_json(root/'complete.json', t['complete_sha256'])
    if complete['questions'] != 5120 or complete['trajectories'] != 25600 or complete['identity_sha256'] != t['identity_sha256']:
        raise ValueError('Teacher completeness binding differs')
    if identity['model_revision'] != REVISION or identity['model_files_sha256'] != sources['model_files_sha256']:
        raise ValueError('Teacher official encoder identity differs')
    desc = []; train = []
    for entry in complete['chunks']:
        name = entry['relative_path']
        if Path(name).name != name: raise ValueError('Chunk path escaped teacher root')
        chunk = checked_json(root/name, entry['sha256']); proof = chunk['raw_rollout']
        if Path(proof['relative_path']).name != proof['relative_path']: raise ValueError('Raw path escaped teacher root')
        train.extend(chunk['groups'])
        desc.append(dict(split='train', raw_path=str(root/proof['relative_path']), raw_sha256=proof['sha256'], raw_bytes=proof['bytes'],
                         label_path=str(root/name), label_sha256=entry['sha256'], groups=chunk['groups']))
    q = sources['test']; new_identity = checked_json(q['identity_path'], q['identity_sha256'])
    labels = checked_json(q['labels_path'], q['labels_sha256']); selection = checked_json(q['selection_path'], q['selection_sha256'])
    heldout = labels['groups']
    for key in ('model_revision','model_files_sha256','train_parquet_sha256','data_manifest_sha256','retriever_version',
                'agent_config','max_turns','scorer_sources','sampling_config','rollout_sources'):
        if new_identity[key] != identity[key]: raise ValueError('Train/test teacher protocol differs: '+key)
    if labels.get('raw_rollout_sha256') != q['raw_sha256'] or labels.get('raw_rollout_bytes') != q['raw_bytes']:
        raise ValueError('New512 raw binding differs')
    if labels.get('selection_sha256') != q['selection_sha256']: raise ValueError('New512 selection binding differs')
    if [(g['qid'],g['question_hash']) for g in heldout] != [(g['qid'],g['question_hash']) for g in selection['questions']]:
        raise ValueError('New512 selected question order differs')
    if [(g['qid'],g['question_hash']) for g in train] != [(g['qid'],g['question_hash']) for g in identity['questions']]:
        raise ValueError('Original teacher question order differs')
    train_ids = {g['qid'] for g in train}; test_ids = {g['qid'] for g in heldout}
    if len(train) != 5120 or len(train_ids) != 5120 or len(heldout) != 512 or len(test_ids) != 512:
        raise ValueError('Expected 5120 train and512 test unique questions')
    if not train_ids.isdisjoint(test_ids) or not {g['question_hash'] for g in train}.isdisjoint({g['question_hash'] for g in heldout}):
        raise ValueError('Train/test question overlap')
    import pyarrow.parquet as pq
    from search_g1.qprobe_features import original_question_and_id
    from search_g1.mixed_selection import question_identity
    data = sources['canonical_train']; parquet = remote(data['path'])
    if sha(parquet) != data['sha256'] or data['sha256'] != identity['train_parquet_sha256']:
        raise ValueError('Canonical train source differs')
    contents = {'train':set(),'test':set()}; found=set()
    for row in pq.read_table(parquet, columns=['prompt','data_source','extra_info']).to_pylist():
        question,qid = original_question_and_id(row)
        if qid in train_ids or qid in test_ids:
            contents['train' if qid in train_ids else 'test'].add(question_identity(question)); found.add(qid)
    if found != train_ids|test_ids or not contents['train'].isdisjoint(contents['test']):
        raise ValueError('Normalized question isolation differs')
    desc.append(dict(split='test',raw_path=q['raw_path'],raw_sha256=q['raw_sha256'],raw_bytes=q['raw_bytes'],
                     label_path=q['labels_path'],label_sha256=q['labels_sha256'],groups=heldout))
    return desc, dict(disjoint_qid=True,disjoint_question_hash=True,disjoint_normalized_content=True,
                     train_questions=5120,test_questions=512,test_regenerated=False,
                     teacher_identity_sha256=t['identity_sha256'],test_identity_sha256=q['identity_sha256'])


def load_verified_raw(descriptor):
    import torch
    p = remote(descriptor['raw_path'])
    with p.open('rb') as f:
        before = os.fstat(f.fileno()); h=hashlib.sha256()
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
        if before.st_size != descriptor['raw_bytes'] or h.hexdigest() != descriptor['raw_sha256']:
            raise ValueError('Raw rollout full SHA differs')
        f.seek(0); raw=torch.load(f,map_location='cpu',weights_only=False);after=os.fstat(f.fileno())
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError('Raw rollout changed while reading')
    return raw


def first_request_prefix(turns, trajectory_id, limit=20):
    """Only native first-request IDs. Never consult whole-turn kept policy masks."""
    if not turns: raise ValueError('Missing original behavior turns')
    turn = turns[0]
    if turn['trajectory_id'] != trajectory_id or turn['turn_index'] != 0:
        raise ValueError('First-request trajectory/ordinal binding differs')
    prompt = turn['actual_request_token_ids']; raw = turn['raw_generated_token_ids']
    if not prompt or any(type(x) is not int or x<0 for x in prompt+raw): raise ValueError('Invalid original token IDs')
    if turn['raw_generated_token_count'] != len(raw): raise ValueError('Raw token count differs')
    # Stop is represented only if actually within the observed raw prefix. No
    # whole-turn finish_reason/final_turn/kept_count is exposed as a feature.
    prefix = raw[:limit]; eos = turn['eos_token_id']
    if eos in prefix: prefix = prefix[:prefix.index(eos)+1]
    return list(prompt), list(prefix), dict(raw_prefix_token_count=len(prefix),observed_eos=eos in prefix,eos_token_id=eos,
        actual_request_sha256=hashlib.sha256(canonical(prompt)).hexdigest(),
        raw_prefix_sha256=hashlib.sha256(canonical(prefix)).hexdigest(),
        mask_position_provenance='reconstructed_unpadded_actual_request_causal_positions',
        whole_turn_future_metadata_used_as_features=False)


def scan(args):
    import numpy as np
    source_path = remote(args.sources); sources=read(source_path); out=remote(args.output)
    out.mkdir(parents=True,exist_ok=False)
    ds,split_proof=descriptors(sources)
    rows=[];ids=[];offsets=[0];seen=set()
    for d in ds:
        raw=load_verified_raw(d); output=raw['output']; identity=raw['identity']; groups=d['groups']
        tids=list(identity['trajectory_ids']); actual=list(output.non_tensor_batch['behavior_trajectory_id'])
        if tids!=actual or tids!=[t for g in groups for t in g['trajectory_ids']] or len(tids)!=len(groups)*5:
            raise ValueError('Trajectory-to-label order differs')
        if raw['question_bindings'] != [dict(qid=g['qid'],question_hash=g['question_hash']) for g in groups]:
            raise ValueError('Original question binding differs')
        turns=output.non_tensor_batch['behavior_turns']
        if len(turns)!=len(tids): raise ValueError('Behavior row count differs')
        for i,tid in enumerate(tids):
            if tid in seen: raise ValueError('Duplicate trajectory identity')
            seen.add(tid); group=groups[i//5]; labels=group['strict_em']
            if len(labels)!=5 or any(type(v)is not int or v not in (0,1) for v in labels): raise ValueError('Invalid trajectory EM labels')
            prompt,prefix,meta=first_request_prefix(turns[i],tid,20); tokens=prompt+prefix
            row=dict(row=len(rows),trajectory_id=tid,split=d['split'],qid=group['qid'],question_hash=group['question_hash'],
                strict_em=labels[i%5],prompt_length=len(prompt),prefix_length=len(prefix),length=len(tokens),offset=offsets[-1],
                source_raw_path=d['raw_path'],source_raw_sha256=d['raw_sha256'],source_raw_row=i,
                source_label_sha256=d['label_sha256'],policy_snapshot_id=identity['policy_snapshot_id'],
                actor_run_id=identity['run_id'],**meta)
            rows.append(row);ids.extend(tokens);offsets.append(len(ids))
        progress(out,0,stage='RAW_ID_SCAN',rows=len(rows));del raw,output
    if sum(r['split']=='train' for r in rows)!=25600 or sum(r['split']=='test' for r in rows)!=2560:
        raise ValueError('Incomplete trajectory dataset')
    model=remote(args.model)
    for name,h in sources['model_files_sha256'].items():
        if Path(name).name!=name or sha(model/name)!=h: raise ValueError('Official encoder asset differs: '+name)
    config=read(model/'config.json')
    pad=sources.get('pad_token_id')
    pad_provenance='sources_json.pad_token_id'
    if pad is None:
        pad=config.get('pad_token_id');pad_provenance='official_config.pad_token_id'
    if pad is None:
        # Query tokenizer metadata only. Original request/prefix IDs are never
        # decoded, encoded or otherwise replaced by the tokenizer.
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
        pad=tokenizer.pad_token_id;pad_provenance='official_tokenizer.pad_token_id'
    if type(pad)is not int or pad<0: raise ValueError('Official pad token ID unavailable')
    np.save(out/'packed_input_ids.npy',np.asarray(ids,dtype=np.int64),allow_pickle=False)
    np.save(out/'offsets.npy',np.asarray(offsets,dtype=np.int64),allow_pickle=False)
    for name in ('packed_input_ids.npy','offsets.npy'):
        with (out/name).open('rb') as f:os.fsync(f.fileno())
    with (out/'index.jsonl').open('x') as f:
        for row in rows:f.write(canonical(row).decode()+'\n')
        f.flush();os.fsync(f.fileno())
    proof=dict(schema_version=2,role='prefix10_20_shared_raw_id_scan',count=len(rows),split_counts=dict(train=25600,test=2560),
        global_max_tokens=max(r['length'] for r in rows),encoder_revision=REVISION,model_dir=str(model),model_files_sha256=sources['model_files_sha256'],
        sources_sha256=sha(source_path),split_proof=split_proof,pad_token_id=pad,pad_token_provenance=pad_provenance,
        position_provenance='reconstructed_contiguous_unpadded_actual_request',prefix_source='first_native_request_raw_including_observed_EOS',
        files={n:dict(sha256=sha(out/n),bytes=(out/n).stat().st_size) for n in ('packed_input_ids.npy','offsets.npy','index.jsonl')})
    once(out/'scan-complete.json',proof);return proof


def validate_prepared_inputs(inputs, sources_path, model_path, *, verify_model=True):
    """Authenticate an original scan, permitting explicit immutable relocation.

    sources.prepared_scan binds the unchanged source scan receipt SHA, original
    sources SHA and encoder file map. Only paths relocate; bytes stay pinned.
    This function performs CPU SHA reads only and never initializes CUDA.
    """
    source = remote(Path(inputs).resolve())
    config_path = remote(sources_path); sources=read(config_path)
    proof_path=source/'scan-complete.json';proof=read(proof_path);model=remote(model_path)
    mapping=proof['model_files_sha256']
    if proof['encoder_revision']!=REVISION:raise ValueError('Prepared encoder revision differs')
    relocated=sources.get('prepared_scan')
    if relocated is None:
        if sha(config_path)!=proof['sources_sha256'] or str(model)!=proof['model_dir']:
            raise ValueError('Prepared input/source/encoder binding differs')
    else:
        if remote(Path(relocated['path']).resolve())!=source or sha(proof_path)!=relocated['scan_complete_sha256']:
            raise ValueError('Relocated original scan receipt differs')
        if proof['sources_sha256']!=relocated['original_sources_sha256'] or mapping!=relocated['source_model_files_sha256']:
            raise ValueError('Relocated data/encoder identity differs')
        if sources.get('model_revision',REVISION)!=REVISION or sources.get('model_files_sha256',mapping)!=mapping:
            raise ValueError('Relocated target encoder mapping differs')
    for name,record in proof['files'].items():
        if Path(name).name!=name or sha(source/name)!=record['sha256'] or (source/name).stat().st_size!=record['bytes']:
            raise ValueError('Prepared input full SHA differs: '+name)
    if verify_model:
        for name,expected in mapping.items():
            if Path(name).name!=name or sha(model/name)!=expected:raise ValueError('Relocated official encoder full SHA differs: '+name)
    relocation=dict(original_scan_sha256=sha(proof_path),original_sources_sha256=proof['sources_sha256'],
                    original_model_dir=proof['model_dir'],target_model_dir=str(model),target_sources_sha256=sha(config_path),
                    same_encoder_files_sha256=mapping,model_full_sha_verified=bool(verify_model),
                    prepared_files_full_sha_verified=True,original_scan_receipt_rewritten=False)
    return proof, relocation


class _LayerCaptured(Exception): pass


def encode(args):
    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import AutoModel
    source=remote(Path(args.inputs).resolve());out=remote(args.output);proof=read(source/'scan-complete.json')
    rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    if world!=4: raise ValueError('Feature extraction uses four actual producer ranks')
    torch.cuda.set_device(local);dist.init_process_group('nccl')
    if rank==0:
        out.mkdir(parents=True,exist_ok=False)
        proof,relocation=validate_prepared_inputs(source,args.sources,args.model)
        once(out/'input-relocation.json',relocation)
    dist.barrier()
    if args.max_tokens!=proof['global_max_tokens'] or args.prefix_length not in (10,20): raise ValueError('Shared twenty-prefix maximum differs')
    rows=[json.loads(line) for line in (source/'index.jsonl').read_text().splitlines()]
    original_offsets=np.load(source/'offsets.npy',allow_pickle=False,mmap_mode='r');packed_ids=np.load(source/'packed_input_ids.npy',allow_pickle=False,mmap_mode='r')
    selected=[];total=0
    for row in rows[rank::world]:
        r=dict(row);r['prefix_length']=min(args.prefix_length,row['prefix_length']);r['length']=row['prompt_length']+r['prefix_length']
        r['source_prefix20_sha256']=row['raw_prefix_sha256']
        original_start=int(original_offsets[row['row']]);prefix_ids=packed_ids[original_start+row['prompt_length']:original_start+r['length']].tolist()
        r['raw_prefix_sha256']=hashlib.sha256(canonical(prefix_ids)).hexdigest();r['raw_prefix_token_count']=len(prefix_ids);r['observed_eos']=r['eos_token_id'] in prefix_ids
        r['feature_shard']=rank;r['feature_offset']=total;r['row']=row['row'];total+=r['length'];selected.append(r)
    model=AutoModel.from_pretrained(args.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,low_cpu_mem_usage=True)
    model.requires_grad_(False);model.eval();model.to('cuda:'+str(local))
    if len(model.layers)!=36 or model.config.hidden_size!=2048: raise ValueError('Official Qwen model dimensions differ')
    captured=[]
    def hook(_module,_inputs,output):
        captured.append(output[0] if isinstance(output,tuple) else output)
        # No layers28..35/final norm/LM vocabulary projection are needed. The
        # original decoder attention mask and RoPE creation before layer27 run.
        raise _LayerCaptured()
    handle=model.layers[27].register_forward_hook(hook)
    target=out/('features-shard-%d.npy'%rank)
    array=np.lib.format.open_memmap(target,mode='w+',dtype=np.uint16,shape=(total,2048))
    index=out/('index-shard-%d.jsonl'%rank)
    try:
        with index.open('x') as log,torch.inference_mode():
            for start in range(0,len(selected),args.micro_batch_size):
                chosen=selected[start:start+args.micro_batch_size];width=max(r['length'] for r in chosen)
                tokens=torch.full((len(chosen),width),proof['pad_token_id'],dtype=torch.long,device=local)
                mask=torch.zeros_like(tokens);positions=torch.zeros_like(tokens)
                for i,r in enumerate(chosen):
                    lo=int(original_offsets[r['row']]);length=r['length']
                    tokens[i,:length]=torch.from_numpy(np.array(packed_ids[lo:lo+length],copy=True)).to(local)
                    mask[i,:length]=1;positions[i,:length]=torch.arange(length,device=local)
                captured.clear()
                try:model(input_ids=tokens,attention_mask=mask,position_ids=positions,use_cache=False,output_hidden_states=False,return_dict=True)
                except _LayerCaptured:pass
                if len(captured)!=1:raise ValueError('Layer27 capture missing')
                hidden=captured.pop()
                for i,r in enumerate(chosen):
                    h=hidden[i,:r['length']].detach().cpu().contiguous()
                    if h.dtype!=torch.bfloat16 or not bool(torch.isfinite(h).all()):raise ValueError('Invalid frozen feature tensor')
                    raw=h.view(torch.int16).numpy().view(np.uint16);lo=r['feature_offset'];array[lo:lo+r['length']]=raw
                    r['feature_sha256']=hashlib.sha256(raw.tobytes()).hexdigest();log.write(canonical(r).decode()+'\n')
                log.flush();array.flush();progress(out,rank,stage='LAYER27_PREFIX_ENCODING',completed_rows=start+len(chosen),total_rows=len(selected),prefix_length=args.prefix_length)
            log.flush();os.fsync(log.fileno());array.flush()
    finally:handle.remove()
    del model,array;torch.cuda.empty_cache()
    shard=dict(rank=rank,count=len(selected),total_tokens=total,files={n:dict(sha256=sha(out/n),bytes=(out/n).stat().st_size) for n in (target.name,index.name)})
    once(out/('shard-%d-complete.json'%rank),shard);dist.barrier()
    if rank==0:
        shards=[read(out/('shard-%d-complete.json'%r)) for r in range(world)];combined=[];files={}
        for sh in shards:
            combined.extend(json.loads(line) for line in (out/('index-shard-%d.jsonl'%sh['rank'])).read_text().splitlines());files.update(sh['files'])
        combined.sort(key=lambda r:r['row'])
        if [r['row'] for r in combined]!=list(range(proof['count'])):raise ValueError('Feature shard coverage differs')
        with (out/'index.jsonl').open('x') as f:
            for r in combined:f.write(canonical(r).decode()+'\n')
            f.flush();os.fsync(f.fileno())
        files['index.jsonl']=dict(sha256=sha(out/'index.jsonl'),bytes=(out/'index.jsonl').stat().st_size)
        spec=dict(representation=REPRESENTATION,prefix_length=args.prefix_length,feature_layer=27,boundary='post_block',feature_dim=2048,
                  encoder_revision=REVISION,encoder_model_dir=str(remote(args.model)),encoder_files_sha256=proof['model_files_sha256'],
                  feature_dtype='bfloat16',encoder_compute_dtype='bfloat16',global_max_tokens=args.max_tokens,pad_token_id=proof['pad_token_id'],
                  pad_token_provenance=proof['pad_token_provenance'],
                  position_provenance=proof['position_provenance'],prefix_source=proof['prefix_source'],split_question_isolation=True,
                  encoder_frozen=True,encoder_optimizer_updates=0,labels_passed_to_encoder=False,executed_decoder_indices=list(range(28)))
        manifest=dict(schema_version=2,representation=REPRESENTATION,count=proof['count'],split_counts=proof['split_counts'],feature_dim=2048,
                      global_max_tokens=args.max_tokens,spec=spec,spec_sha=hashlib.sha256(canonical(spec)).hexdigest(),files=files,shards=shards,
                      inputs_dir=str(source),inputs_scan_sha256=sha(source/'scan-complete.json'),sources_sha256=proof['sources_sha256'],
                      relocated_sources_sha256=sha(args.sources),input_relocation=read(out/'input-relocation.json'),split_proof=proof['split_proof'])
        once(out/'manifest.json',manifest);once(out/'features-complete.json',dict(role='PREFIX_FEATURES_COMPLETE',prefix_length=args.prefix_length,
             manifest_sha256=sha(out/'manifest.json'),count=proof['count'],global_max_tokens=args.max_tokens,actor_optimizer_updates=0,head_optimizer_updates=0))
    dist.barrier();dist.destroy_process_group()


class PrefixFeatureCache:
    def __init__(self,path,expected_manifest_sha=None):
        import numpy as np
        self.path=remote(path);p=self.path/'manifest.json'
        if expected_manifest_sha is not None and sha(p)!=expected_manifest_sha:raise ValueError('Feature manifest SHA differs')
        self.manifest=read(p);self.spec=self.manifest['spec']
        if self.manifest['schema_version']!=2 or self.manifest['representation']!=REPRESENTATION or self.spec['prefix_length'] not in (10,20):
            raise ValueError('Wrong prefix feature schema')
        self.spec_sha=hashlib.sha256(canonical(self.spec)).hexdigest()
        if self.spec_sha!=self.manifest['spec_sha'] or self.spec['feature_layer']!=27 or self.spec['boundary']!='post_block':raise ValueError('Wrong feature encoder spec')
        self.feature_dim=2048;self.max_tokens=self.manifest['global_max_tokens'];self.ids_by_split={'train':[],'test':[]};self.index={}
        self.shards={}
        if sha(self.path/'index.jsonl')!=self.manifest['files']['index.jsonl']['sha256']:raise ValueError('Feature index SHA differs')
        for sh in self.manifest['shards']:
            name='features-shard-%d.npy'%sh['rank'];p=self.path/name
            if p.stat().st_size!=sh['files'][name]['bytes']:raise ValueError('Feature shard size differs')
            self.shards[sh['rank']]=np.load(p,allow_pickle=False,mmap_mode='r')
            if self.shards[sh['rank']].dtype!=np.uint16 or self.shards[sh['rank']].shape!=(sh['total_tokens'],2048):raise ValueError('Feature shard layout differs')
        for line in (self.path/'index.jsonl').read_text().splitlines():
            r=json.loads(line);tid=r['trajectory_id']
            if tid in self.index or type(r['strict_em'])is not int or r['strict_em']not in(0,1) or r['length']>self.max_tokens:raise ValueError('Feature row identity differs')
            self.index[tid]=r;self.ids_by_split[r['split']].append(tid)
        self.ids=list(self.index);self.verified=set()
        self.inputs=Path(self.manifest['inputs_dir']);self.offsets=np.load(self.inputs/'offsets.npy',allow_pickle=False,mmap_mode='r')
        self.input_ids=np.load(self.inputs/'packed_input_ids.npy',allow_pickle=False,mmap_mode='r')
        if len(self.ids)!=self.manifest['count'] or {k:len(v) for k,v in self.ids_by_split.items()}!=self.manifest['split_counts']:
            raise ValueError('Incomplete feature cache')

    def verify_all(self):
        for name,record in self.manifest['files'].items():
            if sha(self.path/name)!=record['sha256']:raise ValueError('Feature full SHA differs: '+name)
        return dict(count=len(self.ids),manifest_sha256=sha(self.path/'manifest.json'),all_files_sha_verified=True)

    def get_batch(self,trajectory_ids,device='cpu'):
        import numpy as np
        import torch
        rows=[self.index[tid] for tid in trajectory_ids];width=max(r['length'] for r in rows)
        features=torch.zeros((len(rows),width,2048),dtype=torch.bfloat16);mask=torch.zeros((len(rows),width),dtype=torch.bool)
        pos=torch.zeros((len(rows),width),dtype=torch.long);tokens=torch.full((len(rows),width),self.spec['pad_token_id'],dtype=torch.long)
        policy=torch.zeros_like(mask)
        for i,r in enumerate(rows):
            lo=r['feature_offset'];n=r['length'];raw=np.array(self.shards[r['feature_shard']][lo:lo+n],copy=True)
            if r['trajectory_id']not in self.verified:
                if hashlib.sha256(raw.tobytes()).hexdigest()!=r['feature_sha256']:raise ValueError('Feature row SHA differs')
                self.verified.add(r['trajectory_id'])
            features[i,:n]=torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16);mask[i,:n]=1;pos[i,:n]=torch.arange(n)
            offset=int(self.offsets[r['row']]);tokens[i,:n]=torch.from_numpy(np.array(self.input_ids[offset:offset+n],copy=True));policy[i,r['prompt_length']:n]=1
        return dict(features=features.to(device),attention_mask=mask.to(device),position_ids=pos.to(device),input_ids=tokens.to(device),policy_mask=policy.to(device))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',required=True);p.add_argument('--scan-only',action='store_true');p.add_argument('--inputs')
    p.add_argument('--prefix-length',type=int,choices=(10,20),default=20);p.add_argument('--max-tokens',type=int)
    p.add_argument('--micro-batch-size',type=int,default=8);a=p.parse_args()
    if not sys.platform.startswith('linux'):raise ValueError('Feature tensors and raw trajectories stay remote Linux')
    if a.scan_only:
        print(json.dumps(scan(a)),flush=True);return
    if not a.inputs or not a.max_tokens:p.error('Encoding requires --inputs and shared --max-tokens')
    if a.micro_batch_size<1:p.error('Positive encoder micro batch required')
    try:encode(a)
    except BaseException as error:
        out=remote(a.output);out.mkdir(parents=True,exist_ok=True)
        once(out/('feature-failure-rank-%s.json'%os.environ.get('RANK','0')),dict(error_type=type(error).__name__,message=str(error),automatic_retry=False))
        raise

if __name__=='__main__':main()
