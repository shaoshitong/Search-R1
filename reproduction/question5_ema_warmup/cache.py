"""Remote packed BF16 question+policy5 features, with accepted A800 reuse.

Builder workers partition complete event G5 groups. Reusable features remain in
their immutable producer files; only newly encoded features are written. The
reader reconstructs fixed Qmax+5 slots and never reruns the frozen encoder.
No token, feature, model or optimizer payload is written to the local computer.
"""
from collections import OrderedDict
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid

import numpy as np

from .data import (ARRAY_NAMES, CompositePrefix5Dataset, FEATURE_DIM, FEATURE_ROLE,
                   POLICY_TOKENS, REVISION, canonical, expected_policy_positions,
                   initial_question_segment, retained_five, sha)

ROLE='question5_packed_bf16_features_v1'


def remote(value):
    path=Path(value)
    if not path.is_absolute() or path.resolve()!=path or not str(path).startswith(('/data/','/mnt/','/backup/')):
        raise ValueError('Question5 feature caches require canonical remote storage')
    return path


def read(path,limit=64*1024**2):
    path=remote(path);s=path.stat()
    if path.is_symlink() or not path.is_file() or s.st_uid!=os.getuid() or s.st_size>limit:
        raise ValueError('Unsafe or oversized question5 cache metadata')
    return json.loads(path.read_bytes())


def signature(path):
    path=remote(path);s=path.stat()
    if path.is_symlink() or not path.is_file() or s.st_uid!=os.getuid():raise ValueError('Foreign cache file')
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def write_once(path,value):
    raw=canonical(value)+b'\n'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def source_binding():
    root=Path(__file__).parent
    return {name:sha(root/name) for name in ('cache.py','data.py','encoder.py')}


def process_birth():
    path=Path('/proc')/str(os.getpid());text=(path/'stat').read_text();fields=text[text.rfind(')')+2:].split()
    return dict(pid=os.getpid(),start_ticks=int(fields[19]),sid=int(fields[3]),uid=path.stat().st_uid)


def admitted_encoder_factory(model_dir,physical_index,device,micro_batch_size,token_budget=65536):
    """Lazy UUID admission, borrowing a validated inherited parent lease if set."""
    def factory(dataset):
        import fcntl
        if device!='cuda:0':raise ValueError('Production encoder uses one born GPU UUID and logical cuda:0')
        values=[entry.split(b'=',1)[1].decode() for entry in Path('/proc/self/environ').read_bytes().split(b'\0') if entry.startswith(b'CUDA_VISIBLE_DEVICES=')]
        if len(values)!=1 or not values[0].startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES')!=values[0]:
            raise ValueError('Cache encoder requires its exact single GPU UUID at process birth')
        inventory=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=15))))
        rows=[row for row in inventory if int(row[0])==physical_index]
        if len(rows)!=1 or rows[0][1].strip()!=values[0] or int(rows[0][2])>1024 or int(rows[0][3])!=0:
            raise ValueError('Assigned cache GPU is occupied or its UUID differs; no foreign stop permitted')
        root=Path('/data/shaoshitong/Search-G1-qprobe-shared/token-feature-gpu-locks');root.mkdir(parents=True,exist_ok=True)
        path=root/(values[0]+'.lock');inherited=os.environ.get('QUESTION5_GPU_LEASE_FD')
        if inherited is not None:
            descriptor=int(inherited);actual=os.fstat(descriptor);expected=path.stat()
            if (descriptor<3 or path.is_symlink() or path.resolve()!=path or actual.st_uid!=os.getuid() or
                    (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino)):
                raise ValueError('Inherited GPU lease FD does not identify the exact bound UUID lock')
            fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            lock=os.fdopen(descriptor,'a')
        else:
            lock=path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            apps=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                '--format=csv,noheader,nounits'],text=True,timeout=15))))
            if any(row and row[0].strip()==values[0] for row in apps):raise ValueError('Assigned cache GPU has a compute process')
            from .encoder import FrozenQuestion5Encoder
            inner=FrozenQuestion5Encoder(dataset,model_dir,device=device,micro_batch_size=micro_batch_size,token_budget=token_budget)
        except BaseException:lock.close();raise
        class AdmittedEncoder:
            admission=dict(physical_index=physical_index,gpu_uuid=values[0],birth_cuda_visible_devices=values[0],
                           producer_identity=process_birth(),hostname=socket.gethostname(),checked_unix=time.time(),
                           inherited_parent_gpu_lease=inherited is not None,
                           lease_device=os.fstat(lock.fileno()).st_dev,lease_inode=os.fstat(lock.fileno()).st_ino)
            @property
            def model(self):return inner.model
            def encode_rows(self,*args,**kwargs):return inner.encode_rows(*args,**kwargs)
            def close(self):
                try:inner.close()
                finally:lock.close()
        return AdmittedEncoder()
    return factory


def source_row(dataset,row):
    """Reference the immutable source index; retain only its small typed layout."""
    source_id,offset,event_id=dataset._refs[row['trajectory_id']]
    source=dataset.sources[source_id];path=remote(source.index_path)
    with path.open('rb') as stream:stream.seek(offset);raw=stream.readline()
    return dict(source_id=source_id,index_path=str(path),index_offset=offset,
        index_line_sha256=hashlib.sha256(raw).hexdigest(),index_file=signature(path),
        source_scan_sha256=source.scan_sha256,event_id=event_id,
        original_trajectory_id=row['original_trajectory_id'])


def layout(dataset,row):
    qlen=initial_question_segment(row)['question_length'];segments,bounds,plen=retained_five(row)
    if not 1<=qlen<=dataset.question_max_tokens or not 1<=plen<=POLICY_TOKENS:raise ValueError('Invalid real-token layout')
    arrays=dataset.source_arrays(row['source_id']);offset=row['policy_offset']
    positions=np.asarray(arrays[ARRAY_NAMES[2]][offset:offset+plen],dtype=np.int64).tolist()
    if positions!=expected_policy_positions(bounds):raise ValueError('Original policy positions changed')
    return dict(trajectory_id=row['trajectory_id'],event_id=row['event_id'],event_group_id=row['event_id'],
        qid=str(row['qid']),strict_em=row['strict_em'],question_length=qlen,policy_length=plen,
        policy_positions=positions,policy_segments=bounds,source=source_row(dataset,row))


class AcceptedA800Reuse:
    """Strict producer-proof reuse; mismatch causes one fresh encoding instead."""
    def __init__(self,spec,max_open=8):
        self.spec=spec;self.max_open=max_open;self.open=OrderedDict();self.accepted={};self.manifests={}

    def _accept(self,row):
        if row.get('source_cache_accepted') is not True:return None
        path=row.get('source_cache_acceptance_path');digest=row.get('source_cache_acceptance_sha256')
        if not path or not digest:return None
        key=(path,digest)
        if key not in self.accepted:
            if sha(remote(path))!=digest:raise ValueError('A800 feature acceptance SHA changed')
            receipt=read(path);proof=receipt.get('audit',receipt)
            if receipt.get('accepted',proof.get('accepted')) is not True or proof.get('accepted') is not True:
                raise ValueError('A800 features have no accepted final resource/source proof')
            if (proof.get('stage')!='PREFIX_SCALE20_FINAL_METADATA_PRODUCER_PROOF_ACCEPTED' or
                    proof.get('runtime_source')!='fcb77204f48d5cb8ec21c68df7d4a2c937b3557a' or
                    proof.get('run_id')!='62407ad0ed404586998439ca1f55b902' or
                    proof.get('execution_id')!='dd3032e313e14ce2b9c8fea996aaf721' or
                    proof.get('only_owned_cuda')!=[] or proof.get('producer_full_sha_proofs_rechecked') is not True):
                raise ValueError('Wrong A800 accepted feature experiment')
            composite_path=Path('/data/shaoshitong/Search-G1-a800-fresh200-0850db2977bd4801b9ced300b0ab387f/features/composite-manifest.json')
            if sha(composite_path)!=proof['composite_manifest_sha256']:raise ValueError('Accepted A800 composite changed')
            composite=read(composite_path);model=composite['spec']
            if (model['encoder_revision']!=REVISION or model['feature_layer']!=27 or model['feature_dim']!=2048 or
                    model['encoder_files_sha256']!=self.spec['encoder_files_sha256']):
                raise ValueError('A800 cached encoder/model identity differs')
            allowed={item['path']:item['manifest_sha256'] for item in composite['shards']}
            allowed[composite['reuse']['cache_dir']]=composite['reuse']['manifest_sha256']
            self.accepted[key]=(proof,allowed)
        return self.accepted[key]

    def _arrays(self,path,expected,proof):
        key=str(path)
        if key in self.open:
            value=self.open.pop(key);self.open[key]=value;return value
        if sha(path/'manifest.json')!=expected:raise ValueError('Accepted A800 shard manifest changed')
        manifest=read(path/'manifest.json')
        if manifest.get('encoder_loaded_state_sha256',proof['frozen_encoder_state_sha256'])!=proof['frozen_encoder_state_sha256']:
            raise ValueError('A800 frozen loaded encoder digest differs')
        names=('packed_features.npy','packed_input_ids.npy','packed_position_ids.npy','packed_policy_mask.npy','offsets.npy')
        seals={name:signature(path/name) for name in names}
        for name in names:
            if seals[name]['bytes']!=manifest['files'][name]['bytes']:raise ValueError('Accepted A800 file size changed')
        arrays={name:np.load(path/name,allow_pickle=False,mmap_mode='r') for name in names}
        if arrays['packed_features.npy'].dtype!=np.uint16 or arrays['packed_features.npy'].shape[1:]!=(2048,):
            raise ValueError('A800 feature storage is not BF16 words')
        self.open[key]=(arrays,seals);self.manifests[key]=expected
        while len(self.open)>self.max_open:self.open.popitem(last=False)
        return arrays,seals

    def find(self,dataset,row,meta):
        row=dict(row);source_proof=dataset.sources[row['source_id']].proof
        if row.get('source_cache_accepted') is True:
            row.setdefault('source_cache_acceptance_path',source_proof.get('source_cache_acceptance_path'))
            if source_proof.get('source_cache_acceptance_sha256')!=row.get('source_cache_acceptance_sha256'):
                raise ValueError('A800 row and source acceptance binding differ')
        accepted=self._accept(row)
        if accepted is None or not row.get('source_cache'):return None
        proof,allowed=accepted;path=remote(row['source_cache']);expected=row['source_cache_manifest_sha256']
        if allowed.get(str(path))!=expected:raise ValueError('A800 source cache is outside accepted composite')
        arrays,seals=self._arrays(path,expected,proof)
        ordinal=row['source_cache_row'];lo,hi=map(int,arrays['offsets.npy'][ordinal:ordinal+2])
        if (lo!=row['source_cache_offset'] or hi-lo!=row['source_cache_length']):raise ValueError('A800 original row boundaries changed')
        chosen=np.asarray(row['selected_feature_indices'],dtype=np.int64)
        qlen,plen=meta['question_length'],meta['policy_length']
        if (chosen.shape!=(qlen+plen,) or np.any(chosen<0) or np.any(chosen>=hi-lo) or
                not np.array_equal(chosen[:qlen],np.arange(qlen)) or np.any(np.diff(chosen)<=0)):
            raise ValueError('A800 selected feature indices do not preserve original question/policy order')
        tokens=arrays['packed_input_ids.npy'][lo:hi];positions=arrays['packed_position_ids.npy'][lo:hi]
        policy=arrays['packed_policy_mask.npy'][lo:hi];source=dataset.source_arrays(row['source_id'])
        contexts,policy_ids,_=(source[name] for name in ARRAY_NAMES)
        segments,bounds,_=retained_five(row);offset=row['policy_offset']
        question=initial_question_segment(row);qstart=question['context_offset']
        wanted=np.concatenate((contexts[qstart:qstart+qlen],policy_ids[offset:offset+plen]))
        if (not np.array_equal(tokens[chosen],wanted) or not np.array_equal(positions[chosen],list(range(qlen))+meta['policy_positions']) or
                np.any(policy[chosen[:qlen]]) or not np.all(policy[chosen[qlen:]])):
            return None
        # Causal equality requires the complete authentic request before each
        # policy segment, not just equality of selected token IDs/positions.
        for segment in segments:
            start,take=segment['policy_start'],segment['policy_length']
            last=int(chosen[qlen+start+take-1])+1
            context=contexts[segment['context_offset']:segment['context_offset']+segment['context_length']]
            if (len(context)!=last or not np.array_equal(tokens[:last],context) or
                    not np.array_equal(positions[:last],np.arange(last))):return None
        return dict(kind='accepted_a800_reference',path=str(path/'packed_features.npy'),
            file=seals['packed_features.npy'],offset=lo,selected_indices=chosen.tolist(),
            producer_manifest_sha256=expected,acceptance_sha256=row['source_cache_acceptance_sha256'],
            loaded_encoder_state_sha256=proof['frozen_encoder_state_sha256'],complete_causal_context_equal=True)


def _batch_records(encoder,dataset,rows,records,packed):
    import torch
    batch=encoder.encode_rows(rows,array_provider=dataset.source_arrays)
    if batch['trajectory_ids']!=[row['trajectory_id'] for row in rows]:raise ValueError('Encoder changed ordered flat IDs')
    for index,record in enumerate(records):
        qlen,plen=record['question_length'],record['policy_length'];mask=batch['attention_mask'][index]
        if int(mask.sum())!=qlen+plen or batch['question_lengths'][index].item()!=qlen:
            raise ValueError('Encoder changed actual question/policy token count')
        expected_positions=list(range(qlen))+record['policy_positions']
        if batch['position_ids'][index][mask].tolist()!=expected_positions:raise ValueError('Encoder changed original positions')
        value=batch['features'][index][mask].detach().to(device='cpu',dtype=torch.bfloat16).contiguous()
        if value.shape!=(qlen+plen,2048) or not torch.isfinite(value).all():raise ValueError('Invalid frozen encoder features')
        words=value.view(torch.int16).numpy().view(np.uint16)
        lo=record['feature']['offset'];packed[lo:lo+len(words)]=words


def build_cache(input_manifest,output_dir,model_dir,*,worker_index=0,world_size=1,
                groups_per_shard=256,micro_batch_size=32,token_budget=65536,device='cuda:0',encoder_factory=None):
    """Build this worker once; a separate CPU finalize binds all worker shards."""
    manifest_path=remote(input_manifest);output=remote(output_dir);model_dir=remote(model_dir)
    if not 0<=worker_index<world_size or groups_per_shard<1 or micro_batch_size<1 or token_budget<1:raise ValueError('Invalid worker bounds')
    datasets={split:CompositePrefix5Dataset(manifest_path,split=split) for split in ('train','test')}
    dataset=datasets['train'];spec=dict(dataset.spec);qmax=dataset.question_max_tokens
    if datasets['test'].question_max_tokens!=qmax or datasets['test'].spec!=spec:raise ValueError('Train/test encoder spec or Qmax differs')
    groups=[(split,group) for split in ('train','test') for group in datasets[split].groups]
    assigned=[(ordinal,split,group) for ordinal,(split,group) in enumerate(groups) if ordinal%world_size==worker_index]
    if not assigned:raise ValueError('Empty cache worker assignment')
    output.mkdir(parents=True,exist_ok=True);worker=output/('worker-%03d'%worker_index);worker.mkdir(mode=0o700,exist_ok=False)
    binding=dict(role=ROLE,source_manifest_path=str(manifest_path),source_manifest_sha256=dataset.manifest_sha,
        spec=spec,question_max_tokens=qmax,max_tokens=qmax+5,source_sha256=source_binding(),
        worker_index=worker_index,world_size=world_size,model_dir=str(model_dir),
        micro_batch_size=micro_batch_size,token_budget=token_budget,
        total_groups={split:len(datasets[split].groups) for split in datasets},assigned_groups=len(assigned),
        no_teacher_generation=True,no_actor_updates=True,no_probe_updates=True,created_unix=time.time(),
        producer_identity=process_birth(),hostname=socket.gethostname())
    write_once(worker/'claim.json',binding);reuse=AcceptedA800Reuse(spec);encoder=None;receipts=[];reused=encoded=0
    try:
        for shard_index,start in enumerate(range(0,len(assigned),groups_per_shard)):
            selected=assigned[start:start+groups_per_shard];name='shard-%06d'%shard_index
            temporary=worker/(name+'.partial-'+uuid.uuid4().hex);temporary.mkdir(mode=0o700)
            records=[];shard_groups=[];misses=[];total=0
            for ordinal,split,group in selected:
                ds=datasets[split];rows=ds.get_rows(group['trajectory_ids'])
                if len(rows)!=5 or [r['strict_em'] for r in rows]!=group['labels']:raise ValueError('Cache requires complete aligned event G5')
                shard_groups.append(dict(group,split=split,global_group_ordinal=ordinal))
                for row in rows:
                    meta=layout(ds,row);meta['split']=split;ref=reuse.find(ds,row,meta)
                    if ref is None:
                        meta['feature']=dict(kind='packed',path='packed_features.npy',offset=total,length=meta['question_length']+meta['policy_length'])
                        total+=meta['feature']['length'];misses.append((ds,row,meta));encoded+=1
                    else:meta['feature']=ref;reused+=1
                    records.append(meta)
            packed=None
            if total:
                if encoder is None:
                    if encoder_factory is None:
                        from .encoder import FrozenQuestion5Encoder
                        encoder=FrozenQuestion5Encoder(dataset,model_dir,device=device,micro_batch_size=micro_batch_size,token_budget=token_budget)
                    else:encoder=encoder_factory(dataset)
                    if hasattr(encoder,'admission'):write_once(worker/'encoder-admission.json',encoder.admission)
                packed=np.lib.format.open_memmap(temporary/'packed_features.npy',mode='w+',dtype=np.uint16,shape=(total,2048))
                cursor=0
                while cursor<len(misses):
                    ds=misses[cursor][0];chunk=[]
                    while cursor<len(misses) and misses[cursor][0] is ds and len(chunk)<micro_batch_size:
                        chunk.append(misses[cursor]);cursor+=1
                    _batch_records(encoder,ds,[x[1] for x in chunk],[x[2] for x in chunk],packed)
                    packed.flush()
                packed.flush();del packed
                with (temporary/'packed_features.npy').open('rb') as stream:os.fsync(stream.fileno())
            with (temporary/'index.jsonl').open('xb') as stream:
                for record in records:stream.write(canonical(record)+b'\n')
                stream.flush();os.fsync(stream.fileno())
            write_once(temporary/'groups.json',shard_groups)
            files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p),file_identity=signature(p)) for p in temporary.iterdir()}
            shard=dict(role=ROLE,source_manifest_sha256=dataset.manifest_sha,source_sha256=binding['source_sha256'],
                worker_index=worker_index,shard_index=shard_index,groups=len(shard_groups),trajectories=len(records),
                newly_encoded_trajectories=sum(r['feature']['kind']=='packed' for r in records),
                reused_trajectories=sum(r['feature']['kind']!='packed' for r in records),packed_real_tokens=total,
                question_max_tokens=qmax,max_tokens=qmax+5,feature_dim=2048,feature_dtype='bfloat16_uint16_words',
                padding_stored=False,complete_event_groups=True,files=files,checked_unix=time.time())
            write_once(temporary/'manifest.json',shard);sync_dir(temporary)
            target=worker/name;os.rename(temporary,target);sync_dir(worker)
            receipts.append(dict(path=str(target),manifest_sha256=sha(target/'manifest.json'),groups=len(shard_groups),trajectories=len(records)))
            write_once(worker/('progress-%06d.json'%shard_index),dict(completed_groups=sum(x['groups'] for x in receipts),
                reused_trajectories=reused,newly_encoded_trajectories=encoded,checked_unix=time.time()))
        if encoder is not None:
            if any(p.requires_grad for p in encoder.model.parameters()) or encoder.model.training:raise ValueError('Encoder lost frozen eval state')
        result=dict(binding,stage='QUESTION5_PACKED_CACHE_WORKER_COMPLETE',shards=receipts,
            reused_trajectories=reused,newly_encoded_trajectories=encoded,encoder_optimizer_updates=0,
            encoded_padding_tokens=0,completed_unix=time.time())
        write_once(worker/'complete.json',result);return result
    finally:
        if encoder is not None:encoder.close()
        for ds in datasets.values():ds.close()


def finalize_cache(output_dir,*,world_size):
    """CPU-only merge of immutable shard references; no feature consolidation."""
    output=remote(output_dir);workers=[read(output/('worker-%03d'%i)/'complete.json') for i in range(world_size)]
    first=workers[0];same=('role','source_manifest_path','source_manifest_sha256','spec','question_max_tokens','max_tokens','source_sha256','total_groups')
    if any(any(w[k]!=first[k] for k in same) or w['worker_index']!=i or w['world_size']!=world_size or
           w['stage']!='QUESTION5_PACKED_CACHE_WORKER_COMPLETE' for i,w in enumerate(workers)):
        raise ValueError('Cache worker source/spec/partition differs')
    if sha(first['source_manifest_path'])!=first['source_manifest_sha256']:raise ValueError('Source composite changed')
    shards=[s for w in workers for s in w['shards']];seen_groups=set();seen_ids=set();counts={'train':0,'test':0};ordinals=set()
    for item in shards:
        path=remote(item['path'])
        if not path.is_relative_to(output) or sha(path/'manifest.json')!=item['manifest_sha256']:raise ValueError('Cache shard manifest changed')
        manifest=read(path/'manifest.json');groups=read(path/'groups.json')
        if manifest['trajectories']!=len(groups)*5 or manifest['groups']!=len(groups):raise ValueError('Incomplete cached event G5')
        for name in ('groups.json','index.jsonl'):
            if sha(path/name)!=manifest['files'][name]['sha256']:raise ValueError('Cached metadata SHA changed')
        for group in groups:
            ids=group['trajectory_ids'];event=group['event_group_id'];ordinal=group['global_group_ordinal']
            if len(ids)!=5 or len(set(ids))!=5 or len(group['labels'])!=5 or event in seen_groups or seen_ids.intersection(ids) or ordinal in ordinals:
                raise ValueError('Duplicate/partial cache event')
            if ordinal%world_size!=manifest['worker_index']:raise ValueError('Wrong group worker partition')
            seen_groups.add(event);seen_ids.update(ids);ordinals.add(ordinal);counts[group['split']]+=1
    if counts!=first['total_groups'] or ordinals!=set(range(sum(counts.values()))):raise ValueError('Cache does not cover complete merged source')
    result=dict(schema_version=1,role=ROLE,stage='QUESTION5_PACKED_FEATURE_CACHE_COMPLETE',
        source_manifest_path=first['source_manifest_path'],source_manifest_sha256=first['source_manifest_sha256'],
        source_sha256=first['source_sha256'],spec=first['spec'],question_max_tokens=first['question_max_tokens'],
        max_tokens=first['max_tokens'],feature_dim=2048,prefix_tokens=5,feature_dtype='bfloat16_uint16_words',
        groups_by_split=counts,trajectories_by_split={k:v*5 for k,v in counts.items()},shards=shards,
        reused_trajectories=sum(w['reused_trajectories'] for w in workers),
        newly_encoded_trajectories=sum(w['newly_encoded_trajectories'] for w in workers),
        padding_stored=False,source_labels_and_positions_referenced=True,encoder_optimizer_updates=0,
        no_teacher_generation=True,no_actor_updates=True,no_probe_updates=True,checked_unix=time.time())
    write_once(output/'manifest.json',result);sync_dir(output);return result


class PackedFeatureReader:
    """Ordered flat-ID reader, mmap LRU, fixed Qmax question slots +5 tail."""
    def __init__(self,manifest_path,split='train',device='cpu',*,max_open_shards=8,expected_manifest_sha=None):
        if split not in ('train','test') or max_open_shards<1:raise ValueError('Invalid cache split/LRU bound')
        self.path=remote(manifest_path);self.manifest=read(self.path);self.manifest_sha=sha(self.path);self.sha=self.manifest_sha
        if expected_manifest_sha is not None and self.manifest_sha!=expected_manifest_sha:raise ValueError('Feature manifest SHA differs')
        m=self.manifest
        if m['role']!=ROLE or m['stage']!='QUESTION5_PACKED_FEATURE_CACHE_COMPLETE' or m['prefix_tokens']!=5 or m['padding_stored'] is not False:
            raise ValueError('Require a completed question+policy5 packed feature cache')
        self.spec=dict(m['spec']);self.spec_sha=hashlib.sha256(canonical(self.spec)).hexdigest();self.split=split;self.device=device
        self.question_max_tokens=m['question_max_tokens'];self.max_tokens=m['max_tokens'];self.feature_dim=2048
        if self.max_tokens!=self.question_max_tokens+5 or self.spec['encoder_revision']!=REVISION:raise ValueError('Cache layout/model differs')
        if sha(remote(m['source_manifest_path']))!=m['source_manifest_sha256']:raise ValueError('Source dataset manifest changed')
        self.groups=[];self.index={};self._open=OrderedDict();self._checked_sources=set();self.max_open_shards=max_open_shards
        for item in m['shards']:
            path=remote(item['path'])
            if sha(path/'manifest.json')!=item['manifest_sha256']:raise ValueError('Cache shard SHA changed')
            shard=read(path/'manifest.json')
            for name in ('index.jsonl','groups.json'):
                if sha(path/name)!=shard['files'][name]['sha256']:raise ValueError('Cache index/group SHA changed')
            groups=read(path/'groups.json');self.groups.extend(g for g in groups if g['split']==split)
            with (path/'index.jsonl').open('rb') as stream:
                for line in stream:
                    row=json.loads(line)
                    if row['split']!=split:continue
                    tid=row['trajectory_id']
                    if tid in self.index:raise ValueError('Duplicate cached trajectory ID')
                    row['_shard']=str(path)
                    if row['feature']['kind']=='packed':
                        seal=shard['files']['packed_features.npy']['file_identity']
                        if signature(path/'packed_features.npy')!=seal:raise ValueError('Encoded shard file identity changed')
                        row['feature']['file']=seal
                    self.index[tid]=row
        self.groups.sort(key=lambda g:g['global_group_ordinal']);self.ids=[tid for g in self.groups for tid in g['trajectory_ids']]
        self.ids_by_split={'train':[],'test':[]};self.ids_by_split[split]=self.ids
        if (len(self.groups)!=m['groups_by_split'][split] or len(self.ids)!=len(self.index) or len(set(self.ids))!=len(self.ids)
                or set(self.ids)!=set(self.index)):raise ValueError('Cache split does not contain all complete G5 groups')
        for group in self.groups:
            if len(group['trajectory_ids'])!=5 or [self.index[t]['strict_em'] for t in group['trajectory_ids']]!=group['labels']:
                raise ValueError('Cache group and trajectory labels differ')

    def __len__(self):return len(self.ids)

    def _array(self,path,seal):
        path=remote(path);key=str(path)
        if signature(path)!=seal:raise ValueError('Sealed feature file identity changed')
        if key in self._open:value=self._open.pop(key);self._open[key]=value;return value
        value=np.load(path,allow_pickle=False,mmap_mode='r')
        if value.dtype!=np.uint16 or value.ndim!=2 or value.shape[1]!=2048:raise ValueError('Packed feature words have wrong dtype/shape')
        self._open[key]=value
        while len(self._open)>self.max_open_shards:self._open.popitem(last=False)
        return value

    def _source_check(self,row):
        source=row['source'];key=(source['index_path'],source['index_offset'])
        if key in self._checked_sources:return
        path=remote(source['index_path'])
        if signature(path)!=source['index_file']:raise ValueError('Source index file identity changed')
        with path.open('rb') as stream:stream.seek(source['index_offset']);raw=stream.readline()
        if hashlib.sha256(raw).hexdigest()!=source['index_line_sha256']:raise ValueError('Source event/label index line changed')
        self._checked_sources.add(key)

    def get_batch(self,ordered_flat_ids,device=None):
        import torch
        ids=list(ordered_flat_ids)
        if not ids:raise ValueError('Require nonempty ordered trajectory IDs')
        rows=[self.index[t] for t in ids];width=self.max_tokens;qmax=self.question_max_tokens
        features=torch.zeros((len(rows),width,2048),dtype=torch.bfloat16)
        question_mask=torch.zeros((len(rows),width),dtype=torch.bool);policy_mask=torch.zeros_like(question_mask)
        positions=torch.zeros((len(rows),width),dtype=torch.long);qlens=[];plens=[];bounds=[]
        for i,row in enumerate(rows):
            self._source_check(row);ref=row['feature'];qlen=row['question_length'];plen=row['policy_length']
            if ref['kind']=='packed':
                packed=self._array(Path(row['_shard'])/ref['path'],ref['file']);value=packed[ref['offset']:ref['offset']+ref['length']]
            else:
                packed=self._array(ref['path'],ref['file']);value=packed[ref['offset']+np.asarray(ref['selected_indices'],dtype=np.int64)]
            if value.shape!=(qlen+plen,2048):raise ValueError('Packed feature row length differs')
            tensor=torch.from_numpy(np.array(value,dtype=np.uint16,copy=True).view(np.int16)).view(torch.bfloat16)
            features[i,:qlen]=tensor[:qlen];features[i,qmax:qmax+plen]=tensor[qlen:]
            question_mask[i,:qlen]=True;policy_mask[i,qmax:qmax+plen]=True
            positions[i,:qlen]=torch.arange(qlen);positions[i,qmax:qmax+plen]=torch.tensor(row['policy_positions'])
            qlens.append(qlen);plens.append(plen);bounds.append(row['policy_segments'])
        target=self.device if device is None else device
        result=dict(features=features,attention_mask=question_mask|policy_mask,question_mask=question_mask,policy_mask=policy_mask,
            position_ids=positions,question_lengths=torch.tensor(qlens,dtype=torch.long),policy_lengths=torch.tensor(plens,dtype=torch.long),
            policy_segments=bounds,trajectory_ids=ids,labels=torch.tensor([r['strict_em'] for r in rows],dtype=torch.float32),
            qids=[r['qid'] for r in rows],event_ids=[r['event_id'] for r in rows],feature_role=FEATURE_ROLE)
        return {key:value.to(target) if isinstance(value,torch.Tensor) else value for key,value in result.items()}

    def close(self):self._open.clear();self._checked_sources.clear()


def cpu_selfcheck():
    """One tiny two-G5 packing/reuse fixture; no GPU, LLM or persistent data."""
    import sys
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    import torch
    from .encoder import encode_rows
    module=sys.modules[__name__];real_sha=sha;cuda_before=torch.cuda.is_initialized()
    with tempfile.TemporaryDirectory(prefix='question5-cache-cpu-') as temporary:
        root=Path(temporary).resolve();source_dir=root/'source';source_dir.mkdir();model_dir=root/'model';model_dir.mkdir()
        manifest=root/'source.json';write_once(manifest,dict(fixture=True))
        qmax=4;rows={};contexts=[];policies=[];native_positions=[];cached_tokens=[];cached_policy=[];offsets=[0]
        groups=[];line_offsets={};calls=[]
        spec=dict(encoder_revision=REVISION,encoder_files_sha256={'config.json':'a'*64},feature_layer=27,
                  question_max_tokens=qmax,head_max_tokens=qmax+5,feature_role=FEATURE_ROLE)
        def words(tokens):
            tokens=(np.asarray(tokens,dtype=np.int64)%32).astype(np.uint16)
            return (np.uint16(0x3f00)+tokens[:,None]*16+(np.arange(2048,dtype=np.uint16)[None,:]%16)).astype(np.uint16)
        for group_index,split in enumerate(('train','test')):
            tids=[];labels=[]
            for sibling in range(5):
                number=group_index*5+sibling;tid='fixture-%d'%number;qlen=2+number%3;plen=3+number%3
                question=np.arange(10+number*20,10+number*20+qlen,dtype=np.int64)
                policy=np.arange(200+number*10,200+number*10+plen,dtype=np.int64)
                observation=np.array([400+number,500+number],dtype=np.int64)
                original=np.concatenate((question,policy[:2],observation,policy[2:]));first=np.concatenate((question,policy[:2]))
                segments=[];poffset=len(policies)
                for turn,(context,start,take,request) in enumerate(((first,0,2,qlen),(original,2,plen-2,qlen+4))):
                    segments.append(dict(context_offset=len(contexts),context_length=len(context),request_length=request,
                        policy_start=start,policy_length=take,position_start=request,turn_index=turn));contexts.extend(context.tolist())
                positions=[qlen,qlen+1]+list(range(qlen+4,qlen+plen+2))
                policies.extend(policy.tolist());native_positions.extend(positions)
                chosen=list(range(qlen))+positions
                mask=np.zeros(len(original),dtype=bool);mask[chosen[qlen:]]=True
                cached_tokens.extend(original.tolist());cached_policy.extend(mask.tolist());offsets.append(len(cached_tokens))
                rows[tid]=dict(trajectory_id=tid,original_trajectory_id=tid,source_id='fixture-source',event_id='event-%d'%group_index,
                    event_group_id='event-%d'%group_index,qid='question-%d'%group_index,split=split,strict_em=sibling%2,
                    question_length=qlen,policy_length=plen,policy_offset=poffset,segments=segments,
                    _fixture_chosen=chosen,_fixture_original=original.tolist(),_fixture_row=number)
                tids.append(tid);labels.append(sibling%2)
            groups.append(dict(event_id='event-%d'%group_index,event_group_id='event-%d'%group_index,
                qid='question-%d'%group_index,trajectory_ids=tids,labels=labels,split=split))
        with (source_dir/'index.jsonl').open('wb') as stream:
            for tid,row in rows.items():line_offsets[tid]=stream.tell();stream.write(canonical(row)+b'\n')
        source_arrays=dict(zip(ARRAY_NAMES,(np.asarray(contexts,dtype=np.int64),np.asarray(policies,dtype=np.int64),np.asarray(native_positions,dtype=np.int64))))
        candidate=root/'accepted-original';candidate.mkdir();candidate_words=words(cached_tokens)
        arrays={'packed_features.npy':candidate_words,'packed_input_ids.npy':np.asarray(cached_tokens,dtype=np.int32),
            'packed_position_ids.npy':np.concatenate([np.arange(offsets[i+1]-offsets[i],dtype=np.int32) for i in range(10)]),
            'packed_policy_mask.npy':np.asarray(cached_policy,dtype=bool),'offsets.npy':np.asarray(offsets,dtype=np.int64)}
        for name,array in arrays.items():np.save(candidate/name,array,allow_pickle=False)
        files={name:dict(bytes=(candidate/name).stat().st_size,sha256=real_sha(candidate/name)) for name in arrays}
        state='b'*64;write_once(candidate/'manifest.json',dict(files=files,encoder_loaded_state_sha256=state))
        composite=root/'composite.json';write_once(composite,dict(spec=dict(encoder_revision=REVISION,feature_layer=27,
            feature_dim=2048,encoder_files_sha256=spec['encoder_files_sha256']),
            shards=[dict(path=str(candidate),manifest_sha256=real_sha(candidate/'manifest.json'))],
            reuse=dict(cache_dir=str(candidate),manifest_sha256=real_sha(candidate/'manifest.json'))))
        acceptance=root/'accepted.json';write_once(acceptance,dict(accepted=True,audit=dict(accepted=True,
            stage='PREFIX_SCALE20_FINAL_METADATA_PRODUCER_PROOF_ACCEPTED',runtime_source='fcb77204f48d5cb8ec21c68df7d4a2c937b3557a',
            run_id='62407ad0ed404586998439ca1f55b902',execution_id='dd3032e313e14ce2b9c8fea996aaf721',only_owned_cuda=[],
            producer_full_sha_proofs_rechecked=True,composite_manifest_sha256=real_sha(composite),frozen_encoder_state_sha256=state)))
        original_feature_sha=real_sha(candidate/'packed_features.npy');acceptance_sha=real_sha(acceptance)
        source_proof=dict(source_cache_acceptance_path=str(acceptance),source_cache_acceptance_sha256=acceptance_sha)
        enabled={'reuse':False}
        class Dataset:
            def __init__(self,path,split):
                self.path=path;self.manifest_sha=real_sha(path);self.spec=dict(spec);self.question_max_tokens=qmax
                self.groups=[dict(g) for g in groups if g['split']==split]
                self._refs={tid:('fixture-source',line_offsets[tid],row['event_id']) for tid,row in rows.items()}
                self.sources={'fixture-source':SimpleNamespace(index_path=source_dir/'index.jsonl',scan_sha256='c'*64,proof=source_proof)}
            def source_arrays(self,_source):return source_arrays
            def get_rows(self,ids):
                result=[]
                for tid in ids:
                    row=dict(rows[tid])
                    if enabled['reuse']:
                        i=row['_fixture_row'];row.update(source_cache_accepted=True,source_cache=str(candidate),
                            source_cache_manifest_sha256=real_sha(candidate/'manifest.json'),source_cache_row=i,
                            source_cache_offset=offsets[i],source_cache_length=offsets[i+1]-offsets[i],
                            selected_feature_indices=row['_fixture_chosen'],source_cache_acceptance_sha256=acceptance_sha)
                    result.append(row)
                return result
            def close(self):pass
        def capture(ids,mask,positions):
            calls.append((len(ids),ids.shape[1]));assert len(ids)*ids.shape[1]<=32
            result=np.stack([words(row.tolist()) for row in ids])
            return torch.from_numpy(result.view(np.int16)).view(torch.bfloat16)
        class Encoder:
            def __init__(self,_dataset):self.model=torch.nn.Linear(1,1).requires_grad_(False).eval()
            def encode_rows(self,requested,array_provider):
                return encode_rows(requested,array_provider,qmax,capture,device='cpu',micro_batch_size=128,token_budget=32)
            def close(self):pass
        fixed=Path('/data/shaoshitong/Search-G1-a800-fresh200-0850db2977bd4801b9ced300b0ab387f/features/composite-manifest.json')
        def redirect(path):return composite if Path(path)==fixed else Path(path).resolve()
        with patch.object(module,'remote',side_effect=redirect),patch.object(module,'sha',side_effect=lambda path:real_sha(redirect(path))),\
             patch.object(module,'CompositePrefix5Dataset',Dataset),patch.object(module,'process_birth',return_value=dict(pid=os.getpid(),start_ticks=1,sid=os.getpid(),uid=os.getuid())):
            def verify(path):
                for split in ('train','test'):
                    reader=PackedFeatureReader(path/'manifest.json',split=split,device='cpu')
                    assert len(reader.groups)==1 and len(reader.groups[0]['trajectory_ids'])==5
                    requested=list(reversed(reader.ids));batch=reader.get_batch(requested)
                    assert batch['trajectory_ids']==requested and batch['features'].shape==(5,qmax+5,2048)
                    for index,tid in enumerate(requested):
                        row=rows[tid];qlen=row['question_length'];plen=row['policy_length'];chosen=row['_fixture_chosen'];original=row['_fixture_original']
                        wanted=words(np.asarray(original)[chosen]);actual=batch['features'][index][batch['attention_mask'][index]].view(torch.int16).numpy().view(np.uint16)
                        assert np.array_equal(actual,wanted)
                        assert batch['labels'][index].item()==row['strict_em'] and batch['question_lengths'][index].item()==qlen
                        assert batch['question_mask'][index].tolist()==[True]*qlen+[False]*(qmax+5-qlen)
                        assert batch['policy_mask'][index].tolist()==[False]*qmax+[True]*plen+[False]*(5-plen)
                        expected_positions=list(range(qlen))+[qlen,qlen+1]+list(range(qlen+4,qlen+plen+2))
                        assert batch['position_ids'][index][batch['attention_mask'][index]].tolist()==expected_positions
                        assert batch['policy_segments'][index]==[{k:s[k] for k in ('turn_index','policy_start','policy_length','position_start')} for s in row['segments']]
                        assert not batch['features'][index,qlen:qmax].any() and not batch['position_ids'][index,qlen:qmax].any()
                        assert not batch['features'][index,qmax+plen:].any() and not batch['position_ids'][index,qmax+plen:].any()
                    reader.close()
            encoded=root/'encoded';built=build_cache(manifest,encoded,model_dir,groups_per_shard=1,
                encoder_factory=Encoder,device='cpu',micro_batch_size=128,token_budget=32)
            finalized=finalize_cache(encoded,world_size=1);verify(encoded)
            assert built['newly_encoded_trajectories']==10 and finalized['reused_trajectories']==0 and calls
            enabled['reuse']=True;reused=root/'reused'
            def forbidden(_dataset):raise AssertionError('Reuse-only cache must not create an encoder')
            built=build_cache(manifest,reused,model_dir,groups_per_shard=1,encoder_factory=forbidden,device='cpu')
            finalized=finalize_cache(reused,world_size=1);verify(reused)
            assert built['reused_trajectories']==10 and finalized['newly_encoded_trajectories']==0
            assert not list(reused.rglob('packed_features.npy')) and real_sha(candidate/'packed_features.npy')==original_feature_sha
            assert torch.cuda.is_initialized()==cuda_before
    return dict(stage='QUESTION5_CACHE_TINY_CPU_FIXTURE_PASS',complete_event_groups=2,trajectories=10,
        packed_bf16_bits_exact=True,ordered_ids_masks_positions_labels_segments_exact=True,
        question_tail_padding_zero=True,reuse_only_encoder_calls=0,reused_feature_files_copied=0,
        real_encoder_batch_assembly_token_budget_checked=True,model_loaded=False,cuda_initialized=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('build-cache','finalize-cache','cpu-selfcheck'),required=True)
    parser.add_argument('--input-manifest',type=Path);parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--model-dir',type=Path);parser.add_argument('--worker-index','--shard-index',type=int,default=0)
    parser.add_argument('--world-size',type=int,default=1);parser.add_argument('--groups-per-shard',type=int,default=256)
    parser.add_argument('--micro-batch-size',type=int,default=32);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--token-budget',type=int,default=65536)
    parser.add_argument('--physical-index',type=int)
    args=parser.parse_args()
    if args.action=='cpu-selfcheck':print(json.dumps(cpu_selfcheck(),allow_nan=False),flush=True);return
    if args.output_dir is None:parser.error('--output-dir is required')
    if args.action=='finalize-cache':result=finalize_cache(args.output_dir,world_size=args.world_size)
    else:
        if args.input_manifest is None or args.model_dir is None:parser.error('build-cache requires --input-manifest and --model-dir')
        if args.physical_index is None:parser.error('build-cache requires --physical-index for lazy owned GPU admission')
        result=build_cache(args.input_manifest,args.output_dir,args.model_dir,worker_index=args.worker_index,world_size=args.world_size,
            groups_per_shard=args.groups_per_shard,micro_batch_size=args.micro_batch_size,device=args.device,
            token_budget=args.token_budget,
            encoder_factory=admitted_encoder_factory(args.model_dir,args.physical_index,args.device,args.micro_batch_size,args.token_budget))
    print(json.dumps(dict(stage=result['stage'],manifest_path=str(args.output_dir/'manifest.json') if args.action=='finalize-cache' else None,
        reused_trajectories=result['reused_trajectories'],newly_encoded_trajectories=result['newly_encoded_trajectories']),allow_nan=False),flush=True)


if __name__=='__main__':main()
