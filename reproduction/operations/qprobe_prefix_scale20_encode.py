"""Independent BF16 prefix encoder workers for a sealed scale20 teacher dataset.

One fresh process/GPU per replica. Encode only new compact shards; reference the
existing 25600 train +2560 fixed-test features without reencoding or copying.
Full native36-layer AutoModel forward, layer27 hook, no probe/actor training.
CPU composite-manifest merge and actual resource-release audit are separate.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

sys.dont_write_bytecode=True
RUNTIME='fcb77204f48d5cb8ec21c68df7d4a2c937b3557a'
REVISION='aa8e72537993ba99e69dfaafa59ed015b17504d1'
FEATURE_HELPER_SHA='4a8adfb2adeef5de6bd80a4abe28172a6a989b56f76ab0b329b04af50567d95f'
REPRESENTATION='actual_prompt_policy_prefix5_alltokens_layer27_v1'
INPUT_FILES=('packed_input_ids.npy','packed_position_ids.npy','packed_policy_mask.npy','offsets.npy','lengths.npy')
CONTRACT=dict(schema_version=1,role='prefix_scale20_parallel_frozen_encoder_worker',
    runtime_commit=RUNTIME,encoder_revision=REVISION,feature_layer=27,full_native_decoder_layers=36,
    feature_dim=2048,encoder_compute_dtype='bfloat16',output_feature_dtype='bfloat16_uint16_words',
    packed_input_ids_dtype='int32',packed_position_ids_dtype='int32',offsets_dtype='int64',lengths_dtype='int32',
    canonical_prefix_hash_integer_dtype='little_endian_int64',input_arrays_hardlinked_without_int64_copy=True,
    default_micro_batch_size=128,default_token_budget=65536,maximum_micro_batch_size=256,
    batching='stable length buckets; batch_size*batch_max_length <= token_budget; no truncation',
    shard_assignment='compact_shard_ordinal modulo world_size equals worker shard_index',
    train_questions=102400,train_trajectories=512000,new_questions=97280,new_trajectories=486400,
    reused_training_trajectories=25600,fixed_test_questions=512,fixed_test_trajectories=2560,
    reused_features_reencoded=False,reused_features_copied=False,
    teacher_regenerated=False,probe_or_actor_training=False,optimizer_created=False,
    actual_positions_from_compact_cache=True,layer27_features_detached=True,
    output='one independent packed-feature directory per compact shard; original row order preserved',
    full_cache_merge='CPU composite references; no 300GB feature consolidation required',
    no_automatic_retry=True,actual_resource_release_audit_required=True)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024**2),b''):h.update(data)
    return h.hexdigest()


def read(path,limit=16*1024**2):
    path=Path(path)
    if path.resolve()!=path or path.is_symlink() or not path.is_file() or path.stat().st_uid!=os.getuid() or path.stat().st_size>limit:
        raise ValueError('SCALE20_ENCODER_UNSAFE_METADATA_PATH')
    return json.loads(path.read_bytes())


def remote(value):
    path=Path(value)
    if not path.is_absolute() or path.resolve()!=path or not str(path).startswith(('/data/','/mnt/','/backup/')):
        raise ValueError('SCALE20_ENCODER_REQUIRES_REAL_REMOTE_STORAGE')
    return path


def once(path,value):
    raw=canonical(value)+b'\n';fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def progress(output,**value):
    row=dict(observed_unix=time.time(),**value)
    with (output/'progress.jsonl').open('ab') as stream:stream.write(canonical(row)+b'\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(row,allow_nan=False),flush=True)


def load_runtime(path):
    path=remote(path)
    if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=RUNTIME or subprocess.check_output(['git','-C',str(path),'status','--porcelain']).strip():
        raise ValueError('SCALE20_ENCODER_REQUIRES_IMMUTABLE_FCB_RUNTIME')
    sys.path.insert(0,str(path))
    from reproduction.a800_common import qprobe_token_warmup as helper
    from reproduction.a800_common.deploy import published_source
    if Path(helper.__file__).resolve()!=path/'reproduction/a800_common/qprobe_token_warmup.py':
        raise ValueError('SCALE20_ENCODER_RUNTIME_IMPORT_SHADOWED')
    mapping=published_source(path,RUNTIME)
    if any(sha(path/name)!=value for name,value in mapping.items()):raise ValueError('SCALE20_ENCODER_RUNTIME_SOURCE_CHANGED')
    old=Path(__file__).with_name('qprobe_prefix_correctness_features.py')
    if sha(old)!=FEATURE_HELPER_SHA:raise ValueError('SCALE20_ENCODER_FROZEN_PROOF_HELPER_SOURCE_CHANGED')
    spec=importlib.util.spec_from_file_location('published_prefix_encoder_frozen_state',old)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return helper,module.frozen_encoder_state,hashlib.sha256(canonical(mapping)).hexdigest()


def load_dataset(path,expected_sha=None):
    path=remote(path);manifest_path=path/'compact-manifest.json';manifest=read(manifest_path)
    digest=sha(manifest_path)
    if expected_sha is not None and digest!=expected_sha:raise ValueError('SCALE20_ENCODER_COMPACT_MANIFEST_SHA_CHANGED')
    expected=dict(stage='PREFIX_TEACHER_SCALE20_COMPACT_COMPLETE',runtime_commit=RUNTIME,
        train_questions=102400,train_trajectories=512000,new_questions=97280,new_trajectories=486400)
    if any(manifest.get(key)!=value for key,value in expected.items()):raise ValueError('SCALE20_ENCODER_DATASET_IDENTITY_OR_COUNT_CHANGED')
    spec=manifest['spec']
    required=dict(representation=REPRESENTATION,encoder_revision=REVISION,feature_layer=27,feature_dim=2048,
        encoder_compute_dtype='bfloat16',feature_dtype='bfloat16')
    if any(spec.get(key)!=value for key,value in required.items()) or not 0<spec['global_max_tokens']<=32768:
        raise ValueError('SCALE20_ENCODER_FROZEN_OFFICIAL_FEATURE_SPEC_CHANGED')
    reuse=manifest['reuse']
    if reuse['train_trajectories']!=25600 or reuse['test_trajectories']!=2560:
        raise ValueError('SCALE20_ENCODER_MUST_REUSE_ORIGINAL_TRAIN_AND_FIXED_TEST')
    old=remote(reuse['cache_dir'])
    if sha(old/'manifest.json')!=reuse['manifest_sha256']:raise ValueError('SCALE20_ENCODER_REUSED_CACHE_MANIFEST_CHANGED')
    shards=manifest['shards']
    if len(shards)!=190 or {row['ordinal'] for row in shards}!=set(range(10,200)) or sum(row['count'] for row in shards)!=486400:
        raise ValueError('SCALE20_ENCODER_COMPACT_SHARD_PARTITION_CHANGED')
    for row in shards:
        if type(row['ordinal']) is not int or type(row['count']) is not int or row['count']!=2560:
            raise ValueError('SCALE20_ENCODER_INVALID_COMPACT_SHARD')
    return path,manifest,digest


def load_compact(root,entry,spec):
    path=Path(entry['path']);path=path if path.is_absolute() else root/path
    path=remote(path)
    if not path.is_relative_to(root):raise ValueError('SCALE20_ENCODER_COMPACT_SHARD_OUTSIDE_DATASET')
    manifest=read(path/'manifest.json')
    if sha(path/'manifest.json')!=entry['manifest_sha256'] or manifest['ordinal']!=entry['ordinal'] or manifest['count']!=entry['count']:
        raise ValueError('SCALE20_ENCODER_COMPACT_SHARD_MANIFEST_CHANGED')
    if set(manifest['files'])!=set(INPUT_FILES)|{'index.jsonl'}:raise ValueError('SCALE20_ENCODER_COMPACT_FILES_CHANGED')
    for name,item in manifest['files'].items():
        target=path/name
        if target.resolve()!=target or target.is_symlink() or target.stat().st_uid!=os.getuid() or target.stat().st_size!=item['bytes'] or sha(target)!=item['sha256']:
            raise ValueError('SCALE20_ENCODER_COMPACT_FILE_SHA_CHANGED: '+name)
    arrays={name:np.load(path/name,allow_pickle=False,mmap_mode='r') for name in INPUT_FILES}
    offsets=arrays['offsets.npy'];lengths=arrays['lengths.npy'];tokens=arrays['packed_input_ids.npy'];positions=arrays['packed_position_ids.npy'];policy=arrays['packed_policy_mask.npy']
    rows=[json.loads(line) for line in (path/'index.jsonl').read_text().splitlines()]
    if (offsets.dtype!=np.int64 or lengths.dtype!=np.int32 or tokens.dtype!=np.int32 or positions.dtype!=np.int32 or policy.dtype!=np.bool_ or
            offsets.shape!=(len(rows)+1,) or lengths.shape!=(len(rows),) or tokens.ndim!=positions.ndim or tokens.ndim!=policy.ndim or tokens.ndim!=1 or
            positions.shape!=tokens.shape or policy.shape!=tokens.shape or offsets[0]!=0 or offsets[-1]!=len(tokens) or
            not np.array_equal(np.diff(offsets),lengths) or len(rows)!=entry['count'] or int(lengths.min())<1 or int(lengths.max())>spec['global_max_tokens']):
        raise ValueError('SCALE20_ENCODER_COMPACT_ARRAY_SHAPE_CHANGED')
    ids=set()
    for index,row in enumerate(rows):
        lo,hi=map(int,offsets[index:index+2]);p=positions[lo:hi];k=int(policy[lo:hi].sum())
        if (row['row']!=index or row['offset']!=lo or row['length']!=hi-lo or row['trajectory_id'] in ids or row.get('split')!='train' or
                type(row['strict_em']) is not int or row['strict_em'] not in (0,1) or not 1<=k<=5 or
                row['policy_prefix_tokens']!=k or np.any(p<0) or np.any(p>=32768) or np.any(np.diff(p)<=0)):
            raise ValueError('SCALE20_ENCODER_COMPACT_TRAJECTORY_OR_ACTUAL_POSITION_CHANGED')
        token_bytes=tokens[lo:hi].astype('<i8').tobytes();position_bytes=p.astype('<i8').tobytes();policy_bytes=policy[lo:hi].tobytes()
        for value,key in ((token_bytes,'input_ids_sha256'),(position_bytes,'position_ids_sha256'),(policy_bytes,'policy_mask_sha256')):
            if hashlib.sha256(value).hexdigest()!=row[key]:raise ValueError('SCALE20_ENCODER_COMPACT_INT64_ROUNDTRIP_HASH_CHANGED')
        actual=hashlib.sha256(canonical(dict(version=1,length=hi-lo,policy_tokens=k))+token_bytes+position_bytes+policy_bytes).hexdigest()
        if actual!=row['prefix_sha256']:raise ValueError('SCALE20_ENCODER_ACTUAL_PREFIX_TOKEN_HASH_CHANGED')
        ids.add(row['trajectory_id'])
    return path,manifest,arrays,rows


def batches(rows,micro_batch_size,token_budget):
    order=sorted(range(len(rows)),key=lambda index:(rows[index]['length'],index));chunk=[];width=0
    for index in order:
        length=rows[index]['length']
        if length>token_budget:raise ValueError('SCALE20_ENCODER_ONE_PREFIX_EXCEEDS_TOKEN_BUDGET_NO_TRUNCATION')
        if chunk and (len(chunk)>=micro_batch_size or max(width,length)*(len(chunk)+1)>token_budget):
            yield chunk;chunk=[];width=0
        chunk.append(index);width=max(width,length)
    if chunk:yield chunk


def encode_shard(model,arrays,rows,path,entry,worker,args,manifest_sha,encoder_before,captured):
    import torch
    target=worker/('shard-%06d'%entry['ordinal']);target.mkdir(mode=0o700,exist_ok=False)
    offsets=arrays['offsets.npy'];total=int(offsets[-1]);required=total*2048*2+1024**3
    if __import__('shutil').disk_usage(worker).free<required:raise ValueError('SCALE20_ENCODER_REMOTE_DISK_ADMISSION_FAILED')
    for name in INPUT_FILES:os.link(path/name,target/name)
    packed=np.lib.format.open_memmap(target/'packed_features.npy',mode='w+',dtype=np.uint16,shape=(total,2048))
    output_rows=[None]*len(rows);done=tokens_written=0;started=time.monotonic()
    with torch.inference_mode():
        for indices in batches(rows,args.micro_batch_size,args.token_budget):
            width=max(rows[index]['length'] for index in indices);shape=(len(indices),width)
            ids=np.full(shape,args.pad_token_id,dtype=np.int64);pos=np.zeros(shape,dtype=np.int64);mask=np.zeros(shape,dtype=np.int64)
            for local,index in enumerate(indices):
                lo,hi=map(int,offsets[index:index+2]);length=hi-lo
                ids[local,:length]=arrays['packed_input_ids.npy'][lo:hi];pos[local,:length]=arrays['packed_position_ids.npy'][lo:hi];mask[local,:length]=1
            captured.clear();model(input_ids=torch.from_numpy(ids).to('cuda:0'),attention_mask=torch.from_numpy(mask).to('cuda:0'),
                position_ids=torch.from_numpy(pos).to('cuda:0'),use_cache=False,output_hidden_states=False,return_dict=True)
            if len(captured)!=1 or captured[0].shape!=(len(indices),width,2048):raise ValueError('SCALE20_ENCODER_LAYER27_HOOK_SHAPE_CHANGED')
            cpu=captured.pop().detach().cpu().contiguous()
            if cpu.dtype!=torch.bfloat16:raise ValueError('SCALE20_ENCODER_LAYER27_MUST_BE_NATIVE_BF16')
            for local,index in enumerate(indices):
                row=rows[index];value=cpu[local,:row['length']].contiguous()
                if not torch.isfinite(value).all():raise ValueError('SCALE20_ENCODER_NONFINITE_REAL_TOKEN_FEATURE')
                data=value.view(torch.int16).numpy().view(np.uint16);lo,hi=map(int,offsets[index:index+2]);packed[lo:hi]=data
                output_rows[index]=dict(row,feature_sha256=hashlib.sha256(data.tobytes()).hexdigest())
                tokens_written+=hi-lo
            done+=len(indices);packed.flush()
            progress(worker,stage='NEW_PREFIX_SHARD_ENCODING',compact_shard_ordinal=entry['ordinal'],completed_trajectories=done,
                shard_trajectories=len(rows),real_tokens_written=tokens_written,actual_bf16_bytes_written=tokens_written*4096,
                feature_file_size_is_preallocated=True,micro_batch_size=len(indices),batch_max_tokens=width,
                padded_tokens_this_batch=len(indices)*width,token_budget=args.token_budget,elapsed_seconds=time.monotonic()-started)
            del cpu
    packed.flush();del packed
    with (target/'index.jsonl').open('xb') as stream:
        for row in output_rows:stream.write(canonical(row)+b'\n')
        stream.flush();os.fsync(stream.fileno())
    names=INPUT_FILES+('packed_features.npy','index.jsonl')
    files={name:dict(bytes=(target/name).stat().st_size,sha256=sha(target/name)) for name in names}
    result=dict(schema_version=1,stage='PREFIX_SCALE20_NEW_SHARD_ENCODED',ordinal=entry['ordinal'],count=len(rows),total_tokens=total,
        representation=REPRESENTATION,feature_dim=2048,feature_dtype='bfloat16',files=files,
        packed_input_ids_dtype='int32',packed_position_ids_dtype='int32',offsets_dtype='int64',lengths_dtype='int32',
        input_arrays_hardlinked_without_int64_copy=True,canonical_prefix_hash_integer_dtype='little_endian_int64',
        compact_shard_path=str(path),compact_shard_manifest_sha256=entry['manifest_sha256'],compact_dataset_manifest_sha256=manifest_sha,
        encoder_loaded_state_sha256=encoder_before['loaded_parameter_and_buffer_sha256'],encoder_final_equality_proof_required=True,
        full_native_decoder_layers=36,feature_layer=27,no_future_tokens=True,actual_position_ids=True,
        actor_updated=False,probe_trained=False,checked_unix=time.time())
    once(target/'manifest.json',result)
    return dict(ordinal=entry['ordinal'],path=str(target),count=len(rows),total_tokens=total,
        manifest_sha256=sha(target/'manifest.json'),packed_feature_bytes=files['packed_features.npy']['bytes'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-repo',type=Path);parser.add_argument('--input-dir',type=Path);parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--compact-manifest-sha256');parser.add_argument('--shard-index',type=int);parser.add_argument('--world-size',type=int,default=4)
    parser.add_argument('--physical-index',type=int);parser.add_argument('--micro-batch-size',type=int,default=128)
    parser.add_argument('--token-budget',type=int,default=65536);parser.add_argument('--print-contract',action='store_true');args=parser.parse_args()
    if args.print_contract:print(json.dumps(CONTRACT,sort_keys=True));return 0
    if any(value is None for value in (args.runtime_repo,args.input_dir,args.output_dir,args.shard_index,args.physical_index)):
        parser.error('--runtime-repo/--input-dir/--output-dir/--shard-index/--physical-index required')
    if not 1<=args.world_size<=4 or not 0<=args.shard_index<args.world_size or not 1<=args.micro_batch_size<=256 or args.token_budget<=0:
        raise ValueError('SCALE20_ENCODER_INVALID_WORKER_OR_TOKEN_BUDGET')
    helper,frozen_state,runtime_map=load_runtime(args.runtime_repo)
    root,manifest,manifest_sha=load_dataset(args.input_dir,args.compact_manifest_sha256)
    assigned=sorted((row for row in manifest['shards'] if row['ordinal']%args.world_size==args.shard_index),key=lambda row:row['ordinal'])
    if not assigned:raise ValueError('SCALE20_ENCODER_EMPTY_WORKER_ASSIGNMENT')
    spec=manifest['spec'];model_dir=remote(spec['encoder_model_dir'])
    if model_dir!=Path('/data/search-g1/assets/qwen2.5-3b-instruct'):raise ValueError('SCALE20_ENCODER_OFFICIAL_MODEL_PATH_CHANGED')
    for name,digest in spec['encoder_files_sha256'].items():
        if sha(model_dir/name)!=digest:raise ValueError('SCALE20_ENCODER_OFFICIAL_MODEL_FILE_SHA_CHANGED: '+name)
    args.pad_token_id=spec['pad_token_id'];output=remote(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    worker=output/('worker-%02d'%args.shard_index);worker.mkdir(mode=0o700,exist_ok=False)
    once(worker/'worker-claim.json',dict(contract=CONTRACT,source_sha256=sha(__file__),runtime_source_map_sha256=runtime_map,
        compact_manifest_sha256=manifest_sha,shard_index=args.shard_index,world_size=args.world_size,
        assigned_shard_ordinals=[row['ordinal'] for row in assigned],assigned_trajectories=sum(row['count'] for row in assigned),
        reused_cache=manifest['reuse'],claimed_unix=time.time(),no_automatic_retry=True))
    lock=None;handle=None
    try:
        values=[entry.split(b'=',1)[1].decode() for entry in Path('/proc/self/environ').read_bytes().split(b'\0') if entry.startswith(b'CUDA_VISIBLE_DEVICES=')]
        if len(values)!=1 or not re.fullmatch(r'GPU-[0-9a-fA-F-]+',values[0]) or os.environ.get('CUDA_VISIBLE_DEVICES')!=values[0]:
            raise ValueError('SCALE20_ENCODER_REQUIRES_EXACT_SINGLE_GPU_UUID_AT_BIRTH')
        inventory=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=15))))
        actual=[row[1].strip() for row in inventory if int(row[0])==args.physical_index]
        if actual!=values:raise ValueError('SCALE20_ENCODER_BORN_UUID_PHYSICAL_MAPPING_CHANGED')
        lock,gpu=helper.gpu_admission(args.physical_index)
        if gpu['actual_cuda_uuids']!=values or gpu['physical_gpu_uuid']!=values[0] or os.environ['CUDA_VISIBLE_DEVICES']!=values[0]:
            raise ValueError('SCALE20_ENCODER_GPU_ADMISSION_CHANGED_BORN_VISIBILITY')
        import torch
        from transformers import AutoModel
        from reproduction.a800.owned_processes import identity
        once(worker/'worker-birth.json',dict(producer=identity(os.getpid()),hostname=socket.gethostname(),gpu=gpu,birth_cuda_visible_devices=values[0]))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        model=AutoModel.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False,low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0')
        model.requires_grad_(False);model.eval()
        if len(model.layers)!=36 or any(p.requires_grad for p in model.parameters()):raise ValueError('SCALE20_ENCODER_MUST_KEEP_ALL36_OFFICIAL_FROZEN_LAYERS')
        before=frozen_state(model);captured=[]
        def hook(_module,_inputs,value):
            value=value[0] if isinstance(value,tuple) else value;captured.append(value.detach())
        handle=model.layers[27].register_forward_hook(hook)
        once(worker/'encoder-before.json',before);results=[]
        progress(worker,stage='FROZEN_OFFICIAL36_LAYER_ENCODER_LOADED',assigned_trajectories=sum(row['count'] for row in assigned),micro_batch_size=args.micro_batch_size,token_budget=args.token_budget)
        for entry in assigned:
            path,compact,arrays,rows=load_compact(root,entry,spec)
            receipt=encode_shard(model,arrays,rows,path,entry,worker,args,manifest_sha,before,captured);results.append(receipt)
            progress(worker,stage='NEW_PREFIX_SHARD_FULL_SHA_COMPLETE',**receipt,total_worker_completed_trajectories=sum(row['count'] for row in results))
        handle.remove();handle=None;after=frozen_state(model)
        if before!=after:raise ValueError('SCALE20_ENCODER_FROZEN_PARAMETERS_OR_BUFFERS_CHANGED')
        once(worker/'frozen-encoder-proof.json',dict(before=before,after=after,exact_equal=True,full_native_decoder_layers=36,
            layer27_hook_only=True,optimizer_created=False,encoder_updated=False,labels_passed_to_encoder=False))
        once(worker/'worker-complete.json',dict(stage='PREFIX_SCALE20_ENCODER_WORKER_COMPLETE',contract=CONTRACT,
            compact_manifest_sha256=manifest_sha,runtime_commit=RUNTIME,worker_index=args.shard_index,world_size=args.world_size,
            completed_trajectories=sum(row['count'] for row in results),shards=results,reused_cache=manifest['reuse'],
            reused_features_reencoded=False,reused_features_copied=False,actor_updated=False,probe_trained=False,
            frozen_encoder_proof_sha256=sha(worker/'frozen-encoder-proof.json'),actual_resource_release_audit_required=True,checked_unix=time.time()))
        return 0
    except BaseException as failure:
        once(worker/'worker-failure.json',dict(failure_type=type(failure).__name__,frames=[dict(file=Path(frame.filename).name,line=frame.lineno,function=frame.name)
            for frame in traceback.extract_tb(failure.__traceback__)[-8:]],partial_shards_preserved=True,no_automatic_retry=True,checked_unix=time.time()))
        raise
    finally:
        if handle is not None:handle.remove()
        if lock is not None:lock.close()


if __name__=='__main__':raise SystemExit(main())
