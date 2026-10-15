"""Produce the full canonical all-token bank, reusing the qualified 5120 rows.

Run --prepare-only first in a fresh remote output directory. A later invocation
without that flag encodes only missing rows on one admitted physical GPU. This
process never generates teacher trajectories, fits a head, or starts an actor.
Its completion receipt requires a separate actual process/CUDA exit audit before
another consumer starts; publication of a receipt does not prove process exit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import time
import traceback

sys.dont_write_bytecode=True
SOURCE_RUN='/data/shaoshitong/Search-G1-a800-fresh200-4a017644c8054de38b05314879527303'
GLOBAL_OUTPUT='/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-a799f7aff3e94045b64791655b878c31/warmup'
COMPLETED_OUTPUT='/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-494eafd18b004cb980121e46b188db44/warmup'
READY_SHA='496e30af7c9c4afb164507f90d14e3f6bfdc7e15e1f6c6f3a4a7c0d5f4198f5f'
SPEC_SHA='fcc9d61db156ddbb4e96dcd6c9437797b978dbbfc9774145a28e0c761de1a9e7'
MANIFEST_SHA='a4f28ede7537d5029076c85747427262d6a275ebeae870cdf6279551972919fb'
GLOBAL_SHA={
    'global-scan.json':'9d700efaa28e3f0debdaef3cfa070753d244ac838e2f0bd3c6b0f4847aa933bc',
    'global-token-bindings.jsonl':'ccb5304332d422f9a3ce6a5961a3b04e0e9b910b1df772134ce7befd43e086a7',
    'global-token-lengths.npy':'3c28b70e08cc2743346a30b293598cc1531ccb789fdeae03f9b3426c3da7dc8e'}
FEATURE_SOURCE_SHA='408ac3c710cbcbb3684c93bdf57fd170c54266038fb6e0aced581accf1e5769b'
EXPECTED_ROWS=169028
EXPECTED_REUSED=5120


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024**2),b''):h.update(part)
    return h.hexdigest()


def once(path,value):
    """An existing attempt is always refused, including failed attempts."""
    path=Path(path);raw=canonical(value)+b'\n'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def canonical_records(rows,bindings,reused_index,question_hash):
    """Validate the exact canonical order before allocating the packed bank."""
    if len(rows)!=len(bindings):raise ValueError('FULL_BANK_CANONICAL_BINDING_COUNT')
    seen=set();records=[]
    for (question,qid),binding in zip(rows,bindings):
        if qid in seen or binding['qid']!=qid or binding['question_hash']!=question_hash(question):
            raise ValueError('FULL_BANK_CANONICAL_ORDER_OR_QUESTION_BINDING')
        if type(binding['length']) is not int or binding['length']<=0:
            raise ValueError('FULL_BANK_INVALID_LENGTH')
        if not re.fullmatch('[0-9a-f]{64}',binding['token_ids_sha256']):
            raise ValueError('FULL_BANK_TOKEN_HASH_MISSING')
        if qid in reused_index:
            prior=reused_index[qid]
            if any(prior[key]!=binding[key] for key in ('qid','question_hash','length','token_ids_sha256')):
                raise ValueError('FULL_BANK_REUSED_ROW_BINDING_CHANGED')
        seen.add(qid);records.append(dict(binding))
    if not set(reused_index)<=seen:raise ValueError('FULL_BANK_REUSED_ID_NOT_CANONICAL')
    return records


def emit_bank(rows,records,reused,writer,extractor,*,batch_size,progress=None):
    """Provider/extractor boundary also permits a tiny, model-free CPU test."""
    if batch_size<=0 or len(rows)!=len(records):raise ValueError('FULL_BANK_BATCH_CONTRACT')
    reused_count=encoded_count=0
    for start in range(0,len(rows),batch_size):
        chunk=rows[start:start+batch_size];ids=[qid for _,qid in chunk]
        existing=[qid for qid in ids if qid in reused.index]
        missing=[(question,qid) for question,qid in chunk if qid not in reused.index]
        prior=list(reused.get_features(existing)) if existing else []
        if len(prior)!=len(existing):raise ValueError('FULL_BANK_REUSED_PROVIDER_COUNT')
        values=dict(zip(existing,prior))
        if missing:
            fresh=list(extractor([question for question,_ in missing],[records[start+i] for i,(_,qid) in enumerate(chunk) if qid not in reused.index]))
            if len(fresh)!=len(missing):raise ValueError('FULL_BANK_EXTRACTOR_COUNT')
            values.update(zip([qid for _,qid in missing],fresh))
        if set(values)!=set(ids):raise ValueError('FULL_BANK_PROVIDER_COUNT')
        writer.append(ids,[values[qid] for qid in ids])
        reused_count+=len(existing);encoded_count+=len(missing)
        if progress:progress(dict(completed_rows=start+len(chunk),total_rows=len(rows),reused_rows=reused_count,newly_encoded_rows=encoded_count))
    manifest=writer.finish()
    return manifest,dict(rows=len(rows),reused_rows=reused_count,newly_encoded_rows=encoded_count)


def output_path(value):
    path=Path(value)
    assert path.resolve()==path and re.fullmatch(r'/data/shaoshitong/Search-G1-qprobe-fulltoken-feature-bank-[0-9a-f]{32}',str(path))
    return path


def task_config(args):
    return dict(source_commit=args.source_commit,source_run_base=SOURCE_RUN,global_token_output=GLOBAL_OUTPUT,
        completed_warmup_output=COMPLETED_OUTPUT,token_feature_spec_sha256=SPEC_SHA,encode_batch_size=args.encode_batch_size,
        expected_rows=EXPECTED_ROWS,reused_rows=EXPECTED_REUSED,newly_encoded_rows=EXPECTED_ROWS-EXPECTED_REUSED)


def progress(output,value):
    event=dict(value,observed_unix=time.time());temporary=output/'progress.partial.json'
    with temporary.open('w') as stream:json.dump(event,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,output/'progress.json')
    print(json.dumps(event,allow_nan=False),flush=True)


def qualify(args,mapping,output):
    """CPU-only full source/model/data/global metadata and reused-bank proof."""
    import numpy as np
    from reproduction.a800_common.qprobe_token_warmup import assets,read
    from search_g1.qprobe_features import question_text_sha
    from search_g1.qprobe_token_features import TokenFeatureCache
    assert mapping['search_g1/qprobe_token_features.py']==FEATURE_SOURCE_SHA
    args.source_run_base=SOURCE_RUN
    source,run,rows,model_hashes,_,_=assets(args,mapping)
    assert len(rows)==EXPECTED_ROWS
    global_root=Path(GLOBAL_OUTPUT);completed=Path(COMPLETED_OUTPUT)
    assert sha(completed/'ready.json')==READY_SHA
    ready=read(completed/'ready.json');resume=ready['parent_feature_resume']
    assert ready['head_optimizer_updates']==ready['head_adam_steps']==200 and ready['epoch_records']==11 and ready['minibatch_records']==200
    assert ready['token_feature_spec_sha256']==SPEC_SHA and ready['token_feature_manifest_sha256']==MANIFEST_SHA
    assert ready['headfit_only'] is True and ready['formal_training_started'] is False
    assert resume['source_output']==GLOBAL_OUTPUT and resume['feature_cache_dir']==str(global_root/'token-features')
    assert resume['token_features_full_sha_verified'] is True and resume['encoder_reencoded'] is False
    for name,expected in GLOBAL_SHA.items():
        assert sha(global_root/name)==expected==resume['global_metadata_files'][name]['sha256']
    scan=read(global_root/'global-scan.json');assert scan['global_train_rows']==EXPECTED_ROWS and scan['global_max_tokens']==244 and scan['total_tokens']==18367021
    lengths=np.load(global_root/'global-token-lengths.npy',allow_pickle=False)
    assert lengths.dtype==np.int32 and lengths.shape==(EXPECTED_ROWS,) and int(lengths.max())==244 and int(lengths.sum())==scan['total_tokens']
    bindings=[json.loads(line) for line in (global_root/'global-token-bindings.jsonl').read_text().splitlines()]
    cache=TokenFeatureCache(global_root/'token-features',expected_spec_sha=SPEC_SHA)
    assert sha(cache.path/'manifest.json')==MANIFEST_SHA and cache.manifest['count']==EXPECTED_REUSED
    cache.verify_all()
    assert cache.manifest['files']==resume['feature_files'] and cache.spec['feature_dtype']=='bfloat16'
    assert cache.spec['train_parquet_sha256']==run['input_sha256']['train.parquet'] and cache.spec['encoder_files_sha256']==hashlib.sha256(canonical(model_hashes)).hexdigest()
    assert cache.spec['global_scan_sha256']==GLOBAL_SHA['global-scan.json'] and cache.spec['global_train_rows']==EXPECTED_ROWS
    records=canonical_records(rows,bindings,cache.index,question_text_sha)
    assert all(record['length']==int(length) for record,length in zip(records,lengths))
    ordered=hashlib.sha256()
    for record in records:ordered.update(json.dumps([record['qid'],record['question_hash'],record['length'],record['token_ids_sha256']],ensure_ascii=False,separators=(',',':')).encode()+b'\n')
    assert ordered.hexdigest()==scan['ordered_bindings_sha256']
    # Validate each reused row's actual finite tensor/hash in bounded CPU batches.
    for start in range(0,len(cache.ids),64):
        value=cache.get_batch(cache.ids[start:start+64],device='cpu');del value
    packed_bytes=int(lengths.sum())*2048*2
    index_allowance=sum(len(canonical(record))+160 for record in records)
    required=packed_bytes+index_allowance+(EXPECTED_ROWS*12+1024)+2*1024**3
    available=shutil.disk_usage(output).free
    assert available>=required, 'FULL_BANK_REMOTE_DISK_ADMISSION_FAILED'
    proof=dict(source_run_base=str(source),model_revision=run['model_revision'],model_files_sha256=model_hashes,
        canonical_input_sha256=run['input_sha256'],completed_warmup_ready_sha256=READY_SHA,
        reused_manifest_sha256=MANIFEST_SHA,reused_feature_files=cache.manifest['files'],token_feature_spec_sha256=SPEC_SHA,
        global_metadata_sha256=GLOBAL_SHA,ordered_bindings_sha256=ordered.hexdigest(),global_rows=EXPECTED_ROWS,
        global_max_tokens=244,total_tokens=scan['total_tokens'],packed_feature_bytes=packed_bytes,
        expected_reused_rows=EXPECTED_REUSED,expected_newly_encoded_rows=EXPECTED_ROWS-EXPECTED_REUSED,
        disk_admission=dict(actual_available_bytes=available,required_free_bytes=required,checked_unix=time.time()),
        no_teacher_generated=True,no_head_loaded_or_fit=True,no_gpu_model_loaded=True)
    return rows,records,cache,proof,run


def prepare(args,output,mapping):
    _,_,cache,proof,_=qualify(args,mapping,output)
    prepared=dict(stage='FULL_CANONICAL_TOKEN_BANK_CPU_PREPARED',task_config=task_config(args),source_sha256=mapping,
        source_map_sha256=hashlib.sha256(canonical(mapping)).hexdigest(),feature_spec=cache.spec,qualification=proof,checked_unix=time.time())
    once(output/'prepared.json',prepared)
    progress(output,dict(stage=prepared['stage'],canonical_rows=EXPECTED_ROWS,reused_rows=EXPECTED_REUSED,newly_encoded_rows=0,required_free_bytes=proof['disk_admission']['required_free_bytes']))


def encode(args,output,mapping,gpu,birth):
    import numpy as np
    import torch
    from transformers import AutoModel,AutoTokenizer
    from reproduction.a800_common.qprobe_token_warmup import read
    from search_g1.qprobe_token_features import TokenFeatureCacheWriter,extract_question_token_features
    prepared=read(output/'prepared.json')
    assert prepared['task_config']==task_config(args) and prepared['source_sha256']==mapping
    rows,records,cache,proof,run=qualify(args,mapping,output)
    previous=dict(prepared['qualification']);current=dict(proof);previous.pop('disk_admission');current.pop('disk_admission');assert previous==current
    assert prepared['feature_spec']==cache.spec and torch.cuda.device_count()==1
    tokenizer=AutoTokenizer.from_pretrained(run['initial_model'],local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(run['initial_model'],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16).to('cuda:0')
    model.requires_grad_(False);model.eval()
    class Reused:
        index=cache.index
        def get_features(self,qids):
            batch=cache.get_batch(qids,device='cpu')
            return [batch['features'][i,:cache.index[qid]['length']].clone() for i,qid in enumerate(qids)]
    def extract(questions,bindings):
        tokens=tokenizer(questions,padding=False,truncation=False,add_special_tokens=True)['input_ids']
        assert len(tokens)==len(bindings)
        for ids,binding in zip(tokens,bindings):
            assert len(ids)==binding['length'] and hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()==binding['token_ids_sha256']
        return extract_question_token_features(model,tokenizer,questions,feature_layer=27,max_tokens=244,device='cuda:0')
    writer=TokenFeatureCacheWriter(output/'token-features',cache.spec,records)
    def report(event):
        if event['completed_rows']%256==0 or event['completed_rows']==EXPECTED_ROWS:progress(output,dict(stage='canonical_token_bank_encoding',**event))
    manifest,counts=emit_bank(rows,records,Reused(),writer,extract,batch_size=args.encode_batch_size,progress=report)
    assert counts==dict(rows=EXPECTED_ROWS,reused_rows=EXPECTED_REUSED,newly_encoded_rows=EXPECTED_ROWS-EXPECTED_REUSED)
    assert manifest['count']==EXPECTED_ROWS and manifest['spec_sha']==SPEC_SHA and manifest['total_tokens']==18367021
    actual_reused=0
    with (output/'token-features/index.jsonl').open() as stream:
        for ordinal,line in enumerate(stream):
            row=json.loads(line);assert row['qid']==records[ordinal]['qid']
            if row['qid'] in cache.index:
                assert row['feature_sha256']==cache.index[row['qid']]['feature_sha256'];actual_reused+=1
    assert actual_reused==EXPECTED_REUSED
    for name,item in manifest['files'].items():assert (output/'token-features'/name).stat().st_size==item['bytes']
    complete=dict(stage='FULL_CANONICAL_TOKEN_BANK_PRODUCED',source_commit=args.source_commit,
        source_map_sha256=prepared['source_map_sha256'],producer_identity=birth,gpu=gpu,task_config=task_config(args),
        canonical_order=True,**counts,feature_spec_sha256=SPEC_SHA,feature_cache_dir=str(output/'token-features'),
        feature_manifest_sha256=sha(output/'token-features/manifest.json'),full_file_sha256=manifest['files'],
        reused_manifest_sha256=MANIFEST_SHA,reused_rows_byte_exact=True,qualification=proof,
        prepared_sha256=sha(output/'prepared.json'),teacher_trajectories_generated=0,head_adam_updates=0,actor_started=False,
        separate_encoder_process=True,encoder_exit_proven=False,actual_process_and_cuda_exit_audit_required=True,
        cuda_allocator_zero_asserted=False,checked_unix=time.time())
    once(output/'complete.json',complete)
    progress(output,dict(stage=complete['stage'],**counts,feature_manifest_sha256=complete['feature_manifest_sha256']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True);parser.add_argument('--source-commit',required=True)
    parser.add_argument('--source-publication',required=True);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--physical-index',type=int);parser.add_argument('--encode-batch-size',type=int,default=4)
    args=parser.parse_args();output=output_path(args.output_dir)
    assert args.encode_batch_size>0 and os.environ.get('SLURM_JOB_ID') is None
    from reproduction.a800_common.qprobe_token_warmup import source_binding,gpu_admission
    mapping=source_binding(args)
    if args.prepare_only:output.mkdir(mode=0o700,exist_ok=False);os.environ['CUDA_VISIBLE_DEVICES']=''
    else:
        assert output.is_dir() and (output/'prepared.json').is_file()
        assert args.physical_index is not None and args.physical_index>=0 and args.physical_index not in (5,6)
        assert not (output/'complete.json').exists() and not list(output.glob('token-features*'))
    for relative in ('cache/hf','cache/triton','cache/inductor','tmp'):(output/relative).mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HOME=str(output/'cache/hf'),XDG_CACHE_HOME=str(output/'cache'),TMPDIR=str(output/'tmp'),TRITON_CACHE_DIR=str(output/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(output/'cache/inductor'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from reproduction.a800.owned_processes import identity
    record=identity(os.getpid());birth={key:record[key] for key in ('pid','start_ticks','sid','uid')};birth['hostname']=socket.gethostname()
    action='prepare' if args.prepare_only else 'encode'
    once(output/(action+'-claim.json'),dict(task_config=task_config(args),producer_identity=birth,no_automatic_retry=True,claimed_unix=time.time()))
    lock=None;exit_code=1
    try:
        if args.prepare_only:prepare(args,output,mapping)
        else:
            lock,gpu=gpu_admission(args.physical_index);encode(args,output,mapping,gpu,birth)
        exit_code=0
    except BaseException as failure:
        frames=[dict(file=Path(frame.filename).name,line=frame.lineno,function=frame.name) for frame in traceback.extract_tb(failure.__traceback__)[-8:]]
        once(output/(action+'-failure.json'),dict(failure_type=type(failure).__name__,frames=frames,producer_identity=birth,partial_artifacts_preserved=True,checked_unix=time.time()))
        print(json.dumps(dict(stage='FULL_CANONICAL_TOKEN_BANK_FAILED',failure_type=type(failure).__name__,partial_artifacts_preserved=True)),flush=True)
    finally:
        if lock is not None:lock.close()
        once(output/(action+'-exit.json'),dict(exit_code=exit_code,producer_identity=birth,process_exit_not_yet_proven=True,checked_unix=time.time()))
    return exit_code


if __name__=='__main__':raise SystemExit(main())
