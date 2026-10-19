"""Frozen manifests and resumable single-GPU CF; no trainer imports/side effects."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from collections import Counter

from .core import (CF_CONFIG, PROTOCOL, aggregate, atomic, canonical, classify,
    digest, messages, prepare_row, read, read_jsonl, sha, utc)
from search_r1.qa_protocol import extract_final_answer

def checkpoint_inventory(directory):
    root=Path(directory).resolve()
    index=read(root/'model.safetensors.index.json')
    shards=sorted(set(index['weight_map'].values()))
    required=shards+['model.safetensors.index.json','config.json','tokenizer_config.json']
    if not (root/'tokenizer.json').is_file(): raise ValueError('Missing tokenizer.json')
    required += [p.name for p in root.iterdir() if p.is_file() and p.name in
        ('tokenizer.json','vocab.json','merges.txt','special_tokens_map.json','generation_config.json','added_tokens.json')]
    files=[]
    for name in sorted(set(required)):
        p=root/name
        if p.stat().st_size<=0: raise ValueError('Empty model file '+name)
        files.append(dict(name=name, bytes=p.stat().st_size, sha256=sha(p)))
    return dict(path=str(root), files=files, checkpoint_hash=digest(files), shard_count=len(shards))

def source_inventory(source, expected=3610):
    rows=list(read_jsonl(source)); keys=[]
    for row in rows:
        if row['data_source']!='nq': raise ValueError('Only fixed NQ evaluation occurrences')
        keys.append([str(row['data_source']),int(row['evaluation_ordinal']),str(row['qid'])])
    if len(rows)!=expected or len({canonical(k) for k in keys})!=expected:
        raise ValueError('Expected exactly %d distinct evaluation occurrence keys (bare qid may repeat)'%expected)
    return rows, dict(path=str(Path(source).resolve()),bytes=Path(source).stat().st_size,
        sha256=sha(source), count=len(rows), occurrence_order_sha256=digest(keys))

def prepare(args):
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('Use a new empty protocol output directory')
    rows, source=source_inventory(args.source,args.expected_count)
    ckpt=checkpoint_inventory(args.checkpoint)
    from .identity import validate_normal_identity
    normal_identity=validate_normal_identity(rows,args.checkpoint,args.normal_config)
    config=dict(CF_CONFIG)
    code_files=[Path(__file__), Path(__file__).with_name('core.py'),
                Path(__file__).with_name('identity.py'),
                Path(__file__).with_name('replacements.py'),
                Path(__file__).parents[2]/'search_r1/qa_protocol.py']
    code={str(p.relative_to(Path(__file__).parents[2])):sha(p) for p in code_files}
    config['implementation_sha256']=digest(code)
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception: commit=None
    manifest=dict(schema_version=1, protocol_version=PROTOCOL, method=args.method, scale='3B', seed=args.seed,
        run_id=args.run_id, actor_version='iter200', checkpoint=ckpt, checkpoint_hash=ckpt['checkpoint_hash'],
        source=source, normal_identity=normal_identity,
        normal_config_path=str(Path(args.normal_config).resolve()),
        pilot_only=args.expected_count!=3610, normal_evaluation_config=read(args.normal_config), cf_config=config,
        cf_config_sha256=digest(config), code_files=code, code_commit=commit, created_at=utc(), output=str(output))
    manifest['normal_evaluation_config_sha256']=digest(manifest['normal_evaluation_config'])
    manifest['manifest_sha256']=digest(manifest)
    prepared=[prepare_row(r,manifest) for r in rows]
    output.mkdir(parents=True,exist_ok=True)
    atomic(output/'manifest.json',manifest)
    with open(output/'prepared.jsonl','xb') as f:
        for r in prepared: f.write(canonical(r)+b'\n')
        f.flush(); os.fsync(f.fileno())
    atomic(output/'prepared-receipt.json',dict(sha256=sha(output/'prepared.jsonl'),count=len(prepared),utc=utc()))
    if args.reuse_results:
        prior=read(Path(args.reuse_results)/'manifest.json')
        if any(prior[k]!=manifest[k] for k in ('method','seed','checkpoint_hash','cf_config_sha256','normal_evaluation_config_sha256')):
            raise ValueError('Pilot cache identity differs from full evaluation')
        allowed={r['request_id']:r for r in prepared}
        for result in result_rows(Path(args.reuse_results)):
            if result['status']=='CLASSIFIED' and result['request_id'] in allowed:
                p=allowed[result['request_id']]
                if result['source_record_hash']!=p['source_record_hash']: raise ValueError('Changed pilot source')
                atomic(output/'results'/(result['request_id']+'.json'),result)
    atomic(output/'summary.json',aggregate(prepared,[]))
    print(json.dumps(dict(manifest=str(output/'manifest.json'),summary=read(output/'summary.json'))))

def validated(manifest_path):
    manifest=read(manifest_path); expected=manifest.pop('manifest_sha256')
    if digest(manifest)!=expected: raise ValueError('Manifest was modified')
    manifest['manifest_sha256']=expected
    out=Path(manifest['output']); receipt=read(out/'prepared-receipt.json')
    if sha(out/'prepared.jsonl')!=receipt['sha256']: raise ValueError('Prepared input changed')
    if sha(manifest['source']['path'])!=manifest['source']['sha256']: raise ValueError('Normal source changed')
    root=Path(__file__).parents[2]
    if {p:sha(root/p) for p in manifest['code_files']}!=manifest['code_files']:
        raise ValueError('Runtime code changed after manifest freeze')
    # Hash all checkpoint files at prepare and final verification. Every run checks
    # size and small-file identities, never silently resolves a different directory.
    for item in manifest['checkpoint']['files']:
        p=Path(manifest['checkpoint']['path'])/item['name']
        if p.stat().st_size!=item['bytes']: raise ValueError('Checkpoint size changed')
        if item['bytes']<16*1024*1024 and sha(p)!=item['sha256']: raise ValueError('Checkpoint metadata changed')
    return manifest, list(read_jsonl(out/'prepared.jsonl'))

def result_rows(out): return [read(p) for p in sorted((out/'results').glob('*.json'))]

def attempt_token_usage(out):
    events=list(read_jsonl(out/'attempts.jsonl')) if (out/'attempts.jsonl').exists() else []
    started={};success={};errors=0
    for event in events:
        if event['status']=='INFERENCE_STARTED':
            for rid,a in event['attempt_ids'].items():started[(rid,a)]=event['input_tokens_by_id'][rid]
        elif event['status']=='CLASSIFIED':success[(event['request_id'],event['attempt_id'])]=event.get('generation_tokens')
        elif event['status']=='INFERENCE_ERROR':errors+=1
    missing=set(started)-set(success)
    known_outputs=[v for v in success.values() if isinstance(v,int)]
    return dict(request_attempts=len(started),known_requested_input_tokens=sum(started.values()),
        successful_attempt_output_tokens=sum(known_outputs),failed_or_interrupted_attempts=len(missing),
        errors=errors,total_generated_tokens=sum(known_outputs) if not missing else None,
        total_generated_tokens_null_reason='failed_or_interrupted_generation_cost_unknown; successful_tokens_are_a_lower_bound' if missing else None)

def run(args):
    manifest, prepared=validated(args.manifest); out=Path(manifest['output'])
    lock=open(out/'worker.lock','a+')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('An existing worker holds this task lock')
    atomic(out/'worker.json',dict(pid=os.getpid(),host=socket.gethostname(),command=sys.argv,utc=utc()),replace=True)
    done={r['request_id'] for r in result_rows(out) if r['status']=='CLASSIFIED'}
    pending=[r for r in prepared if r['status']=='READY' and r['request_id'] not in done]
    if args.limit: pending=pending[:args.limit]
    if not pending:
        atomic(out/'summary.json',aggregate(prepared,result_rows(out)),replace=True); return
    gpu_fd=None
    if args.gpu_uuid:
        from .normal_a800 import gpu_lease
        gpu_fd,physical_index=gpu_lease(args.gpu_uuid)
        # vLLM0.6.3 NVML platform parsing requires integer physical indices.
        os.environ['CUDA_VISIBLE_DEVICES']=str(physical_index)
    from transformers import AutoTokenizer
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    model=manifest['checkpoint']['path']; config=manifest['cf_config']
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    free,total=torch.cuda.mem_get_info(); reserve=4*1024**3
    utilization=min(args.max_memory_utilization,(free-reserve)/total)
    if free<20*1024**3 or utilization<=0: raise RuntimeError('Less than20GiB free: refuse occupied/insufficient GPU')
    requested_max=args.max_batch
    engine=LLM(model=model,tokenizer=model,dtype='bfloat16',tensor_parallel_size=1,
        max_model_len=config['max_model_len'],gpu_memory_utilization=utilization,
        max_num_seqs=requested_max,max_num_batched_tokens=16384,enable_chunked_prefill=True,
        enable_prefix_caching=True,disable_log_stats=True,swap_space=0,trust_remote_code=False,
        seed=config['seed'])
    engine_tokenizer=engine.get_tokenizer()
    if engine_tokenizer.encode('grounding identity check')!=tokenizer.encode('grounding identity check'):
        raise ValueError('Engine tokenizer differs from frozen model')
    generation=read(Path(model)/'generation_config.json') if (Path(model)/'generation_config.json').exists() else {}
    stop=generation.get('eos_token_id',tokenizer.eos_token_id)
    stop=[stop] if isinstance(stop,int) else list(stop)
    stop=sorted(set(stop+[tokenizer.eos_token_id]))
    params=SamplingParams(temperature=0.,top_p=1.,top_k=-1,max_tokens=config['max_new_tokens'],
        stop=[config['answer_stop']],include_stop_str_in_output=config['include_stop_str_in_output'],
        stop_token_ids=stop,skip_special_tokens=False,seed=config['seed'])
    atomic(out/('engine-'+str(os.getpid())+'.json'),dict(checkpoint=model,checkpoint_hash=manifest['checkpoint_hash'],
        torch=torch.__version__,vllm=vllm.__version__,device=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),free_before=free,total=total,
        utilization=utilization,max_batch=requested_max,stop_token_ids=stop,
        sampling_params_repr=repr(params),max_model_len=config['max_model_len'],
        pid=os.getpid(),utc=utc()))
    position=0; batch=min(16,requested_max); completed=len(done); started=time.monotonic()
    attempts=Counter()
    if (out/'attempts.jsonl').exists():
        for item in read_jsonl(out/'attempts.jsonl'):
            if 'attempt_ids' in item:
                for rid,a in item['attempt_ids'].items():attempts[rid]=max(attempts[rid],a)
            elif item.get('request_id'):
                attempts[item['request_id']]=max(attempts[item['request_id']],item.get('attempt_id',1))
    def attempt_event(obj):
        with open(out/'attempts.jsonl','ab') as f:
            f.write(canonical(obj)+b'\n'); f.flush();os.fsync(f.fileno())
    while position<len(pending):
        rows=pending[position:position+batch]; valid=[]; inputs=[]; stamp=utc()
        for row in rows:
            prompt=tokenizer.apply_chat_template(messages(row),tokenize=True,add_generation_prompt=True)
            if len(prompt)+config['max_new_tokens']>config['max_model_len']:
                # Never silently truncate or alter the original seen-doc intervention.
                record=dict(row,status='SOURCE_INCOMPLETE',invalid_reason='cf_context_exceeds_frozen_budget')
                atomic(out/'results'/(row['request_id']+'.json'),record,replace=True); continue
            valid.append(row);inputs.append(dict(prompt_token_ids=prompt))
        t=time.monotonic()
        if not valid: position+=len(rows);continue
        if any(attempts[r['request_id']]>=config['max_attempts'] for r in valid):
            raise RuntimeError('Retry budget exhausted; inspect attempts before explicitly authorizing a new attempt policy')
        for row in valid: attempts[row['request_id']]+=1
        attempt_event(dict(status='INFERENCE_STARTED',attempt_ids={r['request_id']:attempts[r['request_id']] for r in valid},
            input_tokens_by_id={r['request_id']:len(p['prompt_token_ids']) for r,p in zip(valid,inputs)},utc=utc()))
        try:
            outputs=engine.generate(inputs,params,use_tqdm=False)
            if len(outputs)!=len(valid): raise RuntimeError('Incomplete engine output batch')
            for row,request,response in zip(valid,inputs,outputs):
                if list(response.prompt_token_ids)!=request['prompt_token_ids'] or len(response.outputs)!=1:
                    raise RuntimeError('Engine request/response identity mismatch')
                answer=response.outputs[0]
                if answer.finish_reason not in ('stop','length'):
                    raise RuntimeError('Incomplete generation '+str(answer.finish_reason))
                cf_answer=extract_final_answer([answer.text])
                result=dict(request_id=row['request_id'],checkpoint_hash=manifest['checkpoint_hash'],
                    cf_config_sha256=manifest['cf_config_sha256'],source_record_hash=row['source_record_hash'],
                    occurrence_key=row['occurrence_key'],status='CLASSIFIED',
                    classification=classify(row['original_answer'],cf_answer,row['cf_aliases'],row['gold']),
                    cf_answer=cf_answer,cf_response=answer.text,finish_reason=answer.finish_reason,
                    generation_tokens=len(answer.token_ids),stop_reason=answer.stop_reason,
                    model_input_tokens=len(request['prompt_token_ids']),
                    trajectory_total_tokens=len(request['prompt_token_ids'])+len(answer.token_ids),
                    cached_input_tokens=getattr(response,'num_cached_tokens',None),
                    cached_input_tokens_null_reason=None if hasattr(response,'num_cached_tokens') else 'vllm_request_output_does_not_expose_cache_count',
                    response_checksum=digest(dict(text=answer.text,token_ids=answer.token_ids)),
                    prompt_token_ids_sha256=digest(request['prompt_token_ids']),started_at=stamp,
                    attempt_id=attempts[row['request_id']],
                    completed_at=utc(),elapsed_batch_seconds=time.monotonic()-t)
                atomic(out/'results'/(row['request_id']+'.json'),result,replace=True)
                attempt_event(dict(request_id=row['request_id'],attempt_id=attempts[row['request_id']],
                    generation_tokens=len(answer.token_ids),status='CLASSIFIED',started_at=stamp,completed_at=utc()))
                completed+=1
        except Exception as error:
            attempt_event(dict(request_ids=[r['request_id'] for r in valid],status='INFERENCE_ERROR',
                attempt_ids={r['request_id']:attempts[r['request_id']] for r in valid},
                error=type(error).__name__+':'+str(error),batch=batch,utc=utc()))
            # No programming/transport error is converted to a model Amb.
            # CUDA OOM is isolated to this evaluator, never kills other GPU tasks.
            if isinstance(error,torch.cuda.OutOfMemoryError) or 'CUDA out of memory' in str(error):
                torch.cuda.empty_cache()
                if batch>1: batch=max(1,batch//2);continue
            raise
        position+=len(rows)
        progress=dict(completed_unique=completed,total_ready=sum(r['status']=='READY' for r in prepared),
            newly_completed=position,elapsed_seconds=time.monotonic()-started,batch_size=len(valid),
            batch_seconds=time.monotonic()-t,next_batch=min(requested_max,batch*2),
            result_bytes=sum(p.stat().st_size for p in (out/'results').glob('*.json')),
            gpu_max_allocated_bytes=torch.cuda.max_memory_allocated(),latest_success=utc())
        atomic(out/'progress.json',progress,replace=True);print(json.dumps(progress),flush=True)
        batch=min(requested_max,batch*2)
    atomic(out/'summary.json',aggregate(prepared,result_rows(out)),replace=True)
    atomic(out/('worker-exit-'+str(os.getpid())+'.json'),dict(exit_code=0,utc=utc(),summary=read(out/'summary.json')))
    if gpu_fd is not None: os.close(gpu_fd)

def verify(args):
    manifest,prepared=validated(args.manifest); out=Path(manifest['output'])
    actual=checkpoint_inventory(manifest['checkpoint']['path'])
    if actual!=manifest['checkpoint']: raise ValueError('Checkpoint final full SHA mismatch')
    from .identity import validate_normal_identity
    identity=validate_normal_identity(list(read_jsonl(manifest['source']['path'])),manifest['checkpoint']['path'],manifest['normal_config_path'])
    if identity!=manifest['normal_identity']: raise ValueError('Normal identity changed at final readback')
    summary=aggregate(prepared,result_rows(out))
    summary['token_usage']['counterfactual_attempts']=attempt_token_usage(out)
    if manifest['pilot_only']:
        summary['status']='pilot_only'
    summary.update(method=manifest['method'],seed=manifest['seed'],checkpoint_hash=manifest['checkpoint_hash'],
        manifest_sha256=manifest['manifest_sha256'],prepared_sha256=sha(out/'prepared.jsonl'),
        result_file_count=len(result_rows(out)),independent_reclassification=True)
    atomic(out/'verified-summary.json',summary,replace=True)
    print(json.dumps(summary))

def main():
    p=argparse.ArgumentParser(__doc__); sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    for k in ('source','checkpoint','output','method','run-id','normal-config'): prep.add_argument('--'+k,required=True)
    prep.add_argument('--seed',type=int,default=None,help='Actual method training seed if verified; unknown remains null. CF intervention/decode seed is separately frozen.')
    prep.add_argument('--expected-count',type=int,default=3610,help='Other sizes are explicitly pilot-only, never final results')
    prep.add_argument('--reuse-results',help='Identity-checked immutable pilot results from the same source occurrences')
    for name in ('run','pilot','aggregate','verify'):
        cmd=sub.add_parser(name);cmd.add_argument('--manifest',required=True)
        if name in ('run','pilot'):
            cmd.add_argument('--max-batch',type=int,default=256);cmd.add_argument('--limit',type=int,default=0)
            cmd.add_argument('--gpu-uuid',help='A800 physical UUID: acquire the existing task GPU lease before CUDA')
            cmd.add_argument('--max-memory-utilization',type=float,default=.90,help='GPU-total cap; shared H100 requires independently justified conservative budget')
    args=p.parse_args()
    if args.action in ('run','pilot') and not 0<args.max_memory_utilization<=.95:
        raise ValueError('Memory cap must be in (0,.95]')
    if args.action=='prepare':prepare(args)
    elif args.action in ('run','pilot'):
        if args.action=='pilot':args.limit=16
        run(args)
    elif args.action=='verify':verify(args)
    else:
        manifest,prepared=validated(args.manifest);out=Path(manifest['output'])
        summary=aggregate(prepared,result_rows(out));atomic(out/'summary.json',summary,replace=True)
        print(json.dumps(summary))

if __name__=='__main__': main()
