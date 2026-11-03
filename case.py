import os, sys, json, time, argparse, pathlib, traceback

def emit(event, **values):
    item = dict(event=event, time=time.time(), **values)
    print('CAPACITY_EVENT ' + json.dumps(item, sort_keys=True), flush=True)
    with open(os.environ['CASE_EVENTS'], 'a') as f:
        f.write(json.dumps(item, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--memory', type=float, required=True)
    p.add_argument('--max-len', type=int, required=True)
    p.add_argument('--lengths', default='32768')
    p.add_argument('--kv', default='auto')
    p.add_argument('--prefill-chunk', type=int, default=2048)
    args = p.parse_args()
    import importlib.metadata, torch
    from vllm import LLM, SamplingParams
    props = torch.cuda.get_device_properties(0)
    assert torch.cuda.device_count() == 1, 'Only one allocated GPU may be used'
    assert 75 * 2**30 <= props.total_memory <= 85 * 2**30, 'Expected an 80GB GPU'
    emit('gpu_verified', name=props.name, memory_bytes=props.total_memory,
         uuid=str(getattr(props, 'uuid', 'unavailable')), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
         versions={x:importlib.metadata.version(x) for x in ['torch','transformers','vllm']},
         torch_cuda=torch.version.cuda, args=vars(args))
    t=time.monotonic()
    llm = LLM(model=str(pathlib.Path(__file__).parent), load_format='dummy',
              skip_tokenizer_init=True, dtype='bfloat16', kv_cache_dtype=args.kv,
              tensor_parallel_size=1, pipeline_parallel_size=1,
              max_model_len=args.max_len, max_num_seqs=1,
              max_num_batched_tokens=args.prefill_chunk,
              enable_chunked_prefill=True, enable_prefix_caching=False,
              enforce_eager=True, gpu_memory_utilization=args.memory,
              swap_space=0, cpu_offload_gb=0, seed=1234,
              disable_log_stats=False)
    cc=llm.llm_engine.vllm_config.cache_config
    emit('engine_ready', startup_seconds=time.monotonic()-t,
         cache_num_gpu_blocks=getattr(cc,'num_gpu_blocks',None),
         cache_block_size=getattr(cc,'block_size',None),
         max_model_len=llm.llm_engine.vllm_config.model_config.max_model_len)
    for length in [int(x) for x in args.lengths.split(',') if x]:
        assert length+1 <= args.max_len
        tokens = [100+(i%64) for i in range(length)]
        emit('request_started', input_tokens=length, output_tokens_requested=1)
        t=time.monotonic()
        out=llm.generate([dict(prompt_token_ids=tokens)],
                         SamplingParams(temperature=0, max_tokens=1,
                                        ignore_eos=True, detokenize=False), use_tqdm=False)
        torch.cuda.synchronize()
        count=len(out[0].outputs[0].token_ids)
        assert count==1
        assert len(out[0].prompt_token_ids)==length
        emit('request_pass', input_tokens=length, output_tokens=count,
             total_tokens=length+count, elapsed_seconds=time.monotonic()-t,
             finish_reason=out[0].outputs[0].finish_reason,
             parent_allocated_bytes=torch.cuda.memory_allocated(),
             parent_reserved_bytes=torch.cuda.memory_reserved())
    emit('case_pass')

if __name__=='__main__':
    try: main()
    except Exception as e:
        emit('case_failed', error_type=type(e).__name__, error=str(e))
        traceback.print_exc()
        sys.exit(1)
