"""Token-only native-policy vLLM service and semantics-preserving retrieval batching."""
import argparse
import asyncio
import copy
import time
import uuid
from common import OUT, atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['llm', 'retrieval'])
    parser.add_argument('--model')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--memory', type=float, default=.8)
    args = parser.parse_args()
    import uvicorn
    from fastapi import FastAPI, Request
    from contextlib import asynccontextmanager
    state = dict(requests=0, generated_tokens=0, prompt_tokens=0, active=0)

    @asynccontextmanager
    async def lifespan(app):
        if args.kind == 'llm':
            from vllm import AsyncEngineArgs, AsyncLLMEngine
            engine_args = AsyncEngineArgs(model=args.model, tokenizer=args.model,
                dtype='bfloat16', tensor_parallel_size=1,
                gpu_memory_utilization=args.memory, enforce_eager=True,
                max_model_len=24576, max_num_seqs=128,
                max_num_batched_tokens=16384, enable_chunked_prefill=True,
                enable_prefix_caching=True, disable_log_requests=True, seed=42)
            app.state.engine = AsyncLLMEngine.from_engine_args(engine_args)
        else:
            import aiohttp
            app.state.session = aiohttp.ClientSession(trust_env=False,
                timeout=aiohttp.ClientTimeout(total=180))
            app.state.queue = asyncio.Queue(maxsize=512)
            app.state.batch_task = asyncio.create_task(retrieval_batches(app))
        atomic(OUT/('service-%d-ready.json'%args.port),
               dict(port=args.port, kind=args.kind, epoch=time.time()))
        yield
        if args.kind == 'retrieval':
            app.state.batch_task.cancel()
            await app.state.session.close()

    async def retrieval_batches(app):
        while True:
            first = await app.state.queue.get()
            items = [first]
            await asyncio.sleep(.01)
            while len(items) < 32 and not app.state.queue.empty():
                items.append(app.state.queue.get_nowait())
            try:
                payload = dict(queries=[x[0] for x in items], topk=50, return_scores=True)
                async with app.state.session.post('http://127.0.0.1:18086/retrieve',
                                                   json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                assert len(data['result']) == len(items)
                for (_, future), row in zip(items, data['result']):
                    assert len(row) == 50
                    if not future.done():
                        future.set_result(row)
                state['requests'] += len(items)
            except BaseException as error:
                if isinstance(error, asyncio.CancelledError):
                    raise
                for _, future in items:
                    if not future.done():
                        future.set_exception(error)

    app = FastAPI(lifespan=lifespan)

    @app.get('/health')
    async def health():
        return dict(ready=True, kind=args.kind, **state)

    @app.post('/retrieve')
    async def retrieve(request: Request):
        data = await request.json()
        assert args.kind == 'retrieval' and data['topk'] == 50
        assert len(data['queries']) == 1 and data['return_scores'] is True
        future = asyncio.get_running_loop().create_future()
        await app.state.queue.put((data['queries'][0], future))
        return dict(result=[await future])

    @app.post('/generate')
    async def generate(request: Request):
        from vllm import SamplingParams
        data = await request.json()
        prompt = data['prompt_ids']
        assert prompt and len(prompt) < 24576
        params = copy.deepcopy(data['sampling_params'])
        assert params['temperature'] == 0 and params['top_p'] == 1
        params.pop('logprobs', None)
        params.setdefault('repetition_penalty', 1.)
        # Exactly the pinned native async server's remaining-context token budget.
        sampling = SamplingParams(max_tokens=24576-len(prompt), logprobs=None, **params)
        state['active'] += 1
        try:
            final = None
            async for output in app.state.engine.generate(
                    prompt={'prompt_token_ids': prompt}, sampling_params=sampling,
                    request_id=uuid.uuid4().hex):
                final = output
            assert final is not None and len(final.outputs) == 1
            tokens = list(final.outputs[0].token_ids)
            state['requests'] += 1
            state['prompt_tokens'] += len(prompt)
            state['generated_tokens'] += len(tokens)
            return dict(token_ids=tokens, log_probs=None)
        finally:
            state['active'] -= 1

    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
