"""Asynchronous evaluation through the author's actual dual-model agent/tool code."""
import asyncio
import collections
import hashlib
import json
import pathlib
import sys
import time
import types
from common import OUT, SOURCE, TRAIN, N, atomic, sha


class Manager:
    def __init__(self, session, ports, capacities):
        self.session, self.ports, self.capacities = session, ports, capacities
        self.active = [0]*len(ports)
        self.requests = 0

    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None):
        assert image_data is None
        index = min(range(len(self.ports)), key=lambda i: self.active[i]/self.capacities[i])
        self.active[index] += 1
        try:
            async with self.session.post('http://127.0.0.1:%d/generate'%self.ports[index],
                json=dict(prompt_ids=prompt_ids, sampling_params=sampling_params)) as response:
                response.raise_for_status()
                data = await response.json()
            self.requests += 1
            return types.SimpleNamespace(**data)
        finally:
            self.active[index] -= 1


async def main():
    import aiohttp
    import pandas as pd
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.co_search_agent_loop import CoSearchAgentLoop
    from common_em import score, identity, summarize

    config = OmegaConf.create(json.loads((OUT/'native-config.json').read_text()))
    config.actor_rollout_ref.rollout.multi_turn.tool_config_path = str(OUT/'tool.yaml')
    config.data.tool_config_path = str(OUT/'tool.yaml')
    tokenizer = AutoTokenizer.from_pretrained(OUT/'models/main', local_files_only=True)
    rr_tokenizer = AutoTokenizer.from_pretrained(OUT/'models/reranker', local_files_only=True)
    frame = pd.read_parquet(TRAIN/'data/final.parquet')
    references = [json.loads(line) for line in (TRAIN/'data/final.jsonl').open() if line.strip()]
    assert len(frame) == len(references) == N
    expected = collections.Counter(identity(x['question'], x['answers']) for x in references)
    actual = collections.Counter(identity(x['extra_info']['question'], x['reward_model']['ground_truth'])
                                 for _, x in frame.iterrows())
    assert actual == expected

    class AuditedLoop(CoSearchAgentLoop):
        async def _handle_generating_state(self, data, *args, **kwargs):
            start = len(data.response_mask)
            state = await super()._handle_generating_state(data, *args, **kwargs)
            self.actions.append(dict(author='main_model',
                text=self.tokenizer.decode(data.response_ids, skip_special_tokens=True),
                response_token_start=start, response_token_end=start+len(data.response_ids),
                fully_retained=start+len(data.response_ids)<=self.response_length))
            return state

        async def _call_tool(self, tool_call, tools_kwargs, agent_data):
            response = await super()._call_tool(tool_call, tools_kwargs, agent_data)
            _, _, detail = response
            if detail.get('reranker_crashed'):
                raise RuntimeError('Infrastructure failure: '+str(detail))
            self.reranks.append({key: detail.get(key) for key in
                ['reranker_success','reranker_fallback','reranker_fallback_reason',
                 'num_retrieved_docs','num_reranked_docs','sub_query']})
            return response

    results = {}
    binding_sha = sha(OUT/'binding.json')
    for path in sorted((OUT/'rows').glob('*.json')):
        row = json.loads(path.read_text())
        assert row['binding_sha256'] == binding_sha and path.name == '%05d.json'%row['row_id']
        source = frame.iloc[row['row_id']]
        assert row['question_answer_sha256'] == identity(source['extra_info']['question'],
                                                        source['reward_model']['ground_truth'])
        assert row['row_id'] not in results
        results[row['row_id']] = row
    started = time.time()
    inflight = set()
    semaphore = asyncio.Semaphore(128)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False,
                                     connector=aiohttp.TCPConnector(limit=256)) as session:
        main_manager = Manager(session, [18100,18101], [1,1])
        rank_manager = Manager(session, [18102,18103], [3,1])

        async def question(index):
            async with semaphore:
                source = frame.iloc[index]
                for attempt in range(3):
                    agent = AuditedLoop(types.SimpleNamespace(config=config), main_manager,
                        tokenizer, None, reranker_server_manager=rank_manager,
                        reranker_tokenizer=rr_tokenizer)
                    agent.actions, agent.reranks = [], []
                    try:
                        output, _ = await agent.run(
                            sampling_params=dict(temperature=0., top_p=1., top_k=-1, n=1),
                            raw_prompt=list(source['prompt']), uid=str(index),
                            extra_info={'question':source['extra_info']['question']},
                            reward_model={'ground_truth':{'target':[]}}, tools_kwargs={})
                        last = agent.actions[-1] if agent.actions else None
                        refs = source['reward_model']['ground_truth']
                        metrics = score(last['text'] if last else None, refs,
                                        boundary_verified=bool(last and last['fully_retained']))
                        row = dict(row_id=index, question_uid=str(index),
                            question_answer_sha256=identity(source['extra_info']['question'], refs),
                            data_source=str(source['data_source']), model_actions=agent.actions,
                            response_mask=output.response_mask, reranker_events=agent.reranks,
                            inference_received_gold=False, binding_sha256=binding_sha,
                            **metrics)
                        atomic(OUT/('rows/%05d.json'%index), row)
                        results[index] = row
                        if len(results)%32 == 0 or len(results) == N:
                            atomic(OUT/'progress.json', dict(epoch=time.time(),
                                rows=len(results), total=N, seconds=time.time()-started,
                                main_calls=main_manager.requests, reranker_calls=rank_manager.requests,
                                main_active=main_manager.active, reranker_active=rank_manager.active))
                            print('SCORED_ROWS', len(results), flush=True)
                        return
                    except BaseException as error:
                        if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)):
                            raise
                        atomic(OUT/('failures/%05d-attempt%d.json'%(index,attempt)),
                               dict(epoch=time.time(), error=repr(error), row_id=index))
                        if attempt == 2:
                            raise
                        await asyncio.sleep(2**attempt)

        # Bound task objects as well as active questions. Keep engines fed continuously.
        for index in range(N):
            if index in results:
                continue
            inflight.add(asyncio.create_task(question(index)))
            if len(inflight) >= 256:
                done, inflight = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
        if inflight:
            await asyncio.gather(*inflight)
    assert sorted(results) == list(range(N))
    rows = [results[index] for index in range(N)]
    assert collections.Counter(x['question_answer_sha256'] for x in rows) == expected
    metrics = summarize(rows)
    assert metrics['trajectories'] == N and len(metrics['datasets']) == 7
    atomic(OUT/'complete.json', dict(status='COSEARCH_DUAL_MODEL_FINAL51713_COMPLETE',
        epoch=time.time(), training_job=713379, checkpoint_step=200, rows=N,
        binding_sha256=binding_sha, metrics=metrics, no_gold_to_inference=True))


if __name__ == '__main__':
    asyncio.run(main())
