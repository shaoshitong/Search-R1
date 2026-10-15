"""Use the actual author's agent/tool methods with deterministic fake model outputs."""
import asyncio
import ast
import json
import types
import sys
from common import OUT, atomic, env
sys.path[:0] = env()['PYTHONPATH'].split(':')


async def main():
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.co_search_agent_loop import CoSearchAgentLoop
    from verl.experimental.agent_loop.tool_parser import SearchR1ToolParser,FunctionCall
    from question_metadata import install,install_parser
    from common_em import score
    install(CoSearchAgentLoop)
    install_parser(SearchR1ToolParser,FunctionCall)
    config=OmegaConf.create(json.loads((OUT/'native-config.json').read_text()))
    config.actor_rollout_ref.rollout.multi_turn.tool_config_path=str(OUT/'tool.yaml')
    tokenizer=AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path,local_files_only=True)
    prompts=[]

    class FakeManager:
        def __init__(self,role):
            self.role,self.calls=role,0

        async def generate(self,request_id,prompt_ids,sampling_params,**kwargs):
            self.calls+=1
            prompts.append((self.role,tokenizer.decode(prompt_ids)))
            assert sampling_params['temperature']==0
            if self.role!='ranker':
                assert '</search>' in sampling_params['stop'] and sampling_params['include_stop_str_in_output']
            if self.role=='ranker':
                text='<reason>Select relevant passages.</reason><rerank>[2] > [1] > [3] > [4] > [5]</rerank>'
            elif self.calls==1:
                text='<think>Search now.</think><search>test query</search>'
            else:
                text='<think>Use the retrieved evidence.</think><answer>Paris</answer>'
            return types.SimpleNamespace(token_ids=tokenizer.encode(text,add_special_tokens=False),log_probs=None)

    main_manager,ranker=FakeManager('main'),FakeManager('ranker')
    tree=ast.parse((OUT/'tools/evaluate.py').read_text())
    main_node=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='main')
    audit_node=next(n for n in main_node.body if isinstance(n,ast.ClassDef) and n.name=='AuditedLoop')
    namespace={'CoSearchAgentLoop':CoSearchAgentLoop,'json':json}
    exec(compile(ast.Module(body=[audit_node],type_ignores=[]),'actual-evaluation-wrapper','exec'),namespace)
    loop=namespace['AuditedLoop'](types.SimpleNamespace(config=config),main_manager,tokenizer,None,
                          reranker_server_manager=ranker,reranker_tokenizer=tokenizer)
    loop.actions,loop.reranks=[],[]

    async def retrieve(query,top_n):
        assert query=='test query' and top_n==50
        return [dict(id=str(i),contents='Evidence passage %d'%i,score=1./i) for i in range(1,51)]

    loop.tools['search']._call_retrieval_api=retrieve
    output,ranked=await loop.run(dict(temperature=0.,top_p=1.,top_k=-1,n=1),
        raw_prompt=[dict(role='user',content='Answer the given question. Use search tools.\nQuestion: Which city?')],
        extra_info={'index':0,'split':'final'},uid='cpu-fixture',
        reward_model={'ground_truth':{'target':[]}},tools_kwargs={})
    assert main_manager.calls==2 and ranker.calls==1 and len(ranked)==1
    assert any(role=='ranker' and 'Which city?' in text for role,text in prompts)
    assert len(loop.actions)==2 and len(loop.reranks)==1
    assert loop.reranks[0]['reranker_success'] and loop.reranks[0]['num_retrieved_docs']==50
    assert '<rerank>' in loop.reranks[0]['native_retained_reranker_output']
    assert loop.reranks[0]['reranker_prompt_tokens']>0
    assert 'Evidence passage 2' in prompts[-1][1]
    assert prompts[-1][1].index('Evidence passage 2')<prompts[-1][1].index('Evidence passage 1')
    assert score('<answer>Paris</answer>',['Paris'])['em']==1
    assert len(output.response_mask)==len(output.response_ids)
    assert any(x==0 for x in output.response_mask) and any(x==1 for x in output.response_mask)
    result=dict(status='ACTUAL_NATIVE_AGENT_AND_RERANK_TOOL_TWO_MODEL_CPU_CONTRACT_PASS',
                fake_main_calls=2,fake_reranker_calls=1,candidates=50,selected=5,
                selected_order_reaches_main=True,model_tool_boundaries_preserved=True,
                actual_training_shared_search_adapter=True,main_search_stop_matches_training=True,
                actual_production_audit_wrapper=True,
                native_question_restored_from_original_prompt=True,
                GPU_inference_not_tested=True,gold_targets_not_given_to_agent=True)
    atomic(OUT/'native-agent-cpu-contract.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    asyncio.run(main())
