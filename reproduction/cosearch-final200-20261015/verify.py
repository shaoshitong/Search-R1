"""Independent CPU acceptance of saved dual-model final predictions; no training."""
import collections
import json
import hashlib
import pathlib
import subprocess
import time
from common import TRAIN, OUT, N, atomic, sha


def main():
    import pandas as pd
    from common_em import score, identity, summarize
    if not (OUT/'complete.json').exists():
        print(json.dumps(dict(status='COSEARCH_FULL_FINAL_NOT_YET_COMPLETE')))
        return
    complete=json.loads((OUT/'complete.json').read_text())
    assert complete['status']=='COSEARCH_DUAL_MODEL_FINAL51713_COMPLETE'
    assert complete['rows']==N and complete['checkpoint_step']==200
    assert complete['no_gold_to_inference']
    binding=json.loads((OUT/'binding.json').read_text())
    assert binding['training_job']==713379 and binding['checkpoint_step']==200
    assert sha(TRAIN/'data/final.parquet')==binding['data_sha256']
    assert sha(TRAIN/'data/final.jsonl')==binding['data_jsonl_sha256']
    assert sha(OUT/'binding.json')==complete['binding_sha256']
    data=pd.read_parquet(TRAIN/'data/final.parquet')
    references=[json.loads(line) for line in (TRAIN/'data/final.jsonl').open() if line.strip()]
    from question_metadata import with_question
    assert len(references)==N
    assert len(data)==N
    assert len(list((OUT/'rows').glob('*.json')))==N
    rows=[]
    rerank_calls=rerank_success=0
    for index in range(N):
        row=json.loads((OUT/('rows/%05d.json'%index)).read_text())
        source=data.iloc[index]
        reference=references[index]
        assert row['row_id']==index and row['question_uid']==reference['question_id']
        assert row['binding_sha256']==complete['binding_sha256']
        assert row['inference_received_gold'] is False
        assert row['data_source']==source['data_source']
        original=with_question(dict(raw_prompt=list(source['prompt']),extra_info=dict(source['extra_info'])))
        question=original['extra_info']['question']
        assert question.strip()==reference['question'].strip()
        assert row['native_prompt_question_sha256']==hashlib.sha256(question.encode()).hexdigest()
        assert row['question_answer_sha256']==identity(reference['question'],
                                                       source['reward_model']['ground_truth'])
        assert row['question_answer_sha256']==identity(reference['question'],reference['answers'])
        last=row['model_actions'][-1] if row['model_actions'] else None
        if last:
            assert last['author']=='main_model'
        expected=score(last['text'] if last else None,source['reward_model']['ground_truth'],
                       boundary_verified=bool(last and last['fully_retained']))
        for key,value in expected.items():
            assert row[key]==value,(index,key)
        for event in row['reranker_events']:
            if event['reranker_prompt_tokens']>0:
                assert event['num_retrieved_docs']==50
                assert 0<event['reranker_prompt_tokens']<=16384
                assert isinstance(event['native_retained_reranker_output'],str)
                rerank_calls+=1
            else:
                assert event.get('model_tool_input_error',False) and not event['reranker_success']
            if event['reranker_success']:
                assert event['num_reranked_docs']==5
                rerank_success+=1
        rows.append(row)
    assert summarize(rows)==complete['metrics']
    assert len(complete['metrics']['datasets'])==7
    # Invalid learned rerank formatting is a native model outcome, not a failed
    # infrastructure run; its retrieval fallback must remain in the denominator.
    assert rerank_calls>0
    for role in ['main','reranker']:
        proof=json.loads((OUT/('export-%s-verified.json'%role)).read_text())
        assert proof['checkpoint']==200 and proof['all_exported_tensors_equal']
        for name,digest in proof['files'].items():
            assert sha(OUT/'models'/role/name)==digest
    job=(OUT/'job.id').read_text().strip()
    accounting=subprocess.check_output(['sacct','-j',job,'-n','-P',
                                        '--format=JobID,State,ExitCode'],text=True)
    assert job+'|COMPLETED|0:0' in accounting
    cleanup=json.loads((OUT/('cleanup-'+job+'.json')).read_text())
    assert cleanup['all_owned_children_reaped'] and cleanup['only_exact_owned_child_process_groups']
    result=dict(status='COSEARCH_DUAL_MODEL_FINAL51713_INDEPENDENT_ACCEPTANCE_PASS',
                epoch=time.time(),training_job=713379,evaluation_job=int(job),
                checkpoint_step=200,rows=N,metrics=complete['metrics'],
                reranker_calls=rerank_calls,reranker_success=rerank_success,
                full_question_identity_and_independent_EM_F1_pass=True,
                both_exported_models_readback_and_SHA_pass=True,
                owned_cleanup_pass=True,completion_sha256=sha(OUT/'complete.json'))
    atomic(OUT/'acceptance.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
