"""Adapt the immutable DynRPO iter200 native NQ trajectories to the frozen CF schema.

Only CPU reads and task-local writes. Model actions come from native policy masks,
observations from actual truncated tool token IDs; no parsing of mixed tool text.
"""
import argparse,gzip,json,os,re,time
from collections import Counter
from pathlib import Path
from reproduction.grounding_eval.core import canonical,digest,sha,atomic,normalize
from reproduction.grounding_eval.__main__ import checkpoint_inventory
from reproduction.grounding_eval.operations.dynrpo_nq_token_audit import validate_trajectory,RUN,ACTOR,DATA_SHA,ORDER_SHA,SOURCE_COMPLETE_SHA
from search_r1.qa_protocol import extract_final_answer

def decode_parts(row,tokenizer):
    ids=row['response_token_ids'];mask=row['response_info_mask']
    assert len(ids)==len(mask) and all(v in (0,1) for v in mask)
    spans=[];active=[]
    for token,on in zip(ids,mask):
        if on:active.append(token)
        elif active:spans.append(tokenizer.decode(active,skip_special_tokens=False));active=[]
    if active:spans.append(tokenizer.decode(active,skip_special_tokens=False))
    docs=[]
    for t in row['behavior_turns']:
        if not t['observation_token_ids']:continue
        assert t['is_search']==1 and not t['correction_token_ids']
        text=tokenizer.decode(t['observation_token_ids'],skip_special_tokens=False)
        assert '<information>' in text and '</information>' in text
        docs.append(text.split('<information>',1)[1].split('</information>',1)[0])
    assert len(docs)==row['actual_retrieval_calls']
    return spans,docs

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--base',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    b,data,out=Path(a.base),Path(a.data),Path(a.output)
    assert not out.exists();out.mkdir(parents=True)
    c=b/'final-evaluation/complete.json';assert sha(c)==SOURCE_COMPLETE_SHA
    done=json.loads(c.read_text());assert done['run_id']==RUN and done['actor_version']==ACTOR and done['qa_count']==51713
    assert all(v==0 for v in done['actual_worker_exit_codes'].values()) and sha(data)==DATA_SHA
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    questions=pq.read_table(data).to_pylist();assert len(questions)==51713
    model=Path(done['actor_model_dir']);inventory=checkpoint_inventory(model)
    files={x['name']:dict(size=x['bytes'],sha256=x['sha256']) for x in inventory['files']}
    binding=dict(schema_version=1,adapter='dynrpo-native-final-token-mask-v1',method='DynRPO',run_id=RUN,actor_version=ACTOR,
      job=dict(data_path=str(data),checkpoint_path=str(model)),checkpoint_files=files,checkpoint_hash=digest(files),data_sha256=DATA_SHA,
      normal_protocol=done['evaluation_contract'],source_complete_sha256=sha(c),normal_inference_rerun=False)
    atomic(out/'normal-binding.json',binding);atomic(out/'normal-config.json',binding)
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    records={};source_proofs=[];request_ids=set()
    for rank,rr in enumerate(done['rank_receipts']):
        receipt_path=Path(rr['path']);assert sha(receipt_path)==rr['sha256'];receipt=json.loads(receipt_path.read_text())
        full=Path(receipt['full_output']['path']);fullsha=sha(full);assert fullsha==receipt['full_output']['sha256']==done['output_shards'][rank]['sha256']
        count=0
        with gzip.open(full,'rt') as stream:
            for line in stream:
                row=json.loads(line)
                if row['data_source']!='nq':continue
                n=row['evaluation_ordinal'];assert n%4==rank and n not in records
                question=questions[n];assert question['data_source']=='nq'
                validate_trajectory(row,ACTOR,request_ids)
                prompt=tokenizer.apply_chat_template(question['prompt'],add_generation_prompt=True,tokenize=False)
                assert tokenizer.encode(prompt,add_special_tokens=False)[-2048:]==row['initial_prompt_token_ids']
                text='\n'.join(x['content'] for x in question['prompt']);matches=list(re.finditer(r'Question:\s*',text));q=text[matches[-1].end():].strip()
                gold=question['reward_model']['ground_truth']['target'];gold=[gold] if isinstance(gold,str) else list(gold)
                spans,docs=decode_parts(row,tokenizer);answer=extract_final_answer(spans)
                em=int(bool(answer) and normalize(answer) in {normalize(g) for g in gold});assert em==row['em']
                r=dict(data_source='nq',evaluation_ordinal=n,qid=str(question['extra_info']['index']),native_qid=row['qid'],question=q,gold=gold,
                  model_spans=spans,seen_docs=docs,actual_retrieval_calls=row['actual_retrieval_calls'],source_run_id=RUN,actor_version=ACTOR,
                  checkpoint_hash=binding['checkpoint_hash'],normal_config_hash=digest(binding),original_native_em=em,
                  native_response_token_ids_sha256=digest(row['response_token_ids']),native_policy_mask_sha256=digest(row['response_info_mask']),
                  native_observation_token_ids_sha256=digest([t['observation_token_ids'] for t in row['behavior_turns']]))
                r['source_record_hash']=digest(r);r['source_shard_sha256']=fullsha;records[n]=r;count+=1
        source_proofs.append(dict(rank=rank,source=str(full),sha256=fullsha,nq_rows=count))
        atomic(out/'progress.json',dict(phase='EXPORT_NATIVE_NQ',completed_ranks=rank+1,nq_rows=len(records),checked_unix=time.time()),replace=True)
    assert sorted(records)==list(range(3610));rows=[records[n] for n in range(3610)]
    order=[[r['data_source'],r['evaluation_ordinal'],r['question'],sorted(r['gold'])] for r in rows];assert digest(order)==ORDER_SHA
    assert sum(r['original_native_em'] for r in rows)==1706
    with (out/'normal.jsonl').open('xb') as stream:
        for r in rows:stream.write(canonical(r)+b'\n')
        stream.flush();os.fsync(stream.fileno())
    atomic(out/'complete.json',dict(status='DYNRPO_NATIVE_NQ_EXPORT_COMPLETE',count=3610,correct=1706,canonical_question_gold_order_sha256=ORDER_SHA,
      source_proofs=source_proofs,normal_sha256=sha(out/'normal.jsonl'),checkpoint_hash=inventory['checkpoint_hash'],model=str(model),
      normal_inference_rerun=False,GPU_allocations=0,checked_unix=time.time()))

if __name__=='__main__':main()
