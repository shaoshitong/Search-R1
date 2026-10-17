"""Reproduce the saved-raw CPU QA acceptance; never start generation or a model."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import time


def verify(base,expected_count=512):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from .actor import read,sha,remote
    from .eval import launch,load_questions,score_tokens
    from transformers import AutoTokenizer
    base=remote(base);c=read(base/'preparation/config.json');spec=c['qa_standalone'];output=base/'validation'
    if not (output/'complete.json').is_file() or not all((output/('rank-%d-complete.json'%r)).is_file()for r in range(8)):
        raise ValueError('CPU acceptance requires all existing complete receipts; never generate missing QA')
    questions_path=c['validation_questions'] if expected_count==512 else c['final_questions']
    result=launch(base/'preparation/config.json',spec['model_dir'],output,questions_path,expected_count)
    questions=load_questions(questions_path,expected_count)
    tokenizer=AutoTokenizer.from_pretrained(spec['model_dir'],local_files_only=True,trust_remote_code=False)
    ids=set();correct=0;f1=0.;response_tokens=0;retrieval_calls=0;rank_rows=[]
    for rank in range(8):
        receipt=read(output/('rank-%d-complete.json'%rank));full=output/('rank-%d-full.jsonl.gz'%rank)
        if sha(full)!=receipt['full_output']['sha256']:raise ValueError('Retained QA shard hash changed')
        rows={r['evaluation_ordinal']:r for r in receipt['rows']}
        with gzip.open(full,'rt')as stream:
            for line in stream:
                row=json.loads(line);ordinal=row['evaluation_ordinal'];q=questions[ordinal]
                if (ordinal%8!=rank or ordinal in ids or row['qid']!=q['qid'] or row['data_source']!=q['data_source']):
                    raise ValueError('Canonical QA identity was duplicated, omitted or moved')
                score=score_tokens(tokenizer,row['initial_prompt_token_ids'],row['response_token_ids'],
                    row['response_info_mask'],q['reward_model']['ground_truth']['target'])
                if not all(score[k]==row[k]==rows[ordinal][k]for k in score):raise ValueError('Saved score differs from strict raw recompute')
                if not len(row['response_token_ids'])==len(row['response_info_mask'])==len(row['response_log_probs']):
                    raise ValueError('Saved QA token/mask/probability lengths differ')
                if not all(math.isfinite(v)for v in row['response_log_probs']):raise ValueError('Nonfinite native QA probability')
                ids.add(ordinal);correct+=score['em'];f1+=score['f1']
                response_tokens+=row['response_token_count'];retrieval_calls+=row['actual_retrieval_calls']
        rank_rows.append(dict(rank=rank,count=len(rows),gpu_uuid=receipt['device_binding']['gpu_uuid'],
            compute_dtype=receipt['actual_compute_dtype'],elapsed_seconds=receipt['elapsed_seconds'],
            full_sha256=sha(full),receipt_sha256=sha(output/('rank-%d-complete.json'%rank))))
    if ids!=set(range(expected_count)) or correct/expected_count!=result['em'] or abs(f1/expected_count-result['f1'])>1e-12:
        raise ValueError('Aggregate differs from full canonical raw recompute')
    return dict(status='PASS_CPU_RECOMPUTE_CANONICAL%d_ALL8_RAW_SHARDS'%expected_count,qa_count=len(ids),
        correct=correct,em=correct/expected_count,f1=f1/expected_count,run_id=c['run_id'],actor_version=c['actor_version'],
        question_source_sha256=sha(questions_path),config_sha256=sha(base/'preparation/config.json'),
        qa_receipt_sha256=sha(output/'complete.json'),rank_rows=rank_rows,
        actual_response_tokens_including_observations=response_tokens,logical_retrieval_calls=retrieval_calls,
        model_weights_loaded=False,tokenizer_reloaded=True,new_generation=False,observed_unix=time.time())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--expected-count',type=int,choices=(512,51713),default=512)
    a=p.parse_args();print(json.dumps(verify(a.base,a.expected_count)))
