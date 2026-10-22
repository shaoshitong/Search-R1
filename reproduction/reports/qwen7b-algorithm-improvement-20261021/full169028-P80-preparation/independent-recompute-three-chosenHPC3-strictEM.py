from pathlib import Path
import json,time,gzip,hashlib,sys
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');sys.path.insert(0,str(B/'repo'))
from transformers import AutoTokenizer
from search_g1.signals import decode_segments,contexts
from verl.utils.reward_score.qa_em import em_check,normalize_answer
MODEL=B/'assets/qwen25-7b-instruct-a09a35458c70';tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
A=B/'algorithm-improvement-20261021';sealpath=A/'hpc3-tail-newseedG5-20261021-v1/sampling/shard-01/batches/batch-000420/complete.json';seal=json.loads(sealpath.read_bytes());fullref=seal['full']['full'];raw=Path(fullref['path']);rawbytes=raw.read_bytes();assert len(rawbytes)==fullref['bytes']and hashlib.sha256(rawbytes).hexdigest()==fullref['sha256'];wanted={'c107521-g3','c107529-g3','c107543-g2'};result=[]
physical_prefix=A/'hpc3-tail-newseedG5-20261021-v1/sampling/shard-01/batches/batch-000420/outers/outer-000001/prefix/rank-0-prefix.jsonl.gz'
donor=B/'peer-newseed-h100-20261021/audited138-after971-20261021-51057f37/raw/audited138-after971-20261021-51057f37/group-00006/source/sampling/shard-01/batches/batch-000420/outers/outer-000001/full/rank-0-full.jsonl.gz'
def policy(row):return[t for turn in row['behavior_turns']for t in turn['raw_generated_token_ids'][:turn['kept_token_count']]]
donorrows={}
with gzip.open(donor,'rt')as stream:
 for line in stream:
  rr=json.loads(line)
  if any(rr['trajectory_id'].endswith(x)for x in wanted):donorrows[rr['qid']]=rr

with gzip.open(raw,'rt')as f:
 for line in f:
  row=json.loads(line)
  if not any(row['trajectory_id'].endswith(x)for x in wanted):continue
  text,authored=decode_segments(tokenizer,row['response_token_ids'],row['response_info_mask']);context=contexts('',text,authored);target=row['reward_model']['ground_truth']['target'];pred=context['answer']if context else None;score=int(context is not None and em_check(pred,target));gold=[target]if isinstance(target,str)else target
  result.append(dict(qid=row['qid'],trajectory_id=row['trajectory_id'],raw_strict_em=row['strict_em'],independent_current_strict_em=score,answer=pred,normalized_answer=normalize_answer(pred)if pred else None,ground_truth=gold,normalized_ground_truth=[normalize_answer(x)for x in gold],last_model_authored_excerpt=text[authored[-1][0]:authored[-1][1]][-650:]if authored else None,authored_segments=len(authored),response_tokens=len(row['response_token_ids']),strict_answer_from_last_complete_model_authored_segment_only=True,physical_run_id=row.get('run_id'),donor_same_tid=donorrows[row['qid']]['trajectory_id']==row['trajectory_id'],first20_policy_equal=policy(donorrows[row['qid']])[:20]==policy(row)[:20],first80_policy_equal=policy(donorrows[row['qid']])[:80]==policy(row)[:80],full_response_ids_equal=donorrows[row['qid']]['response_token_ids']==row['response_token_ids']))
assert len(result)==3
sources={}
for p in(B/'repo/search_g1/signals.py',B/'repo/search_r1/qa_protocol.py',B/'repo/verl/utils/reward_score/qa_em.py',MODEL/'tokenizer.json',MODEL/'tokenizer_config.json'):
 data=p.read_bytes();sources[str(p)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
print(json.dumps(dict(time=time.time(),status='THREE_CHOSEN_HPC3_EXECUTION_STRICT_EM_RECOMPUTED_WITH_OFFICIAL_TOKENIZER_NO_MODEL',chosen_physical_seal=str(sealpath),chosen_physical_full_ref=fullref,source_resolution_contract=json.loads((A/'hpc3-tail9344-old5-qualified-g10-v1/raw-g10/source-resolution-contract.json').read_bytes()),cases=result,sources=sources,final4096_read=False,model_or_checkpoint_loaded=False,label_or_training_modified=False)))
