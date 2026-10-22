from pathlib import Path
import json,time,gzip,hashlib,sys
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');sys.path.insert(0,str(B/'repo'))
from transformers import AutoTokenizer
from search_g1.signals import decode_segments,contexts
from verl.utils.reward_score.qa_em import em_check,normalize_answer
MODEL=B/'assets/qwen25-7b-instruct-a09a35458c70';tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
root=B/'peer-newseed-h100-20261021/audited138-after971-20261021-51057f37/raw/audited138-after971-20261021-51057f37/group-00006/source';raw=root/'sampling/shard-01/batches/batch-000420/outers/outer-000001/full/rank-0-full.jsonl.gz';wanted={'70188b8e50fd4b45bc420ce58014f46e-outer000001-c107521-g3','70188b8e50fd4b45bc420ce58014f46e-outer000001-c107529-g3','70188b8e50fd4b45bc420ce58014f46e-outer000001-c107543-g2'};result=[]
with gzip.open(raw,'rt')as f:
 for line in f:
  row=json.loads(line)
  if row['trajectory_id']not in wanted:continue
  text,authored=decode_segments(tokenizer,row['response_token_ids'],row['response_info_mask']);context=contexts('',text,authored);target=row['reward_model']['ground_truth']['target'];pred=context['answer']if context else None;score=int(context is not None and em_check(pred,target));gold=[target]if isinstance(target,str)else target
  result.append(dict(qid=row['qid'],trajectory_id=row['trajectory_id'],raw_strict_em=row['strict_em'],independent_current_strict_em=score,answer=pred,normalized_answer=normalize_answer(pred)if pred else None,ground_truth=gold,normalized_ground_truth=[normalize_answer(x)for x in gold],last_model_authored_excerpt=text[authored[-1][0]:authored[-1][1]][-650:]if authored else None,authored_segments=len(authored),response_tokens=len(row['response_token_ids']),strict_answer_from_last_complete_model_authored_segment_only=True))
assert len(result)==3
sources={}
for p in(B/'repo/search_g1/signals.py',B/'repo/search_r1/qa_protocol.py',B/'repo/verl/utils/reward_score/qa_em.py',MODEL/'tokenizer.json',MODEL/'tokenizer_config.json'):
 data=p.read_bytes();sources[str(p)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
print(json.dumps(dict(time=time.time(),status='THREE_RAW_STRICT_EM_RECOMPUTED_WITH_OFFICIAL_TOKENIZER_NO_MODEL',cases=result,sources=sources,final4096_read=False,model_or_checkpoint_loaded=False,label_or_training_modified=False)))
