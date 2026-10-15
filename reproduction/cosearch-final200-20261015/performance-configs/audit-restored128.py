"""Independent CPU audit after preserving all rows across measured performance trials."""
import collections,hashlib,json,pathlib,time,sys
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'tools'))
from common import OUT,TRAIN,atomic,sha
from common_em import score,identity
def main():
 intent=json.loads((OUT/'performance-handoff-intent.json').read_text())
 binding=json.loads((OUT/'binding.json').read_text())
 binding_sha=sha(OUT/'binding.json')
 assert binding_sha==intent['producer_binding_sha256']
 assert sha(OUT/'source-manifest.json')==intent['manifest_sha256']
 assert sha(TRAIN/'data/final.jsonl')==binding['data_jsonl_sha256']
 refs=[json.loads(x) for x in (TRAIN/'data/final.jsonl').open() if x.strip()]
 profiles=collections.Counter();datasets=collections.Counter();rr_calls=rr_success=0;count=0
 for path in sorted((OUT/'rows').glob('*.json')):
  row=json.loads(path.read_text());i=row['row_id'];ref=refs[i]
  assert path.name=='%05d.json'%i
  assert row['question_uid']==ref['question_id']
  assert row['binding_sha256']==binding_sha and row['inference_received_gold'] is False
  assert row['question_answer_sha256']==identity(ref['question'],ref['answers'])
  last=row['model_actions'][-1] if row['model_actions'] else None
  if last:
   assert last['author']=='main_model'
   if last['fully_retained']:
    start,end=last['response_token_start'],last['response_token_end']
    assert end<=len(row['response_mask'])
    assert all(x==1 for x in row['response_mask'][start:end])
  expected=score(last['text'] if last else None,ref['answers'],
   boundary_verified=bool(last and last['fully_retained']))
  assert all(row[k]==v for k,v in expected.items()),(i,'score')
  for event in row['reranker_events']:
   if event['reranker_prompt_tokens']>0:
    assert event['num_retrieved_docs']==50
    assert event['reranker_prompt_tokens']<=16384
    rr_calls+=1
   if event['reranker_success']:
    assert event['num_reranked_docs']==5;rr_success+=1
  profiles[row['question_concurrency']]+=1;datasets[row['data_source']]+=1;count+=1
 previous=json.loads((OUT/'performance-handoff-history/concurrency256-handoff-intent.json').read_text())
 assert profiles[128]>=previous['accepted_rows']
 assert profiles[256]==intent['accepted_rows']-previous['accepted_rows']
 progress=json.loads((OUT/'progress.json').read_text())
 assert progress['question_concurrency']==128 and count>intent['accepted_rows']
 startup=json.loads((OUT/'startup-verified.json').read_text())
 assert startup['first']['question_concurrency']==128
 assert startup['first']['epoch']>intent['epoch']
 assert startup['second']['rows']>startup['first']['rows']
 assert startup['second']['main_calls']>startup['first']['main_calls']
 assert startup['second']['reranker_calls']>startup['first']['reranker_calls']
 samples=[json.loads(x) for x in (OUT/'utilization.jsonl').read_text().splitlines()]
 samples=[s for s in samples if s['epoch']>=startup['second']['epoch']]
 gpu=collections.defaultdict(list)
 for sample in samples[-20:]:
  for line in sample['gpus'].splitlines():
   fields=line.split(',');gpu[fields[0]].append(int(fields[1]))
 failures=[json.loads(p.read_text()) for p in (OUT/'failures').glob('*.json')]
 accepted_errors=sum((OUT/('rows/%05d.json'%x['row_id'])).exists() for x in failures)
 result=dict(status='RESTORED128_ALL_ACCEPTED_ROWS_AND_DUAL_MODEL_PROGRESS_AUDIT_PASS',
  epoch=time.time(),job=721542,rows=count,profiles=dict(profiles),datasets=dict(datasets),
  previous_accepted_rows_preserved=True,immutable_model_and_protocol_binding_pass=True,
  independent_saved_row_score_pass=True,real_reranker_calls=rr_calls,reranker_success=rr_success,
  request_retry_events=len(failures),retry_events_with_accepted_rows=accepted_errors,
  startup_two_samples=startup,latest_progress=progress,gpu_window_samples=min(len(samples),20),
  gpu_window_mean={k:sum(v)/len(v) for k,v in gpu.items()},metrics_final=False)
 atomic(OUT/'performance-restored128-acceptance.json',result);print(json.dumps(result))
if __name__=='__main__':main()
