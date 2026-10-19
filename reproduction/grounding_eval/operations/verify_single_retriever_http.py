# Remote CPU controller / real historical query validation, not new evaluated questions.
import pathlib,json,sys,time,hashlib,urllib.request,os
os.environ['CUDA_VISIBLE_DEVICES']=''
repo='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-54ac33dcebe64e62a11fe4177f41f116-resume-40389d9b00204a80ba68475fac254f9c/repo';sys.path.insert(0,repo)
from verl.utils import hf_tokenizer
from search_r1.llm_agent.generation import LLMGenerationManager
from types import SimpleNamespace
root=pathlib.Path('/data/shaoshitong/grounding-eval-20261019');out=root/'retriever-owned-single-bounded-56df9a1d';tok=hf_tokenizer('/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512/checkpoints/search-g1-common8/actor/global_step_200');manager=LLMGenerationManager.__new__(LLMGenerationManager);manager.tokenizer=tok;manager.config=SimpleNamespace(max_obs_length=500)
selected=[]
for p in sorted((root/'search-g1-normal/records').glob('*.json')):
 row=json.loads(p.read_text())
 for turn in row['behavior_turns']:
  if not turn['observation_token_ids']:continue
  text=tok.decode(turn['raw_generated_token_ids'][:turn['kept_token_count']],skip_special_tokens=True,clean_up_tokenization_spaces=False);actions,contents=manager.postprocess_predictions([text]);assert actions==['search'];selected.append((contents[0],turn['observation_token_ids']))
  if len(selected)==16:break
 if len(selected)==16:break
assert len(selected)==16 and json.loads((out/'parity.json').read_text())['all_equal']
def health():
 with urllib.request.urlopen('http://127.0.0.1:19001/health',timeout=4)as r:return json.loads(r.read())
before=health();assert before['pid']==1151202 and before['run_id']=='grounding-10b412b7c61a4e4cbc5f635fe53f5fa8'
observations=[]
for attempt in range(2):
 req=urllib.request.Request('http://127.0.0.1:19001/retrieve',data=json.dumps(dict(queries=[x[0]for x in selected],topk=3,return_scores=True)).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=40)as r:payload=json.loads(r.read())['result']
 texts=['\n\n<information>'+manager._passages2string(p).strip()+'</information>\n\n'for p in payload]
 ids=manager._process_next_obs(texts).detach().cpu().tolist();ids=[[i for i in v if i!=tok.pad_token_id]for v in ids]
 assert ids==[x[1]for x in selected]
 h=health();assert h['actual_http_requests']>=before['actual_http_requests']+attempt+1
 observations.append(dict(at=time.time(),actual_http_requests=h['actual_http_requests'],actual_queries=h['actual_queries'],cropped_token_parity_count=16,all_equal=True))
report=dict(state='TWO_REAL_QUERY_HTTP_BATCHES_VERIFIED',at=time.time(),pid=1151202,birth=648183650,run_id=before['run_id'],endpoint=before['endpoint'],gpu_shards=1,source_queries='same16historical-normal-queries, validation only not new evaluation occurrences',observations=observations,source_corpus=before['corpus'],new_model_inference_questions=0)
with(out/'independent-http-validation.json').open('x')as f:json.dump(report,f);f.flush();os.fsync(f.fileno())
print(json.dumps(report))
