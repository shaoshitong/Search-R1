"""Small manual-review audit: raw docs/full trajectories stay remote."""
import argparse,os,sys,pathlib,json,time,collections
p=argparse.ArgumentParser(__doc__);p.add_argument('--runtime',required=True);p.add_argument('--cf-root',required=True);p.add_argument('--control-output',required=True);p.add_argument('--output',required=True);a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,a.runtime)
from reproduction.grounding_eval.strict_v5 import validated,result_rows,attempt_token_usage
from reproduction.grounding_eval.core_v5 import prepare_row,classify,messages,digest,sha
from search_r1.qa_protocol import extract_final_answer
from transformers import AutoTokenizer
cf=pathlib.Path(a.cf_root);control=pathlib.Path(a.control_output);complete=json.loads((control/'complete.json').read_text());assert complete['state']=='PILOT_COMPLETE_AWAITING_MANUAL_REVIEW'and complete['results']==16 and not complete['full_run_authorized']
m,prepared=validated(cf/'manifest.json');assert m['protocol_version']=='strict-v5-conservative-typed'and m['cf_config']['replacement_version']=='raw-typed-conservative-v5'
ready=[p for p in prepared if p['status']=='READY'];results=result_rows(cf);byid={r['request_id']:r for r in results};assert len(results)==16 and set(byid)=={p['request_id']for p in ready[:16]}
source={r['evaluation_ordinal']:r for r in map(json.loads,pathlib.Path(m['source']['path']).read_text().splitlines())};tok=AutoTokenizer.from_pretrained(m['checkpoint']['path'],local_files_only=True)
rows=[]
for p in ready[:16]:
 r=byid[p['request_id']];assert prepare_row(source[p['evaluation_ordinal']],m)==p
 assert r['cf_answer']==extract_final_answer([r['cf_response']])and r['classification']==classify(p['original_answer'],r['cf_answer'],p['cf_aliases'],p['gold'])
 ids=tok.apply_chat_template(messages(p),tokenize=True,add_generation_prompt=True);assert digest(ids)==r['prompt_token_ids_sha256']and len(ids)==r['model_input_tokens']
 assert r['generation_tokens']+r['model_input_tokens']==r['trajectory_total_tokens']
 mappings=p['replacement_mapping'];perdoc=collections.defaultdict(list)
 for v in mappings:perdoc[v['doc_index']].append(v)
 match=True;onlymapped=True
 for i,original in enumerate(p['seen_docs']):
  tail=0;parts=[]
  for v in sorted(perdoc[i],key=lambda x:x['start']):
   match=match and original[v['start']:v['end']]==v['original'];parts.extend([original[tail:v['start']],v['injected']]);tail=v['end']
  parts.append(original[tail:]);onlymapped=onlymapped and ''.join(parts)==p['modified_docs'][i]
 assert match and onlymapped and mappings
 ans=r['cf_answer'];question=p['question'];assert len(question)<=500 and all(len(v)<=350 for v in p['gold'])
 rows.append(dict(ordinal=p['evaluation_ordinal'],qid=p['qid'],question=question,gold=p['gold'],original_answer=p['original_answer'],replacement_type=p['replacement_kind'],mapping=[{k:v[k]for k in ['doc_index','start','end','original','injected']}for v in mappings],cf_aliases=p['cf_aliases'],classification=r['classification'],cf_answer=(ans[:300]if ans else ans),cf_answer_truncated=bool(ans and len(ans)>300),finish_reason=r['finish_reason'],stop_reason=r['stop_reason'],generation_tokens=r['generation_tokens'],model_input_tokens=r['model_input_tokens'],document_checks_remote_only=dict(all_original_spans_match=match,only_declared_spans_changed=onlymapped,reprepare_identical=True,source_documents_present=len(p['seen_docs'])>0)))
usage=attempt_token_usage(cf);assert usage['request_attempts']==16 and usage['errors']==0 and usage['failed_or_interrupted_attempts']==0
report=dict(state='PILOT16_COMPLETE_AWAITING_ROOT_MANUAL_REVIEW',at=time.time(),method=m['method'],protocol=m['protocol_version'],replacement_version=m['cf_config']['replacement_version'],source_commit='16c36aa777f684f245b1d38cb2939ef175131831',manifest_sha256=sha(cf/'manifest.json'),prepared_sha256=sha(cf/'prepared.jsonl'),normal_source_sha256=m['source']['sha256'],full_run_started=False,results=16,prepared_total=len(prepared),prepared_status_counts=dict(collections.Counter(p['status']for p in prepared)),pilot_counts=dict(collections.Counter(r['classification']for r in results)),token_usage=usage,rows=rows,privacy='No documents or full trajectories exported; only short QA/mappings and remote boolean checks')
raw=json.dumps(report,ensure_ascii=False,indent=2).encode();assert len(raw)<=40000,('Small audit exceeds40KB',len(raw));out=pathlib.Path(a.output)
with out.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(path=str(out),bytes=len(raw),sha256=sha(out),state=report['state'],counts=report['pilot_counts'])))
