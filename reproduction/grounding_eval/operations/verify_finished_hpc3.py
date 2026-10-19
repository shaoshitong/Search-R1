"""CPU full-read acceptance of preserved normal + diagnostic-v4 CF, no GPU."""
import argparse,os,sys,json,pathlib,collections,time,importlib
p=argparse.ArgumentParser(__doc__);p.add_argument('--method-root',required=True);p.add_argument('--runtime',required=True);p.add_argument('--output',required=True);a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,a.runtime)
from reproduction.grounding_eval.core import read,sha,digest,canonical,read_jsonl,aggregate,messages
from reproduction.grounding_eval.identity import validate_normal_identity
from reproduction.grounding_eval.__main__ import validated,attempt_token_usage
from search_r1.qa_protocol import extract_final_answer
from transformers import AutoTokenizer
import pyarrow.parquet as pq
root=pathlib.Path(a.method_root);job=read(root/'pipeline-job.json');n=pathlib.Path(job['normal_root']);c=pathlib.Path(job['cf_output']);pipe=pathlib.Path(job['pipeline_output'])
assert read(pipe/'complete.json')['exit_code']==0 and not(pipe/'failed.json').exists()
for phase in ['normal','prepare-cf','counterfactual','verify']:assert read(pipe/(phase+'-exit.json'))['exit_code']==0
complete=read(n/'normal-complete.json');assert complete['rows']==3610 and sha(n/'normal.jsonl')==complete['normal_sha256']
rows=list(read_jsonl(n/'normal.jsonl'));identity=validate_normal_identity(rows,job['checkpoint'],job['normal_config'])
assert identity['count']==3610 and not identity['pilot_only']
binding=read(n/'normal-binding.json');tok=AutoTokenizer.from_pretrained(job['checkpoint'],local_files_only=True)
data=pq.read_table(binding['job']['data_path']).to_pylist();totals=collections.Counter();records=[];sizes=[];raw_bytes=0
for file in sorted((n/'normal-batches').glob('batch-*.json')):
 blob=read(file);assert digest(blob['records'])==blob['records_sha256'] and blob['binding_hash']==digest(binding)
 records.extend(dict(r,source_shard_sha256=sha(file))for r in blob['records']);sizes.append(len(blob['records']));raw_bytes+=file.stat().st_size
assert rows==records
for r in rows:
 source=data[r['evaluation_ordinal']];gold=source['reward_model']['ground_truth']['target'];gold=[gold]if isinstance(gold,str)else gold
 assert r['gold']==gold and source['data_source']=='nq' and r['qid']==str(source['extra_info']['index'])
 assert r['question']==[v['content']for v in source['prompt']if v['role']=='user'][0].rsplit('\nQuestion:',1)[1].strip()
 requests=r['llm_requests'];assert requests and all(t['input_tokens']>0 and 0<=t['output_tokens']<=500 for t in requests)
 prompt=tok.apply_chat_template(source['prompt'],add_generation_prompt=True,tokenize=False);ids=tok.encode(prompt,add_special_tokens=False)[-2048:]
 assert r['initial_prompt_tokens']==requests[0]['input_tokens']==len(ids)
 assert r['model_generated_tokens']==sum(r['model_token_mask']) and len(r['model_token_mask'])==len(r['response_token_ids'])
 assert r['raw_generated_tokens']==r['inference_generated_tokens']==sum(t['output_tokens']for t in requests)
 assert r['cumulative_model_input_tokens']==r['cumulative_input_tokens']==sum(t['input_tokens']for t in requests)
 assert r['observation_tokens']==sum(len(t)for t in r['seen_observation_token_ids']) and all(len(t)<=500 for t in r['seen_observation_token_ids'])
 assert r['actual_retrieval_calls']==len(r['seen_docs'])==len(r['seen_observation_token_ids'])
 assert r['trajectory_total_tokens']==r['initial_prompt_tokens']+r['model_generated_tokens']+r['observation_tokens']
 for k in ['model_generated_tokens','raw_generated_tokens','observation_tokens','initial_prompt_tokens','trajectory_total_tokens','cumulative_model_input_tokens']:totals[k]+=r[k]
manifest,prepared=validated(c/'manifest.json');assert manifest['normal_identity']==identity
assert {v['name']:v['sha256']for v in manifest['checkpoint']['files']}==identity['verified_checkpoint_files']
results=[read(f)for f in sorted((c/'results').glob('*.json'))];ready={r['request_id']:r for r in prepared if r['status']=='READY'};assert len(results)==len(ready)
for r in results:
 q=ready[r['request_id']];assert r['cf_answer']==extract_final_answer([r['cf_response']])
 ids=tok.apply_chat_template(messages(q),tokenize=True,add_generation_prompt=True)
 assert r['model_input_tokens']==len(ids) and r['prompt_token_ids_sha256']==digest(ids)
 assert r['trajectory_total_tokens']==r['model_input_tokens']+r['generation_tokens']
summary=aggregate(prepared,results);assert summary['status']=='complete' and summary['counts']['normal_questions']==3610
prior=read(c/'verified-summary.json')
for key in ['counts','coverage','TC','D_Inv','TC_denominator','D_Inv_denominator']:assert summary[key]==prior[key]
attempts=attempt_token_usage(c);assert attempts['errors']==0 and attempts['failed_or_interrupted_attempts']==0 and attempts['request_attempts']==len(results)
assert attempts['successful_attempt_output_tokens']==sum(r['generation_tokens']for r in results)
file_hashes={f.name:sha(f)for f in sorted((c/'results').glob('*.json'))}
report=dict(state='INDEPENDENT_FULL_READ_VERIFIED',at=time.time(),method=job['method'],normal_count=len(rows),normal_raw_bytes=raw_bytes,normal_sha256=sha(n/'normal.jsonl'),normal_identity=identity,normal_token_totals=dict(totals),actual_normal_batch_sizes=sizes,normal_token_capture_limit='Historical adapter persisted request counts, not request token IDs; first prompt identity and all retained masks/observations/count arithmetic verified; no fabricated token-ID readback',cf_summary=summary,cf_attempts=attempts,cf_results=len(results),cf_result_bytes=sum(f.stat().st_size for f in(c/'results').glob('*.json')),cf_results_hash=digest(file_hashes),cf_manifest_sha256=sha(c/'manifest.json'),cf_prepared_sha256=sha(c/'prepared.jsonl'),cf_prompt_token_hashes_reconstructed=True,all_checkpoint_files_fresh_sha_verified=True,scope='Diagnostic typed-v4; not corrected strict or full official reproduction',pipeline_complete=read(pipe/'complete.json')['completed_at'])
out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x')as f:json.dump(report,f);f.flush();os.fsync(f.fileno())
print(json.dumps(report))
