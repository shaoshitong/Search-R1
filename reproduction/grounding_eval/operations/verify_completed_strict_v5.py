"""Full-read CPU acceptance of an existing strict-v5 CF task; does not generate or reuse scores."""
import argparse,sys,os,pathlib,json,time
p=argparse.ArgumentParser(__doc__);p.add_argument('--manifest',required=True);p.add_argument('--runtime',required=True);p.add_argument('--output',required=True);a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,a.runtime)
from reproduction.grounding_eval.core_v5 import read,sha,digest,read_jsonl,aggregate,messages,prepare_row
from reproduction.grounding_eval.strict_v5 import validated,attempt_token_usage
from reproduction.grounding_eval.identity import validate_normal_identity
from search_r1.qa_protocol import extract_final_answer
from transformers import AutoTokenizer
m,prepared=validated(a.manifest);root=pathlib.Path(m['output']);source=list(read_jsonl(m['source']['path']))
assert m['protocol_version']=='strict-v5-conservative-typed'
identity=validate_normal_identity(source,m['checkpoint']['path'],m['normal_config_path']);assert identity==m['normal_identity']
assert len(source)==len(prepared)==3610
for normal,p in zip(source,prepared):
 regenerated=prepare_row(normal,m);regenerated['prepared_at']=p['prepared_at'];assert regenerated==p
cf_files={v['name']:v['sha256']for v in m['checkpoint']['files']};bound_files=identity['verified_checkpoint_files']
extra_normal_receipts=set(bound_files)-set(cf_files)
assert extra_normal_receipts <= {'asset-binding.json','asset-status.json','complete.json'}
assert set(cf_files)<=set(bound_files) and all(bound_files[k]==v for k,v in cf_files.items())
tok=AutoTokenizer.from_pretrained(m['checkpoint']['path'],local_files_only=True)
files=sorted((root/'results').glob('*.json'));results=[read(f)for f in files];ready={r['request_id']:r for r in prepared if r['status']=='READY'};assert len(results)==len(ready)
for r in results:
 q=ready[r['request_id']];assert r['cf_answer']==extract_final_answer([r['cf_response']])
 ids=tok.apply_chat_template(messages(q),tokenize=True,add_generation_prompt=True)
 assert r['model_input_tokens']==len(ids) and r['prompt_token_ids_sha256']==digest(ids)
 assert r['trajectory_total_tokens']==r['model_input_tokens']+r['generation_tokens']
summary=aggregate(prepared,results);assert summary['status']=='complete' and summary['counts']['normal_questions']==3610
old=read(root/'verified-summary.json')
for k in ['counts','coverage','TC','D_Inv','TC_denominator','D_Inv_denominator']:assert summary[k]==old[k]
attempts=attempt_token_usage(root);assert attempts['errors']==0 and attempts['failed_or_interrupted_attempts']==0 and attempts['request_attempts']==len(results)
assert attempts['successful_attempt_output_tokens']==sum(r['generation_tokens']for r in results)
report=dict(state='INDEPENDENT_CF_FULL_READ_VERIFIED',at=time.time(),method=m['method'],normal_identity=identity,normal_sha256=sha(m['source']['path']),all_checkpoint_files_fresh_sha_verified=True,extra_non_HF_normal_receipts_verified=sorted(extra_normal_receipts),cf_summary=summary,cf_attempts=attempts,cf_results=len(results),cf_result_bytes=sum(f.stat().st_size for f in files),cf_results_hash=digest({f.name:sha(f)for f in files}),cf_manifest_sha256=sha(a.manifest),cf_prepared_sha256=sha(root/'prepared.jsonl'),cf_prompt_token_hashes_reconstructed=True,all_prepared_scientific_fields_recomputed=True,scope='Corrected conservative typed strict-v5 operational evaluation; finite-type conditional coverage, not full official reproduction')
out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x')as f:json.dump(report,f);f.flush();os.fsync(f.fileno())
print(json.dumps(report))
