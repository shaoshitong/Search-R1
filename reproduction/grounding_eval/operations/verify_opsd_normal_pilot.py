"""Verify real normal capture on A800; publish only small audit, never raw rows."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess


REMOTE = r'''
import pathlib,json,hashlib,os,re,time,importlib.util
os.environ['CUDA_VISIBLE_DEVICES']=''
import pandas as pd
from transformers import AutoTokenizer
payload=json.loads(INPUT);phase=payload.get('phase','pilot96');target=96 if phase=='pilot96'else 3610
root=pathlib.Path('/data/shaoshitong/grounding-eval-20261019')/('opsd3304-normal-pilot96-v4'if phase=='pilot96'else'opsd3304-normal-v4')
expected=payload['files'];binding=json.loads((root/'normal-binding.json').read_text())
canonical=lambda v:json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
digest=lambda v:hashlib.sha256(canonical(v)).hexdigest()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
complete=json.loads((root/'complete.json').read_text()) if (root/'complete.json').exists()else None
if complete:
 assert json.loads((root/'control/exit.json').read_text())['exit_code']==0
 assert complete['completed']==complete['target']==target
else:assert phase=='full3610' and payload.get('allow_partial') is True
assert binding['normal_protocol']['force_final_answer'] is False and binding['data_sha256']=='55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1'
assert binding['retrieval_provenance']['actual_shards']==2 and binding['retrieval_provenance']['original_shards']==8 and binding['retrieval_provenance']['original_opsd_search_token_parity_verified'] is False
for name,x in expected.items():assert binding['checkpoint_files'][name]==x['sha256'] and (pathlib.Path(binding['checkpoint'])/name).stat().st_size==x['size']
assert sha(binding['data_path'])==binding['data_sha256']
frame=pd.read_parquet(binding['data_path']);assert len(frame)==51713 and sum(frame.data_source=='nq')==3610
tok=AutoTokenizer.from_pretrained(binding['checkpoint'],local_files_only=True)
rows=[];batch_sizes=[];checked_raw_bytes=0
for p in sorted((root/'normalized').glob('*.json')):
 d=json.loads(p.read_text());raw=root/'raw'/p.name
 assert d['binding_sha256']==digest(binding) and d['raw_sha256']==sha(raw)
 original=json.loads(raw.read_text());assert original['binding_sha256']==digest(binding)
 assert [dict(r,source_shard_sha256=d['raw_sha256'])for r in original['rows']]==d['rows']
 rows.extend(d['rows']);batch_sizes.append(len(d['rows']));checked_raw_bytes+=raw.stat().st_size
assert rows and [r['evaluation_ordinal']for r in rows]==list(range(len(rows)))
expected_sizes=[];offset=0;size=32 if phase=='pilot96'else 128;maximum=64 if phase=='pilot96'else 256
while offset<len(rows):
 count=min(size,target-offset);expected_sizes.append(count);offset+=count;size=min(maximum,size*2)
assert offset==len(rows) and batch_sizes==expected_sizes
if complete:assert len(rows)==target
totals={};searches=turns=0
def plain(v):
 if hasattr(v,'tolist'):return plain(v.tolist())
 if isinstance(v,dict):return {k:plain(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [plain(x)for x in v]
 return v
for r in rows:
 source=frame.iloc[r['evaluation_ordinal']];assert source.data_source=='nq' and r['qid']==str(source.extra_info['index'])
 gold=plain(source.reward_model)['ground_truth']['target'];gold=[gold]if isinstance(gold,str)else gold;assert r['gold']==gold
 chat=plain(source['prompt']);assert r['original_prompt']==chat
 text=tok.apply_chat_template(chat,add_generation_prompt=True,tokenize=False) if tok.chat_template else chat[0]['content']
 ids=tok.encode(text,add_special_tokens=False)[-binding['normal_protocol']['max_start_length']:]
 assert r['initial_prompt_token_ids']==ids and r['turns'][0]['generation_context_token_ids']==ids
 prompt_text='\n'.join(t['content']for t in chat);matches=list(re.finditer(r'Question:\s*',prompt_text));assert r['question']==prompt_text[matches[-1].end():].strip()
 kept=raw_tokens=inputs=obs=correction=0
 for t in r['turns']:
  n=len(t['generation_context_token_ids']);g=len(t['raw_response_token_ids']);k=len(t['postprocessed_action_token_ids']);o=len(t['retained_observation_token_ids'])
  assert t['actual_input_tokens']==n and t['actual_generated_tokens']==g and t['retained_model_tokens']==k and t['retained_environment_tokens']==o
  assert o<=500 and g<=500 and not t['final_only']
  kept+=k;raw_tokens+=g;inputs+=n;turns+=1
  if t['actual_search']:obs+=o;searches+=1
  else:correction+=o
 assert r['model_generated_tokens']==kept and r['raw_generated_tokens']==raw_tokens and r['observation_tokens']==obs and r['environment_correction_tokens']==correction
 assert r['initial_prompt_tokens']==len(ids) and r['cumulative_model_input_tokens']==inputs
 assert r['trajectory_total_tokens']==len(ids)+kept+obs+correction
 assert sum(r['model_mask'])==kept and len(r['response_token_ids'])==len(r['model_mask'])
 assert r['actual_retrieval_calls']==sum(t['actual_search']for t in r['turns'])==len(r['seen_docs'])
 for key in ['model_generated_tokens','raw_generated_tokens','observation_tokens','environment_correction_tokens','initial_prompt_tokens','trajectory_total_tokens','cumulative_model_input_tokens']:totals[key]=totals.get(key,0)+r[key]
if complete:assert sha(root/'normal.jsonl')==complete['normal_jsonl_sha256']
report={'state':'VERIFIED'if complete else'PARTIAL_VERIFIED','at':time.time(),'normal_only':True,'count':len(rows),'target':target,'actual_batch_sizes':batch_sizes,'checkpoint_source_sha_files_matched':len(expected),'normal_binding_sha256':digest(binding),'data_sha256':binding['data_sha256'],'canonical_occurrences_gold_prompt_and_question_checked':len(rows),'actual_capture_token_arithmetic_verified':True,'raw_and_normalized_shard_checksums_verified':True,'model_mask_matches_retained_action_tokens':True,'force_final_answer':False,'actual_turns':turns,'actual_searches':searches,'token_totals':totals,'generation_source_sha256':binding['runtime_sources_sha256'],'retrieval_original_vs_actual_shards':[8,2],'original_service_output_token_parity_claimed':False,'progress':json.loads((root/'progress.json').read_text()),'normal_jsonl_sha256':complete['normal_jsonl_sha256']if complete else None,'verified_raw_bytes':checked_raw_bytes,'cf_started':False}
p=root/('pilot-verification.json'if phase=='pilot96'else'normal-independent-verification.json');temp=p.with_name(p.name+'.tmp')
with temp.open('w')as f:json.dump(report,f,indent=2);f.flush();os.fsync(f.fileno())
os.replace(temp,p);print(json.dumps(report))
'''


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--asset-audit', required=True)
    p.add_argument('--audit-output', required=True)
    p.add_argument('--phase', choices=['pilot96', 'full3610'], default='pilot96')
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    expected = json.loads(Path(args.asset_audit).read_text())['files']
    code = REMOTE.replace('INPUT', repr(json.dumps(dict(files=expected,
        phase=args.phase, allow_partial=args.allow_partial))))
    compile(code, 'remote-normal-pilot-verification', 'exec')
    spec = importlib.util.spec_from_file_location('ctl', '/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
    result = subprocess.run(ctl.route('baai-a800', '/data/search-g1/env/bin/python -B -c ' + shlex.quote(code)),
                            capture_output=True, text=True, timeout=60)
    audit = dict(transport_exit=result.returncode, stderr=result.stderr[-1800:])
    if result.returncode:
        audit['output_tail'] = result.stdout[-1000:]
    else:
        audit['report'] = json.loads(result.stdout.strip().splitlines()[-1])
    Path(args.audit_output).write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit))
    if result.returncode:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
