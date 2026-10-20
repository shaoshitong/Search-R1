"""Remote-RAM-only native prefix provenance check for actual first B48 old5+new5."""
import hashlib,json,pathlib,gzip,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
T=B/'algorithm-improvement-20261021/g10-matched6528-v1'
REV='a09a35458c702b33eeacc393d103063234e8bc28'
def read(p):return json.loads(pathlib.Path(p).read_text())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return h.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def check(v):
 p=pathlib.Path(v['path']);assert sha(p)==v['sha256']
 if 'bytes' in v:assert p.stat().st_size==v['bytes']
 return p
prep=read(T/'preparation.json');merge=read(check(prep['g10_complete']));groups=read(check(merge['groups']))['train'];lookup={g['qid']:g for g in groups}
plan=read(check(prep['source_plan']));qids=plan['batches'][0];assert len(qids)==48
needed={tid:(q,j) for q in qids for j,tid in enumerate(lookup[q]['trajectory_ids'])};assert len(needed)==480
original=read(check(merge['original_feature_manifest']));extra_refs=read(check(merge['extra_G5_feature_sources']))
manifests=original+[str(check(r)) for r in extra_refs];found={};retained_refs=[]
for raw in manifests:
 path=pathlib.Path(raw);m=read(path)
 assert m['status']=='LOSSLESS_NATIVE_FEATURES_COMPLETE' and tuple(m[k] for k in ('feature_layer','prefix_tokens','question_max_tokens','dimension'))==(26,20,273,3584)
 index=m['files']['index.jsonl'];check(index);own=[]
 with pathlib.Path(index['path']).open() as f:
  for line in f:
   row=json.loads(line);tid=row['trajectory_id']
   if tid not in needed:continue
   assert tid not in found and row['qid']==needed[tid][0] and row['dimension']==3584
   found[tid]=(path,row);own.append(tid)
 if own:
  check(m['files']['prototype.bin']);retained_refs.append(dict(native_manifest=fp(path),index=index,trajectories=len(own)))
 if len(found)==len(needed):break
assert set(found)==set(needed)
by_prefix={}
for tid,(manifest,row) in found.items():
 prefix=manifest.parent.parent/(manifest.parent.name.replace('-packed','-prefix.jsonl.gz'))
 by_prefix.setdefault(prefix,set()).add(tid)
prefix_refs=[];prompts={};counts={'old5':0,'new5':0};short=0;segments_count=0
for path,tids in by_prefix.items():
 got={};prefix_refs.append(fp(path))
 with gzip.open(path,'rt') as f:
  for line in f:
   v=json.loads(line)
   if v['trajectory_id'] in tids:assert v['trajectory_id'] not in got;got[v['trajectory_id']]=v
 assert set(got)==tids
 for tid,v in got.items():
  q,j=needed[tid];manifest,row=found[tid];group=lookup[q]
  assert v['qid']==q and v['question_hash']==group['question_hash'] and v['split']=='train'
  assert v['actor_version']==v['feature_actor_version']=='official7b-'+REV and v['feature_layer']==26
  assert v['feature_file']==str(manifest) and v['future_information_included'] is False
  body={k:x for k,x in v.items() if k!='prefix_sha256' and not k.startswith('feature_')};assert digest(body)==v['prefix_sha256']
  prompt=v['initial_prompt_token_ids'];policy=v['policy_token_ids'];pos=v['policy_position_ids'];segments=v['prefix_segments']
  assert 1<=len(prompt)<=273 and len(policy)==len(pos)==v['retained_policy_length']<=20
  assert row['question_length']==len(prompt) and row['policy_length']==len(policy) and row['policy_positions']==pos
  if len(policy)<20:assert v['terminal'] is True;short+=1
  offset=0;last_context=None;expected=[]
  for segment in segments:
   start=segment['policy_start'];length=segment['policy_length'];request=segment['request_length'];position=segment['position_start'];context=segment['context_tokens']
   assert start==offset and length>0 and start+length<=len(policy) and request==position and len(context)==request+length
   assert context[request:]==policy[start:start+length]
   if offset==0:assert segment['turn_index']==0 and context[:request]==prompt
   else:assert context[:len(prompt)]==prompt and context[:len(last_context)]==last_context
   expected.extend(range(position,position+length));offset+=length;last_context=context;segments_count+=1
  assert offset==len(policy) and expected==pos
  if q in prompts:assert prompts[q]==prompt
  else:prompts[q]=prompt
  counts['old5' if j<5 else 'new5']+=1
assert len(prompts)==48 and counts==dict(old5=240,new5=240)
print(json.dumps(dict(status='FIRST_ACTUAL_B48_OLD5_NEW5_NATIVE_PREFIX20_PROMPT_CAUSAL_POSITION_PROVENANCE_PASS',
 observed_epoch=time.time(),source_G10_complete=fp(prep['g10_complete']['path']),questions=48,real_trajectories=480,
 event_counts=counts,source_model_revision=REV,native_capture_layer=26,prefix_tokens=20,
 exact_same_initial_prompt_for_all_ten_per_question=True,native_feature_locator_qid_lengths_positions_match=True,
 sealed_prefix_own_SHA_pass=True,all_policy_segments_causal_own_tokens_only=True,future_information_included=False,
 terminal_short_prefixes=short,causal_segments=segments_count,native_manifest_count=len(retained_refs),
 native_manifest_and_index_refs_sha256=digest(retained_refs),sealed_prefix_file_refs_sha256=digest(prefix_refs),
 original_and_extra_are_distinct_native_events=True,bulk_native_tensor_payload_rehashed=False,
 feature_payload_acceptance='Previously sealed merge fullSHA plus actual training FeatureReader per-row lossless hash',
 torch_imported=False,model_or_optimizer_deserialized=False,raw_tokens_or_features_returned=False,untouched_final_read=False)))
