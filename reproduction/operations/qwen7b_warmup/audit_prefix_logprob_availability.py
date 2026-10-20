"""Finite chosen-token behavior-logprob schema/causal audit; never fit or score pools."""
import argparse,fcntl,gzip,hashlib,json,math,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';TARGET=A/'prefix20-chosen-logprob-availability-v1'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ref(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4194304),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(v):
 p=Path(v['path']);assert p.is_absolute() and p.resolve()==p and str(p).startswith(str(B)+'/');actual=ref(p)
 assert all(actual[k]==v[k] for k in ('path','bytes','sha256'));return p
def vector(v):
 assert isinstance(v,list) and all(type(x) in (float,int) and math.isfinite(x) and x<=1e-6 for x in v);return v

def project(prefix,full):
 """Join real native chosen LP by actual retained policy order, never observation0."""
 assert prefix['trajectory_id']==full['trajectory_id'] and prefix['qid']==full['qid'] and prefix['question_hash']==full['question_hash'];body={k:v for k,v in prefix.items() if k!='prefix_sha256' and not k.startswith('feature_')};assert digest(body)==prefix['prefix_sha256']==full['prefix_sha256'];assert prefix['future_information_included'] is False
 prompt=prefix['initial_prompt_token_ids'];assert prompt==full['initial_prompt_token_ids'];ids=[];lp=[];positions=[];segments=[];rebuilt=[];all_lp=[];mask=[];journal=[];settings=[];native_lp_tokens=0
 for turn in full['behavior_turns']:
  request=turn['actual_request_token_ids'];raw=turn['raw_generated_token_ids'];chosen=vector(turn['selected_logprobs']);kept=turn['kept_token_count'];assert request==prompt+rebuilt and len(raw)==len(chosen) and 0<=kept<=len(raw) and turn['prefix_cut']['kept_token_ids']==raw[:kept] and turn['prefix_cut']['kept_token_count']==kept
  chunkids=[];chunklp=[]
  for chunk in turn['actual_sampling_chunks']:
   assert chunk['actual_request_token_ids']==request+chunkids;rawids=chunk['raw_generated_token_ids'];values=vector(chunk['selected_logprobs']);assert len(rawids)==len(values);chunkids+=rawids;chunklp+=values
   if len(ids)<20 and kept:
    settings.append({k:v for k,v in chunk.get('sampling',{}).items() if k in ('temperature','top_p','top_k','logprobs','presence_penalty','frequency_penalty','repetition_penalty','min_tokens','max_tokens')})
  assert chunkids==raw and chunklp==chosen;native_lp_tokens+=len(chunklp);journal+=turn['actual_sampling_chunks'];take=min(kept,max(0,20-len(ids)))
  if take:
   segments.append(dict(turn_index=turn['turn_index'],policy_start=len(ids),policy_length=take,request_length=len(request),position_start=len(request),context_tokens=request+raw[:take]));ids+=raw[:take];lp+=chosen[:take];positions+=list(range(len(request),len(request)+take))
  obs=turn['observation_token_ids'];corr=turn['correction_token_ids'];assert not(obs and corr);env=obs+corr;rebuilt+=raw[:kept]+env;all_lp+=chosen[:kept]+[0.] *len(env);mask += [1]*kept+[0]*len(env)
 assert journal==full['sampling_journal'] and rebuilt==full['response_token_ids'] and mask==full['response_info_mask'] and all_lp==full['response_log_probs'];policy_response=[(token,value) for token,value,flag in zip(rebuilt,all_lp,mask) if flag==1][:20]
 assert [v[0] for v in policy_response]==ids and [v[1] for v in policy_response]==lp and ids==prefix['policy_token_ids'] and positions==prefix['policy_position_ids'] and segments==prefix['prefix_segments'] and len(ids)==prefix['retained_policy_length'] and (len(ids)==20 or prefix['terminal'] is True)
 # Only aggregate/sample metadata escapes this function; no chosen vector/IDs.
 return dict(kept_prefix_policy_tokens=len(ids),native_chosen_logprob_tokens_present=native_lp_tokens,prefix_nonzero_chosen_logprobs=sum(x!=0 for x in lp),prefix_exact_zero_chosen_logprobs=sum(x==0 for x in lp),prefix_min_logprob=min(lp),prefix_max_logprob=max(lp),prefix_mean_logprob=sum(lp)/len(lp),all_values_finite=True,positive_logprob_count=sum(x>0 for x in lp),native_turn_response_chosen_logprobs_exact=True,sealed_tokens_positions_contexts_exact=True,environment_placeholder_positions_excluded=True,prefix_json_has_logprob_fields=[k for k in prefix if 'log_prob' in k or 'logprob' in k],full_schema_chosen_logprob_fields=['sampling_journal[].selected_logprobs','behavior_turns[].actual_sampling_chunks[].selected_logprobs','behavior_turns[].selected_logprobs','response_log_probs (policy positions only)'],sampling_settings=settings,prefix_confidence_is_not_full_vocabulary_entropy=True)

def sample_records(prefixref,fullref,tids,qid):
 checked(prefixref);checked(fullref);prefixes={};fulls={}
 for item,output in ((prefixref,prefixes),(fullref,fulls)):
  with gzip.open(item['path'],'rt') as f:
   for line in f:
    row=json.loads(line);tid=row['trajectory_id']
    if tid in tids:
     assert tid not in output and row['qid']==qid;output[tid]=row
     if len(output)==len(tids):break
 assert set(prefixes)==set(fulls)==set(tids);return [project(prefixes[t],fulls[t]) for t in tids]

def run(commit):
 TARGET.mkdir(parents=True,exist_ok=True)
 if (TARGET/'complete.json').exists():return read(TARGET/'complete.json')
 training=read(A/'peer-newseed8192-g10-train-v1/training-input.json');identity=read(checked(training['identity_plan']));raw=read(checked(training['raw_complete']));events=read(checked(raw['outputs']['source-events']));first=identity['groups'][0];qid=first['qid'];result=[];proofs=[];progress=[]
 for name,start in [('original',0),('newseed20261021',5)]:
  event=next(x for x in events['events'] if x['event']==name);found=[]
  for batch in event['batches']:
   if qid in {x['qid'] for x in read(checked(batch['input']))['rows']}:found.append(batch)
  assert len(found)==1;batch=found[0];tids=first['trajectory_ids'][start:start+3];rows=sample_records(batch['prefix'],batch['full'],tids,qid);result.append(dict(scope='training_original8192_'+name,selection='first frozen identity qid; first3 event-local authentic IDs; no label/score selection',samples=rows));proofs.append(dict(input=batch['input'],prefix=batch['prefix'],full=batch['full']));progress.append(dict(stage=name,samples=sum(len(x['samples']) for x in result),time=time.time()));print(json.dumps(progress[-1]),flush=True)
 # Resolve only the one precommitted validation qid's batch inputs, not all raw validation.
 val= A/'validation-g10-20261021/materialized/bank/complete.json';valref=ref(val);assert valref['sha256']=='5ea5197a841d7bd0b226a82350a2436a8ae3d860143a2ea39fe59b7ecb6837d5';accepted=read(val);fixedref=accepted['consumer_inputs']['fixed_G5_plan'];fixed=read(checked(fixedref));assert fixed['summary']['labels_consulted'] is False;own=fixed['groups'][0];want=set(own['trajectory_ids']);auditref=accepted['outputs']['source-audit.json'];audit=read(checked(auditref));taken=set();rows=[]
 for name,event in audit['events'].items():
  for batch in event:
   metadata=read(checked(batch['input']))
   if own['qid'] not in {x['qid'] for x in metadata['rows']}:continue
   prefixref=batch['phase_payloads']['prefix'];checked(prefixref);matching=[]
   with gzip.open(prefixref['path'],'rt') as f:
    for line in f:
     p=json.loads(line)
     if p['trajectory_id'] in want:matching.append(p['trajectory_id'])
   if not matching:continue
   assert not taken.intersection(matching);taken.update(matching);fullref=batch['phase_payloads']['full'];rows+=sample_records(prefixref,fullref,matching,own['qid']);proofs.append(dict(input=batch['input'],prefix=prefixref,full=fullref))
 assert taken==want and len(rows)==5;result.append(dict(scope='fixed_validation2048_first_qid_G5',selection='existing fixed firstqid all5 own IDs; no new G5 selection, no labels read',samples=rows));progress.append(dict(stage='fixed_validation_firstqid',samples=11,time=time.time()));print(json.dumps(progress[-1]),flush=True)
 for batch in proofs:
  for item in batch.values():checked(item)
 allrows=[r for x in result for r in x['samples']];out=dict(status='FINITE_SHA_BOUND_PREFIX20_NATIVE_CHOSEN_LOGPROBS_AVAILABLE_AND_CAUSALLY_ALIGNED',source_commit=commit,training_identity=training['identity_plan'],training_raw_complete=training['raw_complete'],training_source_events=raw['outputs']['source-events'],validation_complete=valref,validation_fixed_G5_plan=fixedref,validation_source_audit=auditref,source_batch_refs=proofs,samples=len(allrows),prefix_policy_tokens=sum(x['kept_prefix_policy_tokens'] for x in allrows),nonzero_chosen_logprob_tokens=sum(x['prefix_nonzero_chosen_logprobs'] for x in allrows),exact_zero_chosen_logprob_tokens=sum(x['prefix_exact_zero_chosen_logprobs'] for x in allrows),all_three_paths_and_sealed_prefix_exact=True,scopes=result,progress=progress,full_validation_scored_or_scanned=False,only_one_frozen_validation_qid_G5_full_records_read=True,labels_used=False,teacher_scores_used=False,model_or_tensor_read=False,GPU_or_optimizer_used=False,final_read=False,full_vocabulary_entropy_claim=False,probability_semantics='Recorded native selected-token behavior logprob; no new forward/recomputation. Sampling settings reported; not a fabricated entropy or correctness probability.',time=time.time());p=TARGET/'complete.json';temp=p.with_suffix('.json.partial');temp.write_text(json.dumps(out,sort_keys=True));temp.replace(p);return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source-commit',required=True);a=p.parse_args();TARGET.mkdir(parents=True,exist_ok=True)
 with (TARGET/'audit.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);v=run(a.source_commit);print(json.dumps(dict(complete=ref(TARGET/'complete.json'),summary=v),sort_keys=True))
