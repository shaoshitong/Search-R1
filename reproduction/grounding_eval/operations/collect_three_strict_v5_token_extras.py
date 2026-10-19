"""Stream remote normal files once for extra token telemetry; export counts only."""
import argparse,pathlib,json,hashlib,collections,os,time
p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();b=pathlib.Path(a.root);out={}
for method,prefix in [('Search-G1','search-g1'),('Soft-to-Hard','soft-to-hard'),('OPSD3304','opsd3304')]:
 cf=b/(prefix+'-cf-strict-v5');m=json.loads((cf/'manifest.json').read_text());assert m['protocol_version']=='strict-v5-conservative-typed';src=pathlib.Path(m['source']['path']);h=hashlib.sha256();totals=collections.Counter();measured=collections.Counter();count=requests=0;cache=[]
 with src.open('rb')as f:
  for line in f:
   h.update(line)
   if not line.strip():continue
   row=json.loads(line);assert row['evaluation_ordinal']==count and row['data_source']=='nq';count+=1
   for k in ['retrieved_observation_tokens','correction_tokens','environment_correction_tokens']:
    v=row.get(k)
    if isinstance(v,int):totals[k]+=v;measured[k]+=1
   if method=='OPSD3304':totals['aligned_retrieved_observation_tokens']+=row['observation_tokens'];totals['aligned_correction_tokens']+=row['environment_correction_tokens']
   else:totals['aligned_retrieved_observation_tokens']+=row['retrieved_observation_tokens'];totals['aligned_correction_tokens']+=row['correction_tokens']
   turns=row['turns']if method=='OPSD3304'else row['per_turn_token_usage'];assert turns;requests+=len(turns);cache.append(row.get('cached_input_tokens'))
 assert count==3610 and h.hexdigest()==m['source']['sha256']
 for k in ['aligned_retrieved_observation_tokens','aligned_correction_tokens']:measured[k]=count
 cf_cache=[json.loads(p.read_text()).get('cached_input_tokens')for p in(cf/'results').glob('*.json')]
 def cache_stats(values):
  known=[v for v in values if isinstance(v,int)];unknown=len(values)-len(known)
  return dict(total=sum(known)if not unknown else None,mean=sum(known)/len(values)if values and not unknown else None,known_records=len(known),unknown_records=unknown,null_reason='not_all_native_requests_expose_cache_hit_count'if unknown else None)
 types=collections.Counter()
 for line in(cf/'prepared.jsonl').open():
  v=json.loads(line)
  if v['status']=='READY':types[v['replacement_kind']]+=1
 out[method]=dict(at=time.time(),source_sha256=h.hexdigest(),normal_count=count,normal_actual_generation_requests=dict(total=requests,mean=requests/count),normal_extra_tokens={k:dict(total=v,mean=v/measured[k],measured_questions=measured[k])for k,v in totals.items()},normal_cached_input_tokens=cache_stats(cache),CF_cached_input_tokens=cache_stats(cf_cache),ready_replacement_type_counts=dict(types),notes='A800 reported observation_tokens includes all masked environment; aligned retrieval/correction fields use captured per-turn values. OPSD retrieval/correction were already separate. Cache unknown is never zero.')
raw=json.dumps(out,indent=2).encode();p=pathlib.Path(a.output)
with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))
