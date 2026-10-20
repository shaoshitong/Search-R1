"""Fail-closed authentic8192 Prefix20 confidence extraction; no model/score inference."""
import argparse,fcntl,gzip,hashlib,json,math,os,resource,sys,time
from collections import Counter
from pathlib import Path
try:from . import audit_prefix_logprob_availability as a
except ImportError:import audit_prefix_logprob_availability as a
B=a.B;A=a.A;BASE=A/'prefix-confidence-incremental-v1';DATA=BASE/'data';FEATURE_NAMES=('mean_logp','std_logp','min_logp','first5_mean_logp','last5_mean_logp','actual_len_over20')
STATUS='ALL8192_REAL_G10_PREFIX20_NATIVE_CHOSEN_CONFIDENCE_EXTRACTED_AND_ID_VERIFIED'
def read(p):return a.read(p)
def ref(p):return a.ref(p)
def checked(v):return a.checked(v)
def digest(v):return a.digest(v)
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp.%d'%os.getpid());q.write_text(json.dumps(v,sort_keys=True,allow_nan=False));q.replace(p)
def once(p,v):
 p=Path(p)
 if p.exists():assert read(p)==v
 else:atomic(p,v)
def stamp(p):
 s=Path(p).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def split_ids(qids):
 assert len(qids)==len(set(qids))==8192;held=set(sorted(qids,key=lambda q:(hashlib.sha256(('prefix-confidence-qid-9137:'+str(q)).encode()).hexdigest(),str(q)))[:2048]);fit=[q for q in qids if q not in held];hold=[q for q in qids if q in held]
 return dict(seed=9137,fit_qids=fit,held_qids=hold,fit_questions=6144,held_questions=2048,fit_ordered_qid_sha256=digest(fit),held_ordered_qid_sha256=digest(hold),labels_or_scores_used=False,views_whole_question_isolated=True,internal_assessment_indices=[5,6,7,8,9])
def six_features(values):
 assert 0<len(values)<=20 and all(math.isfinite(x) and x<=1e-6 for x in values);n=len(values);mean=math.fsum(values)/n;std=math.sqrt(math.fsum((x-mean)**2 for x in values)/n);return [mean,std,min(values),math.fsum(values[:5])/len(values[:5]),math.fsum(values[-5:])/len(values[-5:]),n/20]
def projection(prefix,full):
 check=a.project(prefix,full);values=[lp for turn in full['behavior_turns'] for lp in turn['selected_logprobs'][:turn['kept_token_count']]][:20];features=six_features(values)
 assert len(values)==check['kept_prefix_policy_tokens'] and math.isclose(features[0],check['prefix_mean_logprob'],rel_tol=1e-14,abs_tol=1e-14);signature=digest(dict(initial_prompt=prefix['initial_prompt_token_ids'],policy_tokens=prefix['policy_token_ids'],positions=prefix['policy_position_ids'],segments=[{k:v for k,v in s.items() if k!='turn_index'} for s in prefix['prefix_segments']]))
 return dict(features=features,lp=values,causal_signature=signature,policy_length=len(values),zero_logprob_count=sum(x==0 for x in values))
def quality():return dict(trajectories=0,questions=0,policy_tokens=0,zero_logprob_tokens=0,constant_logprob_trajectories=0,length_histogram=[0]*21,sum=[0.]*6,sumsq=[0.]*6,min=[None]*6,max=[None]*6,feature_zero_trajectories=[0]*6,within_G5_constant_feature_questions=[0]*6)
def add_quality(q,rows):
 q['questions']+=1
 for j in range(6):q['within_G5_constant_feature_questions'][j]+=len({r['features'][j] for r in rows})==1
 for row in rows:
  q['trajectories']+=1;q['policy_tokens']+=row['policy_length'];q['length_histogram'][row['policy_length']]+=1;q['zero_logprob_tokens']+=row['zero_logprob_count'];q['constant_logprob_trajectories']+=len(set(row['lp']))==1
  for j,x in enumerate(row['features']):q['sum'][j]+=x;q['sumsq'][j]+=x*x;q['feature_zero_trajectories'][j]+=x==0;q['min'][j]=x if q['min'][j] is None else min(x,q['min'][j]);q['max'][j]=x if q['max'][j] is None else max(x,q['max'][j])
def noise():return {s:dict(pairs=0,token_comparisons=0,absolute_difference_sum=0.,max_absolute_difference=0.,pair_mean_absolute_difference_sum=0.,nonidentical_float_pairs=0,max_abs_histogram=[0]*7) for s in ('all','original_original','newseed_newseed','cross_event')}
def add_noise(out,rows):
 thresholds=(0.,1e-6,1e-4,1e-3,1e-2,.1,float('inf'))
 for i in range(10):
  for j in range(i+1,10):
   if rows[i]['causal_signature']!=rows[j]['causal_signature']:continue
   assert len(rows[i]['lp'])==len(rows[j]['lp']);diff=[abs(x-y) for x,y in zip(rows[i]['lp'],rows[j]['lp'])];m=max(diff);s=math.fsum(diff);scope='original_original' if j<5 else ('newseed_newseed' if i>=5 else 'cross_event')
   for key in ('all',scope):
    r=out[key];r['pairs']+=1;r['token_comparisons']+=len(diff);r['absolute_difference_sum']+=s;r['max_absolute_difference']=max(r['max_absolute_difference'],m);r['pair_mean_absolute_difference_sum']+=s/len(diff);r['nonidentical_float_pairs']+=m!=0;r['max_abs_histogram'][next(k for k,t in enumerate(thresholds) if m<=t)]+=1
def merge(values,key=None):
 x=values[0]
 if isinstance(x,dict):return {k:merge([v[k] for v in values],k) for k in x}
 if isinstance(x,list):
  if key in ('min','max'):return [(min if key=='min' else max)(v[i] for v in values if v[i] is not None) for i in range(len(x))]
  return [merge([v[i] for v in values]) for i in range(len(x))]
 return max(values) if key=='max_absolute_difference' else sum(values)
def quality_summary(q):
 n=q['trajectories'];return dict(q,feature_names=list(FEATURE_NAMES),mean=[x/n for x in q['sum']],std=[0. if q['min'][j]==q['max'][j] else math.sqrt(max(0,q['sumsq'][j]/n-(q['sum'][j]/n)**2)) for j in range(6)],global_constant_feature=[q['min'][j]==q['max'][j] for j in range(6)],feature_zero_fraction=[x/n for x in q['feature_zero_trajectories']],within_G5_constant_feature_fraction=[x/q['questions'] for x in q['within_G5_constant_feature_questions']],constant_logprob_trajectory_fraction=q['constant_logprob_trajectories']/n)
def prepare(code,commit):
 code=Path(code);BASE.mkdir(parents=True,exist_ok=True);DATA.mkdir(exist_ok=True);training=read(A/'peer-newseed8192-g10-train-v1/training-input.json');identity=read(checked(training['identity_plan']));qids=[g['qid'] for g in identity['groups']];split=split_ids(qids);once(BASE/'split.json',split) # BEFORE opening G10 truth or probe probabilities.
 raw=read(checked(training['raw_complete']));score=ref(A/'peer8192-frozen-calibration-scores-v2/complete.json');assert score['sha256']=='5660631604565664ee8d6aab20bacc3adf91fbf2407c251bfcd2c8cce6cd5c42';done=read(score['path']);assert done['normal_exit'] and done['actor_updates']==done['probe_updates']==0;e4=done['checkpoints']['E4'];assert e4['checkpoint']['steps']==14418 and e4['checkpoint']['sha256']=='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'
 availability=ref(A/'prefix20-chosen-logprob-availability-v1/complete.json');assert availability['sha256']=='d38529092bb5fc653cefb2cc80181cd06c08fd9951ef36425aa09a924f5e1959';v=read(availability['path']);assert v['all_three_paths_and_sealed_prefix_exact'] and v['samples']==11
 external_meta=read(A/'peer-newseed8192-g10-train-v1/E4-same-HF-validation/complete.json');external=external_meta['checkpoints']['E4']['predictions'];assert external['sha256']=='66cdf3e7c748bda7f508c4f381ad0456a413d5d73c73e5d9f1eb774b71bdb707';valref=ref(A/'validation-g10-20261021/materialized/bank/complete.json');assert valref['sha256']=='5ea5197a841d7bd0b226a82350a2436a8ae3d860143a2ea39fe59b7ecb6837d5';val=read(valref['path'])
 cfg=dict(role='SINGLE_PREFIX20_CONFIDENCE_AUXILIARY_8SCALAR_INCREMENTAL_INFORMATION_TEST',source_commit=commit,source_code={p.name:ref(p) for p in (code/'prefix_confidence_data.py',code/'audit_prefix_logprob_availability.py')},source_protocol=ref(code/'prefix-confidence-PROTOCOL.md'),protocol_git_commit='4e5d2b803954e959cb470d2bf07fa35420d3ed10',identity_plan=training['identity_plan'],raw_complete=training['raw_complete'],source_events=raw['outputs']['source-events'],groups=done['groups'],E4_scoring_complete=score,E4_predictions=e4['predictions'],source_checkpoint=e4['checkpoint'],source_EMA_sha256=e4['ema_sha256'],availability_sample_receipt=availability,split=ref(BASE/'split.json'),feature_names=list(FEATURE_NAMES),external_predictions=external,external_validation_complete=valref,external_fixed_G5_plan=val['consumer_inputs']['fixed_G5_plan'],external_groups=val['consumer_inputs']['groups'],external_source_audit=val['outputs']['source-audit.json'],external_payload_opened=False,CPU=2,RSS_limit_bytes=3*1024**3,wall_limit_seconds=26*60,optimizer='one new8scalar LBFGS; original Probe/raw/EMA/Adam/RNG remain read-only',fixed_regularizer=.01,internal_assessment_indices=[5,6,7,8,9],calibration_only_seen_native_warmup_questions=True)
 once(BASE/'config.json',cfg);return dict(status='FROZEN_QID_SPLIT_AND_TYPED_PREFIX_CONFIDENCE_INPUTS_PREPARED',config=ref(BASE/'config.json'),split=cfg['split'],questions=8192,trajectories=81920,external_payload_opened=False)
def records(batch,wanted):
 stamps={k:stamp(checked(batch[k])) for k in ('input','prefix','full')};prefixes={};seen=set()
 with gzip.open(batch['prefix']['path'],'rt') as f:
  for line in f:
   p=json.loads(line)
   if p['trajectory_id'] in wanted:assert p['trajectory_id'] not in prefixes;prefixes[p['trajectory_id']]=p
 assert set(prefixes)==set(wanted)
 with gzip.open(batch['full']['path'],'rt') as f:
  for line in f:
   full=json.loads(line);tid=full['trajectory_id']
   if tid not in wanted:continue
   own=wanted[tid];assert tid not in seen and full['qid']==own['qid'] and full['question_hash']==own['question_hash'] and type(full['strict_em']) is int and full['strict_em']==own['label'];seen.add(tid);yield tid,projection(prefixes[tid],full)
 assert seen==set(wanted) and stamps=={k:stamp(batch[k]['path']) for k in stamps}
def extract(config_sha):
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));cfg=ref(BASE/'config.json');assert cfg['sha256']==config_sha;c=read(cfg['path']);start=time.monotonic();identity=read(checked(c['identity_plan']));qids=[g['qid'] for g in identity['groups']];assert read(checked(c['split']))==split_ids(qids)
 if (DATA/'complete.json').exists():return read(DATA/'complete.json')
 truth=read(checked(c['groups']));truth=truth['groups'] if isinstance(truth,dict) else truth;assert qids==[g['qid'] for g in truth];by={g['qid']:g for g in truth}
 for g,own in zip(identity['groups'],truth):assert g['trajectory_ids']==own['trajectory_ids'] and g['question_hash']==own['question_hash'] and len(own['labels'])==10
 events=read(checked(c['source_events']));work={}
 for e in events['events']:
  own=[];seen=set()
  for i,batch in enumerate(e['batches']):
   selected=sorted(set(qids).intersection(r['qid'] for r in read(checked(batch['input']))['rows']))
   if selected:assert not seen.intersection(selected);seen.update(selected);own.append((i,batch,selected))
  assert seen==set(qids);work[e['event']]=own
 (DATA/'units').mkdir(exist_ok=True);done=set();unitrefs=[]
 for i,batch,selected in work['newseed20261021']:
  p=DATA/'units'/('new-%04d.json'%i)
  if p.exists():
   v=read(p);assert v['config']==cfg and v['selected_qids_sha256']==digest(selected);checked(v['features'])
   for path,s in v['source_stamps'].items():assert stamp(path)==s
   done.update(selected);unitrefs.append(ref(p))
 old={};oldproof={};count=0;atomic(DATA/'controller.json',dict(pid=os.getpid(),config=cfg,CPU_affinity=list(os.sched_getaffinity(0)),started=time.time(),external_payload_opened=False))
 for i,batch,selected in work['original']:
  selected=[q for q in selected if q not in done]
  if not selected:continue
  wanted={t:dict(qid=q,question_hash=by[q]['question_hash'],label=y) for q in selected for t,y in zip(by[q]['trajectory_ids'][:5],by[q]['labels'][:5])}
  for tid,row in records(batch,wanted):old[tid]=row;count+=1
  for q in selected:oldproof[q]={batch[k]['path']:stamp(batch[k]['path']) for k in ('input','prefix','full')}
  atomic(DATA/'progress.json',dict(stage='TRUE_NATIVE_PREFIX20_OLD5_LP_AND_SIX_FEATURES_IN_RAM',old_trajectories=count,already_accepted_questions=len(done),expected_trajectories=81920,time=time.time()))
  assert time.monotonic()-start<c['wall_limit_seconds'],'Bounded extraction wall budget exceeded before pairing'
 for i,batch,selected in work['newseed20261021']:
  p=DATA/'units'/('new-%04d.json'%i)
  if p.exists():continue
  wanted={t:dict(qid=q,question_hash=by[q]['question_hash'],label=y) for q in selected for t,y in zip(by[q]['trajectory_ids'][5:],by[q]['labels'][5:])};new={t:r for t,r in records(batch,wanted)};stats={'original':quality(),'newseed':quality()};jitter=noise();stamps={batch[k]['path']:stamp(batch[k]['path']) for k in ('input','prefix','full')};payload=p.with_suffix('.jsonl.gz');temp=payload.with_suffix('.gz.partial')
  with gzip.open(temp,'wb') as f:
   for q in selected:
    g=by[q];rows=[old.pop(t) for t in g['trajectory_ids'][:5]]+[new[t] for t in g['trajectory_ids'][5:]];add_quality(stats['original'],rows[:5]);add_quality(stats['newseed'],rows[5:]);add_noise(jitter,rows);stamps.update(oldproof.pop(q));f.write(a.canonical(dict(qid=q,trajectory_ids=g['trajectory_ids'],features=[r['features'] for r in rows]))+b'\n')
  temp.replace(payload);v=dict(config=cfg,features=ref(payload),questions=len(selected),trajectories=10*len(selected),selected_qids_sha256=digest(selected),quality=stats,same_causal_LP_numeric_differences=jitter,source_stamps=stamps,source_batch_sha256=digest(batch),all_native_turn_response_LP_and_P20_exact=True);once(p,v);done.update(selected);unitrefs.append(ref(p));atomic(DATA/'progress.json',dict(stage='REAL_G10_SIX_PREFIX_FEATURES_COMPLETE_BATCHES',questions=len(done),trajectories=10*len(done),expected_questions=8192,completed_units=len(unitrefs),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,external_payload_opened=False,time=time.time()))
  if time.monotonic()-start>c['wall_limit_seconds']:return dict(status='PARTIAL_FEATURE_UNITS_RESUMABLE',questions=len(done))
 assert len(done)==8192 and not old;allrows={};units=[]
 for i,batch,selected in work['newseed20261021']:
  p=DATA/'units'/('new-%04d.json'%i);v=read(p);assert v['source_batch_sha256']==digest(batch)
  for path,s in v['source_stamps'].items():assert stamp(path)==s
  with gzip.open(checked(v['features']),'rt') as f:
   for line in f:
    row=json.loads(line);assert row['qid'] not in allrows and row['trajectory_ids']==by[row['qid']]['trajectory_ids'];allrows[row['qid']]=row
  units.append(v)
 assert set(allrows)==set(qids);features=DATA/'features.jsonl.gz';temp=features.with_suffix('.gz.partial')
 with gzip.open(temp,'wb') as f:
  for q in qids:f.write(a.canonical(allrows[q])+b'\n')
 temp.replace(features);quality_all={name:quality_summary(merge([v['quality'][name] for v in units])) for name in ('original','newseed')};numeric=merge([v['same_causal_LP_numeric_differences'] for v in units]);assert numeric['all']['pairs']==3665
 for r in numeric.values():r['mean_absolute_token_difference']=r['absolute_difference_sum']/r['token_comparisons'] if r['token_comparisons'] else None;r['mean_pair_mean_absolute_difference']=r['pair_mean_absolute_difference_sum']/r['pairs'] if r['pairs'] else None
 out=dict(status=STATUS,config=cfg,features=ref(features),groups=c['groups'],split=c['split'],E4_predictions=c['E4_predictions'],source_checkpoint=c['source_checkpoint'],source_EMA_sha256=c['source_EMA_sha256'],source_protocol=c['source_protocol'],availability_qualified=True,availability_sample_receipt=c['availability_sample_receipt'],questions=8192,trajectories=81920,feature_names=list(FEATURE_NAMES),all_own_ID_EM_order_native_LP_and_P20_exact=True,ordered_qids_sha256=digest(qids),ordered_own_trajectory_ids_sha256=digest([t for g in truth for t in g['trajectory_ids']]),units=[ref(DATA/'units'/('new-%04d.json'%i)) for i,_,_ in work['newseed20261021']],quality_by_seed=quality_all,same_causal_LP_numeric_differences=numeric,numerical_differences_not_semantic_input_information=True,raw_LP_vectors_or_tokens_persisted=False,external_payload_opened=False,model_or_tensor_read=False,actor_updates=0,probe_updates=0,elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,time=time.time());once(DATA/'complete.json',out);return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','extract'),required=True);p.add_argument('--code-root');p.add_argument('--commit');p.add_argument('--config-sha256');args=p.parse_args();BASE.mkdir(exist_ok=True,parents=True);DATA.mkdir(exist_ok=True)
 with (DATA/'extract.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:v=prepare(args.code_root,args.commit) if args.action=='prepare' else extract(args.config_sha256);print(json.dumps(dict(status=v['status'],config=v.get('config'),questions=v.get('questions'),trajectories=v.get('trajectories'),elapsed_seconds=v.get('elapsed_seconds')),sort_keys=True),flush=True)
  except BaseException as e:atomic(DATA/'failure.json',dict(type=type(e).__name__,message=str(e)[:1200],time=time.time()));raise
