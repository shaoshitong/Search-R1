"""Read-only causal prefix5/20/40/80/160 reconstruction of frozen train8192.

Only aggregate checkpoints persist. Per-trajectory signatures live in RAM and
are rebuilt on resume. No generated tokens, hidden/model tensors or evaluation
pool is opened. Token uniqueness is not claimed to imply learnability.
"""
import argparse,fcntl,gzip,hashlib,json,math,os,resource,sys,time
from collections import defaultdict
from pathlib import Path
LENGTHS=(5,20,40,80,160)
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021';TARGET=A/'peer8192-long-prefix-causal-diagnostic-v1'
STATUS='FROZEN8192_TRUE_G10_LONG_CAUSAL_PREFIX_AGGREGATES_COMPLETE'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def fp(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for v in iter(lambda:f.read(4*1024**2),b''):h.update(v)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(v):
 p=Path(v['path']);assert p.is_absolute() and p.resolve()==p and str(p).startswith(str(B)+'/');a=fp(p)
 if any(a[k]!=v[k] for k in ('path','bytes','sha256')):raise ValueError('Frozen source SHA/size differs')
 return p
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.%d.tmp'%os.getpid())
 with q.open('w') as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 q.replace(p)
def immutable(p,v):
 p=Path(p)
 if p.exists():
  if read(p)!=v:raise ValueError('Conflicting immutable aggregate/config')
 else:atomic(p,v)
def stamp(p):
 x=Path(p).stat();return [x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns]
def ids(v):
 if not isinstance(v,list) or any(type(x) is not int or x<0 for x in v):raise ValueError('Actual integer token IDs required')
 return v


def reconstruct(full,n):
 """No suffix after selected policy token is ever included in a causal context."""
 if n not in LENGTHS:raise ValueError('Predeclared prefix length only')
 prompt=ids(full['initial_prompt_token_ids']);rebuilt=[];mask=[];tokens=[];positions=[];segments=[];journal=[]
 past_retrieval=past_correction=0;past_retrieval_events=0;coverage_retrieval=coverage_correction=coverage_events=0;later_policy=0
 for turn in full['behavior_turns']:
  request=ids(turn['actual_request_token_ids']);raw=ids(turn['raw_generated_token_ids']);kept=turn['kept_token_count']
  if request!=prompt+rebuilt or type(kept) is not int or not 0<=kept<=len(raw) or turn['prefix_cut']['kept_token_ids']!=raw[:kept] or turn['prefix_cut']['kept_token_count']!=kept:raise ValueError('Actual causal request/retained cut differs')
  chunks=[]
  for chunk in turn['actual_sampling_chunks']:
   if chunk['actual_request_token_ids']!=request+chunks:raise ValueError('Actual native sampling chunk context differs')
   chunks.extend(ids(chunk['raw_generated_token_ids']))
  if chunks!=raw:raise ValueError('Native sampling chunks do not reconstruct actual turn')
  journal+=turn['actual_sampling_chunks'];take=min(kept,max(0,n-len(tokens)))
  if take:
   start=len(tokens);segments.append(dict(turn_index=turn['turn_index'],policy_start=start,policy_length=take,request_length=len(request),position_start=len(request),context_tokens=request+raw[:take]));tokens.extend(raw[:take]);positions.extend(range(len(request),len(request)+take))
   coverage_retrieval=past_retrieval;coverage_correction=past_correction;coverage_events=past_retrieval_events
   if len(segments)>1:later_policy+=take
  obs=ids(turn['observation_token_ids']);correction=ids(turn['correction_token_ids'])
  if obs and correction:raise ValueError('Two mutually exclusive environment suffixes present')
  rebuilt+=raw[:kept]+obs+correction;mask += [1]*kept+[0]*(len(obs)+len(correction));past_retrieval+=len(obs);past_correction+=len(correction);past_retrieval_events+=bool(obs)
 if (not prompt or not tokens or rebuilt!=full['response_token_ids'] or mask!=full['response_info_mask'] or journal!=full['sampling_journal'] or full['terminal_reason'] not in ('natural_eos','answer','final_action_budget')):raise ValueError('Actual full journal/context/mask/termination differs')
 causal=dict(initial_prompt_token_ids=prompt,policy_token_ids=tokens,policy_position_ids=positions,segments=[{k:v for k,v in seg.items() if k!='turn_index'} for seg in segments])
 return dict(tokens=tokens,positions=positions,segments=segments,token_signature=digest(tokens),causal_signature=digest(causal),policy_tokens=len(tokens),short=len(tokens)<n,terminal_reason=full['terminal_reason'],multi_turn=len(segments)>1,past_retrieval_tokens=coverage_retrieval,past_correction_tokens=coverage_correction,past_retrieval_events=coverage_events,later_turn_policy_tokens=later_policy,max_selected_context_tokens=max(len(seg['context_tokens']) for seg in segments))


def qualify_row(prefix,full,wanted):
 if (prefix['qid']!=wanted['qid'] or full['qid']!=wanted['qid'] or prefix['trajectory_id']!=wanted['trajectory_id'] or full['trajectory_id']!=wanted['trajectory_id'] or prefix['question_hash']!=wanted['question_hash'] or full['question_hash']!=wanted['question_hash'] or full['strict_em']!=wanted['label'] or prefix['initial_prompt_token_ids']!=full['initial_prompt_token_ids']):raise ValueError('Frozen own trajectory/question/strictEM differs')
 body={k:v for k,v in prefix.items() if k!='prefix_sha256' and not k.startswith('feature_')}
 if digest(body)!=prefix['prefix_sha256'] or full['prefix_sha256']!=prefix['prefix_sha256'] or prefix['future_information_included'] is not False:raise ValueError('Sealed actual Prefix20 source SHA differs')
 values={str(n):reconstruct(full,n) for n in LENGTHS};p20=values['20']
 if (p20['tokens']!=prefix['policy_token_ids'] or p20['positions']!=prefix['policy_position_ids'] or p20['segments']!=prefix['prefix_segments'] or p20['policy_tokens']!=prefix['retained_policy_length'] or (p20['short'] and prefix['terminal'] is not True)):raise ValueError('Reconstructed Prefix20 is not exactly the existing sealed input')
 # Only signatures/coverage remain in RAM; no prefix/context record persists.
 for v in values.values():
  for key in ('tokens','positions','segments'):del v[key]
 return dict(label=wanted['label'],trajectory_id=wanted['trajectory_id'],values=values)


def empty_group(g):
 return dict(G=g,questions=0,trajectories=0,Mixed=0,all_pairs=0,correct_wrong_pairs=0,K_histogram=[0]*(g+1),lengths={str(n):dict(coverage=dict(short_trajectories=0,short_terminal_reason={'natural_eos':0,'answer':0,'final_action_budget':0},multi_turn_trajectories=0,past_retrieval_trajectories=0,past_correction_trajectories=0,selected_policy_tokens=0,past_retrieval_tokens=0,past_correction_tokens=0,past_retrieval_events=0,later_turn_policy_tokens=0,max_selected_context_tokens=0),token_context_ambiguous_clusters=0,representations={name:dict(distinct_histogram=[0]*(g+1),duplicate_pairs=0,conflict_pairs=0,any_duplicate_questions=0,all_identical_questions=0,Mixed_questions_with_conflict=0,excess_duplicate_trajectories=0,trajectories_with_duplicate_peer=0) for name in ('policy_tokens','full_causal_input')}) for n in LENGTHS})
def empty():return {k:empty_group(g) for k,g in [('authentic_G10',10),('original_G5',5),('newseed_G5',5)]}
def add(group,acc):
 g=len(group);k=sum(row['label'] for row in group);mixed=0<k<g;acc['questions']+=1;acc['trajectories']+=g;acc['Mixed']+=mixed;acc['all_pairs']+=g*(g-1)//2;acc['correct_wrong_pairs']+=k*(g-k);acc['K_histogram'][k]+=1
 for n in LENGTHS:
  a=acc['lengths'][str(n)];rows=[v['values'][str(n)] for v in group];c=a['coverage']
  for row in rows:
   c['short_trajectories']+=row['short'];c['short_terminal_reason'][row['terminal_reason']]+=row['short'];c['multi_turn_trajectories']+=row['multi_turn'];c['past_retrieval_trajectories']+=bool(row['past_retrieval_tokens']);c['past_correction_trajectories']+=bool(row['past_correction_tokens']);c['selected_policy_tokens']+=row['policy_tokens']
   for key in ('past_retrieval_tokens','past_correction_tokens','past_retrieval_events','later_turn_policy_tokens'):c[key]+=row[key]
   c['max_selected_context_tokens']=max(c['max_selected_context_tokens'],row['max_selected_context_tokens'])
  token_contexts=defaultdict(set)
  for row in rows:token_contexts[row['token_signature']].add(row['causal_signature'])
  a['token_context_ambiguous_clusters']+=sum(len(v)>1 for v in token_contexts.values())
  for name,key in [('policy_tokens','token_signature'),('full_causal_input','causal_signature')]:
   clusters=defaultdict(list)
   for own,row in zip(group,rows):clusters[row[key]].append(own['label'])
   s=a['representations'][name];distinct=len(clusters);s['distinct_histogram'][distinct]+=1;s['any_duplicate_questions']+=distinct<g;s['all_identical_questions']+=distinct==1;s['excess_duplicate_trajectories']+=g-distinct;conflicts=0
   for labels in clusters.values():
    size=len(labels);positive=sum(labels);s['duplicate_pairs']+=size*(size-1)//2;conflicts+=positive*(size-positive)
    if size>1:s['trajectories_with_duplicate_peer']+=size
   s['conflict_pairs']+=conflicts;s['Mixed_questions_with_conflict']+=bool(mixed and conflicts)
def combine(a,b,key=None):
 if isinstance(a,dict):return {k:combine(a[k],b[k],k) for k in a}
 if isinstance(a,list):return [combine(x,y) for x,y in zip(a,b)]
 if key=='G':
  if a!=b:raise ValueError('Aggregate G differs')
  return a
 return max(a,b) if key=='max_selected_context_tokens' else a+b
def report(acc):
 result={}
 for scope,a in acc.items():
  v={k:a[k] for k in ('G','questions','trajectories','Mixed','all_pairs','correct_wrong_pairs','K_histogram')};v['lengths']={}
  for n,row in a['lengths'].items():
   own=dict(coverage=dict(row['coverage'],short_fraction=row['coverage']['short_trajectories']/a['trajectories'],multi_turn_fraction=row['coverage']['multi_turn_trajectories']/a['trajectories'],past_retrieval_fraction=row['coverage']['past_retrieval_trajectories']/a['trajectories']),token_context_ambiguous_clusters=row['token_context_ambiguous_clusters'],representations={})
   for name,s in row['representations'].items():own['representations'][name]=dict(s,duplicate_pair_fraction=s['duplicate_pairs']/a['all_pairs'],conflict_fraction_of_correct_wrong_pairs=s['conflict_pairs']/a['correct_wrong_pairs'] if a['correct_wrong_pairs'] else None,distinguishable_correct_wrong_pair_fraction=1-s['conflict_pairs']/a['correct_wrong_pairs'] if a['correct_wrong_pairs'] else None,any_duplicate_question_fraction=s['any_duplicate_questions']/a['questions'],Mixed_question_conflict_fraction=s['Mixed_questions_with_conflict']/a['Mixed'] if a['Mixed'] else None)
   v['lengths'][n]=own
  result[scope]=v
 return result


def identity_sources():
 training=read(A/'peer-newseed8192-g10-train-v1/training-input.json');identity=read(checked(training['identity_plan']));raw=read(checked(training['raw_complete']))
 if raw['strict_EM_mismatches']!=0 or raw['final_labels_or_rollouts_read'] is not False or len(identity['groups'])!=8192 or identity['summary']['labels_consulted'] is not False:raise ValueError('Exact accepted training8192 required')
 canonical_groups={g['qid']:g for g in read(checked(raw['outputs']['groups']))['train']};groups=[]
 for g in identity['groups']:
  own=canonical_groups[g['qid']]
  if (g['question_hash']!=own['question_hash'] or g['trajectory_ids']!=own['trajectory_ids'] or len(own['trajectory_ids'])!=10 or len(set(own['trajectory_ids']))!=10 or len(own['labels'])!=10 or any(type(y) is not int or y not in (0,1) for y in own['labels'])):raise ValueError('Authentic own G10 identities/EM required')
  groups.append(own)
 return training,identity,raw,groups


def prepare(commit):
 if sys.platform=='darwin':raise ValueError('Real source metadata stays remote')
 TARGET.mkdir(parents=True,exist_ok=True);training,identity,raw,groups=identity_sources();events=read(checked(raw['outputs']['source-events']));wanted={g['qid'] for g in groups};work=[]
 for event in events['events']:
  seen=set()
  for i,batch in enumerate(event['batches']):
   qids=sorted(wanted.intersection(r['qid'] for r in read(checked(batch['input']))['rows']))
   if not qids:continue
   if seen.intersection(qids):raise ValueError('Repeated own question within actual event')
   seen.update(qids);work.append(dict(event=event['event'],index=i,source_batch_sha256=digest(batch),selected_questions=len(qids),selected_qids_sha256=digest(qids)))
  if seen!=wanted:raise ValueError('Selected8192 missing original/new actual source')
 c=dict(role='FROZEN_TRAIN8192_CAUSAL_PREFIX_LENGTH_AGGREGATE_DIAGNOSTIC',source_commit=commit,identity=training['identity_plan'],raw_complete=training['raw_complete'],raw_groups=raw['outputs']['groups'],source_events=raw['outputs']['source-events'],P20_reference=fp(A/'peer8192-g10-learnability-20261021/output/complete.json'),work_plan=work,lengths=list(LENGTHS),CPU=2,address_space_limit_bytes=3*1024**3,wall_budget_seconds=20*60,only_aggregate_checkpoints_persisted=True,signature_cache_RAM_only=True,all_original8192_preserved=True,new_rollouts_generated=0,actor_updates=0,probe_updates=0,final_or_validation_read=False)
 immutable(TARGET/'config.json',c);return dict(status='FROZEN8192_LONG_PREFIX_CPU_DIAGNOSTIC_PREPARED',config=fp(TARGET/'config.json'),source_full_compressed_bytes=sum(batch['full']['bytes'] for e in events['events'] for batch in e['batches']),work_batches=len(work),questions=8192,trajectories=81920)


def source_records(batch,wanted):
 before={k:stamp(checked(batch[k])) for k in ('input','prefix','full')};prefixes={}
 with gzip.open(batch['prefix']['path'],'rt') as f:
  for line in f:
   row=json.loads(line);tid=row['trajectory_id']
   if tid in wanted:
    if tid in prefixes:raise ValueError('Own prefix repeats')
    prefixes[tid]=row
 if set(prefixes)!=set(wanted):raise ValueError('Actual selected Prefix20 coverage missing')
 seen=set()
 with gzip.open(batch['full']['path'],'rt') as f:
  for line in f:
   row=json.loads(line);tid=row['trajectory_id']
   if tid not in wanted:continue
   if tid in seen:raise ValueError('Own full record repeats')
   seen.add(tid);yield tid,qualify_row(prefixes[tid],row,wanted[tid])
 if seen!=set(wanted) or before!={k:stamp(batch[k]['path']) for k in before}:raise ValueError('Full coverage/source stability differs')


def analyze(config_sha):
 if sys.platform=='darwin':raise ValueError('Raw records remain remote')
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));start=time.monotonic();cfg=fp(TARGET/'config.json')
 if cfg['sha256']!=config_sha:raise ValueError('Frozen diagnostic config changed')
 c=read(cfg['path']);training,identity,raw,groups=identity_sources()
 if training['identity_plan']!=c['identity'] or training['raw_complete']!=c['raw_complete'] or raw['outputs']['groups']!=c['raw_groups']:raise ValueError('Frozen training input bindings changed')
 if (TARGET/'complete.json').exists():
  done=read(TARGET/'complete.json')
  if done['config']!=cfg or done['status']!=STATUS:raise ValueError('Existing completion belongs to another analysis')
  for ref in done['aggregate_units']:checked(ref)
  return done
 lookup={g['qid']:g for g in groups};wanted_q=set(lookup);events=read(checked(c['source_events']));batches={e['event']:e['batches'] for e in events['events']};works={e:[] for e in batches}
 for work in c['work_plan']:
  batch=batches[work['event']][work['index']]
  if digest(batch)!=work['source_batch_sha256']:raise ValueError('Work source binding changed')
  qids=sorted(wanted_q.intersection(row['qid'] for row in read(checked(batch['input']))['rows']))
  if len(qids)!=work['selected_questions'] or digest(qids)!=work['selected_qids_sha256']:raise ValueError('Pre-label work identities changed')
  works[work['event']].append((work,batch,qids))
 (TARGET/'units').mkdir(exist_ok=True);done_q=set();unit_paths=[]
 for work,batch,qids in works['newseed20261021']:
  path=TARGET/'units'/('new-%04d.json'%work['index'])
  if path.exists():
   v=read(path)
   if v['config']!=cfg or v['work']!=work:raise ValueError('Existing aggregate belongs to different inputs')
   for p,saved in v['source_stamps'].items():
    if stamp(p)!=saved:raise ValueError('Completed aggregate source changed')
   done_q.update(qids);unit_paths.append(path)
 old_cache={};old_proofs={};matched=0
 atomic(TARGET/'controller.json',dict(pid=os.getpid(),config=cfg,status='CPU2_FROZEN_FULL_JOURNAL_PREFIX_RECONSTRUCTION',CPU_affinity=list(os.sched_getaffinity(0)),address_space_limit_bytes=3*1024**3,already_accepted_questions=len(done_q),time=time.time()))
 for work,batch,qids in works['original']:
  own=[q for q in qids if q not in done_q]
  if not own:continue
  wanted={tid:dict(qid=q,question_hash=lookup[q]['question_hash'],trajectory_id=tid,label=y) for q in own for tid,y in zip(lookup[q]['trajectory_ids'][:5],lookup[q]['labels'][:5])}
  for tid,row in source_records(batch,wanted):old_cache[tid]=row;matched+=1
  for q in own:old_proofs[q]={batch[k]['path']:stamp(batch[k]['path']) for k in ('input','prefix','full')}
  atomic(TARGET/'progress.json',dict(stage='REBUILDING_ORIGINAL_G5_SIGNATURE_CACHE_IN_RAM',old_selected_trajectories=matched,already_accepted_questions=len(done_q),expected_questions=8192,per_trajectory_signatures_persisted=False,time=time.time()))
  if time.monotonic()-start>c['wall_budget_seconds']:atomic(TARGET/'partial.json',dict(status='CPU_WALL_BUDGET_REACHED_RESUMABLE',accepted_questions=len(done_q),phase='old_cache',time=time.time()));return dict(status='PARTIAL_RESUMABLE',accepted_questions=len(done_q))
 for work,batch,qids in works['newseed20261021']:
  path=TARGET/'units'/('new-%04d.json'%work['index'])
  if path.exists():continue
  wanted={tid:dict(qid=q,question_hash=lookup[q]['question_hash'],trajectory_id=tid,label=y) for q in qids for tid,y in zip(lookup[q]['trajectory_ids'][5:],lookup[q]['labels'][5:])};fresh={tid:row for tid,row in source_records(batch,wanted)};acc=empty();stamps={batch[k]['path']:stamp(batch[k]['path']) for k in ('input','prefix','full')}
  for q in qids:
   g=lookup[q];original=[old_cache.pop(t) for t in g['trajectory_ids'][:5]];new=[fresh[t] for t in g['trajectory_ids'][5:]]
   add(original+new,acc['authentic_G10']);add(original,acc['original_G5']);add(new,acc['newseed_G5']);stamps.update(old_proofs.pop(q))
  value=dict(config=cfg,work=work,statistics=acc,questions=len(qids),trajectories=10*len(qids),P20_tokens_positions_contexts_exact_matches=10*len(qids),source_stamps=stamps,only_aggregates=True,source_full_and_prefix_SHA_verified=True)
  immutable(path,value);done_q.update(qids);unit_paths.append(path);atomic(TARGET/'progress.json',dict(stage='TRUE_G10_LONG_PREFIX_BATCH_AGGREGATES_ACCEPTED',accepted_questions=len(done_q),accepted_trajectories=10*len(done_q),expected_questions=8192,accepted_units=len(unit_paths),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,time=time.time()))
  if time.monotonic()-start>c['wall_budget_seconds']:atomic(TARGET/'partial.json',dict(status='CPU_WALL_BUDGET_REACHED_RESUMABLE',accepted_questions=len(done_q),phase='new_units',time=time.time()));return dict(status='PARTIAL_RESUMABLE',accepted_questions=len(done_q))
 if len(done_q)!=8192 or old_cache:raise ValueError('Whole authentic8192/G10 coverage incomplete')
 aggregate=empty();proofs=[]
 for work,batch,qids in works['newseed20261021']:
  path=TARGET/'units'/('new-%04d.json'%work['index']);v=read(path)
  if v['config']!=cfg or v['work']!=work or v['questions']!=len(qids) or v['P20_tokens_positions_contexts_exact_matches']!=10*len(qids):raise ValueError('Aggregate source/count mismatch')
  for p,saved in v['source_stamps'].items():
   if stamp(p)!=saved:raise ValueError('Frozen source changed before final acceptance')
  aggregate=combine(aggregate,v['statistics']);proofs.append(fp(path))
 result=report(aggregate);reference=read(checked(c['P20_reference']))
 for key,name in [('policy_token_only','policy_tokens'),('full_causal_input','full_causal_input')]:
  actual=aggregate['authentic_G10']['lengths']['20']['representations'][name];prior=reference[key]
  if actual['duplicate_pairs']!=prior['identical_input_pairs'] or actual['conflict_pairs']!=prior['identical_input_opposite_EM_pairs'] or actual['Mixed_questions_with_conflict']!=prior['true_G10_Mixed_questions_with_input_conflict']:raise ValueError('P20 collision counts differ from separately accepted existing input analysis')
 final=dict(status=STATUS,config=cfg,statistics=result,aggregate_units=proofs,P20_reference=c['P20_reference'],P20_previous_collision_counts_exact=True,P20_tokens_positions_contexts_exact_matches=81920,questions=8192,trajectories=81920,all_8192_original_identities_unchanged=True,longer_prefixes_are_actual_observed_policy_tokens_only=True,post_cut_observations_and_discarded_generation_excluded=True,no_long_prefix_probe_trained=True,uniqueness_does_not_imply_learnability=True,per_trajectory_signatures_persisted=False,raw_or_reconstructed_rows_persisted=False,model_or_feature_tensor_read=False,final_or_validation_read=False,new_rollouts_generated=0,actor_updates=0,probe_updates=0,CPU_affinity=list(os.sched_getaffinity(0)),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_seconds=time.monotonic()-start,time=time.time());immutable(TARGET/'complete.json',final);return final


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','analyze'),required=True);p.add_argument('--source-commit');p.add_argument('--config-sha256');a=p.parse_args();TARGET.mkdir(exist_ok=True,parents=True)
 with (TARGET/'analysis.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:v=prepare(a.source_commit) if a.action=='prepare' else analyze(a.config_sha256);print(json.dumps(dict(status=v['status'],questions=v.get('questions'),trajectories=v.get('trajectories'),elapsed_seconds=v.get('elapsed_seconds')),sort_keys=True),flush=True)
  except BaseException as exc:atomic(TARGET/'failure.json',dict(type=type(exc).__name__,message=str(exc)[:1500],time=time.time()));raise
