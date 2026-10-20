"""Remote read-only sealed prefix/ownEM diagnostic; no tensor/fullresponse IO."""
import pathlib,json,gzip,hashlib,time,math
from collections import defaultdict,Counter
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
base=B/'algorithm-improvement-20261021/training-extraG5-matched6528-20261021/materialized'
root=base/'sampling/shard-00'
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 p=pathlib.Path(ref['path']);assert p.is_absolute() and p.resolve()==p and str(p).startswith(str(base)+'/')
 assert fp(p)=={k:ref[k] for k in ('path','bytes','sha256')},'Fresh sealed file SHA differs: '+str(p)
 return p
def rows(p):
 opener=gzip.open if str(p).endswith('.gz') else open
 with opener(p,'rt') as f:return [json.loads(line) for line in f]
def ratio(a,b):return a/b if b else None
def entropy(positive,n):
 if not positive or positive==n:return 0.
 p=positive/n;return -p*math.log(p)-(1-p)*math.log1p(-p)
def context_at(row,n):
 segments=[]
 for segment in row['segments']:
  take=min(segment['policy_length'],max(0,n-segment['policy_start']))
  if take:
   segments.append([segment['turn_index'],segment['policy_start'],segment['position_start'],
    segment['context_tokens'][:segment['request_length']+take]])
 return json.dumps([row['position_ids'][:n],segments],sort_keys=True,separators=(',',':'))
config=read(base/'config.json');config_ref=fp(base/'config.json')
assert config['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28'
assert config['capture_prefix_tokens']==20 and config['warmup_only'] is True and config['actor_updates_authorized'] is False
assert config['extra_response_indices']==[5,6,7,8,9]
paths=sorted((root/'batches').glob('batch-*/complete.json'))[:4]
assert paths,'No sealed full-prefix plus ownEM batch yet; never inspect active partial files'
bound=[fp(p) for p in paths];groups=defaultdict(list);sources=[];seen_tids=set()
for path,descriptor in zip(paths,bound):
 receipt=read(path);assert fp(path)==descriptor
 assert receipt['run_id']==config['run_id'] and receipt['model_revision']==config['model_revision'] and receipt['actor_optimizer_updates']==0
 assert receipt['questions']<=128 and receipt['trajectories']==5*receipt['questions']
 batch=path.parent;batch_input=base/'inputs/shard-00'/(batch.name+'.json')
 assert fp(batch_input)['sha256']==receipt['input_sha256']
 batch_config=fp(batch/'config.json')
 for key in ('prefix','full'):
  assert receipt[key]['config_sha256']==batch_config['sha256'] and receipt[key]['run_id']==config['run_id']
  assert receipt[key]['questions']==receipt['questions'] and receipt[key]['trajectories']==receipt['trajectories']
 prefix_ref=receipt['prefix']['prefix'];em_ref=receipt['full']['own_em']
 prefix=rows(checked(prefix_ref));labels=rows(checked(em_ref));by={r['trajectory_id']:r for r in labels}
 assert len(prefix)==len(labels)==len(by)==receipt['trajectories']
 assert set(by)=={r['trajectory_id'] for r in prefix}
 local_qids=set()
 for row in prefix:
  tid=row['trajectory_id'];label=by[tid];qid=row['qid'];tokens=row['policy_token_ids']
  assert tid not in seen_tids and label['qid']==qid and label['prefix_sha256']==row['prefix_sha256']
  assert type(label['strict_em']) is int and label['strict_em'] in (0,1)
  assert row['run_id']==config['run_id'] and row['actor_version']=='official7b-'+config['model_revision']
  assert row['response_index'] in range(5,10) and row['split']=='train'
  assert isinstance(tokens,list) and len(tokens)<=20 and all(type(t) is int for t in tokens)
  assert len(tokens)==20 or row['terminal'] is True
  canonical={k:v for k,v in row.items() if k not in ('prefix_sha256','feature_rank','native_features')}
  # NativeFeatureWriter may attach reference metadata after the original prefix
  # hash: authentic file SHA and ownEM prefix_sha linkage are authoritative here.
  seen_tids.add(tid);local_qids.add(qid)
  groups[qid].append(dict(tokens=tokens,position_ids=row['policy_position_ids'],label=label['strict_em'],
   response_index=row['response_index'],question_hash=row['question_hash'],prompt=row['initial_prompt_token_ids'],
   segments=row['prefix_segments']))
 assert len(local_qids)==receipt['questions']
 assert fp(path)==descriptor;checked(prefix_ref);checked(em_ref)
 sources.append(dict(batch=path.parent.name,complete=descriptor,input=fp(batch_input),config=batch_config,
  prefix=prefix_ref,own_em=em_ref,questions=receipt['questions'],trajectories=receipt['trajectories']))
assert len(groups)==sum(s['questions'] for s in sources)
for group in groups.values():
 assert len(group)==5 and {r['response_index'] for r in group}==set(range(5,10))
 assert len({r['question_hash'] for r in group})==1 and len({tuple(r['prompt']) for r in group})==1
Q=len(groups);N=5*Q;K=Counter(sum(r['label'] for r in group) for group in groups.values())
mixed=sum(K[k] for k in range(1,5));all_cw=sum(k*(5-k)*v for k,v in K.items())
statistics={}
for prefix_length in (5,10,20):
 hist=Counter();duplicate_pairs=conflict_pairs=duplicated_questions=all_identical=conflicted_mixed=0
 excess=has_duplicate_peer=short=0;ce=ce_mixed=0.;context_ambiguous_clusters=0
 for group in groups.values():
  clusters=defaultdict(list)
  for row in group:
   tokens=tuple(row['tokens'][:prefix_length]);clusters[tokens].append(row)
   short+=len(row['tokens'])<prefix_length
  distinct=len(clusters);hist[distinct]+=1;duplicated_questions+=distinct<5;all_identical+=distinct==1;excess+=5-distinct
  local_conflicts=0;is_mixed=0<sum(r['label'] for r in group)<5
  for own in clusters.values():
   n=len(own);positive=sum(row['label'] for row in own)
   duplicate_pairs+=n*(n-1)//2;local_conflicts+=positive*(n-positive)
   if n>1:has_duplicate_peer+=n
   term=n*entropy(positive,n);ce+=term
   if is_mixed:ce_mixed+=term
   # Different RoPE/context can make identical policy tokens insufficient to
   # prove identical native hidden inputs. Count this caveat without reading tensors.
   contexts={context_at(r,prefix_length) for r in own}
   if n>1 and len(contexts)>1:context_ambiguous_clusters+=1
  conflict_pairs+=local_conflicts;conflicted_mixed+=bool(local_conflicts and is_mixed)
 statistics[str(prefix_length)]=dict(questions=Q,trajectories=N,distinct_prefix_count_histogram={str(i):hist[i] for i in range(1,6)},
  any_duplicate_prefix_questions=duplicated_questions,any_duplicate_prefix_question_fraction=ratio(duplicated_questions,Q),
  all_five_identical_questions=all_identical,all_five_identical_question_fraction=ratio(all_identical,Q),
  excess_duplicate_trajectories=excess,excess_duplicate_trajectory_fraction=ratio(excess,N),
  trajectories_with_duplicate_peer=has_duplicate_peer,trajectory_duplicate_peer_fraction=ratio(has_duplicate_peer,N),
  equal_prefix_unordered_pairs=duplicate_pairs,all_within_question_unordered_pairs=10*Q,equal_prefix_pair_fraction=ratio(duplicate_pairs,10*Q),
  equal_prefix_opposite_EM_pairs=conflict_pairs,all_within_question_correct_wrong_pairs=all_cw,
  same_prefix_fraction_of_correct_wrong_pairs=ratio(conflict_pairs,all_cw),
  true_Mixed_questions=mixed,true_Mixed_with_conflicting_prefix=conflicted_mixed,
  true_Mixed_conflicting_prefix_question_fraction=ratio(conflicted_mixed,mixed),
  token_exact_empirical_trajectory_BCE_lower_bound_nats=ratio(ce,N),
  token_exact_empirical_Mixed_only_trajectory_BCE_lower_bound_nats=ratio(ce_mixed,5*mixed),
  shorter_naturally_terminal_trajectories=short,token_equal_clusters_with_different_recorded_context_or_positions=context_ambiguous_clusters)
assert fp(base/'config.json')==config_ref
for descriptor in bound:assert fp(descriptor['path'])==descriptor
result=dict(status='SEALED_NEW_EXTRA_G5_PREFIX_TOKEN_OWN_EM_SMALL_SAMPLE_DIAGNOSTIC_COMPLETE',time=time.time(),
 source_config=config_ref,sources=sources,job_array='743115_0',actual_shard_controller=read(root/'controller.json'),
 questions=Q,trajectories=N,batches=len(sources),K_histogram=[K[i] for i in range(6)],true_Mixed_questions=mixed,
 sampling_scope='first up to4 already-sealed batches of fixed matched6528 training cohort shard0; deterministic subset, not all169028 and not IID population evidence',
 selection_did_not_use_new_EM=True,original_G5_or_final_questions_read=False,tensor_files_read=False,full_response_files_read=False,sampler_modified=False,
 BCE_lower_bound_scope='empirical minimum for a deterministic trajectory scorer using only same-question plus exact prefix-token sequence; not a Mixed BA bound; differing hidden/context inputs can invalidate applying it to a richer native model',
 entropy_units='natural logarithm, nats/trajectory',early_terminal_handling='unpadded actual token tuple including its length; no fabricated tokens',statistics=statistics)
print(json.dumps(result))
