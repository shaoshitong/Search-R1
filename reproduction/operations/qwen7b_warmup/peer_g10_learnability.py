"""Remote-CPU learnability of the fixed NQ8192 authentic old5+new5 bank.

Only accepted own-EM metadata and sealed Prefix20 tokens/contexts are read.
No hidden/model/optimizer/probe score/final files and no scheduler operations.
"""
import argparse,ast
from collections import Counter,defaultdict
from fractions import Fraction
import fcntl,gzip,hashlib,json,math,os,random,sys,time,types
from pathlib import Path
from . import g10_grouping as grouping
from . import replay_probe_layers as replay

STATUS='FIXED_PEER8192_AUTHENTIC_G10_VIEWS_AND_CAUSAL_INPUT_LEARNABILITY_COMPLETE'
Q=8192

def canonical(v):return replay.canonical(v)
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):return replay.fingerprints(p)
def checked(v):
 p=Path(v['path']);a=ref(p)
 if not p.is_absolute() or p.resolve()!=p or not str(p).startswith('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/') or any(a[k]!=v[k] for k in ('bytes','sha256')):
  raise ValueError('Actual remote immutable source SHA/size differs')
 return p

def choose(n,k):return 0 if k<0 or n<k else math.factorial(n)//(math.factorial(k)*math.factorial(n-k))
def entropy(k,n):
 if not k or k==n:return 0.
 p=k/n;return -p*math.log(p)-(1-p)*math.log1p(-p)
def fraction(n,d):return n/d if d else None

def causal_signature(row):
 """Exactly the conceptual frozen actor/probe input, without identity/seed/locators.

 All real full causal contexts retain their positions0..L-1 and attention1.
 The 293 gathered slots retain their own masks/positions and original order.
 """
 own=replay.normalize_prefix(row,row['trajectory_id'],row['qid'],True,max_context_tokens=16384)
 q,p=own['question_length'],own['policy_length'];mask=[i<q or replay.QMAX<=i<replay.QMAX+p for i in range(replay.WIDTH)]
 positions=[0]*replay.WIDTH;positions[:q]=range(q);positions[replay.QMAX:replay.QMAX+p]=own['policy_position_ids']
 segments=[]
 for seg in own['segments']:
  n=len(seg['context_tokens']);segments.append(dict(tokens=seg['context_tokens'],causal_positions=list(range(n)),attention_mask=[1]*n,
   gather_question=list(range(q)) if seg['capture_question'] else [],
   policy_gather_source=list(range(seg['request_length'],seg['request_length']+seg['policy_length'])),
   policy_gather_target=list(range(replay.QMAX+seg['policy_start'],replay.QMAX+seg['policy_start']+seg['policy_length']))))
 return digest(dict(model_revision=replay.REVISION,capture_layer=26,initial_prompt=own['initial_prompt_token_ids'],policy_tokens=own['policy_token_ids'],
  gathered_attention_mask=mask,gathered_position_ids=positions,causal_segments=segments)),digest(own['policy_token_ids'])

def histogram(groups,start=0,end=10):
 g=end-start;h=Counter(sum(x['labels'][start:end]) for x in groups);n=len(groups)
 return dict(questions=n,K_histogram=[h[k] for k in range(g+1)],mixed_questions=sum(v for k,v in h.items() if 0<k<g),
  mixed_fraction=sum(v for k,v in h.items() if 0<k<g)/n,allwrong_questions=h[0],allcorrect_questions=h[g])

def collision_summary(groups,signatures,key=0):
 counters=Counter();distinct=Counter();lower=0.;mixed_lower=0.;mixed_questions=0;all_cw=0
 for group in groups:
  labels=group['labels'];K=sum(labels);mixed=0<K<10;mixed_questions+=mixed;all_cw+=K*(10-K);clusters=defaultdict(list)
  for tid,y in zip(group['trajectory_ids'],labels):clusters[signatures[tid][key]].append(y)
  distinct[len(clusters)]+=1;counters['questions_with_duplicate']+=len(clusters)<10;counters['all_ten_identical']+=len(clusters)==1;counters['excess_duplicate_trajectories']+=10-len(clusters)
  conflicts=0
  for ys in clusters.values():
   n=len(ys);k=sum(ys);counters['duplicate_pairs']+=n*(n-1)//2;conflicts+=k*(n-k)
   if n>1:counters['trajectories_with_duplicate_peer']+=n
   term=n*entropy(k,n);lower+=term
   if mixed:mixed_lower+=term
  counters['opposite_EM_conflict_pairs']+=conflicts;counters['mixed_questions_with_conflict']+=bool(mixed and conflicts)
 n=len(groups);pairs=45*n;total=10*n
 return dict(questions=n,trajectories=total,distinct_inputs_per_question_histogram=[distinct[i] for i in range(1,11)],
  questions_with_any_duplicate=counters['questions_with_duplicate'],question_any_duplicate_fraction=counters['questions_with_duplicate']/n,
  all_ten_identical_questions=counters['all_ten_identical'],all_ten_identical_fraction=counters['all_ten_identical']/n,
  trajectories_with_duplicate_peer=counters['trajectories_with_duplicate_peer'],trajectory_duplicate_peer_fraction=counters['trajectories_with_duplicate_peer']/total,
  excess_duplicate_trajectories=counters['excess_duplicate_trajectories'],excess_duplicate_trajectory_fraction=counters['excess_duplicate_trajectories']/total,
  identical_input_pairs=counters['duplicate_pairs'],all_within_question_pairs=pairs,identical_input_pair_fraction=counters['duplicate_pairs']/pairs,
  identical_input_opposite_EM_pairs=counters['opposite_EM_conflict_pairs'],all_within_question_correct_wrong_pairs=all_cw,
  identical_input_fraction_of_correct_wrong_pairs=fraction(counters['opposite_EM_conflict_pairs'],all_cw),
  opposite_EM_fraction_among_identical_input_pairs=fraction(counters['opposite_EM_conflict_pairs'],counters['duplicate_pairs']),
  true_G10_Mixed_questions=mixed_questions,true_G10_Mixed_questions_with_input_conflict=counters['mixed_questions_with_conflict'],
  true_G10_Mixed_input_conflict_fraction=fraction(counters['mixed_questions_with_conflict'],mixed_questions),
  finite_sample_trajectory_BCE_minimum_nats=lower/total,finite_sample_Mixed_G10_only_BCE_minimum_nats=fraction(mixed_lower,10*mixed_questions))

def view_summary(groups,batches,builder=grouping.build_training_views):
 lookup={g['qid']:g for g in groups};hist=Counter();mixed_counts=Counter();occurrences=Counter();weights=defaultdict(Fraction);selected_digest=hashlib.sha256();perK=defaultdict(lambda:Counter());total_views=0
 for update,qids in enumerate(batches):
  own=[lookup[q] for q in qids];v=builder(own,mode='random',views_per_question=4,seed=1729,epoch=0,on_unbalance='raise',role='train')
  local=defaultdict(set)
  for row in v['groups']:
   q=row['qid'];w=Fraction(row['weight_numerator'],row['weight_denominator'])
   if w!=Fraction(1,4):raise ValueError('Actual four-view question weight must remain1/4')
   local[q].add(tuple(row['source_indices']));weights[q]+=w;hist[row['K']]+=1;mixed_counts[q]+=int(row['true_mixed']);perK[sum(lookup[q]['labels'])]['views']+=1;perK[sum(lookup[q]['labels'])]['mixed']+=int(row['true_mixed'])
   for tid,y in zip(row['trajectory_ids'],row['labels']):
    index=lookup[q]['trajectory_ids'].index(tid)
    if lookup[q]['labels'][index]!=y:raise ValueError('Grouping label does not belong to its own trajectory')
    occurrences[tid]+=1
  if set(local)!=set(qids) or any(len(x)!=4 for x in local.values()):raise ValueError('Four unique actual G5 views per question required')
  selected_digest.update(canonical([update,0,v['groups']]));total_views+=len(v['groups'])
 if set(weights)!=set(lookup) or any(w!=1 for w in weights.values()) or total_views!=4*len(groups):raise ValueError('Every question must appear once with exact total weight1')
 mixed=sum(hist[k] for k in range(1,5));n=len(groups);expected_hist=[Fraction(0)]*6;expected_mixed=Fraction(0);reference_var=0.
 for g in groups:
  k=sum(g['labels']);prob=Fraction(252-choose(k,5)-choose(10-k,5),252);expected_mixed+=prob;reference_var+=float(prob*(1-prob))*Fraction(248,4*251)
  for j in range(6):expected_hist[j]+=Fraction(choose(k,j)*choose(10-k,5-j),252)
 result=dict(mode='random',seed=1729,epoch=0,questions=n,updates=len(batches),questions_per_update=len(batches[0]),
  views=total_views,unique_views_per_question=4,weight_per_view='1/4',total_weight_per_question='1',per_question_total_weight_verified=True,
  realized_G5_K_histogram=[hist[i] for i in range(6)],realized_G5_K_weighted_mass=[str(Fraction(hist[i],4)) for i in range(6)],
  realized_Mixed_views=mixed,realized_nonMixed_views=total_views-mixed,realized_weighted_Mixed_mass=str(Fraction(mixed,4)),realized_weighted_nonMixed_mass=str(Fraction(total_views-mixed,4)),
  realized_Mixed_fraction=mixed/total_views,realized_weighted_Mixed_fraction=mixed/(4*n),count_fraction_equals_weighted_fraction_because_four_equal_views_per_question=True,
  questions_by_number_Mixed_of_four_views=[sum(mixed_counts[q]==i for q in lookup) for i in range(5)],
  trajectory_occurrence_count_histogram=[sum(occurrences[t]==i for g in groups for t in g['trajectory_ids']) for i in range(5)],
  theoretical_uniform252_expected_weighted_G5_K_mass=[str(v) for v in expected_hist],theoretical_uniform252_expected_G5_K_fraction=[float(v/n) for v in expected_hist],
  theoretical_uniform252_expected_weighted_Mixed_mass=str(expected_mixed),theoretical_uniform252_expected_Mixed_fraction=float(expected_mixed/n),
  realized_minus_expected_Mixed_fraction=mixed/total_views-float(expected_mixed/n),uniform_without_replacement_reference_standard_error=math.sqrt(reference_var)/n,
  reference_standard_error_is_sampling_design_reference_not_test_confidence_interval=True,
  per_source_G10_K=[dict(K=k,source_questions=sum(sum(g['labels'])==k for g in groups),actual_views=perK[k]['views'],actual_Mixed_views=perK[k]['mixed'],
   actual_G5_Mixed_fraction=fraction(perK[k]['mixed'],perK[k]['views']),uniform252_G5_Mixed_probability=(252-choose(k,5)-choose(10-k,5))/252) for k in range(11)],
  selected_update_ordered_views_sha256=selected_digest.hexdigest(),G10_Mixed_does_not_imply_each_G5_view_is_Mixed=True)
 return result,occurrences

def pure_training_plan(source,groups):
 tree=ast.parse(Path(source).read_text());definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_plan']
 if len(definitions)!=1:raise ValueError('Unique actual training build_plan source required')
 constants={}
 for n in tree.body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('Q','BATCH','UPDATES'):constants[n.targets[0].id]=ast.literal_eval(n.value)
 if constants!={'Q':8192,'BATCH':32,'UPDATES':256}:raise ValueError('Actual declared B32x256 protocol differs')
 scope=dict(random=random,raw=types.SimpleNamespace(digest=digest),**constants)
 exec(compile(ast.fix_missing_locations(ast.Module(body=definitions,type_ignores=[])),'<SHA-bound-training-build-plan>','exec'),scope)
 return scope['build_plan'](groups,seed=1729)

def validate_factory(source):
 tree=ast.parse(Path(source).read_text());fn=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_config']
 if len(fn)!=1:raise ValueError('Unique actual training make_config factory required')
 values=[n.value for n in ast.walk(fn[0]) if isinstance(n,ast.keyword) and n.arg=='grouping']
 if len(values)!=1 or not isinstance(values[0],ast.Call):raise ValueError('Actual factory grouping missing')
 kwargs={k.arg:ast.literal_eval(k.value) for k in values[0].keywords}
 if kwargs!=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise'):raise ValueError('Actual factory no longer uses requested four random views')
 return kwargs

def analyze(config,output):
 if sys.platform=='darwin':raise ValueError('Real raw/P20 analysis must run only on approved remote CPU storage')
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 with (output/'analysis.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.time()
  if (output/'complete.json').exists():return read(output/'complete.json')
  sources={k:config[k] for k in ('preparation','raw_complete','grouping_code','training_plan_source','training_factory_source')}
  for v in sources.values():checked(v)
  if ref(grouping.__file__)!=config['grouping_code']:raise ValueError('Loaded grouping source differs from published experiment')
  prep=read(config['preparation']['path']);raw=read(config['raw_complete']['path'])
  if raw['status']!='RECEIVED_NEWSEED5_PLUS_CANONICAL_OLD5_RAW_G10_STRICT_EM_CAUSAL_SEEDS_ACCEPTED' or raw['strict_EM_mismatches']!=0 or raw['actor_updates']!=0 or raw['final_labels_or_rollouts_read'] is not False:raise ValueError('Actual own-EM accepted raw G10 required')
  checked(raw['outputs']['groups']);all_groups=read(raw['outputs']['groups']['path'])['train'];by={g['qid']:g for g in all_groups};selection_refs=prep['selections'];selected={};prefix_files=[];identity_ref=None;source_declarations={}
  for item in selection_refs:
   checked(item);selection=read(item['path'])
   if selection['source_ingest']!=config['raw_complete'] or selection['real_trajectories_per_question']!=10 or selection['label_blind_selection'] is not True:raise ValueError('Wrong authentic peer selection')
   own=selection['question_plan'];checked(own)
   if identity_ref is not None and identity_ref!=own:raise ValueError('Four shards must share exactly one question identity plan')
   identity_ref=own;prefix_files+=selection['prefix_files']
   for row in selection['trajectories']:
    if row['trajectory_id'] in source_declarations:raise ValueError('Shard declarations duplicate a trajectory')
    source_declarations[row['trajectory_id']]=row['qid']
  identity=read(identity_ref['path'])
  if identity['raw_complete']!=config['raw_complete'] or identity['summary']['labels_consulted'] is not False or identity['summary']['selected_questions']!=Q:raise ValueError('Explicit precommitted8192 identity required')
  groups=[]
  for g in identity['groups']:
   own=by[g['qid']]
   if set(g)!={'qid','question_hash','trajectory_ids'} or g['question_hash']!=own['question_hash'] or g['trajectory_ids']!=own['trajectory_ids'] or len(own['trajectory_ids'])!=10 or len(set(own['trajectory_ids']))!=10 or len(own['labels'])!=10 or any(type(y) is not int or y not in (0,1) for y in own['labels']):raise ValueError('Actual own identity/EM binding differs')
   if json.loads(g['qid'])[0]!='nq':raise ValueError('This accepted subset is NQ-only')
   if ([x['trajectory_ids'] for x in own['real_G5_events']]!=[own['trajectory_ids'][:5],own['trajectory_ids'][5:]] or
       [x['labels'] for x in own['real_G5_events']]!=[own['labels'][:5],own['labels'][5:]] or
       [x['event'] for x in own['real_G5_events']]!=['original','newseed20261021']):raise ValueError('Original5/new5 ordering or own EM differs')
   groups.append(own)
  if len(groups)!=Q or digest([g['qid'] for g in groups])!=identity['summary']['ordered_qid_sha256'] or source_declarations!={t:g['qid'] for g in groups for t in g['trajectory_ids']}:raise ValueError('Full selected8192/81920 coverage differs')
  del by,all_groups
  validate_factory(config['training_factory_source']['path']);plan=pure_training_plan(config['training_plan_source']['path'],groups)
  if len(plan['batches'])!=256 or any(len(b)!=32 for b in plan['batches']) or set(plan['batch_epochs'])!={0}:raise ValueError('Actual B32x256 epoch0 source plan differs')
  actual_config=dict(status='NOT_PRESENT_PROTOCOL_BOUND_TO_PUBLISHED_FACTORY')
  p=Path(config['optional_actual_fit_config'])
  if p.exists():
   v=read(p)
   if v['grouping']!=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise') or v['update_budget']!=256 or v['global_questions']!=32:raise ValueError('Actual fit config differs from analyzed protocol')
   actual_config=dict(status='ACTUAL_FIT_CONFIG_PRESENT_AND_PROTOCOL_MATCHED',source=ref(p))
  signatures={};counts=Counter();source_proofs=[]
  for item in prefix_files:
   checked(item);before=ref(item['path']);source_proofs.append(before)
   with gzip.open(item['path'],'rt') as f:
    for line in f:
     row=json.loads(line);tid=row['trajectory_id']
     if tid not in source_declarations or row['qid']!=source_declarations[tid] or tid in signatures:raise ValueError('Prepared prefix repeats or leaves selected cohort')
     if any(k in row for k in ('strict_em','labels','response_token_ids','ground_truth')):raise ValueError('Prefix-only projection contains forbidden full/label payload')
     signatures[tid]=causal_signature(row);counts['short_prefix']+=len(row['policy_token_ids'])<20;counts['multisegment_trajectories']+=len(row['prefix_segments'])>1;counts['segments']+=max(1,len(row['prefix_segments']))
     if len(signatures)%2048==0:replay.atomic(output/'progress.json',dict(stage='REAL_PREFIX20_CAUSAL_SIGNATURES',trajectories=len(signatures),expected_trajectories=81920,time=time.time()))
   if before!=ref(item['path']):raise ValueError('Prefix input changed during CPU analysis')
  if set(signatures)!=set(source_declarations):raise ValueError('Incomplete prefix coverage')
  views,occ=view_summary(groups,plan['batches']);weighted_lower=0.;ambiguous_policy_clusters=0
  for g in groups:
   clusters=defaultdict(lambda:[0,0]);token_clusters=defaultdict(set)
   for tid,y in zip(g['trajectory_ids'],g['labels']):
    n=occ[tid];clusters[signatures[tid][0]][0]+=n*y;clusters[signatures[tid][0]][1]+=n;token_clusters[signatures[tid][1]].add(signatures[tid][0])
   for k,n in clusters.values():
    if n:weighted_lower+=n*entropy(k,n)
   ambiguous_policy_clusters+=sum(len(x)>1 for x in token_clusters.values())
  views['actual_question_weighted_trajectory_BCE_empirical_minimum_nats']=weighted_lower/(20*Q)
  result=dict(status=STATUS,source_preparation=config['preparation'],raw_complete=config['raw_complete'],raw_groups=raw['outputs']['groups'],identity_plan=identity_ref,
   grouping_code=config['grouping_code'],training_plan_source=config['training_plan_source'],training_factory_source=config['training_factory_source'],actual_fit_config=actual_config,
   questions=Q,unique_qids=len({g['qid'] for g in groups}),unique_question_hashes=len({g['question_hash'] for g in groups}),duplicate_question_hash_questions=Q-len({g['question_hash'] for g in groups}),
   real_trajectories=10*Q,domain_scope='NQ-only precommitted independent8192 enrichment; not unseen initial-warmup questions',
   ordered_qid_sha256=identity['summary']['ordered_qid_sha256'],ordered_trajectory_sha256=identity['summary']['trajectory_ids_sha256'],
   original_G5=histogram(groups,0,5),newseed_G5=histogram(groups,5,10),authentic_G10=histogram(groups),
   old_new_K_transition=[[sum(sum(g['labels'][:5])==i and sum(g['labels'][5:])==j for g in groups) for j in range(6)] for i in range(6)],
   actual_four_random_G5_views=views,full_causal_input=collision_summary(groups,signatures,0),policy_token_only=collision_summary(groups,signatures,1),
   identical_policy_token_clusters_with_different_causal_inputs=ambiguous_policy_clusters,actual_context_counts=dict(counts),prefix_sources=source_proofs,
   conceptual_input_definition='official frozen revision/L26; initial prompt; all full saved causal contexts and positions0..L-1; context attention1; question/policy gather indices;293-slot masks and positions; own Prefix20 token IDs/length. Ignores qid/tid/seed/run/file locators.',
   BCE_bound_scope='Finite observed training-sample lower bound for one fixed deterministic trajectory-scoring function of exactly the declared signature, within-question conflicts only. Not a test Bayes bound, not a Mixed BA ceiling, and not a lower bound for pre-update losses evaluated at changing model checkpoints. Extra observed inputs/batch numeric artifacts invalidate treating those records as exactly identical model inputs.',
   trajectory_BCE_reduction='G5 mean over5; views each1/4; mean over original questions. Bound concerns only trajectory BCE, not total event/rank loss.',
   question_selection_changed=False,own_EM_labels_read=True,labels_source='SHA-accepted raw G10 groups; no raw full-response/EM payload reopened',
   hidden_or_tensor_files_read=False,model_or_probe_loaded=False,probe_scores_read=False,final_pool_read=False,source_writer_modified=False,actor_updates=0,probe_updates=0,
   elapsed_seconds=time.time()-started,time=time.time())
  for v in sources.values():checked(v)
  checked(raw['outputs']['groups'])
  for v in selection_refs:checked(v)
  if ref(identity_ref['path'])!=identity_ref:raise ValueError('Identity plan changed during analysis')
  replay.atomic(output/'complete.json',result);return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 try:v=analyze(read(a.config),a.output)
 except BaseException as e:
  b=Path(a.output);b.mkdir(parents=True,exist_ok=True);replay.atomic(b/'failure.json',dict(error_type=type(e).__name__,error=str(e)[:1000],time=time.time()));raise
 print(json.dumps(v),flush=True)
if __name__=='__main__':main()
