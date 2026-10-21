"""Resolve real shared cache/CP0 metadata; no torch/model load or GPU action."""
import argparse,array,hashlib,json,math,sys,time,zipfile
from pathlib import Path

CONFIG_SHA='fb637c02812dd070cf392a58df75e0d219bbb10d1a2d24cd6e201f3638147767'
SOURCE='805123aebf8af83a7c1c44947f39c294f7aeb624'
PROTOCOL_SHA='19f5e62fa39f4bde9790cadb29137a7e09cc91bd3fc1698a19f3a444326eed60'
SOURCE_PLAN_SHA='8d6f5b046c8ee459c529f0750c80bac63455b2663f19cdcdce48d0fddd648cbf'
PRIOR=dict(path='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full938-train-prior-decision-diagnostic-v1/train-only-prior-private.json',bytes=2155,sha256='2f76659c49fa6e280b6ef3a3ef528128d9ab29dd41c4e671951e8df5563eae5e')

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def proof(path):
 p=Path(path);h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 if proof(ref['path'])!={k:ref[k]for k in ('path','bytes','sha256')}:raise ValueError('Exact actual source SHA/bytes changed')
 return Path(ref['path'])
def immutable(path,value):
 p=Path(path);raw=canonical(value)+b'\n';p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=raw:raise ValueError('Accepted input metadata cannot be overwritten')
 else:
  tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(raw);tmp.replace(p)
def domain(v):
 if v in ('nq','NQ'):return 'NQ'
 if v in ('hotpotqa','HotpotQA'):return 'HotpotQA'
 raise ValueError('Only frozen NQ/Hotpot domains accepted')
def cp0_zip(path):
 p=Path(path)
 if not 0<p.stat().st_size<=64*1024**2:raise ValueError('Only actual small new-head CP0 accepted; no legacy model load')
 with zipfile.ZipFile(p)as z:
  items=z.infolist();stores=[x for x in items if '/data/'in x.filename];pickles=[x for x in items if x.filename.endswith('/data.pkl')]
  if len(items)>128 or sum(x.file_size for x in items)>64*1024**2 or not stores or any(x.file_size>2*1024**2 for x in stores)or len(pickles)!=1 or pickles[0].file_size>512*1024 or any(x.compress_type!=zipfile.ZIP_STORED for x in items):raise ValueError('Unexpected CP0 ZIP tensor range')
 return dict(file_bytes=p.stat().st_size,stored_tensor_entries=len(stores),largest_storage_bytes=max(x.file_size for x in stores),torch_tensor_loaded=False)

def wait_for_assets(baseline):
 paths=dict(cache_complete=baseline/'cache/complete.json',baseline_CP0=baseline/'checkpoints/cursor-000000.pt',baseline_epoch0_evaluation=baseline/'evaluations/epoch-00/complete.json')
 return paths,[key for key,path in paths.items()if not path.exists()]

def source_order(cohort,orders,plan,fit_epochs):
 fit,held=cohort['fit_qids'],cohort['holdout_qids']
 if len(fit)!=30016 or len(set(fit))!=30016 or len(held)!=5120 or len(set(held))!=5120 or set(fit)&set(held):raise ValueError('Exact disjoint30016/5120 cohort required')
 batches=plan['batches']
 if len(batches)!=938 or any(len(x)!=32 for x in batches)or [q for b in batches for q in b]!=fit:raise ValueError('Original938 ordered B32 batches changed')
 if len(orders['epochs'])!=20 or any(x!=batches for x in orders['epochs']):raise ValueError('All20 epochs must preserve original938 order')
 expected={q:plan['batch_epochs'][i]for i,b in enumerate(batches)for q in b}
 if fit_epochs!=expected:raise ValueError('Original fixed four-view epochs differ')
 return fit,held

def own_ids(groups,fit,held,selection,fixed_rows):
 by={r['qid']:r for r in groups};selected={r['qid']:r for r in selection['groups']};fixed={r['qid']:r for r in fixed_rows}
 if len(by)!=len(groups)or len(selected)!=5120 or len(selection['groups'])!=5120 or len(fixed)!=5120 or len(fixed_rows)!=5120 or set(selected)!=set(held)or set(fixed)!=set(held):raise ValueError('Distinct metadata coverage differs')
 if any('labels'in r or 'strict_em'in r for r in selection['groups']):raise ValueError('Frozen selection must remain identity-only')
 tids=[];counts=dict(NQ=0,HotpotQA=0)
 for q in fit+held:
  g=by[q]
  if len(g['trajectory_ids'])!=10 or len(set(g['trajectory_ids']))!=10 or len(g['labels'])!=10 or any(type(y)is not int or y not in(0,1)for y in g['labels']):raise ValueError('Ten genuine own IDs/strictEM required')
  tids+=g['trajectory_ids']
 if len(set(tids))!=351360 or len({by[q]['question_hash']for q in fit+held})!=35136:raise ValueError('Question/hash/trajectory duplication')
 for q in held:
  g,s,f=by[q],selected[q],fixed[q]
  if s['source_indices']!=[5,6,7,8,9]or s['trajectory_ids']!=g['trajectory_ids'][5:]or f['trajectory_ids']!=s['trajectory_ids']or f['labels']!=g['labels'][5:]or f['question_hash']!=g['question_hash']or s['question_hash']!=g['question_hash']or domain(f['domain'])!=domain(s['domain']):raise ValueError('Committed own-new5 differs')
  counts[domain(s['domain'])]+=1
 if counts!=dict(NQ=2560,HotpotQA=2560):raise ValueError('Frozen held domain counts differ')
 return tids,dict(ordered_holdout_own_new5_IDs_sha256=digest([fixed[q]['trajectory_ids']for q in held]),ordered_holdout_own_new5_strict_EM_sha256=digest([fixed[q]['labels']for q in held]),domain_counts=counts)

def resolve(base,baseline,code_commit):
 base,baseline=Path(base),Path(baseline);protocol_ref=proof(base/'family-protocol.json');protocol=read(protocol_ref['path'])
 if protocol_ref['sha256']!=PROTOCOL_SHA:raise ValueError('Only preregistered36-node family accepted')
 config_ref=proof(baseline/'config.json');c=read(config_ref['path']);pub_ref=proof(baseline/'source-publication.json');pub=read(pub_ref['path'])
 if config_ref['sha256']!=CONFIG_SHA or pub['commit']!=SOURCE or protocol['baseline_config']!=config_ref or protocol['baseline_publication']!=pub_ref:raise ValueError('Exact original805/fb637 source binding required')
 paths,missing=wait_for_assets(baseline)
 if missing:
  value=dict(status='WAITING_FOR_VERIFIED_BASELINE_CACHE_CP0',family_protocol=protocol_ref,baseline_config=config_ref,missing_actual_assets=missing,actual_asset_presence={k:p.exists()for k,p in paths.items()},consumer_packet_written=False,full_CP0_restored=False,GPU_submitted=False)
  path=base/('WAIT-'+digest(value)[:16]+'.json');immutable(path,value)
  return dict(value,receipt=proof(path))
 ready_path=base/'coarse-fork-inputs.json'
 if ready_path.exists():
  packet=read(ready_path)
  for key in ('baseline_config','baseline_publication','cache_complete','baseline_CP0','baseline_epoch0_evaluation','family_protocol','frozen_train_prior','small_checkpoint_verifier'):checked(packet[key])
  return dict(status='EXISTING_REAL_COARSE_FORK_INPUTS_PRESERVED',packet=proof(ready_path),full_CP0_restore_required_on_fork_allocation=True,GPU_submitted=False)
 for key in ('cohort','orders','fit_view_epochs','source_plan'):checked(c[key])
 if c['source_plan']['sha256']!=SOURCE_PLAN_SHA or c['grouping']!=dict(mode='random',on_unbalance='raise',seed=1729,views_per_question=4)or c['head_LR']!=1e-4 or c['EMA_decay']!=.99 or c['update_budget']!=18760:raise ValueError('Matched source order/grouping/budget differs')
 cohort=read(c['cohort']['path']);fit,held=source_order(cohort,read(c['orders']['path']),read(c['source_plan']['path']),read(c['fit_view_epochs']['path']))
 for key in ('merged_groups','internal_holdout_selection','internal_holdout_groups','identity_exclusion_acceptance'):checked(c['inputs'][key])
 exclusion=read(c['inputs']['identity_exclusion_acceptance']['path'])
 if exclusion['labels_consulted_for_selection']or exclusion['scores_consulted_for_selection']or any(x!=0 for x in exclusion['merged_fixed_evaluation_qid_and_hash_overlap'].values())or any(x!=0 for x in exclusion['holdout_fixed_evaluation_qid_and_hash_overlap'].values()):raise ValueError('Fixed evaluation identity exclusion differs')
 tids,held_lineage=own_ids(read(c['inputs']['merged_groups']['path'])['train'],fit,held,read(c['inputs']['internal_holdout_selection']['path']),read(c['inputs']['internal_holdout_groups']['path'])['internal_holdout'])
 cache_ref=proof(paths['cache_complete']);cache=read(cache_ref['path'])
 if cache['status']!='FROZEN_E1_LARGE_NATIVE_CACHE_COMPLETE'or cache['config_sha256']!=CONFIG_SHA or cache['trajectory_ids_sha256']!=digest(tids)or cache['frozen_legacy_before']!=cache['frozen_legacy_after']or cache['actor_updates']or cache['legacy_updates']or cache['actual_native_hidden_dtype']!='bfloat16':raise ValueError('Actual frozen cache lineage differs')
 n=len(tids);expected=dict(hidden=n*20*3584*2,mask=n*20,logits=n*4)
 if n!=351360 or c['memory']['hidden_shape']!=[351360,20,3584]or set(cache['files'])!=set(expected):raise ValueError('Exact351360×20×3584 cache geometry required')
 for key,size in expected.items():
  ref=cache['files'][key];p=Path(ref['path'])
  if p.parent!=baseline/'cache'or ref['bytes']!=size or p.stat().st_size!=size:raise ValueError('Actual cache file size/path differs')
 golden=read(checked(cache['same_api_golden']))
 if golden['status']!='PASSED'or golden['dtype']!='bfloat16'or golden['micro']!=8 or golden['extra_quantization']:raise ValueError('Existing actual native micro8 golden missing')
 if len(cache['chunks'])!=(n+255)//256:raise ValueError('Actual completed cache chunk count differs')
 for index,ref in enumerate(cache['chunks']):
  row=read(checked(ref));start=index*256;end=min(n,start+256)
  if row['start']!=start or row['end']!=end or row['trajectory_ids_sha256']!=digest(tids[start:end]):raise ValueError('Actual native chunk own-ID range differs')
  for key,width in (('hidden',20*3584*2),('mask',20),('logits',4)):
   span=row['spans'][key]
   if span['offset']!=start*width or span['bytes']!=(end-start)*width or len(span['sha256'])!=64:raise ValueError('Bounded cache storage spans differ')
 checked(cache['files']['logits'])
 if sys.byteorder!='little':raise ValueError('Native float32 cache requires little endian')
 z=array.array('f');z.frombytes(Path(cache['files']['logits']['path']).read_bytes())
 if len(z)!=n or not all(math.isfinite(x)for x in z):raise ValueError('Frozen small-logit file not finite/aligned')
 cp0_range=cp0_zip(paths['baseline_CP0']);cp0_ref=proof(paths['baseline_CP0']);eval_ref=proof(paths['baseline_epoch0_evaluation']);evaluation=read(eval_ref['path']);receipt=evaluation['checkpoint']
 if evaluation['epoch']!=0 or evaluation['cursor']!=0 or not evaluation['training_components_unchanged']or evaluation['external_pool_access']or evaluation['final_pool_access']or {k:receipt[k]for k in ('path','bytes','sha256')}!=cp0_ref or receipt['cursor']!=0 or receipt['new_head_Adam_steps']!=0 or not receipt['full_CPU_reload_passed']:raise ValueError('Actual initial checkpoint/evaluation receipt differs')
 four=receipt['accepted_hashes']
 if set(four)!= {'raw','ema','adam','rng'}or any(len(x)!=64 for x in four.values())or four['raw']!=four['ema']or four['adam']!=hashlib.sha256(b'').hexdigest():raise ValueError('Actual initial-head four-component metadata differs')
 if not paths['cache_complete'].stat().st_mtime_ns<=paths['baseline_CP0'].stat().st_mtime_ns<=paths['baseline_epoch0_evaluation'].stat().st_mtime_ns:raise ValueError('Cache-before-CP0-before-evaluation ordering differs')
 checked(PRIOR)
 verifier_base=baseline.parent/'direct-count-large-smallCP1-independent-v2';verifier=verifier_base/'code-8fce2e468313/direct_count_large_small_checkpoint.py'
 verifier_ref=proof(verifier);verifier_publication_ref=proof(verifier_base/'source-publication.json');verifier_pub=read(verifier_publication_ref['path'])
 if verifier_pub['commit']!='8fce2e4683131d1d7c8de35c0e6a40a4fc693af6'or verifier_pub['files']['direct_count_large_small_checkpoint.py']!=verifier_ref:raise ValueError('Published small checkpoint verifier changed')
 row_base=baseline.parent/'direct-count-large-independent-audit-v3';row_pub_ref=proof(row_base/'source-publication.json');row_pub=read(row_pub_ref['path']);row_verifier=row_pub['files']['direct_count_large_independent_audit.py'];checked(row_verifier)
 if row_pub['commit']!='4ca51273ae6c3098e2f0f3c2914189791afcee62':raise ValueError('Original immutable Count row auditor differs')
 adapter_ref=proof(Path(__file__).with_name('direct_head_family_row_audit.py'))
 packet=dict(role='FROZEN_E1_SHARED_CACHE_CP0_COARSE_FORK_INPUTS',baseline_config=config_ref,baseline_publication=pub_ref,cache_complete=cache_ref,baseline_CP0=cp0_ref,baseline_epoch0_evaluation=eval_ref,family_protocol=protocol_ref,frozen_train_prior=PRIOR,
  baseline_CP0_four_hashes=four,small_checkpoint_verifier=verifier_ref,small_checkpoint_verifier_publication=verifier_publication_ref,small_checkpoint_verifier_source_commit='8fce2e4683131d1d7c8de35c0e6a40a4fc693af6',
  original_Count_row_metric_auditor=row_verifier,original_Count_row_metric_auditor_publication=row_pub_ref,coarse_row_schema_adapter=adapter_ref,
  cohort=c['cohort'],orders=c['orders'],fit_view_epochs=c['fit_view_epochs'],source_plan=c['source_plan'],grouping=c['grouping'],inputs=c['inputs'],frozen_NQ_halves=protocol['frozen_NQ_halves'],
  cache_files=cache['files'],cache_shape=[351360,20,3584],cache_dtype='bfloat16',cache_completed_chunks=len(cache['chunks']),cache_own_trajectory_IDs_sha256=digest(tids),ordered_fit_qids_sha256=digest(fit),ordered_holdout_qids_sha256=digest(held),original938_batches_sha256=digest(read(c['source_plan']['path'])['batches']),own_new5_lineage=held_lineage,
  initial_CP0_ZIP_scope=cp0_range,producer_CP0_full_CPU_reload_passed=True,controller_actual_CP0_fourstate_restore_performed=False,full_CP0_restore_required_on_fork_allocation_before_any_update=True,
  CP0_raw_EMA_Adam_RNG_must_equal_baseline_four_hashes=True,old_fourrank_archive_inheritance_required=True,no_optimizer_reset=True,
  controller_hidden_payload_freshly_hashed=False,producer_complete_binds_full_hidden_payload_SHA=True,controller_small_logits_fresh_SHA_and_finite_check=True,all_chunk_metadata_fresh_SHA_and_own_ID_ranges_verified=True,
  implementation=proof(__file__),resolver_source_commit=code_commit,model_or_checkpoint_tensor_loaded=False,GPU_submitted=False,external_payload_opened=False,final_payload_opened=False)
 immutable(ready_path,packet)
 return dict(status='REAL_SHARED_CACHE_CP0_METADATA_INPUTS_RESOLVED_REQUIRES_FORK_ALLOCATION_RESTORE',packet=proof(ready_path),questions=35136,trajectories=351360,cache_chunks=len(cache['chunks']),CP0=cp0_ref,initial_head_four_hashes=four,full_CP0_restore_required_on_fork_allocation=True,GPU_submitted=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--baseline',required=True);p.add_argument('--code-commit',required=True);a=p.parse_args()
 print(json.dumps(resolve(a.base,a.baseline,a.code_commit),sort_keys=True),flush=True)
if __name__=='__main__':main()
