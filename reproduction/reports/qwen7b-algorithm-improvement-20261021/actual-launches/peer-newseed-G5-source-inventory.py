"""Physical NAS10 read-only small-metadata/prefix/ownEM inventory; no tensors."""
import pathlib,json,hashlib,time,socket,gzip
B=pathlib.Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
ALIAS='/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020'
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def physical(p):
 s=str(p)
 return pathlib.Path(str(B)+s[len(ALIAS):]) if s==ALIAS or s.startswith(ALIAS+'/') else pathlib.Path(s)
def ownfp(ref):
 p=physical(ref['path']);x=fp(p)
 assert x['bytes']==ref['bytes'] and x['sha256']==ref['sha256']
 return x
def stat(p):
 s=physical(p).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino,device=s.st_dev)
c=read(B/'config.json');manifest=read(B/'inputs/manifest.json')
assert socket.gethostname().split('.')[0]=='h100-10' and c['run_id']=='70188b8e50fd4b45bc420ce58014f46e'
keys=('base','run_id','seed','source_seed','source_run_id','sampling_attempt_id','model_revision','model_dir','warmup_only',
 'capture_prefix_tokens','feature_layer','question_max_tokens','actor_hidden_native','actor_optimizer_updates','source_commit',
 'source_input_root','retrieval_index','retrieval_corpus','retrieval_encoder','retrieval_data_cache')
out=dict(status='PEER_NEWSEED_G5_READONLY_ASSET_INVENTORY',observed_epoch=time.time(),hostname=socket.gethostname(),physical_root=str(B),shared_alias=ALIAS,
 config=fp(B/'config.json'),configuration={k:c[k] for k in keys if k in c},input_manifest=fp(B/'inputs/manifest.json'),
 canonical_input_summary={k:manifest[k] for k in ('questions','trajectories','seed','diagnostic_not_sampled','ordered_qid_sha256')},
 model_tensor_read=False,feature_tensor_read=False,source_mutated=False,external_processes_queried=False)
original=physical(c['source_input_root']).parent/'config.json'
if original.exists():
 orig=read(original);out['original_source_config']=fp(original);out['original_source_configuration']={k:orig[k] for k in keys if k in orig}
 model=physical(c['model_dir'])/'complete.json'
if model.exists():out['official_model_complete']=dict(ref=fp(model),metadata=read(model))
inputs={physical(v['path']).resolve():v for v in manifest['batches']};paths=sorted(B.glob('sampling/shard-*/batches/batch-*/complete.json'))
members=[];payloads={k:0 for k in ('prefix_states','prefix_tokens','full_responses','own_em')};countq=countt=0
for p in paths:
 v=read(p);assert v['run_id']==c['run_id'] and v['model_revision']==c['model_revision'] and v['actor_optimizer_updates']==0
 assert v['trajectories']==5*v['questions'] and v['hidden_cached'] is False
 inp=B/'inputs'/p.parent.parent.parent.name/(p.parent.name+'.json');assert inp.resolve() in inputs
 assert v['input_sha256']==inputs[inp.resolve()]['sha256']
 members.append(dict(seal=str(p.relative_to(B)),sha256=fp(p)['sha256'],questions=v['questions'],trajectories=v['trajectories']))
 countq+=v['questions'];countt+=v['trajectories']
 for name,ref in [('prefix_states',v['prefix']['states']),('prefix_tokens',v['prefix']['prefix']),('full_responses',v['full']['full']),('own_em',v['full']['own_em'])]:payloads[name]+=ref['bytes']
out['sealed']=dict(batches=len(paths),questions=countq,trajectories=countt,declared_payload_bytes=payloads,
 membership_sha256=hashlib.sha256(json.dumps(members,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 global_complete_exists=(B/'complete.json').exists(),bulk_full_response_payloads_fresh_SHA_reread=False)
if (B/'audit/latest.json').exists():
 a=read(B/'audit/latest.json');out['existing_independent_CPU_audit']=dict(ref=fp(B/'audit/latest.json'),summary={k:a[k] for k in (
  'status','run_id','seed','model_revision','verified_batches','verified_unique_qids','verified_unique_trajectories','questions','trajectories',
  'failed_batches','unsealed_batches','sealed_pending_batches','cache_reused_batches','independently_read_batches','strict_em_recomputed',
  'strict_em_verification','states_shards_independently_audited','auditor_sha256','epoch') if k in a})
out['sample_sealed_batches']=[]
chosen=paths[:2]+paths[-2:]
seen=set()
for p in chosen:
 if p in seen:continue
 seen.add(p);v=read(p);before=fp(p);inp=B/'inputs'/p.parent.parent.parent.name/(p.parent.name+'.json')
 assert fp(inp)['sha256']==v['input_sha256'];rows=read(inp)['rows'];byqid={r['qid']:r for r in rows}
 assert len(byqid)==v['questions']
 cfg=physical(v['prefix']['config_path']) if 'config_path' in v['prefix'] else p.parent/'config.json';cc=read(cfg)
 assert fp(cfg)['sha256']==v['prefix']['config_sha256']==v['full']['config_sha256']
 pr=ownfp(v['prefix']['prefix']);er=ownfp(v['full']['own_em']);prefix=[]
 with gzip.open(pr['path'],'rt') as f:prefix=[json.loads(line) for line in f]
 with open(er['path']) as f:labels=[json.loads(line) for line in f]
 by={r['trajectory_id']:r for r in prefix};assert len(by)==len(prefix)==len(labels)==v['trajectories']
 for label in labels:
  row=by[label['trajectory_id']];assert row['qid']==label['qid'] and row['prefix_sha256']==label['prefix_sha256']
  assert row['initial_prompt_token_ids']==byqid[row['qid']]['initial_prompt_token_ids']
  assert row['question_hash']==byqid[row['qid']]['question_hash'] and 0<=row['response_index']<5
  assert type(label['strict_em']) is int and label['strict_em'] in (0,1)
  assert len(row['policy_token_ids'])<=20 and (len(row['policy_token_ids'])==20 or row['terminal'] is True)
 cache=B/'audit/batches'/inp.parent.name/(inp.stem+'.json');audit=None
 if cache.exists():
  cached=read(cache);binding=cached['binding'];same=all(stat(B/('sampling/'+inp.parent.name+'/batches/'+inp.stem+'/complete.json') if name=='batch_seal' else (p.parent/'config.json') if name=='config' else p.parent/'outers/outer-000001'/('questions.json' if name=='questions' else 'prefix/complete.json' if name=='prefix_seal' else 'prefix/scan.json' if name=='prefix_scan' else 'prefix/rank-0-complete.json' if name=='prefix_rank' else 'full/complete.json' if name=='full_seal' else 'full/rank-0-complete.json' if name=='full_rank' else 'probe-score/selection.json' if name=='selection' else 'prefix/rank-0-prefix.jsonl.gz' if name=='prefix' else 'full/rank-0-full.jsonl.gz' if name=='full' else 'full/rank-0-own-em.jsonl'))==value for name,value in binding['files'].items())
  audit=dict(ref=fp(cache),status=cached['status'],all_cached_file_stats_still_match=same,
   independent_sha256=cached['proof']['independent_sha256'],actual_trajectory_ids_sha256=cached['proof']['actual_trajectory_ids_sha256'])
 assert fp(p)==before;ownfp(v['prefix']['prefix']);ownfp(v['full']['own_em'])
 sample={k:v['prefix'][k] for k in ('sampling_contract','action_boundary','interaction_protocol','actual_compute_dtype','engine_seed','sampling_attempt_id') if k in v['prefix']}
 out['sample_sealed_batches'].append(dict(complete=before,input=fp(inp),batch_configuration=fp(cfg),prefix=pr,own_em=er,
  full_response=v['full']['full'],prefix_states=v['prefix']['states'],phase_contract=sample,
  configuration={k:cc[k] for k in ('run_id','seed','source_seed','source_run_id','sampling_attempt_id','model_revision','actor_hidden_native','prefix_policy_tokens') if k in cc},
  questions=v['questions'],trajectories=v['trajectories'],sample_ownEM_ones=sum(r['strict_em'] for r in labels),
  all_own_prompt_prefix_label_links_fresh_verified=True,existing_independent_batch_audit=audit))
out['existing_newseed_HF_HKUST_receipt_candidates']=[str(p.relative_to(B)) for p in B.glob('preparation/*') if any(s in p.name.lower() for s in ('hf','upload','archive','transfer','export'))]
out['active_native_or_pool_workers_modified']=False
print(json.dumps(out))
