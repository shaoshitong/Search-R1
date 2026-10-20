"""Remote stdlib-only controller for the published pure hard-enrichment module."""
import pathlib,json,hashlib,fcntl,os,time,importlib.util,traceback
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021';P=B/'balanced-rank1-from-epoch1-prefix20-20261020'
ROOT=A/'hard-enrichment-20261021';COMMIT='dfb183ddf35a7116ef118d1d2e1e5b49dcbb21c7'
def read(p):return json.loads(p.read_text())
def fp(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def atomic(p,value):
 raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
 tmp=p.with_name(p.name+'.tmp')
 with tmp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def progress(phase,**kw):
 v=dict(phase=phase,time=time.time(),pid=os.getpid(),GPU_used=False,tensor_loaded=False);v.update(kw)
 atomic(ROOT/'prepare-progress.json',v);print(json.dumps(v),flush=True)
def once(p,value):
 raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
 if p.exists():assert p.read_bytes()==raw,'Preserve changed immutable preparation '+str(p)
 else:
  with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return fp(p)
def main():
 with (ROOT/'prepare.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  pub=read(ROOT/'source-publication.json');assert pub['commit']==COMMIT
  code=pathlib.Path(pub['files']['hard_enrichment.py']['path'])
  assert fp(code)==pub['files']['hard_enrichment.py']
  spec=importlib.util.spec_from_file_location('published_hard_enrichment',str(code));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  identity=dict(pid=os.getpid(),start_ticks=pathlib.Path('/proc/self/stat').read_text().split()[21],source_commit=COMMIT,code=fp(code))
  atomic(ROOT/'prepare-controller.json',identity)
  if (ROOT/'prepare-complete.json').exists():
   prior=read(ROOT/'prepare-complete.json');assert fp(prior['candidate_plan']['path'])==prior['candidate_plan']
   print(json.dumps(dict(status='EXISTING_CPU_CANDIDATES_FRESH_SHA_VERIFIED',complete=fp(ROOT/'prepare-complete.json'),candidate_plan=prior['candidate_plan'],summary=prior['summary'])),flush=True);return
  progress('READ_CANONICAL_AND_BALANCED_BANKS')
  original=read(B/'bank/groups.json');bank=read(P/'bank/groups.json');canonical={g['qid']:g for g in original['train']}
  assert len(canonical)==len(original['train'])==len(bank['train'])==169028
  assert set(canonical)=={g['qid'] for g in bank['train']}
  for g in bank['train']:assert all(g[k]==canonical[g['qid']][k] for k in ('qid','question_hash','trajectory_ids','labels'))
  assert len(bank['diagnostic'])==512
  progress('VERIFY_NATIVE_PREFIX20_METADATA',canonical_questions=169028)
  feature_path=B/'bank/features.json';features=read(feature_path);assert features and len(features)==len(set(features))
  for i,path in enumerate(features):
   v=read(pathlib.Path(path));assert v['status']=='LOSSLESS_NATIVE_FEATURES_COMPLETE'
   assert (v['feature_layer'],v['prefix_tokens'],v['dimension'],v['question_max_tokens'])==(26,20,3584,273)
   if i%128==0:progress('VERIFY_NATIVE_PREFIX20_METADATA',verified_manifests=i+1,total_manifests=len(features))
  matched_path=A/'matched-loss128-v2/plan.json';matched=read(matched_path)
  core=module.core_groups(bank,matched);assert len(core)==6528
  selection_path=B/'selection-2048-20261020/bank/groups.json';selection=read(selection_path)['selection_validation'];assert len(selection)==2048
  final_candidates=(A/'training-extraG5-matched6528-20261021/final-exclusion-plan.json',A/'validation-g10-20261021/materialized/final-exclusion-plan.json')
  final_path=next(p for p in final_candidates if p.exists());final=read(final_path)
  assert final['role']=='final' and final['frozen'] is True and final['labels_read'] is False and len(final['questions'])==4096
  assert all(set(r)<={'qid','question_hash','canonical_question_sha256','pool'} and {'qid','question_hash'}<=set(r) for r in final['questions'])
  materialized=B/'confirmation-materialization-deploy-20261021/materialized/materialization.json'
  proof=fp(materialized);assert proof['sha256']=='ff68f243b1daa7d8a617a2c7a825e1126d0771eee2e92b96a85dc5f9b4f2bfe3'
  assert any(r['path']==str(materialized) and r['sha256']==proof['sha256'] for r in final['source_proofs'])
  scopes=dict(matched_fit=core,diagnostic512=bank['diagnostic'],selection2048=selection,frozen_final=final['questions'])
  qs=set();hs=set();audit_rows=[];scope_counts={}
  for role,rows in scopes.items():
   q={r['qid'] for r in rows};h={r['question_hash'] for r in rows}
   assert len(q)==len(rows) and not qs&q and not hs&h,'Actual exclusion scopes overlap by qid/hash: '+role
   qs.update(q);hs.update(h);scope_counts[role]=len(rows)
   audit_rows.extend(dict(qid=r['qid'],question_hash=r['question_hash']) for r in rows)
  audit_rows.sort(key=lambda r:r['qid'])
  refs=[]
  for role,path in [('matched_fit',matched_path),('diagnostic512',P/'bank/groups.json'),('selection2048',selection_path),('frozen_final',final_path)]:
   r=fp(path);r['role']=role;refs.append(r)
  audit=dict(role='training_mining_exclusions',groups=audit_rows,source_refs=refs,scopes=scope_counts,
   identity_only=True,labels_read_from_final=False,final_generated_responses_opened=False,
   source_canonical_bank=fp(B/'bank/groups.json'),frozen_final_materialization=proof,
   all_four_scopes_qid_and_question_hash_disjoint=True)
  progress('SAVE_IDENTITY_ONLY_EXCLUSION_AUDIT',scope_counts=scope_counts)
  exclusion_ref=once(ROOT/'identity-exclusions.json',audit)
  r=read(P/'probe/epoch-0004.json');c=read(P/'config.json')
  assert r['epoch']==4 and r['steps']==14418 and r['sha256']=='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'
  assert r['cursor']==0 and r['completed_epoch'] is True and r['independent_cpu_reload'] is True
  cp=pathlib.Path(r['path']);assert cp.is_file() and cp.stat().st_size==r['bytes']
  checkpoint={k:r[k] for k in ('path','bytes','sha256','steps')};checkpoint.update(name='E4',epoch=4,source_receipt=fp(P/'probe/epoch-0004.json'),source_config=fp(P/'config.json'),model_dir=c['model_dir'])
  config=dict(source_bank=fp(P/'bank/groups.json'),features=fp(feature_path),checkpoint=checkpoint,
   matched_plan=fp(matched_path),exclusion_refs=[exclusion_ref],seed=20261021,
   canonical_bank=fp(B/'bank/groups.json'),native_features=dict(feature_layer=26,prefix_tokens=20,dimension=3584,question_max_tokens=273,verified_manifests=len(features)),
   canonical_train_own_qid_hash_ids_EM_equal_to_balanced=True,source_commit=COMMIT)
  config_ref=once(ROOT/'prepare-config.json',config)
  progress('FREEZE_HASH_SELECTED32768_CANDIDATES')
  candidate=module.prepare_candidates(bank,matched,[audit],source_bank=config['source_bank'],features=config['features'],
   checkpoint=checkpoint,exclusion_refs=[exclusion_ref],matched_plan_ref=config['matched_plan'],seed=config['seed'])
  candidate_ref=module.once(ROOT/'candidates.json',candidate)
  actual=read(pathlib.Path(candidate_ref['path']));assert actual==candidate
  assert len(actual['groups'])==32768 and sum(len(g['trajectory_ids']) for g in actual['groups'])==163840
  assert not qs&{g['qid'] for g in actual['groups']} and not hs&{g['question_hash'] for g in actual['groups']}
  result=dict(status='TRAIN_ONLY32768_HASH_CANDIDATES_PREPARED_NOT_SCORED',candidate_plan=candidate_ref,summary=candidate['summary'],
   source_config=config_ref,exclusions=exclusion_ref,source_commit=COMMIT,source_code=fp(code),canonical_train_equivalence=True,
   source_bank=config['source_bank'],features=config['features'],source_checkpoint=checkpoint,native_metadata=config['native_features'],
   actual_saved_candidate_fresh_SHA_verified=True,checkpoint_tensor_read=False,GPU_used=False,probe_updates=0,actor_updates=0)
  complete=module.once(ROOT/'prepare-complete.json',result)
  progress('COMPLETE',questions=32768,trajectories=163840,candidate_plan=candidate_ref,complete=complete)
  print(json.dumps(result),flush=True)
try:main()
except BaseException:
 atomic(ROOT/'prepare-failure.json',dict(time=time.time(),traceback=traceback.format_exc(),GPU_used=False))
 raise
