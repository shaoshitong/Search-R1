import json,hashlib,pathlib,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021';P=B/'balanced-rank1-from-epoch1-prefix20-20261020'
def read(p):return json.loads(p.read_text())
def fp(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
out=dict(status='TRAIN_ONLY_HARD_ENRICHMENT_READONLY_SOURCE_PREFLIGHT',time=time.time(),tensor_loaded=False,remote_mutations=False)
original=read(B/'bank/groups.json');balanced=read(P/'bank/groups.json')
assert len(original['train'])==len(balanced['train'])==169028
oi={g['qid']:g for g in original['train']};bi={g['qid']:g for g in balanced['train']}
assert len(oi)==len(bi)==169028 and set(oi)==set(bi)
for qid in oi:
 assert all(oi[qid][k]==bi[qid][k] for k in ('qid','question_hash','trajectory_ids','labels'))
out['source_bank']=fp(P/'bank/groups.json');out['canonical_bank']=fp(B/'bank/groups.json')
out['canonical_train_equivalence']=dict(questions=169028,qid_hash_ownIDs_EM_all_equal=True)
out['diagnostic_questions']=len(balanced['diagnostic']);assert out['diagnostic_questions']==512
out['matched_plan']=fp(A/'matched-loss128-v2/plan.json');plan=read(pathlib.Path(out['matched_plan']['path']))
assert len(plan['cohort_qids'])==len(set(plan['cohort_qids']))==6528
out['features']=fp(B/'bank/features.json');features=read(pathlib.Path(out['features']['path']))
assert isinstance(features,list) and features and len(features)==len(set(features))
metadata=set()
for p in features:
 v=read(pathlib.Path(p));assert v['status']=='LOSSLESS_NATIVE_FEATURES_COMPLETE'
 metadata.add((v['feature_layer'],v['prefix_tokens'],v['dimension'],v['question_max_tokens']))
assert metadata=={(26,20,3584,273)}
out['feature_metadata']=dict(manifests=len(features),all_metadata_same=True,feature_layer=26,prefix_tokens=20,dimension=3584,question_max_tokens=273)
r=read(P/'probe/epoch-0004.json');c=read(P/'config.json')
assert r['epoch']==4 and r['steps']==14418 and r['sha256']=='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'
assert r['cursor']==0 and r['completed_epoch'] is True and r['independent_cpu_reload'] is True
cp=pathlib.Path(r['path']);assert cp.is_file() and cp.stat().st_size==r['bytes']
d={k:r[k] for k in ('path','bytes','sha256','steps')};d.update(name='E4',epoch=4,source_receipt=fp(P/'probe/epoch-0004.json'),source_config=fp(P/'config.json'),model_dir=c['model_dir'])
out['checkpoint']=d;out['checkpoint_tensor_read']=False
selection=B/'selection-2048-20261020/bank/groups.json';v=read(selection)
out['selection_source']=fp(selection);out['selection_keys']=list(v.keys())
out['selection_counts']={k:len(x) for k,x in v.items() if isinstance(x,list)}
out['final_candidates']=[]
for p in (A/'training-extraG5-matched6528-20261021/final-exclusion-plan.json',A/'validation-g10-20261021/materialized/final-exclusion-plan.json'):
 if p.exists():
  v=read(p);row=dict(ref=fp(p),keys=list(v.keys()),role=v.get('role'),questions=len(v.get('questions',[])))
  assert isinstance(v.get('questions'),list) and all(set(g)=={'qid','question_hash'} for g in v['questions'])
  out['final_candidates'].append(row)
assert out['final_candidates'],'No existing identity-only final exclusion proof'
print(json.dumps(out))
