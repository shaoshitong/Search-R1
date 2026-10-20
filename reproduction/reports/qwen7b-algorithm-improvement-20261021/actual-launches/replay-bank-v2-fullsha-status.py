"""Read-only actual whole-bank merge lifecycle and accepted feature-list proofs."""
import pathlib,json,hashlib,fcntl,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/replay-matched9088-bank-v2-20261021');prepared=b/'prepared'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def live(pid,ticks):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]==str(ticks)
 except OSError:return False
out=dict(time=time.time(),base=str(b))
for name in ('full-sha-merge-launch.json','full-sha-merge-exit.json'):
 if (b/name).exists():out[name]=read(b/name)
launch=out.get('full-sha-merge-launch.json');out['process_identity_live']=live(launch['pid'],launch['start_ticks']) if launch else False
def proc_tree(pid):
 seen=set();todo=[pid];result=[]
 while todo:
  own=todo.pop()
  if own in seen:continue
  seen.add(own);root=pathlib.Path('/proc/%d'%own)
  if not root.exists():continue
  item=dict(pid=own)
  try:
   children=root/'task'/str(own)/'children';todo.extend(int(n) for n in children.read_text().split())
   item['io']={line.split(':',1)[0]:int(line.split(':',1)[1]) for line in (root/'io').read_text().splitlines() if line.split(':',1)[0] in ('rchar','read_bytes','syscr')}
   current=[]
   for fd in (root/'fd').iterdir():
    try:
     target=fd.readlink() if hasattr(fd,'readlink') else pathlib.Path(__import__('os').readlink(str(fd)))
     if str(target).startswith(str(prepared)+'/'):
      info=(root/'fdinfo'/fd.name).read_text();position=next((int(line.split(':',1)[1]) for line in info.splitlines() if line.startswith('pos:')),None)
      current.append(dict(path=str(target),position=position))
    except OSError:pass
   item['open_verified_bank_files']=current[:8]
  except OSError:pass
  result.append(item)
 return result
if launch and out['process_identity_live']:out['actual_cpu_readers']=proc_tree(launch['pid'])
out['accepted_layer_lists_present']=[p.name for p in prepared.glob('features-layer-*.json')]
for name in ('full-sha-merge.log','full-sha-merge-controller.log'):
 p=b/name
 if p.exists():
  size=p.stat().st_size
  with p.open('rb') as f:f.seek(max(0,size-5000));tail=f.read(5000)
  out[name]=dict(bytes=size,tail=tail.decode(errors='replace'))
p=prepared/'bank-complete.json'
if p.exists():
 d=read(p);assert d['status']=='ALL_FOUR_MATCHED9088Q_MULTILAYER_HF_REPLAY_BANKS_SHA_VERIFIED' and d['full_replay_shared_file_SHA_verified'] is True
 assert d['questions']==9088 and d['trajectories']==45440 and d['roles']=={'fit':6528,'diagnostic512':512,'selection2048':2048}
 assert d['native_bitexact_equivalence_claim'] is False and d['actor_updates']==0
 assert sha(pathlib.Path(d['source_preparation']['path']))==d['source_preparation']['sha256']=='a7fa015dd655ef40d371172375ae30e4280503375485bd61ed6ab49baf5c1fd9'
 identities=read(pathlib.Path(d['group_identities']['path']));assert sha(pathlib.Path(d['group_identities']['path']))==d['group_identities']['sha256']
 groups=identities['groups'];assert len(groups)==9088 and not identities['labels_included'];assert len({g['qid'] for g in groups})==9088
 assert len({g['question_hash'] for g in groups})==9088 and len({t for g in groups for t in g['trajectory_ids']})==45440
 actual_roles=dict(__import__('collections').Counter(g['role'] for g in groups));assert actual_roles==d['roles']
 prep=read(prepared/'preparation.json');plan_ref=prep['matched_plan'];plan_path=pathlib.Path(plan_ref['path']);assert sha(plan_path)==plan_ref['sha256'];plan=read(plan_path)
 assert {g['qid'] for g in groups if g['role']=='fit'}==set(plan['cohort_qids']) and len(plan['cohort_qids'])==6528
 held_ref=prep['source_banks']['selection'];held_path=pathlib.Path(held_ref['path']);assert sha(held_path)==held_ref['sha256'];held=read(held_path)['selection_validation']
 assert {g['qid'] for g in groups if g['role']=='selection2048'}=={g['qid'] for g in held} and len(held)==2048
 out['actual_role_counts']=actual_roles;out['fit_qids_match_original_matched128_plan']=True;out['selection_qids_match_original2048_bank']=True
 out['source_own_EM_banks']=prep['source_banks'];out['source_matched_plan']=plan_ref
 layers={}
 for layer,source in d['features_by_capture_layer'].items():
  pp=pathlib.Path(source['path']);assert ref(pp)==source;refs=read(pp);count=0;seen=set()
  for item in refs:
   mp=pathlib.Path(item['path']);assert ref(mp)==item;meta=read(mp);assert meta['status']=='LOSSLESS_HF_REPLAY_FEATURES_COMPLETE' and meta['capture_layer']==int(layer) and meta['source_runtime']=='huggingface_frozen_causal_prefill'
   assert meta['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28' and meta['actor_optimizer_updates']==0
   count+=meta['count'];seen.add(item['path'])
  assert count==45440 and len(seen)==len(refs)==144
  layers[layer]=dict(features=source,manifest_count=len(refs),trajectories=count)
 out['bank_complete']=ref(p);out['status']=d['status'];out['roles']=d['roles'];out['layers']=layers;out['group_identities']=d['group_identities'];out['full_shared_file_SHA_verified']=True;out['native_bitexact_equivalence_claim']=False
else:out['status']='FULL_SHA_MERGE_NOT_YET_ACCEPTED'
print(json.dumps(out))
