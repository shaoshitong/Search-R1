from pathlib import Path
import json,hashlib,datetime,csv,io,re,importlib.util,time
w=Path.cwd();b=w/'outputs/tree-nested-branch-handoff-20261013';ms=[w/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
def read(p):return json.loads(p.read_bytes())
def pin(p,sha=None,size=None,mirror=True):
 raw=p.read_bytes();hh=hashlib.sha256(raw).hexdigest()
 if sha:assert hh==sha,p
 if size is not None:assert len(raw)==size,p
 if mirror:
  for m in ms:assert (m/p.relative_to(b)).read_bytes()==raw,p
 return dict(path=str(p),bytes=len(raw),sha256=hh)
cp=b/'LEARNING_CURVE_COMPLETE_THROUGH170.json';op=b/'LEARNING_CURVE_COMPLETE_THROUGH165.json';c=read(cp);o=read(op)
assert c['training_points'][:165]==o['training_points'] and c['completed_dev512_points'][:33]==o['completed_dev512_points']
assert len(c['training_points'])==170 and len(c['completed_dev512_points'])==34
def segments(p,key):
 s=p.read_text();match=re.search('"'+key+'"\\s*:\\s*\\[',s);assert match
 i=match.end();out=[];dec=json.JSONDecoder()
 while True:
  while s[i].isspace() or s[i]==',':i+=1
  if s[i]==']':return out
  _,end=dec.raw_decode(s,i);out.append(s[i:end]);i=end
assert segments(cp,'training_points')[:165]==segments(op,'training_points')
assert segments(cp,'completed_dev512_points')[:33]==segments(op,'completed_dev512_points')
oldcsv=(b/'LEARNING_CURVE_COMPLETE_THROUGH165.csv').read_bytes();newcsv=(b/'LEARNING_CURVE_COMPLETE_THROUGH170.csv').read_bytes()
assert len(oldcsv)==174892 and hashlib.sha256(oldcsv).hexdigest()=='bc33520eaf13018b4e2c62623e60642c60c95f070b1fd77d7be6ef1d1503bd73'
assert newcsv.startswith(oldcsv) and len(list(csv.DictReader(io.StringIO(newcsv.decode()))))==204
bp=b/'LEARNING_CURVE_THROUGH170_COST_SUPPLEMENT_BINDING.json';binding=read(bp);assert binding['through_outer']==170 and binding['actual_remote_incremental_collection_invocations']==1 and binding['preceding_pending_remote_operations'] is False
refs={k:pin(Path(x['path']),x['sha256'],x['bytes'],not k.startswith('source_poll') and Path(x['path']).is_relative_to(b)) for k,x in binding['references'].items()}
assert refs['curve']==pin(cp,'5c542620c45852c1570f5dcfca79e8b4ae6e4352335a8452d31abed0ecb791c9',992265)
new=[]
for n in range(166,171):
 t=next(x for x in c['training_points'] if x['outer']==n);fp=b/f'FULL_OUTER{n}_PROGRESS_RECEIPT.json';f=read(fp);pp=w/t['source_poll'];p=read(pp);m=f['iteration_metrics']
 assert str(pp)==f['input_bindings']['poll']['path'] and hashlib.sha256(pp.read_bytes()).hexdigest()==f['input_bindings']['poll']['sha256']
 assert next(x for x in p['iteration_metrics'] if int(x['step'])==n)==m
 for k,mk in [('pure_intra_adv_population_variance','tree_variance/intra_adv_population_variance'),('raw_within_tree_reward_population_variance','tree_variance/within_root_raw_reward_population_variance'),('total_actual_adv_population_variance','tree_variance/group_actual_adv_population_variance'),('training_reward_mean','critic/score/mean'),('step_wall_seconds','timing_s/step')]:assert t[k]==m[mk],(n,k)
 assert t['actual_cumulative_adam_calls_per_rank']==f['per_rank_actual_total_adam'] and t['actual_outer_adam_calls_per_rank']==f['actual_outer_successful_writes_per_rank']
 assert t['source_commit']==f['source_commit'] and t['config_sha256']==f['config_sha256']
 new.append(dict(outer=n,full=pin(fp,t['source_receipt_sha256']),poll=pin(pp,t['source_poll_sha256'],mirror=False)))
full=read(b/'FULL_OUTER170_PROGRESS_RECEIPT.json');dev=next(x for x in c['completed_dev512_points'] if x['outer']==170);actual=full['actual_dev512']
assert dev['rows']==actual['rows']==512 and dev['ledger_sha256']==actual['ledger_sha256'] and dev['f1']==actual['metrics']['val/f1/all'] and dev['em']==actual['metrics']['val/em/all']
sp=w/'work/tree-speed-repair-20261008/nested-dev-cumulative-parts-source/reproduction/tree-eval-resume-20261013/local-control-backup/collect_nested_dev_costs.py'
assert hashlib.sha256(sp.read_bytes()).hexdigest()=='545bc2cb668da2b672ee3a859c031171927f3b2b6a2aaa3d368bee7d09b732d0'
spec=importlib.util.spec_from_file_location('published_dev_decoder',sp);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
d=json.loads(mod.read_bound(b,ms,dict(path='DEV5_170_NATIVE_COST_SUPPLEMENT.json',bytes=7058,sha256='4b34f52a1a3a8447f23c4ebdb8c55a83c9254c8fc72cf8ba3ad367d69b2e6722')));rows=mod.rows_from_summary(b,ms,d);old=read(b/'DEV5_165_NATIVE_COST_SUPPLEMENT.json');oldrows=mod.rows_from_summary(b,ms,old)
assert len(rows)==34 and len(oldrows)==33 and rows[:33]==oldrows
parts=[pin(b/x['file'],x['sha256'],x['bytes']) for x in d['rows']['parts']]
costparts=[pin(Path(x['path']),x['sha256'],x['bytes']) for x in binding['cumulative_cost_part_references']]
rootdevpath=b/'ROOT_DEV170_NATIVE_COST_ACTUAL_READBACK.json';rd=read(rootdevpath)
assert len(costparts)==rd['actual_ref_counts']['cum_parts']
row=rows[-1];assert row['outer']==170 and row['ledger_sha256']==dev['ledger_sha256']
control=read(Path(refs['dev170_original_control_summary']['path']));assert control['ledger_sha256']==row['ledger_sha256'] and control['metrics']==actual['metrics']
rootdev=pin(rootdevpath,'f1afb5d6c6c686fe771acb9f04f065bfd10d5f46199024f1fe6711a780007d13',80637);assert refs['root_dev_native_cost_readback']==rootdev
r=dict(schema='root_curve_and_dev_native_cost_through170_actual_paged_readback_v2',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),executor='/root',scope='Actual local immutable-source and first-FULL original-poll reconciliation; complete paged dev rows read through published decoder; no new remote read/evaluation/test/code review/tensor access.',references=refs,curve170=refs['curve'],csv170=refs['csv'],old_curve165=pin(op,'754d0ef57d8182dcda085ecab576051715e039a50d05e30878d4b3c0c19834cc',942131),old165_training_dev_objects_raw_fragments_and_CSV174892_prefix_preserved=True,new166_170_bindings=new,actual_completed_dev170=dev,new_dev5_170=refs['dev_native_cost'],rows_storage=d['rows']['storage'],actual_decoded_rows=34,all_complete_rows_parts=parts,cumulative_cost_complete_field_parts=costparts,old33_dev_control_rows_unchanged=True,actual_dev170_control=refs['dev170_original_control_summary'],root_dev170_actual_readback=rootdev,cost_supplement_binding=pin(bp,'533545b6c7661cf2ecf5f50c2f990b5eacf756ef596d2d4e4bbc0ff8053e2150',28387),native=refs['native170_metadata'],actual_native_counters=row['counters'],observer_wall_seconds=row['wall_seconds'],fullmetric_dev_seconds=full['normal_dev512_wall_seconds'],wall_CPU_added_again=False,both_mirrors_actual_byte_readback=True,goal_status='active',goal_complete=False)
rp=b/'ROOT_CURVE_DEV_COST_THROUGH170_ACTUAL_READBACK.json';assert not rp.exists();raw=(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
for p in [rp,*[m/rp.name for m in ms]]:p.write_bytes(raw)
first=pin(rp);time.sleep(1);assert pin(rp)==first
print(json.dumps(dict(root_readback=first)))
