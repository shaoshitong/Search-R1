from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,math,importlib.util,time,subprocess
W=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2');B=W/'outputs/tree-nested-branch-handoff-20261013';M=[W/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')];S=W/'work/tree-speed-repair-20261008/nested-dev-cumulative-parts-source/reproduction/tree-eval-resume-20261013/local-control-backup/collect_nested_dev_costs.py';ENTRY='1edaf50bb8e14de51db936e0859e369792011233'
C='778b2b43dad90fc0c1c5d95d77c2326286011b73';P='ba9f6627ec9a4e9749830d7b6acca53f6feec6b4';CFG='355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3';MODEL='744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0';DATA='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24';RUN='/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:'+C;UNIT='sst-tree-nested-branch-fresh200-778b2b43.service';INV='a313bda919834d24b4c5d491698d39ce';INNER='846e3e75def4d13d2d130b90dff648cbd8cef3cf7ced987ffc606d335661867b'
F={};R={};PARTS=[];FIELDS={};OBS=[]
def sh(r):return hashlib.sha256(r).hexdigest()
def unique(pairs):
 d={}
 for k,v in pairs:assert k not in d,k;d[k]=v
 return d
def load(r):return json.loads(r,object_pairs_hook=unique)
def path(n):
 p=Path(n);return p if p.is_absolute() else (W/p if p.parts[0] in ('outputs','work') else B/p)
def read(n,h=None,z=None):
 p=path(n).resolve();assert p.is_relative_to(W) and p.suffix in ('.json','.md','.py','.patch','.csv'),str(p);r=p.read_bytes();assert len(r)<4000000
 if h is not None:assert sh(r)==h,'SHA '+str(p)
 if z is not None:assert len(r)==z,'size '+str(p)
 mirror=p.is_relative_to(B) and not p.name.startswith(('NATIVE5_CONTINUE_POLL_','NATIVE5_CONTINUE_RESOURCE_'))
 ref=dict(path=str(p),bytes=len(r),sha256=sh(r),mirror_checked=mirror)
 if str(p) in F:assert F[str(p)]==ref,'Changed '+str(p)
 if mirror:
  for m in M:assert (m/p.relative_to(B)).read_bytes()==r,'mirror '+str(p)
 F[str(p)]=ref;R[str(p)]=r
 return load(r) if p.suffix=='.json' else r
def bound(ref,hint=None):
 n=ref.get('path') or ref.get('file') or hint;assert isinstance(n,str),(ref,hint)
 if n.startswith(('/dataset/','/mnt/')):OBS.append(ref);return None
 return read(n,ref['sha256'],ref['bytes'])
def refs(v):
 if isinstance(v,list):
  for x in v:refs(x)
 elif isinstance(v,dict):
  if isinstance(v.get('sha256'),str) and len(v['sha256'])==64 and type(v.get('bytes')) is int and (v.get('path') or v.get('file')):bound(v);return
  for k,x in v.items():
   if isinstance(x,dict) and isinstance(x.get('sha256'),str) and len(x['sha256'])==64 and type(x.get('bytes')) is int and Path(k).suffix in ('.json','.md','.py','.patch','.csv'):bound(x,k)
   else:refs(x)
def expand(v,k):
 if isinstance(v,dict) and v.get('storage')=='complete_original_field_parts':
  kind=v['container_type'];out={} if kind=='mapping' else [];seen=set()
  for ref in v['parts']:
   assert ref.get('json_selector')=='value' and ref['file'] not in seen;seen.add(ref['file']);part=bound(ref);assert part['schema']=='cumulative_scalar_field_part_v1' and part['original_field']==k
   val=part['value'];assert isinstance(val,dict if kind=='mapping' else list)
   if kind=='mapping':assert not set(out)&set(val);out.update(val)
   else:out.extend(val)
   PARTS.append(dict(original_field=k,**ref))
  assert len(out)==v['original_entries'];FIELDS[k]=dict(entries=len(out),parts=len(v['parts']),container_type=kind);return expand(out,k)
 if isinstance(v,dict):return {n:expand(x,k+'.'+n) for n,x in v.items()}
 if isinstance(v,list):return [expand(x,k+'['+str(i)+']') for i,x in enumerate(v)]
 return v
ORIGINAL=W/'work/tree-speed-repair-20261008/nested-local-observer-snapshot-source/reproduction/tree-local-observer-snapshot-20261016/dev165-independent-reader/dev165_independent_actual_readback.py';read(str(ORIGINAL),'8fb7855a1d38204874fd72fb8e7759fc0df462df1cc05c8fc8316f14069cc5a7',18352)
sr=read(str(S),'545bc2cb668da2b672ee3a859c031171927f3b2b6a2aaa3d368bee7d09b732d0',27911);assert subprocess.run(['git','-C',str(S.parent),'show',ENTRY+':reproduction/tree-eval-resume-20261013/local-control-backup/collect_nested_dev_costs.py'],check=True,stdout=subprocess.PIPE).stdout==sr
spec=importlib.util.spec_from_file_location('dev170_actual_independent_reader',S);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
owner=read('DEV5_170_NATIVE_COST_LOCAL_RECEIPT.json','29c0ea107ed07fe91b2d76742246c3909b0a32380927015022153c79cc364415',59548);summ=read('DEV5_170_NATIVE_COST_SUPPLEMENT.json','4b34f52a1a3a8447f23c4ebdb8c55a83c9254c8fc72cf8ba3ad367d69b2e6722',7058);cost=read('COST_OUTER170_LOCAL_RECEIPT.json','d4f3410f492be4a8d3e3bd8742ed06ef0b717876f9d8e50c2b66767a37ea4800',66204);raw=read('CUMULATIVE_COST_THROUGH170.json','3156615c0354cece824cde4dd7754e603664699e5028831648c9ee513e85080b',77080);cum={k:expand(v,k) for k,v in raw.items()}
read('FULL_OUTER170_PROGRESS_RECEIPT.json','a1d5f8b8a8af07324594657e5b26b16a670dba5d3ecd27fb7d27dac8040390be',31896);read('NATIVE_HISTORY_170_METADATA_READBACK.json','923b57c2d99790babfc843ea66ce497dd6b94d7fa616dcaed94e432bf8e0b635',8355);read('NATIVE5_CONTINUE_POLL_2104.json','bc6db0b12f047a547c051f9b96eb1862d911b625df2f9265a12ec8ece61c9b12',633979);read('NATIVE5_CONTINUE_RESOURCE_2104.json','9e12d5c3adc33646eb79734e117f99ed0b6a5b99c0cef99d391407aacb61f9e3',18808)
oldroot=read('ROOT_DEV165_NATIVE_COST_ACTUAL_READBACK.json','ed1c63d0b2abee62f3696b2b3a906c5d792c917dc036670bf7d4b2cc1293f66d',68410);oldowner=read('DEV5_165_NATIVE_COST_LOCAL_RECEIPT.json','22408dab7242eb6f60297b2714fb80a200b648fa67ea2d73a5c1bfdf89e6a982',21484);old=read('DEV5_165_NATIVE_COST_SUPPLEMENT.json','4dc7dbab1990b9088f272b6b77b63eaaf0020bdaaa1f19c4432e709b42d3ac6c',7041)
assert len(cost['mirrored_artifacts'])==12 and len(FIELDS)==5 and len(PARTS)==5
refs(cost)
for k in ['collector_source','reviewed_entry_selection','source_FULL','source_CUM','source_native_metadata','cost_outer_receipt','supplement','markdown','prior_immutable_supplement','same_entry_prior_actual_executor_attestation','execution_record','pre_execution_input_pins','owner_receipt_builder']:
 if k in owner:bound(owner[k])
for k in ['source_CUM_all_complete_field_parts','rows_part_refs','prior_rows_part_refs','local_original_control_refs']:refs(owner.get(k,[]))
for ref in summ['rows']['parts']+old['rows']['parts']:bound(ref)
rows=mod.rows_from_summary(B,M,summ);prior=mod.rows_from_summary(B,M,old);assert len(rows)==34 and len(prior)==33 and rows[:-1]==prior and [r['outer'] for r in rows]==list(range(5,171,5));assert summ['newly_read_remote_control_outers']==[170] and summ['reused_original_control_outers']==list(range(5,166,5))
full=bound(owner['source_FULL']);native=bound(owner['source_native_metadata']);poll=bound(full['input_bindings']['poll']);res=bound(full['input_bindings']['resource']);assert Path(full['input_bindings']['poll']['path']).name=='NATIVE5_CONTINUE_POLL_2104.json' and Path(full['input_bindings']['resource']['path']).name=='NATIVE5_CONTINUE_RESOURCE_2104.json'
for k,v in dict(source_commit=C,config_sha256=CFG,unit=UNIT,run=RUN,completed_outer=170,actual_training_pid=503255,actual_training_invocation_id=INV,goal_status='active').items():assert full[k]==v,k
ids=native['identity'];assert all(ids.get(k)==v for k,v in dict(code_commit=C,config_sha256=CFG,model_identity=MODEL,data_identity=DATA,completed_step=170,actual_update_limit=200,run_id=Path(RUN).name,job_id='not-slurm;systemd:'+C).items());assert native['archive_matches_original_manifest'] is True and native['manifest_status']=='COMPLETE_NATIVE_ROLLING_STATE_SAVED'
totals={str(i):3400 for i in range(8)};assert full['per_rank_actual_total_adam']==full['per_rank_successful_adam']==totals==cum['successful_optimizer_accounting']['successful_calls_per_rank_through170']==cum['successful_optimizer_accounting']['direct_observed_calls_per_rank'];workers=native['receipt']['workers'];assert len(workers)==8 and {w['rank'] for w in workers}==set(range(8))
for w in workers:assert w['actual_total_optimizer_calls']==3400 and w['observed_adam_steps']==[3400] and w['scheduler_last_epoch']==170 and w['full_worker_state_finite'] is True
assert native['receipt']['driver']['next_step']==171 and [r['outer'] for r in cum['rows_counted_once']]==list(range(1,171)) and cum['through_effective_outer']==170
actual=full['actual_dev512'];met=actual['metrics'];assert actual['outer']==170 and actual['rows']==512 and actual['phase']=='periodic' and actual['ledger_sha256']==INNER;assert met['val/qa_count/all']==512 and met['val/qa_count/nq']==met['val/qa_count/hotpotqa']==256 and met['val/f1/all']==0.5742947048611112 and met['val/em/all']==0.490234375
assert len(cum['completed_dev_evaluations'])==34 and [x['outer'] for x in cum['completed_dev_evaluations']]==list(range(5,171,5));dev=cum['completed_dev_evaluations'][-1];assert dev['F1']==met['val/f1/all'] and dev['EM']==met['val/em/all'] and dev['ledger_sha256']==INNER and dev['ledger']==actual['path']
assert poll['source_commit']==C and poll['unit']==UNIT and poll['run']==RUN and poll['progress']['actual_completed_outer_from_attempt_results']==170 and any(x.get('ledger_sha256')==INNER and x.get('metrics')==met for x in poll['development']);assert summ['model_identity']==MODEL and summ['data_identity']==DATA and summ['split_identity']['independent_split_file_sha256'] is None
oldmap={r['outer']:r for r in oldroot['control_file_and_inner_ledger_bindings']};controls=[];pins=[]
for row,dev in zip(rows,cum['completed_dev_evaluations']):
 n=row['outer'];sc=P if n==5 else C;run='/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:'+sc
 pin=dict(outer=n,code_commit=sc,config_sha256=mod.CONFIGS[sc],model_identity=MODEL,data_identity=DATA,job_id='not-slurm;systemd:'+sc,run=run,path=run+'/output/validation-cost/'+dev['ledger'],phase=dev['phase'],ledger_sha256=dev['ledger_sha256'],F1=dev['F1'],EM=dev['EM']);pins.append(pin);control=bound(row['local_original_summary']);cr=mod.read_bound(B,M,row['local_original_summary']);assert mod.control_row(pin,cr,row['source_control_summary'],row['local_original_summary'])==row
 assert control['metrics']['val/qa_count/all']==512 and control['metrics']['val/qa_count/nq']==control['metrics']['val/qa_count/hotpotqa']==256 and row['precise_engine_emitted_tokens'] is None and row['scoring_call_event_total'] is None
 if n<170:assert len(cr)==oldmap[n]['bytes'] and sh(cr)==oldmap[n]['control_file_sha256'] and row==prior[n//5-1]
 controls.append(dict(outer=n,path=row['local_original_summary']['path'],bytes=len(cr),control_file_sha256=sh(cr),inner_ledger_sha256=row['ledger_sha256'],source_commit=sc,prior33_object_and_original_bytes_preserved=n<170))
assert controls[-1]['bytes']==3606 and controls[-1]['control_file_sha256']=='17998fd4d10e221ddc095f2d8251c6890c5cb810dad7a19fef388301954cd94a' and controls[-1]['inner_ledger_sha256']==INNER
program=sh(mod.REMOTE.replace('PIN_DATA',repr(pins[-1:])).encode());assert program==summ['remote_program_sha256']==owner['remote_program_sha256'] and summ['collector_source_sha256']==owner['collector_source_sha256']==sh(sr) and owner['collector_source_commit']==ENTRY
assert owner['new_entry_main_invocations_this_task']==owner['actual_remote_incremental_collection_invocations']==1 and owner['preceding_local_pending_invocations']==0 and owner['remote_collector_exit_code']==0 and owner['same_entry_executor_attested_cumulative_main_invocations']==owner['same_entry_executor_attested_cumulative_remote_incremental_invocations']==3;assert oldowner['new_entry_main_invocations_this_task']==oldowner['actual_remote_incremental_collection_invocations']==1
assert owner['outer']==170 and owner['status']=='ACTUAL_DEV170_COST_COLLECTED_AND_LOCAL_MIRRORS_READ_BACK' and owner['same_entry_prior_confirmed_main_invocations']==owner['same_entry_prior_confirmed_remote_incremental_invocations']==2
assert oldowner['same_entry_executor_attested_cumulative_main_invocations']==oldowner['same_entry_executor_attested_cumulative_remote_incremental_invocations']==2
execution=bound(owner['execution_record']);assert execution['exit_code']==0 and execution['new_entry_main_invocations_this_task']==execution['actual_remote_incremental_collection_invocations']==1 and execution['preceding_local_pending_invocations']==0 and execution['source_commit']==ENTRY and execution['source_sha256']==sh(sr)
assert rows[-1]['counters']==dict(generation_attempts=63,generation_completed=63,generation_failed=0,submitted_rows=1664,returned_rows=1664,submitted_prefill_token_events=1273770,returned_nonpad_model_token_events=237339,real_sample_rows=512,active_row_events=1597,submitted_padding_row_events=67,http_attempts=63,http_completed=63,http_failed=0,http_query_events=1080,http_returned_query_rows=1080)
assert rows[-1]['wall_seconds']==586.9193585850298 and rows[-1]['trainer_process_cpu_seconds']==196.65070707700215
assert native['tensor_contents_read'] is False and native['independent_cold_reload_performed'] is False and native['receipt']['tensor_contents_read'] is False and native['receipt']['independent_cold_reload_performed'] is False
for key,record in summ['native_counter_totals'].items():
 vals=[(r['counters'] or {}).get(key) for r in rows];good=[x for x in vals if mod.numeric(x)];total=math.fsum(good) if good else None;missing=[r['outer'] for r,x in zip(rows,vals) if not mod.numeric(x)];assert record==dict(total=total if len(good)==len(vals) else None,known_sum=total,missing_outers=missing),key
for key,field in [('original_validation_observer_wall_seconds','wall_seconds'),('original_validation_trainer_process_cpu_seconds','trainer_process_cpu_seconds')]:
 vals=[r[field] for r in rows];good=[x for x in vals if mod.numeric(x)];total=math.fsum(good) if good else None;missing=[r['outer'] for r,x in zip(rows,vals) if not mod.numeric(x)];assert summ[key]==dict(total=total if len(good)==len(vals) else None,known_sum=total,missing_outers=missing)
assert summ['precise_engine_emitted_tokens'] is None and summ['scoring_call_event_total'] is None
input_observations=[]
for i in range(2):
 for name,ref in F.items():
  p=Path(name);r=p.read_bytes();assert r==R[name] and sh(r)==ref['sha256'] and len(r)==ref['bytes'],name
  if ref['mirror_checked']:
   for m in M:assert (m/p.relative_to(B)).read_bytes()==r,name
 observed=dict(input_readback_pass=i+1,observed_utc=datetime.now(timezone.utc).isoformat(),unique_files=len(F),mirrored_files=sum(x['mirror_checked'] for x in F.values()));input_observations.append(observed);print(json.dumps(observed))
 if i==0:time.sleep(1)
receipt=dict(schema='root_dev170_native_cost_actual_local_readback_v1',status='PASS_ACTUAL_DEV170_LOCAL_ARTIFACT_READBACK',observed_utc=datetime.now(timezone.utc).isoformat(),executor='/root/dev170_actual_readback',outer=170,goal_status='active',scope='Independent actual local artifact/reference/object readback only; actual DEV170 owner gate supplied by root. No collector invocation, remote query, new code review or training test.',candidate_entry_commit=ENTRY,adapted_from_reader_source=F[str(ORIGINAL)],collector_source=F[str(S)],source_pins={n:dict(bytes=F[str(B/n)]['bytes'],sha256=F[str(B/n)]['sha256']) for n in ['CUMULATIVE_COST_THROUGH170.json','COST_OUTER170_LOCAL_RECEIPT.json','DEV5_170_NATIVE_COST_SUPPLEMENT.json','DEV5_170_NATIVE_COST_LOCAL_RECEIPT.json','FULL_OUTER170_PROGRESS_RECEIPT.json','NATIVE_HISTORY_170_METADATA_READBACK.json']},actual_ref_counts=dict(cost_mirrored_artifacts=len(cost['mirrored_artifacts']),cum_complete_fields=len(FIELDS),cum_parts=len(PARTS),cum_completed_dev_entries=34,new_dev_row_parts=len(summ['rows']['parts']),new_dev_rows=34,old_dev_row_parts=len(old['rows']['parts']),old_dev_rows=33,all_original_control_summaries=34,preserved_old_control_summaries=33,new_remote_control_summaries=1,unique_audited_files=len(F),unique_mirrored_files=sum(x['mirror_checked'] for x in F.values()),input_byte_readback_passes=2,mirror_roots=2),cumulative_fields_dynamic=FIELDS,cumulative_part_refs=PARTS,new_row_part_refs=summ['rows']['parts'],old_row_part_refs=old['rows']['parts'],control_file_and_inner_ledger_bindings=controls,prior33_rows_exact_object_equal=True,prior33_control_original_bytes_SHA_preserved=True,original_CUM170_raw_bytes_SHA_preserved=True,actual170_identity=dict(source_commit=C,config_sha256=CFG,model_identity=MODEL,data_identity=DATA,rows=512,nq_rows=256,hotpotqa_rows=256,F1=met['val/f1/all'],EM=met['val/em/all'],phase='periodic',ledger=actual['path'],inner_ledger_sha256=INNER,control_file_bytes=3606,control_file_sha256=controls[-1]['control_file_sha256'],actual_total_adam_per_rank=totals,scheduler_last_epoch=170,native_driver_next_step=171,training_pid=503255,training_invocation=INV,first_FULL_poll='NATIVE5_CONTINUE_POLL_2104.json',first_FULL_resource='NATIVE5_CONTINUE_RESOURCE_2104.json',native_metadata_only=True,tensor_contents_read=False,independent_cold_reload_performed=False,PT_file_sha256_recomputed=False,PT_current_availability_after_capture_is_unknown=True),actual_new_collection_binding=dict(executor_receipt_attested_new_entry_main_invocations=1,executor_receipt_attested_remote_incremental_invocations=1,executor_receipt_attested_preceding_pending_invocations=0,executor_exit_code=0,execution_record=F[str(path(owner['execution_record']['path']).resolve())],same_entry_executor_attested_cumulative_main_invocations_160_plus165_plus170=3,same_entry_executor_attested_cumulative_remote_incremental_invocations_160_plus165_plus170=3,artifact_corroboration_new_remote_outers=[170],artifact_corroboration_reused_outers=list(range(5,166,5)),independently_reconstructed_remote_program_sha256=program,remote_program_pin_count=1,evidence_limit='Invocation counts are SHA-bound executor attestations corroborated by sole new170 row/control, prior165 cumulative2 receipt and one-pin remote-program hash; no independent system-wide process-log audit or remote execution observation is claimed.',this_readback_collector_or_SSH_invocations=0),actual170_counters=rows[-1]['counters'],validated_native_dev_counter_totals=summ['native_counter_totals'],original_validation_observer_wall_seconds=summ['original_validation_observer_wall_seconds'],original_validation_trainer_process_cpu_seconds=summ['original_validation_trainer_process_cpu_seconds'],accounting=dict(validation_wall_CPU_already_in_outer_service_not_added_again=True,precise_engine_emitted_tokens=None,scoring_call_event_total=None,independent_split_file_sha256=None,final_service_CPU_seconds=None,submitted_prefill_is_repeated_events=True,returned_nonpad_is_not_precise_engine_emitted=True,HTTP_calls_and_query_events_distinct=True),two_pass_local_input_and_mirror_bytes_verified=True,input_readback_observations=input_observations,mirrors=[str(m) for m in M],files=sorted(F.values(),key=lambda x:x['path']),remote_descriptors_not_followed=OBS,inconsistencies=[],owner_completion_gate='Root explicitly authorized actual170 reader after receiving owner59548B SHA29c0ea107ed07fe91b2d76742246c3909b0a32380927015022153c79cc364415, exit0/main1/remote1/PENDING0 and pinned170 artifacts.',raw_input_mirror_policy='Historical ROOT_DEV165 and explicit root scope confirm raw poll/resource are BASE-only. Both local SHA/bytes passes use mirror_checked=false; product artifacts are checked in both mirrors.',reviewer_actions=dict(remote_operations=False,collector_or_main_invoked=False,training_or_evaluation_modified=False,code_review_or_training_tests=False,tensor_model_rollout_JSONL_read=False,canonical_STATUS_REPORT_FUP_observer_modified=False,old_DEV165_outputs_rewritten=False,immutable_CUM_rewritten=False),receipt_publication=dict(new_file_only=True,copies=3,byte_readback_passes=2),raw_observation_sources_scope='Original FULL170/native/poll/resource/control values are local SHA-bound actual artifacts, with raw poll/resource mirrored=false by historical policy. No new remote observation/cold reload/final51713/HF/cleanup/goal completion is claimed.')
out=(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();assert len(out)<=100000;name='ROOT_DEV170_NATIVE_COST_ACTUAL_READBACK.json'
for folder in [B]+M:assert not (folder/name).exists() or (folder/name).read_bytes()==out
for folder in [B]+M:
 if not (folder/name).exists():(folder/name).write_bytes(out)
for i in range(2):
 for folder in [B]+M:assert (folder/name).read_bytes()==out
 for p in F:assert Path(p).read_bytes()==R[p]
 print(json.dumps(dict(receipt_readback_pass=i+1,observed_utc=datetime.now(timezone.utc).isoformat(),copies=3,source_inputs_preserved=True)))
 if i==0:time.sleep(1)
print(json.dumps(dict(status=receipt['status'],file=str(B/name),bytes=len(out),sha256=sh(out),dev_parts=len(summ['rows']['parts']),dev_rows=34,cum_parts=len(PARTS),cost_refs=len(cost['mirrored_artifacts']),unique_audited_files=len(F),unique_mirrored_files=sum(x['mirror_checked'] for x in F.values()))))
