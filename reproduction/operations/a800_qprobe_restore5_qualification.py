"""One published, bounded metadata qualification of an authoritative failed CP5.

No tensor, weights, QA, CUDA, verifier, recovery, launch, signal or retry. File
SHA/finite statements below are producer declarations, not new readback proof.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
OPS = Path(__file__).resolve().parents[2]
PRIMARY = OPS/'reproduction/operations/a800_qprobe_async_milestone_observer.py'
PRIMARY_SHA = '322975cd5d12b1ac5bb288588fcdb6e723c0e2fa5e838be93680e5cb2f1abe68'
PRIMARY_COMMIT = 'ade1289a25de192d1bf0e6c6bfeb8223d2b9a449'

REMOTE_QUALIFICATION = r'''
def keys(value):
 need(type(value) is dict and len(value)<=128,'metadata_key_schema')
 need(all(type(k) is str and re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,95}',k) for k in value),'metadata_key_name')
 return sorted(value)
def safe_path(path,parent):
 path=Path(path);need(path.resolve()==path and not path.is_symlink() and path.is_relative_to(parent),'canonical_checkpoint_path')
 value=path.stat();need(stat.S_ISREG(value.st_mode) and value.st_uid==os.getuid(),'checkpoint_file_owner')
 need(re.fullmatch(r'[A-Za-z0-9._/-]{1,512}',str(path.relative_to(base))),'checkpoint_filename')
 return dict(path=str(path),relative_path=str(path.relative_to(base)),bytes=value.st_size,mtime_ns=value.st_mtime_ns)
def terminal_ownership():
 stage,_=document('logs/pipeline-status.json');ledger,_=document('logs/process-ledger.json',PROCESS_LEDGER_LIMIT)
 need(stage['run_id']==ledger['run_id']==run['run_id'] and stage['stage']=='FAILED','not_failed_authoritative_run')
 need(not terminal_births['pipeline']['birth_matches'] and not terminal_births['trainer']['birth_matches'],'leader_birth_not_gone')
 records={};jobs=[]
 for name,job in ledger['jobs'].items():
  need(re.fullmatch('[A-Za-z0-9_.-]{1,64}',name) and job['state']=='STOPPED','owned_job_not_stopped')
  for record in list(job['ledger'].values())+[job['leader']]:
   records[tuple(record[k] for k in ('pid','start_ticks','sid','uid'))]=record
  jobs.append(dict(name=name,state=job['state'],exit_code=scalar(job.get('exit_code'))))
 need(0<len(records)<=8192,'owned_birth_count')
 states=[process(record) for record in records.values()]
 need(not any(value['birth_matches'] for value in states),'recorded_owned_birth_not_gone')
 trainer_job=ledger['jobs']['trainer'];need(trainer_job['exit_code']==1,'unexpected_trainer_terminal_exit')
 project=[];scanned=0
 for proc in Path('/proc').iterdir():
  check()
  if not proc.name.isdigit():continue
  scanned+=1;need(scanned<=65536,'proc_scan_limit')
  try:
   if proc.stat().st_uid!=os.getuid():continue
   text=(proc/'stat').read_text();fields=text[text.rfind(')')+2:].split()
   if fields[0] in ('Z','X'):continue
   cwd=(proc/'cwd').resolve()
   namespace=next((p for p in (cwd,*cwd.parents) if re.fullmatch('/data/shaoshitong/Search-G1-a800-fresh200-[0-9a-f]{32}',str(p))),None)
   if namespace is None:continue
   with (proc/'cmdline').open('rb') as stream:command=stream.read(8193)
   need(len(command)<=8192,'proc_command_limit')
   labels=[]
   if any(x in command for x in (b'main_ppo',b'TaskRunner',b'train.sh',b'ActorRolloutRefWorker')):labels.append('trainer_or_training_worker')
   if b'run_pipeline' in command:labels.append('pipeline')
   if not labels:continue
   identity=dict(pid=int(proc.name),start_ticks=int(fields[19]),sid=int(fields[3]),uid=os.getuid())
   value=process(identity)
   if not value['alive']:continue
   need(len(project)<128,'project_process_count');project.append(dict(**value,namespace=str(namespace),roles=labels))
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 need(not project,'project_namespace_training_process_live')
 return dict(status='observed',pipeline_stage='FAILED',pipeline_failed_unix=scalar(stage.get('unix')),
  recorded_unique_births=len(states),all_recorded_births_gone=True,current_namespace_trainers=[],
  all_project_namespace_trainers=[],all_project_namespace_trainers_gone=True,jobs=jobs,
  scope='exact PID/start/session/UID, plus bounded project trainer classification; command text never returned')
def evidence_trace():
 raw,info=snapshot(base/'logs/trainer.log',65536,append=True,tail=True);need(raw is not None,'trainer_log_missing')
 text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',raw.decode(errors='replace'))
 text=re.sub(r'^\((?:TaskRunner|ActorRolloutRefWorker|WorkerDict|RayWorker) pid=[0-9]+\)\s*','',text,flags=re.MULTILINE)
 frames=[dict(file=Path(m.group(1)).name[:160],line=int(m.group(2)),function=m.group(3)[:96]) for m in
  re.finditer(r'^[ \t]*File "([^"\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)[ \t]*$',text,re.MULTILINE)][-8:]
 allowed={'Missing actual common8 monitor tensor hashes':'missing_actual_common8_monitor_tensor_hashes',
  'Unexpected monitor identity schema':'unexpected_monitor_identity_schema',
  'Missing local monitor identity':'missing_local_monitor_identity',
  'Unsupported local monitor count':'unsupported_local_monitor_count',
  'Incomplete local monitor indices':'incomplete_local_monitor_indices',
  'Incomplete local monitor lengths':'incomplete_local_monitor_lengths'}
 matches=re.findall(r'^[ \t]*(?:[A-Za-z_][A-Za-z0-9_]*\.)*(EvidenceError): ([^\r\n]{1,256})\r?$',text,re.MULTILINE)
 return dict(status='observed',bounded_tail_sha256=info['scanned_sha256'],frames=frames,
  exception_type=matches[-1][0] if matches else None,
  fixed_message_code=allowed.get(matches[-1][1]) if matches else None,
  message_status='fixed_whitelist_matched' if matches and matches[-1][1] in allowed else 'unknown_message_not_returned',
  scope='only exact known evidence-error message codes and frame metadata; no raw message/source/log text')
def monitor_keys():
 result=[]
 for rank in range(4):
  values,info=rows('logs/optimizer-rank-'+str(rank)+'.jsonl');need(values is not None and info['scan_from_start'],'monitor_ledger_missing')
  seen=set();items=[]
  for row in values:
   need(row.get('run_id')==run['run_id'] and row.get('rank')==rank,'monitor_row_identity')
   if row.get('kind')!='POLICY_MONITOR':continue
   context=row['context'];aid=context['attempt_id'];ordinal=context['attempt_index'];outer=context['logical_update']
   need(re.fullmatch('[0-9a-f]{32}',aid) and integer(ordinal,1) and integer(outer,1),'monitor_attempt_identity')
   if aid in seen:continue
   seen.add(aid);identity=row['monitor_identity'];hashes=identity['tensor_sha256']
   names=keys(hashes)
   allowed_tensor_names={'responses','input_ids','attention_mask','position_ids','old_log_probs','advantages','loss_mask','ref_log_prob'}
   need(set(names)<=allowed_tensor_names,'unexpected_monitor_tensor_name')
   for name,value in hashes.items():hash_value(value)
   summary=dict(attempt_id=aid,attempt_index=ordinal,logical_update=outer,monitor_identity_keys=keys(identity),
    tensor_keys=names,tensor_sha256=hashes,monitor_identity_sha256=digest(canonical(identity)),
    selection_indices_sha256=digest(canonical(identity.get('selection_indices'))),
    actor_token_lengths_sha256=digest(canonical(identity.get('actor_token_lengths'))),
    has_ref_log_prob='ref_log_prob' in hashes,source_row_sha256=digest(canonical(row)))
   need(len(canonical(summary))<=2048,'monitor_identity_summary_limit');items.append(summary)
  need(len(items)==5 and [x['attempt_index'] for x in items]==list(range(1,6)),'monitor_attempt_prefix')
  result.append(dict(rank=rank,first_per_attempt=items,prefix_sha256=info['complete_prefix_sha256']))
 return dict(status='observed',per_rank=result,scope='first monitor per rank/attempt, hash and key names only; no token or tensor values')
def scheduler_metadata():
 output=[]
 names=('logical_update','attempt_index','completed_outer_before','completed_outer_after','scheduler_last_epoch_before',
  'scheduler_last_epoch_after','scheduler_step_count_before','scheduler_step_count_after','writes_before','writes_after',
  'actual_optimizer_writes','opportunities_processed','scheduler_advances','process_pid','process_start_ticks','base_learning_rate','warmup_steps')
 kinds={'BEGIN','COMMIT_PREPARED','COMPLETE','REJECTED','POST_COMMIT_FAILURE','PRE_COMMIT_EVIDENCE_FAILURE','ABORTED_ATTEMPT'}
 for rank in range(4):
  values,info=rows('logs/scheduler-rank-'+str(rank)+'.jsonl');need(values is not None and info['scan_from_start'] and not info['partial_final_bytes'],'scheduler_prefix')
  need(len(values)<=1000,'scheduler_row_limit');complete=[]
  for value in values:
   same_execution(value);need(value['rank']==rank and value['source_map_sha256']==source_map_sha and value['kind'] in kinds,'scheduler_identity')
   if value['kind']!='COMPLETE':continue
   item=select(value,names);item['kind']='COMPLETE';item['attempt_id']=value['attempt_id']
   need(re.fullmatch('[0-9a-f]{32}',item['attempt_id']),'scheduler_attempt')
   for key in ('consumed_learning_rates','next_learning_rates'):
    rates=value[key];need(type(rates) is list and 0<len(rates)<=16 and all(number(x) for x in rates),'scheduler_lr');item[key]=rates
   complete.append(item)
  need([x['logical_update'] for x in complete]==list(range(1,6)) and complete[-1]['completed_outer_after']==5 and
   complete[-1]['scheduler_last_epoch_after']==5 and complete[-1]['writes_after']==100,'scheduler_cp5_boundary')
  output.append(dict(rank=rank,complete_rows=complete,prefix_sha256=info['complete_prefix_sha256']))
 return dict(status='observed',per_rank=output,scope='durable scheduler scalar records only; no saved optimizer/scheduler state opened')
def checkpoint_metadata_full():
 directory=base/'checkpoints/search-g1-common8/actor/global_step_5';prefix=str(directory.relative_to(base))
 need(Path(run['checkpoint_dir'])==base/'checkpoints/search-g1-common8','checkpoint_namespace')
 complete,ci=document(prefix+'/complete.json');manifest,mi=document(prefix+'/training_state/manifest.json')
 need(complete is not None and manifest is not None and complete['run_id']==run['run_id'] and
  complete['step']==manifest['completed_outer']==5 and complete['checkpoint']==str(directory),'cp5_identity')
 need(complete['finite_all_saved_tensors'] is True and complete['actual_optimizer_writes_per_rank']==100,'producer_cp5_finite_counter')
 def proof(value,rank=None):
  names=keys(value);out=safe_path(value['path'],directory/'training_state')
  need(integer(value['bytes'],1) and out['bytes']==value['bytes'],'declared_proof_bytes')
  out.update(declared_bytes=value['bytes'],declared_sha256=hash_value(value['sha256']),proof_keys=names)
  if rank is not None:
   need(value['rank']==rank and value['run_id']==run['run_id'] and value['actual_optimizer_writes']==100,'worker_proof_identity')
   out.update(rank=rank,run_id=value['run_id'],declared_actual_optimizer_writes=100)
  return out
 workers=manifest['worker_files'];need(sorted(x['rank'] for x in workers)==[0,1,2,3],'cp5_ranks')
 q=manifest['question_probe'];need(q['completed_outer']==5 and q['actor_snapshot_id']==run['run_id']+':5','qprobe_actor_boundary')
 head=q['head'];need(head['completed_outer']==5 and head['actor_snapshot_id']==q['actor_snapshot_id'] and
  head['feature_spec_sha']==q['feature_spec_sha'] and head['version']==q['head_version'],'head_boundary')
 head_proof=proof(head);head_proof.update(checkpoint_sha=hash_value(head['checkpoint_sha']),version=version(head['version']),
  actor_snapshot_id=head['actor_snapshot_id'],completed_outer=5,feature_spec_sha=hash_value(head['feature_spec_sha']),update_count=scalar(head.get('update_count')))
 need(head_proof['declared_sha256']==head_proof['checkpoint_sha'],'head_declared_hash_alias')
 declared=complete['files_sha256'];need(type(declared) is dict and len(declared)<=128,'cp5_declared_files')
 declared_files=[]
 for name,sha in sorted(declared.items()):
  need(type(name) is str and re.fullmatch(r'[A-Za-z0-9._/-]{1,512}',name) and '..' not in Path(name).parts,'declared_file_name')
  declared_files.append(dict(**safe_path(directory/name,directory),declared_sha256=hash_value(sha)))
 inventory=[]
 for path in directory.rglob('*'):
  check();need(path.resolve()==path and not path.is_symlink(),'checkpoint_symlink')
  value=path.stat();need(value.st_uid==os.getuid(),'checkpoint_owner')
  if stat.S_ISREG(value.st_mode):
   need(len(inventory)<128,'checkpoint_inventory_limit');inventory.append(safe_path(path,directory))
 config,config_info=document(prefix+'/config.json');index,index_info=document(prefix+'/model.safetensors.index.json')
 need(config is not None and index is not None and type(index['weight_map']) is dict,'hf_schema_metadata')
 shards=sorted(set(index['weight_map'].values()));need(len(shards)<=16 and len(index['weight_map'])==complete['validated_tensor_count'],'hf_declared_layout')
 shape=select(config,('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads',
  'vocab_size','max_position_embeddings','tie_word_embeddings'))
 if type(config.get('torch_dtype')) is str:shape['torch_dtype']=version(config['torch_dtype'])
 architectures=config.get('architectures');need(type(architectures) is list and len(architectures)<=4,'hf_architectures')
 shape['architectures']=[version(x) for x in architectures]
 return dict(status='observed',checkpoint=str(directory),completed_outer=5,training_world_size=4,
  source_run_manifest_sha256=run_manifest_sha,source_run_id=run['run_id'],source_execution_id=run['execution_id'],source_commit=run['source_commit'],
  source_execution_backend=run['execution_backend'],source_slurm_job_id=run['slurm_job_id'],
  complete_metadata_sha256=ci['scanned_sha256'],manifest_metadata_sha256=mi['scanned_sha256'],
  producer_complete=select(complete,('step','finite_all_saved_tensors','validated_tensor_count','actual_optimizer_writes_per_rank')),
  declared_files=declared_files,checkpoint_inventory=inventory,manifest_keys=keys(manifest),
  worker_proofs=[proof(w,w['rank']) for w in workers],driver_proof=proof(manifest['driver_file']),
  question_probe=dict(**proof(q),completed_outer=5,actor_snapshot_id=q['actor_snapshot_id'],head_version=version(q['head_version']),
   feature_spec_sha=hash_value(q['feature_spec_sha']),head=head_proof),
  hf_config_sha256=config_info['scanned_sha256'],hf_index_sha256=index_info['scanned_sha256'],hf_shape_metadata=shape,
  hf_shard_declared_files=[safe_path(directory/name,directory) for name in shards],
  restore_mode_metadata=dict(candidate_kind='question_probe_completed_actor_head_candidate',saved_tensor_shapes='unknown_not_opened',
   driver_cursor_and_rng='unknown_not_opened',actual_full_sha_readback=False,actual_tensor_finite_recheck=False,actual_cuda_fsdp_restore_accepted=False),
  scope='canonical JSON and declared SHA/finite producer fields plus stat only; this does not grant reload/recovery acceptance')
raw_payload=sys.stdin.read(1024*1024+1);need(len(raw_payload.encode())<=1024*1024,'payload_limit')
request=json.loads(raw_payload);sys.stdin=io.StringIO(raw_payload)
result=dict(role='failed_qprobe_cp5_metadata_qualification',run_id=request['expected_run'],execution_id=request['expected_execution'],
 source_commit=request['intent']['source_commit'],base=request['intent']['base'],status='unknown',metadata_only=True,read_once=True,
 remote_mutation=False,no_model_or_tensor_reads=True,no_qa_or_token_files=True,no_native_verifier=True,recovery_accepted=False)
try:
 exec(common_code,globals());run=pinned_run();need(run.get('question_probe') is not None,'not_qprobe')
 source_map_sha=digest(canonical(run['source_sha256']));run_manifest_sha=digest(budget_bytes(base/'logs/run-manifest.json',1024*1024))
 complete_updates=None;closed_attempts=None;read_pass='terminal_births';terminal_births=births();result['births']=terminal_births
 read_pass='terminal_ownership';result['terminal_ownership']=terminal_ownership()
 read_pass='driver_prefix';result['driver_prefix']=reward_window();need(len(complete_updates)==5,'driver_not_exact_cp5')
 read_pass='four_rank_prefix';counts=opportunity_counts();result['four_rank_prefix']=counts
 need(counts['status']=='observed' and counts['four_rank_complete_prefix_agreement'] is True and
  counts['cumulative_processed']==counts['cumulative_successful_adam']==100 and counts['cumulative_skip']==0 and
  counts['current_partial'] is None,'four_rank_not_exact_cp5')
 read_pass='evidence_trace';result['evidence_trace']=section(evidence_trace)
 read_pass='monitor_keys';result['monitor_identity_metadata']=section(monitor_keys)
 read_pass='scheduler';result['scheduler_metadata']=section(scheduler_metadata)
 read_pass='checkpoint';result['checkpoint5_metadata']=section(checkpoint_metadata_full)
 read_pass='existing_scores';result['existing_scores']=section(scores)
 result['metadata_fields_complete']=all(result[name].get('status')=='observed' for name in
  ('monitor_identity_metadata','scheduler_metadata','checkpoint5_metadata'))
 need('torch' not in sys.modules,'unexpected_torch_import')
 result['status']='observed';result['metadata_files']=metadata_files
except (Unknown,AssertionError,KeyError,IndexError,TypeError,ValueError,OverflowError,OSError) as failure:
 result.update(status='unknown',reason=str(failure) if isinstance(failure,Unknown) else type(failure).__name__)
finally:
 signal.alarm(0);result.update(observed_unix=time.time(),aggregate_bytes_read=spent,bounded_failures=bounded_failures,
  limits=dict(aggregate_read_bytes=TOTAL_LIMIT,ledger_bytes=LEDGER_LIMIT,process_ledger_bytes=PROCESS_LEDGER_LIMIT,
   output_bytes=128*1024,deadline_seconds=65,monitor_summary_bytes=2048,trace_tail_bytes=65536))
encoded=json.dumps(result,allow_nan=False)
if len(encoded.encode())>128*1024:encoded=json.dumps(dict(role=result['role'],run_id=result['run_id'],execution_id=result['execution_id'],
 source_commit=result['source_commit'],base=result['base'],status='unknown',reason='output_limit',recovery_accepted=False),allow_nan=False)
print('RESTORE5_QUALIFICATION_JSON:'+encoded,flush=True)
'''


def load_reader():
    raw = PRIMARY.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PRIMARY_SHA
    assert subprocess.check_output(['git','-C',str(OPS),'show',PRIMARY_COMMIT+':'+str(PRIMARY.relative_to(OPS))],timeout=30) == raw
    head = subprocess.check_output(['git','-C',str(OPS),'rev-parse','HEAD'],text=True,timeout=30).strip()
    own = Path(__file__).resolve()
    assert subprocess.check_output(['git','-C',str(OPS),'show',head+':'+str(own.relative_to(OPS))],timeout=30) == own.read_bytes(), 'Publish exact operator first'
    spec = importlib.util.spec_from_file_location('published_restore5_primary',PRIMARY)
    primary = importlib.util.module_from_spec(spec);spec.loader.exec_module(primary)
    flex,commit,remote,common_sha = primary.load_flex()
    assert remote.endswith(primary.REMOTE_READ)
    prefix = remote[:-len(primary.REMOTE_READ)]
    marker = 'raw_payload=sys.stdin.read(1024*1024+1)'
    assert primary.REMOTE_READ.count(marker)==1
    definitions = primary.REMOTE_READ.split(marker,1)[0]
    remote = prefix+definitions+'\n'+REMOTE_QUALIFICATION
    ast.parse(remote)
    return flex,commit,remote,common_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handle',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--expected-run',required=True)
    parser.add_argument('--expected-execution',required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    assert all(re.fullmatch('[0-9a-f]{32}',v) for v in (args.expected_run,args.expected_execution))
    args.handle.relative_to(ROOT/'preparation');args.receipt.relative_to(ROOT/'preparation')
    assert args.receipt.is_absolute() and args.receipt.resolve()==args.receipt and args.receipt.suffix=='.json' and not args.receipt.exists()
    flex,commit,remote,common_sha=load_reader();intent=flex.read(args.handle)
    authority_sha=flex.authority(args.handle,intent,args.expected_run,args.expected_execution)
    module=flex.runtime_module(args.runtime,intent);route=module.ssh_command(remote)
    assert route[-1].startswith('python3 -c ')
    route[-1]='/data/search-g1/env/bin/python -c '+route[-1][len('python3 -c '):]
    flex.authority(args.handle,intent,args.expected_run,args.expected_execution)
    result=dict(role='single_failed_qprobe_cp5_metadata_qualification_receipt',operator_commit=commit,
        operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),primary_sha256=PRIMARY_SHA,
        reused_common_sha256=common_sha,remote_reader_sha256=hashlib.sha256(remote.encode()).hexdigest(),
        authority_sha256=authority_sha,handle_sha256=hashlib.sha256(args.handle.read_bytes()).hexdigest(),
        run_id=args.expected_run,execution_id=args.expected_execution,source_commit=intent['source_commit'],base=intent['base'],
        no_automatic_retry=True,recovery_accepted=False,observation=None)
    flex.once(args.receipt.with_suffix('.claim.json'),dict(role='once_readonly_restore5_qualification_claim',
        run_id=args.expected_run,execution_id=args.expected_execution,source_commit=intent['source_commit'],
        operator_sha256=result['operator_sha256'],created_unix=time.time(),remote_mutation=False))
    payload=json.dumps(dict(intent=intent,expected_run=args.expected_run,expected_execution=args.expected_execution,after_step=0),allow_nan=False).encode()
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        try:
            actual=subprocess.run(route,input=payload,stdout=out,stderr=err,timeout=90)
            result.update(exit_code=actual.returncode,status='READ_RETURNED')
        except (OSError,subprocess.TimeoutExpired) as failure:
            result.update(exit_code=None,status='TIMEOUT_OR_TRANSPORT_UNKNOWN',failure_type=type(failure).__name__)
        result.update(stdout_bytes=out.tell(),stderr_bytes=err.tell());out.seek(0);raw=out.read(128*1024+1)
        err.seek(0);stderr=err.read(64*1024);result['bounded_stderr_sha256']=hashlib.sha256(stderr).hexdigest()
        result['remote_exception_frames']=[dict(file=Path(m.group(1)).name[:160],line=int(m.group(2)),function=m.group(3)[:96])
            for m in re.finditer(r'^[ \t]*File "([^"\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)[ \t]*$',stderr.decode(errors='replace'),re.MULTILINE)][-8:]
    if result['exit_code']==0 and len(raw)<=128*1024:
        try:
            rows=[json.loads(line.split(':',1)[1]) for line in raw.decode().splitlines() if line.startswith('RESTORE5_QUALIFICATION_JSON:')]
            assert len(rows)==1;value=rows[0]
            assert (value['run_id'],value['execution_id'],value['source_commit'],value['base'])==(args.expected_run,args.expected_execution,intent['source_commit'],intent['base'])
            assert value['role']=='failed_qprobe_cp5_metadata_qualification' and value['metadata_only'] is True and value['read_once'] is True
            result['observation']=value
        except (AssertionError,KeyError,TypeError,ValueError,UnicodeError) as failure:
            result.update(status='SANITIZED_OUTPUT_REJECTED',failure_type=type(failure).__name__)
    elif len(raw)>128*1024:result['status']='OUTPUT_LIMIT_EXCEEDED'
    result['returned_unix']=time.time();flex.once(args.receipt,result)
    print(json.dumps(dict(receipt=str(args.receipt),exit_code=result['exit_code'],status=result['status'],
        observation=result['observation'],no_automatic_retry=True,recovery_accepted=False),allow_nan=False))
    return 0 if result['observation'] is not None and result['observation']['status']=='observed' else 1


if __name__=='__main__':
    raise SystemExit(main())
