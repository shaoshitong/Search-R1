"""Read-only CPU metadata/producer-proof final audit of the completed 0850 build.

Never opens features, tensors, raw rollouts or model weights; never kills or
launches work. One SSH query checks immutable manifests, recorded full-SHA
producer proofs and actual owned-process/CUDA release. This does not repeat
the feature producer's full-byte verification.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import textwrap
import time

ROOT=Path('/Users/shaoshitong/Documents/Search-G1')
CONTROLLER=Path(__file__).with_name('a800_qprobe_prefix_scale20.py')
FINISHED=ROOT/'preparation/qprobe-prefix-scale20-native-observation-1792140512182782000.json'
BASE='/data/shaoshitong/Search-G1-a800-fresh200-0850db2977bd4801b9ced300b0ab387f'
RUN='62407ad0ed404586998439ca1f55b902'
EXECUTION='dd3032e313e14ce2b9c8fea996aaf721'
SOURCE='fcb77204f48d5cb8ec21c68df7d4a2c937b3557a'
OPERATOR='0277a4b0b5466b5f8767624b544c0001b47a89d6'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


AUDIT=r'''
import re
phase='source_and_stage'
metadata_bytes=0;metadata_hashes={};sealed_file_count=0;sealed_bytes=0
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def file(q):
 q=pathlib.Path(q);assert q.is_absolute() and q.resolve()==q and not q.is_symlink()
 s=q.stat();assert q.is_file() and s.st_uid==os.getuid();return s
def document(q,expected=None,limit=4*1024**2):
 global metadata_bytes
 q=pathlib.Path(q);s=file(q);assert s.st_size<=limit
 metadata_bytes+=s.st_size;assert metadata_bytes<=64*1024**2
 raw=q.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 if expected is not None:assert digest==expected
 metadata_hashes[str(q.relative_to(b)) if q.is_relative_to(b) else 'reused-cache-manifest']=digest
 return json.loads(raw)
def seal_files(path,files):
 global sealed_file_count,sealed_bytes
 for name,proof in files.items():
  assert pathlib.Path(name).name==name and re.fullmatch(r'[0-9a-f]{64}',proof['sha256'])
  s=file(path/name);assert s.st_size==proof['bytes']
  sealed_file_count+=1;sealed_bytes+=s.st_size
source();assert str(b)==p['expected']['base']
staged=document(b/'logs/staged.json');stage=document(b/'stage-complete.json',staged['producer_stage_sha256'])
run=document(b/'logs/run-manifest.json');expect=p['expected']
assert run['run_id']==stage['run_id']==expect['run_id'] and run['execution_id']==stage['execution_id']==expect['execution_id']
assert run['source_commit']==h['runtime_source']==expect['runtime_source']
assert staged['operator_commit']==h['operator_commit']==expect['operator_commit']
assert run['execution_backend']=='ssh_local' and run['slurm_job_id'] is None
assert run['actor_training_budget']==run['head_training_budget']==0 and run['diagnostic_only'] is True
assert run['training_physical_indices']==h['preferred_training_indices']==[0,2,3,4]
assert len(run['training_gpu_uuids'])==len(set(run['training_gpu_uuids']))==4
assert digest(b/run['gpu_placement_binding_receipt'])==run['gpu_placement_binding_sha256']
launch=document(b/'logs/launch.json');claim=document(b/'logs/launch-once.json')
assert launch['training_indices']==claim['training_indices']==run['training_physical_indices']
assert claim['training_uuids']==run['training_gpu_uuids']
assert stage['config_sha256']==digest(b/'teacher-config.json')
assert stage['selection_sha256']==run['diagnostic_selection_sha256']==digest(b/'selection.json')
root=pathlib.Path(h['dataset_dir']);assert root==b/'dataset'
plan=document(root/'plan.json',stage['dataset_plan_sha256']);plan_sha=metadata_hashes['dataset/plan.json']
assert plan['runtime_source_commit']==run['source_commit'] and plan['training_forbidden'] is True
assert plan['question_count']==102400 and plan['trajectories']==512000 and plan['reused_questions']==5120 and plan['new_questions']==97280
assert plan['raw_policy']=='compact-after-verified' and stage['dataset_dir']==str(root)
assert plan['selection_sha256']==stage['selection_sha256']
phase='dataset_and_compact_manifests'
complete=document(root/'complete.json',expect['dataset_complete_sha256'])
teacher=document(b/'teacher-complete.json');teacher_exit=document(b/'teacher-exit.json');driver_exit=document(b/'diagnostic-driver-exit.json')
for value in (teacher,teacher_exit,driver_exit):
 assert value['run_id']==run['run_id'] and value['execution_id']==run['execution_id']
assert teacher_exit['exit_code']==driver_exit['exit_code']==0
assert all(teacher[k]==v for k,v in complete.items())
assert complete['question_count']==102400 and complete['trajectories']==512000 and complete['chunks']==200
assert complete['reused_questions']==5120 and complete['new_questions']==97280 and complete['fixed_heldout_questions']==512
assert complete['heldout_regenerated'] is False and complete['actual_actor_optimizer_updates']==complete['actual_head_optimizer_updates']==0
assert complete['dataset_plan_sha256']==plan_sha and complete['selection_sha256']==plan['selection_sha256']
assert complete['old_identity_sha256']==plan['old_identity_sha256'] and complete['old_complete_sha256']==plan['old_complete_sha256']
assert complete['protocol_sha256']==hashlib.sha256(canonical(plan['protocol'])).hexdigest()
frozen_final=document(b/'teacher-cache/frozen-final.json');teacher_frozen=frozen_final['frozen_workers']
assert teacher_frozen['actual_frozen_rank_readback'] is True and teacher_frozen['actual_world_size']==4
before=sorted(teacher_frozen['before'],key=lambda x:x['rank']);after=sorted(teacher_frozen['after'],key=lambda x:x['rank'])
assert [x['rank'] for x in before]==[x['rank'] for x in after]==list(range(4))
frozen_keys=('rank','parameter_sha256','optimizer_sha256','scheduler_sha256','actual_optimizer_writes','completed_outer')
assert all({k:a[k] for k in frozen_keys}=={k:z[k] for k in frozen_keys} and a['actual_optimizer_writes']==a['completed_outer']==0 for a,z in zip(before,after))
assert len(frozen_final['worker_rng_restored'])==4 and all(x['rng_equal'] is True for x in frozen_final['worker_rng_restored'])
split=complete['split_proof'];assert split==plan['split_proof']
assert all(split[k] is True for k in ('disjoint_qid','disjoint_question_hash','disjoint_normalized_question_content','fixed_heldout_teacher_reused'))
assert split['heldout_labels_used_for_training'] is False and split['fixed_heldout_questions']==512
chunks=document(root/'chunk-manifest.json',complete['chunk_manifest_sha256'])
compact=document(root/'compact-manifest.json',complete['compact_manifest_sha256'])
assert chunks['dataset_plan_sha256']==compact['dataset_plan_sha256']==plan_sha
assert compact['stage']=='PREFIX_TEACHER_SCALE20_COMPACT_COMPLETE' and compact['runtime_commit']==run['source_commit']
assert compact['train_questions']==102400 and compact['train_trajectories']==512000 and compact['new_questions']==97280 and compact['new_trajectories']==486400
assert compact['selection_sha256']==plan['selection_sha256'] and compact['reuse']==plan['reuse'] and compact['split_proof']==split
spec=compact['spec'];assert spec['representation']=='actual_prompt_policy_prefix5_alltokens_layer27_v1'
assert spec['encoder_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1' and spec['feature_layer']==27 and spec['feature_dim']==2048
assert spec['encoder_compute_dtype']==spec['feature_dtype']=='bfloat16'
assert {k:v for k,v in spec.items() if k!='global_max_tokens'}=={k:v for k,v in plan['encoder_spec'].items() if k!='global_max_tokens'}
assert spec['global_max_tokens']==complete['global_max_tokens']
reuse=compact['reuse'];assert reuse['train_trajectories']==25600 and reuse['test_trajectories']==2560 and reuse['count']==28160
assert reuse['actual_full_sha_verified'] is True and reuse['source_ids_and_labels_exact_equal'] is True
qualified=reuse['qualification'];assert qualified['role']=='stage_actual_reused_prefix_cache_full_sha_seal'
assert qualified['actual_full_sha_verified'] is True and qualified['runtime_source_commit']==run['source_commit']
assert qualified['producer_source_sha256']==h['operator_sha256']['qprobe_prefix_teacher_scale20.py']
assert qualified['manifest_sha256']==reuse['manifest_sha256']
old=pathlib.Path(reuse['cache_dir']);old_manifest=document(old/'manifest.json',reuse['manifest_sha256'])
assert old_manifest['count']==28160 and old_manifest['split_counts']==dict(train=25600,test=2560)
assert qualified['files']==old_manifest['files'];seal_files(old,qualified['files'])
assert len(chunks['chunks'])==200 and [x['index'] for x in chunks['chunks']]==list(range(200))
assert len(compact['shards'])==190 and [x['ordinal'] for x in compact['shards']]==list(range(10,200))
new_entries={x['ordinal']:x for x in compact['shards']};total_tokens=0;minimum=None;maximum=0
phase='sealed_chunk_metadata'
for entry in chunks['chunks']:
 i=entry['index'];path=root/'shards'/('shard-%02d'%(i//10))/('chunk-%02d'%(i%10))
 assert entry['relative_path']==str((path/'commit.json').relative_to(root))
 value=document(path/'commit.json',entry['sha256']);assert value['index']==i and value['shard']==i//10 and value['shard_chunk']==i%10
 assert value['dataset_plan_sha256']==plan_sha and value['questions']==512 and value['trajectories']==2560
 assert value['actual_zero_optimizer_frozen_readback'] is True and value['actual_actor_optimizer_updates']==value['actual_head_optimizer_updates']==0
 prefix=document(path/'prefix/manifest.json',value['files']['prefix/manifest.json']['sha256'])
 assert prefix['ordinal']==i and prefix['count']==prefix['trajectories']==2560 and prefix['total_tokens']==value['total_prefix_tokens']
 assert prefix['exact_int64_roundtrip_verified'] is True and prefix['unknown_zero_policy_count']==0
 assert prefix['input_ids_dtype']==prefix['position_ids_dtype']=='int32' and prefix['policy_mask_dtype']=='bool'
 assert prefix['actual_pad_token_id']==spec['pad_token_id'] and prefix['global_max_tokens']<=spec['global_max_tokens']
 seal_files(path/'prefix',prefix['files'])
 assert set(value['files'])=={'labels.json','prefix/manifest.json'}|{'prefix/'+name for name in prefix['files']}
 for name,proof in value['files'].items():
  assert file(path/name).st_size==proof['bytes']
  if name.startswith('prefix/') and name!='prefix/manifest.json':assert proof==prefix['files'][name[7:]]
 assert value['raw_rollout']['origin']==('reused_qualified_teacher' if i<10 else 'new_official_teacher')
 if i>=10:
  e=new_entries[i];assert e['path']==str(path/'prefix') and e['count']==2560 and e['manifest_sha256']==value['files']['prefix/manifest.json']['sha256']
  removed=document(path/'raw-compacted.json');assert removed['accepted_commit_sha256']==entry['sha256']
  assert removed['raw_sha256']==value['raw_rollout']['sha256'] and removed['raw_bytes']==value['raw_rollout']['bytes']
  assert removed['removed_after_labels_and_prefix_full_sha'] is True and not (path/'rollout.pt').exists()
 total_tokens+=prefix['total_tokens'];maximum=max(maximum,prefix['global_max_tokens']);minimum=prefix['minimum_tokens'] if minimum is None else min(minimum,prefix['minimum_tokens'])
assert total_tokens==complete['total_prefix_tokens'] and maximum<=complete['global_max_tokens'] and minimum==complete['minimum_tokens']
phase='encoder_and_composite_proofs'
features=b/'features';finished=document(features/'complete.json');assert finished==expect['features_complete']
composite=document(features/'composite-manifest.json',finished['composite_manifest_sha256'])
assert composite['stage']==finished['stage']=='PREFIX_SCALE20_COMPOSITE_FEATURES_COMPLETE'
assert composite['train_questions']==finished['train_questions']==102400 and composite['train_trajectories']==finished['train_trajectories']==512000
assert composite['new_trajectories']==486400 and composite['reused_train_trajectories']==25600
assert composite['fixed_test_trajectories']==finished['fixed_test_trajectories']==2560
assert composite['reuse']==reuse and composite['spec']==spec and composite['compact_manifest_sha256']==complete['compact_manifest_sha256']
assert composite['no_large_feature_consolidation'] is True and finished['all_four_encoder_processes_exited'] is True
assert composite['actual_actor_optimizer_updates']==composite['actual_head_optimizer_updates']==finished['actual_actor_optimizer_updates']==finished['actual_head_optimizer_updates']==0
worker_shards=[];worker_births=[];proof_hashes=[];encoder_state=None;worker_contract=None
for i in range(4):
 path=features/('worker-%02d'%i);worker=document(path/'worker-complete.json');claim_worker=document(path/'worker-claim.json');producer=document(path/'worker-birth.json')
 assert worker['stage']=='PREFIX_SCALE20_ENCODER_WORKER_COMPLETE' and worker['worker_index']==i and worker['world_size']==4
 assert worker['runtime_commit']==run['source_commit'] and worker['compact_manifest_sha256']==complete['compact_manifest_sha256'] and worker['reused_cache']==reuse
 assert all(worker[k] is False for k in ('reused_features_reencoded','reused_features_copied','actor_updated','probe_trained'))
 contract=worker['contract'];assert contract==claim_worker['contract'] and claim_worker['source_sha256']==h['operator_sha256']['qprobe_prefix_scale20_encode.py']
 assert contract['runtime_commit']==run['source_commit'] and contract['full_native_decoder_layers']==36 and contract['feature_layer']==27 and contract['feature_dim']==2048
 assert contract['optimizer_created'] is False and contract['probe_or_actor_training'] is False and contract['layer27_features_detached'] is True
 assert claim_worker['shard_index']==i and claim_worker['world_size']==4 and claim_worker['compact_manifest_sha256']==complete['compact_manifest_sha256']
 assert claim_worker['reused_cache']==reuse and [x['ordinal'] for x in worker['shards']]==claim_worker['assigned_shard_ordinals']
 assert worker['completed_trajectories']==claim_worker['assigned_trajectories']==sum(x['count'] for x in worker['shards'])
 assert all(x['ordinal']%4==i for x in worker['shards'])
 proof=document(path/'frozen-encoder-proof.json',worker['frozen_encoder_proof_sha256']);proof_hashes.append(worker['frozen_encoder_proof_sha256'])
 assert proof['exact_equal'] is True and proof['before']==proof['after']==document(path/'encoder-before.json')
 assert proof['full_native_decoder_layers']==36 and proof['layer27_hook_only'] is True and proof['optimizer_created'] is False and proof['encoder_updated'] is False and proof['labels_passed_to_encoder'] is False
 assert proof['before']['requires_grad'] is False and proof['before']['model_eval'] is True
 state=proof['before']['loaded_parameter_and_buffer_sha256'];assert re.fullmatch('[0-9a-f]{64}',state)
 if encoder_state is not None:assert encoder_state==state and worker_contract==contract
 encoder_state=state;worker_contract=contract;worker_shards.extend(worker['shards']);worker_births.append(producer['producer'])
 assert producer['gpu']['physical_index']==run['training_physical_indices'][i] and producer['gpu']['physical_gpu_uuid']==run['training_gpu_uuids'][i]
 assert producer['birth_cuda_visible_devices']==run['training_gpu_uuids'][i] and producer['gpu']['actual_cuda_uuids']==[run['training_gpu_uuids'][i]]
assert len(worker_shards)==190 and {x['ordinal'] for x in worker_shards}==set(range(10,200)) and sum(x['count'] for x in worker_shards)==486400
assert sorted(worker_shards,key=lambda x:x['ordinal'])==composite['shards']
for entry in worker_shards:
 i=entry['ordinal'];path=features/('worker-%02d'%(i%4))/('shard-%06d'%i)
 assert entry['path']==str(path) and entry['count']==2560
 value=document(path/'manifest.json',entry['manifest_sha256']);assert value['stage']=='PREFIX_SCALE20_NEW_SHARD_ENCODED'
 assert value['ordinal']==i and value['count']==2560 and value['total_tokens']==entry['total_tokens']
 assert value['compact_shard_manifest_sha256']==new_entries[i]['manifest_sha256'] and value['compact_dataset_manifest_sha256']==complete['compact_manifest_sha256']
 assert value['encoder_loaded_state_sha256']==encoder_state and value['encoder_final_equality_proof_required'] is True
 assert value['full_native_decoder_layers']==36 and value['feature_layer']==27 and value['feature_dim']==2048 and value['feature_dtype']=='bfloat16'
 assert value['no_future_tokens'] is True and value['actual_position_ids'] is True and value['actor_updated'] is False and value['probe_trained'] is False
 assert value['packed_input_ids_dtype']==value['packed_position_ids_dtype']=='int32' and value['input_arrays_hardlinked_without_int64_copy'] is True
 assert value['files']['packed_features.npy']['bytes']==entry['packed_feature_bytes']
 seal_files(path,value['files'])
 source_prefix=document(pathlib.Path(new_entries[i]['path'])/'manifest.json',new_entries[i]['manifest_sha256'])
 for name in ('packed_input_ids.npy','packed_position_ids.npy','packed_policy_mask.npy','offsets.npy','lengths.npy'):
  assert value['files'][name]==source_prefix['files'][name]
  a=file(path/name);c=file(pathlib.Path(new_entries[i]['path'])/name);assert (a.st_dev,a.st_ino)==(c.st_dev,c.st_ino)
phase='actual_owned_resource_release'
status=document(b/'logs/status.json');exit_status=document(b/'logs/exit.json');assert status==exit_status
assert status['stage']=='PREFIX_SCALE20_DATASET_AND_FEATURES_COMPLETE' and status['exit_code']==status['teacher_exit_code']==0
assert status['complete_sha256']==expect['dataset_complete_sha256'] and status['actual_actor_optimizer_updates']==status['actual_head_optimizer_updates']==0
assert len(status['encoder_children'])==4 and {x['worker'] for x in status['encoder_children']}==set(range(4))
births=[launch['process'],status['child']]+worker_births+[x['process'] for x in status['encoder_children']]
teacher_launch=document(b/'teacher-generation-once.json');assert teacher_launch['run_id']==run['run_id'] and teacher_launch['execution_id']==run['execution_id'];births.append(teacher_launch['supervisor'])
ledger=document(b/'logs/process-ledger.json',limit=16*1024**2);assert ledger['run_id']==run['run_id']
leaders=[launch['process']]
for job in ledger['jobs'].values():
 assert job['state']=='STOPPED';leader=job['leader'];assert leader['pid']==leader['sid'];leaders.append(leader);births.append(leader)
 for record in job['ledger'].values():
  assert record['sid']==leader['sid'] and record['uid']==leader['uid'];births.append(record)
known={tuple(x[k] for k in ('pid','start_ticks','sid','uid')) for x in births};sessions={(x['sid'],x['uid']) for x in leaders}
assert all(x[3]==os.getuid() and (x[2],x[3]) in sessions for x in known)
for pid,ticks,sid,uid in known:
 try:current=birth(pid)
 except (FileNotFoundError,ProcessLookupError):continue
 assert current!=dict(pid=pid,start_ticks=ticks,sid=sid,uid=uid), 'Recorded owned birth still present'
survivors=[]
for q in pathlib.Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:
  current=birth(int(q.name));text=(q/'stat').read_text();state=text[text.rfind(')')+2:].split()[0]
 except (FileNotFoundError,ProcessLookupError):continue
 if (current['sid'],current['uid']) in sessions and state not in ('Z','X'):survivors.append(current)
assert not survivors, 'Bound owned UID/SID session has surviving or unknown members'
raw_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
owned_cuda=[];foreign_count=0;apps_exited=0
for row in csv.reader(io.StringIO(raw_apps)):
 if not row:continue
 assert len(row)==2;pid=int(row[1].strip())
 try:current=birth(pid)
 except (FileNotFoundError,ProcessLookupError):apps_exited+=1;continue
 if (current['sid'],current['uid']) in sessions or tuple(current[k] for k in ('pid','start_ticks','sid','uid')) in known:owned_cuda.append(dict(pid=pid,gpu_uuid=row[0].strip()))
 else:foreign_count+=1
assert owned_cuda==[]
return_value=dict(stage='PREFIX_SCALE20_FINAL_METADATA_PRODUCER_PROOF_ACCEPTED',accepted=True,checked_unix=time.time(),
 run_id=run['run_id'],execution_id=run['execution_id'],runtime_source=run['source_commit'],operator_commit=h['operator_commit'],
 dataset_complete_sha256=expect['dataset_complete_sha256'],chunk_manifest_sha256=complete['chunk_manifest_sha256'],
 compact_manifest_sha256=complete['compact_manifest_sha256'],composite_manifest_sha256=finished['composite_manifest_sha256'],
 training_questions=102400,training_trajectories=512000,new_trajectories=486400,reused_training_trajectories=25600,
 fixed_test_questions=512,fixed_test_trajectories=2560,completed_chunks=200,new_feature_shards=190,encoder_workers=4,
 global_max_tokens=spec['global_max_tokens'],total_training_prefix_tokens=complete['total_prefix_tokens'],
 frozen_encoder_state_sha256=encoder_state,frozen_encoder_proof_sha256=proof_hashes,
 sealed_files_stat_checked=sealed_file_count,sealed_bytes_stat_checked=sealed_bytes,metadata_bytes_read=metadata_bytes,
 metadata_manifest_set_sha256=hashlib.sha256(canonical(metadata_hashes)).hexdigest(),
 producer_full_sha_proofs_rechecked=True,feature_full_bytes_rehashed_this_audit=False,npy_or_tensor_opened=False,
 actual_actor_optimizer_updates=0,actual_head_optimizer_updates=0,all_exact_owned_births_absent=True,
 recorded_unique_births=len(known),bound_uid_sid_sessions=len(sessions),owned_session_survivors=[],only_owned_cuda=[],
 foreign_compute_processes_untouched=foreign_count,compute_apps_exited_during_query=apps_exited,read_only=True)
print(json.dumps(return_value,allow_nan=False),flush=True)
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handle',required=True,type=Path)
    parser.add_argument('--finished-receipt',type=Path,default=FINISHED)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();h=json.loads(args.handle.read_bytes())
    assert h['base']==BASE and h['runtime_source']==SOURCE and h['operator_commit']==OPERATOR
    assert sha(CONTROLLER)==h['operator_sha256'][CONTROLLER.name]
    finished=json.loads(args.finished_receipt.read_bytes());assert finished['return_code']==0
    assert finished['handle_sha256']==sha(args.handle)
    observation=finished['observation'];run=observation['native']['run_manifest'];docs=observation['metadata']
    assert run['run_id']==RUN and run['execution_id']==EXECUTION and run['source_commit']==SOURCE
    assert docs['logs/exit.json']['exit_code']==0 and docs['features/complete.json']['all_four_encoder_processes_exited'] is True
    expected=dict(base=BASE,run_id=RUN,execution_id=EXECUTION,runtime_source=SOURCE,operator_commit=OPERATOR,
        dataset_complete_sha256=docs['logs/exit.json']['complete_sha256'],features_complete=docs['features/complete.json'])
    spec=importlib.util.spec_from_file_location('immutable_scale20_final_controller',CONTROLLER)
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    caught="\nexcept BaseException as error:\n import traceback\n print(json.dumps(dict(accepted=False,phase=phase,error_class=type(error).__name__,frames=[dict(file=pathlib.Path(x.filename).name,line=x.lineno,function=x.name) for x in traceback.extract_tb(error.__traceback__)[-6:]],read_only=True)),flush=True)\n raise SystemExit(1)\n"
    code=controller.COMMON+"\nphase='common_source'\ntry:\n"+textwrap.indent(AUDIT,' ')+caught
    command=controller.route().ssh_command(code);assert command[-1].startswith('python3 -c ')
    command[-1]=controller.PYTHON+' -c '+command[-1][len('python3 -c '):]
    receipt=dict(role='scale20_read_only_final_metadata_producer_proof_audit',observer_sha256=sha(__file__),
        original_controller_sha256=sha(CONTROLLER),handle_sha256=sha(args.handle),finished_receipt_sha256=sha(args.finished_receipt),
        expected=expected,started_unix=time.time(),read_only=True,accepted=False)
    try:
        result=subprocess.run(command,input=json.dumps(dict(handle=h,expected=expected)).encode(),capture_output=True,timeout=65)
        receipt.update(return_code=result.returncode,stderr_sha256=hashlib.sha256(result.stderr).hexdigest())
        assert len(result.stdout)<=128*1024
        if result.stdout:receipt['audit']=json.loads(result.stdout)
        receipt['accepted']=result.returncode==0 and receipt.get('audit',{}).get('accepted') is True
    except subprocess.TimeoutExpired:receipt['status']='UNKNOWN_SSH_TIMEOUT'
    receipt['finished_unix']=time.time()
    output=args.output or ROOT/'preparation'/('qprobe-prefix-scale20-final-audit-'+str(time.time_ns())+'.json')
    with output.open('x') as stream:
        json.dump(receipt,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(receipt=str(output),**receipt),allow_nan=False),flush=True)
    if not receipt['accepted']:raise SystemExit(1)


if __name__=='__main__':main()
