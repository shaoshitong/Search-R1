"""Publish/stage/CPU-prepare and run one native-Qwen-block warmup10.

No actor launch, old stop, teacher generation, feature encoding, or trained-head
adoption is exposed. --admission adds one read-only idle inventory to a prepared
prelaunch observation. Completion requires the actual private session and CUDA
contexts to exit after the recorded 200 Adam steps.
"""
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

sys.dont_write_bytecode=True
ROOT=Path('/Users/shaoshitong/Documents/Search-G1')
OPS=Path(__file__).resolve().parents[2]
RUNTIME=ROOT/'qprobe-qwen-lastblock-warmup20'
HUMAN=ROOT/'preparation/qprobe-qwen-lastblock-warmup20-human-intent.json'
SCAFFOLD='reproduction/operations/a800_qprobe_token_binary_headfit_resume.py'
SCAFFOLD_COMMIT='fa2d02128c09829354a7a0e6b3eeb88c091c550c'
SCAFFOLD_SHA='28e2622472080167a9342dd04db3c7341f72ad8b2d5908e5bdf4e313d40777c8'
PRIOR_AUDIT=ROOT/'preparation/qprobe-token-binary-headfit-resume-run-observation2-completion-audit.json'
PRIOR_AUDIT_SHA='bd4b7bb5c9ebbf21bf5e66cf80999a9a4107b364ad84104e573de0cc4b79ba0c'
COMPLETED_HANDLE=ROOT/'preparation/qprobe-token-binary-headfit-resume-deployment.json'
ROLE='a800_native_qwen_lastblock_warmup10_only_intent'
PREFIX='/data/shaoshitong/Search-G1-qprobe-qwen-lastblock-warmup10-'
NATIVE_CLASS='transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'


def load_support():
    raw=(OPS/SCAFFOLD).read_bytes();assert hashlib.sha256(raw).hexdigest()==SCAFFOLD_SHA
    assert subprocess.check_output(['git','-C',str(OPS),'show',SCAFFOLD_COMMIT+':'+SCAFFOLD])==raw
    spec=importlib.util.spec_from_file_location('published_qwen_warmup_transport',OPS/SCAFFOLD)
    scaffold=importlib.util.module_from_spec(spec);spec.loader.exec_module(scaffold)
    return scaffold,scaffold.load_base()


def validate_human(human):
    assert human['role']=='latest_human_qwen_native_lastblock_binary_warmup10_and_formal200_override'
    expected=dict(native_block_required=True,native_block_class=NATIVE_CLASS,manual_transformer_reimplementation_forbidden=True,
        frozen_token_encoder_layer=27,detach_inputs=True,global_max_tokens=244,warmup_groups=5120,warmup_epochs=10,
        head_batch_size=256,expected_actual_head_adam_updates=200,fresh_copied_official_block=True,
        fresh_linear_adam_private_rng=True,old_trained_head_not_inherited=True,reuse_qualified_token_features=True,
        reencode_teacher_or_warmup_features=False,formal_selected=512,formal_exploration=0,all_large_artifacts_remote_only=True,
        public_source_publication_explicitly_authorized=True)
    assert all(human.get(key)==value for key,value in expected.items())
    assert human['official_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    assert human['native_config_verified_official']==dict(hidden_size=2048,num_attention_heads=16,num_key_value_heads=2,
        num_hidden_layers=36,copied_layer_index=35,intermediate_size=11008,rms_norm_eps=1e-6,rope_theta=1000000.0)
    assert human['head_learning_rate']==.0003 and human['head_seed']==1729 and human['runtime_worktree']==str(RUNTIME)


def authority(base,intent):
    human=base.document(HUMAN);validate_human(human)
    assert base.digest(HUMAN)==intent['human_intent_sha256']
    assert intent['role']==ROLE and intent['host']=='baai-a800' and re.fullmatch('[0-9a-f]{32}',intent['deployment_id'])
    assert intent['base']==PREFIX+intent['deployment_id'] and intent['warmup_only'] is True
    assert intent['actor_trainer_launch_authorized'] is False and intent['resume_token_feature_output']==base.PREFIX+'a799f7aff3e94045b64791655b878c31/warmup'
    assert intent['native_block_class']==NATIVE_CLASS and intent['warmup_epochs']==10 and intent['expected_head_adam_updates']==200
    assert base.digest(PRIOR_AUDIT)==PRIOR_AUDIT_SHA and intent['prior_completed_resource_audit_sha256']==PRIOR_AUDIT_SHA


def scope(code):
    old_role="'a800_token_binary_warmup10_only_intent'"
    old_base="'/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-'+intent['deployment_id']"
    assert code.count(old_role) in (1,2) and code.count(old_base) in (1,2)
    return code.replace(old_role,repr(ROLE)).replace(old_base,repr(PREFIX)+"+intent['deployment_id']")


SUPPORT=r'''
assert intent['warmup_only'] is True and intent['actor_trainer_launch_authorized'] is False
assert intent['native_block_class']=='transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'
prior=intent['prior_completed_resource_audit'];assert prior['accepted'] is True and prior['live_pinned_births']==prior['actual_private_session_survivors']==prior['owned_cuda_contexts_remaining']==0
assert sha(Path('/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-494eafd18b004cb980121e46b188db44/warmup/ready.json'))==prior['ready_sha256']=='496e30af7c9c4afb164507f90d14e3f6bfdc7e15e1f6c6f3a4a7c0d5f4198f5f'
def runner(gpu=None):
 args=['/data/search-g1/env/bin/python','-m','reproduction.a800_common.qprobe_qwen_block_warmup','--source-run-base',intent['old_intent']['base'],'--output-dir',str(output),'--source-commit',intent['source_commit'],'--source-publication',str(base/'preparation/source-publication.json'),'--teacher-identity-sha256',intent['teacher_identity_sha256'],'--resume-token-feature-output',intent['resume_token_feature_output']]
 return args+(['--prepare-only'] if gpu is None else ['--physical-index',str(gpu)])
def native_proof(value,*,copied):
 assert value['native_class']=='transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'
 assert type(value['transformers_version']) is str and re.fullmatch(r'[0-9][A-Za-z0-9.+_-]{0,63}',value['transformers_version'])
 assert value['block_index']==35 and value['official_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
 assert value['source_tensors_verified'] is True and value['official_block_copy_verified'] is copied
 assert value['head_initialized'] is copied and value['fresh_linear'] is copied and value['fresh_adam'] is copied
 return value
def prepared_metadata():
 value=document(output/'prepared.json');assert document(base/'logs/prepare-exit.json')['exit_code']==0
 assert value['stage']=='QWEN_LAST_BLOCK_WARMUP10_CPU_PREPARED' and value['task_config']['source_commit']==intent['source_commit']
 assert value['source_map_sha256']==hashlib.sha256(canonical(intent['source_sha256'])).hexdigest()
 native_proof(value['native_block_proof'],copied=False)
 expected=dict(epochs=10,batch_size=256,learning_rate=.0003,auxiliary_weight=.3,target_statistic='sample_std',head_seed=1729,top_k=512,exploration_questions=0,block_index=35,feature_layer=27,global_max_tokens=244,architecture='native_transformers_Qwen2DecoderLayer')
 assert all(value['task_config'][key]==item for key,item in expected.items())
 assert value['no_gpu_model_loaded'] is True and value['encoder_reencoded'] is False and value['trajectories_regenerated'] is False and value['old_head_inherited'] is False and value['head_initialized'] is False and value['formal_training_started'] is False
 implementation=value['native_implementation'];assert implementation['transformers_version']==value['native_block_proof']['transformers_version'] and implementation['attention_implementation']=='sdpa'
 assert implementation['implementations']['decoder_layer']['qualified_name']==intent['native_block_class']
 assert all(re.fullmatch('[0-9a-f]{64}',item['source_sha256']) for item in implementation['implementations'].values())
 assert all(implementation['native_config'][key]==item for key,item in dict(hidden_size=2048,num_attention_heads=16,num_key_value_heads=2,intermediate_size=11008,num_hidden_layers=36,rms_norm_eps=1e-6,rope_theta=1000000.0).items())
 assert value['official_block_source']['source_layer']==35 and value['official_block_source']['tensor_count']==12
 checks=value['cpu_checks'];assert checks['module']=='search_g1.test_qprobe_qwen_block_model' and checks['passed'] is True
 assert checks['tests_run']>0 and checks['failures']==checks['errors']==checks['skipped']==0 and checks['cuda_visible_devices']=='' and checks['cuda_initialized'] is False
 assert sha(output/'cpu-checks.log')==checks['log_sha256']
 return value
def idle_inventory():
 queried=time.time();rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20))))
 apps=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20))))
 inventory=[dict(physical_index=int(row[0]),gpu_uuid=row[1].strip(),used_mib=float(row[2]),util_percent=float(row[3])) for row in rows]
 assert len({row['physical_index'] for row in inventory})==len(inventory) and len({row['gpu_uuid'] for row in inventory})==len(inventory)
 assert all(row['physical_index']>=0 and math.isfinite(row['used_mib']) and row['used_mib']>=0 and math.isfinite(row['util_percent']) and 0<=row['util_percent']<=100 for row in inventory)
 occupied={row[0].strip() for row in apps if row}
 eligible=[row['physical_index'] for row in inventory if row['physical_index'] not in (5,6) and row['used_mib']<=256 and row['util_percent']==0 and row['gpu_uuid'] not in occupied]
 return dict(physical_gpu_inventory=inventory,eligible_idle_nonauxiliary_indices=eligible,queried_unix=queried)
def terminal_helpers():
 library=Path(intent['cleanup_library_repo']);source(library,intent['cleanup_library_source'],intent['cleanup_library_source_sha256'],allow_hydra=True)
 for name in tuple(sys.modules):
  if name in ('reproduction','search_g1') or name.startswith(('reproduction.','search_g1.')):del sys.modules[name]
 sys.path[:]=[str(library)]+[x for x in sys.path if x not in (str(repo),str(library),'')]
 from reproduction.a800.owned_processes import owned_session_survivors,remaining_cuda_contexts
 for name,module in tuple(sys.modules.items()):
  if name in ('reproduction','search_g1') or name.startswith(('reproduction.','search_g1.')):
   path=getattr(module,'__file__',None)
   if path is not None:
    path=Path(path).resolve();assert sha(path)==intent['cleanup_library_source_sha256'][str(path.relative_to(library))]
 return owned_session_survivors,remaining_cuda_contexts
def completion_resource_audit():
 pending=dict(status='pending',accepted=False,metadata_only=True,no_remote_mutation=True,no_tensor_readback=True,nvml_queries=0);queries=0
 if not (output/'ready.json').is_file() or not (base/'logs/run-exit.json').is_file():return dict(pending,reason='ready_and_worker_exit_not_both_recorded')
 try:
  if document(base/'logs/run-exit.json')['exit_code']!=0:return dict(pending,reason='worker_not_successful')
  prepared=prepared_metadata();ready=document(output/'ready.json');launch=document(base/'logs/run-launch.json');claim=document(output/'run-claim.json')
  keys=('pid','start_ticks','sid','uid');worker=launch['process'];producer={key:claim[key] for key in keys}
  assert all(type(row[key]) is int for row in (worker,producer) for key in keys) and worker['pid']==worker['sid']==producer['sid'] and worker['pid']!=producer['pid']
  assert worker['uid']==producer['uid']==os.getuid() and all(row['pid']>0 and row['start_ticks']>0 for row in (worker,producer))
  assert {key:ready[key] for key in keys}==producer and ready['hostname']==claim['hostname']==socket.gethostname()
  assert ready['stage']=='QWEN_LAST_BLOCK_FRESH_HEAD_WARMUP10_COMPLETE' and ready['source_commit']==launch['source_commit']==intent['source_commit']
  assert ready['task_config']==prepared['task_config']==claim['task_config'] and ready['source_map_sha256']==prepared['source_map_sha256']
  assert ready['prepared_sha256']==sha(output/'prepared.json');native_proof(ready['native_block_proof'],copied=True)
  assert all(ready['native_block_proof'][key]==prepared['native_block_proof'][key] for key in ('native_class','transformers_version','block_index','official_revision'))
  assert ready['official_block_source']==prepared['official_block_source'] and ready['native_implementation']==prepared['native_implementation']
  assert ready['cpu_checks']==prepared['cpu_checks']
  copied=ready['initial_block_proof'];official=prepared['official_block_source'];assert copied['source_layer']==official['source_layer']==35 and copied['tensor_count']==official['tensor_count']==12
  assert copied['all_copied_parameters_trainable'] is True and set(copied['tensors'])==set(official['tensors']) and ready['initial_readout_proof']['fresh'] is True
  for key,value in copied['tensors'].items():
   assert value['exact_cast_equal'] is True and re.fullmatch('[0-9a-f]{64}',value['copied_fp32_sha256'])
   assert all(value[field]==official['tensors'][key][field] for field in ('source_key','shape','source_dtype','source_sha256'))
  assert ready['block_index']==35 and ready['head_optimizer_updates']==ready['head_adam_steps']==200 and ready['epoch_records']==11 and ready['minibatch_records']==200
  assert ready['formal_training_started'] is False and ready['automatic_formal_200_launch'] is False and ready['old_head_inherited'] is False and ready['no_reencode'] is True and ready['encoder_reencoded'] is False and ready['trajectories_generated']==0
  assert ready['physical_index']==launch['physical_index'] and ready['physical_index'] not in (5,6) and ready['physical_gpu_uuid']==launch['gpu_uuid']
  metrics=document(output/'warmup-metrics.json',2*1024**2);metrics_sha=sha(output/'warmup-metrics.json',2*1024**2)
  assert ready['metrics_sha256']==metrics_sha and metrics['role']=='qwen_last_block_question_head_training_warmup' and metrics['training_groups']==5120 and metrics['fit_epochs']==10 and metrics['actual_head_adam_updates']==metrics['expected_head_adam_updates']==200
  assert metrics['actor_updated'] is False and metrics['reference_updated'] is False and metrics['formal_training_started'] is False
  epochs=metrics['epoch_history'];minis=metrics['minibatch_history'];assert len(epochs)==11 and len(minis)==200
  assert all(row['epoch']==epoch and row['actual_head_adam_updates']==epoch*20 for epoch,row in enumerate(epochs))
  assert all(row['head_update_ordinal']==index+1 and row['head_adam_steps_before']==index and row['head_adam_steps_after']==index+1 and row['epoch']==index//20+1 and row['minibatch_index']==index%20 and row['batch_size']==256 for index,row in enumerate(minis))
  assert all(type(row.get('binary_accuracy')) in (int,float) and math.isfinite(row['binary_accuracy']) and 0<=row['binary_accuracy']<=1 for row in minis)
  assert metrics['training_summary']==epochs[-1] and not (output/'run-failure.json').exists()
  survivors,contexts=terminal_helpers();worker_state=process(worker);producer_state=process(producer);live,reused_sessions=survivors([worker])
  queried=time.time();queries=1;apps=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20))))
  remaining,reused_pids=contexts([worker,producer],apps);accepted=not worker_state['alive'] and not producer_state['alive'] and not live and not remaining
  return dict(status='accepted' if accepted else 'pending',accepted=accepted,metadata_only=True,no_remote_mutation=True,no_tensor_readback=True,worker=worker,runner=producer,
   live_pinned_births=int(worker_state['alive'])+int(producer_state['alive']),actual_private_session_survivors=len(live),owned_cuda_contexts_remaining=len(remaining),
   reused_session_count=len(reused_sessions),reused_pid_count=len(reused_pids),head_adam_steps=200,epoch_records=11,minibatch_records=200,
   source_commit=intent['source_commit'],ready_sha256=sha(output/'ready.json'),metrics_sha256=metrics_sha,nvml_queries=1,nvml_query_started_unix=queried,checked_unix=time.time())
 except (AssertionError,KeyError,TypeError,ValueError,OSError,subprocess.SubprocessError) as failure:
  return dict(pending,reason='terminal_metadata_or_resources_not_qualified',failure_type=type(failure).__name__,nvml_queries=queries,checked_unix=time.time())
'''

RUN=r'''
prepared=prepared_metadata();review=p['ready'];assert review['role']=='qwen_native_lastblock_warmup10_reviewed_prepared_ready'
assert review['source_commit']==intent['source_commit'] and review['source_map_sha256']==prepared['source_map_sha256']
assert review['human_intent_sha256']==intent['human_intent_sha256'] and review['source_publication_sha256']==intent['source_publication_sha256']
assert review['independent_code_review_approved'] is True and review['actual_deployment_review_approved'] is True and review['authorized_source_publication_approved'] is True
assert review['remote_prepared_sha256']==sha(output/'prepared.json') and review['native_block_proof']==prepared['native_block_proof']
assert not (base/'logs/run-launch.json').exists() and not (output/'run-claim.json').exists()
index=p['physical_index'];assert type(index) is int and index>=0 and index not in (5,6)
actual=idle_inventory();assert index in actual['eligible_idle_nonauxiliary_indices']
gpu=next(row['gpu_uuid'] for row in actual['physical_gpu_inventory'] if row['physical_index']==index)
once(base/'logs/run-once.json',dict(source_commit=intent['source_commit'],physical_index=index,gpu_uuid=gpu,reviewed_prepared_sha256=review['remote_prepared_sha256'],no_automatic_retry=True,claimed_unix=time.time()))
with (base/'logs/run.log').open('xb') as log:
 child=subprocess.Popen(['/data/search-g1/env/bin/python','-c',p['worker'],json.dumps(dict(base=str(base),repo=str(repo),action='run',args=runner(index)))],cwd=repo,env=environment(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record=dict(stage='ONE_NATIVE_QWEN_LASTBLOCK_WARMUP10_LAUNCHED',process=birth(child.pid),physical_index=index,gpu_uuid=gpu,source_commit=intent['source_commit'],actor_trainer_started=False,formal_training_started=False,expected_head_adam_updates=200,launched_unix=time.time())
 once(base/'logs/run-launch.json',record)
emit(record)
'''


def program(scaffold,base,action):
    if action=='stage':return scope(scaffold.program(base,'stage'))
    common=scope(base.COMMON)+'\nbind(json.load(sys.stdin))\n'+SUPPORT
    if action=='run-once':return common+'\n'+RUN
    original=base.PREPARE if action=='prepare' else base.OBSERVE
    assert original.startswith(base.COMMON);body=original[len(base.COMMON):]
    assert body.count('bind(json.load(sys.stdin));')==1;body=body.replace('bind(json.load(sys.stdin));','',1)
    if action=='observe':
        assert body.count('value[:256]')==1;body=body.replace('value[:256]','value[:512]',1)
        assert body.count("'warmup/warmup-metrics.json'")==1;body=body.replace("'warmup/warmup-metrics.json'","'warmup/partial-head.json','warmup/warmup-metrics.json'",1)
        addition=r'''
path=output/'progress.jsonl'
if path.exists():
 assert path.resolve()==path and not path.is_symlink() and path.stat().st_uid==os.getuid()
 info=path.stat();start=max(0,info.st_size-256*1024)
 with path.open('rb') as stream:stream.seek(start);raw=stream.read(256*1024)
 if start:raw=raw[raw.find(b'\n')+1:] if b'\n' in raw else b''
 complete=raw[:raw.rfind(b'\n')+1];events=[json.loads(line) for line in complete.splitlines()]
 result['progress_stream']=dict(file_bytes=info.st_size,scan_start_byte=start,partial_tail_bytes=len(raw)-len(complete),complete_tail_rows=len(events),latest=scalars(events[-1]) if events else None,recent=[scalars(value) for value in events[-16:]])
if p.get('admission'):
 prepared_metadata();assert not (base/'logs/run-once.json').exists();result['prelaunch_idle_admission']=idle_inventory()
result['completion_resource_audit']=completion_resource_audit()
emit(result)
'''
        assert body.count('emit(result)')==1;body=body.replace('emit(result)',addition,1)
    return common+'\n'+body


def create(scaffold,base,args):
    human=base.document(HUMAN);validate_human(human);repo=args.source_repo.resolve();commit=args.source_commit
    assert re.fullmatch('[0-9a-f]{40}',commit) and subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==commit
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip()
    module=base.runtime_module();mapping=module.published_source(repo,commit);publication=base.document(args.source_publication)
    assert 'reproduction/a800_common/qprobe_qwen_block_warmup.py' in mapping
    assert all(base.digest(repo/name)==value for name,value in mapping.items())
    assert publication['source_commit']==publication['remote_commit']==commit and publication['remote_fresh_object_store_disk_git_hash_equal'] is True
    assert publication['source_map_sha256']==hashlib.sha256(base.canonical(mapping)).hexdigest()
    assert re.fullmatch('refs/heads/[A-Za-z0-9_./-]+',publication['published_ref'])
    bundle=args.bundle.resolve();subprocess.run(['git','-C',str(repo),'bundle','verify',str(bundle)],capture_output=True,check=True)
    assert subprocess.check_output(['git','-C',str(repo),'bundle','list-heads',str(bundle)],text=True).split()[0]==commit
    assert base.digest(PRIOR_AUDIT)==PRIOR_AUDIT_SHA;prior=base.document(PRIOR_AUDIT)['observation']['completion_resource_audit'];assert prior['accepted'] is True
    assert base.digest(scaffold.PARENT_HANDLE)==scaffold.PARENT_HANDLE_SHA
    old=base.document(COMPLETED_HANDLE);identifier=uuid.uuid4().hex
    intent=dict(role=ROLE,host='baai-a800',deployment_id=identifier,base=PREFIX+identifier,created_unix=time.time(),
        source_commit=commit,source_sha256=mapping,published_ref=publication['published_ref'],source_repo=str(repo),
        source_publication=str(args.source_publication.resolve()),source_publication_bytes=args.source_publication.stat().st_size,source_publication_sha256=base.digest(args.source_publication),
        bundle=str(bundle),bundle_bytes=bundle.stat().st_size,bundle_sha256=base.digest(bundle),human_intent_sha256=base.digest(HUMAN),
        old_intent=old['old_intent'],parent_intent=base.document(scaffold.PARENT_HANDLE),headfit_resume_only=True,resume_token_feature_output=scaffold.PARENT_BASE+'/warmup',
        prior_completed_resource_audit=prior,prior_completed_resource_audit_sha256=PRIOR_AUDIT_SHA,
        cleanup_library_repo=old['cleanup_library_repo'],cleanup_library_source=old['cleanup_library_source'],cleanup_library_source_sha256=old['cleanup_library_source_sha256'],
        teacher_identity_sha256=old['teacher_identity_sha256'],warmup_only=True,actor_trainer_launch_authorized=False,
        native_block_class=NATIVE_CLASS,native_config_verified_official=human['native_config_verified_official'],warmup_epochs=10,expected_head_adam_updates=200,
        no_automatic_retry=True,all_large_artifacts_remote_only=True)
    authority(base,intent);handle=args.handle.resolve() if args.handle else ROOT/'preparation'/('qprobe-qwen-lastblock-warmup10-'+identifier+'.json')
    base.once(handle,intent);print(json.dumps(dict(handle=str(handle),base=intent['base'],source_commit=commit)))


def perform(scaffold,base,args):
    commit=subprocess.check_output(['git','-C',str(OPS),'rev-parse','HEAD'],text=True).strip();relative=str(Path(__file__).resolve().relative_to(OPS))
    assert subprocess.check_output(['git','-C',str(OPS),'show',commit+':'+relative])==Path(__file__).read_bytes(), 'Publish exact operator first'
    handle=args.handle.resolve();intent=base.document(handle);authority(base,intent)
    payload=dict(intent=intent,operator_commit=commit,worker=base.WORKER,admission=args.admission)
    if args.action=='run-once':
        assert args.physical_index is not None and args.ready is not None;payload.update(physical_index=args.physical_index,ready=base.document(args.ready))
    code=program(scaffold,base,args.action);ast.parse(code)
    if args.action!='observe':base.once(handle.with_name(handle.stem+'.'+args.action+'.claim.json'),dict(action=args.action,operator_commit=commit,source_commit=intent['source_commit'],claimed_unix=time.time(),no_automatic_retry=True))
    if args.action=='stage':
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w') as archive:
            raw=base.canonical(payload);item=tarfile.TarInfo('request.json');item.size=len(raw);archive.addfile(item,io.BytesIO(raw))
            for path,name,expected in ((Path(intent['bundle']),'preparation/published-source.bundle',intent['bundle_sha256']),(Path(intent['source_publication']),'preparation/source-publication.json',intent['source_publication_sha256'])):
                assert base.digest(path)==expected
                with path.open('rb') as source:item=tarfile.TarInfo(name);item.size=path.stat().st_size;archive.addfile(item,source)
        data=stream.getvalue()
    else:data=base.canonical(payload)
    route=base.runtime_module().ssh_command(code);assert route[-1].startswith('python3 -c ');route[-1]=base.PYTHON+' -c '+route[-1][len('python3 -c '):]
    receipt=args.receipt.resolve() if args.receipt else handle.with_name(handle.stem+'.'+args.action+'-'+str(time.time_ns())+'.json')
    result=dict(action=args.action,operator_commit=commit,source_commit=intent['source_commit'],base=intent['base'],no_automatic_retry=True,observation=None)
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        try:
            done=subprocess.run(route,input=data,stdout=out,stderr=err,timeout=400 if args.action=='stage' else 120);result['exit_code']=done.returncode
        except (OSError,subprocess.TimeoutExpired) as failure:result.update(exit_code=None,failure_type=type(failure).__name__,status='TRANSPORT_UNKNOWN_OBSERVE_SAME_HANDLE')
        result.update(stdout_bytes=out.tell(),stderr_bytes=err.tell());out.seek(0);raw=out.read(512*1024+1);err.seek(0);errors=err.read(256*1024)
        result['bounded_stderr_sha256']=hashlib.sha256(errors).hexdigest();result['remote_exception_types']=sorted(set(value.decode() for value in re.findall(rb'(?m)^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))(?::|$)',errors)))
        result['remote_frames']=[dict(file=Path(name.decode(errors='replace')).name,line=int(line)) for name,line in re.findall(rb'File "([^"]+)", line ([0-9]+)',errors)[-8:]]
    if len(raw)<=512*1024:
        rows=[json.loads(row.split(':',1)[1]) for row in raw.decode(errors='strict').splitlines() if row.startswith('TOKEN_WARMUP_JSON:')];assert len(rows)<=1
        if rows:result['observation']=rows[0]
    result['checked_unix']=time.time();base.once(receipt,result);print(json.dumps(dict(receipt=str(receipt),**result),allow_nan=False))
    return 0 if result['exit_code']==0 and result['observation'] is not None else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('create','stage','prepare','run-once','observe'))
    parser.add_argument('--handle',type=Path);parser.add_argument('--receipt',type=Path);parser.add_argument('--source-repo',type=Path,default=RUNTIME)
    parser.add_argument('--source-commit');parser.add_argument('--source-publication',type=Path);parser.add_argument('--bundle',type=Path)
    parser.add_argument('--ready',type=Path);parser.add_argument('--physical-index',type=int);parser.add_argument('--admission',action='store_true')
    args=parser.parse_args();assert not args.admission or args.action=='observe';scaffold,base=load_support()
    if args.action=='create':
        assert all(value is not None for value in (args.source_commit,args.source_publication,args.bundle));return create(scaffold,base,args)
    assert args.handle is not None;return perform(scaffold,base,args)


if __name__=='__main__':raise SystemExit(main())
