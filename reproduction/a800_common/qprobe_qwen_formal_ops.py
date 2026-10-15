"""Source-bound, one-shot preparation and launch of fresh native-Qwen formal200.

CPU preparation, one real missing-row encoder, native-head adoption, initial0
reward-probe adoption, release qualification, and actor launch have separate
durable claims. No action retries a failed or uncertain execution.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode=True
ROLE='a800_qwen_native_formal_prospective_intent'
PREFIX='/data/shaoshitong/Search-G1-a800-fresh200-'
SPEC_SHA='fcc9d61db156ddbb4e96dcd6c9437797b978dbbfc9774145a28e0c761de1a9e7'
INITIAL0='/data/shaoshitong/Search-G1-a800-fresh200-7895330a2c584bba9940720e98f006ef'
QUALIFIED_WARMUP_BASE='/data/shaoshitong/Search-G1-qprobe-qwen-lastblock-warmup10-096c8297cba74f8089063e763686671e'
QUALIFIED_WARMUP_SOURCE='e81445905bf8b15dbb0bdec3afa29d99ab0b11d8'
QUALIFIED_WARMUP_MAP='006912cc7817b2ae8abde32a85c42d21b2c1976bf0b5618066c6aa9d2f0674ad'
QUALIFIED_BANK_SOURCE='927163041cf0ddf2c2c517efc466177d76dc2a18'
QUALIFIED_BANK_MAP='7ce4d02996393270a47e1d3840ab5ee793205e17e3154de18193c19b4120851e'
QUALIFIED_BANK_ORIGIN=PREFIX+'92bb58c18ae74ca9bc59fe1ba5dc02b6'
QUALIFIED_BANK_BASE='/data/shaoshitong/Search-G1-qprobe-fulltoken-feature-bank-8be88f0d4ac1470f95fe13b4fe146a3f'
QUALIFIED_BANK_RUN='862bccb5fcf348ce8454f9a332f37bdd'
QUALIFIED_BANK_EXECUTION='027e65da351a4e018dd377b806c19434'
QUALIFIED_BANK_COMPLETE='e27f0afa69c31a9248977a78fe2093bfd1049a324fc3c860bc963c75a84b68e9'
QUALIFIED_BANK_MANIFEST='fa8c9513f89b3700851f8fe75665bf515a82e7f491bc3a52a643a6e82aa243e9'
QUALIFIED_BANK_ADMIT='efaa3a8bc9912cdc5a51884f5e626645ffe565980d3145edffad2bf7f241bdf1'
QUALIFIED_BANK_TERMINAL_AUDIT='214270332335330c32851b8db31e5e4d14d70cd734633837a16bf4adc13c2191'
QUALIFIED_BANK_PRODUCERS={
    'reproduction/a800_common/qprobe_token_feature_bank.py':'4b03e3e980a5e79516804ec4e263942759a3ea048e6ea34071eaa1fcad52cb35',
    'search_g1/qprobe_token_features.py':'408ac3c710cbcbb3684c93bdf57fd170c54266038fb6e0aced581accf1e5769b'}
BANK_FILES=('manifest.json','index.jsonl','offsets.npy','lengths.npy','packed_features.npy')
PYTHON='/data/search-g1/env/bin/python'
CPU_MODULES=(
    'search_g1.test_qprobe_qwen_block_online',
    'reproduction.common8.test_native_qprobe_configuration',
    'reproduction.common8.test_qprobe_configuration',
    'reproduction.a800_common.test_qprobe_native_initial_adoption',
    'search_g1.test_qprobe_saved_native_checkpoint',
    'search_g1.test_qprobe_driver_gpu',
)
CPU_PROGRAM=r'''
import hashlib,json,os,sys,time,traceback,unittest
from pathlib import Path
sys.dont_write_bytecode=True
target=Path(sys.argv[1]);expected=json.loads(sys.argv[2]);repo=Path(sys.argv[3])
proof=dict(expected,stage='NATIVE_FORMAL_CPU_COMPATIBILITY_CHECKS',tests_run=0,
    failures=0,errors=0,skips=0,passed=False,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    cuda_initialized_before=None,cuda_initialized_after=None,cuda_visible_devices_after=None)
try:
    assert proof['cuda_visible_devices']==''
    for name,digest in expected['test_source_sha256'].items():
        assert hashlib.sha256((repo/(name.replace('.','/')+'.py')).read_bytes()).hexdigest()==digest
    import torch,transformers,hydra
    proof.update(torch_version=torch.__version__,transformers_version=transformers.__version__,hydra_version=hydra.__version__)
    proof['cuda_initialized_before']=torch.cuda.is_initialized()
    assert proof['cuda_initialized_before'] is False
    suite=unittest.defaultTestLoader.loadTestsFromNames(expected['modules'])
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
    proof.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped))
    proof['cuda_initialized_after']=torch.cuda.is_initialized()
    proof['cuda_visible_devices_after']=os.environ.get('CUDA_VISIBLE_DEVICES')
    proof['passed']=result.wasSuccessful() and result.testsRun>0 and not result.skipped and proof['cuda_initialized_after'] is False and proof['cuda_visible_devices_after']==''
except BaseException as failure:
    traceback.print_exc();proof['check_failure_type']=type(failure).__name__
finally:
    proof['checked_unix']=time.time()
    raw=json.dumps(proof,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fd=os.open(target.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
sys.exit(0 if proof['passed'] else 1)
'''


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def owned(path):
    path=Path(path);assert path.resolve()==path and not path.is_symlink() and path.stat().st_uid==os.getuid()
    return path


def read(path,limit=16*1024**2):
    path=owned(path);assert path.is_file() and path.stat().st_size<=limit
    return json.loads(path.read_bytes())


def once(path,value):
    raw=canonical(value)+b'\n';fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def birth(pid):
    path=Path('/proc')/str(pid);raw=(path/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
    return dict(pid=pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=path.stat().st_uid)


def alive(record):
    try:
        actual=birth(record['pid']);raw=(Path('/proc')/str(record['pid'])/'stat').read_text();state=raw[raw.rfind(')')+2:].split()[0]
        return actual=={key:record[key] for key in ('pid','start_ticks','sid','uid')} and state not in ('Z','X')
    except (FileNotFoundError,ProcessLookupError):return False


def validate_qualified_bank_origin(origin,source_sha256):
    expected=dict(role='native_qwen_preupdate_qualified_bank_origin_v1',base=QUALIFIED_BANK_ORIGIN,
        source_commit=QUALIFIED_BANK_SOURCE,source_map_sha256=QUALIFIED_BANK_MAP,run_id=QUALIFIED_BANK_RUN,
        execution_id=QUALIFIED_BANK_EXECUTION,bank_base=QUALIFIED_BANK_BASE,complete_sha256=QUALIFIED_BANK_COMPLETE,
        feature_manifest_sha256=QUALIFIED_BANK_MANIFEST,admitted_sha256=QUALIFIED_BANK_ADMIT,
        terminal_audit_sha256=QUALIFIED_BANK_TERMINAL_AUDIT,producer_source_sha256=QUALIFIED_BANK_PRODUCERS,
        formal_completed_outers=0,actor_or_head_updates_inherited=False)
    assert all(origin.get(key)==value for key,value in expected.items())
    assert origin['formal_completed_outers'] is not False and origin['actor_or_head_updates_inherited'] is False
    assert all(source_sha256.get(name)==digest for name,digest in QUALIFIED_BANK_PRODUCERS.items())
    for key in ('old_handle_sha256','old_intent_canonical_sha256','bank_admitted_observation_sha256'):
        assert re.fullmatch('[0-9a-f]{64}',origin[key])
    assert type(origin['terminal_audit_bytes']) is int and 0<origin['terminal_audit_bytes']<=2*1024**2
    admitted=origin['admitted']
    assert admitted['stage']=='FULL_BANK_SHA_AND_ENCODER_RESOURCE_RELEASE_ADMITTED' and admitted['complete_sha256']==QUALIFIED_BANK_COMPLETE
    assert admitted['cache_dir']==QUALIFIED_BANK_BASE+'/token-features' and admitted['feature_manifest_sha256']==QUALIFIED_BANK_MANIFEST
    assert admitted['feature_spec_sha256']==SPEC_SHA and admitted['owned_sessions_remaining']==admitted['owned_cuda_contexts_remaining']==0
    assert len(admitted['files'])==5 and {row['path'] for row in admitted['files']}=={admitted['cache_dir']+'/'+name for name in BANK_FILES}
    for row in admitted['files']:
        assert all(type(row[key]) is int and row[key]>=0 for key in ('device','inode','bytes','mtime_ns'))


def validate_terminal_bank_origin(origin,audit):
    assert audit['role']=='native_qwen92bb_actual_terminal_failure_and_onlyowned_resource_audit'
    assert all(audit[key]==origin[key] for key in ('source_commit','run_id','execution_id'))
    assert audit['terminal_owned_cleanup_accepted'] is True and audit['owned_birth_records']==34
    assert audit['live_owned_session_processes']==audit['owned_cuda_contexts_remaining']==audit['all_project_trainers_live']==[]
    assert audit['one_nvml_read_only_query'] is True and audit['no_tensor_or_gpu_test'] is True and audit['no_stop_or_restart'] is True
    metadata=audit['metadata_files'];cleanup=metadata['logs/cleanup-status.json']['value']
    assert cleanup['cleaned'] is True and cleanup['execution_backend']=='ssh_local' and cleanup['slurm_job_id'] is None
    assert all(cleanup[key]==origin[key] for key in ('run_id','execution_id'))
    assert metadata['logs/pipeline-status.json']['value']['stage']=='FAILED'
    assert all(job['state']=='STOPPED' for job in metadata['logs/process-ledger.json']['value']['jobs'].values())


def check_bank_file_seal(value):
    assert len(value['files'])==5 and {row['path'] for row in value['files']}=={value['cache_dir']+'/'+name for name in BANK_FILES}
    for row in value['files']:
        st=owned(row['path']).stat()
        assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(row['device'],row['inode'],row['bytes'],row['mtime_ns'])


def require_recorded_sessions_gone(records,survivors):
    records=list(records)
    leaders={tuple(record[key] for key in ('pid','start_ticks','sid','uid')):record
        for record in records if record['pid']==record['sid']}
    known={(record['sid'],record['uid']) for record in leaders.values()}
    assert leaders and all((record['sid'],record['uid']) in known for record in records), 'Recorded child has no bound private-session leader'
    assert all(not alive(record) for record in records)
    live,_=survivors(list(leaders.values()));assert not live


def validate_intent(intent):
    assert intent['role']==ROLE and intent['host']=='baai-a800' and re.fullmatch('[0-9a-f]{32}',intent['deployment_id'])
    assert intent['base']==PREFIX+intent['deployment_id'] and re.fullmatch('[0-9a-f]{40}',intent['source_commit'])
    assert re.fullmatch('/data/shaoshitong/Search-G1-qprobe-fulltoken-feature-bank-[0-9a-f]{32}',intent['bank_base'])
    assert intent['native_formal_200_authorized'] is True and intent['target_updates']==200
    assert re.fullmatch('/data/shaoshitong/Search-G1-qprobe-qwen-lastblock-warmup10-[0-9a-f]{32}',intent['warmup_base'])
    assert re.fullmatch('[0-9a-f]{40}',intent['warmup_source_commit']) and re.fullmatch('[0-9a-f]{64}',intent['warmup_source_map_sha256'])
    assert (intent['warmup_base'],intent['warmup_source_commit'],intent['warmup_source_map_sha256'])==(QUALIFIED_WARMUP_BASE,QUALIFIED_WARMUP_SOURCE,QUALIFIED_WARMUP_MAP)
    assert intent['warmup_handle']['base']==intent['warmup_base'] and intent['warmup_handle']['source_commit']==intent['warmup_source_commit']
    assert hashlib.sha256(canonical(intent['warmup_handle']['source_sha256'])).hexdigest()==intent['warmup_source_map_sha256']
    assert intent['initial_probes_source_base']==INITIAL0 and intent['head_device']=='cuda:0'
    assert intent['source_publication_authorized'] is True and intent.get('saved_checkpoint_restore') is None
    if intent.get('qualified_bank_origin') is not None:
        origin=intent['qualified_bank_origin'];validate_qualified_bank_origin(origin,intent['source_sha256'])
        assert intent['bank_base']==origin['bank_base'] and intent['source_commit']!=origin['source_commit'] and intent['base']!=origin['base']


def load(base):
    base=owned(Path(base));intent=read(base/'preparation/intent.json');validate_intent(intent)
    assert base==Path(intent['base']);repo=base/'repo';assert Path(__file__).resolve().parents[2]==repo
    from reproduction.a800_common.deploy import published_source
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==intent['source_commit']
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip()
    assert published_source(repo,intent['source_commit'])==intent['source_sha256']
    for name,digest in intent['source_sha256'].items():assert sha(repo/name)==digest
    from reproduction.common8.source_files import runtime_files
    for path in runtime_files(repo):
        name=str(path.relative_to(repo))
        assert name in intent['source_sha256'] or re.fullmatch(r'outputs/\d{4}-\d{2}-\d{2}/\d{2}-\d{2}-\d{2}/\.hydra/(config|hydra|overrides)\.yaml',name)
    assert sha(base/'preparation/source-publication.json')==intent['source_publication_sha256']
    publication=read(base/'preparation/source-publication.json')
    assert publication['source_commit']==publication['remote_commit']==intent['source_commit'] and publication['remote_fresh_object_store_disk_git_hash_equal'] is True
    return base,repo,intent


def environment(base,repo,gpu=''):
    result=dict(os.environ)
    for key in tuple(result):
        if key.startswith('SLURM_') or key in ('PYTHONHOME','RAY_ADDRESS','LD_LIBRARY_PATH','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):result.pop(key,None)
    result.update(SEARCH_G1_BASE=str(base),SEARCH_G1_PROFILE='common8',SEARCH_G1_TRAINING_WORLD_SIZE='4',CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(repo),PATH='/data/search-g1/env/bin:'+result.get('PATH','/usr/bin:/bin'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8')
    for key,name in dict(TMPDIR='tmp',HF_HOME='cache/hf',XDG_CACHE_HOME='cache',TRITON_CACHE_DIR='cache/triton',TORCHINDUCTOR_CACHE_DIR='cache/inductor').items():result[key]=str(base/name)
    return result


def unlaunched(base):
    for name in ('pipeline-launch-claim.json','training-launch-claim.json','process-ledger.json','actual-attempts.jsonl','actual-updates.jsonl','deployment-launch-receipt.json'):
        assert not (base/'logs'/name).exists(), 'Existing actor execution must be observed, never retried'


def command(base,repo,label,args,*,timeout=None):
    with (base/'logs'/(label+'.log')).open('xb') as log:
        result=subprocess.run(args,cwd=repo,env=environment(base,repo),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        log.flush();os.fsync(log.fileno())
    assert result.returncode==0, label+' failed; preserve this claim'


def cpu_expectation(repo,intent):
    test_source={name:intent['source_sha256'][name.replace('.','/')+'.py'] for name in CPU_MODULES}
    return dict(source_commit=intent['source_commit'],source_map_sha256=hashlib.sha256(canonical(intent['source_sha256'])).hexdigest(),modules=list(CPU_MODULES),test_source_sha256=test_source,runner_source_sha256=intent['source_sha256']['reproduction/a800_common/qprobe_qwen_formal_ops.py'])


def validate_cpu_checks(proof,expected):
    assert proof['stage']=='NATIVE_FORMAL_CPU_COMPATIBILITY_CHECKS'
    assert all(proof.get(key)==value for key,value in expected.items())
    assert proof['passed'] is True and type(proof['tests_run']) is int and proof['tests_run']>0
    assert proof['failures']==proof['errors']==proof['skips']==0
    assert proof['cuda_visible_devices']==proof['cuda_visible_devices_after']=='' and proof['cuda_initialized_before'] is proof['cuda_initialized_after'] is False
    assert 'check_failure_type' not in proof


def cpu_checks(base,repo,intent):
    expected=cpu_expectation(repo,intent);path=base/'logs/native-formal-cpu-checks.json';log_path=base/'logs/native-formal-cpu-checks.log'
    # Synthetic manifests own their test topology; never inherit production's
    # four-GPU override into this CPU-only child. Production commands retain it.
    production_env=environment(base,repo)
    assert production_env['SEARCH_G1_TRAINING_WORLD_SIZE']=='4'
    fixture_env=dict(production_env)
    fixture_env.pop('SEARCH_G1_TRAINING_WORLD_SIZE',None)
    with log_path.open('xb') as log:
        result=subprocess.run([PYTHON,'-c',CPU_PROGRAM,str(path),canonical(expected).decode(),str(repo)],cwd=repo,env=fixture_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        log.flush();os.fsync(log.fileno())
    assert production_env['SEARCH_G1_TRAINING_WORLD_SIZE']=='4'
    proof=read(path);validate_cpu_checks(proof,expected);assert result.returncode==0
    return dict(receipt_sha256=sha(path),log_sha256=sha(log_path),**proof)


def required_cpu_checks(base,repo,intent):
    prepared=read(base/'logs/native-prepared.json');proof=read(base/'logs/native-formal-cpu-checks.json')
    validate_cpu_checks(proof,cpu_expectation(repo,intent))
    assert prepared['cpu_checks']['receipt_sha256']==sha(base/'logs/native-formal-cpu-checks.json') and prepared['cpu_checks']['log_sha256']==sha(base/'logs/native-formal-cpu-checks.log')
    assert all(prepared['cpu_checks'].get(key)==value for key,value in proof.items())
    return prepared['cpu_checks']


def prepare(base,repo,intent):
    unlaunched(base);compatibility=cpu_checks(base,repo,intent);baseline=owned(intent['baseline_base']);original=baseline/'preparation/deployment-intent.json'
    assert sha(original)==intent['baseline_intent_sha256'];old=read(original)
    assert old['source_commit']==intent['baseline_commit'] and old['base']==str(baseline)
    allowed={'data/'+name for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json')}
    allowed|={'preparation/official-model-manifest-original.json','preparation/formal-inline-method-template.json'}
    rows=[{key:row[key] for key in ('target','bytes','sha256')} for row in old['inputs'] if row['target'] in allowed]
    assert len(rows)==len(allowed) and {row['target'] for row in rows}==allowed
    for directory in ('data','ray','artifacts','checkpoints'):(base/directory).mkdir(parents=True,exist_ok=True)
    for row in rows:
        origin=owned(baseline/row['target']);assert origin.stat().st_size==row['bytes'] and sha(origin)==row['sha256']
        target=base/row['target'];assert not target.exists()
        with origin.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2);dst.flush();os.fsync(dst.fileno())
        assert sha(target)==row['sha256'] and sha(origin)==row['sha256']
    deployment=dict(schema_version=1,role='fresh_a800_common_deployment_intent',deployment_id=intent['deployment_id'],created_unix=intent['created_unix'],host='baai-a800',base=str(base),source_commit=intent['source_commit'],source_sha256=intent['source_sha256'],inputs=rows,training_world_size=4,training_physical_indices=None,gpu_placement_policy='launch_time_four_idle_excluding_auxiliary_v1',gpu_placement_bound=False,auxiliary_physical_indices=[5,6],target_updates=200,same_process_after_five=True,initial_probes_source_base=INITIAL0)
    once(base/'preparation/deployment-intent.json',deployment)
    once(base/'logs/deployment-input-receipt.json',dict(deployment_id=intent['deployment_id'],inputs=rows,exact_original_remote_bytes=True,controller_data_or_model_bytes_transferred=False))
    once(base/'logs/deployment-source-receipt.json',dict(deployment_id=intent['deployment_id'],source_commit=intent['source_commit'],source_sha256=intent['source_sha256'],source_transport='published_git_bundle_relay',source_is_exact_published_commit=True,uncommitted_source_copied=False))
    from reproduction.common8.policy_calibration_plan import document_sha256
    method=read(base/'preparation/formal-inline-method-template.json');method.update(source_commit=intent['source_commit'],source_map_sha256=document_sha256(intent['source_sha256']),created_unix=time.time())
    once(base/'preparation/formal-inline-method.json',method)
    command(base,repo,'native-prepare-run',[PYTHON,'-m','reproduction.a800_common.prepare_run','--base',str(base),'--model','/data/search-g1/assets/qwen2.5-3b-instruct','--official-manifest',str(base/'preparation/official-model-manifest-original.json'),'--source-commit',intent['source_commit'],'--initial-probes-source-base',INITIAL0],timeout=900)
    run=read(base/'logs/run-manifest.json');validate_fresh(run,bound=False)
    from reproduction.common8.formal_inline_safety import verify_method
    verify_method(method,run)
    value=dict(stage='NATIVE_FORMAL_CPU_PREPARED',run_id=run['run_id'],execution_id=run['execution_id'],source_commit=intent['source_commit'],cpu_checks=compatibility,actor_started=False,bank_pending=True,head_binding_pending=True,checked_unix=time.time())
    once(base/'logs/native-prepared.json',value);return value


def validate_fresh(run,*,bound):
    assert run['execution_backend']=='ssh_local' and run['slurm_job_id'] is None and run['execution_host']==socket.gethostname()
    assert run['profile']=='common8' and run['completed_updates']==0 and run['target_updates']==200 and run['training_world_size']==4 and run['actor_micro_batch_per_rank']==4
    assert run['initial_model']==run['reference_model']=='/data/search-g1/assets/qwen2.5-3b-instruct' and run['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    for key in ('full_state_restore','actor_checkpoint','reference_checkpoint','resume_path','inherited_policy_run_id','inherited_optimizer_run_id','inherited_scheduler_run_id'):assert run.get(key) is None
    budget=run['budget'];assert budget['world_size']==4 and budget['per_rank_actor_micro_batch']==4 and budget['ppo_micro_batch_size']==16 and budget['ppo_mini_batch_size']==128 and budget['lr_warmup_steps']==57 and budget['per_rank_log_prob_micro_batch']==8
    assert (128//4)//4==8 and run['initial_probe_version'] is None and run['auxiliary_physical_indices']==[5,6]
    if bound:
        q=run['question_probe'];expected=dict(head_kind='qwen_native_lastblock_binary_v1',head_device='cuda:0',candidate_questions=5120,selected_questions=512,responses_per_question=5,exploration_questions=0,ranking='binary_mixed_probability',source_layer=35,head_epochs_per_outer=2,replay_window_outer=8,teacher_replay_excluded=True,std_loss_weight=.3,count_aux_loss_weight=0.,target_statistic='sample_std')
        assert all(q.get(key)==value for key,value in expected.items())
        assert q['initial_head_checkpoint'] and q['initial_head_ready'] and run['mixed_first_selection'] is False
    else:assert run.get('question_probe') is None


def warmup_admit(base,repo,intent,path):
    warmup=Path(intent['warmup_base']);source_commit=intent['warmup_source_commit']
    evidence=read(path,2*1024**2);observation=evidence.get('observation',evidence);audit=observation.get('completion_resource_audit',observation)
    assert audit['accepted'] is True and audit['source_commit']==source_commit and audit['head_adam_steps']==200 and audit['epoch_records']==11 and audit['minibatch_records']==200
    assert audit['live_pinned_births']==audit['actual_private_session_survivors']==audit['owned_cuda_contexts_remaining']==0 and audit['nvml_queries']==1
    ready_path=warmup/'warmup/ready.json';ready=read(ready_path)
    assert sha(ready_path)==audit['ready_sha256'] and ready['source_commit']==source_commit and ready['stage']=='QWEN_LAST_BLOCK_FRESH_HEAD_WARMUP10_COMPLETE'
    assert read(warmup/'preparation/intent.json')==intent['warmup_handle'] and ready['source_map_sha256']==intent['warmup_source_map_sha256']
    publication=read(warmup/'preparation/source-publication.json');assert publication['source_commit']==publication['remote_commit']==source_commit and publication['source_map_sha256']==intent['warmup_source_map_sha256']
    assert publication['remote_fresh_object_store_disk_git_hash_equal'] is True
    assert ready['head_adam_steps']==ready['head_optimizer_updates']==200 and ready['old_head_inherited'] is False and ready['no_reencode'] is True and ready['formal_training_started'] is False
    native=ready['native_block_proof'];assert native['native_class']=='transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer' and native['block_index']==35 and native['official_block_copy_verified'] is True and native['fresh_linear'] is True and native['fresh_adam'] is True
    assert sha(warmup/'warmup/warmup-metrics.json')==audit['metrics_sha256']==ready['metrics_sha256']
    assert sha(warmup/'warmup/fresh-head.pt')==ready['head_sha256']
    launch=read(warmup/'logs/run-launch.json');claim=read(warmup/'warmup/run-claim.json')
    assert launch['process']==audit['worker'] and {key:claim[key] for key in ('pid','start_ticks','sid','uid')}==audit['runner']
    assert not alive(audit['worker']) and not alive(audit['runner'])
    assert read(warmup/'logs/run-exit.json')['exit_code']==0
    value=dict(stage='ACTUAL_NATIVE_WARMUP_AND_RESOURCE_RELEASE_ADMITTED',source_commit=source_commit,source_map_sha256=intent['warmup_source_map_sha256'],warmup_base=str(warmup),ready_sha256=audit['ready_sha256'],metrics_sha256=audit['metrics_sha256'],head_sha256=ready['head_sha256'],actual_resource_audit=audit,evidence_sha256=sha(path),checked_unix=time.time())
    once(base/'logs/native-warmup-admitted.json',value);return value


def required_warmup(base,intent):
    value=read(base/'logs/native-warmup-admitted.json');assert value['stage']=='ACTUAL_NATIVE_WARMUP_AND_RESOURCE_RELEASE_ADMITTED'
    assert value['warmup_base']==intent['warmup_base'] and value['source_commit']==intent['warmup_source_commit'] and value['source_map_sha256']==intent['warmup_source_map_sha256']
    assert sha(Path(intent['warmup_base'])/'warmup/ready.json')==value['ready_sha256'] and sha(Path(intent['warmup_base'])/'warmup/fresh-head.pt')==value['head_sha256']
    return value


def bank_command(base,repo,intent,*,prepare_only,physical_index=None):
    assert intent.get('qualified_bank_origin') is None, 'Qualified existing bank must be admitted, never prepared or encoded again'
    if not prepare_only:required_warmup(base,intent)
    args=[PYTHON,'-m','reproduction.a800_common.qprobe_token_feature_bank','--output-dir',intent['bank_base'],'--source-commit',intent['source_commit'],'--source-publication',str(base/'preparation/source-publication.json')]
    if prepare_only:args+=['--prepare-only']
    else:
        assert type(physical_index) is int and physical_index>=0 and physical_index not in (5,6)
        args+=['--physical-index',str(physical_index)]
    command(base,repo,'bank-prepare' if prepare_only else 'bank-encode',args,timeout=None)
    return dict(stage='FULL_BANK_CPU_PREPARED' if prepare_only else 'FULL_BANK_ENCODER_RETURNED',bank_base=intent['bank_base'],actor_started=False,checked_unix=time.time())


def helpers(intent):
    library=Path(intent['cleanup_library_repo'])
    from reproduction.a800_common.deploy import published_source
    assert subprocess.check_output(['git','-C',str(library),'rev-parse','HEAD'],text=True).strip()==intent['cleanup_library_source']
    assert published_source(library,intent['cleanup_library_source'])==intent['cleanup_library_source_sha256']
    for name,digest in intent['cleanup_library_source_sha256'].items():assert sha(library/name)==digest
    # Load the standalone C4 ownership module without shadowing formal packages.
    import importlib.util
    spec=importlib.util.spec_from_file_location('native_formal_c4_ownership',library/'reproduction/a800/owned_processes.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.owned_session_survivors,module.remaining_cuda_contexts


def bank_admit(base,repo,intent):
    if intent.get('qualified_bank_origin') is not None:return qualified_bank_admit(base,repo,intent)
    required_warmup(base,intent);bank=Path(intent['bank_base']);complete=read(bank/'complete.json');ended=read(bank/'encode-exit.json')
    assert ended['exit_code']==0 and complete['stage']=='FULL_CANONICAL_TOKEN_BANK_PRODUCED'
    assert ended['producer_identity']==complete['producer_identity']
    assert complete['source_commit']==intent['source_commit'] and complete['rows']==169028 and complete['reused_rows']==5120 and complete['newly_encoded_rows']==163908 and complete['reused_rows_byte_exact'] is True
    assert complete['feature_spec_sha256']==SPEC_SHA and complete['teacher_trajectories_generated']==complete['head_adam_updates']==0 and complete['actor_started'] is False
    leader=read(base/'logs/bank-encode-launch.json')['process'];producer=complete['producer_identity'];assert not alive(leader) and not alive(producer)
    assert leader['pid']==leader['sid']==producer['sid'] and leader['uid']==producer['uid']==os.getuid()
    survivors,contexts=helpers(intent);live,reused_sessions=survivors([leader]);assert not live
    import csv
    rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20))))
    remaining,reused_pids=contexts([leader,producer],rows);assert not remaining
    from search_g1.qprobe_token_features import TokenFeatureCache
    cache=TokenFeatureCache(bank/'token-features',expected_spec_sha=SPEC_SHA);cache.verify_all()
    assert cache.manifest['count']==169028 and cache.manifest['total_tokens']==18367021 and sha(cache.path/'manifest.json')==complete['feature_manifest_sha256'] and cache.manifest['files']==complete['full_file_sha256']
    files=[]
    for name in BANK_FILES:
        path=owned(cache.path/name);st=path.stat();files.append(dict(path=str(path),device=st.st_dev,inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns))
    value=dict(stage='FULL_BANK_SHA_AND_ENCODER_RESOURCE_RELEASE_ADMITTED',feature_manifest_sha256=complete['feature_manifest_sha256'],feature_spec_sha256=SPEC_SHA,cache_dir=str(cache.path),files=files,complete_sha256=sha(bank/'complete.json'),encoder_worker=leader,encoder_producer=producer,owned_sessions_remaining=0,owned_cuda_contexts_remaining=0,reused_session_count=len(reused_sessions),reused_pid_count=len(reused_pids),checked_unix=time.time())
    once(base/'logs/native-bank-admitted.json',value);return value


def qualified_bank_metadata(base,repo,intent):
    origin=intent['qualified_bank_origin'];validate_qualified_bank_origin(origin,intent['source_sha256'])
    old=Path(origin['base']);old_intent=read(old/'preparation/intent.json')
    assert hashlib.sha256(canonical(old_intent)).hexdigest()==origin['old_intent_canonical_sha256']
    assert old_intent['source_commit']==origin['source_commit'] and old_intent['base']==str(old) and old_intent['bank_base']==intent['bank_base']
    assert hashlib.sha256(canonical(old_intent['source_sha256'])).hexdigest()==origin['source_map_sha256']
    for name,digest in QUALIFIED_BANK_PRODUCERS.items():assert old_intent['source_sha256'][name]==digest==sha(old/'repo'/name)==sha(repo/name)
    audit_path=base/'preparation/qualified-bank-terminal-audit.json'
    assert audit_path.stat().st_size==origin['terminal_audit_bytes'] and sha(audit_path)==origin['terminal_audit_sha256']
    audit=read(audit_path,2*1024**2);validate_terminal_bank_origin(origin,audit)
    for relative,item in audit['metadata_files'].items():assert sha(old/relative)==item['sha256']
    old_run=read(old/'logs/run-manifest.json')
    assert all(old_run[key]==origin[key] for key in ('source_commit','run_id','execution_id')) and old_run['completed_updates']==0
    assert old_run.get('full_state_restore') is None
    for name in ('actual-attempts.jsonl','actual-updates.jsonl')+tuple('optimizer-rank-'+str(rank)+'.jsonl' for rank in range(4)):
        path=old/'logs'/name;assert not path.exists() or path.stat().st_size==0, 'Origin is not pre-update'
    assert not list((old/'artifacts/question-probe/completed').glob('*.json'))
    admitted_path=old/'logs/native-bank-admitted.json';assert sha(admitted_path)==origin['admitted_sha256']
    admitted=read(admitted_path);assert admitted==origin['admitted'];check_bank_file_seal(admitted)
    bank=Path(intent['bank_base']);complete=read(bank/'complete.json');ended=read(bank/'encode-exit.json')
    assert sha(bank/'complete.json')==origin['complete_sha256'] and complete['source_commit']==origin['source_commit'] and complete['source_map_sha256']==origin['source_map_sha256']
    assert ended['exit_code']==0 and ended['producer_identity']==complete['producer_identity']==admitted['encoder_producer']
    assert read(old/'logs/bank-encode-launch.json')['process']==admitted['encoder_worker']
    assert complete['stage']=='FULL_CANONICAL_TOKEN_BANK_PRODUCED' and complete['rows']==169028 and complete['reused_rows']==5120 and complete['newly_encoded_rows']==163908
    assert complete['canonical_order'] is complete['reused_rows_byte_exact'] is True and complete['teacher_trajectories_generated']==complete['head_adam_updates']==0 and complete['actor_started'] is False
    cache=Path(admitted['cache_dir']);manifest=read(cache/'manifest.json')
    assert sha(cache/'manifest.json')==complete['feature_manifest_sha256']==origin['feature_manifest_sha256']
    assert manifest['files']==complete['full_file_sha256'] and manifest['count']==169028 and manifest['total_tokens']==18367021
    assert manifest['spec_sha']==complete['feature_spec_sha256']==SPEC_SHA and hashlib.sha256(canonical(manifest['spec'])).hexdigest()==SPEC_SHA
    run=read(base/'logs/run-manifest.json');validate_fresh(run,bound=False if run.get('question_probe') is None else True)
    science=complete['qualification'];assert science['model_revision']==run['model_revision']
    assert all(science['canonical_input_sha256'][name]==digest for name,digest in run['input_sha256'].items())
    assert manifest['spec']['train_parquet_sha256']==run['input_sha256']['train.parquet']
    assert manifest['spec']['encoder_files_sha256']==hashlib.sha256(canonical(science['model_files_sha256'])).hexdigest()
    records={}
    launch=audit['metadata_files']['logs/deployment-launch-receipt.json']['value']
    for record in [launch,admitted['encoder_worker'],admitted['encoder_producer']]+[record for job in audit['metadata_files']['logs/process-ledger.json']['value']['jobs'].values() for record in [job['leader'],*job['ledger'].values()]]:
        identity={key:record[key] for key in ('pid','start_ticks','sid','uid')};records[tuple(identity.values())]=identity
    survivors,_=helpers(intent);require_recorded_sessions_gone(records.values(),survivors)
    return admitted


def qualified_bank_admit(base,repo,intent):
    required_warmup(base,intent);admitted=qualified_bank_metadata(base,repo,intent);origin=intent['qualified_bank_origin']
    value=dict(admitted,source_commit=intent['source_commit'],qualified_bank_origin=dict(base=origin['base'],source_commit=origin['source_commit'],
        source_map_sha256=origin['source_map_sha256'],admitted_sha256=origin['admitted_sha256'],terminal_audit_sha256=origin['terminal_audit_sha256']),
        reused_qualified_bank=True,bank_reencoded=False,packed_features_rehashed=False,original_full_sha_seal_revalidated=True,
        actor_or_head_updates_inherited=False,formal_completed_outers=0,checked_unix=time.time())
    once(base/'logs/native-bank-admitted.json',value);return value


def required_bank(base,intent):
    value=read(base/'logs/native-bank-admitted.json');assert value['stage']=='FULL_BANK_SHA_AND_ENCODER_RESOURCE_RELEASE_ADMITTED'
    assert value['cache_dir']==intent['bank_base']+'/token-features' and sha(Path(value['cache_dir'])/'manifest.json')==value['feature_manifest_sha256']
    assert sha(Path(intent['bank_base'])/'complete.json')==value['complete_sha256']
    check_bank_file_seal(value)
    assert not alive(value['encoder_worker']) and not alive(value['encoder_producer'])
    if intent.get('qualified_bank_origin') is not None:
        admitted=qualified_bank_metadata(base,base/'repo',intent);origin=intent['qualified_bank_origin']
        assert value['reused_qualified_bank'] is True and value['source_commit']==intent['source_commit']
        assert value['bank_reencoded'] is value['packed_features_rehashed'] is value['actor_or_head_updates_inherited'] is False
        assert value['original_full_sha_seal_revalidated'] is True and value['files']==admitted['files']
        assert value['qualified_bank_origin']=={key:origin[key] for key in ('base','source_commit','source_map_sha256','admitted_sha256','terminal_audit_sha256')}
    return value


def bind(base,repo,intent):
    unlaunched(base);warmup=required_warmup(base,intent);bank=required_bank(base,intent)
    from search_g1.rollout_phase_contract import DEFAULT_ROLLOUT_PHASE
    from reproduction.common8.stage_retrieval import ASSETS
    from search_g1.qprobe_features import sha_json
    phase=base/'preparation/native-infra-rollout.json';once(phase,DEFAULT_ROLLOUT_PHASE)
    command(base,repo,'native-bind',[PYTHON,'-m','reproduction.common8.qprobe_qwen_bind','--base',str(base),'--cache-dir',bank['cache_dir'],'--warmup-ready',intent['warmup_base']+'/warmup/ready.json','--warmup-ready-sha256',warmup['ready_sha256'],'--retriever-version',sha_json(ASSETS),'--infra-rollout-json',str(phase)],timeout=None)
    run=read(base/'logs/run-manifest.json');validate_fresh(run,bound=True)
    return dict(stage='NATIVE_HEAD_AND_FULL_BANK_BOUND_CPU_ONLY',run_id=run['run_id'],head_device=run['question_probe']['head_device'],actor_started=False,checked_unix=time.time())


def adoption(base,repo,intent):
    unlaunched(base);validate_fresh(read(base/'logs/run-manifest.json'),bound=True)
    command(base,repo,'native-initial0-adopt',[PYTHON,'-m','reproduction.a800_common.adopt_initial_probes','--base',str(base),'--source-base',INITIAL0],timeout=None)
    return verify_initial0(base,read(base/'logs/run-manifest.json'))


def verify_initial0(base,run):
    proof=read(base/'logs/initial-probe-adoption.json');live=read(base/'artifacts/live/manifest.json')
    assert proof['adopted'] is True and proof['run_id']==live['run_id']==run['run_id'] and proof['source_commit']==run['source_commit'] and proof['execution_id']==run['execution_id']
    assert proof['new_calibration_generated'] is False and proof['inherited_training_state'] is False and proof['actual_original_calibration_exit_code']==0
    assert proof['origin']['base']==INITIAL0 and proof['version']==live['version'] and live['round']==0
    assert sha(base/'artifacts/live/manifest.json')==proof['live_manifest_sha256'] and live['target_source_commit']==run['source_commit'] and live['target_source_sha256']==run['source_sha256']
    from reproduction.a800.probe_admission import calibrated_version
    calibrated_version(run,live)
    for key,name in (('commit','commit_probe.pkl'),('boundary','boundary_probe.pkl')):assert sha(owned(live[key]))==live['head_sha256'][name]
    assert sha(owned(proof['target_archive']))==proof['target_archive_sha256']
    return proof


def prepare_launch(base,repo,intent):
    unlaunched(base);cpu=required_cpu_checks(base,repo,intent);run=read(base/'logs/run-manifest.json');validate_fresh(run,bound=True)
    warmup=required_warmup(base,intent);bank=required_bank(base,intent);initial0=verify_initial0(base,run)
    from reproduction.common8.formal_inline_safety import verify_method
    verify_method(read(base/'preparation/formal-inline-method.json'),run)
    from reproduction.a800_common.qprobe_adoption_source_delta import validate_native_initial_head
    validate_native_initial_head(run,run['question_probe'])
    value=dict(stage='NATIVE_FRESH_FORMAL200_PRELAUNCH_QUALIFIED',run_id=run['run_id'],execution_id=run['execution_id'],source_commit=intent['source_commit'],source_map_sha256=hashlib.sha256(canonical(intent['source_sha256'])).hexdigest(),cpu_checks_receipt_sha256=cpu['receipt_sha256'],warmup_ready_sha256=warmup['ready_sha256'],warmup_head_sha256=warmup['head_sha256'],bank_manifest_sha256=bank['feature_manifest_sha256'],initial0_adoption_sha256=sha(base/'logs/initial-probe-adoption.json'),run_manifest_sha256=sha(base/'logs/run-manifest.json'),head_device='cuda:0',actor_started=False,actual_cuda_restore_accepted=False,target_updates=200,checked_unix=time.time())
    once(base/'logs/native-prelaunch.json',value);return value


def validate_release(review,intent,qualified):
    assert review['role']=='native_qwen_formal200_independent_release_review' and review['approved'] is True and review['independent'] is True and review['context_free'] is True
    assert review['reviewer']!=review['implementation_actor'] and review['reviewer'] and review['implementation_actor']
    assert review['source_commit']==intent['source_commit'] and review['source_map_sha256']==qualified['source_map_sha256']
    assert review['human_intent_sha256']==intent['human_intent_sha256'] and review['authorized_source_publication_approved'] is True
    assert review['warmup_ready_sha256']==qualified['warmup_ready_sha256'] and review['warmup_head_sha256']==qualified['warmup_head_sha256'] and review['bank_manifest_sha256']==qualified['bank_manifest_sha256']
    assert review['actual_deployment_review_approved'] is True


def launch(base,repo,intent,review_path):
    unlaunched(base);qualified=read(base/'logs/native-prelaunch.json');review=read(review_path);validate_release(review,intent,qualified)
    assert required_cpu_checks(base,repo,intent)['receipt_sha256']==qualified['cpu_checks_receipt_sha256']
    assert sha(base/'logs/run-manifest.json')==qualified['run_manifest_sha256'];required_warmup(base,intent);required_bank(base,intent)
    run=read(base/'logs/run-manifest.json');validate_fresh(run,bound=True);verify_initial0(base,run)
    once(base/'logs/native-formal-launch-once.json',dict(run_id=run['run_id'],execution_id=run['execution_id'],source_commit=intent['source_commit'],review_sha256=sha(review_path),no_automatic_retry=True,claimed_unix=time.time()))
    env=environment(base,repo);env['RAY_TMPDIR']=run['ray_tmp_dir']
    with (base/'logs/deployment-pipeline.log').open('xb') as log:
        process=subprocess.Popen([PYTHON,'-m','reproduction.a800_common.qprobe_qwen_formal_ops','--base',str(base),'--action','pipeline'],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=birth(process.pid);assert record['pid']==record['sid']
        receipt=dict(deployment_id=intent['deployment_id'],run_id=run['run_id'],execution_id=run['execution_id'],source_commit=intent['source_commit'],execution_backend='ssh_local',slurm_job_id=None,hostname=socket.gethostname(),stdout_log=str(base/'logs/deployment-pipeline.log'),no_automatic_retry=True,launched_unix=time.time(),**record)
        once(base/'logs/deployment-launch-receipt.json',receipt)
    return dict(stage='ONE_FRESH_NATIVE_FORMAL200_PIPELINE_LAUNCHED',**receipt)


def pipeline(base,repo,intent):
    """Original local pipeline with pre-adopted initial0 verification only."""
    import fcntl
    import signal
    from reproduction.a800_common.run_pipeline import LocalPipeline,bound_ray_tmp_dir,request_interrupt
    from reproduction.common8.formal_contract import assert_formal_run
    from reproduction.common8.topology import validate_base,execution_job_id
    from reproduction.common8.run_pipeline import supported_execution_role
    from reproduction.a800.owned_processes import publish
    class NativePipeline(LocalPipeline):
        def initial_probe_calibration(self):
            proof=verify_initial0(self.base,self.run)
            self.stage('PREPARED_ORIGINAL_INITIAL0_ADOPTION_REVALIDATED',version=proof['version'],new_calibration_generated=False)
    run=read(base/'logs/run-manifest.json');validate_fresh(run,bound=True);assert_formal_run(run);validate_base(base,run)
    assert os.environ.get('RAY_TMPDIR')==bound_ray_tmp_dir(run) and execution_job_id(run) is None and supported_execution_role(run,base)=='production'
    assert os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE')=='4'
    lock=(base/'gpu-job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,request_interrupt);signal.signal(signal.SIGINT,request_interrupt)
    assert not (base/'logs/process-ledger.json').exists() and not (base/'logs/pipeline-launch-claim.json').exists()
    for port in (18010,18012):
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    from reproduction.common8.native_allocation import admit,bind_idle_local_placement,local_placement
    run=bind_idle_local_placement(base);training,_,_,_=local_placement(run);os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,training));allocation=admit(base)
    claim=dict(run_id=run['run_id'],execution_backend='ssh_local',execution_id=run['execution_id'],slurm_job_id=None,node=allocation['node'],pipeline_identity=allocation['pipeline_identity'],training_physical_indices=allocation['training_physical_indices'],auxiliary_physical_indices=allocation['auxiliary_physical_indices'],excluded_physical_indices=allocation['excluded_physical_indices'],gpu_placement_binding_sha256=allocation['gpu_placement_binding_sha256'],assigned_gpu_uuids=allocation['assigned_gpu_uuids'],allocation_sha256=sha(base/'logs/native-allocation-admission.json'),claimed_unix=time.time())
    once(base/'logs/pipeline-launch-claim.json',claim);run['pipeline_identity']=allocation['pipeline_identity'];publish(base/'logs/run-manifest.json',run)
    from reproduction.common8.disk_admission import admit as admit_disk
    admit_disk(base);NativePipeline(base,run).execute()
    return dict(stage='NATIVE_FORMAL_PIPELINE_RETURNED',run_id=run['run_id'],execution_id=run['execution_id'])


def observe(base,repo,intent):
    result=dict(stage='NATIVE_FORMAL_EXISTING_METADATA_ONLY',base=str(base),source_commit=intent['source_commit'],metadata_only=True,files={})
    paths=['logs/native-formal-cpu-checks.json','logs/native-prepared.json','logs/native-warmup-admitted.json','logs/native-bank-admitted.json','logs/qprobe-bound.json','logs/initial-probe-adoption.json','logs/native-prelaunch.json','logs/deployment-launch-receipt.json','logs/pipeline-status.json','logs/cleanup-status.json']
    for action in ('prepare','bank-prepare','bank-encode','warmup-admit','bank-admit','bind','adopt-initial0','prepare-launch','launch'):
        paths += ['logs/'+action+'-'+suffix+'.json' for suffix in ('launch','exit','helper-exit','failure')]
    paths += [str(Path(intent['bank_base'])/name) for name in ('prepare-claim.json','prepare-exit.json','prepare-failure.json','prepared.json','encode-claim.json','encode-exit.json','encode-failure.json','progress.json','complete.json')]
    for relative in paths:
        path=Path(relative) if relative.startswith('/') else base/relative
        if not path.exists():continue
        value=read(path,2*1024**2);item=dict(bytes=path.stat().st_size,sha256=sha(path),metadata=value)
        if isinstance(value.get('process'),dict):item['actual_birth_alive']=alive(value['process'])
        result['files'][relative]=item
    if (base/'preparation/deployment-intent.json').exists():
        from reproduction.a800_common.deploy import REMOTE_OBSERVE
        observation=subprocess.run([PYTHON,'-c',REMOTE_OBSERVE],input=canonical(read(base/'preparation/deployment-intent.json')),capture_output=True,timeout=75)
        assert len(observation.stdout)<=512*1024
        rows=[json.loads(line.split(':',1)[1]) for line in observation.stdout.decode().splitlines() if line.startswith('DEPLOYMENT_JSON:')]
        result['original_pipeline_metadata']=rows[-1] if observation.returncode==0 and len(rows)==1 else dict(status='unknown',exit_code=observation.returncode)
    result['checked_unix']=time.time();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True)
    parser.add_argument('--action',required=True,choices=('prepare','warmup-admit','bank-prepare','bank-encode','bank-admit','bind','adopt-initial0','prepare-launch','launch','pipeline','observe'))
    parser.add_argument('--warmup-receipt');parser.add_argument('--physical-index',type=int);parser.add_argument('--review')
    args=parser.parse_args()
    if args.action!='pipeline':os.environ['CUDA_VISIBLE_DEVICES']=''
    base,repo,intent=load(args.base)
    if args.action=='observe':print('NATIVE_FORMAL_JSON:'+json.dumps(observe(base,repo,intent),allow_nan=False));return 0
    if args.action!='pipeline':
        unlaunched(base);once(base/'logs'/(args.action+'-once.json'),dict(source_commit=intent['source_commit'],producer_identity=birth(os.getpid()),no_automatic_retry=True,claimed_unix=time.time()))
    exit_code=1
    try:
        if args.action=='prepare':value=prepare(base,repo,intent)
        elif args.action=='warmup-admit':assert args.warmup_receipt;value=warmup_admit(base,repo,intent,args.warmup_receipt)
        elif args.action in ('bank-prepare','bank-encode'):value=bank_command(base,repo,intent,prepare_only=args.action=='bank-prepare',physical_index=args.physical_index)
        elif args.action=='bank-admit':value=bank_admit(base,repo,intent)
        elif args.action=='bind':value=bind(base,repo,intent)
        elif args.action=='adopt-initial0':value=adoption(base,repo,intent)
        elif args.action=='prepare-launch':value=prepare_launch(base,repo,intent)
        elif args.action=='launch':assert args.review;value=launch(base,repo,intent,args.review)
        else:value=pipeline(base,repo,intent)
        print('NATIVE_FORMAL_JSON:'+json.dumps(value,allow_nan=False),flush=True);exit_code=0
    except BaseException as failure:
        frames=[dict(file=Path(frame.filename).name,line=frame.lineno,function=frame.name) for frame in traceback.extract_tb(failure.__traceback__)[-8:]]
        once(base/'logs'/(args.action+'-failure.json'),dict(failure_type=type(failure).__name__,frames=frames,partial_artifacts_preserved=True,no_automatic_retry=True,checked_unix=time.time()))
        print('NATIVE_FORMAL_JSON:'+json.dumps(dict(stage='NATIVE_FORMAL_ACTION_FAILED',action=args.action,failure_type=type(failure).__name__,no_automatic_retry=True)),flush=True)
    finally:
        once(base/'logs'/(args.action+'-helper-exit.json'),dict(exit_code=exit_code,producer_identity=birth(os.getpid()),process_exit_not_yet_proven=True,checked_unix=time.time()))
    return exit_code


if __name__=='__main__':raise SystemExit(main())
