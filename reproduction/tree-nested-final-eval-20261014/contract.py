"""Fixed778 trained200 final-only handoff; never stops or resumes training."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
TRAIN_SOURCE = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
TRAIN_JOB = 'not-slurm;systemd:' + TRAIN_SOURCE
TRAIN_UNIT = 'sst-tree-nested-branch-fresh200-778b2b43.service'
RUN = ROOT / 'runs' / ('tree-nested-branch-fresh200-' + TRAIN_JOB)
PACKAGE = ROOT / 'instruct300/candidates' / ('nested-branch-fresh200-' + TRAIN_SOURCE)
PACKAGE_SHA = 'c2c449c925a3b482b8152f09390789165aa48f3481f15caa62f477c7ca953188'
FROZEN_SOURCE = PACKAGE / 'source/repo'
CONFIG = RUN / 'resolved.yaml'
CONFIG_SHA = '355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3'
NATIVE = RUN / 'checkpoints/training_state/global_step_200/manifest.json'
HF = RUN / 'checkpoints/actor/global_step_200'
OUTPUT = RUN / 'output/final-eval-fast'
COUNTS = dict(nq=3610, triviaqa=11313, popqa=14267, hotpotqa=7405,
              **{'2wikimultihopqa':12576}, musique=2417, bamboogle=125)
GPU_UUIDS = ('GPU-2c3bdc08-b976-0fb5-79bd-04def458046b', 'GPU-6e00764e-3916-65f2-436e-568aa0c70fc5',
             'GPU-db94211b-42d5-88d0-f626-60e777af3be6', 'GPU-308b13c1-44ea-47b7-7b33-01a928099618',
             'GPU-525a9efc-446e-333d-11ff-e1f2213d15c8', 'GPU-d61d94cd-3d24-fa07-8761-33961a14d995',
             'GPU-f3a967b0-0b4a-4d13-0917-0a4f70aa740b', 'GPU-fca5fcba-4036-0e79-4cb6-ceb9ed121ad5')
REUSED_FILES = {'eval_worker.py':'906095da3e22b4f667fdb36cc635aa52bd8b0d968d116aeb3dcc59f4fda3fcea',
                'eval_merge.py':'9159f0b2021a97cb4aa24bcfe8e67c08c6cafc44f7d567976aae8a851e1b205a',
                'shared_guard.py':'b750ae444454106b002138b79e188769f46562bc95066e9dcfc488acb5740086'}
RESTORE_RECEIPT = RUN/'output/nested-native5-loaded.json'
RESTORE_RECEIPT_SHA = 'cceb1cfe9263fcf7213a375585d97a06cf54ca179f11c0f6a03985b7c2c348d6'
ORIGIN_SOURCE = '0befbaf99fc44942105831b3fbec94e4bf1b8ccb'
MIDDLE_SOURCE = 'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4'
BOOTSTRAP_SOURCE = 'eb6a69e47163b179b20b66cf4992c0d5e152ea6c'
BOOTSTRAP_FAULT = dict(path=str(ROOT/'instruct300/candidates'/('nested-branch-fresh200-'+BOOTSTRAP_SOURCE)/'bootstrap-fault.json'),
    sha256='bb6d682432f93802253d16fe01c9ec1d1eb990b3c24502232d2f833d1e18853d')
BOOTSTRAP_JOURNAL = dict(path=str(ROOT/'instruct300/submission-history'/('TERMINAL_BOOTSTRAP_FAILURE-SUBMISSION_ONCE-nested-branch-fresh200-'+BOOTSTRAP_SOURCE+'.json')),
    sha256='4260b4bb508417c091bcbaeee010beee90bd078d22ad4b2eab85ec4df1d5dddb')


def effective_segments():
    specs=[(ORIGIN_SOURCE,'2a317628c497bba932698bf6d7bf64ca3503233d32080a389a36e1f69b362243',1,1),
           (MIDDLE_SOURCE,'22beaec8aa2b449b496754aacfd9b1e7c89009c9c27a8eff3aef92b768b28b59',2,5),
           (TRAIN_SOURCE,CONFIG_SHA,6,200)]
    return [dict(source_run=str(ROOT/'runs'/('tree-nested-branch-fresh200-not-slurm;systemd:'+source)),
                 run_id='tree-nested-branch-fresh200-not-slurm;systemd:'+source,
                 code_commit=source,config_sha256=config,first_outer=first,last_outer=last)
            for source,config,first,last in specs]

HANDOFF_STATUS = 'EXPLICIT_NESTED_TRAIN200_FINAL_EVALUATION_HANDOFF_AUTHORIZED'


def require(condition, message):
    if not condition:
        raise ValueError('Nested final handoff: ' + message)


def sha(path):
    path = Path(path)
    require(stat.S_ISREG(path.lstat().st_mode), 'ordinary file required: '+str(path))
    value = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY|os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda:stream.read(8*1024**2), b''):
            value.update(block)
        after = os.fstat(stream.fileno())
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
            == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'file changed while held')
    return value.hexdigest()


def read_json(path, expected=None):
    path = Path(path)
    require(sha(path) == expected if expected is not None else True, 'JSON hash changed: '+str(path))
    with os.fdopen(os.open(path, os.O_RDONLY|os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno()); raw = stream.read(); after = os.fstat(stream.fileno())
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
            == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
            and (expected is None or hashlib.sha256(raw).hexdigest() == expected), 'JSON changed while held')
    return json.loads(raw)


def descriptor(item, expected_path=None):
    require(isinstance(item,dict) and set(item)=={'path','sha256'}, 'path/SHA descriptor required')
    path = Path(item['path'])
    require(path.is_absolute() and '..' not in path.parts and path.resolve().is_relative_to(ROOT.resolve()),
            'descriptor must remain in this retained remote task storage')
    require(expected_path is None or path == expected_path, 'descriptor path differs')
    require(re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None, 'full SHA required')
    return read_json(path, item['sha256'])


def verify_code_manifest(tools, manifest, expected):
    tools = Path(tools).resolve(); value = read_json(manifest, expected)
    require(socket.gethostname().split('.')[0]=='h100-10', 'only retained H10018 is permitted')
    commit = value['eval_wrapper_source_commit']
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None and commit != TRAIN_SOURCE
            and tools == (ROOT/'instruct300'/('eval-nested-'+commit)).resolve(), 'distinct committed wrapper identity required')
    external={str(ROOT/'tools/tree_shared_owned_runtime_fixed.py'):'4288a4dcfa923df4d8d74f009bf4c56c8446d04fa8ef4202c5e55d3371539ba0',
              str(ROOT/'repo/reproduction/h20/retriever_local.py'):'a78f0b493c538dddc6619e0e462995fd9945c25a187c2165e99d4c10622ce459'}
    require(value['external_files']==external and all(sha(name)==expected for name,expected in external.items()),
            'unchanged own runtime/retriever helpers must remain pinned')
    require(value['schema']=='nested_final_eval_code_v1' and value['training_source_commit']==TRAIN_SOURCE
            and value['operational_changes_independently_reviewed'] is False
            and value['training_chain']==dict(package_sha256=PACKAGE_SHA,config_sha256=CONFIG_SHA,
                native5_restore_receipt_sha256=RESTORE_RECEIPT_SHA,effective_training_segments=effective_segments())
            and value['readiness']=='PENDING_REAL_TRAIN200_NATIVE_HF_ARCHIVE_AND_EXPLICIT_HANDOFF',
            'wrapper identity or unclaimed operational review differs')
    names = {'contract.py','prepare.py','supervisor.py','dispatch_once.py','build_code_package.py',
             'README.md',*REUSED_FILES}
    require(set(value['files'])==names, 'complete standalone wrapper inventory required')
    for name, expected_sha in value['files'].items():
        require(sha(tools/name)==expected_sha, 'wrapper source changed: '+name)
    require(all(value['files'][name]==digest for name,digest in REUSED_FILES.items()),
            'actually used worker/merge/guard bytes must remain unchanged')
    return value


def restore_and_fault_evidence():
    """Real loaded5 identity and fault costs are retained apart from effective credit."""
    loaded=read_json(RESTORE_RECEIPT,RESTORE_RECEIPT_SHA);p=loaded['provenance'];d=loaded['driver']
    require(loaded['status']=='COLD_NESTED_NATIVE5_STATE_LOADED'
            and loaded['protocol']=='tree_nested_fresh200_native5_ray_thread_continuation_v1'
            and p['child_code_commit']==TRAIN_SOURCE and p['child_config_sha256']==CONFIG_SHA
            and p['child_job_id']==TRAIN_JOB and p['child_run_id']==RUN.name
            and p['parent_manifest_sha256']=='b112caffda6514c34a07f2f6b6ecc62096c0717eb09ebae46aac71c6b9dfe09a'
            and p['ancestry_receipt_sha256']=='dc6f1f065a21d048b21a9462b55f992f9762b07f9b00633cb7028af99443f096'
            and p['effective_ancestor_segments']==effective_segments()[:2]
            and p['ancestry_provenance']['child_code_commit']==MIDDLE_SOURCE
            and p['ancestry_provenance']['parent_identity']['code_commit']==ORIGIN_SOURCE
            and d['completed_outer_updates']==d['training_attempts']==5 and d['next_outer_update']==6
            and d['sampler_cursor']==2560 and d['sampler_epoch']==0
            and d['actual_cold_driver_state_equal'] is True and d['actual_file_sha256_verified'] is True
            and d['completed_optimizer_calls_per_rank']=={str(r):100 for r in range(8)},
            'actual native5 loaded driver/ancestry identity differs')
    workers=sorted(loaded['workers'],key=lambda w:w['rank'])
    require(len(workers)==8 and [w['rank'] for w in workers]==list(range(8))
            and all(w['status']=='COLD_NESTED_NATIVE5_WORKER_LOADED' and w['world_size']==8
                and w['adam_steps']==[100] and w['actual_total_optimizer_calls']==100 and w['scheduler_step']==5
                and w['child_code_commit']==TRAIN_SOURCE and w['actual_cold_native_state_equal'] is True
                and w['actual_file_sha256_verified'] is True
                and w['serialized_logical_state_sha256']==w['actual_loaded_logical_state_sha256']
                and w['exact_executor_resume'] is False for w in workers)
            and loaded['exact_executor_resume'] is False and loaded['executor_equivalence'] is False,
            'actual native5 loaded eight-rank equality differs')
    failed2,failed6=p['failed_parent_outer2'],p['failed_parent_outer6']
    require(failed2['failed_attempt_in_effective_history'] is False
            and failed6['failed_attempt_in_effective_history'] is False
            and failed2['actual_completed_outer_before_failure']==2
            and failed6['actual_completed_outer_before_failure']==5,
            'lost/failed parent attempts cannot become extra trained200 credit')
    evidence={}
    for label,item in [('original_lost_outer2',failed2),('ba9_failed_outer6',failed6)]:
        terminal=item['failed_terminal']
        pins=dict(attempt=dict(path=item['path'],sha256=item['sha256']),
                  terminal=dict(path=terminal['terminal_path'],sha256=terminal['terminal_sha256']),
                  fault=dict(path=terminal['fault_snapshot_path'],sha256=terminal['fault_snapshot_sha256']))
        for pin in pins.values():descriptor(pin)
        evidence[label]=dict(pins,credited_in_effective_history=False,
            cost_evidence_retained=True,cost_independently_accepted=False,
            unrecorded_partial_optimizer_calls=None)
    bootstrap=descriptor(BOOTSTRAP_FAULT);descriptor(BOOTSTRAP_JOURNAL)
    fields=dict(line.split('=',1) for line in bootstrap['systemd'].splitlines() if '=' in line)
    require(bootstrap['status']=='OWN_BOOTSTRAP_EXIT_BEFORE_TRAINING'
            and bootstrap['source_commit']==BOOTSTRAP_SOURCE
            and bootstrap['child_run_created'] is False and bootstrap['native_load_attempted'] is False
            and bootstrap['training_outer_started'] is False
            and fields['MainPID']=='0' and fields['Result']=='exit-code' and fields['ExecMainStatus']=='1'
            and bootstrap['actual_own_cgroup_exists'] is False,
            'unloaded eb6 startup cannot receive effective training credit')
    evidence['eb6_unloaded_bootstrap']=dict(fault=BOOTSTRAP_FAULT,journal=BOOTSTRAP_JOURNAL,
        credited_in_effective_history=False,actual_optimizer_writes=0,
        observed_cpu_usage_nsec=int(fields['CPUUsageNSec']),cost_evidence_retained=True,
        cost_independently_accepted=False)
    return dict(native5_restore_receipt=dict(path=str(RESTORE_RECEIPT),sha256=RESTORE_RECEIPT_SHA),
                effective_training_segments=effective_segments(),failed_attempts=evidence,
                exact_executor_resume=False)


def training_metadata():
    """JSON-only current-chain completion checks; no native tensor deserialization."""
    require(socket.gethostname().split('.')[0]=='h100-10', 'retained host required')
    require(sha(CONFIG)==CONFIG_SHA, 'actual778 configuration changed')
    package = read_json(PACKAGE/'manifest.json',PACKAGE_SHA)
    require(package['code_commit']==TRAIN_SOURCE, 'frozen training package changed')
    for path in (FROZEN_SOURCE,PACKAGE):
        sys.path.insert(0,str(path))
    os.environ.update(SEARCH_R1_BASELINE_COMMIT=TRAIN_SOURCE,SEARCH_R1_BASELINE_CONFIG_SHA256=CONFIG_SHA,
        SEARCH_R1_BASELINE_RUN_ID=RUN.name,SEARCH_R1_BASELINE_OUTPUT=str(RUN/'output'),
        SEARCH_R1_BASELINE_DATA_IDENTITY=package['common_data_identity'],
        SEARCH_R1_MODEL_IDENTITY=package['model_identity']['sha256'],SLURM_JOB_ID=TRAIN_JOB)
    from search_r1.baselines.tree_nested_chain_resume import install_package_environment
    install_package_environment(package)
    import production_contract
    chain=restore_and_fault_evidence()
    ranks = production_contract.direct_rank_records(RUN,package,CONFIG_SHA,200)
    native, native_sha = production_contract.checkpoint_manifest(RUN,package,CONFIG_SHA,200)
    accounting = native['identity']['guard_continue_accounting']
    require(ranks['completed_outer']==200 and accounting['completed_outer_updates']==200
            and ranks['actual_writes_per_rank']==accounting['total_optimizer_calls_per_rank']
            and ranks['attempt_accounting']['inherited_completed_outers']==5
            and ranks['attempt_accounting']['failed_parent_outer2_in_effective_history'] is False
            and ranks['attempt_accounting']['failed_parent_outer6_in_effective_history'] is False,
            'real effective200 origin1/ba9 2..5/778 6..200 accounting required')
    for event in ranks['attempt_accounting']['credited']:
        segment=next(x for x in chain['effective_training_segments'] if x['first_outer']<=event['outer']<=x['last_outer'])
        require(event['source_run']==segment['source_run']
                and event['expected_identity']['code_commit']==segment['code_commit']
                and event['expected_identity']['config_sha256']==segment['config_sha256'],
                'credited outer leaves actual effective source/config segment')
    for record in native['worker_states']:
        rank = str(record['identity']['rank']); calls = accounting['total_optimizer_calls_per_rank'][rank]
        require(record['actual_total_optimizer_calls']==calls and record['observed_adam_steps']==[calls]
                and record['scheduler_last_epoch']==200 and record['full_worker_state_finite'] is True,
                'native200 rank actual optimizer/scheduler metadata differs')
    return package,native,native_sha,ranks


def original_terminal():
    raw = subprocess.check_output(['systemctl','show',TRAIN_UNIT,
        '--property=MainPID,ActiveState,SubState,Result,ExecMainStatus,ControlGroup'],text=True)
    value = dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
    require(value.get('MainPID')=='0' and value.get('ActiveState') in ('inactive','failed'),
            'current training must remain running until a separately authorized true200 final-only handoff')
    relative = value.get('ControlGroup','') or '/system.slice/'+TRAIN_UNIT
    require(relative=='/system.slice/'+TRAIN_UNIT,'original exact own cgroup identity differs')
    observations=[]
    for base in ('/sys/fs/cgroup','/sys/fs/cgroup/systemd','/sys/fs/cgroup/freezer'):
        scope=Path(base)/relative.lstrip('/')
        if not scope.exists():continue
        require(scope.is_dir() and not scope.is_symlink(),'original own cgroup redirected')
        require((scope/'cgroup.procs').is_file(),'original own process list unavailable')
        for path in sorted(scope.rglob('cgroup.procs')):
            pids=path.read_text().split()
            require(all(pid.isdecimal() for pid in pids) and not pids,'original live descendants remain')
            observations.append(str(path))
    value['own_cgroup_live_descendants_empty']=True
    value['own_cgroup_process_lists_observed']=observations
    return value


def verify_archive(value):
    require(value['status']=='NESTED_NATIVE_HF200_DURABLE_ARCHIVE_VERIFIED'
            and value['training_source_commit']==TRAIN_SOURCE and value['training_run']==str(RUN)
            and value['actual_completed_outer']==200 and value['native_manifest']['path']==str(NATIVE)
            and value['config_sha256']==CONFIG_SHA and value['native_state_deserialized'] is False,
            'real complete200 durable archive proof required')
    require(sha(NATIVE)==value['native_manifest']['sha256'], 'native200 manifest changed since archive')
    native = read_json(NATIVE)
    ids=native['identity']
    require(native['status']=='COMPLETE_SAME_WORLD_STATE_SAVED'
            and ids['job_id']==TRAIN_JOB and ids['code_commit']==TRAIN_SOURCE and ids['config_sha256']==CONFIG_SHA
            and ids['completed_step']==ids['actual_update_limit']==200
            and value['native_identity']==ids and value['hf_checkpoint']==str(HF)
            and value['hf_identity']==native['hf_identity']
            and value['actual_optimizer_calls_per_rank']==ids['guard_continue_accounting']['total_optimizer_calls_per_rank']
            and value['effective_training_segments']==effective_segments()
            and value['restore_and_fault_evidence']==restore_and_fault_evidence()
            and value['inherited_completed_outers']==5 and value['current_child_completed_outers']==195
            and value['effective_parent_outer1_included'] is True and value['failed_parent_original_outer2_included'] is False
            and value['failed_parent_ba9_outer6_included'] is False and value['failed_eb6_bootstrap_included'] is False
            and value['all_files_hash_verified'] is True and value['all_files_fsynced'] is True,
            'archive source/effective200/optimizer/HF/durability binding differs')
    paths = {str(NATIVE),native['driver']['path'],*[r['path'] for r in native['worker_states']],
             *[str(HF/name) for name in native['hf_identity']['files']]}
    require(set(value['files'])==paths, 'archive omitted a native/HF file or added unrelated state')
    for path,record in value['files'].items():
        require(Path(path).stat().st_size==record['bytes'] and sha(path)==record['sha256'], 'archived checkpoint changed: '+path)
    return native


def verify_handoff(path, expected):
    return verify_handoff_value(read_json(path,expected),path)


def verify_handoff_value(value, path):
    require(Path(path).is_absolute() and Path(path).resolve().parent==OUTPUT.resolve(), 'handoff must stay in own final-eval directory')
    require(value['status']==HANDOFF_STATUS and value['training_source_commit']==TRAIN_SOURCE
            and value['training_run']==str(RUN) and value['training_unit']==TRAIN_UNIT
            and value['actual_training_completed_outer']==200 and value['exact_executor_resume'] is False,
            'explicit fixed-chain trained200 handoff required')
    authority = value['authority']
    require(authority['scope']=='only_final_evaluation_after_completed_training200'
            and isinstance(authority['human_instruction'],str) and authority['human_instruction'].strip()
            and isinstance(authority['original_instruction_time'],str) and authority['original_instruction_time'].strip()
            and value['training_stop_or_resume_by_wrapper'] is False,
            'original explicit human final-only handoff authority required; wrapper cannot stop training')
    archive = descriptor(value['archive'],OUTPUT/'archive-proof.json')
    native = verify_archive(archive)
    assets = descriptor(value['asset_manifest'],OUTPUT/'asset-manifest.json')
    prefix = descriptor(value['prefix'],OUTPUT/'bound-prefix-empty.json')
    from eval_merge import identities
    require({r['name']:r['sha256'] for r in assets['model_files']}==native['hf_identity']['files']
            and assets['config_sha256']==CONFIG_SHA and assets['dataset_count']==51713
            and assets['expected_source_counts']==COUNTS and prefix['rows']==[]
            and all(prefix[key]==val for key,val in identities(assets).items())
            and value['score_output']==str(OUTPUT/'scores'), 'new HF200/data/empty prefix binding differs')
    done = read_json(str(OUTPUT/'bound-prefix-empty.json')+'.done.json')
    require(done['count']==0 and done['sha256']==value['prefix']['sha256']
            and done['actual_trained_outer']==200 and done['identity']==identities(assets), 'empty prefix durable proof differs')
    terminal = descriptor(value['original_terminal'],OUTPUT/'original-terminal.json')
    live = original_terminal()
    require(terminal['unit']==TRAIN_UNIT and terminal['systemd']==live
            and terminal['normal_exit_claimed'] is False and terminal['original_final_evaluation_complete'] is False,
            'truthful original interrupted final-only terminal required')
    original = descriptor(value['original_run_result'],RUN/'run-result.json')
    require(original['status']=='FAILED' and original['code_commit']==TRAIN_SOURCE
            and original['job_id']==TRAIN_JOB and original['resumed_current_chain'] is True
            and live['own_cgroup_live_descendants_empty'] is True
            and original['actual_progress_evidence']['actual_completed_outer_from_attempt_results']==200
            and original['actual_progress_evidence']['completed_actual_optimizer_calls_per_rank']
                ==archive['actual_optimizer_calls_per_rank'],
            'original supervisor terminal must retain real trained200 and actual cleanup; no synthesized exit0')
    return value
