"""Small immutable bindings. Importing this module does not initialize CUDA."""
import copy
import hashlib
import json
from pathlib import Path

RUNTIME='c5fc09603a10244655b98ec0ce8eada32668f83b'
RUN='f77c3245dcfe49a88a5a99b1dec161e6'
STEPS=(5,50,100,150,200)
ROLE='post_completion_full_state_restore_only'
SOURCE=Path('/mnt/data/shaoshitong/search-g1-qprobe3-binary10-c5fc096-20261015')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def bind_step(parent,receipt,path,output):
    """Separate acceptance binding allows CP200 without changing training bind."""
    require(parent['run_id']==receipt['run_id']==RUN,'RUN_CHANGED')
    require(parent['source_commit']==RUNTIME,'FROZEN_RUNTIME_CHANGED')
    step=receipt['step'];require(step in STEPS,'UNSUPPORTED_SAVE_POINT')
    require(receipt['status']=='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS','PAIR_NOT_VALIDATED')
    require(receipt['world_size']==8 and type(receipt['attempt_progress']['cumulative_writes']) is int and receipt['attempt_progress']['cumulative_writes']>=step,'ACTUAL_WRITES_OR_WORLD_DIFFERS')
    require(Path(receipt['source_base']).resolve()==SOURCE,'SOURCE_CHANGED')
    require(parent['precision_profile']=='native8_bf16_varlen_fp32_master','PRECISION_CHANGED')
    require(parent.get('execution_site')=='h1008' and parent['actor_micro_batch_per_rank']==4,'MICRO_CHANGED')
    require(parent['initial_model']==parent['reference_model'] and parent['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1','OFFICIAL_FIXED_REF_CHANGED')
    run=copy.deepcopy(parent)
    run['completed_updates']=0
    run['initial_probe_version']=receipt['reward_probe_version']
    run['question_probe']['initial_head_checkpoint']=None
    run['question_probe']['snapshot_path']=None
    run['question_probe']['output_dir']=str(Path(output)/'artifacts/question-probe')
    run['full_state_restore']=dict(kind='question_probe_pair',source_receipt=str(path),source_receipt_sha256=sha(path),
        source_checkpoint=receipt['checkpoint'],source_base=str(SOURCE),
        completed_outer=step,actual_optimizer_writes_per_rank=receipt['attempt_progress']['cumulative_writes'],
        reward_probe_version=receipt['reward_probe_version'],reward_probe_round=receipt['reward_probe_round'])
    run['restore_only_acceptance']=dict(role=ROLE,source_runtime_commit=RUNTIME,step=step,training_updates_allowed=False)
    return run


def validate_binding(document):
    require(document['role']==ROLE and document['run_id']==RUN,'WRONG_ACCEPTANCE_IDENTITY')
    require(document['source_runtime_commit']==RUNTIME and Path(document['source_base']).resolve()==SOURCE,'WRONG_SOURCE')
    require(document['steps']==list(STEPS),'ALL_SAVE_POINTS_REQUIRED_IN_ORDER')
    require(set(document['receipts'])=={str(v) for v in STEPS},'RECEIPT_SET_DIFFERS')
    base=Path(document['base']).resolve();require(SOURCE not in base.parents and base!=SOURCE,'OUTPUT_CANNOT_WRITE_TRAINING')
    require(len(document['physical_gpu_uuids'])==len(set(document['physical_gpu_uuids']))==8,'EIGHT_UUIDS_REQUIRED')
    require(document['allocation_mode']=='shared' and document['cpu_request']==96 and document['memory_request']=='512G','ALLOCATION_BUDGET_CHANGED')
    require('3402' in document['forbidden_training_job_ids'],'TRAINER_OVERLAP_FORBIDDEN')
    for step,row in document['receipts'].items():
        expected=SOURCE/'artifacts/question-probe/completed-pairs'/('global_step_'+step)/'source-receipt.json'
        require(Path(row['path']).resolve()==expected,'RECEIPT_PATH_CHANGED')
        require(sha(expected)==row['sha256'],'RECEIPT_PIN_CHANGED')
    require(sha(document['resolved_config'])==document['resolved_config_sha256'],'RESOLVED_CONFIG_CHANGED')
    require(Path(document['resolved_config']).resolve()==SOURCE/'logs/resolved-training-config.json','CONFIG_SOURCE_CHANGED')
    return document
