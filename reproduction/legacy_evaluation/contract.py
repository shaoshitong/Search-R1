"""Fail closed on old-evaluation identity; has no training runtime hooks."""
import hashlib
from collections import Counter
import json
from pathlib import Path

PROTOCOL_PATH=Path(__file__).with_name('protocol.json')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def protocol():
    return json.loads(PROTOCOL_PATH.read_text())


def verify_source_tree(repo):
    """Use an isolated original-source tree, never the active new trainer."""
    repo=Path(repo).resolve();expected=protocol()['code_sha256']
    for name,sha in expected.items():
        path=repo/name
        if not path.is_file() or path.is_symlink() or digest(path)!=sha:
            raise ValueError('LEGACY_SOURCE_MISMATCH:'+name)
    return dict(recipe_sha256=digest(PROTOCOL_PATH),source_files=len(expected),actual_gpu_execution=False)


def fp32_loader_options(fsdp_config):
    """A rollout-only worker otherwise defaults to BF16 before vLLM sync."""
    value=dict(fsdp_config or {})
    value['model_dtype']='fp32'
    value['mixed_precision']=dict(param_dtype='fp32',reduce_dtype='fp32',buffer_dtype='fp32')
    return dict(fsdp_config=value,optim_config=None,use_remove_padding=False,
                enable_gradient_checkpointing=False)


def verify_loaded_workers(rows):
    if len(rows)!=8 or sorted(row['rank'] for row in rows)!=list(range(8)):
        raise ValueError('LEGACY_EIGHT_ACTUAL_WORKERS_REQUIRED')
    if len({row['cuda_uuid'] for row in rows})!=8:
        raise ValueError('LEGACY_EIGHT_DISTINCT_UUIDS_REQUIRED')
    for row in rows:
        if row.get('world_size')!=8 or row.get('hf_parameter_dtypes')!=['torch.float32'] or \
                row.get('vllm_parameter_dtypes')!=['torch.float32'] or row.get('hf_parameter_numel',0)<=0:
            raise ValueError('LEGACY_ACTUAL_FP32_REQUIRED')
        if row.get('optimizer_constructed') is not False or row.get('optimizer_steps')!=0:
            raise ValueError('LEGACY_EVALUATION_MUST_NOT_TRAIN')
        if row.get('recipe_sha256')!=digest(PROTOCOL_PATH):
            raise ValueError('LEGACY_WORKER_RECIPE_MISMATCH')
    return True


def occurrence_sha256(question_ids,data_sources):
    rows=[[i,q,s] for i,(q,s) in enumerate(zip(question_ids,data_sources))]
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def verify_result(result,split,checkpoint_step,question_ids,data_sources):
    spec=protocol()
    if type(checkpoint_step) is not int or checkpoint_step not in spec['checkpoints']['admitted_steps'] or split not in ('validation','final'):
        raise ValueError('LEGACY_STEP_OR_SPLIT_INVALID')
    if split=='final' and checkpoint_step!=200:
        raise ValueError('LEGACY_FINAL_REQUIRES200')
    expected=spec['datasets'][split]
    if result.get('recipe_sha256')!=digest(PROTOCOL_PATH) or result.get('recipe_id')!=spec['recipe_id']:
        raise ValueError('LEGACY_RESULT_RECIPE_MISSING_OR_CHANGED')
    if result.get('step')!=checkpoint_step or result.get('split')!=split or \
            result.get('qa_count')!=expected['qa_count'] or result.get('question_count')!=expected['qa_count'] or \
            result.get('validation_sha256')!=expected['sha256']:
        raise ValueError('LEGACY_RESULT_DATASET_OR_COUNT_MISMATCH')
    if len(question_ids)!=expected['qa_count'] or len(data_sources)!=len(question_ids) or result.get('canonical_occurrence_order_verified') is not True or result.get('occurrence_sha256')!=occurrence_sha256(question_ids,data_sources):
        raise ValueError('LEGACY_ORDER_NOT_VERIFIED')
    if split=='validation' and result.get('source_counts')!=expected['source_counts']:
        raise ValueError('LEGACY_VALIDATION_SOURCE_COUNTS_CHANGED')
    counts=result.get('source_counts',{})
    if counts!=dict(Counter(data_sources)) or sum(counts.values())!=expected['qa_count'] or (split=='final' and len(counts)!=7):
        raise ValueError('LEGACY_SOURCE_COUNTS_INCOMPLETE')
    return True
