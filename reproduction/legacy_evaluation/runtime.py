"""CPU binding/authentication for an isolated old-recipe evaluator."""
import copy
import json
from pathlib import Path
import re
from reproduction.legacy_evaluation.contract import PROTOCOL_PATH,digest,protocol,verify_source_tree

ROLE='strict_legacy_checkpoint_evaluation'


LEGACY_RESOLVED_CONFIG_SHA='12e4b148cfbfbb8baa67baf80aa0f3796ac93f6c33cb056b1e8e9f155489104e'


def reference_configuration(document):
    if 'resolved_config' in document:
        if document.get('run_id')!=protocol()['protocol_authoritative_source']['historical_run']:
            raise ValueError('LEGACY_REFERENCE_CONFIG_RUN_CHANGED')
        document=document['resolved_config']
    canonical=json.dumps(document,sort_keys=True,separators=(',',':')).encode()
    if __import__('hashlib').sha256(canonical).hexdigest()!=LEGACY_RESOLVED_CONFIG_SHA:
        raise ValueError('LEGACY_REFERENCE_RESOLVED_CONFIG_NOT_BYTE_BOUND')
    return document


def checkpoint_hf_names(checkpoint):
    checkpoint=Path(checkpoint)
    index=json.loads((checkpoint/'model.safetensors.index.json').read_text())
    names=set(index['weight_map'].values())|{'model.safetensors.index.json','config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','generation_config.json'}
    names|={name for name in ('added_tokens.json','special_tokens_map.json','tokenizer.model') if (checkpoint/name).is_file()}
    if any(Path(name).name!=name for name in names):raise ValueError('LEGACY_HF_INDEX_PATH_INVALID')
    return names


def validate_binding(binding,config_path,*,verify_weights=True):
    spec=protocol();step=binding.get('checkpoint_step');split=binding.get('split')
    if binding.get('role')!=ROLE or type(step) is not int or step not in spec['checkpoints']['admitted_steps'] or \
            split not in ('validation','final') or (split=='final' and step!=200):
        raise ValueError('LEGACY_BINDING_STEP_SPLIT_ROLE')
    if binding.get('recipe_sha256')!=digest(PROTOCOL_PATH) or binding.get('recipe_id')!=spec['recipe_id']:
        raise ValueError('LEGACY_BINDING_RECIPE')
    if binding.get('legacy_source_commit')!=spec['protocol_authoritative_source']['source_commit'] or binding.get('legacy_source_sha256')!=spec['code_sha256']:
        raise ValueError('LEGACY_BINDING_SOURCE')
    verify_source_tree(binding['legacy_source_root'])
    if not re.fullmatch('[0-9a-f]{40}',binding.get('control_source_commit','')):
        raise ValueError('LEGACY_CONTROL_COMMIT_REQUIRED')
    control=Path(__file__).resolve().parents[2]
    required={'legacy_eval_control/__init__.py','legacy_eval_control/worker.py','legacy_eval_control/entry.py'} | {
        'reproduction/legacy_evaluation/'+name for name in ('runner.py','runtime.py','journal.py','pipeline.py','contract.py','protocol.json')}
    mapping=binding.get('control_source_sha256',{})
    if not required<=set(mapping):raise ValueError('LEGACY_CONTROL_SOURCE_MAP_INCOMPLETE')
    for name,sha in mapping.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or not (control/rel).is_file() or (control/rel).is_symlink() or digest(control/rel)!=sha:
            raise ValueError('LEGACY_CONTROL_SOURCE_MISMATCH:'+name)
    cp=Path(binding['checkpoint'])
    if not cp.is_absolute() or cp.resolve()!=cp or not cp.is_dir():raise ValueError('LEGACY_CHECKPOINT_PATH')
    if binding.get('validation_sha256')!=spec['datasets'][split]['sha256']:
        raise ValueError('LEGACY_BINDING_DATASET')
    if digest(config_path)!=binding['resolved_config_sha256']:
        raise ValueError('LEGACY_REFERENCE_CONFIG_SHA')
    reference_configuration(json.loads(Path(config_path).read_text()))
    if digest(binding['checkpoint_manifest'])!=binding['checkpoint_manifest_sha256']:
        raise ValueError('LEGACY_CHECKPOINT_MANIFEST_SHA')
    manifest=json.loads(Path(binding['checkpoint_manifest']).read_text())
    if step==0:
        if manifest.get('repo')!='Qwen/Qwen2.5-3B-Instruct' or manifest.get('revision')!='aa8e72537993ba99e69dfaafa59ed015b17504d1':
            raise ValueError('LEGACY_OFFICIAL_ZERO_MANIFEST')
        certified={r['file']:r['sha256'] for r in manifest['files']}
    else:
        if manifest.get('run_id')!=binding['run_id'] or manifest.get('step')!=step or Path(manifest.get('checkpoint','')).resolve()!=cp or manifest.get('status')!='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS':
            raise ValueError('LEGACY_SAME_EXPERIMENT_COMPLETE_RECEIPT')
        certified={r['relative_path']:r['full_read_sha256'] for r in manifest['files'] if '/' not in r['relative_path']}
    names=checkpoint_hf_names(cp)
    hashes=binding.get('checkpoint_files_sha256',{})
    if set(hashes)!=names or any(certified.get(n)!=sha for n,sha in hashes.items()):
        raise ValueError('LEGACY_HF_FILES_NOT_CERTIFIED')
    for name,sha in hashes.items():
        if Path(name).name!=name or not (cp/name).is_file() or (cp/name).is_symlink() or (verify_weights and digest(cp/name)!=sha):
            raise ValueError('LEGACY_CHECKPOINT_FILE_SHA:'+name)
    if hashes['generation_config.json']!=binding['generation_config_sha256']:
        raise ValueError('LEGACY_GENERATION_FILE_BINDING')
    official=Path(binding['official_model']);original=official/'generation_config.json'
    if digest(original)!=spec['sampling']['official_generation_config_sha256']:
        raise ValueError('LEGACY_ORIGINAL_GENERATION_SHA')
    a=json.loads(original.read_text());b=json.loads((cp/'generation_config.json').read_text())
    for d in (a,b):d.pop('transformers_version',None)
    if a!=b:raise ValueError('LEGACY_GENERATION_FIELDS_CHANGED')
    return dict(recipe_sha256=digest(PROTOCOL_PATH),checkpoint_files_verified=len(names) if verify_weights else 0,
        checkpoint_manifest_bindings_verified=len(names),actual_gpu_execution=False)


def legacy_config(original,checkpoint,retrieval_url):
    """Rebuild only from the actual old3364 resolved config, never new config."""
    from omegaconf import OmegaConf
    config=OmegaConf.create(copy.deepcopy(original));OmegaConf.set_struct(config,False)
    spec=protocol()
    reference_configuration(original)
    config.actor_rollout_ref.model.path=str(Path(checkpoint).resolve())
    config.actor_rollout_ref.model.pop('actor_checkpoint_path',None)
    config.actor_rollout_ref.model.use_remove_padding=False
    config.actor_rollout_ref.model.enable_gradient_checkpointing=False
    config.trainer.pop('full_state_restore_receipt',None);config.trainer.pop('weight_continuation_step',None)
    rollout=config.actor_rollout_ref.rollout
    if rollout.get('async_phase_enabled',False):raise ValueError('LEGACY_REFERENCE_CONFIG_IS_ASYNC')
    if rollout.dtype!='float32' or rollout.gpu_memory_utilization!=0.1 or config.data.val_batch_size!=64:
        raise ValueError('LEGACY_REFERENCE_CONFIG_NOT_OLD3364')
    if [config.data[k] for k in ('max_start_length','max_prompt_length','max_response_length','max_obs_length')]!=[2048,4096,500,500] or config.max_turns!=4:
        raise ValueError('LEGACY_REFERENCE_CONTEXT_CHANGED')
    if config.trainer.n_gpus_per_node!=8 or config.trainer.nnodes!=1 or rollout.tensor_model_parallel_size!=1:
        raise ValueError('LEGACY_EIGHT_TP1_REQUIRED')
    if rollout.n!=1 or rollout.temperature!=1 or rollout.top_p!=1 or rollout.top_k!=-1 or rollout.response_length!=500 or not config.do_search or config.retriever.topk!=3:
        raise ValueError('LEGACY_REFERENCE_SAMPLING_CHANGED')
    config.retriever.url=retrieval_url
    return config
