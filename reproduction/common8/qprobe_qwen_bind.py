"""CPU-only adoption of this experiment's completed native Qwen warmup10.

Authenticate the pinned actual warmup, original teacher and full token bank,
then bind a pristine official-actor formal200 run. Never fit, sample or launch.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

from reproduction.common8.qprobe_token_bind import _json, _once, _fresh_run
from reproduction.common8.qprobe_configuration import arguments
from search_g1.qprobe_features import COMMON_MANIFEST_SHA, OFFICIAL_REVISION, sha_file
from search_g1.qprobe_token_features import TokenFeatureCache, remote_path


WARMUP = Path('/data/shaoshitong/Search-G1-qprobe-qwen-lastblock-warmup10-096c8297cba74f8089063e763686671e/warmup')
WARMUP_SOURCE = 'e81445905bf8b15dbb0bdec3afa29d99ab0b11d8'
WARMUP_SOURCE_MAP = '006912cc7817b2ae8abde32a85c42d21b2c1976bf0b5618066c6aa9d2f0674ad'
WARMUP_RUNTIME_FILES = 490
NATIVE_MODEL_SHA = '69ba7c7437e01ca21e96266ce515d3998717669342df8ef6f578dcc73666730e'
OFFICIAL_MODEL = '/data/search-g1/assets/qwen2.5-3b-instruct'
TEACHER = Path('/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v4-sigma3-fullouter')
TEACHER_SHA = '88998de7a30455b9135b02cc3c82251fbc1ba1765708244538b6a66c1371e871'
TEACHER_COMPLETE_SHA = '07f4e2e3937c814637b783468ff1062af7ba701e1ac1e136da2dd56f838b45b1'
TOKEN_SPEC_SHA = 'fcc9d61db156ddbb4e96dcd6c9437797b978dbbfc9774145a28e0c761de1a9e7'
NATIVE_CLASS = 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'


def _sha(value, name):
    if not isinstance(value,str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('QWEN_BIND_INVALID_SHA: '+name)
    return value


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def recipe(cache_dir):
    return dict(head_kind='qwen_native_lastblock_binary_v1',loss_mode='token_binary_std_v1',
        ranking='binary_mixed_probability',candidate_questions=5120,selected_questions=512,
        responses_per_question=5,exploration_questions=0,seed=20261013,head_seed=1729,
        cache_dir=str(cache_dir),max_tokens=244,feature_dim=2048,head_hidden_size=2048,hidden_size=2048,
        num_attention_heads=16,num_key_value_heads=2,intermediate_size=11008,rms_norm_eps=1e-6,
        rope_theta=1e6,max_position_embeddings=32768,source_layer=35,num_hidden_layers=36,
        attention_implementation='sdpa',micro_batch_size=32,head_device='cuda:0',
        head_learning_rate=3e-4,head_batch_size=256,head_epochs_per_outer=2,bootstrap_head_epochs=10,
        bootstrap_max_groups_per_outer=5120,mean_loss_weight=0.,std_loss_weight=.3,count_aux_loss_weight=0.,
        target_statistic='sample_std',replay_window_outer=8,teacher_replay_excluded=True,
        teacher_warmup_cache=str(TEACHER))


def _block_shapes():
    return {'input_layernorm.weight':[2048],'post_attention_layernorm.weight':[2048],
        'self_attn.q_proj.weight':[2048,2048],'self_attn.q_proj.bias':[2048],
        'self_attn.k_proj.weight':[256,2048],'self_attn.k_proj.bias':[256],
        'self_attn.v_proj.weight':[256,2048],'self_attn.v_proj.bias':[256],
        'self_attn.o_proj.weight':[2048,2048],'mlp.gate_proj.weight':[11008,2048],
        'mlp.up_proj.weight':[11008,2048],'mlp.down_proj.weight':[2048,11008]}


def validate_ready_metadata(ready, prepared, publication, run):
    """Pure metadata checks; a missing actual ready is never manufactured."""
    if (ready.get('stage') != 'QWEN_LAST_BLOCK_FRESH_HEAD_WARMUP10_COMPLETE' or
            ready.get('source_commit') != WARMUP_SOURCE or ready.get('source_map_sha256') != WARMUP_SOURCE_MAP or
            publication.get('source_commit') != WARMUP_SOURCE or publication.get('remote_commit') != WARMUP_SOURCE or
            publication.get('source_map_sha256') != WARMUP_SOURCE_MAP or publication.get('runtime_file_count') != WARMUP_RUNTIME_FILES or
            publication.get('remote_fresh_object_store_disk_git_hash_equal') is not True or
            prepared.get('stage') != 'QWEN_LAST_BLOCK_WARMUP10_CPU_PREPARED' or
            prepared.get('source_map_sha256') != WARMUP_SOURCE_MAP or
            prepared.get('source_sha256') != publication.get('source_sha256') or
            len(prepared.get('source_sha256',{})) != WARMUP_RUNTIME_FILES or
            hashlib.sha256(_canonical(prepared['source_sha256'])).hexdigest() != WARMUP_SOURCE_MAP or
            prepared['source_sha256'].get('search_g1/qprobe_qwen_block_model.py') != NATIVE_MODEL_SHA):
        raise ValueError('QWEN_BIND_WARMUP_NOT_THIS_PUBLISHED_NATIVE_EXPERIMENT')
    expected = dict(questions=5120,teacher_trajectories_reused=25600,trajectories_generated=0,
        block_index=35,global_max_tokens=244,token_feature_spec_sha256=TOKEN_SPEC_SHA,
        head_optimizer_updates=200,head_adam_steps=200,epoch_records=11,minibatch_records=200,
        old_head_inherited=False,encoder_reencoded=False,no_reencode=True,
        encoder_loaded_in_this_process=False,trajectories_regenerated=False,formal_training_started=False)
    if any(ready.get(key) != value or type(ready.get(key)) is not type(value) for key,value in expected.items()):
        raise ValueError('QWEN_BIND_NATIVE_WARMUP_INCOMPLETE_OR_INHERITED')
    if (ready.get('feature_cache_dir') != prepared.get('feature_cache_dir') or
            ready.get('token_feature_manifest_sha256') != prepared.get('token_feature_manifest_sha256')):
        raise ValueError('QWEN_BIND_WARMUP_TOKEN_CACHE_CHANGED')
    task = ready.get('task_config',{})
    required_task = dict(source_commit=WARMUP_SOURCE,teacher_identity_sha256=TEACHER_SHA,
        epochs=10,batch_size=256,micro_batch_size=32,learning_rate=3e-4,auxiliary_weight=.3,
        target_statistic='sample_std',head_seed=1729,top_k=512,exploration_questions=0,block_index=35,
        architecture='native_transformers_Qwen2DecoderLayer',native_attention_implementation='sdpa',global_max_tokens=244)
    if any(task.get(key) != value for key,value in required_task.items()) or task != prepared.get('task_config'):
        raise ValueError('QWEN_BIND_NATIVE_WARMUP_RECIPE_CHANGED')
    replay = ready.get('replay_contract',{})
    if (replay.get('window_completed_formal_outers') != 8 or replay.get('teacher_warmup_in_replay') is not False or
            replay.get('ages_included') != list(range(8)) or replay.get('normalize_weights') is not False or
            replay.get('weight_formula') != '2*2**(-age) if 0 <= age <= 7 else 0' or
            replay.get('reduction') != 'mean(weight * per_group_loss)'):
        raise ValueError('QWEN_BIND_FORMAL_REPLAY_OBJECTIVE_CHANGED')
    block = ready.get('native_block_proof',{})
    implementation = ready.get('native_implementation',{})
    if (block.get('native_class') != NATIVE_CLASS or block.get('block_index') != 35 or
            block.get('official_revision') != OFFICIAL_REVISION or
            any(block.get(key) is not True for key in ('source_tensors_verified','official_block_copy_verified','head_initialized','fresh_linear','fresh_adam')) or
            not block.get('transformers_version') or implementation != prepared.get('native_implementation') or
            implementation.get('transformers_version') != block['transformers_version'] or
            implementation.get('attention_implementation') != 'sdpa' or
            implementation.get('implementations',{}).get('decoder_layer',{}).get('qualified_name') != NATIVE_CLASS):
        raise ValueError('QWEN_BIND_ACTUAL_NATIVE_IMPLEMENTATION_OR_COPY_MISSING')
    config = prepared.get('official_model_config',{})
    expected_config = dict(model_type='qwen2',hidden_size=2048,num_hidden_layers=36,num_attention_heads=16,
        num_key_value_heads=2,intermediate_size=11008,rms_norm_eps=1e-6,rope_theta=1e6,
        max_position_embeddings=32768,hidden_act='silu',attention_dropout=0.)
    if (any(config.get(key) != value for key,value in expected_config.items()) or
            config.get('use_sliding_window',False) is not False or
            ready.get('official_model_config_sha256') != run['initial_model_sha256'].get('config.json') or
            ready['official_model_config_sha256'] != prepared.get('official_model_config_sha256')):
        raise ValueError('QWEN_BIND_OFFICIAL_NATIVE_ARCHITECTURE_CHANGED')
    original,copied = ready.get('official_block_source',{}),ready.get('initial_block_proof',{})
    shapes = _block_shapes()
    if (original != prepared.get('official_block_source') or original.get('source_layer') != 35 or
            original.get('tensor_count') != 12 or copied.get('source_layer') != 35 or copied.get('tensor_count') != 12 or
            copied.get('all_copied_parameters_trainable') is not True or
            set(original.get('tensors',{})) != set(shapes) or set(copied.get('tensors',{})) != set(shapes) or
            original.get('tensor_index_sha256') != run['initial_model_sha256'].get('model.safetensors.index.json')):
        raise ValueError('QWEN_BIND_OFFICIAL_LASTBLOCK_PROOF_CHANGED')
    for name,shape in shapes.items():
        source,target = original['tensors'][name],copied['tensors'][name]
        if (source.get('source_key') != 'model.layers.35.'+name or source.get('shape') != shape or
                source.get('source_dtype') != 'bfloat16' or target.get('exact_cast_equal') is not True or
                any(target.get(key) != source.get(key) for key in ('source_key','shape','source_dtype','source_sha256')) or
                source.get('bytes') != math.prod(shape)*2 or
                original.get('official_shards',{}).get(source.get('shard'),{}).get('sha256') !=
                    run['initial_model_sha256'].get(source.get('shard'))):
            raise ValueError('QWEN_BIND_LASTBLOCK_PARAMETER_COPY_CHANGED: '+name)
        _sha(source.get('source_sha256'),'official_tensor'); _sha(target.get('copied_fp32_sha256'),'copied_fp32_tensor')
    readout = ready.get('initial_readout_proof',{})
    if readout.get('fresh') is not True or readout.get('seed') != 1729:
        raise ValueError('QWEN_BIND_READOUT_NOT_FRESH')
    for key in ('weight_sha256','bias_sha256'): _sha(readout.get(key),key)
    checks = ready.get('cpu_checks',{})
    if (checks != prepared.get('cpu_checks') or checks.get('passed') is not True or
            checks.get('tests_run',0) <= 0 or checks.get('skipped') != 0 or checks.get('failures') != 0 or
            checks.get('errors') != 0 or checks.get('cuda_initialized') is not False):
        raise ValueError('QWEN_BIND_NATIVE_CPU_FIXTURES_NOT_PASSED')
    for key in ('head_sha256','metrics_sha256','prepared_sha256','token_feature_manifest_sha256'):
        _sha(ready.get(key),key)
    return ready


def _histories(metrics):
    if (metrics.get('fit_epochs') != 10 or metrics.get('training_groups') != 5120 or
            metrics.get('actual_head_adam_updates') != 200 or metrics.get('expected_head_adam_updates') != 200 or
            len(metrics.get('epoch_history',[])) != 11 or len(metrics.get('minibatch_history',[])) != 200):
        raise ValueError('QWEN_BIND_ACTUAL_WARMUP_HISTORY_INCOMPLETE')
    for epoch,row in enumerate(metrics['epoch_history']):
        if row.get('epoch') != epoch or row.get('actual_head_adam_updates') != 20*epoch:
            raise ValueError('QWEN_BIND_EPOCH_COUNTER_CHANGED')
    for ordinal,row in enumerate(metrics['minibatch_history'],1):
        if (row.get('head_update_ordinal') != ordinal or row.get('head_adam_steps_before') != ordinal-1 or
                row.get('head_adam_steps_after') != ordinal or row.get('epoch') != (ordinal-1)//20+1 or
                row.get('minibatch_index') != (ordinal-1)%20 or row.get('batch_size') != 256):
            raise ValueError('QWEN_BIND_MINIBATCH_COUNTER_CHANGED')
    for row in metrics['epoch_history']+metrics['minibatch_history']:
        keys = ('binary_bce','regression_mse_raw','regression_mse_normalized','regression_contribution','binary_accuracy')
        if any(type(row.get(key)) not in (int,float) or not math.isfinite(row[key]) for key in keys):
            raise ValueError('QWEN_BIND_NONFINITE_OR_MISSING_ACTUAL_METRICS')
        total = row.get('total_loss',row.get('pre_step_loss'))
        if (type(total) not in (int,float) or not math.isfinite(total) or not 0 <= row['binary_accuracy'] <= 1 or
                not math.isclose(total,row['binary_bce']+.3*row['regression_mse_normalized'],abs_tol=2e-6,rel_tol=1e-6) or
                not math.isclose(row['regression_mse_raw'],.3*row['regression_mse_normalized'],abs_tol=2e-6,rel_tol=1e-6) or
                not math.isclose(row['regression_contribution'],.3*row['regression_mse_normalized'],abs_tol=2e-6,rel_tol=1e-6)):
            raise ValueError('QWEN_BIND_ACTUAL_LOSS_OR_ACCURACY_CHANGED')


def _read_warmup(ready_path,ready_sha,run):
    if Path(ready_path) != WARMUP/'ready.json':
        raise ValueError('QWEN_BIND_REQUIRES_THIS_NATIVE_WARMUP_READY_PATH')
    _sha(ready_sha,'actual_ready_sha256')
    if sha_file(ready_path) != ready_sha: raise ValueError('QWEN_BIND_ACTUAL_READY_SHA_CHANGED')
    ready = _json(ready_path)
    prepared = _json(WARMUP/'prepared.json')
    publication = _json(WARMUP.parent/'preparation/source-publication.json')
    validate_ready_metadata(ready,prepared,publication,run)
    for name,key in (('prepared.json','prepared_sha256'),('fresh-head.pt','head_sha256'),('warmup-metrics.json','metrics_sha256')):
        if sha_file(WARMUP/name) != ready[key]: raise ValueError('QWEN_BIND_ACTUAL_WARMUP_FILE_CHANGED: '+name)
    if sha_file(WARMUP/'cpu-checks.log') != ready['cpu_checks']['log_sha256']:
        raise ValueError('QWEN_BIND_NATIVE_CPU_FIXTURE_LOG_CHANGED')
    _histories(_json(WARMUP/'warmup-metrics.json'))
    from reproduction.common8.source_files import runtime_files
    repo = WARMUP.parent/'repo'
    actual = {str(path.relative_to(repo)):sha_file(path) for path in runtime_files(repo)}
    if actual != prepared['source_sha256']: raise ValueError('QWEN_BIND_FROZEN_WARMUP_RUNTIME_CHANGED')
    return ready,prepared


def _teacher_labels(run,cache,retriever_version,resolved,ready,prepared):
    from search_g1.qprobe_warmup import TeacherCache
    from search_g1.qprobe_runtime import strict_scoring_sources
    if sha_file(TEACHER/'identity.json') != TEACHER_SHA or sha_file(TEACHER/'complete.json') != TEACHER_COMPLETE_SHA:
        raise ValueError('QWEN_BIND_ORIGINAL_TEACHER_IDENTITY_CHANGED')
    identity = _json(TEACHER/'identity.json')
    provenance = prepared['label_producer_provenance']
    if (provenance.get('teacher_identity_sha256') != TEACHER_SHA or provenance.get('teacher_complete_sha256') != TEACHER_COMPLETE_SHA or
            identity['feature_spec_sha'] != provenance.get('original_feature_spec_sha256') or
            identity['model_revision'] != run['model_revision'] or identity['model_files_sha256'] != run['initial_model_sha256'] or
            identity['train_parquet_sha256'] != run['input_sha256']['train.parquet'] or
            identity['data_manifest_sha256'] != COMMON_MANIFEST_SHA or identity['retriever_version'] != retriever_version or
            identity['sampling_config'] != resolved['actor_rollout_ref']['rollout'] or
            identity['agent_config'] != {key:resolved['data'][key] for key in ('max_start_length','max_prompt_length','max_response_length','max_obs_length')} or
            identity['max_turns'] != resolved['max_turns'] or identity['training_world_size'] != 4 or
            identity['scorer_sources'] != strict_scoring_sources()):
        raise ValueError('QWEN_BIND_TEACHER_MODEL_DATA_RETRIEVAL_OR_GENERATION_CHANGED')
    for name,digest in dict(identity['scorer_sources'],**identity['rollout_sources']).items():
        if run['source_sha256'].get(name) != digest: raise ValueError('QWEN_BIND_TEACHER_SCIENCE_CHANGED: '+name)
    teacher = TeacherCache.__new__(TeacherCache); teacher.root,teacher.identity = TEACHER,identity
    with (TEACHER/'writer.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        proof,groups,_ = teacher.verify_complete()
    if proof != _json(TEACHER/'complete.json'): raise ValueError('QWEN_BIND_TEACHER_COMPLETE_PROOF_CHANGED')
    labels = [dict(row,complete=True,selection_source='bootstrap_all_candidate',**proof['protocol']) for row in groups]
    qids = [row['qid'] for row in labels]
    if len(qids) != len(set(qids)) or len(qids) != 5120:
        raise ValueError('QWEN_BIND_REQUIRES_ALL_5120_TEACHER_LABELS')
    if hashlib.sha256(_canonical(qids)).hexdigest() != prepared['ordered_warmup_qids_sha256']:
        raise ValueError('QWEN_BIND_TEACHER_ORDER_DIFFERS_FROM_ACTUAL_WARMUP')
    warmup_cache = TokenFeatureCache(remote_path(ready['feature_cache_dir']),expected_spec_sha=TOKEN_SPEC_SHA)
    if sha_file(warmup_cache.path/'manifest.json') != ready['token_feature_manifest_sha256'] or warmup_cache.ids != qids:
        raise ValueError('QWEN_BIND_WARMUP_TOKEN_ROW_ORDER_CHANGED')
    for row in labels:
        if row['source_outer'] != 0 or cache.index[row['qid']]['question_hash'] != row['question_hash']:
            raise ValueError('QWEN_BIND_TEACHER_QUESTION_TOKEN_BINDING_CHANGED')
        if any(cache.index[row['qid']].get(key) != warmup_cache.index[row['qid']].get(key)
               for key in ('question_hash','length','token_ids_sha256','feature_sha256')):
            raise ValueError('QWEN_BIND_FULL_BANK_DID_NOT_REUSE_BYTE_EXACT_WARMUP_TOKENS')
    return labels,qids


def _resolved_prospective(base,run,phase):
    from reproduction.common8 import configuration
    from reproduction.common8.formal_contract import verify_formal_config
    from search_g1.rollout_phase_contract import hydra_arguments
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    _,overrides = configuration.arguments(base)
    if run.get('infra_rollout') is not None and run['infra_rollout'] != phase:
        raise ValueError('QWEN_BIND_PREPARED_ROLLOUT_PHASE_CHANGED')
    prospective = dict(run,infra_rollout=phase)
    if run.get('infra_rollout') is None: overrides += hydra_arguments(prospective)
    with initialize_config_dir(config_dir=str(configuration.REPO/'verl/trainer/config'),version_base=None):
        config = compose(config_name='ppo_trainer',overrides=overrides)
    OmegaConf.resolve(config)
    resolved = OmegaConf.to_container(config,resolve=True)
    verify_formal_config(resolved,prospective,base)
    return resolved


def _verify_token_bank(base,run,cache):
    intent_path=base/'preparation/intent.json'
    intent=_json(intent_path) if intent_path.exists() else {}
    if intent.get('qualified_bank_origin') is None:
        cache.verify_all()
        return dict(reused_qualified_bank=False,full_file_sha_verified_in_this_bind=True)
    from reproduction.a800_common.qprobe_qwen_formal_ops import validate_intent,required_bank
    if (intent['base']!=str(base) or intent['source_commit']!=run['source_commit'] or
            intent['source_sha256']!=run['source_sha256'] or cache.path!=Path(intent['bank_base'])/'token-features'):
        raise ValueError('QWEN_BIND_QUALIFIED_BANK_TARGET_CHANGED')
    validate_intent(intent)
    admitted=required_bank(base,intent)
    if (admitted.get('reused_qualified_bank') is not True or admitted.get('original_full_sha_seal_revalidated') is not True or
            admitted.get('bank_reencoded') is not False or admitted.get('packed_features_rehashed') is not False or
            admitted.get('actor_or_head_updates_inherited') is not False or admitted.get('formal_completed_outers')!=0 or
            type(admitted.get('formal_completed_outers')) is not int or admitted.get('source_commit')!=run['source_commit'] or
            admitted['cache_dir']!=str(cache.path) or admitted['feature_spec_sha256']!=TOKEN_SPEC_SHA or
            sha_file(cache.path/'manifest.json')!=admitted['feature_manifest_sha256'] or
            cache.manifest!=_json(cache.path/'manifest.json')):
        raise ValueError('QWEN_BIND_QUALIFIED_BANK_SEAL_NOT_ADMITTED')
    return dict(reused_qualified_bank=True,original_full_sha_seal_revalidated=True,
        full_file_sha_verified_in_this_bind=False,packed_features_rehashed=False,
        feature_manifest_sha256=admitted['feature_manifest_sha256'],complete_sha256=admitted['complete_sha256'],
        qualified_bank_origin=copy.deepcopy(admitted['qualified_bank_origin']))


def bind(base,cache_dir,*,warmup_ready,warmup_ready_sha256,retriever_version,infra_rollout):
    import torch
    from search_g1.qprobe_qwen_block_online import OnlineQwenBlockQProbe,QwenBlockOnlineConfig
    from search_g1.rollout_phase_contract import validate as validate_rollout
    if torch.cuda.is_initialized(): raise ValueError('QWEN_BIND_MUST_RUN_WITHOUT_INITIALIZED_CUDA')
    base,cache_dir = remote_path(base),remote_path(cache_dir)
    if Path(__file__).resolve().parents[2] != base/'repo':
        raise ValueError('QWEN_BIND_MUST_EXECUTE_THE_TARGET_BOUND_SOURCE')
    run = _fresh_run(base)
    if (run['initial_model'] != OFFICIAL_MODEL or run['reference_model'] != OFFICIAL_MODEL or
            run['model_revision'] != OFFICIAL_REVISION or run['target_updates'] != 200 or
            run['source_sha256'].get('search_g1/qprobe_qwen_block_model.py') != NATIVE_MODEL_SHA or
            any(run.get(key) for key in ('actor_checkpoint_path','resume_from_path','initial_actor_checkpoint','restored_from'))):
        raise ValueError('QWEN_BIND_REQUIRES_FRESH_OFFICIAL_ACTOR_200')
    ready,prepared = _read_warmup(warmup_ready,warmup_ready_sha256,run)
    cache = TokenFeatureCache(cache_dir,expected_spec_sha=TOKEN_SPEC_SHA)
    bank_verification = _verify_token_bank(base,run,cache)
    if (cache.manifest['count'] != 169028 or cache.spec['global_max_tokens'] != 244 or
            cache.manifest['total_tokens'] != 18367021 or cache.spec['train_parquet_sha256'] != run['input_sha256']['train.parquet'] or
            cache.spec['encoder_revision'] != OFFICIAL_REVISION):
        raise ValueError('QWEN_BIND_TOKEN_BANK_NOT_FULL_CANONICAL_TRAIN')
    phase = validate_rollout(infra_rollout,4)
    resolved = _resolved_prospective(base,run,phase)
    if (resolved['actor_rollout_ref']['model'].get('actor_checkpoint_path') is not None or
            resolved['trainer'].get('full_state_restore_receipt') is not None):
        raise ValueError('QWEN_BIND_CANNOT_RESTORE_ACTOR_OR_OPTIMIZER')
    labels,qids = _teacher_labels(run,cache,retriever_version,resolved,ready,prepared)
    settings = recipe(cache_dir)
    directory = base/'artifacts/question-probe/native-qwen-warmup-adoption'
    preflight = dict(settings,output_dir=str(base/'artifacts/question-probe'),
        initial_head_checkpoint=str(directory/'initial.pt'),initial_head_checkpoint_sha256='0'*64,
        initial_head_ready=str(directory/'ready.json'),initial_head_ready_sha256='0'*64)
    arguments(dict(run,question_probe=preflight),base)
    _once(base/'logs/qprobe-bind-claim.json',dict(run_id=run['run_id'],source_commit=run['source_commit'],
        actual_warmup_source=WARMUP_SOURCE,actual_warmup_ready_sha256=warmup_ready_sha256,
        actual_warmup_sha256=ready['head_sha256'],claimed_unix=time.time(),no_automatic_retry=True))
    directory.mkdir(parents=True,exist_ok=False)
    provenance = dict(actor_snapshot_id=run['run_id']+':0',warmup_ready=str(warmup_ready),warmup_ready_sha256=warmup_ready_sha256,
        warmup_source_commit=WARMUP_SOURCE,warmup_source_map_sha256=WARMUP_SOURCE_MAP,
        teacher_identity_sha256=TEACHER_SHA,teacher_complete_sha256=TEACHER_COMPLETE_SHA)
    head = OnlineQwenBlockQProbe.adopt_warmup(WARMUP/'fresh-head.pt',labels,qids,
        expected_checkpoint_sha=ready['head_sha256'],expected_feature_spec_sha=cache.spec_sha,
        provenance=provenance,config=QwenBlockOnlineConfig.from_mapping(settings),feature_cache=cache,device='cpu')
    if (head.replay_labels or head.replay_features or head.protocol is not None or head.update_count or head.completed_outer or
            head._adam_steps() != 200 or head.initial_block_proof != ready['initial_block_proof'] or
            head.initial_readout_proof != ready['initial_readout_proof'] or head.native_implementation != ready['native_implementation']):
        raise ValueError('QWEN_BIND_ADOPTED_HEAD_OR_EMPTY_FORMAL_REPLAY_CHANGED')
    head_path = directory/'initial.pt'; receipt = head.save(head_path,actor_snapshot_id=run['run_id']+':0')
    loaded = OnlineQwenBlockQProbe.load(head_path,expected_actor_snapshot_id=run['run_id']+':0',
        expected_feature_spec_sha=cache.spec_sha,expected_checkpoint_sha=receipt['sha256'],device='cpu')
    if (loaded.version != head.version or loaded._adam_steps() != 200 or loaded.replay_labels or
            not torch.equal(loaded.rng.get_state(),head.rng.get_state())):
        raise ValueError('QWEN_BIND_SAVED_INITIAL_HEAD_READBACK_CHANGED')
    loaded.verify_native_provenance()
    if torch.cuda.is_initialized(): raise ValueError('QWEN_BIND_CPU_ADOPTION_INITIALIZED_CUDA')
    adopted = dict(stage='QWEN_NATIVE_FORMAL_INITIAL_HEAD_ADOPTED',run_id=run['run_id'],source_commit=run['source_commit'],
        head_receipt=receipt,raw_warmup_checkpoint_sha256=ready['head_sha256'],feature_spec_sha=cache.spec_sha,
        actual_head_adam_updates=200,completed_outer=0,warmup_questions=5120,teacher_trajectories_reused=25600,
        trajectories_regenerated=False,no_head_refit=True,teacher_replay_excluded=True,
        native_block_proof=copy.deepcopy(ready['native_block_proof']),actor_optimizer_inherited=False,actor_checkpoint_inherited=False,
        token_bank_verification=bank_verification,
        initialization_provenance=head.initialization_provenance,
        actual_warmup=dict(ready_path=str(warmup_ready),ready_sha256=warmup_ready_sha256,source_commit=WARMUP_SOURCE,
            source_map_sha256=WARMUP_SOURCE_MAP,prepared_sha256=ready['prepared_sha256'],head_sha256=ready['head_sha256']),
        actual_saved_head_cpu_readback=True,formal_training_started=False,checked_unix=time.time())
    ready_path = directory/'ready.json'; _once(ready_path,adopted)
    settings.update(output_dir=str(base/'artifacts/question-probe'),initial_head_checkpoint=str(head_path),
        initial_head_checkpoint_sha256=receipt['sha256'],initial_head_ready=str(ready_path),initial_head_ready_sha256=sha_file(ready_path))
    proposed = dict(run,question_probe=settings,retriever_version=retriever_version,mixed_first_selection=False,infra_rollout=phase)
    arguments(proposed,base)
    return _publish_bound(base,run,proposed)


def _publish_bound(base,run,proposed):
    if _json(base/'logs/run-manifest.json') != run: raise ValueError('QWEN_BIND_MANIFEST_CHANGED_DURING_ADOPTION')
    for name in ('training-launch-claim.json','pipeline-launch-claim.json','actual-attempts.jsonl',
                 'actual-updates.jsonl','resolved-training-config.json','policy-safety-calibration.json'):
        if (base/'logs'/name).exists(): raise ValueError('QWEN_BIND_RUN_CLAIMED_DURING_ADOPTION: '+name)
    from reproduction.common8.source_files import runtime_files
    repo = base/'repo'
    if {str(path.relative_to(repo)):sha_file(path) for path in runtime_files(repo)} != run['source_sha256']:
        raise ValueError('QWEN_BIND_PREPARED_SOURCE_CHANGED_DURING_ADOPTION')
    backup = base/'preparation/question-probe-pre-bind.json'
    with backup.open('xb') as stream:
        stream.write((base/'logs/run-manifest.json').read_bytes()); stream.flush(); os.fsync(stream.fileno())
    from reproduction.a800.owned_processes import publish
    publish(base/'logs/run-manifest.json',proposed)
    _once(base/'logs/qprobe-bound.json',dict(run_id=run['run_id'],source_commit=run['source_commit'],question_probe=proposed['question_probe'],
        initialized_from_this_native_experiment_warmup=True,teacher_replay_excluded=True,
        head_warmup_adam_updates_excluded_from_formal_outers=200,checked_unix=time.time()))
    return proposed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True,type=Path); parser.add_argument('--cache-dir',required=True,type=Path)
    parser.add_argument('--warmup-ready',required=True,type=Path); parser.add_argument('--warmup-ready-sha256',required=True)
    parser.add_argument('--retriever-version',required=True); parser.add_argument('--infra-rollout-json',required=True,type=Path)
    args = parser.parse_args()
    run = bind(args.base,args.cache_dir,warmup_ready=args.warmup_ready,warmup_ready_sha256=args.warmup_ready_sha256,
        retriever_version=args.retriever_version,infra_rollout=_json(remote_path(args.infra_rollout_json)))
    print(json.dumps(dict(run_id=run['run_id'],question_probe=run['question_probe'],formal_training_started=False),allow_nan=False))


if __name__ == '__main__': main()
