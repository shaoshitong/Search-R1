"""CPU-only preparation reusing the accepted immutable CP5 and identity assets.

The live warmup's latest state is deliberately never consulted or loaded.
Only bounded old receipts and full file hashes are read; the previously
accepted finite/CPU-load proof is explicitly inherited for identical bytes.
No GPU inventory, idle check, model load, canonical rescan or launch occurs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .a800_prepare import (PREFIX, stable_bytes, checked_json, full_sha, once,
                           process_identity, signature, remote)
from .contract import (WORLD, PREFIX_TOKENS, HIDDEN_CAPTURE_CONTRACT, QMAX, WIDTH, DIM,
                       CANDIDATE_QUESTIONS, SELECTED_QUESTIONS, G, FINAL_OUTER,
                       PROBE_PERIOD, PROBE_UPDATES, PROBE_LR)

OLD_RUN = '411c44aad7ba4e8c90f5fbec9e5ecbaa'
OLD_EXECUTION = '9074575c76e3484baa5fd0f920e7ff91'
OLD_SOURCE = '2c0b5fbfe5a61d8bb1c0d407b0109c4012a2d286'
OLD_CONFIG_SHA = 'cf16231dc74f8f1fa74c5e2d999d9334e1e82930dd6b918efe57dbdcbada5773'
OLD_QUALIFICATION_SHA = '133b8e4bf5212b54feaea24cd2c09630b5e3a015e9f8c2c7375e7a8a3f004fb7'
CP5_SHA = 'b5c81ae76c12f677a0663d6e6b0f0482ca63fde0507c2a018aa6034282105594'
WARMUP_RUN = '8af926a49630428097b94027864da68b'


def _same_asset(template, old, key):
    value = old[key]
    if template.get(key) not in (None, value):
        raise ValueError('Accepted asset identity changed: '+key)
    return value


def _reuse_source(base, old_base, old_config, qualification):
    source = remote(old_config['probe_source_checkpoint'])
    expected = old_base/'preparation/warmup-source-epoch-0005.pt'
    if (source != expected or source.is_symlink() or not source.is_file()
            or qualification.get('source_checkpoint') != str(source)
            or qualification.get('path') != str(source)
            or old_config.get('probe_source_epoch') != 5
            or old_config.get('probe_source_sha256') != CP5_SHA
            or old_config.get('probe_source_model_kind') != 'ema'
            or old_config.get('probe_source_run') != WARMUP_RUN
            or qualification.get('source_epoch') != 5
            or qualification.get('source_sha256') != CP5_SHA
            or qualification.get('source_warmup_run') != WARMUP_RUN
            or qualification.get('model_kind') != 'ema'
            or qualification.get('model_only_import') is not True
            or qualification.get('finite') is not True
            or qualification.get('full_sha_verified') is not True
            or qualification.get('cpu_load_verified') is not True
            or qualification.get('warmup_optimizer_imported') is not False
            or qualification.get('warmup_rng_imported') is not False
            or qualification.get('question_max_tokens') != QMAX
            or qualification.get('prefix_tokens') != PREFIX_TOKENS
            or qualification.get('head_max_tokens') != WIDTH
            or qualification.get('hidden_size') != DIM):
        raise ValueError('Only the previously accepted immutable warmup CP5 EMA is authorized')
    extraction = qualification.get('native_model_extraction', {})
    if (extraction.get('learned_tensor_count') != 14 or extraction.get('source_epoch') != 5
            or extraction.get('source_warmup_run') != WARMUP_RUN
            or extraction.get('source_model_kind') != 'ema'
            or extraction.get('source_actual_adam_updates') != 30450
            or extraction.get('source_ema_updates') != 30450
            or extraction.get('warmup_model_weights_only') is not True
            or not isinstance(extraction.get('nonpersistent_buffer_names'), list)
            or 'rotary.inv_freq' not in extraction['nonpersistent_buffer_names']):
        raise ValueError('Accepted native CP5 model/buffer extraction proof differs')
    before = source.stat()
    if (before.st_size != qualification['source_bytes']
            or before.st_size != old_config['probe_source_bytes']
            or before.st_ino != qualification['source_inode']
            or before.st_dev != qualification['source_device']):
        raise ValueError('Accepted immutable CP5 inode or size changed')
    target = base/'preparation/warmup-source-epoch-0005.pt'
    os.link(source, target, follow_symlinks=False)
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if signature(before) != signature(target.stat()) or signature(before) != signature(source.stat()):
        raise ValueError('Immutable accepted CP5 changed during hardlink; candidate preserved')
    if full_sha(target) != CP5_SHA:
        raise ValueError('Reused CP5 full SHA differs; candidate preserved')
    result = dict(qualification, source_checkpoint=str(target), path=str(target),
        source_inode=target.stat().st_ino, source_device=target.stat().st_dev,
        source_bytes=target.stat().st_size, source_sha256=CP5_SHA,
        reused_accepted_cpu_qualification=True, cpu_load_not_reexecuted=True,
        qualification_mode='full_sha_identical_accepted_cpu_qualification_reuse',
        parent_source_qualification_path=str(old_base/'preparation/source-qualification.json'),
        parent_source_qualification_sha256=OLD_QUALIFICATION_SHA,
        accepted_parent_checkpoint_path=str(source), accepted_parent_actor_run=OLD_RUN,
        accepted_parent_actor_execution=OLD_EXECUTION, accepted_parent_actor_source=OLD_SOURCE,
        cpu_load_verified_scope='inherited_parent_exact_CP5_bytes_after_new_full_SHA',
        current_cpu_checkpoint_tensor_loads=0, current_tensor_deserializations=0,
        current_full_checkpoint_byte_hash_reads=1,
        full_sha_verified=True, cuda_reload_verified=False, observed_unix=time.time())
    once(base/'preparation/source-qualification.json', result)
    return result


def prepare(template_config_path, old_actor_handle_path):
    """Run once in an isolated staged namespace, without touching either trainer."""
    template_config_path, old_actor_handle_path = remote(template_config_path), remote(old_actor_handle_path)
    template_raw, handle_raw = stable_bytes(template_config_path), stable_bytes(old_actor_handle_path)
    template, handle = json.loads(template_raw), json.loads(handle_raw)
    base = remote(template['base'])
    if (not str(base).startswith(PREFIX) or not base.is_dir() or base.resolve(strict=True) != base
            or handle.get('run_id') != OLD_RUN or handle.get('execution_id') != OLD_EXECUTION
            or handle.get('source_commit') != OLD_SOURCE
            or template.get('accepted_source_config_sha256') != OLD_CONFIG_SHA
            or template.get('accepted_source_qualification_sha256') != OLD_QUALIFICATION_SHA):
        raise ValueError('Use the isolated new namespace and exact accepted old actor source pins')
    old_base = remote(handle['base']).resolve(strict=True)
    if old_base == base or old_base.name != 'search-g1-prefix-selection200-a800-prefix5-'+OLD_RUN:
        raise ValueError('Parent actor namespace differs or aliases this fresh experiment')
    directory = base/'preparation'
    directory.mkdir(exist_ok=True)
    intent = dict(schema_version=1, role='a800_prefix5_period10_cpu_accepted_source_reuse_once',
        base=str(base), run_id=template['run_id'], source_commit=template['source_commit'], process=process_identity(),
        template_config_path=str(template_config_path), template_config_sha256=hashlib.sha256(template_raw).hexdigest(),
        old_actor_handle_path=str(old_actor_handle_path), old_actor_handle_sha256=hashlib.sha256(handle_raw).hexdigest(),
        old_actor_run_id=OLD_RUN, old_actor_execution_id=OLD_EXECUTION, old_actor_source_commit=OLD_SOURCE,
        old_training_stopped=False, cuda_model_or_forward_used=False, nvml_queried=False,
        warmup_latest_consulted=False, reused_accepted_cpu_qualification=True)
    once(directory/'source-cpu-prepare-intent.json', intent)
    try:
        old_directory = old_base/'preparation'
        old_config = checked_json(old_directory/'launch-config.json', OLD_CONFIG_SHA)
        qualification = checked_json(old_directory/'source-qualification.json', OLD_QUALIFICATION_SHA)
        cpu_exit = json.loads(stable_bytes(old_directory/'source-cpu-prepare-exit.json'))
        if cpu_exit.get('exit_code') != 0 or not isinstance(cpu_exit.get('complete_sha256'), str):
            raise ValueError('Parent CPU preparation has no actual accepted exit0')
        old_complete = checked_json(old_directory/'source-cpu-prepare-complete.json', cpu_exit['complete_sha256'])
        if (old_config.get('run_id') != OLD_RUN or old_config.get('execution_id') != OLD_EXECUTION
                or old_config.get('source_commit') != OLD_SOURCE or old_config.get('base') != str(old_base)
                or old_complete.get('status') != 'A800_PREFIX5_CPU_PREPARE_COMPLETE'
                or old_complete.get('run_id') != OLD_RUN or old_complete.get('source_commit') != OLD_SOURCE
                or old_complete.get('launch_config_sha256') != OLD_CONFIG_SHA
                or old_complete.get('source_qualification_sha256') != OLD_QUALIFICATION_SHA
                or old_config.get('probe_source_cpu_proof_sha256') != OLD_QUALIFICATION_SHA
                or old_complete.get('source_epoch') != 5 or old_complete.get('source_sha256') != CP5_SHA):
            raise ValueError('Parent config/qualification/complete/exit chain differs')
        source = _reuse_source(base, old_base, old_config, qualification)
        # These two accepted identity artifacts are hash-checked and referenced.
        # No169028-row parsing or feature/tensor copying is required.
        pool_path = remote(old_config['source_question_pool'])
        fixed_path = remote(old_config['fixed_test_identity_path'])
        pool_sha, fixed_sha = old_config['source_question_pool_sha256'], old_config['fixed_test_identity_sha256']
        if (pool_sha != old_complete.get('source_question_pool_sha256')
                or fixed_sha != old_complete.get('fixed_test_identity_sha256')
                or old_complete.get('source_question_pool') != str(pool_path)
                or old_complete.get('fixed_test_identity_path') != str(fixed_path)
                or full_sha(pool_path) != pool_sha or full_sha(fixed_path) != fixed_sha):
            raise ValueError('Accepted source question pool or fixed512 identities changed')
        fixed = json.loads(stable_bytes(fixed_path))
        if fixed.get('questions') != 512 or fixed.get('trajectories') != 2560 or len(fixed.get('records', [])) != 512:
            raise ValueError('Accepted original fixed512 identity metadata counts differ')
        if old_config['fixed_test_manifest_sha256'] != source['source_feature_manifest_sha256']:
            raise ValueError('Accepted fixed feature manifest differs from qualified warmup CP5')
        for key in ('model_revision', 'model_files_sha256', 'canonical_train_sha256', 'common_manifest_sha256'):
            _same_asset(template, old_config, key)
        config = dict(template, base=str(base), world_size=WORLD, prefix_policy_tokens=PREFIX_TOKENS,
            candidate_questions=CANDIDATE_QUESTIONS, selected_questions=SELECTED_QUESTIONS, responses_per_question=G,
            outers=FINAL_OUTER, probe_update_period=PROBE_PERIOD, probe_updates_per_period=PROBE_UPDATES,
            probe_learning_rate=PROBE_LR, probe_epochs_per_update=1, probe_update_at_final=False,
            nonmixed_advantage_mode='zero_anchor_all_correct_only', actual_em_selection_fallback='probe_score_fill',
            hidden_capture_contract=HIDDEN_CAPTURE_CONTRACT, independent_encoder_forward=False,
            probe_source_checkpoint=source['source_checkpoint'], probe_source_sha256=CP5_SHA,
            probe_source_bytes=source['source_bytes'], probe_source_inode=source['source_inode'],
            probe_source_epoch=5, probe_source_model_kind='ema', probe_source_run=WARMUP_RUN,
            probe_source_cpu_proof=str(directory/'source-qualification.json'),
            probe_source_cpu_proof_sha256=full_sha(directory/'source-qualification.json'),
            fixed_test_identity_path=str(fixed_path), fixed_test_identity_sha256=fixed_sha,
            fixed_test_manifest_sha256=old_config['fixed_test_manifest_sha256'],
            fixed_test_features=_same_asset(template, old_config, 'fixed_test_features'),
            source_question_pool=str(pool_path), source_question_pool_sha256=pool_sha,
            pool_state_path=str(base/'question-pool/state.json'), prefix_probe_receipt=str(base/'probe/latest.json'),
            reused_accepted_cpu_qualification=True, reused_identity_assets=True,
            source_asset_parent_base=str(old_base), source_asset_parent_config_sha256=OLD_CONFIG_SHA,
            source_asset_parent_qualification_sha256=OLD_QUALIFICATION_SHA)
        # Only reuse accepted bindings here. Launch must perform its actual
        # UUID ownership and idle admission after the authorized old-job stop.
        for key in ('physical_gpu_indices', 'physical_gpu_uuids', 'auxiliary_gpu_indices', 'auxiliary_gpu_uuids'):
            config[key] = _same_asset(template, old_config, key)
        if (len(config['physical_gpu_indices']) != WORLD or len(config['physical_gpu_uuids']) != WORLD
                or config['auxiliary_gpu_indices'] != [5, 6] or len(config['auxiliary_gpu_uuids']) != 2
                or set(config['physical_gpu_indices']) & {5, 6}
                or len(set(config['physical_gpu_uuids']+config['auxiliary_gpu_uuids'])) != WORLD+2):
            raise ValueError('Accepted training/auxiliary physical UUID bindings differ')
        from .launch import validate_config
        validate_config(config, base)
        once(directory/'launch-config.json', config)
        complete = dict(schema_version=1, status='A800_PREFIX5_CPU_PREPARE_COMPLETE', base=str(base),
            run_id=template['run_id'], execution_id=template['execution_id'], source_commit=template['source_commit'],
            source_epoch=5, source_checkpoint=source['source_checkpoint'], source_sha256=CP5_SHA, model_kind='ema',
            model_only_import=True, finite=True, source_qualification=str(directory/'source-qualification.json'),
            source_qualification_sha256=full_sha(directory/'source-qualification.json'),
            launch_config=str(directory/'launch-config.json'), launch_config_sha256=full_sha(directory/'launch-config.json'),
            fixed_test_identity_path=str(fixed_path), fixed_test_identity_sha256=fixed_sha,
            source_question_pool=str(pool_path), source_question_pool_sha256=pool_sha,
            canonical_questions=169028, eligible_unique_questions=old_complete['eligible_unique_questions'], fixed_test_questions=512,
            data_source=old_complete['data_source'], reused_accepted_cpu_qualification=True,
            parent_source_qualification_sha256=OLD_QUALIFICATION_SHA, parent_launch_config_sha256=OLD_CONFIG_SHA,
            parent_cpu_prepare_exit=cpu_exit, cpu_load_not_reexecuted=True, warmup_latest_consulted=False,
            identity_canonical_rescan_performed=False, nvml_queried=False, idle_admission_deferred_to_launch=True,
            old_training_stopped=False, optimizer_or_rng_imported=False, feature_tensor_files_read=0,
            checkpoint_tensor_loads=0, token_array_files_read=0, cuda_model_or_forward_used=False,
            new_generation=0, new_em_scoring=0, process=process_identity(), observed_unix=time.time())
        once(directory/'source-cpu-prepare-complete.json', complete)
        once(directory/'source-cpu-prepare-exit.json', dict(exit_code=0, process=process_identity(),
            complete_sha256=full_sha(directory/'source-cpu-prepare-complete.json')))
        return complete
    except BaseException as error:
        once(directory/'source-cpu-prepare-exit.json', dict(exit_code=1, process=process_identity(),
            error_type=type(error).__name__, error=str(error), failed_candidate_preserved=True,
            same_once_claim_must_not_be_replayed=True, old_training_stopped=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template-config', required=True)
    parser.add_argument('--old-actor-handle', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template_config, args.old_actor_handle), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
