"""Authenticate frozen all-token assets, then fit one fresh official Qwen block.

The only inherited trainable parameters are the official, never-trained layer35
weights. The readout, Adam and private RNG are new. Existing qualified token
features and G5 teacher labels are reused without encoder/actor allocation.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import traceback

from reproduction.a800_common import qprobe_token_warmup as qualified

sys.dont_write_bytecode = True
BLOCK_INDEX = 35
PREFIX = 'model.layers.35.'
PREPARED_STAGE = 'QWEN_LAST_BLOCK_WARMUP10_CPU_PREPARED'
READY_STAGE = 'QWEN_LAST_BLOCK_FRESH_HEAD_WARMUP10_COMPLETE'
QUALIFIED_READY = '/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-494eafd18b004cb980121e46b188db44/warmup/ready.json'
QUALIFIED_READY_SHA = '496e30af7c9c4afb164507f90d14e3f6bfdc7e15e1f6c6f3a4a7c0d5f4198f5f'
REPLAY_CONTRACT = dict(window_completed_formal_outers=8, teacher_warmup_in_replay=False,
    age_zero='current_complete_formal_outer', ages_included=list(range(8)),
    weight_formula='2*2**(-age) if 0 <= age <= 7 else 0',
    reduction='mean(weight * per_group_loss)', normalize_weights=False)
DTYPES = {'BF16': ('bfloat16', 2), 'F32': ('float32', 4), 'F16': ('float16', 2)}
sha, read, remote, once, canonical = qualified.sha, qualified.read, qualified.remote, qualified.once, qualified.canonical


def progress(output, value):
    event = dict(value, observed_unix=time.time())
    with (output / 'progress.jsonl').open('ab') as stream:
        stream.write(canonical(event) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    temporary = output / 'progress.partial.json'
    with temporary.open('wb') as stream:
        stream.write(canonical(event) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, output / 'progress.json')
    print(json.dumps(event, allow_nan=False), flush=True)


def task_config(args):
    return dict(source_commit=args.source_commit, source_run_base=str(args.source_run_base),
        teacher_identity_sha256=args.teacher_identity_sha256,
        resume_token_feature_output=str(args.resume_token_feature_output),
        qualified_token_warmup_ready=QUALIFIED_READY, qualified_token_warmup_ready_sha256=QUALIFIED_READY_SHA,
        feature_dtype='bfloat16', feature_layer=27, block_index=35,
        architecture='native_transformers_Qwen2DecoderLayer', native_attention_implementation='sdpa',
        global_max_tokens=244, trainable='official_layer35_and_fresh_flatten_linear_readout',
        epochs=10, batch_size=256, micro_batch_size=args.micro_batch_size,
        learning_rate=3e-4, auxiliary_weight=.3, target_statistic='sample_std',
        head_seed=args.head_seed, top_k=512, exploration_questions=0,
        classification_report='binary_accuracy', replay_contract=REPLAY_CONTRACT)


def qualification_mapping(mapping, parent):
    """Keep the frozen old qualification's exact source namespace intact."""
    from reproduction.a800_common.deploy import published_source
    old = published_source(Path(parent).parent / 'repo', qualified.RESUME_SOURCE)
    if not set(old).issubset(mapping):
        raise ValueError('QWEN_BLOCK_QUALIFICATION_SOURCE_FILES_MISSING')
    result = {name: mapping[name] for name in old}
    runner = 'reproduction/a800_common/qprobe_token_warmup.py'
    if any(result[name] != old[name] for name in old if name != runner):
        raise ValueError('QWEN_BLOCK_FROZEN_FEATURE_OR_TEACHER_SCIENTIFIC_SOURCE_CHANGED')
    return result


def qualify_features(args, output, mapping):
    # The old helper's epochs10 describes its historical qualification namespace.
    # It never loads a head or fits in this route; our task_config is epochs20.
    legacy_args = copy.copy(args)
    legacy_args.target_statistic = 'sample_std'
    legacy_args.auxiliary_weight = .3
    legacy_args.feature_dtype = 'bfloat16'
    result = qualified.resume_encoded_features(legacy_args, output,
        qualification_mapping(mapping, args.resume_token_feature_output))
    cache, qids, groups, provenance, scan, recovery = result
    if len(qids) != 5120 or cache.ids != qids or scan['global_max_tokens'] != 244:
        raise ValueError('QWEN_BLOCK_REQUIRES_SAME_5120_ALLTOKEN_GLOBAL244_FEATURES')
    if provenance['trajectories_regenerated'] is not False or provenance['old_head_inherited'] is not False:
        raise ValueError('QWEN_BLOCK_QUALIFICATION_CANNOT_INHERIT_TRAINED_HEAD_OR_NEW_LABELS')
    qualify_completed_token_ready(cache)
    return result


def qualify_completed_token_ready(cache):
    """Read the prior completion's scalar lineage, never its trained Head file."""
    path = remote(QUALIFIED_READY)
    if sha(path) != QUALIFIED_READY_SHA:
        raise ValueError('QWEN_BLOCK_QUALIFIED_TOKEN_WARMUP_READY_SHA_CHANGED')
    ready = read(path)
    if (ready.get('stage') != 'TOKEN_BINARY_FRESH_HEAD_WARMUP10_COMPLETE' or
            not ready.get('source_commit', '').startswith('f3807ac') or
            ready.get('feature_cache_dir') != str(cache.path) or ready.get('token_feature_spec_sha256') != cache.spec_sha or
            ready.get('token_feature_manifest_sha256') != sha(cache.path/'manifest.json') or
            ready.get('questions') != 5120 or ready.get('head_optimizer_updates') != 200 or
            ready.get('head_adam_steps') != 200 or ready.get('trajectories_generated') != 0 or
            ready.get('old_head_inherited') is not False or ready.get('formal_training_started') is not False):
        raise ValueError('QWEN_BLOCK_QUALIFIED_TOKEN_WARMUP_FEATURE_LINEAGE_CHANGED')
    return dict(path=str(path), sha256=QUALIFIED_READY_SHA, source_commit=ready['source_commit'],
                feature_cache_dir=str(cache.path), token_feature_spec_sha256=cache.spec_sha,
                token_feature_manifest_sha256=ready['token_feature_manifest_sha256'], trained_head_read=False)


def block_shapes(hidden_size=2048, num_attention_heads=16, num_key_value_heads=2, intermediate_size=11008):
    head_dim = hidden_size // num_attention_heads
    kv_size = num_key_value_heads * head_dim
    return {'input_layernorm.weight': [hidden_size], 'post_attention_layernorm.weight': [hidden_size],
        'self_attn.q_proj.weight': [hidden_size, hidden_size], 'self_attn.q_proj.bias': [hidden_size],
        'self_attn.k_proj.weight': [kv_size, hidden_size], 'self_attn.k_proj.bias': [kv_size],
        'self_attn.v_proj.weight': [kv_size, hidden_size], 'self_attn.v_proj.bias': [kv_size],
        'self_attn.o_proj.weight': [hidden_size, hidden_size],
        'mlp.gate_proj.weight': [intermediate_size, hidden_size],
        'mlp.up_proj.weight': [intermediate_size, hidden_size],
        'mlp.down_proj.weight': [hidden_size, intermediate_size]}


def official_config(model):
    value = read(Path(model) / 'config.json')
    expected = dict(model_type='qwen2', hidden_size=2048, num_hidden_layers=36,
        num_attention_heads=16, num_key_value_heads=2, intermediate_size=11008,
        rms_norm_eps=1e-6, rope_theta=1000000.0, hidden_act='silu', attention_dropout=0.0)
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError('QWEN_BLOCK_OFFICIAL_NATIVE_ARCHITECTURE_CHANGED')
    if value.get('use_sliding_window', False) is not False or value.get('max_position_embeddings', 0) < 244:
        raise ValueError('QWEN_BLOCK_OFFICIAL_CONTEXT_CHANGED')
    return value


def safetensor_header(path):
    """Read metadata only, with no tensor allocation or model construction."""
    with Path(path).open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('QWEN_BLOCK_INVALID_SAFETENSOR_HEADER')
        size = struct.unpack('<Q', raw)[0]
        if not 2 <= size <= 64 * 1024**2:
            raise ValueError('QWEN_BLOCK_INVALID_SAFETENSOR_HEADER')
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError('QWEN_BLOCK_TRUNCATED_SAFETENSOR_HEADER')
        header = json.loads(raw)
    return header, 8 + size


def region_sha(path, offset, size):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        while size:
            block = stream.read(min(size, 4 * 1024**2))
            if not block:
                raise ValueError('QWEN_BLOCK_TRUNCATED_OFFICIAL_TENSOR')
            digest.update(block); size -= len(block)
    return digest.hexdigest()


def block_index_binding(model, hashes, *, expected_shapes=None):
    model = Path(model)
    index_path = model / 'model.safetensors.index.json'
    if sha(index_path) != hashes['model.safetensors.index.json']:
        raise ValueError('QWEN_BLOCK_OFFICIAL_TENSOR_INDEX_SHA_CHANGED')
    index = read(index_path)['weight_map']
    expected_shapes = block_shapes() if expected_shapes is None else expected_shapes
    selected = {name: shard for name, shard in index.items() if name.startswith(PREFIX)}
    if set(selected) != {PREFIX + name for name in expected_shapes}:
        raise ValueError('QWEN_BLOCK_INCOMPLETE_OR_EXTRA_LASTBLOCK_TENSORS')
    shards, tensors, total_bytes = {}, {}, 0
    for shard in sorted(set(selected.values())):
        path = model / shard
        if Path(shard).name != shard or path.resolve() != path or path.is_symlink() or sha(path) != hashes.get(shard):
            raise ValueError('QWEN_BLOCK_OFFICIAL_TENSOR_SHARD_BINDING_CHANGED')
        header, data_start = safetensor_header(path)
        shards[shard] = dict(sha256=hashes[shard], bytes=path.stat().st_size)
        for name in sorted(name for name, filename in selected.items() if filename == shard):
            key = name[len(PREFIX):]
            item = header[name]
            if item['shape'] != expected_shapes[key] or item['dtype'] != 'BF16':
                raise ValueError('QWEN_BLOCK_OFFICIAL_TENSOR_SHAPE_OR_DTYPE_CHANGED')
            start, end = item['data_offsets']
            if type(start) is not int or type(end) is not int or start < 0 or end < start:
                raise ValueError('QWEN_BLOCK_INVALID_OFFICIAL_TENSOR_OFFSETS')
            elements = 1
            for size in item['shape']: elements *= size
            if end - start != elements * 2 or data_start + end > path.stat().st_size:
                raise ValueError('QWEN_BLOCK_OFFICIAL_TENSOR_BYTE_LENGTH_CHANGED')
            tensors[key] = dict(source_key=name, shard=shard, shape=item['shape'], source_dtype='bfloat16',
                bytes=end-start, source_sha256=region_sha(path, data_start+start, end-start))
            total_bytes += end-start
    return dict(source_layer=35, tensor_count=len(tensors), tensor_bytes=total_bytes,
        tensor_index_sha256=hashes['model.safetensors.index.json'], official_shards=shards,
        tensors=tensors, full_encoder_loaded=False, actor_loaded=False)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def load_block_weights(model, proof, *, opener=None):
    if opener is None:
        from safetensors import safe_open
        opener = safe_open
    model = Path(model); state = {}
    for shard, item in proof['official_shards'].items():
        path = model / shard
        if sha(path) != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError('QWEN_BLOCK_OFFICIAL_SHARD_CHANGED_AFTER_PREPARATION')
        with opener(str(path), framework='pt', device='cpu') as handle:
            for key, tensor_proof in proof['tensors'].items():
                if tensor_proof['shard'] != shard: continue
                value = handle.get_tensor(tensor_proof['source_key'])
                if (list(value.shape) != tensor_proof['shape'] or str(value.dtype) != 'torch.bfloat16' or
                        tensor_sha(value) != tensor_proof['source_sha256']):
                    raise ValueError('QWEN_BLOCK_LOADED_OFFICIAL_TENSOR_DIFFERS')
                state[key] = value
    if len(state) != proof['tensor_count']:
        raise ValueError('QWEN_BLOCK_LOADED_TENSOR_COUNT_DIFFERS')
    return state


def verify_copied_block(actual, expected):
    if (actual.get('source_layer') != 35 or actual.get('tensor_count') != 12 or
            actual.get('all_copied_parameters_trainable') is not True or
            set(actual.get('tensors', {})) != set(expected['tensors'])):
        raise ValueError('QWEN_BLOCK_NATIVE_COPY_PROOF_INCOMPLETE')
    for key, source in expected['tensors'].items():
        target = actual['tensors'][key]
        if any(target.get(field) != source[field] for field in ('source_key', 'shape', 'source_dtype', 'source_sha256')):
            raise ValueError('QWEN_BLOCK_NATIVE_COPY_DOES_NOT_MATCH_OFFICIAL_SOURCE')
        if target.get('exact_cast_equal') is not True or not target.get('copied_fp32_sha256'):
            raise ValueError('QWEN_BLOCK_NATIVE_COPY_NOT_EXACT_FLOAT32_CAST')


def native_binding():
    from search_g1.qprobe_qwen_block_model import native_implementation_binding
    return native_implementation_binding()


def native_block_proof(binding, *, copied):
    from search_g1.qprobe_features import OFFICIAL_REVISION
    native_class = binding['implementations']['decoder_layer']['qualified_name']
    if native_class != 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer' or not binding.get('transformers_version'):
        raise ValueError('QWEN_BLOCK_REQUIRES_ACTUAL_NATIVE_TRANSFORMERS_LAYER')
    return dict(native_class=native_class, transformers_version=binding['transformers_version'],
        block_index=35, source_tensors_verified=True, official_block_copy_verified=copied,
        head_initialized=copied, fresh_linear=copied, fresh_adam=copied, official_revision=OFFICIAL_REVISION)


def cpu_checks(output):
    """Exercise the staged native implementation in the existing CPU environment."""
    module = 'search_g1.test_qprobe_qwen_block_model'
    program = '''import json, os, sys, torch, unittest
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert not torch.cuda.is_initialized()
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(sys.argv[1]))
summary = dict(module=sys.argv[1], tests_run=result.testsRun, failures=len(result.failures),
    errors=len(result.errors), skipped=len(result.skipped), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
    cuda_initialized=torch.cuda.is_initialized())
summary['passed'] = result.wasSuccessful() and result.testsRun > 0 and not result.skipped and not summary['cuda_initialized']
print('QWEN_BLOCK_CPU_CHECKS=' + json.dumps(summary, sort_keys=True))
sys.exit(0 if summary['passed'] else 1)
'''
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([sys.executable, '-c', program, module], cwd=Path(__file__).resolve().parents[2],
        env=environment, text=True, capture_output=True, timeout=180)
    (output/'cpu-checks.log').write_text(result.stdout + result.stderr)
    markers = [line[len('QWEN_BLOCK_CPU_CHECKS='):] for line in result.stdout.splitlines()
               if line.startswith('QWEN_BLOCK_CPU_CHECKS=')]
    if len(markers) != 1:
        raise ValueError('QWEN_BLOCK_STAGED_NATIVE_CPU_CHECK_RESULT_MISSING')
    summary = json.loads(markers[0])
    if (result.returncode != 0 or summary.get('passed') is not True or summary.get('tests_run', 0) < 1 or
            summary.get('failures') != 0 or summary.get('errors') != 0 or summary.get('skipped') != 0 or
            summary.get('cuda_visible_devices') != '' or summary.get('cuda_initialized') is not False):
        raise ValueError('QWEN_BLOCK_STAGED_NATIVE_CPU_CHECKS_DID_NOT_PASS')
    summary['log_sha256'] = sha(output/'cpu-checks.log')
    return summary


def prepare(args, output, mapping):
    checks = cpu_checks(output)
    progress(output, dict(stage='qwen_block_read_only_asset_qualification_started'))
    cache, qids, groups, provenance, scan, recovery = qualify_features(args, output, mapping)
    run_manifest = read(Path(args.source_run_base) / 'logs/run-manifest.json')
    model = remote(run_manifest['initial_model'])
    config = official_config(model)
    block = block_index_binding(model, run_manifest['initial_model_sha256'])
    implementation = native_binding()
    prepared = dict(stage=PREPARED_STAGE, task_config=task_config(args), source_sha256=mapping,
        source_map_sha256=hashlib.sha256(canonical(mapping)).hexdigest(),
        label_producer_provenance=provenance, global_scan=scan, parent_feature_resume=recovery,
        official_model_dir=str(model), official_model_config=config, official_model_config_sha256=sha(model/'config.json'),
        official_block_source=block, native_implementation=implementation,
        native_block_proof=native_block_proof(implementation, copied=False),
        cpu_checks=checks,
        qualified_token_warmup_ready=qualify_completed_token_ready(cache),
        token_feature_spec_sha256=cache.spec_sha, feature_cache_dir=str(cache.path),
        token_feature_manifest_sha256=sha(cache.path/'manifest.json'),
        ordered_warmup_qids_sha256=hashlib.sha256(canonical(qids)).hexdigest(),
        no_gpu_model_loaded=True, encoder_reencoded=False, trajectories_regenerated=False,
        old_head_inherited=False, head_initialized=False, formal_training_started=False, checked_unix=time.time())
    once(output / 'prepared.json', prepared)
    progress(output, dict(stage=PREPARED_STAGE, output_dir=str(output), prepared_sha256=sha(output/'prepared.json'),
        global_max_tokens=244, questions=5120, block_index=35, official_block_tensor_bytes=block['tensor_bytes'],
        no_gpu_model_loaded=True, formal_training_started=False))
    return prepared


def fit_and_save(args, output, prepared, gpu, cache, qids, groups, provenance, recovery, block_state):
    import numpy as np
    import torch
    from search_g1.qprobe_qwen_block_model import QwenBlockQProbeConfig, QwenBlockWarmup
    config = QwenBlockQProbeConfig(max_tokens=244, epochs=10, batch_size=256,
        micro_batch_size=args.micro_batch_size, learning_rate=3e-4, auxiliary_weight=.3, seed=args.head_seed, top_k=512)
    fresh = QwenBlockWarmup(config, block_state, device='cuda:0')
    del block_state
    verify_copied_block(fresh.initial_block_proof, prepared['official_block_source'])
    if fresh.update_count or fresh._adam_steps() or fresh.fitted or fresh.optimizer.state:
        raise ValueError('QWEN_BLOCK_REQUIRES_NEW_READOUT_ADAM_AND_PRIVATE_RNG')
    if (fresh.initial_readout_proof.get('fresh') is not True or fresh.initial_readout_proof.get('seed') != args.head_seed or
            not fresh.initial_readout_proof.get('weight_sha256') or not fresh.initial_readout_proof.get('bias_sha256') or
            fresh.native_implementation != prepared['native_implementation']):
        raise ValueError('QWEN_BLOCK_FRESH_NATIVE_READOUT_OR_IMPLEMENTATION_PROOF_DIFFERS')
    progress(output, dict(stage='fresh_official_qwen_lastblock_head_initialized', head_optimizer_updates=0,
        head_adam_steps=0, block_index=35, official_parameter_bytes=prepared['official_block_source']['tensor_bytes'],
        old_head_inherited=False, encoder_loaded_in_this_process=False))
    counts = np.asarray([group['correct_count'] for group in groups], dtype=np.int64)
    metadata = dict(token_feature_spec_sha256=cache.spec_sha, label_producer_provenance=provenance,
        parent_feature_resume=recovery, official_block_source=prepared['official_block_source'],
        native_implementation=prepared['native_implementation'], official_model_config=prepared['official_model_config'],
        replay_contract=REPLAY_CONTRACT, task_config=task_config(args), initialization_kind='fresh_official_qwen_lastblock35')
    try:
        metrics = fresh.fit(cache, cache.lengths.copy(), counts, qids,
            progress_callback=lambda event: progress(output, event))
        if fresh.update_count != 200 or fresh._adam_steps() != 200 or fresh.fitted is not True:
            raise ValueError('QWEN_BLOCK_WARMUP10_ADAM_INCOMPLETE')
        if len(metrics['epoch_history']) != 11 or len(metrics['minibatch_history']) != 200:
            raise ValueError('QWEN_BLOCK_WARMUP10_HISTORY_INCOMPLETE')
        if any('binary_accuracy' not in row and 'accuracy' not in row for row in metrics['minibatch_history']):
            raise ValueError('QWEN_BLOCK_MINIBATCH_BINARY_ACCURACY_MISSING')
        verify_histories(metrics)
    except BaseException:
        preserve_partial(output, fresh, metadata)
        raise
    try:
        once(output / 'warmup-metrics.json', metrics)
        checkpoint = output / 'fresh-head.pt'
        qualified.save_remote_state(checkpoint, dict(metadata, head=fresh.state_dict()))
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if loaded['head']['update_count'] != 200 or loaded['head']['fitted'] is not True:
            raise ValueError('QWEN_BLOCK_PUBLISHED_HEAD_STATE_INCOMPLETE')
        for key, value in fresh.state_dict()['model'].items():
            if not torch.equal(value.detach().cpu(), loaded['head']['model'][key]) or not bool(torch.isfinite(loaded['head']['model'][key]).all()):
                raise ValueError('QWEN_BLOCK_PUBLISHED_PARAMETER_READBACK_DIFFERS')
        final = metrics['epoch_history'][-1]
        ready = dict(stage=READY_STAGE, source_commit=args.source_commit, source_map_sha256=prepared['source_map_sha256'],
            task_config=task_config(args), **gpu, questions=5120, teacher_trajectories_reused=25600, trajectories_generated=0,
            block_index=35, global_max_tokens=244, feature_cache_dir=str(cache.path), token_feature_spec_sha256=cache.spec_sha,
            token_feature_manifest_sha256=sha(cache.path/'manifest.json'), official_block_source=prepared['official_block_source'],
            official_model_config_sha256=prepared['official_model_config_sha256'], native_implementation=prepared['native_implementation'],
            initial_block_proof=fresh.initial_block_proof, initial_readout_proof=fresh.initial_readout_proof,
            native_block_proof=native_block_proof(prepared['native_implementation'], copied=True),
            qualified_token_warmup_ready=prepared['qualified_token_warmup_ready'],
            cpu_checks=prepared['cpu_checks'],
            head_optimizer_updates=200, head_adam_steps=200, epoch_records=11, minibatch_records=200,
            primary_report='binary_accuracy', primary_binary_accuracy=final.get('binary_accuracy', final.get('accuracy')),
            head_sha256=sha(checkpoint), head_bytes=checkpoint.stat().st_size,
            metrics_sha256=sha(output/'warmup-metrics.json'), prepared_sha256=sha(output/'prepared.json'),
            metrics_scope='same5120_training_fit_only_no_independent_validation', old_head_inherited=False,
            encoder_reencoded=False, no_reencode=True, encoder_loaded_in_this_process=False,
            trajectories_regenerated=False, formal_training_started=False, automatic_formal_200_launch=False,
            replay_contract=REPLAY_CONTRACT, checked_unix=time.time())
        progress(output, ready)
        once(output / 'ready.json', ready)
        return ready
    except BaseException:
        preserve_partial(output, fresh, metadata)
        raise


def verify_histories(metrics):
    if (metrics.get('fit_epochs') != 10 or metrics.get('training_groups') != 5120 or
            metrics.get('actual_head_adam_updates') != 200 or metrics.get('expected_head_adam_updates') != 200 or
            len(metrics.get('epoch_history', [])) != 11 or len(metrics.get('minibatch_history', [])) != 200):
        raise ValueError('QWEN_BLOCK_WARMUP10_METRICS_BUDGET_DIFFERS')
    for epoch, row in enumerate(metrics['epoch_history']):
        if row.get('epoch') != epoch or row.get('actual_head_adam_updates') != epoch * 20:
            raise ValueError('QWEN_BLOCK_EPOCH_ADAM_HISTORY_DIFFERS')
    for ordinal, row in enumerate(metrics['minibatch_history'], 1):
        if (row.get('head_update_ordinal') != ordinal or row.get('head_adam_steps_before') != ordinal - 1 or
                row.get('head_adam_steps_after') != ordinal or row.get('epoch') != (ordinal - 1)//20 + 1 or
                row.get('minibatch_index') != (ordinal - 1) % 20 or row.get('batch_size') != 256 or
                not isinstance(row.get('binary_accuracy'), (float, int)) or not 0 <= row['binary_accuracy'] <= 1):
            raise ValueError('QWEN_BLOCK_MINIBATCH_ADAM_HISTORY_DIFFERS')


def preserve_partial(output, fresh, metadata):
    partial = output / 'partial-head.pt'
    qualified.save_remote_state(partial, dict(metadata, head=fresh.state_dict()))
    once(output / 'partial-head.json', dict(stage='QWEN_LAST_BLOCK_PARTIAL_HEAD_PRESERVED',
        head_optimizer_updates=fresh.update_count, head_adam_steps=fresh._adam_steps(), fitted=fresh.fitted,
        head_sha256=sha(partial), head_bytes=partial.stat().st_size, rollback_performed=False,
        automatic_retry=False, old_head_inherited=False, formal_training_started=False))


def run(args, output, mapping, gpu):
    import torch
    if torch.cuda.device_count() != 1: raise ValueError('QWEN_BLOCK_REQUIRES_ONE_ADMITTED_GPU')
    prepared = read(output / 'prepared.json')
    if prepared['stage'] != PREPARED_STAGE or prepared['task_config'] != task_config(args) or prepared['source_sha256'] != mapping:
        raise ValueError('QWEN_BLOCK_PREPARED_SOURCE_OR_TASK_CHANGED')
    if prepared['cpu_checks'].get('passed') is not True or sha(output/'cpu-checks.log') != prepared['cpu_checks']['log_sha256']:
        raise ValueError('QWEN_BLOCK_NATIVE_CPU_CHECK_RECEIPT_CHANGED')
    cache, qids, groups, provenance, scan, recovery = qualify_features(args, output, mapping)
    if (provenance != prepared['label_producer_provenance'] or recovery != prepared['parent_feature_resume'] or
            scan != prepared['global_scan'] or cache.spec_sha != prepared['token_feature_spec_sha256'] or
            sha(cache.path/'manifest.json') != prepared['token_feature_manifest_sha256'] or
            hashlib.sha256(canonical(qids)).hexdigest() != prepared['ordered_warmup_qids_sha256']):
        raise ValueError('QWEN_BLOCK_QUALIFIED_FROZEN_ASSETS_CHANGED')
    model = remote(prepared['official_model_dir'])
    if (sha(model/'config.json') != prepared['official_model_config_sha256'] or
            official_config(model) != prepared['official_model_config'] or native_binding() != prepared['native_implementation']):
        raise ValueError('QWEN_BLOCK_NATIVE_IMPLEMENTATION_OR_CONFIG_CHANGED')
    if torch.cuda.memory_allocated() != 0:
        raise ValueError('QWEN_BLOCK_UNEXPECTED_PRE_HEAD_CUDA_ALLOCATION')
    state = load_block_weights(model, prepared['official_block_source'])
    return fit_and_save(args, output, prepared, gpu, cache, qids, groups, provenance, recovery, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run-base', 'output-dir', 'source-publication'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--teacher-identity-sha256', required=True)
    parser.add_argument('--resume-token-feature-output', type=Path, default=Path(qualified.RESUME_OUTPUT))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--physical-index', type=int)
    parser.add_argument('--head-seed', type=int, default=1729)
    parser.add_argument('--micro-batch-size', type=int, default=32)
    args = parser.parse_args()
    output, origin, parent = remote(args.output_dir), remote(args.source_run_base), remote(args.resume_token_feature_output)
    if str(parent) != qualified.RESUME_OUTPUT or args.head_seed != 1729 or args.micro_batch_size < 1 or 256 % args.micro_batch_size:
        raise ValueError('QWEN_BLOCK_EXPLICIT_QUALIFICATION_OR_FRESH_FIT_SETTINGS_CHANGED')
    if output == origin or origin in output.parents or Path(qualified.TEACHER) in output.parents or output == parent or parent.parent in output.parents or output in parent.parents:
        raise ValueError('QWEN_BLOCK_OUTPUT_CANNOT_OVERWRITE_ORIGINAL_ASSETS')
    if os.environ.get('SLURM_JOB_ID') is not None or not (args.prepare_only or args.physical_index is not None):
        raise ValueError('QWEN_BLOCK_REQUIRES_EXPLICIT_LOCAL_ONE_GPU_ADMISSION')
    mapping = qualified.source_binding(args)
    if args.prepare_only:
        output.mkdir(parents=True, exist_ok=False); os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif not output.is_dir() or not (output/'prepared.json').is_file() or (output/'ready.json').exists():
        raise ValueError('QWEN_BLOCK_RUN_REQUIRES_FRESH_PREPARED_OUTPUT')
    for name in ('cache/hf', 'cache/triton', 'cache/inductor', 'tmp'):
        (output/name).mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(output/'cache/hf'), XDG_CACHE_HOME=str(output/'cache'), TMPDIR=str(output/'tmp'),
        TRITON_CACHE_DIR=str(output/'cache/triton'), TORCHINDUCTOR_CACHE_DIR=str(output/'cache/inductor'),
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    stat = (Path('/proc')/str(os.getpid())/'stat').read_text(); fields = stat[stat.rfind(')')+2:].split()
    birth = dict(pid=os.getpid(), start_ticks=int(fields[19]), sid=int(fields[3]), uid=os.getuid(), hostname=socket.gethostname())
    once(output/('prepare-claim.json' if args.prepare_only else 'run-claim.json'),
        dict(task_config=task_config(args), **birth, no_automatic_retry=True, claimed_unix=time.time()))
    lock = None
    try:
        if args.prepare_only: prepare(args, output, mapping)
        else:
            lock, gpu = qualified.gpu_admission(args.physical_index)
            run(args, output, mapping, dict(birth, **gpu))
    except BaseException as failure:
        frames = [dict(file=Path(frame.filename).name, line=frame.lineno, function=frame.name)
                  for frame in traceback.extract_tb(failure.__traceback__)[-8:]]
        once(output/('prepare-failure.json' if args.prepare_only else 'run-failure.json'),
            dict(failure_type=type(failure).__name__, frames=frames, **birth, no_automatic_retry=True, checked_unix=time.time()))
        print(json.dumps(dict(stage='QWEN_LAST_BLOCK_WARMUP_FAILED', failure_type=type(failure).__name__,
            output_dir=str(output), formal_training_started=False)), flush=True)
        return 1
    finally:
        if lock is not None: lock.close()
    return 0


if __name__ == '__main__': raise SystemExit(main())
