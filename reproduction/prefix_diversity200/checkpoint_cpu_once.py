"""Once-only CPU/full-file readback of this run's actual joint milestones.

This reads no historical training checkpoint, invokes no model forward or CUDA API,
and never changes training state. CPU deserialization is not a GPU restore or an
independent backup. I1-I6 compares only selected small official initial tensors.
Missing joint completion is pending without an audit intent.
"""
import argparse
import fcntl
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from .actor import MILESTONES, REVISION, _state_binding, learning_rate
from .diversity import select_groups, summarize
from .checkpoint_cpu_correction import proof_name, verify_predecessor

STATUS = 'PREFIX_SELECTION200_JOINT_MILESTONE_FULL_SHA_CPU_PASS'
SOURCE_PROBE_SHA = '8df6bd445945ffbbf6a94171f24934495f0ce3ff17a3d2577bbd53d4b8fefd56'
TEST_SHA = 'a125fc5af90e34fcd728c82077d9d62070a78d5ef703ac11f3748687d088e64f'


def need(condition, message):
    if not condition: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def signature(info):
    # Link-count/ctime may change when training rotates another hardlink to an
    # immutable probe milestone. Byte identity uses inode/size/data mtime.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def owned(base, path):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts and base in path.parents,
         'Checkpoint asset is outside this current run')
    cursor = path
    while cursor != base:
        need(not cursor.is_symlink(), 'Checkpoint asset must not redirect into another run')
        cursor = cursor.parent
    need(path.resolve(strict=False) == path, 'Checkpoint asset is not canonical')
    return path


def hash_file(path, expected=None):
    path = Path(path)
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno()); h = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b''): h.update(chunk)
        after = os.fstat(handle.fileno())
    need(signature(before) == signature(after) == signature(path.stat()), 'File changed during independent full read')
    result = dict(path=str(path), bytes=before.st_size, sha256=h.hexdigest(), dev=before.st_dev,
                  inode=before.st_ino, mtime_ns=before.st_mtime_ns, independent_full_sha=True)
    if expected:
        need(result['sha256'] == expected['sha256'] and result['bytes'] == expected['bytes'],
             'Independent full file bytes differ from the actual checkpoint producer')
    return result


def load_torch_cpu(path, producer):
    import torch
    path = Path(path)
    print(json.dumps(dict(stage='TORCH_FULL_SHA_CPU_BEGIN', path=str(path), unix=time.time())), flush=True)
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno()); h = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b''): h.update(chunk)
        need(before.st_size == producer['bytes'] and h.hexdigest() == producer['sha256'],
             'Independent full Torch checkpoint SHA/size differs')
        handle.seek(0)
        saved = torch.load(handle, map_location='cpu', weights_only=False)
        after = os.fstat(handle.fileno())
    need(signature(before) == signature(after) == signature(path.stat()), 'Immutable CPU checkpoint changed during load')
    proof = dict(path=str(path), bytes=before.st_size, sha256=h.hexdigest(), dev=before.st_dev,
                 inode=before.st_ino, mtime_ns=before.st_mtime_ns,
                 independent_full_sha=True, actual_cpu_deserialization=True)
    print(json.dumps(dict(stage='TORCH_FULL_SHA_CPU_COMPLETE', path=str(path), bytes=before.st_size,
                         unix=time.time())), flush=True)
    return saved, proof


def validate_selection(selection, labels, selected, rollout, actor_rollout, joint, hashes):
    """Recompute the actual EM-based whole-G5 subset, without any model call."""
    chosen, all_groups = select_groups(selection, {row['trajectory_id']: row for row in labels})
    need(selected['selected'] == chosen and joint['actor_selection'] == selected
         and selected['continued1024'] == summarize(all_groups) and selected['actor512'] == summarize(chosen)
         and selected['prefix_selection_sha256'] == hashes['selection']['sha256']
         and selected['full_rollout_sha256'] == hashes['rollout']['sha256']
         and selected['outcome_selection'] == 'population_strict_EM_variance_K_times_5minusK_over25'
         and selected['tie_break'] == 'frozen_prefix_q_descending_then_original_candidate_ordinal'
         and selected['probe_uses_all_continued1024'] is True
         and selected['uncontinued_candidate_questions'] == 4096
         and selected['all_candidate_accuracy_available'] is False,
         'Actual actor subset is not the deterministic strict-EM population variance Top512')
    need(actor_rollout['actor_selection_metrics'] == selected
         and actor_rollout['actor_selection_sha256'] == hashes['actor_selection']['sha256']
         and actor_rollout['source_full_rollout_sha256'] == hashes['rollout']['sha256']
         and actor_rollout['questions'] == 512 and actor_rollout['trajectories'] == 2560
         and actor_rollout['actor_version'] == rollout['actor_version']
         and actor_rollout['raw_native_records_preserved'] is True,
         'Selected actor rollout is not bound to the complete original1024 continuation')
    return dict(deterministic_population_variance_top512_recomputed=True,
                continued1024=summarize(all_groups), actor512=summarize(chosen))


def validate_actor_rollout_binding(saved, path, proof):
    need(saved['original_rollout_sha256'] == proof['sha256']
         and saved['original_rollout_path'] == str(path),
         'Actor state belongs to another selected512 actor rollout')


def validate_probe_clock(metrics, rank_rows, global_rows, outer):
    """Verify original rank writes and global true-mean loss denominators."""
    need(metrics['global_step_loss_curve']['points'] == global_rows and len(global_rows) == 20
         and metrics['diagnostic_semantics'] == 'same_post_update_head_complete_selected1024G5_and_fixed512G5',
         'Probe metric prediction clock or original global loss series differs')
    need(len(rank_rows) == 8, 'Probe original write logs must cover all8 ranks')
    for rank, rows in enumerate(rank_rows):
        need(len(rows) == 20 and sum(r['local_questions'] for r in rows) == 128
             and sum(r['local_trajectories'] for r in rows) == 640,
             'Original probe rank did not consume128 unique whole-G5 groups in20 writes')
        for step, row in enumerate(rows, 1):
            from .checkpoint_i1i6_cpu import finite_numbers
            need(finite_numbers(row), 'Original probe rank write contains nonfinite loss/gradient diagnostics')
            need(row['rank'] == rank and row['outer'] == outer and row['step'] == step
                 and row['actual_adam_steps'] == 20*(outer-1)+step and row['actual_learning_rate'] == 1e-6
                 and row['global_questions'] == (52 if step % 5 == 0 else 51)
                 and row['local_trajectories'] == 5*row['local_questions']
                 and math.isclose(row['backward_loss'], row['total_loss']*8*row['local_questions']/row['global_questions'], rel_tol=2e-6, abs_tol=2e-6),
                 'Original probe actual rank Adam/DDP true-mean scale differs')
    for step, global_row in enumerate(global_rows, 1):
        n = 52 if step % 5 == 0 else 51
        need(global_row['outer'] == outer and global_row['step'] == step
             and global_row['actual_adam_steps_per_rank'] == 20*(outer-1)+step
             and global_row['actual_learning_rate'] == 1e-6
             and global_row['global_questions'] == n and global_row['global_trajectories'] == n*5
             and global_row['loss_prediction_clock'] == 'before_this_step_adam_update',
             'Probe global diagnostics do not follow the actual synchronous20-write clock')
        for key in ('trajectory_bce', 'group_bce', 'total_loss'):
            value = math.fsum(rows[step-1][key]*rows[step-1]['local_questions'] for rows in rank_rows)/n
            need(math.isfinite(value) and math.isclose(value, global_row[key], rel_tol=1e-12, abs_tol=1e-12),
                 'Probe reported global BCE is not the original rank-weighted true mean')
    for key in ('trajectory_bce', 'group_bce', 'total_loss'):
        expected = math.fsum(row[key]*row['global_questions'] for row in global_rows)/1024
        need(math.isclose(metrics['train_online_'+key], expected, rel_tol=1e-12, abs_tol=1e-12),
             'Probe mean BCE denominator must be1024 groups/5120 trajectories')
    return dict(original_rank_write_logs=8, writes_per_rank=20, group_denominator=1024,
                trajectory_denominator=5120, pre_step_prediction_clock_verified=True)


def audit_rollout_records(base, rollout, actor_rollout, selected, labels, *, i1i6=False):
    """Full SHA/read original native records; compare actor subset byte semantics."""
    from .actor import validate_record
    wanted = {tid for row in selected['selected'] for tid in row['trajectory_ids']}
    by_id = {row['trajectory_id']: row for row in labels}
    originals, proofs, policy_counts = {}, [], {}
    for role, receipt in (('full1024', rollout), ('actor512', actor_rollout)):
        seen = set()
        for item in receipt['shards']:
            path = owned(base, item['path']); proofs.append(dict(role=role, **hash_file(path, item)))
            before = path.stat(); count = 0
            with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as handle:
                for line in handle:
                    row = json.loads(line); tid = row['trajectory_id']; count += 1
                    need(tid not in seen and tid in by_id, 'Duplicate or foreign completed native trajectory')
                    seen.add(tid); validate_record(row, receipt['actor_version'])
                    if i1i6:
                        from .checkpoint_i1i6_cpu import native_budget
                        native_budget(row)
                        if role == 'actor512': policy_counts[(row['candidate_ordinal'],row['response_index'])] = sum(row['response_info_mask'])
                    need(row['qid'] == by_id[tid]['qid'] and row['prefix_sha256'] == by_id[tid]['prefix_sha256']
                         and row['selection_sha256'] == by_id[tid]['selection_sha256'], 'Native trajectory lost original prefix/outcome binding')
                    if role == 'full1024' and tid in wanted: originals[tid] = digest(row)
                    if role == 'actor512': need(tid in wanted and digest(row) == originals[tid], 'Actor changed an original same-G5 native record')
            need(count == item['count'] and signature(before) == signature(path.stat()), 'Native trajectory shard changed or has wrong count')
        need(seen == (set(by_id) if role == 'full1024' else wanted), 'Completed native trajectory set differs')
    return dict(files=proofs, original_full_trajectories=5120, actor_selected_trajectories=2560,
                original_native_records_equal=True, actor_policy_counts=[v for k,v in sorted(policy_counts.items())])


def tensor_cpu(value, *, dtype=None, shape=None, finite=False):
    import torch
    need(isinstance(value, torch.Tensor) and value.device.type == 'cpu', 'Expected actual CPU checkpoint tensor')
    if dtype is not None: need(value.dtype == dtype, 'Checkpoint tensor dtype differs')
    if shape is not None: need(tuple(value.shape) == tuple(shape), 'Checkpoint tensor shape differs')
    if finite:
        flat = value.reshape(-1)
        for start in range(0, flat.numel(), 2 * 1024 ** 2):
            need(bool(torch.isfinite(flat[start:start + 2 * 1024 ** 2]).all()), 'Actual saved tensor contains nonfinite values')


def validate_rng(rng):
    import numpy as np
    import torch
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    tensor_cpu(rng['torch_cpu'], dtype=torch.uint8)
    torch.Generator(device='cpu').set_state(rng['torch_cpu'])
    tensor_cpu(rng['torch_cuda'], dtype=torch.uint8)
    need(rng['torch_cuda'].ndim == 1 and rng['torch_cuda'].numel() > 0, 'Missing saved rank CUDA RNG byte state')
    return dict(python_numpy_cpu_torch_state_loaded=True, saved_cuda_rng_bytes=rng['torch_cuda'].numel(),
                cuda_rng_applied=False)


def validate_adam(optimizer, shapes, steps, rate, *, weight_decay):
    import torch
    groups = optimizer['param_groups']
    ids = [key for group in groups for key in group['params']]
    states = optimizer['state']
    need(len(ids) == len(set(ids)) == len(shapes) == len(states) and set(ids) == set(states),
         'Every learned parameter must retain one actual Adam state in the same order')
    need(all(group['lr'] == rate and group['weight_decay'] == weight_decay
             and tuple(group['betas']) == (.9, .999) for group in groups), 'Actual Adam recipe/LR differs')
    for key, shape in zip(ids, shapes):
        state = states[key]
        step = state['step']
        if isinstance(step, torch.Tensor):
            tensor_cpu(step, finite=True); need(step.numel() == 1, 'Adam step must be scalar')
            step = step.item()
        need(step == steps, 'Actual saved Adam counter differs from the complete outer')
        for name in ('exp_avg', 'exp_avg_sq'):
            tensor_cpu(state[name], dtype=torch.float32, shape=shape, finite=True)
    return len(states)


def validate_actor_state(saved, actor, config, outer, rank):
    import math
    need(saved['run_id'] == config['run_id'] and saved['completed_outer'] == outer and saved['rank'] == rank
         and saved['binding'] == actor['binding'] == _state_binding(config), 'Actor state belongs to another run/rank/recipe')
    need(saved['actual_adam_steps'] == 20 * outer and saved['scheduler_completed_outer'] == outer,
         'Actual actor Adam/scheduler boundary differs')
    layout = saved['optimizer_parameter_layout']
    need(len(layout) == len(saved['optimizer']['param_groups']), 'FSDP optimizer group layout differs')
    shapes = []
    for group, records in zip(saved['optimizer']['param_groups'], layout):
        need(len(group['params']) == len(records), 'FSDP flat parameter ordering is incomplete')
        for row in records:
            need(row['dtype'] == 'torch.float32' and isinstance(row['shape'], list)
                 and all(type(n) is int and n >= 0 for n in row['shape'])
                 and math.prod(row['shape']) == row['numel'], 'Saved FSDP FP32 shard shape/layout differs')
            shapes.append(row['shape'])
    count = validate_adam(saved['optimizer'], shapes, 20 * outer, learning_rate(outer), weight_decay=.01)
    scheduler = saved['scheduler']
    need(scheduler['last_epoch'] == outer and scheduler['_step_count'] == outer + 1
         and scheduler['base_lrs'] == [1e-6] and scheduler['_last_lr'] == [learning_rate(outer)],
         'Actual restored outer-counted LambdaLR state differs')
    need(saved['model_master_dtype'] == 'torch.float32'
         and saved['optimizer_layout'] == 'FSDP_FULL_SHARD_same_Qwen2DecoderLayer_wrap_world8',
         'Actor FP32 master or FSDP restore layout is unsupported')
    return dict(rank=rank, actual_adam_steps=20 * outer, actual_adam_states=count,
                scheduler_completed_outer=outer, next_learning_rate=learning_rate(outer), rng=validate_rng(saved['rng']))


def head_shapes():
    return {'block.self_attn.q_proj.weight': (2048, 2048), 'block.self_attn.q_proj.bias': (2048,),
        'block.self_attn.k_proj.weight': (256, 2048), 'block.self_attn.k_proj.bias': (256,),
        'block.self_attn.v_proj.weight': (256, 2048), 'block.self_attn.v_proj.bias': (256,),
        'block.self_attn.o_proj.weight': (2048, 2048), 'block.mlp.gate_proj.weight': (11008, 2048),
        'block.mlp.up_proj.weight': (11008, 2048), 'block.mlp.down_proj.weight': (2048, 11008),
        'block.input_layernorm.weight': (2048,), 'block.post_attention_layernorm.weight': (2048,),
        'output.weight': (1, 293 * 2048), 'output.bias': (1,)}


def validate_probe_state(saved, receipt, config, outer, sampling_actor_version):
    import torch
    binding = saved['binding']
    need(saved['run_id'] == config['run_id'] and saved['completed_outer'] == outer
         and saved['actual_adam_steps_per_rank'] == [20 * outer] * 8
         and saved['learning_rate'] == 1e-6 and saved['actor_version'] == sampling_actor_version,
         'Online probe has the wrong own-trajectory version/Adam boundary')
    need(saved['binding_sha256'] == digest(binding) == receipt['binding_sha256'], 'Probe binding hash differs')
    required = dict(role='prefix_selection200_online_correctness_probe', run_id=config['run_id'],
        encoder_revision=REVISION, question_max_tokens=273, head_max_tokens=293,
        feature_layer=27, feature_boundary='post_block', prefix_tokens=20,
        source_probe_sha256=SOURCE_PROBE_SHA, warmup_model_weights_only=True,
        warmup_optimizer_imported=False, online_initial_adam_steps=0, learning_rate=1e-6,
        group_weight=1., loss='mean_trajectory_BCE_plus_mean_G5_mixed_BCE', updates_per_outer=20,
        global_batch_average=256, actual_global_batches=[255, 260], world_size=8, accumulation=1,
        diagnostic_manifest_sha256=TEST_SHA, selected_questions=1024, candidate_questions=5120, exploration=0)
    need(all(binding.get(key) == value for key, value in required.items()), 'Unsupported probe scientific binding')
    model, shapes = saved['model'], head_shapes()
    need(set(model) == set(shapes), 'Probe must contain the full copied official block and flattened Linear')
    for name, value in model.items(): tensor_cpu(value, dtype=torch.float32, shape=shapes[name], finite=True)
    count = validate_adam(saved['optimizer'], [model[name].shape for name in model], 20 * outer, 1e-6, weight_decay=0.)
    rng = saved['rank_rng']
    need(len(rng) == 8 and [row['rank'] for row in rng] == list(range(8)), 'Probe checkpoint must contain all eight rank RNG states')
    for row in rng:
        need(row['actual_adam_steps'] == 20 * outer, 'Probe rank RNG belongs to another Adam boundary')
        validate_rng(row)
    return dict(model_fp32_tensors=len(model), adam_fp32_states=count,
                actual_adam_steps_per_rank=[20 * outer] * 8, rank_rng_count=8, learning_rate=1e-6)


def validate_pool_state(state, reservation, binding, source_pool, config, outer):
    size = binding['eligible_questions']
    need(size == 168510 == source_pool['eligible_unique_questions'] == len(source_pool['records'])
         and source_pool['canonical_rows'] == 169028 and source_pool['excluded_diagnostic_questions'] == 512
         and source_pool['canonical_train_sha256'] == config['canonical_train_sha256'], 'Actual canonical question pool source differs')
    need(state['next_outer'] == outer + 1 and state['consumed_candidates'] == 5120 * outer
         and state['candidate_questions'] == 5120 and state['seed'] == int(config.get('seed', 1729))
         and type(state['cycle']) is int and state['cycle'] >= 0 and 0 <= state['cursor'] <= size
         and state['cycle'] * size + state['cursor'] == 5120 * outer,
         'Saved candidate pool cycle/cursor does not describe the complete outer')
    need(len(state['order']) == size and all(type(i) is int for i in state['order'])
         and sorted(state['order']) == list(range(size)), 'Saved real candidate permutation is incomplete')
    need(reservation['outer'] == outer and reservation['pool_sha256'] == binding['sha256']
         and reservation['after'] == state and reservation['after_sha256'] == digest(state),
         'Saved pool boundary differs from the actual once-only reservation')
    ids, cycles = reservation['indices'], reservation['cycles']
    need(len(ids) == 5120 and len(set(ids)) == 5120 and all(type(i) is int and 0 <= i < size for i in ids)
         and len(cycles) == 5120 and all(type(cycle) is int and 0 <= cycle <= state['cycle'] for cycle in cycles),
         'Candidate reservation repeated a question or lost an actual pool index')
    return dict(eligible_questions=size, cycle=state['cycle'], cursor=state['cursor'],
                next_outer=state['next_outer'], consumed_candidates=state['consumed_candidates'],
                actual_permutation_entries=len(state['order']), source_pool_sha256=binding['sha256'])


def qwen_shapes(configuration):
    h, layers = configuration['hidden_size'], configuration['num_hidden_layers']
    intermediate, vocabulary = configuration['intermediate_size'], configuration['vocab_size']
    head = h // configuration['num_attention_heads']
    kv = configuration['num_key_value_heads'] * head
    need(configuration['model_type'] == 'qwen2' and h == 2048 and layers == 36,
         'Unsupported official3B HF architecture')
    shapes = {'model.embed_tokens.weight': (vocabulary, h), 'model.norm.weight': (h,), 'lm_head.weight': (vocabulary, h)}
    for i in range(layers):
        p = 'model.layers.%d.' % i
        shapes.update({p+'self_attn.q_proj.weight': (h, h), p+'self_attn.q_proj.bias': (h,),
            p+'self_attn.k_proj.weight': (kv, h), p+'self_attn.k_proj.bias': (kv,),
            p+'self_attn.v_proj.weight': (kv, h), p+'self_attn.v_proj.bias': (kv,),
            p+'self_attn.o_proj.weight': (h, h), p+'mlp.gate_proj.weight': (intermediate, h),
            p+'mlp.up_proj.weight': (intermediate, h), p+'mlp.down_proj.weight': (h, intermediate),
            p+'input_layernorm.weight': (h,), p+'post_attention_layernorm.weight': (h,)})
    return shapes


def audit_hf(base, actor, config):
    import torch
    from safetensors import safe_open
    directory = owned(base, actor['hf_model_dir'])
    need(actor['model_master_dtype'] == 'torch.float32' and actor['rollout_compute_dtype'] == 'bfloat16', 'Unsupported actor master/export precision')
    need(set(path.name for path in directory.iterdir() if path.is_file()) == set(actor['hf_files']), 'Actual HF file set differs from producer')
    from tokenizers import Tokenizer
    original_dir = Path(config['official_model_dir'])
    original_tokenizer = hash_file(original_dir / 'tokenizer.json')
    need(original_tokenizer['sha256'] == config['model_files_sha256']['tokenizer.json']
         and Tokenizer.from_file(str(directory/'tokenizer.json')).to_str()
         == Tokenizer.from_file(str(original_dir/'tokenizer.json')).to_str(),
         'Saved tokenizer semantics differ from the pinned official tokenizer')
    need(hash_file(original_dir/'tokenizer_config.json')['sha256'] == config['model_files_sha256']['tokenizer_config.json']
         and hash_file(original_dir/'config.json')['sha256'] == config['model_files_sha256']['config.json'],
         'Official tokenizer/architecture configuration differs from pinned initial files')
    original_tokens, saved_tokens = read(original_dir/'tokenizer_config.json'), read(directory/'tokenizer_config.json')
    need(all(original_tokens.get(key) == saved_tokens.get(key)
             for key in ('chat_template', 'bos_token', 'eos_token', 'pad_token', 'model_max_length')),
         'Saved tokenizer chat template/special token semantics differ')
    print(json.dumps(dict(stage='HF_FULL_SHA_CPU_BEGIN', path=str(directory), unix=time.time())), flush=True)
    files = [hash_file(owned(base, directory / name), producer) for name, producer in actor['hf_files'].items()]
    saved_config = read(directory / 'config.json')
    # Architecture JSON is small; official weight access below is limited to selected small tensors.
    original = read(Path(config['official_model_dir']) / 'config.json')
    for key in ('model_type', 'hidden_size', 'num_hidden_layers', 'intermediate_size', 'vocab_size',
                'num_attention_heads', 'num_key_value_heads', 'tie_word_embeddings'):
        need(saved_config.get(key) == original.get(key), 'Learned HF architecture differs from the pinned official actor')
    shapes = qwen_shapes(saved_config)
    index = read(directory / 'model.safetensors.index.json')['weight_map']
    tied_alias_omitted = False
    if saved_config.get('tie_word_embeddings', False) and 'lm_head.weight' not in index:
        # save_pretrained may omit only the explicitly tied LM-head alias. Its
        # FP32 values are then exactly the retained shared embedding parameter.
        shapes.pop('lm_head.weight'); tied_alias_omitted = True
    need(set(index) == set(shapes), 'HF shard index does not describe the complete official actor')
    change_rows, official_index = [], None
    if config.get('interaction_contract'):
        from .checkpoint_i1i6_cpu import INITIAL_CHANGE_TENSORS, CHANGE_FALLBACK, compare_official_small_tensor
        official_index_path = original_dir/'model.safetensors.index.json'
        official_index_file = hash_file(official_index_path)
        need(official_index_file['sha256'] == config['model_files_sha256']['model.safetensors.index.json'],
             'Official small-tensor index differs from the pinned initial model')
        official_index = read(official_index_path)['weight_map']
        need(INITIAL_CHANGE_TENSORS.issubset(index) and INITIAL_CHANGE_TENSORS.issubset(official_index),
             'Deterministic initial parameter-change probes missing from learned/official actor')
    seen, elements = set(), 0
    for name in sorted(set(index.values())):
        path = owned(base, directory / name); before = path.stat()
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            for key in sorted(handle.keys()):
                need(key not in seen and index.get(key) == name, 'Duplicate or misplaced HF tensor')
                tensor = handle.get_tensor(key)
                tensor_cpu(tensor, dtype=torch.float32, shape=shapes[key], finite=True)
                if official_index is not None:
                    primary_done = INITIAL_CHANGE_TENSORS.issubset({row['tensor'] for row in change_rows})
                    fallback = (key == CHANGE_FALLBACK and primary_done
                                and not any(row['nonzero_difference_elements'] for row in change_rows))
                    if key in INITIAL_CHANGE_TENSORS or fallback:
                        change_rows.append(compare_official_small_tensor(tensor, key, original_dir, official_index, config))
                seen.add(key); elements += tensor.numel(); del tensor
        need(signature(before) == signature(path.stat()), 'HF tensor file changed during actual CPU read')
    need(seen == set(shapes), 'Actor HF CPU read omitted a learned tensor')
    parameter_change = None
    if official_index is not None:
        need(INITIAL_CHANGE_TENSORS.issubset({row['tensor'] for row in change_rows}),
             'Same HF CPU traversal omitted a deterministic parameter-change probe')
        changed = sum(row['nonzero_difference_elements'] for row in change_rows)
        need(changed > 0, 'Selected official-initial tensor comparison proves no learned parameter change')
        parameter_change = dict(status='ACTUAL_PARAMETER_CHANGE_OBSERVED' if changed else 'SELECTED_TENSORS_EQUAL_INITIAL_NO_PARAMETER_CHANGE_PROVEN',
            compared_tensors=change_rows, compared_elements=sum(row['elements'] for row in change_rows),
            nonzero_difference_elements=changed, max_absolute_difference=max(row['max_absolute_difference'] for row in change_rows),
            fallback_used=any(row['tensor'] == CHANGE_FALLBACK for row in change_rows),
            official_index_small_file=official_index_file, official_full_weight_hash_repeated=False,
            learned_hf_second_read_performed=False, all_model_parameters_compared=False,
            first_outer_lr_zero_is_intentional=True, positive_lr_completed_outers=list(range(2,actor['completed_outer']+1)))
        print(json.dumps(dict(stage='HF_INITIAL_PARAMETER_CHANGE_COMPARISON_COMPLETE', result=parameter_change, unix=time.time())), flush=True)
    print(json.dumps(dict(stage='HF_FULL_SHA_CPU_COMPLETE', tensors=len(seen), unix=time.time())), flush=True)
    return dict(files=files, actual_cpu_tensor_reads=len(seen), actual_fp32_elements=elements,
                official_initial_parameter_change=parameter_change,
                tokenizer_semantics_equal=True, original_tokenizer_small_file=original_tokenizer,
                tied_lm_head_alias_omitted=tied_alias_omitted,
                master_low_bits_preserved_in_hf=True, forward_performed=False)


def _audit_complete(base, config, joint, outer):
    import torch
    torch.set_num_threads(1)
    root = base / 'outers' / ('outer-%06d' % outer)
    i1i6 = bool(config.get('interaction_contract'))
    actor, probe = joint['actor'], joint['probe']
    need(joint['run_id'] == config['run_id'] and joint['outer'] == outer
         and joint['source_commit'] == config['source_commit'], 'Joint savepoint belongs to another experiment')
    need(actor['completed_outer'] == outer and probe['completed_outer'] == outer
         and actor['run_id'] == probe['run_id'] == config['run_id']
         and actor['source_commit'] == config['source_commit']
         and actor['actual_adam_steps_per_rank'] == [20 * outer] * 8
         and probe['actual_adam_steps_per_rank'] == [20 * outer] * 8
         and actor['scheduler_completed_outer'] == outer and actor['milestone'] is True,
         'Actor/probe counters do not share a complete milestone')
    contract_proof = None
    if i1i6:
        from .checkpoint_i1i6_cpu import audit_contract
        contract_proof = audit_contract(base, config, actor, root)
    grounding_proof, attempt_config = None, config
    if i1i6:
        from .checkpoint_i1i6_cpu import audit_attempt_grounding
        grounding_proof, attempt_config = audit_attempt_grounding(base, config, actor, root, outer)
    metadata = {}
    paths = {'joint': root / 'complete.json', 'rollout': root / 'full/complete.json', 'actor_rollout': root / 'actor-selection/complete.json',
        'actor_selection': root / 'actor-selection/selection.json',
        'prefix': root / 'prefix/complete.json',
        'own_em': root / 'full/own-em.jsonl', 'score_manifest': root / 'probe-score/manifest.json',
        'selection': root / 'probe-score/selection.json', 'update': root / 'probe-update/complete.json',
        'metrics': root / 'probe-update/metrics.json',
        'pool_binding': base / 'question-pool/binding.json', 'reservation': root / 'pool-reservation.json',
        'candidate_selection': root / 'selection.json', 'questions': root / 'questions.json'}
    for name, path in paths.items(): metadata[name] = hash_file(owned(base, path))
    rollout = read(paths['rollout']); prefix = read(paths['prefix'])
    actor_rollout, actor_selection = read(paths['actor_rollout']), read(paths['actor_selection'])
    manifest = read(paths['score_manifest']); update = read(paths['update'])
    selection, candidates, questions = read(paths['selection']), read(paths['candidate_selection']), read(paths['questions'])
    validate_actor_rollout_binding(actor, paths['actor_rollout'], metadata['actor_rollout'])
    need(joint['rollout'] == rollout
         and rollout['actor_version'] == actor['sampling_actor_version']
         and rollout['questions'] == 1024 and rollout['trajectories'] == 5120,
         'Actual actor savepoint is not bound to its own same-prefix completed rollout')
    need(update['checkpoint'] == probe and update['status'] == 'SELECTED1024G5_ONLINE_PROBE_20_ADAM_COMPLETE'
         and update['actual_adam_steps_per_rank'] == [20 * outer] * 8 and update['new_adam_steps_per_rank'] == 20
         and update['scored_features_manifest_sha256'] == metadata['score_manifest']['sha256']
         and update['selection_sha256'] == joint['selection_sha256'] == metadata['selection']['sha256']
         and update['own_em_sha256'] == rollout['own_em']['sha256'] == metadata['own_em']['sha256']
         and update['metrics_sha256'] == metadata['metrics']['sha256']
         and joint['probe_metrics'] == read(paths['metrics']), 'Probe savepoint lost its own scored-prefix/outcome binding')
    need(selection['actor_version'] == manifest['actor_version'] == rollout['actor_version']
         and selection['prefix_scan_sha256'] == manifest['prefix_scan_sha256'] == rollout['prefix_scan_sha256']
         and selection['probe_before_checkpoint_sha256'] == manifest['probe_before_checkpoint_sha256']
         and len(selection['selected']) == 1024 and manifest['count'] == 5120,
         'Selected G5 prefixes/versions differ across score and continuation')
    need(prefix['run_id'] == config['run_id'] and prefix['outer'] == outer
         and prefix['questions'] == 5120 and prefix['trajectories'] == 25600
         and prefix['actor_version'] == rollout['actor_version']
         and prefix['scan_sha256'] == manifest['prefix_scan_sha256']
         and prefix['candidate_questions_sha256'] == metadata['questions']['sha256'],
         'Actual5120G5 prefix generation did not use this fixed candidate question bundle')
    selected_ids = {tid: str(row['qid']) for row in selection['selected'] for tid in row['trajectory_ids']}
    need(len(selected_ids) == 5120 and len({str(row['qid']) for row in selection['selected']}) == 1024,
         'Milestone selection is not complete unique512G5')
    score_path = owned(base, root/'probe-score/candidate-scores.jsonl')
    metadata['candidate_scores'] = hash_file(score_path)
    scores = [json.loads(line) for line in score_path.read_text().splitlines()]
    need(metadata['candidate_scores']['sha256'] == manifest['candidate_scores_sha256'] and len(scores) == 5120
         and sorted(row['candidate_ordinal'] for row in scores) == list(range(5120))
         and sorted(scores, key=lambda row: (-row['q'], row['candidate_ordinal']))[:1024] == selection['selected'],
         'Actual prefix selection is not stable Top1024 of all5120 original scores')
    for row in scores:
        z = row['trajectory_logits']
        need(len(z) == 5 and all(math.isfinite(value) for value in z), 'Invalid original prefix logits')
        p = [1/(1+math.exp(-value)) if value >= 0 else math.exp(value)/(1+math.exp(value)) for value in z]
        q = 1-math.prod(p)-math.prod(1-value for value in p)
        need(all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(p,row['trajectory_probabilities']))
             and math.isclose(q,row['q'],rel_tol=1e-12,abs_tol=1e-12), 'Original score does not implement1-prod(p)-prod(1-p)')
    labels = [json.loads(line) for line in paths['own_em'].read_text().splitlines()]
    need(len(labels) == 5120 and len({row['trajectory_id'] for row in labels}) == 5120
         and all(selected_ids.get(row['trajectory_id']) == str(row['qid'])
                 and type(row['strict_em']) is int and row['strict_em'] in (0, 1)
                 and row['actor_version'] == rollout['actor_version']
                 and row['prefix_scan_sha256'] == manifest['prefix_scan_sha256']
                 and row['selection_sha256'] == metadata['selection']['sha256'] for row in labels),
         'Actual own-trajectory outcomes do not cover the selected sameG5')
    diversity = validate_selection(selection, labels, actor_selection, rollout, actor_rollout, joint, metadata)
    native_records = audit_rollout_records(base, rollout, actor_rollout, actor_selection, labels, i1i6=i1i6)
    actor_policy_counts = native_records.pop('actor_policy_counts')
    rank_rows = []
    for rank in range(8):
        path = owned(base, root / 'probe-update' / ('rank-%d-updates.jsonl' % rank))
        metadata['probe_rank_updates_%d' % rank] = hash_file(path)
        rank_rows.append([json.loads(line) for line in path.read_text().splitlines()])
    path = owned(base, root / 'probe-update/global-loss-updates.jsonl')
    metadata['probe_global_updates'] = hash_file(path)
    probe_clock = validate_probe_clock(read(paths['metrics']), rank_rows,
                                      [json.loads(line) for line in path.read_text().splitlines()], outer)
    binding = read(paths['pool_binding']); source_path = Path(config['source_question_pool'])
    source_proof = hash_file(source_path)
    need(source_proof['sha256'] == binding['sha256'] and binding['source'] == str(source_path)
         and binding['fixed_test_features_manifest'] == config['fixed_test_manifest_sha256'] == TEST_SHA,
         'Actual candidate source pool/fixed diagnostic binding differs')
    source_pool = read(source_path)
    snapshot = owned(base, joint['pool_snapshot']); metadata['pool_snapshot'] = hash_file(snapshot)
    need(metadata['pool_snapshot']['sha256'] == joint['pool_sha256'], 'Joint pool snapshot changed')
    pool = read(snapshot); reservation = read(paths['reservation'])
    pool_summary = validate_pool_state(pool, reservation, binding, source_pool, config, outer)
    for key in ('fixed_test_qids_sha256', 'fixed_test_normalized_text_sha256'):
        need(binding[key] == source_pool[key] == candidates[key], 'Candidate diagnostic exclusion fingerprint differs')
    need(candidates['questions'] == questions['questions'] == 5120
         and questions['selection_sha256'] == metadata['candidate_selection']['sha256']
         and [row['candidate_ordinal'] for row in candidates['rows']] == list(range(5120)),
         'Actual candidate question bundle/order is incomplete')
    from reproduction.prefix20_online.pool import identity
    need(len(questions['rows']) == 5120, 'Actual candidate prompt/gold count differs')
    for ordinal, row in enumerate(candidates['rows']):
        i = reservation['indices'][ordinal]; original = source_pool['records'][i]
        need(row['pool_index'] == i and row['pool_cycle'] == reservation['cycles'][ordinal]
             and all(row.get(key) == value for key, value in original.items()), 'Actual reserved candidate identity differs from its source pool')
        actual = questions['rows'][ordinal]
        actual_identity = identity(actual)
        need(all(actual.get(key) == value for key, value in row.items())
             and all(actual_identity[key] == original[key] for key in actual_identity),
             'Actual persisted prompt/gold question identity differs from the reserved canonical source')
    state_dir = owned(base, actor['state_path'])
    need(read(state_dir / 'complete.json') == actor and actor['learning_rate_used'] == learning_rate(outer-1)
         and actor['next_learning_rate'] == learning_rate(outer), 'Actor producer receipt or original scheduler clock differs')
    hf = audit_hf(base, actor, config)
    need(len(actor['rank_states']) == len(actor['rank_receipts']) == 8, 'Actor milestone must contain8 rank-local full states')
    ranks = []
    for rank, item in enumerate(actor['rank_states']):
        path = owned(base, item['path']); need(path == state_dir / ('rank-%d.pt' % rank), 'Rank state escapes its actual actor attempt')
        saved, file_proof = load_torch_cpu(path, item)
        summary = validate_actor_state(saved, actor, config, outer, rank)
        validate_actor_rollout_binding(saved, paths['actor_rollout'], metadata['actor_rollout'])
        receipt = actor['rank_receipts'][rank]
        ledger = owned(base, receipt['ledger_path']); ledger_proof = hash_file(ledger)
        need(receipt['rank'] == rank and receipt['outer'] == outer and receipt['new_adam_steps'] == 20
             and receipt['actual_adam_steps'] == 20 * outer and receipt['opportunities'] == 20
             and receipt['scheduler_completed_outer'] == outer and receipt['next_learning_rate'] == learning_rate(outer)
             and ledger_proof['sha256'] == receipt['ledger_sha256'], 'Actor milestone rank receipt differs from its original update ledger')
        all_records = [json.loads(line) for line in ledger.read_text().splitlines()]
        writes = [row for row in all_records if row['event'] == 'ACTUAL_ADAM_WRITE']
        need(len(writes) == 20 and all(row['rank'] == rank and row['outer'] == outer
             and row['opportunity'] == j and row['writes_this_attempt'] == j + 1
             and row['actual_adam_steps'] == 20 * (outer - 1) + j + 1
             and row['actual_learning_rate'] == learning_rate(outer-1) for j, row in enumerate(writes)),
             'Original actor ledger does not prove20 actual rank-local writes')
        if i1i6:
            from .checkpoint_i1i6_cpu import validate_pair_writes
            summary['logical_pair_objective'] = validate_pair_writes(writes, all_records, actor_policy_counts, config['seed'], outer, rank)
        summary.update(file=file_proof, actual_update_ledger=ledger_proof)
        ranks.append(summary); del saved; gc.collect()
    driver_path = owned(base, actor['driver_state']['path'])
    driver, driver_file = load_torch_cpu(driver_path, actor['driver_state'])
    need(driver['run_id'] == config['run_id'] and driver['completed_outer'] == outer
         and driver['actor_version'] == actor['actor_version']
         and driver['candidate_pool_state']['value'] == pool
         and driver['candidate_pool_state']['sha256'] == hashlib.sha256(canonical(pool)+b'\n').hexdigest(),
         'Actual actor driver state did not preserve the real candidate boundary')
    if i1i6:
        from .checkpoint_i1i6_cpu import validate_driver_binding
        validate_driver_binding(driver, base, config, attempt_config, outer, metadata['score_manifest']['sha256'])
    previous_probe = driver['prefix_probe_receipt']['value']
    need(previous_probe['run_id'] == config['run_id'] and previous_probe['completed_outer'] == outer - 1
         and previous_probe['actual_adam_steps_per_rank'] == [20 * (outer - 1)] * 8
         and previous_probe['sha256'] == manifest['probe_before_checkpoint_sha256'],
         'Actor driver lost its actual selection-time probe state/version')
    validate_rng(driver['rng']); del driver; gc.collect()
    probe_path = owned(base, base / 'probe' / ('head-outer-%04d.pt' % outer))
    probe_snapshot = owned(base, joint['probe_snapshot'])
    need(signature(probe_path.stat()) == signature(probe_snapshot.stat()), 'Probe milestone alias and joint snapshot are not the same immutable inode')
    saved_probe, probe_file = load_torch_cpu(probe_path, probe)
    probe_summary = validate_probe_state(saved_probe, probe, config, outer, rollout['actor_version'])
    test_path = Path(config['fixed_test_features']) / 'manifest.json'
    fixed_test_proof = hash_file(test_path)
    fixed_test = read(test_path)
    need(fixed_test_proof['sha256'] == config['fixed_test_manifest_sha256'] == TEST_SHA
         and saved_probe['binding']['encoder_files_sha256'] == manifest['encoder_files_sha256']
         == fixed_test['spec']['encoder_files_sha256']
         and fixed_test['spec_sha'] == digest(fixed_test['spec']),
         'Probe checkpoint changed the frozen official encoder/fixed diagnostic manifest')
    need(saved_probe['scored_features_manifest_sha256'] == metadata['score_manifest']['sha256']
         and saved_probe['own_em_sha256'] == metadata['own_em']['sha256'], 'CPU-loaded probe lost its actual training data binding')
    del saved_probe; gc.collect()
    prior = Path(config['probe_source_cpu_proof']); prior_file = hash_file(prior)
    need(prior_file['sha256'] == config['probe_source_cpu_proof_sha256'], 'Pinned warmup source small CPU proof changed')
    source = read(prior)
    need(source['status'] == 'JOINT100_CHECKPOINT_INDEPENDENT_FULL_SHA_CPU_PASS'
         and source['epoch'] == 100 and source['checkpoint_sha256'] == SOURCE_PROBE_SHA
         and source['independent_full_file_sha_performed'] is True and source['cpu_load_performed'] is True,
         'Unsupported source warmup CPU proof; never infer PASS from a producer SHA alone')
    return dict(i1i6_contract=contract_proof, actor_hf=hf, actor_rank_states=ranks, actor_driver_file=driver_file,
        actual_grounding_binding=grounding_proof,
        probe=dict(file=probe_file, summary=probe_summary), pool=pool_summary,
        strict_em_diversity=diversity, original_native_records=native_records, probe_clock=probe_clock,
        source_pool_file=source_proof, reused_prior_warmup_small_proof=prior_file,
        fixed_test_manifest_file=fixed_test_proof, own_selected_trajectory_count=len(labels), metadata=metadata)


def _once(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False); handle.flush(); os.fsync(handle.fileno())


def audit(base, outer, *, audit_version="v1", correction_config=None):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Run with CUDA_VISIBLE_DEVICES= for CPU-only audit')
    base = Path(base)
    need(base.is_absolute() and base.parent == Path('/mnt/data/shaoshitong')
         and base.name.startswith('search-g1-prefix-diversity200-') and base.resolve(strict=True) == base,
         'Audit only this current isolated prefix-selection200 run')
    need(type(outer) is int and outer in MILESTONES, 'Only actual5/50/100/150/200 milestones are supported')
    config_path = base / 'preparation/launch-config.json'; config = read(config_path)
    need(config['base'] == str(base) and config['model_revision'] == REVISION
         and config['outers'] == 200 and config['world_size'] == 8, 'Current run scientific identity differs')
    proof_path = base / 'preparation' / (proof_name(outer, audit_version)+'.json')
    intent = proof_path.with_name(proof_path.stem + '-intent.json')
    joint_path = base / 'outers' / ('outer-%06d' % outer) / 'complete.json'
    with proof_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if proof_path.exists():
            prior = read(proof_path)
            need(prior['run_id'] == config['run_id'] and prior['outer'] == outer, 'Existing once proof belongs to another run/milestone')
            return dict(status='EXISTING_CPU_MILESTONE_PROOF_NO_REPLAY', proof_status=prior['status'],
                proof=hash_file(proof_path), outer=outer, independent_large_file_read_this_call=False,
                gpu_forward_performed=False, backup_performed=False)
        if intent.exists():
            return dict(status='EXISTING_CPU_MILESTONE_INTENT_NO_BLIND_REPLAY', outer=outer, intent_path=str(intent), training_modified=False)
        if not joint_path.is_file():
            return dict(status='PENDING_ACTUAL_JOINT_MILESTONE_NO_INTENT', outer=outer, missing=str(joint_path), once_intent_created=False)
        correction = None
        if audit_version == 'v2':
            need(correction_config and correction_config['outer'] == outer, 'v2 requires explicit predecessor correction binding')
            correction = verify_predecessor(base, correction_config)
        else:
            need(correction_config is None, 'v1 must not accept a correction configuration')
        joint = read(joint_path); joint_proof = hash_file(joint_path); config_proof = hash_file(config_path)
        _once(intent, dict(run_id=config['run_id'], outer=outer, joint_complete_sha256=joint_proof['sha256'],
            config_sha256=config_proof['sha256'], source_commit=config['source_commit'], audit_version=audit_version,
            corrected_predecessor=correction, started_unix=time.time()))
        started = time.monotonic()
        try:
            result = _audit_complete(base, config, joint, outer)
            need(hash_file(config_path)['sha256'] == config_proof['sha256']
                 and hash_file(joint_path)['sha256'] == joint_proof['sha256'],
                 'Pinned launch configuration or immutable joint changed during CPU acceptance')
            proof = dict(status=STATUS, run_id=config['run_id'], outer=outer, source_commit=config['source_commit'],
                joint_complete_sha256=joint_proof['sha256'], config_sha256=config_proof['sha256'],
                actor_actual_adam_steps_per_rank=[20 * outer] * 8, probe_actual_adam_steps_per_rank=[20 * outer] * 8,
                independent_full_file_sha_performed=True, actual_cpu_deserialization_performed=True,
                details=result)
        except BaseException as error:
            proof = dict(status='PREFIX_SELECTION200_CPU_MILESTONE_AUDIT_FAILED_TRAINING_UNCHANGED',
                run_id=config['run_id'], outer=outer, joint_complete_sha256=joint_proof['sha256'],
                error_type=type(error).__name__, error=str(error),
                independent_full_file_sha_performed=False, actual_cpu_deserialization_performed=False)
        proof.update(audit_version=audit_version, corrected_predecessor=correction, gpu_forward_performed=False, cuda_rng_applied=False, independent_backup_performed=False,
            training_modified=False, old_weight_files_read=bool(config.get('interaction_contract')),
            historical_training_checkpoint_weight_files_read=False,
            official_initial_small_tensor_reads_performed=bool(config.get('interaction_contract')),
            large_payload_transferred_to_Mac=False,
            reader_source_sha256=hash_file(__file__)['sha256'], elapsed_seconds=time.monotonic() - started)
        _once(proof_path, proof)
        return dict(status=proof['status'], outer=outer, proof=hash_file(proof_path),
            actor_actual_adam_steps_per_rank=proof.get('actor_actual_adam_steps_per_rank'),
            probe_actual_adam_steps_per_rank=proof.get('probe_actual_adam_steps_per_rank'),
            error=proof.get('error'), gpu_forward_performed=False, independent_backup_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True); parser.add_argument('--outer', required=True, type=int)
    parser.add_argument('--audit-version', choices=('v1','v2'), default='v1')
    parser.add_argument('--correction-config', help='Explicit small v2 controller configuration JSON')
    args = parser.parse_args(); result = audit(args.base, args.outer, audit_version=args.audit_version,
        correction_config=json.loads(args.correction_config) if args.correction_config else None)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if 'FAILED' in result['status'] or ('proof_status' in result and result['proof_status'] != STATUS):
        raise SystemExit(1)


if __name__ == '__main__': main()
