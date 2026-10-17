"""Small I1-I6 evidence checks added to the existing once-only full CPU reader.

No forward/CUDA call. Source identity proves configured shared BF16 execution;
actual QA execution is only claimed when a completed original QA receipt exists.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

OBJECTIVE = 'mean_of_eight_rank_eight_logical_two_row_policy_means'
RUNTIME = '68d2d131aee91d194c487f54a0204f53397d87ea'
RUNTIME_PINS = {RUNTIME: 'checkpoint_i1i6_runtime_files.json',
    '33ef232e605f7864086d1685d9a4d8ea0d67f4ab': 'checkpoint_i1i6_runtime_files_33ef.json'}


def finite_numbers(value):
    """Every recorded numeric diagnostic must be finite, including early writes."""
    if isinstance(value, dict): return all(finite_numbers(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(finite_numbers(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def validate_pair_writes(writes, all_records, policy_counts, seed, outer, rank):
    from .actor import rank_plan
    from .checkpoint_cpu_once import need
    need(len(writes) == 20, 'I1 must inspect all20 original writes')
    for row, ids in zip(writes, rank_plan(seed, outer)[rank]):
        need(row['objective'] == OBJECTIVE and row['local_trajectories'] == 16
             and row['global_mini'] == 128 and row['grad_norm'] >= 0
             and finite_numbers(row), 'I1 original logical-pair objective/mini/loss/gradient differs')
        expected = [sum(policy_counts[i] for i in ids[j:j+2]) for j in range(0,16,2)]
        parts = defaultdict(list)
        for part in row['metrics']:
            pair = part['logical_pair']; need(type(pair) is int and 0 <= pair < 8, 'Invalid logical pair')
            total = expected[pair]; mass = part['policy_tokens']
            need(type(mass) is int and 0 <= mass <= total and part['pair_policy_tokens'] == total
                 and math.isclose(part['gradient_weight'], mass/(8*total) if total else 0., rel_tol=1e-12, abs_tol=1e-12)
                 and all(math.isfinite(part[k]) for k in ('loss','clip_fraction','old_policy_kl')),
                 'I1 physical micro weight lost its original two-row token denominator')
            parts[pair].append(part)
        need(set(parts) == set(range(8)) and all(sum(p['policy_tokens'] for p in parts[j]) == n for j,n in enumerate(expected)),
             'I1 original eight logical pairs were not each consumed exactly once')
    monitors = [r for r in all_records if r['event'] == 'POLICY_MONITOR']
    need([(r['stage'],r['optimizer_writes']) for r in monitors] ==
         [(stage,writes) for n in range(1,21) for stage,writes in (('before_minibatch',n-1),('after_minibatch',n))],
         'I6 before/after monitor does not cover all20 actual writes')
    need(all(finite_numbers(r['report']) and r['report']['invalid_count'] == 0 for r in monitors),
         'I6 original before/after monitor recorded nonfinite values')
    return dict(original_two_row_means_verified=True, physical_micro_weights_recomputed_from_native_masks=True,
                actual_writes=20, logical_pairs_per_write=8, before_after_monitor_records=40)


def native_budget(row):
    from .checkpoint_cpu_once import need
    need(len(row['initial_prompt_token_ids']) <= 2048, 'I2 initial prompt exceeds shared budget')
    for turn in row['behavior_turns']:
        need(len(turn['actual_request_token_ids'])+500 <= 4096
             and len(turn['raw_generated_token_ids']) <= 500
             and len(turn['observation_token_ids']) <= 500,
             'I2 actual native request/action/observation exceeds the shared contract')


def validate_resume_origin(base, config, actor):
    """33ef import result stores Adam counters; terminal and actual joint store scheduler."""
    from .checkpoint_cpu_once import owned, read, hash_file, need
    spec = config['resume_joint']; source_base = Path(spec['source_base'])
    need(source_base != base and source_base.parent == base.parent
         and config['initial_official_fresh_run_source'] == str(source_base),
         'I1-I6 continuation lost its original fresh official run')
    initial_path = source_base/'preparation/launch-config.json'; initial = read(initial_path)
    resume_path = owned(base, base/'preparation/joint-resume-receipt.json')
    terminal_path = owned(base, base/'preparation/joint-source-terminal-proof.json')
    resume, terminal = read(resume_path), read(terminal_path)
    source_joint_path = owned(source_base, source_base/'outers'/('outer-%06d' % spec['completed_outer'])/'complete.json')
    source_joint = read(source_joint_path); recovered_actor, recovered_probe = source_joint['actor'], source_joint['probe']
    need(resume['status'] == 'COMPLETE_JOINT_BOUNDARY_IMPORTED_NO_RECOMPUTATION'
         and resume['run_id'] == initial['run_id'] == config['run_id']
         and resume['source_base'] == terminal['source_base'] == str(source_base)
         and resume['completed_outer'] == terminal['completed_outer'] == spec['completed_outer']
         and resume['terminal_proof_sha256'] == hash_file(terminal_path)['sha256']
         and terminal['source_config_sha256'] == hash_file(initial_path)['sha256']
         and initial['fresh_actor_from_official'] is True and initial['source_commit'] == RUNTIME
         and initial['actor_version'] == config['actor_version']
         and initial['grounding_initial_probe_version'] == config['grounding_initial_probe_version']
         and resume['actor_steps'] == resume['probe_steps'] == [20*spec['completed_outer']]*8
         and terminal['scheduler_completed_outer'] == recovered_actor['scheduler_completed_outer'] == spec['completed_outer']
         and terminal['actor_steps'] == recovered_actor['actual_adam_steps_per_rank'] == resume['actor_steps']
         and terminal['probe_steps'] == recovered_probe['actual_adam_steps_per_rank'] == resume['probe_steps']
         and source_joint['run_id'] == recovered_actor['run_id'] == recovered_probe['run_id'] == config['run_id']
         and source_joint['outer'] == recovered_actor['completed_outer'] == recovered_probe['completed_outer'] == spec['completed_outer']
         and terminal['joint_sha256'] == hash_file(source_joint_path)['sha256']
         and spec['completed_outer'] < actor['completed_outer'],
         'I1-I6 continuation is not a proven intact joint boundary from the fresh official run')
    science = ('model_revision','model_files_sha256','candidate_questions','selected_questions',
        'actor_selected_questions','responses_per_question','prefix_policy_tokens','seed',
        'official_model_dir','canonical_train_sha256','validation_questions_sha256',
        'final_questions_sha256','grounding_calibration_sha256','interaction_contract',
        'grounding_manifest_sha256','grounding_refit_rounds','policy_monitor',
        'probe_learning_rate','probe_updates_per_outer','probe_epochs_per_outer',
        'probe_global_batch_average','actor_global_mini_batch','actor_ppo_epochs',
        'fixed_test_manifest_sha256')
    need(all(initial[key] == config[key] for key in science), 'I1-I6 resumed scientific identity differs')
    origin = dict(initial_config=hash_file(initial_path), resume=hash_file(resume_path),
                  terminal=hash_file(terminal_path), original_joint=hash_file(source_joint_path),
                  scheduler_completed_outer=terminal['scheduler_completed_outer'], completed_outer=spec['completed_outer'])
    return origin


def audit_contract(base, config, actor, root):
    from .checkpoint_cpu_once import hash_file, read, need
    from .interaction import ROLE, contract
    need(config['source_commit'] in RUNTIME_PINS and config['interaction_contract'] == ROLE
         and config['actor_version'] == config['run_id']+'-official0'
         and config['grounding_initial_probe_version'] == config['grounding_probe_version']
         and actor['binding']['grounding_probe_version'] == config['grounding_initial_probe_version']
         and actor['binding']['loss_reduction'] == OBJECTIVE and actor['rollout_compute_dtype'] == 'bfloat16',
         'I1-I6 new-run source/fresh-start/stable-initial-grounding/BF16 binding differs')
    origin = None
    if config['fresh_actor_from_official'] is not True:
        origin = validate_resume_origin(base, config, actor)
    else:
        need(not config.get('resume_joint'), 'Fresh actor must not claim a joint import')
    pinned = read(Path(__file__).with_name(RUNTIME_PINS[config['source_commit']]))
    source_files = read(base/'preparation/source-files.json')
    source = {}
    for name, expected in pinned.items():
        proof = hash_file(base/'repo'/name)
        need(proof['sha256'] == expected == source_files[name], 'Frozen I1-I6 source differs: '+name)
        source[name] = proof
    splits = {}
    for key in ('canonical_train','validation_questions','final_questions','grounding_calibration_questions','grounding_manifest'):
        proof = hash_file(config[key]); expected_key = 'grounding_calibration_sha256' if key == 'grounding_calibration_questions' else key+'_sha256'
        need(proof['sha256'] == config[expected_key], 'I1-I6 pinned source/split differs: '+key)
        splits[key] = proof
    # QA is sequential and may still be pending while this independent reader runs.
    qa_path = root/'validation/complete.json'
    qa = dict(status='NOT_COMPLETED_AT_CPU_READ_NO_ADDITIONAL_QA', actual_qa_execution_verified=False)
    if qa_path.is_file():
        receipt = read(qa_path)
        need(receipt['run_id'] == config['run_id'] and receipt['actor_version'] == actor['actor_version']
             and receipt['evaluation_contract'] == contract(0.,1)
             and receipt['qa_count'] == 512 and receipt['question_source_sha256'] == config['validation_questions_sha256']
             and receipt['actual_worker_exit_codes'] == {str(r):0 for r in range(8)}, 'I2 original QA completion contract differs')
        ranks=[]
        for item in receipt['rank_receipts']:
            proof=hash_file(item['path']); need(proof['sha256']==item['sha256'], 'I2 original QA rank receipt differs')
            value=read(item['path'])
            need(value['evaluation_contract']==contract(0.,1) and value['actual_compute_dtype']=='torch.bfloat16', 'I2 actual QA dtype/contract differs')
            ranks.append(proof)
        qa=dict(status='ORIGINAL_QA_SHARED_BF16_RECEIPTS_VERIFIED',actual_qa_execution_verified=True,
                complete=hash_file(qa_path),rank_receipts=ranks)
    return dict(frozen_source_files=source, pinned_source_splits=splits, original_fresh_run=origin,
                stable_initial_grounding_binding=True, shared_bf16_train_qa_source_contract_verified=True,
                train_native_request_budgets_verified=True, original_qa=qa,
                cuda_executor_exact_restore_verified=False)


def audit_attempt_grounding(base, config, actor, root, outer):
    """The saved attempt uses the preceding refit; stable binding retains round0."""
    from .checkpoint_cpu_once import read, need, hash_file, owned
    from .cost_snapshot import grounding_binding
    from .actor import REWARD_ENV, _state_binding
    state_dir = owned(base, actor['state_path'])
    prefix = 'outer-%04d-attempt-' % outer
    need(state_dir.parent == base/'actor/outers' and state_dir.name.startswith(prefix),
         'Actor save does not identify its original current-base attempt')
    attempt_id = state_dir.name[len(prefix):]
    need(bool(attempt_id) and all(c in '0123456789abcdef' for c in attempt_id), 'Invalid actor attempt identity')
    attempt_path = owned(base, root/('actor-attempt-%s-config.json' % attempt_id))
    attempt = read(attempt_path)
    need(attempt['actor_attempt_id'] == attempt_id
         and all(attempt.get(key) == value for key,value in config.items()
                 if key not in ('actor_global_micro_batch_size', 'grounding_probe_version'))
         and _state_binding(attempt) == actor['binding'] == _state_binding(config),
         'Original actor attempt changed the pinned launch configuration or stable initial binding')
    need(attempt.get('actor_global_micro_batch_size', 128) == actor['actual_global_micro']
         and actor['actual_global_micro'] == 8*actor['actual_local_micro']
         and actor['actual_global_micro']*actor['gradient_accumulation'] == 128,
         'Saved actor micro/accumulation differs from its original attempt')
    reward_round = min(150, ((outer-1)//50)*50)
    expected = grounding_binding(base, config, reward_round)
    need(attempt['grounding_probe_version'] == expected['version']
         and attempt['grounding_active_manifest'] == expected['manifest'],
         'Original actor attempt used the wrong actual grounding refit clock')
    reward = actor['reward']
    need(reward['version'] == expected['version'] and reward['trajectories'] == 2560
         and reward['actual_reward_world_size'] == 8 and reward['prefix_correctness_head_used_for_reward'] is False
         and len(reward['ranks']) == 8 and all(row['rank'] == rank
             and row['version'] == expected['version'] and row['reward_environment'] == REWARD_ENV
             and row['prefix_correctness_head_used_for_reward'] is False and finite_numbers(row)
             for rank,row in enumerate(reward['ranks'])) and finite_numbers(reward),
         'Actual original soft reward recipe/probe binding or finite scores changed')
    proofs = dict(attempt=hash_file(attempt_path), manifest=hash_file(expected['manifest']))
    for key in ('completed_receipt','adoption_receipt'):
        if expected[key]: proofs[key] = hash_file(expected[key])
    if reward_round:
        manifest = read(expected['manifest'])
        need(manifest['calibration_data_sha256'] == config['grounding_calibration_sha256'],
             'Refitted grounding uses another calibration split')
        for name,key in (('commit_probe.pkl','commit'),('boundary_probe.pkl','boundary')):
            proof = hash_file(manifest[key]); need(proof['sha256'] == manifest['head_sha256'][name],
                'Actually used fitted grounding head changed')
            proofs[key] = proof
    return dict(round=reward_round, version=expected['version'], files=proofs), attempt


# All probes are whole small learned tensors, never an embedding or LM-head
# matrix. If these first-block probes are unchanged, inspect only the final
# RMSNorm already encountered later in the same learned-HF CPU traversal.
INITIAL_CHANGE_TENSORS = frozenset({
    'model.layers.0.self_attn.q_proj.bias',
    'model.layers.0.self_attn.k_proj.bias',
    'model.layers.0.self_attn.v_proj.bias',
    'model.layers.0.input_layernorm.weight',
    'model.layers.0.post_attention_layernorm.weight',
})
CHANGE_FALLBACK = 'model.norm.weight'


def compare_official_small_tensor(learned, key, official_dir, official_index, config):
    """CPU read only one chosen official tensor, promoted to the FP32 master."""
    import torch
    from safetensors import safe_open
    from .checkpoint_cpu_once import need, signature, tensor_cpu
    name = official_index[key]
    need(Path(name).name == name and name in config['model_files_sha256'],
         'Official initial tensor escapes its pinned weight shard')
    path = official_dir/name; before = path.stat()
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        original = handle.get_tensor(key)
        tensor_cpu(original, shape=learned.shape, finite=True)
        original_dtype = str(original.dtype)
        promoted = original.to(dtype=torch.float32)
        difference = learned - promoted
        tensor_cpu(difference, dtype=torch.float32, shape=learned.shape, finite=True)
        changed = int(torch.count_nonzero(difference).item())
        maximum = float(difference.abs().max().item())
        result = dict(tensor=key, elements=learned.numel(), nonzero_difference_elements=changed,
                      max_absolute_difference=maximum, learned_dtype=str(learned.dtype),
                      official_saved_dtype=original_dtype, comparison_dtype='torch.float32',
                      official_file=str(path), official_file_pinned_sha256=config['model_files_sha256'][name],
                      official_full_file_hash_repeated=False, official_tensor_only_cpu_read=True)
    need(signature(before) == signature(path.stat()), 'Official small tensor source changed during CPU read')
    return result


def validate_driver_binding(driver, base, config, attempt, outer, score_sha):
    from .checkpoint_cpu_once import need
    need(driver['driver_metadata_binding'] == dict(source='current_execution_base', outer=outer,
             run_id=config['run_id'], score_manifest_sha256=score_sha)
         and driver['candidate_pool_state']['path'] == str(base/'question-pool/state.json')
         and driver['prefix_probe_receipt']['path'] == str(base/'probe/latest.json')
         and driver['prefix_probe_receipt']['sha256'] == hashlib.sha256(json.dumps(driver['prefix_probe_receipt']['value'],
             sort_keys=True, allow_nan=False).encode()).hexdigest()
         and driver['driver_state'] == attempt.get('driver_state', {})
         and driver['grounding_probe_version'] == attempt['grounding_probe_version'],
         'I4 driver did not bind current execution metadata and actual attempt reward version')
