"""Small I1-I6 evidence checks added to the existing once-only full CPU reader.

No forward/CUDA call. Source identity proves configured shared BF16 execution;
actual QA execution is only claimed when a completed original QA receipt exists.
"""
from collections import defaultdict
import json
import math
from pathlib import Path

OBJECTIVE = 'mean_of_eight_rank_eight_logical_two_row_policy_means'
RUNTIME = '68d2d131aee91d194c487f54a0204f53397d87ea'


def validate_pair_writes(writes, all_records, policy_counts, seed, outer, rank):
    from .actor import rank_plan
    from .checkpoint_cpu_once import need
    for row, ids in zip(writes, rank_plan(seed, outer)[rank]):
        need(row['objective'] == OBJECTIVE and row['local_trajectories'] == 16
             and row['global_mini'] == 128, 'I1 original logical-pair objective/mini differs')
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


def audit_contract(base, config, actor, root):
    from .checkpoint_cpu_once import hash_file, read, need
    from .interaction import ROLE, contract
    need(config['source_commit'] == RUNTIME and config['interaction_contract'] == ROLE
         and config['fresh_actor_from_official'] is True
         and config['actor_version'] == config['run_id']+'-official0'
         and config['grounding_initial_probe_version'] == config['grounding_probe_version']
         and actor['binding']['grounding_probe_version'] == config['grounding_initial_probe_version']
         and actor['binding']['loss_reduction'] == OBJECTIVE and actor['rollout_compute_dtype'] == 'bfloat16',
         'I1-I6 new-run source/fresh-start/stable-initial-grounding/BF16 binding differs')
    pinned = read(Path(__file__).with_name('checkpoint_i1i6_runtime_files.json'))
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
    return dict(frozen_source_files=source, pinned_source_splits=splits,
                stable_initial_grounding_binding=True, shared_bf16_train_qa_source_contract_verified=True,
                train_native_request_budgets_verified=True, original_qa=qa,
                cuda_executor_exact_restore_verified=False)


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
