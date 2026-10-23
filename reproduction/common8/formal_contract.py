"""The new formal200 contract, distinct from immutable constant300 diagnostics."""
import copy
import hashlib
from pathlib import Path

from reproduction.common8.policy_calibration_plan import MATHEMATICS, canonical_bytes
from reproduction.common8.topology import world_size, actor_global_micro, actor_logical_micro, execution_job_id, monitor_trajectories_per_rank
from reproduction.common8.precision_contract import bf16_policy, easy2560_native4, native4_7b, remove_padding, verify as verify_precision_config

CONTRACT_KIND = 'fresh200_warmup57'
TARGET_UPDATES = 200
WARMUP_STEPS = 57
SAVE_STEPS = (5, 50, 100, 150, 200)
PROBE_ROUNDS = (0, 50, 100, 150)
SAFETY_MODE = 'inline_first_five'
SKIP_INITIAL_EVALUATION_POLICY = 'skip_initial_evaluation_start_training'
FORMAL_COUNTING = 'all20_opportunities_processed_and_at_least_one_write'
FORMAL_SAFEGUARDS = dict(dual_clip=3.0,epsilon=0.2,target_old_policy_kl=0.2,
                         kl_log_bound=30.0,negative_tail_limit=0.01)
FORMAL_MATHEMATICS = copy.deepcopy(MATHEMATICS)
FORMAL_MATHEMATICS.update(target_actual_outer_updates=TARGET_UPDATES,
    lr_warmup_steps=WARMUP_STEPS,
    learning_rate_schedule='linear_fixed_outer_warmup_then_constant',
    entropy_coefficient=0.0, reference_kl_coefficient=0.0,
    policy_safety_mode=SAFETY_MODE,safeguards=copy.deepcopy(FORMAL_SAFEGUARDS),
    optimizer_opportunities_per_outer=20,optimizer_completion_rule=FORMAL_COUNTING,
    optimizer_step_count_semantics='opportunities; actual successful writes reported separately')


def formal_mathematics(run=None):
    """The same formal objective with manifest-bound four/eight-rank placement."""
    size = world_size(run)
    values = copy.deepcopy(FORMAL_MATHEMATICS)
    values.update(world_size=size, ppo_micro_batch_size=actor_global_micro(run),
                  per_rank_actor_micro_batch=actor_global_micro(run)//size,
                  per_rank_log_prob_micro_batch=32 // size,
                  objective_reduction=('original four-row micro masked_mean'
                      if actor_global_micro(run)//size == 4 else 'original two-row micro masked_mean'))
    if run is not None and (run.get('easy2560_selection') is True or run.get('dynrpo7b_selection') is True):
        assert native4_7b(run)
        logical = actor_logical_micro(run)
        values.update(per_rank_actor_logical_micro_batch=logical,
                      objective_reduction=('original two-row micro masked_mean' if logical==2 else 'original four-row micro masked_mean'))
    if run is not None and bf16_policy(run):
        values.update(compute_dtype='bfloat16', rollout_dtype='bfloat16', allow_tf32=None)
    return values


def consumed_lr(step):
    assert type(step) is int and 1 <= step <= TARGET_UPDATES
    return 1e-6 * min((step - 1) / WARMUP_STEPS, 1.0)



def has_formal_markers(run):
    """Conflicting child-role labels cannot bypass recognizable formal fields."""
    budget=run.get('budget') if isinstance(run.get('budget'),dict) else {}
    return ('policy_safety_mode' in run or 'policy_safety_mode' in budget or
            run.get('contract_kind')==CONTRACT_KIND or budget.get('contract_kind')==CONTRACT_KIND or
            run.get('target_updates')==TARGET_UPDATES or
            budget.get('target_actual_outer_updates')==TARGET_UPDATES or
            run.get('lr_warmup_steps')==WARMUP_STEPS or budget.get('lr_warmup_steps')==WARMUP_STEPS)


def skip_initial_evaluation(run):
    """An explicit fresh native4 policy, without a historical baseline claim."""
    if run.get('pretraining_baseline_policy') != SKIP_INITIAL_EVALUATION_POLICY:
        return False
    assert (run.get('profile') == 'common8' and run.get('execution_site') == 'hpc3'
        and run.get('execution_backend', 'slurm') == 'slurm' and run.get('training_world_size') == 4
        and run.get('startup_admission_mode') == 'actual_first_five'
        and ((run.get('data_selection_mode') == 'prefix20_probe'
              and run.get('question_probe', {}).get('probe_architecture') == 'qwen2_prefix20_trajectory_em_v1')
             or native4_7b(run))
        and not run.get('full_state_restore')), 'Initial evaluation skip requires explicit fresh native4 Prefix20 or Easy'
    return True


def val_before_train(run):
    return run.get('pretraining_baseline_policy') != 'reuse_historical' and not skip_initial_evaluation(run)


def assert_formal_run(run):
    assert run['profile'] == 'common8' and run['contract_kind'] == CONTRACT_KIND
    assert run.get('policy_calibration_role', 'production') == 'production'
    assert run['policy_safety_mode']==SAFETY_MODE, 'Formal training requires the explicit prospective inline mode'
    assert type(run['completed_updates']) is int and run['completed_updates'] == 0
    assert type(run['target_updates']) is int and run['target_updates'] == TARGET_UPDATES
    assert type(run['acceptance_update']) is int and run['acceptance_update'] == 5
    assert run['initial_model'] == run['reference_model']
    from reproduction.common8.model_identity import identity
    model_id, model_revision = identity(run)
    skip_initial_evaluation(run)
    size = world_size(run)
    if run.get('execution_backend', 'slurm') == 'ssh_local':
        assert size == 4 and run['training_world_size'] == 4
        assert execution_job_id(run) is None
    else:
        assert run.get('execution_backend', 'slurm') == 'slurm'
        assert size == 8 or (size == 4 and run.get('execution_site') == 'hpc3')
    budget = run['budget']
    assert budget['contract_kind'] == CONTRACT_KIND and budget['profile']=='common8'
    assert budget['model_id']==model_id and budget['model_revision']==model_revision
    for name, value in formal_mathematics(run).items():
        assert canonical_bytes(budget[name]) == canonical_bytes(value), 'Formal budget differs: ' + name
    assert budget['common_manifest_sha256'] == '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
    return run


def verify_formal_config(document, run, base):
    """Validate actual resolved fields, paths and bound source, not a label alone."""
    assert_formal_run(run)
    base = Path(base).resolve()
    actor = document['actor_rollout_ref']['actor']
    reference = document['actor_rollout_ref']['ref']
    rollout = document['actor_rollout_ref']['rollout']
    def exact(part, values, label):
        for name, value in values.items():
            actual = part[name]
            equal = (type(actual) in (int, float) and actual == value) if type(value) is float else (
                canonical_bytes(actual) == canonical_bytes(value))
            assert equal, label + ' differs: ' + name
    exact(actor, dict(ppo_mini_batch_size=128, ppo_micro_batch_size=actor_global_micro(run), ppo_epochs=1,
        grad_clip=1.0, clip_ratio=0.2, state_masking=True, use_kl_loss=True,
        entropy_coeff=0.0, kl_loss_coef=0.0, kl_loss_type='low_var_kl',
        use_dynamic_bsz=False, ulysses_sequence_parallel_size=1), 'Actor mathematics')
    if run.get('easy2560_selection') is True:
        assert easy2560_native4(run)
        exact(actor, dict(ppo_logical_micro_batch_size=4), 'Logical actor objective')
        exact(document['algorithm'], dict(all_correct_virtual_zero=False), 'Easy real-group advantages')
        exact(document['data'], dict(selection_mode='mixed_first'), 'Easy selection')
    if run.get('dynrpo7b_selection'):
        exact(actor, dict(ppo_logical_micro_batch_size=2), 'DynRPO virtual micro2 objective')
        exact(document['algorithm'], dict(all_correct_virtual_zero=True), 'All-correct virtual zero')
        exact(document['data'], dict(selection_mode='dynrpo7b'), 'DynRPO selection')
    exact(actor['optim'], dict(lr=1e-6, lr_warmup_steps_ratio=0.0,
        lr_warmup_steps=WARMUP_STEPS, warmup_style='constant'), 'Optimizer mathematics')
    verify_precision_config(document, run)
    exact(rollout, dict(n=1, n_agent=5, temperature=1.0, top_p=1.0,
        log_prob_micro_batch_size=32, dtype=run['budget']['rollout_dtype'], tensor_model_parallel_size=1,
        name='vllm', gpu_memory_utilization=run.get('vllm_gpu_memory_utilization',0.4)), 'Rollout mathematics')
    exact(reference, dict(log_prob_micro_batch_size=32), 'Reference micro batch')
    if run.get('infra_rollout') is not None:
        from search_g1.rollout_phase_contract import validate
        phase=validate(run['infra_rollout'],world_size(run))
        assert rollout['async_phase_enabled'] is True
        exact(rollout,{k:v for k,v in phase.items() if k!='mode'},'Rollout phase execution limits')
    exact(document['algorithm'], dict(adv_estimator='grpo', no_think_rl=False), 'Advantages')
    exact(document['data'], dict(train_batch_size=512, val_batch_size=64,
        max_prompt_length=4096, max_response_length=500, max_start_length=2048,
        max_obs_length=500, train_data_num=None, val_data_num=None,
        shuffle_train_dataloader=True), 'Data budget')
    assert document['data']['train_files'] == str(base / 'data/train.parquet')
    assert document['data']['val_files'] == str(base / 'data/stage.parquet')
    assert document['actor_rollout_ref']['model']['path'] == run['initial_model']
    exact(document['actor_rollout_ref']['model'], dict(enable_gradient_checkpointing=True,
        use_remove_padding=remove_padding(run)), 'Model execution graph')
    exact(actor['fsdp_config'], dict(param_offload=True,grad_offload=True,
        optimizer_offload=True), 'Actor FSDP graph')
    assert reference['fsdp_config']['param_offload'] is True
    exact(document['trainer'], dict(n_gpus_per_node=world_size(run), nnodes=1,
        total_training_steps=TARGET_UPDATES + 1, total_epochs=15,
        save_freq=(1 if run.get('easy2560_rolling_checkpoint') is True or run.get('dynrpo7b_selection') is True else 50),
        test_freq=50, acceptance_step=5, val_only=False,
        val_before_train=val_before_train(run)), 'Training horizon')
    assert document['trainer']['default_local_dir'] == run['checkpoint_dir']
    if run.get('full_state_restore'):
        restore=run['full_state_restore']
        assert document['actor_rollout_ref']['model']['actor_checkpoint_path']==restore['source_checkpoint']
        assert document['trainer']['full_state_restore_receipt']==restore['source_receipt']
    assert Path(run['checkpoint_dir']).resolve() == base / 'checkpoints/search-g1-common8'
    assert document['max_turns'] == 4 and document['do_search'] is True
    exact(document['retriever'], dict(url=('http://127.0.0.1:18110/retrieve' if run.get('dynrpo7b_selection') else 'http://127.0.0.1:18010/retrieve'), topk=3), 'Retrieval')
    policy = actor['policy_safety']
    exact(policy, dict(enabled=True,mode=SAFETY_MODE,dual_clip=FORMAL_SAFEGUARDS['dual_clip'],
        target_old_policy_kl=FORMAL_SAFEGUARDS['target_old_policy_kl'],
        kl_log_bound=FORMAL_SAFEGUARDS['kl_log_bound'],negative_tail_limit=FORMAL_SAFEGUARDS['negative_tail_limit'],
        monitor_trajectories_per_rank=monitor_trajectories_per_rank(run)), 'Prospective formal safeguards')
    assert policy['calibration_manifest'] == str(base / 'logs/policy-safety-calibration.json')
    repo = Path(__file__).resolve().parents[2]
    for name in ('reproduction/common8/recipe.json', 'reproduction/common8/budget.json',
                 'verl/trainer/config/ppo_trainer.yaml'):
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == run['source_sha256'][name]
    return dict(contract_kind=CONTRACT_KIND,policy_safety_mode=SAFETY_MODE,
        mathematical_config_sha256=hashlib.sha256(canonical_bytes(formal_mathematics(run))).hexdigest(),
        resolved_config_sha256=hashlib.sha256(canonical_bytes(document)).hexdigest(),
        source_map_sha256=hashlib.sha256(canonical_bytes(run['source_sha256'])).hexdigest(),
        source_snapshot_semantics='bound disk source hashes; not proof of loaded modules')
