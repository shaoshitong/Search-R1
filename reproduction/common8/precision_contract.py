"""Manifest-bound BF16 execution; legacy diagnostic manifests remain FP32."""

BF16_PROFILE = 'bf16_policy_fp32_master'
NATIVE_BF16_PROFILE = 'native8_bf16_varlen_fp32_master'
NATIVE4_BF16_PROFILE = 'native4_bf16_varlen_fp32_master'
SAMPLING_JOURNAL_POLICY = 'sampling_journal_no_extra_hf_forward'


def easy2560_native4(run):
    """The explicit Easy experiment may change physical, never logical, micros."""
    return (run.get('easy2560_selection') is True and run.get('profile') == 'common8'
        and run.get('execution_site') == 'hpc3' and run.get('execution_backend', 'slurm') == 'slurm'
        and type(run.get('training_world_size')) is int and run['training_world_size'] == 4
        and run.get('data_selection_mode') == 'mixed_first' and not run.get('question_probe')
        and type(run.get('actor_logical_micro_batch_per_rank')) is int
        and run['actor_logical_micro_batch_per_rank'] == 4)


def native4_7b(run):
    """Shared measured infrastructure; selectors remain independent recipes."""
    if easy2560_native4(run):
        return True
    from reproduction.dynrpo7b.infra_profile import native4
    return native4(run)


def bf16_policy(run):
    profile = run.get('precision_profile')
    assert profile in (None, BF16_PROFILE, NATIVE_BF16_PROFILE, NATIVE4_BF16_PROFILE), 'Unknown policy precision profile'
    if profile == BF16_PROFILE:
        assert run.get('execution_backend') == 'ssh_local' and run.get('training_world_size') == 4
        return True
    if profile == NATIVE_BF16_PROFILE:
        assert run.get('training_world_size') == 8 and run.get('execution_backend') == 'slurm'
        if run.get('execution_site')=='h1008':
            assert run.get('actor_micro_batch_per_rank')==4
        else:
            assert run.get('actor_micro_batch_per_rank')==2
        assert run.get('vllm_gpu_memory_utilization')==0.1
        return True
    if profile == NATIVE4_BF16_PROFILE:
        assert run.get('training_world_size') == 4 and run.get('execution_backend') == 'slurm'
        assert run.get('execution_site') == 'hpc3'
        if run.get('easy2560_selection') is True or run.get('dynrpo7b_selection') is True:
            assert native4_7b(run), 'Large physical micros require an explicit native4 recipe'
            micro = run.get('actor_micro_batch_per_rank')
            assert type(micro) is int and micro in (4, 8, 16, 32)
        else:
            assert run.get('actor_micro_batch_per_rank') == 4
        assert run.get('vllm_gpu_memory_utilization') == 0.1
        return True
    return False


def remove_padding(run):
    bf16_policy(run)
    return run.get('precision_profile') in (NATIVE_BF16_PROFILE, NATIVE4_BF16_PROFILE)


def overrides(run):
    if not bf16_policy(run):
        return {}
    result = {
        'actor_rollout_ref.actor.compute_dtype': 'bfloat16',
        'actor_rollout_ref.actor.allow_tf32': 'null',
        'actor_rollout_ref.ref.compute_dtype': 'bfloat16',
        'actor_rollout_ref.ref.allow_tf32': 'null',
        '+actor_rollout_ref.actor.fsdp_config.model_dtype': 'fp32',
        '+actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype': 'bf16',
        '+actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype': 'fp32',
        '+actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype': 'fp32',
        '+actor_rollout_ref.ref.fsdp_config.model_dtype': 'fp32',
        'actor_rollout_ref.rollout.dtype': 'bfloat16',
    }
    if remove_padding(run):
        result['actor_rollout_ref.model.use_remove_padding'] = 'true'
    return result


def sampling_journal_only(run):
    policy = run.get('behavior_probability_policy')
    assert policy in (None, SAMPLING_JOURNAL_POLICY), 'Unknown behavior probability policy'
    if policy == SAMPLING_JOURNAL_POLICY:
        assert bf16_policy(run), 'Sampling-journal-only mode requires the explicit BF16 profile'
        return True
    return False


def verify(document, run):
    parts = document['actor_rollout_ref']
    mixed = parts['actor']['fsdp_config']['mixed_precision']
    use_bf16 = bf16_policy(run)
    for role in ('actor', 'ref'):
        part = parts[role]
        assert part['compute_dtype'] == ('bfloat16' if use_bf16 else 'float32')
        assert part['allow_tf32'] is (None if use_bf16 else False)
        assert part['fsdp_config']['model_dtype'] == 'fp32'
    assert mixed['param_dtype'] == ('bf16' if use_bf16 else 'fp32')
    assert mixed['reduce_dtype'] == mixed['buffer_dtype'] == 'fp32'
    assert parts['rollout']['dtype'] == ('bfloat16' if use_bf16 else 'float32')
    return True
