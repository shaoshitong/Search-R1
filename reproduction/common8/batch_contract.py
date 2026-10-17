"""Check the actual worker config before and after global-to-rank normalization."""
from reproduction.common8.topology import (
    actor_global_micro, actor_local_micro, actor_local_mini, actor_logical_micro, logprob_local_micro,
    world_size as training_world_size,
)


def independent_worker_config(config):
    # Colocated actor and reference constructors receive the same object.
    # Never let one role's in-place normalization change another role's input.
    import copy
    return copy.deepcopy(config)


def check_global(config, world_size, sequence_parallel, run=None):
    assert world_size == training_world_size(run) and sequence_parallel == 1, 'Wrong actual common8 data-parallel topology'
    assert config.rollout.n == 1 and config.rollout.n_agent == 5
    assert config.actor.ppo_mini_batch_size == 128
    assert config.actor.ppo_micro_batch_size == actor_global_micro(run)
    assert getattr(config.actor, 'ppo_logical_micro_batch_size', None) in (None, actor_logical_micro(run))
    if actor_local_micro(run) > 4:
        assert config.actor.ppo_logical_micro_batch_size == actor_logical_micro(run)
    assert config.actor.ppo_epochs == 1
    assert config.rollout.log_prob_micro_batch_size == 32
    assert config.ref.log_prob_micro_batch_size == 32
    assert config.actor.optim.lr == 1e-6
    assert config.actor.optim.lr_warmup_steps_ratio == 0
    assert config.actor.optim.lr_warmup_steps == 57
    assert config.actor.optim.warmup_style == 'constant'


def check_local(config, *, actor, rollout, reference):
    if actor:
        assert config.actor.ppo_mini_batch_size == actor_local_mini()
        assert config.actor.ppo_micro_batch_size == actor_local_micro()
        assert getattr(config.actor, 'ppo_logical_micro_batch_size', None) in (None, actor_logical_micro())
        if actor_local_micro() > 4:
            assert config.actor.ppo_logical_micro_batch_size == actor_logical_micro()
    if rollout:
        assert config.rollout.log_prob_micro_batch_size == logprob_local_micro()
    if reference:
        assert config.ref.log_prob_micro_batch_size == logprob_local_micro()
