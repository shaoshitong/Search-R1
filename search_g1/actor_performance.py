"""Explicit actor hot-path policy; scientific math and checkpointing are separate.

The default preserves every historical numerical guard and rollback snapshot.
``lean_v1`` removes per-micro/per-Adam numerical guards and rollback copies, while
leaving forward/loss/clipping/Adam and the full outer checkpoint unchanged.
Its production binding is checked once with the existing actor configuration.
"""

LEAN_PROFILE = 'lean_v1'


def lean_actor(config):
    """Recognize only an explicit opt-in, never an environment-wide bypass."""
    get = getattr(config, 'get', None)
    profile = get('performance_profile', None) if callable(get) else getattr(config, 'performance_profile', None)
    if profile not in (None, '', 'legacy', LEAN_PROFILE):
        raise ValueError('Unknown actor performance_profile: ' + str(profile))
    return profile == LEAN_PROFILE


def lean_run(run):
    """Authenticate the manifest-only scope for scheduler and audit readers."""
    if not lean_actor(run):
        return False
    from reproduction.common8.precision_contract import easy2560_native4
    if not (run.get('performance_profile') == LEAN_PROFILE
            and easy2560_native4(run)
            and run.get('policy_calibration_role', 'production') == 'production'
            and run.get('extra_policy_monitor_forwards') is False):
        raise ValueError('lean_v1 requires explicit native4 Easy production and no extra monitor forwards')
    return True


def validate_lean_binding(config, run):
    """Fail closed if lean execution is not bound to native four-rank Easy."""
    enabled = lean_actor(config)
    bound = lean_run(run)
    if enabled != bound:
        raise ValueError('Run and actor performance_profile differ')
    if not enabled:
        return False
    safety = config.get('policy_safety')
    if safety is None or safety.get('extra_policy_monitor_forwards') is not False:
        raise ValueError('lean_v1 actor must explicitly disable extra monitor forwards')
    return True


def rollback_metadata(*, lean):
    """Describe the actual per-write capability, not checkpoint recovery."""
    return dict(no_automatic_rollback=lean, current_step_rollback=not lean)
