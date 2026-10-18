"""Ten-outer probe clocks; outer200 deliberately has no probe update."""
from .contract import FINAL_OUTER, PROBE_PERIOD, PROBE_UPDATES
from .candidate_schedule import candidate_count, consumed_candidates_through

PERIOD = PROBE_PERIOD
UPDATES = PROBE_UPDATES


def is_probe_update(outer):
    if type(outer) is not int or not 1 <= outer <= FINAL_OUTER:
        raise ValueError('Actor outer must be1..200')
    from .contract import ACTOR50_BRANCH
    return outer < FINAL_OUTER and outer % PERIOD == 0 and (not ACTOR50_BRANCH or outer > 50)


def probe_boundary(completed_outer):
    from .contract import ACTOR50_BRANCH
    if ACTOR50_BRANCH:
        from reproduction.actor50_grpo.schedule import probe_boundary as actor50_boundary
        return actor50_boundary(completed_outer)
    if type(completed_outer) is not int or not 0 <= completed_outer <= FINAL_OUTER:
        raise ValueError('Completed actor outer must be0..200')
    last = min(completed_outer // PERIOD * PERIOD, FINAL_OUTER - PERIOD)
    return dict(last_update_outer=last, actual_adam_per_rank=last // PERIOD * UPDATES)


def complete_pool_plan(seed, outer):
    from .probe_plan import complete_pool_plan as plan
    return plan(seed, outer)
