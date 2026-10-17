"""Ten-outer probe clocks; outer200 deliberately has no probe update."""
from .contract import FINAL_OUTER, PROBE_PERIOD, PROBE_UPDATES
from .candidate_schedule import candidate_count, consumed_candidates_through

PERIOD = PROBE_PERIOD
UPDATES = PROBE_UPDATES


def is_probe_update(outer):
    if type(outer) is not int or not 1 <= outer <= FINAL_OUTER:
        raise ValueError('Actor outer must be1..200')
    return outer < FINAL_OUTER and outer % PERIOD == 0


def probe_boundary(completed_outer):
    if type(completed_outer) is not int or not 0 <= completed_outer <= FINAL_OUTER:
        raise ValueError('Completed actor outer must be0..200')
    last = min(completed_outer // PERIOD * PERIOD, FINAL_OUTER - PERIOD)
    return dict(last_update_outer=last, actual_adam_per_rank=last // PERIOD * UPDATES)


def complete_pool_plan(seed, outer):
    from .probe_plan import complete_pool_plan as plan
    return plan(seed, outer)
