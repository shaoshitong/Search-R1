"""Segmented candidate counts; history and the periodic head-update math stay fixed."""
from .contract import (FINAL_OUTER, G, ORDINARY_CANDIDATES, PERIODIC_CANDIDATES,
                       PROBE_PERIOD, SELECTED_QUESTIONS)

FIELDS = frozenset(('effective_from_outer', 'legacy_candidates', 'ordinary_candidates',
    'update_candidates', 'probe_update_period', 'probe_update_at_final', 'selected_questions'))


def validate_schedule(config, *, required=False):
    if config.get('grpo_from_actor50'):
        from reproduction.actor50_grpo.schedule import validate_branch
        validate_branch(config)
        return dict(config['candidate_schedule'])
    value = config.get('candidate_schedule')
    if value is None:
        if required:
            raise ValueError('New execution requires its explicit candidate_schedule')
        return None
    expected = dict(legacy_candidates=PERIODIC_CANDIDATES,
        ordinary_candidates=ORDINARY_CANDIDATES, update_candidates=PERIODIC_CANDIDATES,
        probe_update_period=PROBE_PERIOD, probe_update_at_final=False,
        selected_questions=SELECTED_QUESTIONS)
    if (not isinstance(value, dict) or set(value) != FIELDS
            or type(value.get('effective_from_outer')) is not int
            or not 1 <= value['effective_from_outer'] <= FINAL_OUTER + 1
            or any(type(value.get(key)) is not type(item) or value[key] != item for key, item in expected.items())):
        raise ValueError('Candidate schedule must bind historical2560/ordinary5120/periodic2560/Top512/P10/no-update200')
    return dict(value)


def schedule_binding(config):
    """Return an independent canonical value; None means immutable legacy history."""
    return validate_schedule(config)


def candidate_count(config, outer):
    if type(outer) is not int or not 1 <= outer <= FINAL_OUTER:
        raise ValueError('Candidate count requires actual outer1..200')
    value = validate_schedule(config)
    if config.get('grpo_from_actor50') and outer > 50:
        return PERIODIC_CANDIDATES
    if value is None or outer < value['effective_from_outer']:
        return PERIODIC_CANDIDATES
    update = outer < FINAL_OUTER and outer % PROBE_PERIOD == 0
    return value['update_candidates'] if update else value['ordinary_candidates']


def consumed_candidates_through(config, completed_outer):
    """Sum the historical prefix and authorized suffix, without rewriting either."""
    if type(completed_outer) is not int or not 0 <= completed_outer <= FINAL_OUTER:
        raise ValueError('Candidate clock requires completed outer0..200')
    value = validate_schedule(config)
    if config.get('grpo_from_actor50') and completed_outer >= 50:
        return config['actor50_pool_consumed_candidates'] + (completed_outer-50)*PERIODIC_CANDIDATES
    if value is None:
        return completed_outer * PERIODIC_CANDIDATES
    before = min(completed_outer, value['effective_from_outer'] - 1)
    updates = (min(completed_outer, FINAL_OUTER - 1) // PROBE_PERIOD -
               min(before, FINAL_OUTER - 1) // PROBE_PERIOD)
    ordinary = completed_outer - before - updates
    return (before * value['legacy_candidates'] + updates * value['update_candidates'] +
            ordinary * value['ordinary_candidates'])


def schedule_metadata(config, outer):
    value = validate_schedule(config)
    if config.get('grpo_from_actor50') and outer > 50:
        from reproduction.actor50_grpo.schedule import outer_plan
        return dict(candidate_schedule=value, scheduled_candidate_questions=PERIODIC_CANDIDATES,
            grpo_from_actor50=True, actor50_branch_schedule=config['actor50_branch_schedule'],
            actor50_pool_consumed_candidates=config['actor50_pool_consumed_candidates'],
            actual_outer_plan=outer_plan(outer))
    if value is None or outer < value['effective_from_outer']:
        return {}
    return dict(candidate_schedule=value, scheduled_candidate_questions=candidate_count(config, outer))


def scoring_candidate_count(manifest):
    """Admit a scheduled native scan or a retained legacy/fixed diagnostic scan."""
    count = manifest.get('count')
    if type(count) is not int or count <= 0 or count % G:
        raise ValueError('Native scoring manifest must contain complete authentic G5 groups')
    questions = count // G
    if questions == SELECTED_QUESTIONS:
        return questions  # Existing fixed512 diagnostics predate an explicit flag.
    expected = candidate_count(manifest, manifest['outer'])
    if questions != expected:
        raise ValueError('Native scoring count differs from its actual scheduled candidate pool')
    return questions
