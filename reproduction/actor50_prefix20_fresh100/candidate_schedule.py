"""Every new warmup cycle uses exactly512 fresh questions, never actor pools."""
def candidate_count(config,outer):
    assert type(outer) is int and 1<=outer<=50
    assert config.get('candidate_schedule') is None
    return 512

def schedule_metadata(config,outer):
    assert candidate_count(config,outer)==512
    return dict(probe_warmup_cycle=outer,fresh_training_questions=512)
