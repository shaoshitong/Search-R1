"""Model-specific identity and budget fields; learning mathematics is shared."""
THREE = ('Qwen/Qwen2.5-3B-Instruct', 'aa8e72537993ba99e69dfaafa59ed015b17504d1')
SEVEN = ('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28')


def identity(run):
    pair = run['budget']['model_id'], run['model_revision']
    assert pair in (THREE, SEVEN), 'Only pinned official model revisions are supported'
    if pair == SEVEN:
        from reproduction.common8.precision_contract import native4_7b
        assert native4_7b(run), '7B adapter requires an explicit native4 7B recipe'
    assert run['budget']['model_revision'] == pair[1]
    return pair


def budget_for_run(budget, run):
    pair = identity(run)
    return dict(budget, model_id=pair[0], model_revision=pair[1])
