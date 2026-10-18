"""Normalize already captured native20/40 rows without a legacy twenty-token cut."""


def normalized_native_row(original,qmax,prefix):
    from reproduction.prefix20_question_replay.features import initial_question_segment
    from reproduction.prefix20_online.model import expected_policy_positions
    if prefix not in (20,40):raise ValueError('Only the explicit native20/40 branch is admitted')
    question=initial_question_segment(original);qlen=question['question_length']
    n=original.get('policy_length',original.get('length',original.get('prefix_length')))
    if type(n) is not int or not 1<=n<=prefix or not 1<=qlen<=qmax:
        raise ValueError('Native captured row length or original question width differs')
    segments=[];offset=0
    for source in original['segments']:
        if source['policy_start']!=offset or source['policy_length']<=0:
            raise ValueError('Actual retained native segments must be contiguous and nonempty')
        if source['context_length']!=source['request_length']+source['policy_length']:
            raise ValueError('The captured segment must keep its exact original causal request')
        segments.append(dict(source));offset+=source['policy_length']
    bounds=[{k:s[k] for k in ('turn_index','policy_start','policy_length','position_start')} for s in segments]
    if offset!=n or len(expected_policy_positions(bounds))!=n:
        raise ValueError('Missing native retained positions; never trim Prefix40 back to20')
    return dict(original,question_length=qlen,question_context_offset=question['context_offset'],
        length=n,policy_length=n,prefix_length=n,segments=segments,segment_bounds=bounds)
