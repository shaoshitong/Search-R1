"""CPU entry and full40-token metadata check before any allocation is stopped."""
import hashlib
from pathlib import Path
import os
import json


def main():
    import torch
    from reproduction.prefix_selection200 import contract
    from reproduction.prefix_selection200.probe import prefix_row
    from reproduction.prefix_selection200.rollout import question_count
    from reproduction.prefix_selection200.prefix5_model import PARAMETER_SHAPES
    from reproduction.prefix_selection200.actor_hidden import PREFIX,WIDTH
    from .actor import ACTOR_SOURCE_SHA256
    from . import worker,rerank,lift40,bootstrap,run,mixed_floor,restart
    from .schedule import BRANCH_SCHEDULE,outer_plan
    from reproduction.prefix_selection200 import actor
    assert hashlib.sha256(Path(actor.__file__).read_bytes()).hexdigest()==ACTOR_SOURCE_SHA256
    n=int(os.environ['SEARCH_G1_GRPO_PREFIX']);assert n in (20,40)
    assert PREFIX==n and WIDTH==273+n and contract.PREFIX_TOKENS==n
    assert PARAMETER_SHAPES['output.weight']==(1,(273+n)*2048)
    segments=[dict(turn_index=0,policy_start=0,policy_length=17,position_start=4,
        request_length=4,context_length=21,context_offset=0),dict(turn_index=1,
        policy_start=17,policy_length=n-17,position_start=10,request_length=10,
        context_length=10+n-17,context_offset=21)]
    row=prefix_row(dict(length=n,prefix_length=n,policy_length=n,segments=segments))
    assert row['policy_length']==n and sum(s['policy_length'] for s in row['segments'])==n
    assert row['segment_bounds'][-1]['policy_length']==n-17
    c=dict(grpo_from_actor50=True,actor50_branch_schedule=BRANCH_SCHEDULE,
        actor50_pool_consumed_candidates=230400,candidate_schedule=dict(effective_from_outer=6,
        legacy_candidates=2560,ordinary_candidates=5120,update_candidates=2560,
        probe_update_period=10,probe_update_at_final=False,selected_questions=512))
    for outer in (51,60,99,100,149,150,151,190,200):
        p=outer_plan(outer);x=dict(c,rollout_continuation_questions=p['continuation_questions'])
        assert question_count(x,'prefix',outer)==2560
        assert question_count(x,'continue',outer)==p['continuation_questions']
        diagnostic=dict(x,diagnostic_only=True,rollout_candidate_questions=512,rollout_continuation_questions=512)
        assert question_count(diagnostic,'prefix',outer)==question_count(diagnostic,'continue',outer)==512
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='CPU_NATIVE_ENTRY_AND_FULL_PREFIX_PASS',prefix=n,width=273+n,
        all_native_policy_positions_preserved=True,no_CUDA=True)))


if __name__=='__main__':main()
