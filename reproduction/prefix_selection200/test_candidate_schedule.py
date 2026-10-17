"""CPU-only candidate clocks, retained pool continuity, and authentic G5 score shapes."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import unittest

from .candidate_schedule import (candidate_count, consumed_candidates_through,
    schedule_binding, scoring_candidate_count, validate_schedule)
from .contract import G, WORLD, SELECTED_QUESTIONS, CANDIDATE_QUESTIONS, PERIODIC_CANDIDATES, ORDINARY_CANDIDATES
from .periodic import is_probe_update, probe_boundary
from .pool import initial_state, reserve_indices, reserve_for_outer, state_sha
from .rollout import question_count

ROOT = Path(__file__).resolve().parents[2]


def configured(boundary=79):
    return dict(candidate_schedule=dict(effective_from_outer=boundary+1,
        legacy_candidates=2560, ordinary_candidates=5120, update_candidates=2560,
        probe_update_period=10, probe_update_at_final=False, selected_questions=512))


class CandidateScheduleTests(unittest.TestCase):
    def test_historical_prefix_and_new_suffix_have_exact_segmented_counts(self):
        config = configured(79)
        self.assertEqual((CANDIDATE_QUESTIONS, PERIODIC_CANDIDATES, ORDINARY_CANDIDATES), (2560,2560,5120))
        for outer in range(1,201):
            expected = 2560 if outer <= 79 or is_probe_update(outer) else 5120
            self.assertEqual(candidate_count(config,outer), expected)
            self.assertEqual(consumed_candidates_through(config,outer),
                             sum(candidate_count(config,index) for index in range(1,outer+1)))
        self.assertEqual(consumed_candidates_through(config,79),79*2560)
        self.assertEqual(candidate_count(config,80),2560)
        self.assertEqual(candidate_count(config,81),5120)
        self.assertEqual(candidate_count(config,190),2560)
        self.assertEqual(candidate_count(config,200),5120)
        self.assertFalse(is_probe_update(200))
        self.assertEqual(probe_boundary(200),dict(last_update_outer=190,actual_adam_per_rank=950))
        self.assertEqual(consumed_candidates_through(config,200),791040)
        self.assertEqual(consumed_candidates_through(configured(0),200),975360)
        self.assertEqual(consumed_candidates_through({},200),512000)
        self.assertEqual(consumed_candidates_through(configured(200),200),512000)

    def test_schedule_schema_is_explicit_and_legacy_history_stays_readable(self):
        self.assertIsNone(validate_schedule({}))
        with self.assertRaisesRegex(ValueError,'explicit'):
            validate_schedule({},required=True)
        for key,value in (('effective_from_outer',True),('ordinary_candidates',2560),
                          ('update_candidates',5120),('probe_update_at_final',True),('selected_questions',1024)):
            config=configured();config['candidate_schedule'][key]=value
            with self.assertRaises(ValueError):validate_schedule(config)
        config=configured();copy_binding=schedule_binding(config);copy_binding['ordinary_candidates']=1
        self.assertEqual(config['candidate_schedule']['ordinary_candidates'],5120)

    def test_pool_resume_keeps_original_state_order_cursor_cycle_and_rng(self):
        state=initial_state(17011,1729)
        for outer in range(1,4):
            _,_,state=reserve_indices(state,outer)
        state['controller_rng']='retained opaque controller snapshot'
        source=copy.deepcopy(state);source_sha=state_sha(source);global_rng=random.getstate()
        config=configured(3)
        indices,cycles,after=reserve_for_outer(state,config,4)
        self.assertEqual(state_sha(state),source_sha)
        self.assertEqual(state,source)
        self.assertEqual(after['order'],source['order'])
        self.assertEqual(after['cursor'],source['cursor']+5120)
        self.assertEqual(after['cycle'],source['cycle'])
        self.assertEqual(after['seed'],source['seed'])
        self.assertEqual(after['controller_rng'],source['controller_rng'])
        self.assertEqual(after['candidate_questions'],2560)  # Historical annotation is preserved.
        self.assertEqual(len(set(indices)),5120);self.assertEqual(cycles,[0]*5120)
        for outer in range(5,13):
            before=copy.deepcopy(after)
            selected,_,after=reserve_for_outer(after,config,outer)
            self.assertEqual(len(set(selected)),candidate_count(config,outer))
            self.assertEqual(after['consumed_candidates'],consumed_candidates_through(config,outer))
            self.assertEqual(after['cycle']*len(after['order'])+after['cursor'],after['consumed_candidates'])
            self.assertEqual(reserve_for_outer(json.loads(json.dumps(before)),config,outer)[2],after)
        self.assertEqual(random.getstate(),global_rng)
        wrong=copy.deepcopy(after);wrong['consumed_candidates']+=2560
        with self.assertRaisesRegex(ValueError,'segmented'):
            reserve_for_outer(wrong,config,13)

    def test_native_prefix_and_full_continuation_scopes_follow_actual_outer(self):
        config=configured(79)
        self.assertEqual(question_count(config,'prefix',79),2560)
        self.assertEqual(question_count(config,'prefix',80),2560)
        self.assertEqual(question_count(config,'prefix',81),5120)
        self.assertEqual(question_count(config,'prefix',200),5120)
        self.assertEqual(question_count(config,'continue',81),512)
        self.assertEqual(question_count(dict(config,rollout_continuation_questions=2560),'continue',80),2560)
        self.assertEqual(question_count(dict(config,diagnostic_only=True,rollout_candidate_questions=512),'prefix',81),512)
        with self.assertRaises(ValueError):
            question_count(dict(config,rollout_continuation_questions=2560),'continue',81)
        with self.assertRaises(ValueError):
            question_count(dict(config,rollout_candidate_questions=2560),'prefix',81)
        self.assertEqual(question_count({},'prefix'),2560)

    def test_periodic_head_epoch_retains_2560_g5_and_fifty_global256_average_updates(self):
        from .contract import PROBE_LR, PROBE_UPDATES
        from .probe_plan import complete_pool_plan, rank_question_slice
        plan=complete_pool_plan(1729,80)
        self.assertEqual(PROBE_LR,5e-6);self.assertEqual(PROBE_UPDATES,50)
        self.assertEqual(len(plan),50)
        self.assertEqual(sorted(index for batch in plan for index in batch['question_indices']),list(range(2560)))
        self.assertEqual(sum(batch['global_questions']*G for batch in plan),50*256)
        self.assertEqual({batch['global_questions']*G for batch in plan},{255,260})
        self.assertEqual([sum(len(rank_question_slice(batch,rank))for batch in plan)for rank in range(WORLD)],[640]*4)
        with self.assertRaises(ValueError):complete_pool_plan(1729,200)

    def test_5120_score_shape_retains_original_tie_break_top512_and_periodic_shape(self):
        import torch
        tree=ast.parse((ROOT/'reproduction/prefix_selection200/probe.py').read_text())
        names={'candidate_groups','mixed_probabilities','stable_top512'}
        nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name in names]
        scope=dict(torch=torch,G=G,SELECTED_QUESTIONS=SELECTED_QUESTIONS,
                   PERIODIC_CANDIDATES=PERIODIC_CANDIDATES,ORDINARY_CANDIDATES=ORDINARY_CANDIDATES)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-probe-pure-score-functions','exec'),scope)
        index={f'q{question}-g{response}':dict(qid=f'q{question}',question_hash=f'h{question}',
            candidate_ordinal=question,response_index=response)
            for question in reversed(range(5120))for response in range(5)}
        groups=scope['candidate_groups'](index)
        self.assertEqual(len(groups),5120)
        logits={tid:0. for tid in index}
        selected,all_scores=scope['stable_top512'](groups,logits)
        self.assertEqual(len(all_scores),5120)
        self.assertEqual([row['candidate_ordinal']for row in selected],list(range(512)))
        self.assertEqual(selected[0]['trajectory_ids'],[f'q0-g{response}'for response in range(5)])
        self.assertAlmostEqual(selected[0]['q'],30/32)
        tiny={key:value for key,value in index.items() if value['candidate_ordinal'] < 3}
        self.assertEqual(len(scope['candidate_groups'](tiny,3)),3)
        config=configured(79)
        self.assertEqual(scoring_candidate_count(dict(config,count=25600,outer=81)),5120)
        self.assertEqual(scoring_candidate_count(dict(config,count=12800,outer=80)),2560)
        self.assertEqual(scoring_candidate_count(dict(config,count=25600,outer=200)),5120)
        self.assertEqual(scoring_candidate_count(dict(count=12800,outer=79)),2560)
        self.assertEqual(scoring_candidate_count(dict(config,count=2560,outer=81)),512)
        with self.assertRaises(ValueError):scoring_candidate_count(dict(config,count=25600,outer=80))
        with self.assertRaises(ValueError):scoring_candidate_count(dict(count=25600,outer=81))
        index.pop('q0-g0')
        with self.assertRaises(ValueError):scope['candidate_groups'](index,5120)

    def test_probe_training_loading_and_optimizer_critical_ast_are_identical(self):
        proof=json.loads((ROOT/'results/candidate-schedule/probe-critical-ast-unchanged-20261017.json').read_text())
        self.assertEqual(proof['before'],proof['after']);self.assertEqual(len(proof['after']),14)
        tree=ast.parse((ROOT/proof['source']).read_text())
        actual={node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
                for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))and node.name in proof['after']}
        self.assertEqual(actual,proof['after'])


if __name__=='__main__':unittest.main()
