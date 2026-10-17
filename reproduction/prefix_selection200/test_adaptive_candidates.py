"""Small CPU metadata/formula fixtures; no model, dataset, GPU or rollout."""
import copy
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from . import adaptive_candidates as adaptive
from .candidate_schedule import candidate_count,consumed_candidates_through,schedule_metadata,scoring_candidate_count
from .pool import initial_state,reserve_for_outer,state_sha
from .periodic import probe_boundary


def config(base,maximum=10004):
    return dict(base=str(base),run_id='cpu-fixture',adaptive_candidates=adaptive.policy(maximum),
        candidate_schedule=dict(effective_from_outer=1,legacy_candidates=2560,ordinary_candidates=5120,
            update_candidates=2560,probe_update_period=10,probe_update_at_final=False,selected_questions=512))


def complete(c,outer,mixed):
    root=Path(c['base'])/'outers'/('outer-%06d'%outer);root.mkdir(parents=True,exist_ok=True)
    value=adaptive.ensure(c,outer)
    joint=dict(run_id=c['run_id'],outer=outer,rollout=dict(run_id=c['run_id'],outer=outer,
        questions=512,trajectories=2560,actual_selected_mixed_questions=mixed,
        actual_selected_mixed_fraction=mixed/512))
    (root/'complete.json').write_text(json.dumps(joint,sort_keys=True))
    return value


class AdaptiveCandidateTests(unittest.TestCase):
    def test_formula_full_mixed_branch_rounding_and_both_bounds(self):
        p=adaptive.policy(168510)
        ordinary=adaptive.prediction(5120,384,p)
        self.assertEqual(ordinary['denominator'],.75+1e-6)
        self.assertEqual(ordinary['questions'],6828)
        full=adaptive.prediction(5120,512,p)
        self.assertEqual(full['denominator'],1.1+1e-6)
        self.assertEqual(full['questions'],4656)
        below_full=adaptive.prediction(5120,511,p)
        self.assertEqual(below_full['denominator'],511/512+1e-6)
        self.assertGreater(below_full['questions'],5120)
        self.assertEqual(adaptive.prediction(5120,0,p)['questions'],168508)
        self.assertEqual(adaptive.prediction(512,512,p)['questions'],512)
        # A displayed70% is rounded; feed the exact real integer count/512.
        real70=adaptive.prediction(5120,358,p)
        self.assertEqual(real70['questions'],math.ceil((5120/(358/512+1e-6))/4)*4)
        for previous,mixed in ((511,1),(5121,1),(5120,513),(5120,True),(True,1)):
            with self.assertRaises(ValueError):adaptive.prediction(previous,mixed,p)

    def test_committed_feedback_and_idempotent_saved_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=config(tmp)
            self.assertEqual(candidate_count(c,1),5120)
            with self.assertRaises(FileNotFoundError):candidate_count(c,2)
            first=complete(c,1,384)
            self.assertEqual(candidate_count(c,2),6828)
            second=complete(c,2,512);raw=adaptive.path(c,2).read_bytes()
            self.assertEqual(adaptive.ensure(c,2),second)
            self.assertEqual(adaptive.path(c,2).read_bytes(),raw)
            self.assertEqual(consumed_candidates_through(c,2),5120+6828)
            self.assertEqual(candidate_count(c,3),6208)
            joint=Path(tmp)/'outers/outer-000001/complete.json';v=json.loads(joint.read_text())
            v['rollout'].update(actual_selected_mixed_questions=512,actual_selected_mixed_fraction=1.)
            joint.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'never silently resize'):adaptive.ensure(c,2)

    def test_periodic_fixed_2560_then_next_ordinary_uses_previous_actual_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=config(tmp)
            for outer in range(1,201):
                value=complete(c,outer,512)
                periodic=outer<200 and outer%10==0
                if periodic:
                    self.assertEqual(value['questions'],2560);self.assertEqual(value['mode'],'fixed_probe_update')
                elif outer>1 and (outer-1)%10==0:
                    self.assertEqual(value['questions'],2328)
            self.assertEqual(adaptive.budget(c,200)['mode'],'adaptive')
            self.assertEqual(probe_boundary(200),dict(last_update_outer=190,actual_adam_per_rank=950))
            self.assertEqual(consumed_candidates_through(c,200),sum(adaptive.budget(c,n)['questions']for n in range(1,201)))

    def test_pool_cross_cycle_and_resume_keep_exact_reserved_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=config(root/'a');state=initial_state(10007,1729)
            complete(c,1,384);_,_,state=reserve_for_outer(state,c,1)
            complete(c,2,512);indices,cycles,state=reserve_for_outer(state,c,2)
            self.assertEqual(len(indices),6828);self.assertEqual(len(set(indices)),6828)
            self.assertEqual(set(cycles),{0,1});before=state_sha(state)
            shutil.copytree(root/'a',root/'resumed');resumed=dict(c,base=str(root/'resumed'))
            self.assertEqual(candidate_count(c,3),candidate_count(resumed,3))
            self.assertEqual(reserve_for_outer(state,c,3),reserve_for_outer(copy.deepcopy(state),resumed,3))
            self.assertEqual(state_sha(state),before)

    def test_score_manifest_carries_budget_without_outcome_payload_or_filesystem(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=config(tmp);complete(c,1,384);complete(c,2,512)
            metadata=schedule_metadata(c,2)
            manifest=dict(run_id=c['run_id'],outer=2,count=6828*5,**metadata)
            shutil.rmtree(tmp)
            self.assertEqual(scoring_candidate_count(manifest),6828)
            for key,value in [('questions',6824),('denominator',.7),('consumed_after',1),('run_id','other')]:
                altered=copy.deepcopy(manifest);altered['candidate_budget'][key]=value
                with self.assertRaises(ValueError):scoring_candidate_count(altered)

    def test_invalid_or_nonfinite_feedback_never_becomes_a_prediction(self):
        for change in ('run','outer','size','fraction','nan'):
            with tempfile.TemporaryDirectory() as tmp:
                c=config(tmp);complete(c,1,384)
                p=Path(tmp)/'outers/outer-000001/complete.json';joint=json.loads(p.read_text())
                if change=='run':joint['run_id']='other'
                elif change=='outer':joint['outer']=0
                elif change=='size':joint['rollout']['questions']=2560
                elif change=='fraction':joint['rollout']['actual_selected_mixed_fraction']=.7
                else:joint['rollout']['actual_selected_mixed_fraction']=float('nan')
                p.write_text(json.dumps(joint))
                with self.assertRaises(ValueError):candidate_count(c,2)

    def test_cached_prepare_installs_policy_and_refuses_started_runtime(self):
        from reproduction.hpc3_prefix5 import runtime
        for started in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);prep=base/'preparation';prep.mkdir()
                for name,value in [('assets-ready.json',dict(commit='old')),('source-publication.json',dict(commit='new')),
                                   ('launch-config-template.json',dict(source_commit='old'))]:
                    (prep/name).write_text(json.dumps(value))
                if started:(prep/'submit-intent.json').write_text('{}')
                with patch.object(runtime,'BASE',base),patch.object(runtime,'adaptive_policy',return_value=adaptive.policy(10004)):
                    if started:
                        with self.assertRaisesRegex(ValueError,'claimed'):runtime.prepare()
                        self.assertNotIn('adaptive_candidates',json.loads((prep/'launch-config-template.json').read_text()))
                    else:
                        result=runtime.prepare();saved=json.loads((prep/'launch-config-template.json').read_text())
                        self.assertTrue(result['adaptive_policy_changed']);self.assertEqual(saved['source_commit'],'new')
                        self.assertEqual(saved['adaptive_candidates'],adaptive.policy(10004))


if __name__=='__main__':unittest.main()
