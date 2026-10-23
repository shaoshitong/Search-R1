"""CPU-only aggregation of actual DynRPO prefix/continuation rank receipts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from search_g1.rollout_phase_contract import scalar_metrics


def output(continuations=1):
    events=[]
    for i in range(continuations+1):
        phase_id='phase-%d'%i
        events.append(dict(phase='candidate_prefix' if i==0 else 'selected_continuation',
            phase_id=phase_id,phase_metrics=[dict(phase_id=phase_id,
                policy_snapshot_id='actor-0',aborted=False,weight_syncs=1,
                engine_steps=10*i+rank+1,completed_requests=100*i+rank+1)
                for rank in range(4)]))
    return SimpleNamespace(meta_info=dict(async_rollout=dict(
        role='dynrpo7b_pause_select_continue',completed_trajectories=2560,
        generation_requests=9123,environment_actions=5678,peak_inflight_trajectories=256,
        retrieval_concurrency=16,phase_elapsed_seconds=87.5,phases=events)))


class DynRPOPhaseMetricsTests(unittest.TestCase):
    def test_actual_751127_closed_phase_metadata(self):
        reports=Path(__file__).resolve().parents[2]/'reproduction/reports/dynrpo7b-formal-20261023'
        meta=json.loads((reports/'actual-751127-rollout-async-meta.json').read_text())
        expected=json.loads((reports/'actual-751127-phase-metrics-cpu.json').read_text())
        self.assertTrue(all('policy_snapshot_id' not in event for event in meta['phases']))
        self.assertEqual(scalar_metrics(SimpleNamespace(meta_info={'async_rollout':meta})),expected['metrics'])

    def test_two_real_four_rank_phases(self):
        value=output()
        result=scalar_metrics(value)
        self.assertEqual(result['infra/rollout/weight_syncs_per_rank'],2)
        self.assertEqual(result['infra/rollout/engine_steps_sum'],60)
        self.assertEqual(result['infra/rollout/completed_requests_sum'],420)
        self.assertEqual(result['infra/rollout/completed_trajectories'],2560)
        self.assertEqual(result['infra/rollout/generation_requests'],9123)
        self.assertEqual(result['infra/rollout/phase_elapsed_seconds'],87.5)
        self.assertNotIn('phase_metrics',value.meta_info['async_rollout'])

    def test_backstop_is_additional_real_work(self):
        result=scalar_metrics(output(continuations=2))
        self.assertEqual(result['infra/rollout/weight_syncs_per_rank'],3)
        self.assertEqual(result['infra/rollout/engine_steps_sum'],150)
        self.assertEqual(result['infra/rollout/completed_requests_sum'],1230)

    def test_incomplete_or_divergent_receipts_rejected(self):
        for mutation in ('missing_rank','wrong_snapshot','aborted'):
            value=copy.deepcopy(output())
            receipts=value.meta_info['async_rollout']['phases'][1]['phase_metrics']
            if mutation=='missing_rank':receipts.pop()
            elif mutation=='wrong_snapshot':receipts[0]['policy_snapshot_id']='actor-1'
            else:receipts[0]['aborted']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                scalar_metrics(value)


if __name__=='__main__':
    unittest.main()
