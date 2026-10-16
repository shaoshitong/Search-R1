"""CPU tests of native residual/identity alignment and the new update clock."""
import importlib.util
from types import SimpleNamespace
import unittest

from .actor_hidden import ActorHiddenCapture, query_mapping, QMAX, WIDTH, DIM
from .periodic import complete_pool_plan, is_probe_update, probe_boundary


class PeriodicProtocolTests(unittest.TestCase):
    def test_only_twenty_boundaries_write_and_full_budget(self):
        self.assertEqual([n for n in range(1,201) if is_probe_update(n)], list(range(20,201,20)))
        self.assertEqual(probe_boundary(19), dict(last_update_outer=0,actual_adam_per_rank=0))
        self.assertEqual(probe_boundary(20), dict(last_update_outer=20,actual_adam_per_rank=100))
        self.assertEqual(probe_boundary(39), probe_boundary(20))
        self.assertEqual(probe_boundary(200), dict(last_update_outer=200,actual_adam_per_rank=1000))

    def test_unselected_whole_pool_natural_once_and_unequal_ddp(self):
        p=complete_pool_plan(1729,20)
        self.assertEqual(len(p),100)
        self.assertEqual(sorted(i for b in p for i in b['question_indices']),list(range(5120)))
        self.assertEqual(sum(b['global_questions']*5 for b in p),25600)
        for rank in range(8):
            self.assertEqual(sum(b['rank_questions'][rank] for b in p),640)
        with self.assertRaises(ValueError):complete_pool_plan(1729,19)

    def test_native_query_order_uses_actual_lengths_not_fixed_stride(self):
        m=[SimpleNamespace(request_id='prefill',seq_data={5:None}),
           SimpleNamespace(request_id='decode',seq_data={9:None})]
        q=query_mapping(m,[3,1],[4,5,6,22],[0,1,2,18])
        self.assertEqual([(r['request_id'],r['start'],r['end']) for r in q],
                         [('prefill',0,3),('decode',3,4)])
        self.assertEqual(q[1]['positions'],[18])
        with self.assertRaises(ValueError):query_mapping(m,[3,1],[4,5,6],[0,1,2])


class NativeRetentionClockTests(unittest.TestCase):
    def test_native_periodic_config_does_not_require_legacy_ten_writes_every_outer(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from .storage import validate_base
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory).resolve();b=parent/'search-g1-prefix-selection200-native-test'
            (b/'preparation').mkdir(parents=True)
            c=dict(base=str(b),run_id='test',outers=200,world_size=8,actor_hidden_native=True,
                probe_updates_per_outer=0,probe_update_period=20,probe_updates_per_period=100)
            (b/'preparation/launch-config.json').write_text(json.dumps(c))
            with patch('reproduction.prefix_selection200.storage.REMOTE_PARENT',parent):
                self.assertEqual(validate_base(b)[1],c)



@unittest.skipUnless(importlib.util.find_spec('torch'), 'Run tensor tests in existing remote CPU environment')
class NativeHiddenTensorTests(unittest.TestCase):
    def make_capture(self):
        import torch
        layers=torch.nn.ModuleList([torch.nn.Identity() for _ in range(36)])
        core=SimpleNamespace(layers=layers,config=SimpleNamespace(hidden_size=DIM))
        runner=SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True),
            cache_config=SimpleNamespace(enable_prefix_caching=False),model=SimpleNamespace(model=core),
            prepare_model_input=lambda *a,**k:None,execute_model=lambda *a,**k:None)
        engine=SimpleNamespace(model_executor=SimpleNamespace(driver_worker=SimpleNamespace(model_runner=runner)))
        c=ActorHiddenCapture(engine,'actor0');self.addCleanup(c.close)
        state=dict(trajectory_id='tid',actor_version='actor0',prompt=[7,8],ordinal=0,info_mask=[],
                   action=dict(actual_request_token_ids=[7,8]))
        request=dict(request_id='rid',actual_request_token_ids=[7,8])
        c.register(state,request)
        return c

    def feed(self,c,tokens,positions,hidden_value=3,residual_value=5):
        import torch
        c.query_rows=[dict(request_id='rid',start=0,end=len(tokens),tokens=tokens,positions=positions)]
        h=torch.full((len(tokens),DIM),hidden_value,dtype=torch.bfloat16)
        r=torch.full_like(h,residual_value)
        c._layer_output(None,None,(h,r))
        # Simulate next native layer mutating the original residual in place.
        r.fill_(99)
        c._consume_native()

    def test_post_block_is_residual_sum_and_padding_never_contains_future(self):
        import torch
        c=self.make_capture();self.feed(c,[7,8,10,11,999],[0,1,2,3,4])
        prefix=dict(trajectory_id='tid',actor_version='actor0',retained_policy_length=2,
                    initial_prompt_token_ids=[7,8],policy_token_ids=[10,11],policy_position_ids=[2,3])
        x=c.finish(prefix);self.assertEqual(tuple(x['features'].shape),(WIDTH,DIM))
        self.assertTrue(torch.equal(x['features'][0],torch.full((DIM,),8,dtype=torch.bfloat16)))
        self.assertTrue(bool((x['features'][QMAX+2:]==0).all()))
        self.assertEqual(x['attention_mask'].nonzero().flatten().tolist(),[0,1,QMAX,QMAX+1])

    def test_token20_not_processed_cannot_pass_as_twenty_token_representation(self):
        c=self.make_capture();self.feed(c,[7,8]+list(range(19)),list(range(21)))
        prefix=dict(trajectory_id='tid',actor_version='actor0',retained_policy_length=20,
                    initial_prompt_token_ids=[7,8],policy_token_ids=list(range(20)),
                    policy_position_ids=list(range(2,22)))
        with self.assertRaisesRegex(ValueError,'Missing/off-by-one'):c.finish(prefix)

    def test_twentieth_actual_decode_input_completes_the_correct_representation(self):
        c=self.make_capture();self.feed(c,[7,8]+list(range(19)),list(range(21)))
        self.feed(c,[19],[21])
        prefix=dict(trajectory_id='tid',actor_version='actor0',retained_policy_length=20,
                    initial_prompt_token_ids=[7,8],policy_token_ids=list(range(20)),
                    policy_position_ids=list(range(2,22)))
        x=c.finish(prefix)
        self.assertEqual(int(x['attention_mask'].sum()),22)
        self.assertEqual(x['position_ids'][QMAX:QMAX+20].tolist(),list(range(2,22)))

    def test_original_token_identity_mismatch_is_detected(self):
        c=self.make_capture()
        with self.assertRaisesRegex(ValueError,'registered real token'):self.feed(c,[77,8],[0,1])


if __name__=='__main__':unittest.main()
