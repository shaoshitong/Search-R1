"""Focused runner/evaluation gates with real synthetic optimizer transactions."""
import copy,gzip,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from . import direct_hidden_count_train as r,probe_ablation_train as u,direct_hidden_checkpoint as cp
from .test_direct_hidden_count import setup,update,Reader,forward


class EntryTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_actual_count_scoring_uses_six_classes_not_product_of_old_p(self):
        values=setup();m,e,o,h,he,ho,own,*_=values
        groups=[dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'][:5],labels=g['labels'][:5]) for g in own]
        def pure(model,data,head):
            x=torch.tanh(model.block(data['features'].detach()))
            return model.output(x.flatten(1)).squeeze(-1),head.encode(x[:,2:],data['attention_mask'][:,2:])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.jsonl.gz';r.score_fixed(m,h,Reader(),groups,torch.device('cpu'),p,forward_adapter=pure)
            with gzip.open(p,'rt') as f:before=[json.loads(line) for line in f]
            with torch.no_grad():m.output.bias.add_(20)
            q=Path(d)/'b.jsonl.gz';r.score_fixed(m,h,Reader(),groups,torch.device('cpu'),q,forward_adapter=pure)
            with gzip.open(q,'rt') as f:after=[json.loads(line) for line in f]
            self.assertEqual([v['predicted_mixed'] for v in before],[v['predicted_mixed'] for v in after])
            self.assertNotEqual(before[0]['p_j'],after[0]['p_j'])
            for row in before:
                self.assertAlmostEqual(row['predicted_mixed'],sum(row['count_probabilities'][1:5]),places=12)
                self.assertEqual(row['score_kind'],'hidden_direct_count6')
            changed=copy.deepcopy(groups)
            for g in changed:g['labels']=[1-y for y in g['labels']]
            q=Path(d)/'c.jsonl.gz';r.score_fixed(m,h,Reader(),changed,torch.device('cpu'),q,forward_adapter=pure)
            with gzip.open(q,'rt') as f:other=[json.loads(line) for line in f]
            self.assertEqual([v['predicted_mixed'] for v in after],[v['predicted_mixed'] for v in other])

    def test_actual_CP1_standalone_CPU_hardgate_checks_nine_components(self):
        values=setup();update(values);m,e,o,h,he,ho,own,old,*_=values
        hashes=cp.state_hashes(m,e,o,h,he,ho,torch.device('cpu'))
        with tempfile.TemporaryDirectory() as d,patch.object(r,'TARGET',Path(d)):
            (Path(d)/'config.json').write_text('{}')
            binding=dict(plan=dict(sha256='a'*64),config=r.a.proof(Path(d)/'config.json'))
            state=cp.joint_blob(m,e,o,h,he,ho,torch.device('cpu'),1,binding,old['source_all_rank_states'],dict(cursor=1,plan_sha256='a'*64))
            cp.save_joint(Path(d),state)
            (Path(d)/'progress.json').write_text(json.dumps(dict(cursor=1)))
            (Path(d)/'next-update-replay.json').write_text(json.dumps(dict(status='PASSED',actual=hashes,expected=hashes)))
            result=r.audit_first()
            self.assertEqual(result['cursor'],1);self.assertEqual(len(result['component_hashes']),9)
            self.assertEqual(result['legacy_Adam'],10465);self.assertEqual(result['new_head_Adam'],1)

    def test_same_full938_recipe_has_all_four_checkpoint_first_milestones(self):
        # The existing API qualification already rejects CP256 as a new E1 start.
        original=dict(source_cursor=0,source_adam_steps=10464,update_budget=938,global_questions=32,
            fit_questions=30016,fit_real_trajectories=300160,evaluation_cumulative_cursors=[256,512,768,938],
            grouping=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise'),
            recipe=dict(id='hidden-direct-count6-from-E1',trajectory_BCE_weight=1.,count_NLL_weight=1.,product_group_loss_weight=0.))
        result=r.configured(original,dict(commit='source',files={}),dict(sha256='a'*64))
        self.assertFalse(result['prepare_only']);self.assertEqual(result['checkpoint_components'],9)
        self.assertTrue(result['first_checkpoint_independent_CPU_gate'])
        with self.assertRaises(ValueError):r.configured(dict(original,source_cursor=256),dict(commit='s',files={}),{})
        text=r.render(Path('/remote/immutable-code'),'b'*64)
        self.assertIn('--gres=gpu:1',text);self.assertIn('--qos=normal',text);self.assertNotIn('dependency',text)

    def test_submission_is_guarded_and_separate_from_preparation(self):
        from . import deploy_direct_hidden_count_train as d
        from types import SimpleNamespace
        for output in ('{"status":"PASS"}\n',b'{"status":"PASS"}\n'):
            self.assertEqual(d.parse_result(SimpleNamespace(stdout=output,stderr=b'',returncode=0)),dict(status='PASS'))
        self.assertIn("joint-training.lock",d.SUBMIT)
        self.assertIn("q7b-direct-count938",d.SUBMIT)
        self.assertIn("submit-intent.json",d.SUBMIT)
        for action in ('init','chunk','extract','prepare','submit','status'):
            compile(d.remote_source(dict(base='/remote/isolated'),action),'<remote-'+action+'>','exec')
        self.assertNotEqual(r.TARGET,r.prep.TARGET)

if __name__=='__main__':unittest.main()
