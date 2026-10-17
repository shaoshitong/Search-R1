import unittest
from collections import Counter
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from .ab_policy import balanced_plan, ema_step, validate_receipt, BALANCED, FROZEN
from .probe_plan import rank_question_slice, ddp_mean_scale
from reproduction.prefix20_group_joint100.loss import joint_loss


class ABProbeTests(unittest.TestCase):
    def test_imbalanced_pool_resampling_preserves_g5_and_ddp_mean(self):
        for n,outer in ((512,1),(2560,10),(512,200)):
            labels=[1]*17+[0]*(n-17)
            plan,proof=balanced_plan(labels,1729,outer)
            self.assertEqual(len(plan),n//512*10)
            indices=[i for step in plan for i in step['question_indices']]
            self.assertEqual(Counter(indices),Counter(proof['sampled_indices']))
            self.assertEqual(sum(labels[i] for i in indices),n//2)
            self.assertEqual(proof['sampled_counts'],[n//2,n//2])
            self.assertEqual(Counter(indices),Counter({int(k):v for k,v in proof['group_multiplicities'].items() if v}))
            for step in plan:
                parts=[rank_question_slice(step,r) for r in range(4)]
                self.assertEqual([i for rows in parts for i in rows],step['question_indices'])
                reference=sum(i*i for i in step['question_indices'])/step['global_questions']
                reduced=sum(sum(i*i for i in rows)/len(rows)*ddp_mean_scale(len(rows),step['global_questions']) for rows in parts)/4
                self.assertAlmostEqual(reference,reduced,places=7)
            self.assertEqual(balanced_plan(labels,1729,outer),(plan,proof))

    def test_missing_class_is_explicit_zero_updates(self):
        for value in (0,1):
            plan,proof=balanced_plan([value]*512,1729,1)
            self.assertEqual(plan,[])
            self.assertEqual(proof['status'],'SKIPPED_MISSING_CLASS')
            self.assertEqual(proof['sampled_indices'],[])

    def test_ema_actual_step_closed_form_and_integer_buffers(self):
        raw=torch.nn.Linear(2,1); ema=torch.nn.Linear(2,1)
        raw.register_buffer('count',torch.tensor(3))
        ema.register_buffer('count',torch.tensor(0))
        raw.register_buffer('rope',torch.tensor([4.]),persistent=False)
        ema.register_buffer('rope',torch.tensor([0.]),persistent=False)
        with torch.no_grad():
            for p in raw.parameters(): p.fill_(2.)
            for p in ema.parameters(): p.zero_()
        for _ in range(10): ema_step(ema,raw)
        for p in ema.parameters(): torch.testing.assert_close(p,torch.full_like(p,2*(1-.99**10)))
        self.assertEqual(ema.count.item(),3)
        torch.testing.assert_close(ema.rope,torch.tensor([4*(1-.99**10)]))
        for p in raw.parameters(): torch.testing.assert_close(p,torch.full_like(p,2.))

    def test_weight_five_gradient_and_binary_k1_target(self):
        z=torch.zeros((3,5),requires_grad=True)
        labels=torch.tensor([[0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.],[1.,1.,1.,1.,1.]])
        loss=joint_loss(z,labels,group_weight=5.)
        self.assertEqual(loss['mixed_label'].tolist(),[0.,1.,0.])
        torch.testing.assert_close(loss['loss'],loss['trajectory_bce']+5*loss['group_bce'])
        loss['loss'].backward();self.assertTrue(torch.isfinite(z.grad).all())

    def test_actual_clock_allows_skips_not_fabricated_ema_updates(self):
        frozen=dict(completed_outer=0,actual_adam_steps_per_rank=[0]*4)
        self.assertEqual(validate_receipt(dict(selection_probe_experiment=FROZEN),frozen,200)['actual_adam_per_rank'],0)
        balanced=dict(completed_outer=20,actual_adam_steps_per_rank=[240]*4,ema_updates=240)
        validate_receipt(dict(selection_probe_experiment=BALANCED),balanced,20)
        with self.assertRaises(ValueError):
            validate_receipt(dict(selection_probe_experiment=BALANCED),dict(balanced,ema_updates=241),20)

    def test_joint_commit_uses_actual_ab_clock_not_original_period_clock(self):
        from .test_candidate_schedule_commit import write
        from .test_candidate_schedule import configured
        from .test_grounding_protocol import actor as actor_fixture
        from .pool import initial_state, reserve_indices
        from .candidate_schedule import candidate_count
        from .launch import commit_outer
        from reproduction.prefix20_online.launch import sha
        for arm,outer,previous,added in ((FROZEN,10,0,0),(BALANCED,1,0,10),
                                         (BALANCED,10,90,50),(BALANCED,200,2750,10),
                                         (BALANCED,1,0,0)):
            with self.subTest(arm=arm,outer=outer,added=added),tempfile.TemporaryDirectory() as d:
                base=Path(d);root=base/'outers'/('outer-%06d'%outer)
                config=configured();config.update(run_id='run',source_commit='new',selection_probe_experiment=arm)
                state={'version':'grounding0'}
                actor=actor_fixture(outer,state,actual=20*outer,new=20)
                write(base/'actor/latest.json',actor)
                actual=previous+added
                probe=write(base/'probe/latest.json',dict(run_id='run',completed_outer=0 if arm==FROZEN else outer,
                    actual_adam_steps_per_rank=[actual]*4,ema_updates=actual))
                (base/'probe/latest.pt').write_bytes(b'tiny CPU receipt fixture')
                if outer>1:write(base/'latest-joint.json',dict(probe=dict(actual_adam_steps_per_rank=[previous]*4)))
                pool=initial_state(168510)
                for n in range(1,outer+1):_,_,pool=reserve_indices(pool,n,candidate_count(config,n))
                write(base/'question-pool/state.json',pool)
                selection=root/'probe-score/selection.json';write(selection,dict(outer=outer))
                write(root/'full/complete.json',dict(actor_version='actor%d'%outer))
                update=dict(outer=outer,new_adam_steps_per_rank=added,actual_adam_steps_per_rank=[actual]*4)
                if arm==FROZEN:
                    update.update(status='PROBE_FROZEN_NO_OPTIMIZER_UPDATE')
                    write(root/'probe-frozen.json',update)
                else:
                    update.update(status='BALANCED_CURRENT_OUTER_PROBE_COMPLETE',checkpoint=probe,
                                  ema_updates=actual,sampling=dict(status='SKIPPED_MISSING_CLASS' if not added else 'BALANCED_RESAMPLED_ONE_PASS'))
                    write(root/'probe-update/metrics.json',update)
                    write(root/'probe-update/complete.json',dict(update,metrics_sha256=sha(root/'probe-update/metrics.json')))
                with patch('reproduction.prefix_selection200.grounding.checked_state',return_value=state), patch('reproduction.prefix_selection200.ab_lifecycle.selection_metrics',return_value={'mixed':256}):
                    result=commit_outer(base,config,outer,root,selection,{},grounding_state=state)
                self.assertEqual(result['probe_boundary']['actual_adam_per_rank'],actual)
                self.assertEqual(result['probe_boundary']['last_update_outer'],0 if arm==FROZEN else outer)
                self.assertTrue(Path(result['probe_snapshot']).exists())


if __name__=='__main__':unittest.main()
