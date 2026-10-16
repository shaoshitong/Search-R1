"""The paired restart keeps its original explicit periodic schedules."""
import copy
import unittest

from reproduction.hpc3_dynrpo_periodic.resume import inherit_periodic_recipe


class ResumeRecipeTests(unittest.TestCase):
    def fixture(self):
        parent=dict(run_id='same-learner',target_updates=200,
            question_probe=dict(head_kind='qwen_native_periodic_lastblock_binary_v1'),
            question_probe_refit_outers=list(range(0,200,25)),extra_checkpoint_steps=list(range(25,201,25)))
        run={k:copy.deepcopy(parent[k]) for k in ('run_id','target_updates','question_probe')}
        run['full_state_restore']=dict(completed_outer=5,actual_optimizer_writes_per_rank=100)
        return run,parent

    def test_schedule_preserved_without_resetting_paired_prefix(self):
        run,parent=self.fixture();binding=copy.deepcopy(run['full_state_restore'])
        self.assertIs(inherit_periodic_recipe(run,parent),run)
        self.assertEqual(run['question_probe_refit_outers'],[0,25,50,75,100,125,150,175])
        self.assertEqual(run['extra_checkpoint_steps'],[25,50,75,100,125,150,175,200])
        self.assertEqual(run['full_state_restore'],binding)
        run['extra_checkpoint_steps'].append(225)
        self.assertNotIn(225,parent['extra_checkpoint_steps'])

    def test_changed_horizon_schedule_or_fresh_binding_rejected(self):
        for change in ('horizon','fit200','missing_schedule','changed_run_id','fresh'):
            run,parent=self.fixture()
            if change=='horizon':run['target_updates']=100
            if change=='fit200':parent['question_probe_refit_outers'].append(200)
            if change=='missing_schedule':parent.pop('extra_checkpoint_steps')
            if change=='changed_run_id':run['run_id']='new-learner'
            if change=='fresh':run['full_state_restore']={}
            with self.subTest(change=change),self.assertRaises(ValueError):inherit_periodic_recipe(run,parent)


if __name__=='__main__':unittest.main()
