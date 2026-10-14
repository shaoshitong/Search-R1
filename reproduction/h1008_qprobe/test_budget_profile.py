import unittest
from reproduction.common8.precision_contract import bf16_policy,NATIVE_BF16_PROFILE,remove_padding
from reproduction.common8.topology import actor_global_micro,actor_local_mini


class H1008BudgetTests(unittest.TestCase):
    def run_config(self):return dict(profile='common8',training_world_size=8,execution_backend='slurm',
        execution_site='h1008',actor_micro_batch_per_rank=4,vllm_gpu_memory_utilization=.1,
        precision_profile=NATIVE_BF16_PROFILE)
    def test_eight_ranks_keep_mini128_with_local_micro4_accumulation4(self):
        run=self.run_config();self.assertTrue(bf16_policy(run));self.assertTrue(remove_padding(run))
        self.assertEqual(actor_global_micro(run),32);self.assertEqual(actor_local_mini(run),16)
        self.assertEqual(actor_local_mini(run)//run['actor_micro_batch_per_rank'],4)
        self.assertEqual(512*5//128,20)
    def test_existing_native_eight_profile_stays_local_micro2(self):
        run=self.run_config();run.pop('execution_site')
        with self.assertRaises(AssertionError):bf16_policy(run)
        run['actor_micro_batch_per_rank']=2;self.assertTrue(bf16_policy(run))
    def test_new_same_budget_recovery_can_keep_micro4(self):
        run=self.run_config();run['full_state_restore']={'kind':'question_probe_pair'}
        self.assertTrue(bf16_policy(run))

if __name__=='__main__':unittest.main()
