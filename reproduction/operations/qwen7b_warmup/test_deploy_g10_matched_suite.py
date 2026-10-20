"""CPU metadata/planning/render checks; no real-bank, tensor or remote access."""
import copy
from fractions import Fraction
from pathlib import Path
import subprocess
import unittest
from . import probe_ablation_train as u
from .g10_matched_inputs import (TARGET,VARIANTS,VALIDATION_STATUS,validation_consumer,
    shared_plan,make_config,render_sbatch)
from .g10_weighted_balanced import build_training_views
from .deploy_g10_matched_suite import self_test


def groups(h=3000):
    return [dict(qid='q%05d'%i,trajectory_ids=['q%05d:t%d'%(i,j) for j in range(10)],
        labels=[int(j<(0 if i<h else 5)) for j in range(10)]) for i in range(6528)]


class MatchedG10DeploymentTests(unittest.TestCase):
    def test_same6528_B48_136_shared_plan_and_four_equal_unique_views(self):
        own=groups();reference=dict(cohort_qids=[g['qid'] for g in own])
        plan=shared_plan(own,reference)
        self.assertEqual(len(plan['batches']),136);self.assertTrue(all(len(b)==48 for b in plan['batches']))
        self.assertLessEqual(max(plan['batch_homogeneous_counts']),24)
        self.assertEqual(set(plan['cohort_qids']),set(reference['cohort_qids']))
        self.assertTrue(plan['same_plan_required_for_random_and_balanced'])
        by={g['qid']:g for g in own};batch=[by[q] for q in plan['batches'][0]]
        balanced=build_training_views(batch);random=build_training_views(batch,mode='random')
        self.assertEqual(balanced['summary']['mixed_views'],96);self.assertEqual(balanced['summary']['nonmixed_views'],96)
        for selected in (balanced,random):
            self.assertEqual(len(selected['groups']),192)
            self.assertTrue(all(Fraction(v['weight_numerator'],v['weight_denominator'])==Fraction(1,4) for v in selected['groups']))
            for qid in plan['batches'][0]:
                views=[v for v in selected['groups'] if v['qid']==qid]
                self.assertEqual(len({tuple(v['source_indices']) for v in views}),4)
        wrong=copy.deepcopy(reference);wrong['cohort_qids'][0]='other'
        with self.assertRaisesRegex(ValueError,'SAME6528'):shared_plan(own,wrong)

    def test_infeasible_true_cohort_never_fallbacks_or_changes_questions(self):
        own=groups(4000)
        with self.assertRaisesRegex(ValueError,'impossible'):
            shared_plan(own,dict(cohort_qids=[g['qid'] for g in own]))

    def test_configs_share_source_plan_and_validation_but_explicit_changed_budget(self):
        reference=dict(source_checkpoint=dict(path='/remote/E1',sha256=u.SOURCE_SHA),
            source_config=dict(path='/remote/config',sha256='1'*64),source_bank=dict(path='/remote/bank',sha256='2'*64),
            model_config=dict(path='/remote/model/config.json',sha256='3'*64),model_dir='/remote/model')
        plan=dict(path='/remote/new-plan',sha256='4'*64);evaluations=[dict(name='selection2048FreshG5',role='validation')]
        configs=[make_config(reference,n,{},'a'*40,{'path':'/remote/train','sha256':'5'*64},
            {'path':'/remote/validation','sha256':'6'*64},plan,evaluations) for n in VARIANTS]
        for key in ('source_checkpoint','source_config','source_bank','model_config','model_dir','plan','evaluations'):
            self.assertEqual(configs[0][key],configs[1][key])
        for c in configs:
            self.assertEqual((c['global_questions'],c['update_budget'],c['micro_trajectories']),(48,136,5))
            self.assertTrue(c['smaller_question_budget_acknowledged']);self.assertFalse(c['equal_budget_to_B51_128_baseline_claim'])
            self.assertFalse(c['optimizer_reset']);self.assertFalse(c['readout_reset'])
        self.assertEqual(configs[1]['grouping']['weighting'],'equal-view-quota')
        self.assertEqual(configs[1]['grouping']['on_unbalance'],'raise');self.assertNotIn('weighting',configs[0]['grouping'])

    def test_precommitted_validation_role_and_consumer_gate(self):
        val=dict(status=VALIDATION_STATUS,role='validation',untouched_final=False,tuning_allowed=True,
            fitting_on_validation_allowed=False,questions=2048,trajectories=20480,real_fresh_events=2,
            real_trajectories_per_event=10240,actor_optimizer_updates=0,synthetic_EM=False,
            fixed_G5_was_chosen_before_own_EM=True,consumer_inputs=dict(groups={},features={},fixed_G5_plan={},
                group_split='validation',features_format='flat_list_of_native_complete_paths',selection_labels_consulted=False))
        self.assertEqual(validation_consumer(val),val['consumer_inputs'])
        for key,value in [('status','PARTIAL'),('untouched_final',True),('fixed_G5_was_chosen_before_own_EM',False),('questions',512)]:
            bad=copy.deepcopy(val);bad[key]=value
            with self.assertRaises(ValueError):validation_consumer(bad)
        bad=copy.deepcopy(val);bad['consumer_inputs']['selection_labels_consulted']=True
        with self.assertRaisesRegex(ValueError,'label-blind'):validation_consumer(bad)

    def test_safe_sbatch_and_bounded_published_source_closure(self):
        for name in VARIANTS:
            script=render_sbatch(name,TARGET+'/'+name,TARGET+'/code-'+'a'*12)
            subprocess.run(['bash','-n'],input=script,text=True,check=True)
            self.assertIn('#SBATCH --gres=gpu:1\n',script);self.assertIn('#SBATCH --qos=normal\n',script)
            self.assertNotIn('--dependency',script)
        result=self_test();self.assertFalse(result['training_launched']);self.assertFalse(result['remote_actions_executed'])
        self.assertLess(max(result['command_bytes'].values()),16000)


if __name__=='__main__':unittest.main()
