"""Pure paired-metadata, resource and bounded-transport checks; no remote IO."""
import copy,unittest
from .hard_enrichment_train_prepare import make_config,check_selection
from .hard_enrichment_train import SOURCE_SHA,LOSS
from .deploy_hard_enrichment_suite import self_test


def ref(name):return dict(path='/data/source/'+name,bytes=100,sha256=(name[0] if name[0] in 'abcdef' else 'a')*64)


def selection():
    outputs={}
    for name in ('targeted','random_control'):
        outputs[name]=dict(groups=ref(name+'-groups'),plan=ref(name+'-plan'),summary=dict(branch=name,total_unique_questions=22896,
            core_questions=6528,supplement_questions=16368,supplement_class_counts=dict(allwrong=4092,allcorrect=4092,Mixed=8184),
            update_budget=477,global_groups=48,epochs=1,seed=20261021,validation_or_final_questions_used=False,
            question_selection_uses_scores=name=='targeted'))
    return dict(status='TRAIN_ONLY_HARD_AND_RANDOM22896Q_CLASS_MATCHED_PLANS_COMPLETE',outputs=outputs,
        checkpoint=dict(sha256=SOURCE_SHA,steps=14418,epoch=4),GPU_used=False,actor_updates=0,probe_updates=0,
        controls=dict(same_core=True,same_unique_count=True,same_macro_class_counts=True,same_seed=True,same_update_budget=True,
            Mixed_K1_K2_K3_K4_matched=False,initial_warmup_unseen_claim=False,previous_finetuning_unseen_claim=False),
        source_bank=ref('bank'),features=ref('features'),source_candidates=ref('candidate'),source_scoring_receipt=ref('complete'))


class HardEnrichmentDeploymentTests(unittest.TestCase):
    def test_shared_E4_loss_optimizer_budget_and_external_fixture_only_data_differs(self):
        selected=selection();source=dict(path='/data/source/E4.pt',bytes=100,sha256=SOURCE_SHA,steps=14418,epoch=4,model_dir='/data/source/model')
        configs=[make_config(name,selected,ref('selection'),source,ref('receipt'),ref('config'),ref('model'),ref('manifests'),{},'a'*40) for name in ('targeted','random_control')]
        differing={k for k in configs[0] if configs[0][k]!=configs[1][k]}
        self.assertEqual(differing,{'variant','bank','plan'})
        for c in configs:
            self.assertEqual(c['recipe']['loss_kwargs'],LOSS);self.assertEqual(c['source_checkpoint']['steps'],14418)
            self.assertFalse(c['E1_reset']);self.assertTrue(c['source_raw_ema_adam_RNG_preserved'])
            self.assertEqual((c['update_budget'],c['global_groups'],c['fit_questions']),(477,48,22896))
            self.assertFalse(c['four_rank_bitexact_resume_claim'])

    def test_partial_scoring_or_unmatched_selection_budget_rejected(self):
        original=selection();check_selection(original)
        for key,value in [('status','PARTIAL'),('probe_updates',1)]:
            v=copy.deepcopy(original);v[key]=value
            with self.assertRaises(ValueError):check_selection(v)
        v=copy.deepcopy(original);v['outputs']['targeted']['summary']['supplement_class_counts']['allcorrect']=4091
        with self.assertRaises(ValueError):check_selection(v)
        v=copy.deepcopy(original);v['checkpoint']['steps']=10464
        with self.assertRaises(ValueError):check_selection(v)

    def test_actual_import_closure_and_each_control_command_below16KB(self):
        result=self_test();self.assertFalse(result['remote_actions_executed'])
        self.assertLess(max(result['command_bytes'].values()),16000)
        self.assertGreaterEqual(result['runtime_files'],13)


if __name__=='__main__':unittest.main()
