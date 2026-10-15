import unittest
from pathlib import Path
from search_g1.qprobe_driver_metrics import selection_diagnostics

class DriverDiagnosticsTests(unittest.TestCase):
    def audit(self):return dict(selected_all_wrong=30,selected_all_correct=40,
        selected_random_exploration=64,selected_predicted_mixed=448,
        prequential={'predicted_mixed':{'questions':448,'actual_mixed_fraction':.9},
                     'random_exploration':{'questions':64,'actual_mixed_fraction':.4}})
    def test_binary_producer_keys_are_consumed_by_actual_driver(self):
        result=selection_diagnostics(self.audit())
        self.assertEqual(result['selection/selected_predicted_mixed'],448)
        self.assertEqual(result['question_probe/prequential/predicted_mixed/actual_mixed_fraction'],.9)
        driver=(Path(__file__).resolve().parents[1]/'verl/trainer/ppo/ray_trainer.py').read_text()
        self.assertIn('metrics.update(selection_diagnostics(mixed_audit))',driver)
        self.assertNotIn('selected_predicted_diversity',driver)
        self.assertNotIn("'predicted_diversity'",driver)
    def test_diagnostic_failure_boolean_and_nonfinite_metrics_do_not_stop_driver(self):
        audit=self.audit();audit['prequential']={'diagnostic_error':{'stage':'metric'},'diagnostics_complete':False}
        result=selection_diagnostics(audit)
        self.assertEqual(result['question_probe/diagnostic_reporting_incomplete'],1)
        self.assertEqual(result['selection/selected_predicted_mixed'],448)
        audit['prequential']={'all':{'nan':float('nan'),'unavailable':None,'valid':.7}}
        result=selection_diagnostics(audit)
        self.assertEqual(result['question_probe/prequential/all/valid'],.7)
        self.assertNotIn('question_probe/prequential/all/nan',result)
        self.assertEqual(result['question_probe/diagnostic_reporting_incomplete'],1)
    def test_missing_diagnostic_schema_is_only_an_incomplete_report(self):
        for audit in (None,{},dict(self.audit(),prequential=None)):
            self.assertEqual(selection_diagnostics(audit)['question_probe/diagnostic_reporting_incomplete'],1)

if __name__=='__main__':unittest.main()
