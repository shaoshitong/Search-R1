"""Remote CPU tests: finite heuristics warn; actual invalid numerics still reject."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from search_g1.policy_safety import monitor,dual_clip_terms,bind_actual_config
from reproduction.common8.formal_attempt_audit import _monitor_statistics,_audit_transaction_rows
from reproduction.common8.formal_contract import consumed_lr
from reproduction.common8.safety_calibration_evidence import MONITOR_TENSORS,OBJECTIVE_WEIGHT_DEFINITION
from reproduction.common8.topology import monitor_scope


class ThresholdDiagnosticTests(unittest.TestCase):
    def observe(self,delta,*,advantages=None,weights=None,mask=None,diagnostic=True):
        d=torch.tensor(delta,dtype=torch.float64)
        a=-torch.ones_like(d) if advantages is None else torch.tensor(advantages,dtype=torch.float64)
        w=torch.ones_like(d) if weights is None else torch.tensor(weights,dtype=torch.float64)
        m=torch.ones_like(d) if mask is None else torch.tensor(mask,dtype=torch.float64)
        return monitor(d,a,m,w,c=3.,kl_log_bound=30.,target_kl=.2,negative_tail_limit=.01,
            finite_thresholds_diagnostic_only=diagnostic)
    def test_finite_kl_and_tail_thresholds_warn_without_skip(self):
        result=self.observe([2.])
        self.assertTrue(result['allowed']);self.assertEqual(result['reasons'],[])
        self.assertEqual(result['diagnostic_warnings'],['old_policy_kl','negative_advantage_tail'])
        self.assertEqual(result['enforcement']['dual_clip'],'enforced')
        audited=_monitor_statistics(result,diagnostic_only=True)
        self.assertTrue(audited['allowed'])
    def test_finite_delta_over_logbound_allowed_without_fabricated_kl(self):
        result=self.observe([1000.])
        self.assertTrue(result['allowed']);self.assertFalse(result['kl_estimate_complete'])
        self.assertIsNone(result['sampled_old_policy_kl'])
        self.assertIn('delta_above_kl_log_bound',result['diagnostic_warnings'])
        self.assertEqual(result['overflow_count'],1)
        self.assertTrue(_monitor_statistics(result,diagnostic_only=True)['allowed'])
    def test_old_default_mode_still_enforces_original_thresholds(self):
        result=self.observe([2.],diagnostic=False)
        self.assertFalse(result['allowed']);self.assertEqual(result['diagnostic_warnings'],[])
        self.assertIn('old_policy_kl',result['reasons'])
        self.assertFalse(_monitor_statistics(result)['allowed'])
    def test_real_nan_negative_weights_and_empty_effective_input_rejected(self):
        for kwargs in (dict(delta=[float('nan')]),dict(delta=[0.],weights=[-1.]),dict(delta=[0.],mask=[0.])):
            result=self.observe(**kwargs)
            self.assertFalse(result['allowed']);self.assertTrue(result['fatal_reasons'])
            self.assertFalse(_monitor_statistics(result,diagnostic_only=True)['allowed'])
    def test_auditor_does_not_accept_warning_as_fatal_or_unbound_mode(self):
        result=self.observe([2.])
        with self.assertRaises(Exception):_monitor_statistics(result)
        with self.assertRaises(Exception):_monitor_statistics(dict(result,allowed=False,reasons=['old_policy_kl']),True)
    def test_dual_clip_three_still_binds_finite_extreme_objective(self):
        objective=dual_clip_terms(torch.tensor([1000.,1000.]),torch.tensor([-1.,1.]),torch.ones(2),epsilon=.2,c=3.)
        torch.testing.assert_close(objective,torch.tensor([3.,-1.2],dtype=torch.float64))
    def test_new_mode_is_manifest_and_binary_architecture_bound(self):
        config=OmegaConf.create(dict(clip_ratio=.2,use_dynamic_bsz=False,policy_safety=dict(enabled=True,
            dual_clip=3.,target_old_policy_kl=.2,kl_log_bound=30.,monitor_trajectories_per_rank=4,
            finite_thresholds_diagnostic_only=True)))
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);(base/'logs').mkdir()
            run=dict(profile='common8',training_world_size=8,finite_thresholds_diagnostic_only=True,
                question_probe=dict(probe_architecture='qwen2_decoder3_binary_mix_v1'))
            manifest=base/'logs/run-manifest.json';manifest.write_text(json.dumps(run))
            with patch.dict(os.environ,SEARCH_G1_PROFILE='common8',SEARCH_G1_BASE=str(base)):
                bind_actual_config(config)
                manifest.write_text(json.dumps(dict(run,finite_thresholds_diagnostic_only=False)))
                with self.assertRaises(RuntimeError):bind_actual_config(config)
                manifest.write_text(json.dumps(dict(run,question_probe=dict(probe_architecture='qwen2_decoder3_count6'))))
                with self.assertRaises(RuntimeError):bind_actual_config(config)
    def audit_complete_fake_attempt(self, monitor_result, *, extra_monitor_forwards=True):
        # Structural synthetic fixture only: no CUDA/model or actual Adam claim.
        run=dict(profile='common8',training_world_size=8,finite_thresholds_diagnostic_only=True,extra_policy_monitor_forwards=extra_monitor_forwards)
        binding=dict(run_id='fixture',calibration_sha256='a'*64)
        attempt=dict(logical_update=1,attempt_id='b'*32,attempt_index=1,
            behavior_probability_gate_sha256='c'*64,opportunities_processed=20,
            optimizer_writes_before=0,optimizer_writes_after=20,
            opportunity_statuses=['written']*20,complete_outer=True)
        identity=dict(selection_indices=[0,1,2,3],actor_token_lengths=[1,2,3,4],
            tensor_sha256={name:'d'*64 for name in MONITOR_TENSORS})
        rows_by_rank={}
        for rank in range(8):
            rows=[]
            for mini in range(20):
                context=dict(logical_update=1,minibatch_index=mini,prior_optimizer_writes=mini,
                    attempt_id=attempt['attempt_id'],attempt_index=1,
                    behavior_probability_gate_sha256=attempt['behavior_probability_gate_sha256'])
                stages=(('before_minibatch',mini),('after_minibatch',mini+1)) if extra_monitor_forwards else ()
                for stage,write_count in stages:
                    rows.append(dict(monitor_result,kind='POLICY_MONITOR',rank=rank,run_id='fixture',
                        context=context,stage=stage,optimizer_writes=write_count,
                        calibration_sha256='a'*64,monitoring_scope=monitor_scope(run),
                        monitor_identity=identity,rejection_count=0,
                        objective_weight_definition=OBJECTIVE_WEIGHT_DEFINITION))
                rows.append(dict(kind='OPTIMIZER_WRITE',rank=rank,run_id='fixture',context=context,
                    write_count=mini+1,learning_rates=[consumed_lr(1)],gradient_norm_before=.5,gradient_norm_after=.5))
                rows.append(dict(kind='OPPORTUNITY',rank=rank,run_id='fixture',context=context,
                    opportunity_index=mini,processed=True,status='written',optimizer_write_succeeded=True,
                    optimizer_write_rolled_back=False,optimizer_call_attempted=True,
                    optimizer_writes_before=mini,optimizer_writes_after=mini+1,guard_stage=None,guard_reasons=[]))
            rows.append(dict(kind='OUTER_BUDGET',rank=rank,run_id='fixture',opportunities_processed=20,
                optimizer_writes=20,cumulative_writes=20,extra_policy_monitor_forwards=extra_monitor_forwards,
                actual_monitor_forward_count=40 if extra_monitor_forwards else 0,reused_monitor_forward_count=0))
            rows_by_rank[rank]=rows
        return _audit_transaction_rows(rows_by_rank,binding,attempt,0,run=run)

    def test_complete_twenty_opportunity_attempt_accepts_finite_threshold_warnings(self):
        result=self.audit_complete_fake_attempt(self.observe([1000.]))
        self.assertEqual(result['global_monitor_count'],40)
        self.assertEqual(result['rejection_count'],0)
        self.assertEqual(result['guard_triggered_monitor_count'],0)

    def test_complete_attempt_cannot_claim_written_adam_after_real_nonfinite(self):
        with self.assertRaises(Exception):self.audit_complete_fake_attempt(self.observe([float('nan')]))

    def test_actor_passes_flag_while_dualclip_computation_remains_enabled(self):
        source=(Path(__file__).resolve().parents[1]/'verl/workers/actor/dp_actor.py').read_text()
        self.assertIn("finite_thresholds_diagnostic_only=safety.get('finite_thresholds_diagnostic_only',False)",source)
        self.assertIn('dual_clip=float(safety.dual_clip) if safety_enabled else None',source)

if __name__=='__main__':unittest.main()
