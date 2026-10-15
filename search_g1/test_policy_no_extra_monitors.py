"""Remote CPU tests for actual no-extra-monitor execution and honest receipts."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from search_g1.numerics import require_finite


class NoExtraMonitorTests(unittest.TestCase):
    def run_real_minibatch(self, extra):
        source=(Path(__file__).resolve().parents[1]/'verl/workers/actor/dp_actor.py').read_text()
        tree=ast.parse(source)
        actor_class=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DataParallelPPOActor')
        method=next(n for n in actor_class.body if isinstance(n,ast.FunctionDef) and n.name=='_run_minibatch')
        dualclips=[];monitors=[];forward_calls=[];actual_writes=[]
        parameter=torch.nn.Parameter(torch.tensor(.1));optimizer=torch.optim.SGD([parameter],lr=.01)
        def policy_loss(**kwargs):
            dualclips.append(kwargs['dual_clip'])
            loss=(kwargs['log_prob']*kwargs['advantages']).mean()
            return loss,loss.detach()*0,loss.detach()*0
        def append(metrics,values):
            for k,v in values.items():metrics.setdefault(k,[]).append(v)
        namespace=dict(torch=torch,FSDP=type('NoFSDP',(),{}),core_algos=SimpleNamespace(compute_policy_loss=policy_loss),
            require_finite=require_finite,append_to_dict=append)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'actual_actor_minibatch.py','exec'),namespace)
        class Micro(dict):
            def cuda(self,**kwargs):return self
        micro=Micro(responses=torch.ones(2,3,dtype=torch.long),loss_mask=torch.ones(2,3),
            old_log_probs=torch.zeros(2,3),advantages=-torch.ones(2,3))
        mini=SimpleNamespace(split=lambda size:[micro])
        config=SimpleNamespace(use_dynamic_bsz=False,ppo_micro_batch_size=2,state_masking=True,
            entropy_coeff=0.,use_kl_loss=False,clip_ratio=.2)
        def forward(**kwargs):
            forward_calls.append(True);return None,parameter.expand(2,3)
        def step():
            self.assertTrue(torch.isfinite(parameter.grad));optimizer.step();actual_writes.append(True)
            return torch.tensor(.1)
        actor=SimpleNamespace(actor_module=object(),config=config,actor_optimizer=optimizer,gradient_accumulation=1,
            _forward_micro_batch=forward,_optimizer_step=step,
            _monitor_old_policy=lambda *args:monitors.append(args[-1]))
        safety={'dual_clip':3.}
        class Safety(dict):
            __getattr__=dict.__getitem__
        safety=Safety(safety)
        if extra is not None:safety['extra_policy_monitor_forwards']=extra
        with patch('torch.cuda.current_device',return_value=0):
            namespace['_run_minibatch'](actor,mini,temperature=1.,safety_enabled=True,safety=safety,
                frozen=None,tail_limit=.01,metrics={})
        return dict(monitors=monitors,forwards=len(forward_calls),dualclips=dualclips,writes=len(actual_writes))

    def test_no_extra_forward_real_call_count_zero_dualclip_still_three_and_write_occurs(self):
        result=self.run_real_minibatch(False)
        self.assertEqual(result,dict(monitors=[],forwards=1,dualclips=[3.],writes=1))
    def test_old_default_still_calls_both_original_monitors(self):
        result=self.run_real_minibatch(None)
        self.assertEqual(result['monitors'],['before_minibatch','after_minibatch'])
        self.assertEqual(result['dualclips'],[3.]);self.assertEqual(result['writes'],1)
    def test_full_eight_rank_twenty_write_attempt_with_no_monitor_receipts(self):
        from search_g1.test_policy_threshold_diagnostics import ThresholdDiagnosticTests
        fixture=ThresholdDiagnosticTests()
        result=fixture.audit_complete_fake_attempt(None,extra_monitor_forwards=False)
        self.assertEqual(result['global_monitor_count'],0)
        self.assertEqual(result['actual_monitor_forward_count'],0)
        self.assertEqual(result['reused_monitor_forward_count'],0)
        self.assertEqual(result['guard_triggered_monitor_count'],0)

if __name__=='__main__':unittest.main()
