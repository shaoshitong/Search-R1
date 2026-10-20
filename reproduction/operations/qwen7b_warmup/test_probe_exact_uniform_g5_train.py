"""Focused synthetic Adam/VJP/fullstate adapter tests; no GPU qualification."""
import copy,inspect,unittest
import torch
from . import probe_g10_train as original
from . import probe_ablation_train as u
from .probe_exact_uniform_g5_train import bounded_exact_update,make_config,REQUIRED,MODE
from .test_probe_g10_train import setup,ToyReader
from .test_exact_uniform_g5_loss import brute


def options():return dict(loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.),micro_trajectories=3,combination_chunk=17)


class ExactUniformRunnerTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_bounded_two_pass_adam_ema_gradient_matches_independent_joined_brute(self):
        model,ema,opt,gs,_,source=setup();d=torch.device('cpu');result=bounded_exact_update(model,ema,opt,ToyReader(),gs,d,**options());grad=[p.grad.clone() for p in model.parameters()];params=[p.clone() for p in model.parameters()];ep=[p.clone() for p in ema.parameters()];u.restore_blob(source,model,ema,opt,d);model.train();opt.zero_grad();tids=[t for g in gs for t in g['trajectory_ids']];parts=[]
        for start in range(0,len(tids),3):parts.append(model(ToyReader().get(tids[start:start+3],d)))
        loss=brute(torch.cat(parts).reshape(3,10),gs);loss.backward();opt.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):a.mul_(.99).add_(b,alpha=.01)
        for a,b in zip(grad,[p.grad for p in model.parameters()]):torch.testing.assert_close(a,b,rtol=2e-12,atol=2e-13)
        for a,b in zip(params,model.parameters()):torch.testing.assert_close(a,b,rtol=2e-13,atol=2e-13)
        for a,b in zip(ep,ema.parameters()):torch.testing.assert_close(a,b,rtol=2e-13,atol=2e-13)
        self.assertAlmostEqual(result['loss'],float(loss.detach()),places=12);self.assertEqual(u.steps(opt),10465);self.assertEqual(result['actual_forward_trajectory_exposures'],60);self.assertEqual(result['maximum_retained_transformer_graph_microbatches'],1)
    def test_dropout_native_RNG_double_replay_and_complete_blob_next_update(self):
        model,ema,opt,gs,_,source=setup(dropout=True);d=torch.device('cpu');bounded_exact_update(model,ema,opt,ToyReader(),gs,d,**options());expected=u.state_hashes(model,ema,opt,d);u.restore_blob(source,model,ema,opt,d);bounded_exact_update(model,ema,opt,ToyReader(),gs,d,**options());self.assertEqual(expected,u.state_hashes(model,ema,opt,d));saved=copy.deepcopy(original.blob(model,ema,opt,d,1,dict(config_sha256='toy',matched_binding_sha256='toy'),source['source_all_rank_states']));bounded_exact_update(model,ema,opt,ToyReader(),gs,d,**options());expected=u.state_hashes(model,ema,opt,d);u.restore_blob(saved,model,ema,opt,d);bounded_exact_update(model,ema,opt,ToyReader(),gs,d,**options());self.assertEqual(expected,u.state_hashes(model,ema,opt,d));self.assertEqual(u.steps(opt),10466);self.assertEqual(saved['cursor'],1);self.assertFalse(saved['optimizer_reset'])
    def test_original_API_defaults_and_actual_reference_plan_preserved(self):
        self.assertIs(inspect.signature(original.run).parameters['update_callback'].default,original.run_update)
        qids=['synthetic-q%d'%i for i in range(6528)];plan=dict(cohort_qids=qids,batches=[qids[i:i+48] for i in range(0,6528,48)],batch_epochs=[0]*136);reference=dict(source_checkpoint=dict(sha256=u.SOURCE_SHA),global_questions=48,update_budget=136,fit_questions=6528,fit_real_trajectories=65280,grouping=dict(mode='random'),plan=dict(path='/synthetic/actual-inherited-plan',sha256='a'*64),source_config=dict(sha256='b'*64),source_bank=dict(sha256='c'*64));hashes={'reproduction/operations/qwen7b_warmup/'+n:'d'*64 for n in REQUIRED};c=make_config(reference,plan,hashes=hashes,commit='e'*40);self.assertEqual(c['plan'],reference['plan']);self.assertEqual(c['source_checkpoint'],reference['source_checkpoint']);self.assertEqual(c['grouping']['mode'],MODE);self.assertEqual(c['view_weight_denominator'],252);self.assertFalse(c.get('explicit_launch_authorized',False));self.assertEqual(reference['update_budget'],136)

if __name__=='__main__':unittest.main()
