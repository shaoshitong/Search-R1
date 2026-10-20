"""Narrow CPU G20 VJP/fullstate tests, not real-GPU qualification claims."""
import copy,unittest
import torch
from . import probe_ablation_train as u
from .g20_grouping import build_training_views,build_batch_plan
from .g20_question_loss import g20_question_loss
from .probe_g20_train import bounded_question_update,validate_plan,blob
from .test_probe_g10_train import ToyReader,setup as old_setup


def setup(dropout=False):
    model,ema,opt,_,_,source=old_setup(dropout)
    groups=[dict(qid='q'+str(q),question_hash=str(q)*64,
        trajectory_ids=['q%d:t%d'%(q,t) for t in range(20)],
        labels=[int(t<k) for t in range(20)]) for q,k in enumerate((1,10,19))]
    labels=torch.tensor([g['labels'] for g in groups]);mask=(labels[:,:,None]==1)&(labels[:,None,:]==0)
    return model,ema,opt,groups,mask,source


def kwargs(mode='random'):
    return dict(grouping_kwargs=dict(mode=mode,seed=17,epoch=2),
        loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.,within_rank_weight=.4,
            inter_rank_weight=.6,stat_mean_weight=.2,stat_variance_weight=.3),micro_trajectories=3)


def joined(model,ema,opt,reader,groups,device,mask,options):
    model.train();opt.zero_grad(set_to_none=True);tids=[t for g in groups for t in g['trajectory_ids']]
    parts=[model(reader.get(tids[start:start+3],device)).reshape(-1) for start in range(0,len(tids),3)]
    views=build_training_views(groups,role='train',**options['grouping_kwargs'])['groups']
    result=g20_question_loss(torch.cat(parts).reshape(len(groups),20),groups,views,
        distinguishable_pair_mask=mask,**options['loss_kwargs'])
    result['loss'].backward();opt.step()
    with torch.no_grad():
        for left,right in zip(ema.parameters(),model.parameters()):left.mul_(.99).add_(right,alpha=.01)
    return result


class ProbeG20Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_bounded_complete20_gradient_and_adam_match_joined_reference(self):
        for mode in ('random','global-weighted-balanced'):
            model,ema,opt,groups,mask,source=setup();device=torch.device('cpu');options=kwargs(mode)
            result=bounded_question_update(model,ema,opt,ToyReader(),groups,device,distinguishable_pair_mask=mask,**options)
            gradients=[p.grad.clone() for p in model.parameters()];params=[p.clone() for p in model.parameters()];emas=[p.clone() for p in ema.parameters()]
            native=u.tensor_hash(u.buffers(model));rng=u.rng_hash(u.rng_state(device));adam=u.optimizer_hash(opt)
            u.restore_blob(source,model,ema,opt,device)
            reference=joined(model,ema,opt,ToyReader(),groups,device,mask,options)
            for got,want in zip(gradients,[p.grad for p in model.parameters()]):torch.testing.assert_close(got,want,rtol=1e-12,atol=1e-12)
            for got,want in zip(params,model.parameters()):torch.testing.assert_close(got,want,rtol=1e-13,atol=1e-13)
            for got,want in zip(emas,ema.parameters()):torch.testing.assert_close(got,want,rtol=1e-13,atol=1e-13)
            self.assertEqual(result['components'],{k:float(v.detach()) for k,v in reference['components'].items()})
            self.assertEqual(u.steps(opt),10465);self.assertEqual(result['unique_real_trajectories'],60)
            self.assertEqual(result['actual_forward_trajectory_exposures'],120);self.assertTrue(result['supports_G20'])
            self.assertEqual(u.tensor_hash(u.buffers(model)),native);self.assertEqual(u.rng_hash(u.rng_state(device)),rng)
            # Floating summation may differ; numeric Adam tensor comparison is
            # meaningful, not a monolithic bitwise-state claim.

    def test_real_same_source_double_update_replay_and_fullstate_resume(self):
        model,ema,opt,groups,mask,source=setup(True);device=torch.device('cpu')
        bounded_question_update(model,ema,opt,ToyReader(),groups,device,distinguishable_pair_mask=mask,**kwargs())
        first=u.state_hashes(model,ema,opt,device)
        saved=copy.deepcopy(blob(model,ema,opt,device,1,dict(config_sha256='toy',matched_binding_sha256='toy'),source['source_all_rank_states']))
        u.restore_blob(source,model,ema,opt,device)
        bounded_question_update(model,ema,opt,ToyReader(),groups,device,distinguishable_pair_mask=mask,**kwargs())
        self.assertEqual(first,u.state_hashes(model,ema,opt,device))
        bounded_question_update(model,ema,opt,ToyReader(),groups,device,distinguishable_pair_mask=mask,**kwargs())
        second=u.state_hashes(model,ema,opt,device);u.restore_blob(saved,model,ema,opt,device)
        bounded_question_update(model,ema,opt,ToyReader(),groups,device,distinguishable_pair_mask=mask,**kwargs())
        self.assertEqual(second,u.state_hashes(model,ema,opt,device));self.assertEqual(u.steps(opt),10466)
        self.assertFalse(saved['optimizer_reset']);self.assertFalse(saved['block_reset'])

    def test_wrong10_pool_and_changed_second_pass_fail_before_adam(self):
        model,ema,opt,groups,mask,source=setup();device=torch.device('cpu');before=u.model_hash(model)
        wrong=copy.deepcopy(groups)
        for g in wrong:g['trajectory_ids']=g['trajectory_ids'][:10];g['labels']=g['labels'][:10]
        with self.assertRaises(ValueError):bounded_question_update(model,ema,opt,ToyReader(),wrong,device,**kwargs())
        class Changed(ToyReader):
            calls=0
            def get(self,ids,device):
                self.calls+=1;data=super().get(ids,device)
                if self.calls>20:data['features'].add_(.5)
                return data
        with self.assertRaisesRegex(ValueError,'micro logits differ'):
            bounded_question_update(model,ema,opt,Changed(),groups,device,distinguishable_pair_mask=mask,**kwargs())
        self.assertEqual(u.steps(opt),10464);self.assertEqual(u.model_hash(model),before)

    def test_exact_shared6528_plan_rejects_duplication_or_unbalanced_batches(self):
        groups=[dict(qid='q%d'%q,trajectory_ids=['q%d:t%d'%(q,t) for t in range(20)],
            labels=[int(t<([0,1,10,20][q%4])) for t in range(20)]) for q in range(6528)]
        plan=build_batch_plan(groups,batch_questions=48,seed=1729)
        config=dict(global_questions=48,update_budget=136,smaller_question_budget_acknowledged=True,grouping=dict(mode='random',seed=1729))
        self.assertEqual(len(validate_plan(plan,groups,config)),6528)
        config['grouping']['mode']='global-weighted-balanced';self.assertEqual(len(validate_plan(plan,groups,config)),6528)
        bad=copy.deepcopy(plan);bad['batches'][0][0]=bad['batches'][0][1]
        with self.assertRaises(ValueError):validate_plan(bad,groups,config)


class SameG10QuestionOrderTests(unittest.TestCase):
    def test_inherits_exact_question_order_and_rejects_changed_old_EM(self):
        from .g20_matched_inputs import shared_plan,assert_old10_prefix
        from .g10_weighted_balanced import build_weighted_batch_plan
        old=[];expanded=[]
        for q in range(6528):
            k=(0,1,5,10)[q%4];old_labels=[int(t<k) for t in range(10)]
            own=dict(qid='q%d'%q,question_hash=str(q),trajectory_ids=['q%d:t%d'%(q,t) for t in range(10)],labels=old_labels)
            old.append(own);expanded.append(dict(own,trajectory_ids=['q%d:t%d'%(q,t) for t in range(20)],labels=old_labels+[int(t<3) for t in range(10)]))
        previous=build_weighted_batch_plan(old,batch_questions=48,seed=1729,epoch=0,on_unbalance='raise')
        assert_old10_prefix(expanded,old)
        got=shared_plan(expanded,{'cohort_qids':[g['qid'] for g in old]},previous,{'path':'actual-plan','sha256':'frozen'})
        self.assertEqual(got['batches'],previous['batches']);self.assertEqual(got['cohort_qids'],previous['cohort_qids']);self.assertTrue(got['inherited_G10_batches_unchanged'])
        altered=copy.deepcopy(expanded);altered[0]['labels'][0]^=1
        with self.assertRaisesRegex(ValueError,'prefix'):assert_old10_prefix(altered,old)

if __name__=='__main__':unittest.main()
