"""CPU toy real next-update/VJP/fullstate tests; no GPU or real-bank claims."""
import copy
import gc
import json
import random
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from . import probe_ablation_train as u
from .g10_grouping import fixed_g5
from .probe_g10_train import (bounded_question_update,joined_graph_reference,
    validate_plan,frozen_validation,blob,accepted_g10,MERGED)


class ToyProbe(torch.nn.Module):
    def __init__(self,dropout=False):
        super().__init__();self.linear=torch.nn.Linear(2,1,dtype=torch.float64)
        self.drop=torch.nn.Dropout(.2 if dropout else 0.)
        self.register_buffer('native_clock',torch.zeros(1,dtype=torch.float64),persistent=False)
    def forward(self,data):
        # A changing nonpersistent native cache and stochastic forward test that
        # the second pass uses each original micro's exact state and random mask.
        self.native_clock.add_(1)
        return self.linear(self.drop(data['features'])).flatten()+self.native_clock.clone()*.0001


class ToyReader:
    def get(self,ids,device):
        values=[]
        for tid in ids:
            q,t=tid.split(':');values.append([int(q[1:])*.1+.2,int(t[1:])*.07])
        return dict(features=torch.tensor(values,dtype=torch.float64,device=device))


def setup(dropout=False):
    random.seed(23);np.random.seed(23);torch.manual_seed(23)
    model=ToyProbe(dropout);ema=copy.deepcopy(model).requires_grad_(False)
    opt=torch.optim.Adam(model.parameters(),lr=5e-6)
    model(dict(features=torch.ones(2,2,dtype=torch.float64))).sum().backward();opt.step()
    # Toy-only representation of populated old Adam10464, never a real source.
    for state in opt.state.values():state['step'].fill_(10464)
    groups=[dict(qid='q'+str(q),question_hash=str(q)*64,
                trajectory_ids=['q%d:t%d'%(q,t) for t in range(10)],
                labels=[int(t<k) for t in range(10)]) for q,k in enumerate((1,5,9))]
    labels=torch.tensor([g['labels'] for g in groups])
    mask=(labels[:,:,None]==1)&(labels[:,None,:]==0)
    ranks=[dict(rank=r,rng=copy.deepcopy(u.rng_state(torch.device('cpu')))) for r in range(4)]
    source=copy.deepcopy(blob(model,ema,opt,torch.device('cpu'),0,
        dict(config_sha256='toy',matched_binding_sha256='toy'),ranks))
    return model,ema,opt,groups,mask,source


def kwargs(mode='per-question-balanced'):
    return dict(grouping_kwargs=dict(mode=mode,views_per_question=4,seed=17,epoch=2),
        loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.,within_rank_weight=.4,
            inter_rank_weight=.6,stat_mean_weight=.2,stat_variance_weight=.3),
        micro_trajectories=3)


class ProbeG10TrainTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_bounded_vjp_matches_joined_gradient_adam_and_ema(self):
        for mode in ('random','per-question-balanced','global-balanced'):
            model,ema,opt,groups,mask,source=setup()
            result=bounded_question_update(model,ema,opt,ToyReader(),groups,torch.device('cpu'),
                distinguishable_pair_mask=mask,**kwargs(mode))
            gradients=[p.grad.clone() for p in model.parameters()]
            actual=[p.clone() for p in model.parameters()];actual_ema=[p.clone() for p in ema.parameters()]
            native=u.buffers(model);rng=u.rng_hash(u.rng_state(torch.device('cpu')))
            u.restore_blob(source,model,ema,opt,torch.device('cpu'))
            reference=joined_graph_reference(model,ema,opt,ToyReader(),groups,torch.device('cpu'),
                distinguishable_pair_mask=mask,**kwargs(mode))
            for got,want in zip(gradients,[p.grad for p in model.parameters()]):
                torch.testing.assert_close(got,want,rtol=1e-12,atol=1e-12)
            for got,want in zip(actual,model.parameters()):torch.testing.assert_close(got,want,rtol=1e-13,atol=1e-13)
            for got,want in zip(actual_ema,ema.parameters()):torch.testing.assert_close(got,want,rtol=1e-13,atol=1e-13)
            self.assertEqual(result['components'],reference['components'])
            self.assertEqual(result['denominators']['question'],3)
            self.assertEqual(result['actual_forward_trajectory_exposures'],60)
            self.assertEqual(result['unique_real_trajectories'],30)
            self.assertEqual(result['maximum_retained_transformer_graph_microbatches'],1)
            self.assertTrue(result['grouping']['per_question_total_weight_exactly_one'])
            self.assertEqual(u.steps(opt),10465)
            self.assertEqual(u.tensor_hash(native),u.tensor_hash(u.buffers(model)))
            self.assertEqual(rng,u.rng_hash(u.rng_state(torch.device('cpu'))))

    def test_actual_same_source_double_replay_includes_dropout_and_native_state(self):
        model,ema,opt,groups,mask,source=setup(dropout=True);device=torch.device('cpu')
        bounded_question_update(model,ema,opt,ToyReader(),groups,device,distinguishable_pair_mask=mask,**kwargs())
        expected=u.state_hashes(model,ema,opt,device)
        u.restore_blob(source,model,ema,opt,device)
        bounded_question_update(model,ema,opt,ToyReader(),groups,device,distinguishable_pair_mask=mask,**kwargs())
        self.assertEqual(expected,u.state_hashes(model,ema,opt,device))
        self.assertEqual(source['steps'],10464)
        self.assertEqual(int(source['optimizer']['state'][0]['step']),10464)

    def test_accepted_fullstate_save_load_next_update_is_equal(self):
        model,ema,opt,groups,mask,source=setup(dropout=True);device=torch.device('cpu')
        bounded_question_update(model,ema,opt,ToyReader(),groups,device,distinguishable_pair_mask=mask,**kwargs())
        saved=copy.deepcopy(blob(model,ema,opt,device,1,
            dict(config_sha256='toy',matched_binding_sha256='toy'),source['source_all_rank_states']))
        with tempfile.TemporaryDirectory() as directory:
            receipt=u.save_checkpoint(Path(directory),saved)
            bounded_question_update(model,ema,opt,ToyReader(),groups,device,distinguishable_pair_mask=mask,**kwargs())
            expected=u.state_hashes(model,ema,opt,device)
            loaded=torch.load(u.checked(receipt),map_location='cpu',mmap=True,weights_only=False)
            u.restore_blob(loaded,model,ema,opt,device)
            self.assertEqual(loaded['cursor'],1);self.assertEqual(u.steps(opt),10465)
            self.assertFalse(loaded['optimizer_reset']);self.assertFalse(loaded['block_reset'])
            bounded_question_update(model,ema,opt,ToyReader(),groups,device,distinguishable_pair_mask=mask,**kwargs())
            self.assertEqual(expected,u.state_hashes(model,ema,opt,device))
            self.assertEqual(u.steps(opt),10466);del loaded;gc.collect()

    def test_changed_second_pass_input_fails_before_any_adam_write(self):
        class ChangedReader(ToyReader):
            def __init__(self):self.calls=0
            def get(self,ids,device):
                self.calls+=1;data=super().get(ids,device)
                if self.calls>10:data['features'].add_(.5)
                return data
        model,ema,opt,groups,mask,source=setup();before=u.model_hash(model)
        with self.assertRaisesRegex(ValueError,'micro logits differ'):
            bounded_question_update(model,ema,opt,ChangedReader(),groups,torch.device('cpu'),
                distinguishable_pair_mask=mask,**kwargs())
        self.assertEqual(u.steps(opt),10464);self.assertEqual(u.model_hash(model),before)

    def test_loss_memory_budget_and_smaller_Q_require_explicit_acknowledgment(self):
        model,ema,opt,groups,mask,_=setup()
        with self.assertRaisesRegex(ValueError,'memory budget'):
            bounded_question_update(model,ema,opt,ToyReader(),groups,torch.device('cpu'),
                distinguishable_pair_mask=mask,max_virtual_views=2,**kwargs())
        plan=dict(cohort_qids=[g['qid'] for g in groups],batches=[[g['qid'] for g in groups]],batch_epochs=[0])
        config=dict(global_questions=3,update_budget=1)
        with self.assertRaisesRegex(ValueError,'acknowledge'):validate_plan(plan,groups,config)
        config['smaller_question_budget_acknowledged']=True
        self.assertEqual(len(validate_plan(plan,groups,config)),3)
        with self.assertRaisesRegex(ValueError,'external'):validate_plan(plan,groups,config,forbidden=['q0'])

    def test_validation_uses_frozen_labelblind_ids_then_own_EM(self):
        *_,groups,mask,source=setup()
        plan=fixed_g5(groups,selection='random',role='validation',seed=19)
        first=frozen_validation(groups,plan)
        changed=copy.deepcopy(groups)
        for g in changed:g['labels']=[1-y for y in g['labels']]
        second=frozen_validation(changed,plan)
        self.assertEqual([g['trajectory_ids'] for g in first],[g['trajectory_ids'] for g in second])
        self.assertEqual(second[0]['labels'],[1-y for y in first[0]['labels']])
        bad=copy.deepcopy(plan);bad['summary']['labels_consulted']=True
        with self.assertRaisesRegex(ValueError,'label-blind'):frozen_validation(groups,bad)
        bad=copy.deepcopy(plan);bad['groups'][0]['trajectory_ids'][0]='foreign'
        with self.assertRaisesRegex(ValueError,'own actual'):frozen_validation(groups,bad)
        final_plan=fixed_g5(groups,selection='random',role='final',seed=19)
        with self.assertRaisesRegex(ValueError,'label-blind'):frozen_validation(groups,final_plan)

    def test_global_balance_routes_real_update_and_rejects_infeasible_plan(self):
        from .probe_g10_train import run_update
        model,ema,opt,groups,mask,source=setup(dropout=True)
        groups=groups[:2];groups[0]['labels']=[0]*10
        qids=[g['qid'] for g in groups];lookup={g['qid']:g for g in groups}
        c=dict(global_questions=2,update_budget=1,smaller_question_budget_acknowledged=True,
            grouping=dict(mode='global-weighted-balanced',views_per_question=4,seed=17,
                          on_unbalance='raise',weighting='equal-view-quota'),
            recipe=dict(loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.)),
            micro_trajectories=3,max_virtual_views=64,max_inter_view_pairs=4096)
        plan=dict(cohort_qids=qids,batches=[qids],batch_epochs=[0])
        self.assertEqual(len(validate_plan(plan,groups,c)),2)
        result=run_update(model,ema,opt,ToyReader(),lookup,qids,c,torch.device('cpu'),0)
        self.assertEqual(result['grouping']['mixed_views'],4)
        self.assertEqual(result['grouping']['nonmixed_views'],4)
        self.assertTrue(result['grouping']['count_balanced'])
        self.assertTrue(result['grouping']['weighted_class_mass_balanced'])
        self.assertTrue(result['grouping']['per_question_total_weight_exactly_one'])
        self.assertEqual(u.steps(opt),10465)
        groups[1]['labels']=[1]*10
        with self.assertRaisesRegex(ValueError,'cannot attain'):validate_plan(plan,groups,c)

    def test_accepted_merge_rejoins_actual_old_and_extra_own_EM(self):
        *_,groups,mask,source=setup()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def put(name,value):
                path=root/name;path.write_text(json.dumps(value))
                return dict(path=str(path),bytes=path.stat().st_size,sha256=u.sha(path))
            old=[dict(g,trajectory_ids=g['trajectory_ids'][:5],labels=g['labels'][:5]) for g in groups]
            extra=[dict(g,trajectory_ids=g['trajectory_ids'][5:],labels=g['labels'][5:]) for g in groups]
            merged=[dict(g,original_fixed5_trajectory_ids=g['trajectory_ids'][:5]) for g in groups]
            old_ref=put('old.json',dict(train=old))
            extra_ref=put('shard.json',dict(status='EXTRA_REAL_G5_SHARD_NATIVE_FULL_OWN_EM_SHA_VERIFIED',
                actor_updates=0,synthetic_EM=False,groups=extra))
            complete=dict(status=MERGED,actor_updates=0,synthetic_EM=False,
                model_revision='a09a35458c702b33eeacc393d103063234e8bc28',
                original_fixed5_preserved=True,validation_or_final_questions_generated=False,
                questions=3,trajectories=30,extra_real_trajectories=15,
                groups=put('groups.json',dict(train=merged)),original_G5_bank=old_ref,
                extra_shards=[extra_ref],ordered_qid_sha256=u.digest([g['qid'] for g in groups]),
                original_feature_manifest=put('original-features.json',[]),
                extra_G5_feature_sources=put('extra-features.json',[]))
            reference=put('complete.json',complete)
            accepted,_,_=accepted_g10(reference)
            self.assertEqual(accepted,merged)
            bad=copy.deepcopy(merged);bad[0]['labels'][5]=1-bad[0]['labels'][5]
            complete['groups']=put('bad-groups.json',dict(train=bad))
            bad_ref=put('bad-complete.json',complete)
            with self.assertRaisesRegex(ValueError,'own EM'):accepted_g10(bad_ref)
            Path(reference['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):accepted_g10(reference)


if __name__=='__main__':unittest.main()
