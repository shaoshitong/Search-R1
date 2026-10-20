"""Small CPU tests of global graph, lineage, cursor and scoring; no GPU claims."""
import copy
import random
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from .probe_ablation_loss import probe_ablation_loss
from .probe_ablation_train import (validate_plan,run_update,rng_state,restore_rng,
    state_hashes,restore_blob,checkpoint_blob,save_checkpoint,checked,
    prediction_metrics,roc_auc,steps)


class ToyProbe(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(2,1,dtype=torch.float64)
    def forward(self,data):return self.linear(data['features']).flatten()


class ToyReader:
    def get(self,ids,device):
        return dict(features=torch.tensor([[i/100.,(i%7)/10.] for i in ids],
                                          dtype=torch.float64,device=device))


def toy():
    random.seed(17);np.random.seed(17);torch.manual_seed(17)
    model=ToyProbe();ema=copy.deepcopy(model).requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-6)
    # Represent an already-populated source Adam; the real source uses10464.
    optimizer.zero_grad();model(dict(features=torch.ones(1,2,dtype=torch.float64))).sum().backward();optimizer.step()
    groups={}
    for i in range(51):
        labels=[1,0,0,0,0] if i%2 else [0]*5
        groups[i]=dict(qid=i,trajectory_ids=list(range(i*5,i*5+5)),labels=labels,
                       pair_mask=[[bool(labels[a]==1 and labels[b]==0) for b in range(5)] for a in range(5)])
    config=dict(micro_groups=2,recipe=dict(loss_kwargs=dict(trajectory_weight=1.,
                binary_mixed_weight=1.,within_rank_weight=.3,inter_rank_weight=.2,
                stat_mean_weight=.1,stat_variance_weight=.1)))
    return model,ema,optimizer,groups,config


def cpu_blob(model,ema,optimizer):
    state=rng_state(torch.device('cpu'))
    archive=[dict(rank=i,rng=copy.deepcopy(state)) for i in range(4)]
    return copy.deepcopy(checkpoint_blob(model,ema,optimizer,torch.device('cpu'),0,
                     dict(config_sha256='config',matched_binding_sha256='matched'),archive))


class ProbeAblationTrainTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_plan_exact_cohort_budget_complete_batch_and_exclusion(self):
        _,_,_,groups,_=toy();plan=dict(cohort_qids=list(groups),batches=[list(groups)])
        self.assertEqual(len(validate_plan(plan,list(groups.values()),1)),51)
        for invalid in (dict(cohort_qids=list(groups),batches=[list(groups)[:-1]]),
                        dict(cohort_qids=list(groups)+[0],batches=[list(groups)])):
            with self.assertRaises(ValueError):validate_plan(invalid,list(groups.values()),1)
        with self.assertRaises(ValueError):validate_plan(plan,list(groups.values()),1,forbidden=[0])
        with self.assertRaises(ValueError):validate_plan(plan,list(groups.values()),2)

    def test_microforward_retains_complete_global_graph_and_exact_replay(self):
        model,ema,opt,groups,c=toy();device=torch.device('cpu');reader=ToyReader()
        source=cpu_blob(model,ema,opt);qids=list(groups)
        metrics=run_update(model,ema,opt,reader,groups,qids,c,device)
        first=state_hashes(model,ema,opt,device)
        self.assertEqual(metrics['global_questions'],51);self.assertEqual(steps(opt),2)
        self.assertEqual(metrics['inter_actual_pairs'],25*26)
        # SAME source, same variant and full batch must produce the same update.
        restore_blob(source,model,ema,opt,device)
        run_update(model,ema,opt,reader,groups,qids,c,device)
        self.assertEqual(first,state_hashes(model,ema,opt,device))
        self.assertEqual(int(source['optimizer']['state'][0]['step']),1)
        # Compare microforwards to one monolithic forward on this deterministic
        # toy (not an unsupported claim about real FA batching kernels).
        direct,_,direct_opt,_,_=toy();restore_blob(source,direct,copy.deepcopy(ema),direct_opt,device)
        direct_opt.zero_grad()
        z=direct(reader.get([tid for g in groups.values() for tid in g['trajectory_ids']],device)).reshape(51,5)
        y=torch.tensor([g['labels'] for g in groups.values()],dtype=z.dtype)
        kwargs=dict(c['recipe']['loss_kwargs']);kwargs['within_pair_mask']=torch.tensor([g['pair_mask'] for g in groups.values()],dtype=torch.bool)
        probe_ablation_loss(z,y,**kwargs)['loss'].backward();direct_opt.step()
        for actual,expected in zip(model.parameters(),direct.parameters()):
            torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)

    def test_checkpoint_restores_adam_rng_and_next_cursor_without_reset(self):
        model,ema,opt,groups,c=toy();device=torch.device('cpu');reader=ToyReader()
        source=cpu_blob(model,ema,opt)
        run_update(model,ema,opt,reader,groups,list(groups),c,device)
        accepted=cpu_blob(model,ema,opt);accepted['cursor']=1
        before=state_hashes(model,ema,opt,device)
        with tempfile.TemporaryDirectory() as d:
            receipt=save_checkpoint(Path(d),accepted)
            loaded=torch.load(checked(receipt),map_location='cpu',mmap=True,weights_only=False)
            self.assertEqual(loaded['cursor'],1);self.assertEqual(loaded['steps'],2)
            self.assertEqual(len(loaded['source_all_rank_states']),4)
            self.assertTrue(loaded['hardware_topology_changed'])
            self.assertFalse(loaded['four_rank_bitexact_resume_claim'])
            run_update(model,ema,opt,reader,groups,list(groups),c,device)
            restore_blob(loaded,model,ema,opt,device)
            self.assertEqual(state_hashes(model,ema,opt,device),before)
            with self.assertRaises(ValueError):save_checkpoint(Path(d),accepted)
        self.assertEqual(source['steps'],1)

    def test_mixed_balanced_accuracy_auc_and_class_errors_are_primary(self):
        rows=[dict(strict_em=[1,0,0,0,0],predicted_mixed=.9),
              dict(strict_em=[1,1,0,0,0],predicted_mixed=.4),
              dict(strict_em=[0]*5,predicted_mixed=.6),
              dict(strict_em=[1]*5,predicted_mixed=.1)]
        metrics=prediction_metrics(rows)
        self.assertEqual(metrics['confusion'],dict(tp=1,fp=1,fn=1,tn=1))
        self.assertEqual(metrics['mixed_balanced_accuracy'],.5)
        self.assertEqual(metrics['classwise']['allwrong']['false_positive_rate'],1.)
        self.assertEqual(metrics['classwise']['allcorrect']['false_positive_rate'],0.)
        self.assertEqual(metrics['k_histogram'],[1,1,1,0,0,1])
        self.assertEqual(metrics['mixed_roc_auc'],.75)
        self.assertEqual(roc_auc([.5,.5],[0,1]),.5)


if __name__=='__main__':unittest.main()
