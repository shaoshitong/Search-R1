"""Small direct-logit/label-blind-selection fixtures, no real tensor assets."""
import copy,gzip,json,math,tempfile,unittest
from pathlib import Path
import torch
from torch import nn
from .mixed_direct_evaluation import score_reader,summarize,selected_qids


class DirectToy(nn.Module):
    def __init__(self):super().__init__();self.bias=nn.Parameter(torch.tensor(0.))
    def encode_trajectories(self,batch):return dict(hidden=batch['features'],attention_mask=batch['attention_mask'],position_ids=batch['position_ids'])
    def group_logits(self,encoded):return encoded['hidden'].mean((1,2,3))+self.bias


class Reader:
    def __init__(self,groups):
        self.values={tid:z for g,z in zip(groups,[-2.,1.,3.])for tid in g['trajectory_ids']}
    def get(self,ids,device):
        return dict(features=torch.tensor([[[self.values[t]]]for t in ids],device=device),attention_mask=torch.ones(len(ids),1,device=device,dtype=torch.bool),position_ids=torch.zeros(len(ids),1,device=device,dtype=torch.long))


def groups():return [dict(qid='q%d'%i,trajectory_ids=['q%d-t%d'%(i,j)for j in range(5)],labels=y)for i,y in enumerate([[0]*5,[1,0,0,0,0],[1]*5])]


class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_direct_sigmoid_stable_BCE_three_class_counts_and_no_trajectory_probability(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);g=groups();model=DirectToy();before=copy.deepcopy(model.state_dict())
            result=score_reader(model,Reader(g),g,torch.device('cpu'),root/'predictions.jsonl.gz',root/'progress.json','synthetic',micro_groups=2)
            with gzip.open(root/'predictions.jsonl.gz','rt')as f:rows=[json.loads(x)for x in f]
            self.assertEqual(len(rows),3)
            for r,z in zip(rows,[-2.,1.,3.]):
                self.assertEqual(set(r),{'qid','trajectory_ids','strict_em','true_mixed','predicted_mixed'})
                self.assertAlmostEqual(r['predicted_mixed'],1/(1+math.exp(-z)),places=6)
            expected=(math.log1p(math.exp(-2))+math.log1p(math.exp(-1))+math.log1p(math.exp(3)))/3
            self.assertAlmostEqual(result['binaryMixedBCE'],expected,places=6)
            self.assertEqual(result['pool']['k_histogram'],[1,1,0,0,0,1]);self.assertEqual(result['pool']['mixed_balanced_accuracy'],.75)
            self.assertEqual(result['classwise_binaryMixedBCE']['allwrong']['questions'],1);self.assertEqual(result['classwise_binaryMixedBCE']['Mixed']['questions'],1);self.assertEqual(result['classwise_binaryMixedBCE']['allcorrect']['questions'],1)
            self.assertFalse(result['trajectory_probability_or_accuracy_produced']);self.assertFalse(result['product_formula_used'])
            for name,value in model.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)

    def test_label_permutation_does_not_change_Top_selected_identity(self):
        rows=[dict(qid=str(i),trajectory_ids=['%d-%d'%(i,j)for j in range(5)],strict_em=y,true_mixed=0<sum(y)<5,predicted_mixed=p)for i,(y,p)in enumerate([([0]*5,.8),([1,0,0,0,0],.8),([1]*5,.3)])]
        changed=copy.deepcopy(rows)
        for r,source in zip(changed,reversed(rows)):
            r['strict_em']=source['strict_em'];r['true_mixed']=source['true_mixed']
        a=summarize(rows,2);b=summarize(changed,2)
        self.assertEqual(selected_qids(rows,2),['0','1']);self.assertEqual(a['selected_qids_sha256'],b['selected_qids_sha256']);self.assertFalse(a['labels_used_to_select'])

    def test_bad_ownG5_or_old_probability_schema_fails_closed(self):
        row=dict(qid='q',trajectory_ids=['t%d'%j for j in range(5)],strict_em=[0]*5,true_mixed=False,predicted_mixed=.4,p_j=[.1]*5)
        with self.assertRaises(ValueError):summarize([row])
        del row['p_j'];row['trajectory_ids'][-1]=row['trajectory_ids'][0]
        with self.assertRaises(ValueError):summarize([row])
        g=groups();g[0]['trajectory_ids']=g[0]['trajectory_ids'][:-1]
        with tempfile.TemporaryDirectory()as folder:
            with self.assertRaisesRegex(ValueError,'own G5'):score_reader(DirectToy(),Reader(g),g,torch.device('cpu'),Path(folder)/'p.gz',Path(folder)/'progress.json','bad')


if __name__=='__main__':unittest.main()
