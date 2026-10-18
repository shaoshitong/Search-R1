import copy,os,unittest
from unittest.mock import patch
import torch
from .b_progressive import prefix,continuation_questions,expand,ordinary,MODE
from .periodic_data import select_actual_mixed_groups

class ProgressiveBTests(unittest.TestCase):
    def test_exact_stage_boundaries_and_periodic_budget_unchanged(self):
        self.assertEqual([prefix(i)for i in (50,51,100,101,150,151,200)],[5,20,20,40,40,80,80])
        self.assertEqual([continuation_questions(i)for i in (51,101,151)],[768,1024,1536])
        config={'b_progressive':{'mode':MODE}}
        self.assertTrue(ordinary(config,51));self.assertFalse(ordinary(config,60));self.assertTrue(ordinary(config,200));self.assertFalse(ordinary({},51))
    def test_expansion_preserves_raw_ema_adam_clock_and_zero_new_slots(self):
        model={f'block.param{i}':torch.tensor([float(i)])for i in range(12)}
        model.update({'output.weight':torch.randn(1,278*2048),'output.bias':torch.tensor([.3])})
        saved=dict(model=model,ema_model={k:v*.7 for k,v in model.items()},binding={'prefix_tokens':5,'head_max_tokens':278},optimizer={'param_groups':[{'params':list(range(14)),'lr':5e-6}],'state':{i:{'exp_avg':v*.1,'exp_avg_sq':v.square(),'step':torch.tensor(700.)}for i,v in enumerate(model.values())}},rank_rng=['same']*4,ema_updates=700)
        out=expand(saved,20)
        for key in ('model','ema_model'):
            self.assertTrue(torch.equal(out[key]['output.weight'][:,:278*2048],saved[key]['output.weight']))
            self.assertEqual(out[key]['output.weight'][:,278*2048:].count_nonzero(),0)
            for name in model:
                if name!='output.weight':self.assertTrue(torch.equal(out[key][name],saved[key][name]))
        self.assertEqual(out['ema_updates'],700);self.assertEqual(out['rank_rng'],saved['rank_rng'])
        self.assertEqual(out['optimizer']['state'][12]['step'],700)
        self.assertEqual(out['binding']['prefix_tokens'],20)
        self.assertEqual(saved['binding']['prefix_tokens'],5)
    def test_expanded_rerank_uses_same_g5_true_labels_then_prior_score(self):
        groups=[];records=[];expected={}
        for i in range(768):
            tids=[f'{i}-{j}'for j in range(5)];groups.append(dict(qid=f'q{i}',candidate_ordinal=i*2,question_hash=f'h{i}',q=1-i/768,trajectory_ids=tids))
            for j,tid in enumerate(tids):
                expected[tid]='prefix-'+tid
                records.append(dict(trajectory_id=tid,qid=f'q{i}',strict_em=int(j==0 and i>=400),actor_version='a50',prefix_scan_sha256='scan',selection_sha256='pool',prefix_sha256=expected[tid]))
        chosen,stats=select_actual_mixed_groups(groups,records,fallback_mode='probe_score_fill',provenance={'actor_version':'a50','prefix_scan_sha256':'scan','pool_selection_sha256':'pool'},expected_prefixes=expected,admitted_questions=768,ordinal_limit=5120)
        self.assertEqual([g['qid']for g in chosen[:368]],[f'q{i}'for i in range(400,768)])
        self.assertEqual(stats['chosen_mixed_count'],368);self.assertEqual(stats['fallback_count'],144)
        self.assertEqual(chosen[368]['qid'],'q0');self.assertEqual(len(chosen),512)
        records[0]['prefix_sha256']='different'
        with self.assertRaises(ValueError):select_actual_mixed_groups(groups,records,fallback_mode='probe_score_fill',provenance={'actor_version':'a50','prefix_scan_sha256':'scan','pool_selection_sha256':'pool'},expected_prefixes=expected,admitted_questions=768,ordinal_limit=5120)
if __name__=='__main__':unittest.main()
