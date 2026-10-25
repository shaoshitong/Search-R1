import copy
import json
import tempfile
import unittest
from pathlib import Path
from reproduction.dynrpo7b.objective_revision import all_correct_rows, bind_transition, REVISION
from reproduction.dynrpo7b.selection import Candidate, select_actual_mixed
from reproduction.dynrpo7b.collection_resume import find_resume

class ObjectiveRevisionTests(unittest.TestCase):
    def test_permuted_groups(self):
        ids=['a','b','c']*5
        labels=[1,1,0, 1,1,0, 1,0,0, 1,1,0, 1,1,0]
        self.assertEqual(all_correct_rows(ids,labels),{0,3,6,9,12})
        with self.assertRaises(ValueError):all_correct_rows(ids[:-1],labels[:-1])

    def test_fallback_excludes_high_scoring_wrong(self):
        rows=[Candidate('wrong',0,100,strict_em=(0,)*5),
              Candidate('correct',1,0,strict_em=(1,)*5),
              Candidate('mixed',2,-1,strict_em=(1,0,0,0,0))]
        chosen=select_actual_mixed(rows,2,allow_all_wrong=False)
        self.assertEqual([x.question_id for x in chosen],['mixed','correct'])
        with self.assertRaises(ValueError):select_actual_mixed(rows,3,allow_all_wrong=False)

    def test_only_authorized_transition(self):
        old=dict(budget=dict(per_rank_actor_logical_micro_batch=2,objective_reduction='two',learning_rate=1e-6))
        new=dict(budget=dict(per_rank_actor_logical_micro_batch=4,objective_reduction='four',learning_rate=1e-6),
                 objective_revision=REVISION,actor_logical_micro_batch_per_rank=4,
                 all_correct_virtual_zero=False,all_correct_reward_one=True,selection_allow_all_wrong=False)
        receipt=dict(completed_outer=50,status='DYNRPO7B_ROLLING_READY',source_base='/old')
        bind_transition(new,old,receipt)
        self.assertTrue(new['objective_transition']['old_run_suffix_not_imported'])
        bad=copy.deepcopy(new);bad['budget']['learning_rate']=2e-6
        with self.assertRaises(ValueError):bind_transition(bad,old,receipt)
        with self.assertRaises(ValueError):bind_transition(new,old,dict(receipt,completed_outer=51))
        bind_transition(new,new,dict(receipt,completed_outer=51))

    def test_fork_never_reads_old_completed_outer51_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';new=root/'new'
            p=old/'preparation/dynrpo7b-collection/outer-0051.json';p.parent.mkdir(parents=True)
            p.write_text('invalid: this old suffix must not be read')
            run=dict(run_id='r',full_state_restore=dict(source_base=str(old)),
                     objective_transition=dict(old_run_suffix_not_imported=True,source_base=str(old)))
            self.assertIsNone(find_resume(new,run,51,1000))

if __name__=='__main__':unittest.main()
