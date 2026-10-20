import copy
import json
from pathlib import Path
import unittest
from . import peer_newseed_subset_export as m

class SubsetTests(unittest.TestCase):
    def test_actual_published_metadata_contract(self):
        p=Path(__file__).resolve().parents[2]/'reports/qwen7b-algorithm-improvement-20261021/peer-newseed-delta/existing-audited-delta-plan.json'
        v=json.loads(p.read_bytes());self.assertEqual(m.validate_selection(v)['questions'],15616)
        for key,value in [('questions',31232),('batches',123),('subset','all-sealed')]:
            bad=copy.deepcopy(v);bad[key]=value
            with self.assertRaises(ValueError):m.validate_selection(bad)
        bad=copy.deepcopy(v);bad['selected_batches'][0]['input']='inputs/foreign.json'
        with self.assertRaises(ValueError):m.validate_selection(bad)
    def test_own_identity_no_old_or_duplicate(self):
        old=dict(qids={'old'},hashes={'oldhash'},tids={'oldtid'})
        seen=dict(qids=set(),hashes=set(),tids=set())
        g=dict(qid='new',question_hash='hash',trajectory_ids=['a','b','c','d','e'])
        m.identity_guard(old,[g],seen);self.assertEqual(len(seen['tids']),5)
        with self.assertRaises(ValueError):m.identity_guard(old,[g],seen)
        for key,val in [('qid','old'),('question_hash','oldhash'),('trajectory_ids',['oldtid'])]:
            bad=dict(g);bad[key]=val
            with self.assertRaises(ValueError):m.identity_guard(old,[bad],dict(qids=set(),hashes=set(),tids=set()))

if __name__=='__main__':unittest.main()
